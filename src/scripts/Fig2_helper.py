#############################
########## Purpose ##########
#############################

# Figures 2, 3, and 4 require a 4-th order non-linear limb-darkening law for the injection / simulation of the LC.
# Given that we are working with a made up fiducial system, we need to identify the limb-darkening values to use for this.
# In order to do this, we explore all available intensity profiles for a given grid of stellar models, and perform a PCA 
# analysis to identify both the median/mode and an outlier intensity profile which can be used in our analyses. 
# We perform such decomposition on each individual grid of stellar models, and in doing so this allows us to highlight the choice of 
# 1. stellar model and 2. limb-darkening prescription on the transit depth amplification factor and bias 



######################################
########## Import libraries ##########
######################################


import numpy as np
import matplotlib
import os
matplotlib.use('TkAgg')  # or 'Qt5Agg' or 'MacOSX'
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import jit, vmap
import exotic_ld as el
import pickle
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.interpolate import CubicSpline
from lmfit import minimize, Parameters


######################################
########## Hyper-parameters ##########
######################################

LD_data_path = '/Volumes/Pandora/Work/PhD/Research/TIC/LD simulation'
orig_save_data_path = '/Users/samsonmercier/Desktop/Work/PhD/Research/TIC/Fig2_helper_Storage/'

models = ['mps1'] #['phoenix','kurucz', 'stagger', 'mps1', 'mps2']

Teffs = {
    'phoenix' : [2300, 15000], 
    'kurucz' : [3500, 6500],
    'stagger' : [4000, 7000],
    'mps2' : [3500, 9000],
    'mps1' : [3500, 9000],
}

loggs = {
    'phoenix' : [0.0, 6.0], 
    'kurucz' : [4.0, 5.0],
    'stagger' : [1.5, 5.0],
    'mps1' : [3.0, 5.0],
    'mps2' : [3.0, 5.0],
}

metallicitys = {
    'phoenix' : [-1.5, 1.0], 
    'kurucz' : [-5.0, 1.0],
    'stagger' : [-3.0, 0.0],
    'mps1' : [-5.0, 1.5],
    'mps2' : [-5.0, 1.5],
}


mu_resolution = {
    'phoenix' : 78,
    'kurucz' : 17,
    'stagger' : 10,
    'mps1' : 24,
    'mps2' : 24,
}

lambda_resolution = {
    'phoenix' : 54500,
    'kurucz' : 1221,
    'stagger' : 105767,
    'mps1' : 1221,
    'mps2' : 1221,
}

N = 10

bs = jnp.linspace(0, 1, N)
ps = jnp.logspace(-3, -1, N)

n_components = 5

n_clusters = 2

mode = 'build' # 'build' or 'load'

############################
###### Function block ######
############################
@jit
def calculate_segment_area(r, y_interior, y_exterior):
    """
    Calculate occulted area within circle of radius r,
    between horizontal lines at y_interior and y_exterior.
    
    Cases:
    1. Both edges outside circle: area = 0
    2. Only interior edge intersects: area = interior_segment
    3. Both edges intersect: area = interior_segment - exterior_segment
    """
    
    # Clip ratios for arccos
    ratio_interior = jnp.clip((y_interior) / r, -1.0, 1.0)
    ratio_exterior = jnp.clip((y_exterior) / r, -1.0, 1.0)
    
    # Calculate segment areas
    # Segment = area between chord and circle edge
    theta_interior = 2 * jnp.arccos(ratio_interior)
    theta_exterior = 2 * jnp.arccos(ratio_exterior)
    
    segment_interior = 0.5 * r**2 * (theta_interior - jnp.sin(theta_interior))
    segment_exterior = 0.5 * r**2 * (theta_exterior - jnp.sin(theta_exterior))
    
    # Band area (area between two chords)
    band_area = segment_interior - segment_exterior
    
    return band_area

@jit
def chord_extracter_single(b, p, intensity_profiles, mus):
    """
    Vectorized version - Extract the intensity profile for a given chord size (p) and location (b).
    
    :param b: Impact parameter of the transit chord.
    :param p: Planet-to-star radius ratio.
    :param intensity_profiles: 2D array of intensity spectra (shape: n_wavelengths x n_mus) 
    :param mus: Array of mu values (outer edges of annuli).
    """
    # Define the grid of radii on the stellar grid
    rs = jnp.sqrt(1 - mus**2)
    
    # Calculate the occulted area for each annulus using vectorized operations
    # Each annulus i goes from r=0 (if i=0) or r=rs[i-1] to r=rs[i]
    
    # Chord edges
    y_interior = b - p
    y_exterior = b + p
    
    # Calculate segment area up to each radius
    segment_areas = vmap(calculate_segment_area, in_axes=(0, None, None))(rs, y_interior, y_exterior)
    
    # Calculate occulted area in each annulus by taking differences
    # For annulus 0: full segment area at rs[0]
    # For annulus i>0: segment_areas[i] - segment_areas[i-1]
    occulted_area = jnp.concatenate([
        segment_areas[0:1],  # First annulus
        jnp.diff(segment_areas)  # Remaining annuli
    ])
    
    # Calculate the area of each annulus
    annulus_area = jnp.concatenate([
        jnp.pi * rs[0:1]**2,  # First annulus (disk)
        jnp.diff(jnp.pi * rs**2)  # Remaining annuli (rings)
    ])
    
    # Calculate proportion (avoid division by zero)
    occulted_proportion = jnp.where(
        annulus_area > 0,
        occulted_area / annulus_area,
        0.0
    )

    # Weight intensity profiles
    weighted = intensity_profiles * occulted_proportion[jnp.newaxis, :]
    
    return weighted

# Vectorize over both b and p
chord_extracter_vectorized = jit(vmap(
    vmap(chord_extracter_single, in_axes=(None, 0, None, None)),  # vmap over p
    in_axes=(0, None, None, None)  # vmap over b
))
    
################################
########## Code block ##########
################################
if not os.path.exists(orig_save_data_path):os.makedirs(orig_save_data_path)

# Instantiate dictionary to store information 
gen_dict = {}

gen_dict['stellar_mus']={model : np.zeros(mu_resolution[model], dtype=float) for model in models}
gen_dict['stellar_wavelengths']={model : np.zeros(lambda_resolution[model], dtype=float) for model in models}

gen_dict['global_intensity_profiles']={model : np.zeros((N, N, N, mu_resolution[model]), dtype=float) for model in models}

gen_dict['local_intensity_profiles']={model : np.zeros((N, N, N, N, N, mu_resolution[model]), dtype=float) for model in models}

#Iterate over all the stellar models available 
for model in models:
    
    # Create save path for each model
    save_data_path = orig_save_data_path + f'{model}/'
    if not os.path.exists(save_data_path):os.makedirs(save_data_path)

    #Build intensity profiles grid
    if mode == 'build':
        #Iterate over the three stellar parameters and retrieve intensity profiles
        #Temperature
        for i, T in enumerate(jnp.linspace(Teffs[model][0], Teffs[model][1], N)):
            #Surface gravity
            for j, g in enumerate(jnp.linspace(loggs[model][0], loggs[model][1], N)):
                #Metallicity
                for k, m in enumerate(jnp.linspace(metallicitys[model][0], metallicitys[model][1], N)):
                    
                    #Calculate stellar spectrum - across wavelength and viewing angle
                    print('GENERATING Teff =', T, 'logg =', g, 'metallicty =', m, 'for model', model)
                    sld = el.StellarLimbDarkening(M_H=m, Teff=T, logg=g,
                                ld_model=model,
                                ld_data_path=LD_data_path,
                                interpolate_type="nearest")
                    
                    #Store the wavelength and mu arrays
                    if (i == 0) and (j == 0) and (k == 0):
                        gen_dict['stellar_mus'][model] = jnp.copy(sld.mus)
                        gen_dict['stellar_wavelengths'][model] = jnp.copy(sld.stellar_wavelengths)

                    #Store the global stellar intensity spectrum
                    global_stellar_intensities = jnp.copy(sld.stellar_intensities)
                    
                    # Integrate stellar spectrum over wavelength
                    global_intensity_profile = jnp.trapezoid(global_stellar_intensities, gen_dict['stellar_wavelengths'][model], axis=0)

                    # Normalize and store the global intensity profile
                    gen_dict['global_intensity_profiles'][model][i, j, k] = global_intensity_profile/global_intensity_profile[0]

                    ##############################################################################
                    ########## Extract intensity profile for each transit chord ##################
                    ##############################################################################

                    # Define the annuli edges
                    annuli_mus = jnp.append(
                        gen_dict['stellar_mus'][model][:-1] + jnp.diff(gen_dict['stellar_mus'][model])/2, 
                        gen_dict['stellar_mus'][model][-1] + (jnp.diff(gen_dict['stellar_mus'][model])[-1]/2)
                    )
                    
                    # Compute for all (b, p) combinations at once
                    local_stellar_intensities = chord_extracter_vectorized(
                        bs, ps, 
                        global_stellar_intensities, 
                        annuli_mus
                    )

                    # Compute for all (b, p) combinations the integral of the local stellar intensity spectrum over wavelength 
                    local_intensity_profiles = jnp.trapezoid(
                        local_stellar_intensities, 
                        gen_dict['stellar_wavelengths'][model], 
                        axis=2
                    )
                    
                    #Normalize and store this local intensity profile
                    gen_dict['local_intensity_profiles'][model][i, j, k, :, :, :] = local_intensity_profiles / local_intensity_profiles[:, :, 0:1]

        #Store the stellar spectrum
        with open(save_data_path + 'data.pkl', 'wb') as f:pickle.dump(gen_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    #Load intensity profiles grid
    elif mode == 'load':
        with open(save_data_path + 'data.pkl', 'rb') as f:gen_dict = pickle.load(f)
    
    else:
        raise KeyboardInterrupt('Mode not recognized.')

    ##########################################
    ########## PCA analysis ##################
    ##########################################
    # Retrieve the grid of mu values
    mus = jnp.copy(gen_dict['stellar_mus'][model])
    rs = jnp.sqrt(1 - mus**2)

    # Reshaping intensity profiles for PCA
    intensity_profiles = gen_dict['local_intensity_profiles']
    pca_int_profile = intensity_profiles[model].reshape((N*N*N*N*N, mu_resolution[model]))

    # Perform PCA analysis 
    pca = PCA(n_components=n_components)
    profiles_pca = pca.fit_transform(pca_int_profile)

    # Extract eigen intensity profile
    eigen_profiles = pca.components_  # Shape: (n_components, n_mu_points)

    # Clustering in PCA space
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(profiles_pca[:, :3])  # Use first 3 PCs

    # Visualization
    fig = plt.figure(figsize=(16, 12))

    # Plot 1: All original intensity profiles
    ax1 = plt.subplot(3, 3, 1)
    for prof in pca_int_profile:
        ax1.plot(mus, prof, alpha=0.3, color='gray', linewidth=0.5)
    ax1.set_xlabel('μ = cos(θ)')
    ax1.set_ylabel('Intensity')
    ax1.set_title('Sample of Original Intensity Profiles')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Scree plot (explained variance)
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(range(1, n_components + 1), pca.explained_variance_ratio_, 'bo-', linewidth=2)
    ax2.set_xlabel('Principal Component')
    ax2.set_ylabel('Explained Variance Ratio')
    ax2.set_title('Scree Plot')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Cumulative explained variance
    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(range(1, n_components + 1), jnp.cumsum(pca.explained_variance_ratio_), 'ro-', linewidth=2)
    ax3.axhline(y=0.95, color='g', linestyle='--', label='95% variance')
    ax3.set_xlabel('Number of Components')
    ax3.set_ylabel('Cumulative Explained Variance')
    ax3.set_title('Cumulative Variance Explained')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4-8: First 5 Eigen-intensity profiles
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    for i_plot in range(n_components):
        ax = plt.subplot(3, 3, 4 + i_plot)
        ax.plot(mus, eigen_profiles[i_plot], color=colors[i_plot], linewidth=2)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('μ = cos(θ)')
        ax.set_ylabel('Component Value')
        ax.set_title(f'Eigen-profile {i_plot+1} ({pca.explained_variance_ratio_[i_plot]*100:.1f}%)')
        ax.grid(True, alpha=0.3)

    # Plot 9: PCA space with clusters
    ax9 = plt.subplot(3, 3, 9)
    scatter = ax9.scatter(profiles_pca[:, 0], profiles_pca[:, 1], 
                        c=cluster_labels, cmap='viridis', s=50, alpha=0.6)
    ax9.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax9.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax9.set_title('Stellar Models in PCA Space')
    plt.colorbar(scatter, ax=ax9, label='Cluster')
    ax9.grid(True, alpha=0.3)

    plt.savefig(save_data_path + 'PCA_Analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Find the mode (median/typical) and outlier profiles
    # Calculate distances from the cluster centers
    distances_from_centers = kmeans.transform(profiles_pca[:, :3])

    # Find the most "typical" profile (closest to its cluster center)
    typical_idx = jnp.argmin(jnp.min(distances_from_centers, axis=1))
    typical_profile = pca_int_profile[typical_idx]

    # Find outlier profiles (one from each cluster edge)
    outlier_indices = []
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_distances = distances_from_centers[cluster_mask, cluster_id]
        # Get the farthest point in this cluster
        outlier_in_cluster = jnp.where(cluster_mask)[0][jnp.argmax(cluster_distances)]
        outlier_indices.append(outlier_in_cluster)

    # Alternatively, reconstruct profiles using different numbers of components
    # This shows how well PCA approximates the original profiles
    from matplotlib.gridspec import GridSpec

    # Create figure with custom grid
    fig2 = plt.figure(figsize=(15, 12))
    gs = GridSpec(3, 1+len(outlier_indices), figure=fig2, height_ratios=[1, 0.66, 0.33], hspace=0.05)

    # ===== Top Row: Typical and Outlier Profiles =====
    # Plot typical profile surrounded by rest of profiles
    ax = fig2.add_subplot(gs[0, 0])
    for prof in pca_int_profile:
        ax.plot(mus, prof, alpha=0.3, color='gray', linewidth=0.5)
    ax.plot(mus, typical_profile, 'b-', linewidth=2, label='Typical (Mode)', zorder=10)
    ax.set_xlabel('μ = cos(θ)')
    ax.set_ylabel('Normalized Intensity')
    ax.set_title('Most Typical Intensity Profile')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot outlier profiles
    for i_plot, outlier_idx in enumerate(outlier_indices):
        ax = fig2.add_subplot(gs[0, i_plot+1])
        for prof in pca_int_profile:
            ax.plot(mus, prof, alpha=0.3, color='gray', linewidth=0.5)
        ax.plot(mus, pca_int_profile[outlier_idx], 'r-', linewidth=2, 
                label=f'Outlier {i_plot+1}', zorder=10)
        ax.set_xlabel('μ = cos(θ)')
        ax.set_ylabel('Normalized Intensity')
        ax.set_title(f'Outlier Profile {i_plot+1}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # ===== Bottom Two Rows: Reconstruction Quality =====
    test_profile_idx = typical_idx
    original = pca_int_profile[test_profile_idx]

    n_comp_list = [1, 2, n_components]
    for col_idx, n_comp_plot in enumerate(n_comp_list):
        # Reconstruct using only first n_comp_plot components
        pca_temp = PCA(n_components=n_comp_plot)
        pca_temp.fit(pca_int_profile)
        reduced = pca_temp.transform(original.reshape(1, -1))
        reconstructed = pca_temp.inverse_transform(reduced)[0]
        
        residual = original - reconstructed
        rmse = np.sqrt(np.mean(residual**2))
        
        # Top part: Original vs Reconstructed (2/3 of height)
        ax_top = fig2.add_subplot(gs[1, col_idx])
        ax_top.plot(mus, original, 'k-', linewidth=2, label='Original', alpha=0.7)
        ax_top.plot(mus, reconstructed, 'r--', linewidth=2, label='Reconstructed')
        ax_top.set_ylabel('Normalized Intensity')
        ax_top.set_title(f'{n_comp_plot} Components (RMSE={rmse:.4f})')
        ax_top.legend(loc='best')
        ax_top.grid(True, alpha=0.3)
        ax_top.set_xticklabels([])  # Remove x-axis labels for top subplot
        
        # Bottom part: Residuals (1/3 of height)
        ax_bottom = fig2.add_subplot(gs[2, col_idx], sharex=ax_top)
        ax_bottom.plot(mus, 100 * residual / original, 'g-', linewidth=1.5, label='Rel. Diff.')
        ax_bottom.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax_bottom.set_xlabel('μ = cos(θ)')
        ax_bottom.set_ylabel('Relative Diff. (%)')
        ax_bottom.legend(loc='best')
        ax_bottom.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_data_path + 'Mode&Outliers.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Print summary statistics
    print(f"\n=== PCA Analysis Summary ===")
    print(f"Total profiles analyzed: {len(pca_int_profile)}")
    print(f"Variance captured by {n_components} components: {jnp.sum(pca.explained_variance_ratio_)*100:.2f}%")
    print(f"\nTypical profile index: {typical_idx}")
    print(f"Outlier profile indices: {outlier_indices}")

    # Save the profiles for use in your transit simulations
    np.save(save_data_path + f'mode_intensity_profile_{model}.npy', typical_profile)
    for i_save, outlier_idx in enumerate(outlier_indices):
        np.save(save_data_path + f'outlier{i_save+1}_intensity_profile_{model}.npy', 
                pca_int_profile[outlier_idx])
    np.save(save_data_path + f'mu_values_{model}.npy', mus)

    print(f"\nSaved profiles to {save_data_path}")


    #Fitting the mode and outlier profiles with a 4-th order non-linear limb-darkening law
    for j_fit, special_profile in enumerate([typical_profile] + [pca_int_profile[idx] for idx in outlier_indices]):
        
        #Interpolate intensity profile from its grid to a grid of 100 mu values going from 0.01 to 1.0 with increments of 0.01 with cubic spline (Claret & Bloemen 2011)
        new_mus = jnp.linspace(0.01, 1.0, 100)
        inter_special_profile = CubicSpline(mus[::-1], special_profile[::-1])(new_mus)

        #Define 4-th order non-linear LD law
        def fourNLLD(x, coeffs):
            return 1 - coeffs[0] * (1 - x**(1/2)) - coeffs[1] * (1 - x) - coeffs[2] * (1 - x**(3/2)) - coeffs[3] * (1 - x**2)
        
        #Define residual function to minimize
        def residual(params, x, base_prof):
            return fourNLLD(x, [params[f'c{i_coeff+1}'].value for i_coeff in range(4)]) - base_prof
    
        #Define lmfit parameters
        params = Parameters()
        for i_param in range(4):
            params.add(f'c{i_param+1}', value=np.random.uniform(0, 1))

        #Perform the minimization
        result = minimize(residual, params, args=(new_mus, inter_special_profile))

        #Plot base profile, interpolated profile, and best-fit profile
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        ax1.plot(mus, special_profile, 'bo', label='Original Profile', alpha=0.5)
        ax1.plot(new_mus, inter_special_profile, 'g-', label='Interpolated Profile', alpha=0.7)
        ax1.plot(new_mus, fourNLLD(new_mus, [result.params[f'c{i_coeff+1}'].value for i_coeff in range(4)]), 'r--', label='Best-fit 4th Order NLLD', linewidth=2)
        ax2.plot(new_mus, 100 * (inter_special_profile - fourNLLD(new_mus, [result.params[f'c{i_coeff+1}'].value for i_coeff in range(4)]))/inter_special_profile, 'r--', linewidth=2)
        ax2.set_xlabel('μ = cos(θ)')
        ax1.set_ylabel('Normalized Intensity')
        ax2.set_ylabel('Relative Difference (%)')
        ax1.set_title('4th Order Non-Linear Limb-Darkening Fit - ' + ('Typical Profile' if j_fit==0 else f'Outlier Profile {j_fit}'))
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax2.grid(True, alpha=0.3)
        plt.savefig(save_data_path + f'4thOrderNLLD_Fit_Profile_{"mode" if j_fit==0 else f"outlier{j_fit}"}_{model}.png', dpi=150, bbox_inches='tight')
        plt.show()