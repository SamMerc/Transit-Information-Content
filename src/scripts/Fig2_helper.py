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
import gc


######################################
########## Hyper-parameters ##########
######################################

LD_data_path = '/Volumes/Pandora/Work/PhD/Research/TIC/LD simulation'
orig_save_data_path = '/Volumes/Pandora/Work/PhD/Research/TIC/Gen_Storage/Fig2_helper_Storage/'

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

N_star = 10

N_chords = 100

N_bs_ps = 7
bs = jnp.linspace(0, 1, N_bs_ps)
ps = jnp.logspace(-3, -1, N_bs_ps)

n_components = 5

n_clusters = 2

mode = 'build' # 'build' or 'load'

############################
###### Function block ######
############################
@jit
def calculate_annulus_overlap(r_planet, p, r_inner, r_outer):
    """
    JAX-optimized annulus overlap calculation.
    """
    overlap_outer = calculate_circle_overlap(r_planet, p, r_outer)
    overlap_inner = calculate_circle_overlap(r_planet, p, r_inner)
    return overlap_outer - overlap_inner


@jit
def calculate_circle_overlap(d, r1, r2):
    """
    JAX-optimized circle overlap calculation.
    Handles arrays efficiently.
    """
    # No overlap cases
    no_overlap = d >= r1 + r2
    
    # Complete overlap cases
    complete_overlap = d <= jnp.abs(r2 - r1)
    
    # Partial overlap - use lens formula
    d2 = d * d
    r1_2 = r1 * r1
    r2_2 = r2 * r2
    
    # Safe division - avoid division by zero
    denom = 2 * d * r1
    safe_denom = jnp.where(denom == 0, 1.0, denom)
    alpha_arg = (d2 + r1_2 - r2_2) / safe_denom
    alpha_arg = jnp.clip(alpha_arg, -1.0, 1.0)
    alpha = jnp.arccos(alpha_arg)
    
    denom2 = 2 * d * r2
    safe_denom2 = jnp.where(denom2 == 0, 1.0, denom2)
    beta_arg = (d2 + r2_2 - r1_2) / safe_denom2
    beta_arg = jnp.clip(beta_arg, -1.0, 1.0)
    beta = jnp.arccos(beta_arg)
    
    partial_area = r1_2 * alpha + r2_2 * beta - 0.5 * (r1_2 * jnp.sin(2 * alpha) + r2_2 * jnp.sin(2 * beta))
    
    # Combine all cases
    area = jnp.where(no_overlap, 0.0,
                     jnp.where(complete_overlap, jnp.pi * jnp.minimum(r1, r2)**2,
                              partial_area))
    
    return area
    
@jit
def chord_intensity(b, p, intensity_spectra, stellar_radii):
    '''
    JAX-optimized version for a single (b, p) pair.
    :param b: Transit chord impact parameter.
    :param p: Planet-to-star radius ratio.
    :param intensity_spectra: 2D array of intensity spectra (shape : n_wavelengths * n_stellar_mus).
    :param stellar_radii: Array of r values for the edges of the annuli discretizing the stellar disk. (shape : n_stellar_mus)
    :param N_chords: Number of points discretizing the transit chord.
    '''
    # Calculate the possible positions of the planet along the (half) transit chord based on the impact parameter
    # We only need half of the transit chord to trace out the intensity profile needed.
    x_min = 0.0
    x_max = jnp.sqrt((1+p)**2 - b**2)
    x_vals = jnp.linspace(x_min, x_max, N_chords)
    r_ps = jnp.sqrt(b**2 + x_vals**2)  # Shape: (N_chords,)
    
    # Add inner edge (r=0) and outer edge (r=1 or last stellar radius)
    r_inner_edges = jnp.concatenate([jnp.array([0.0]), stellar_radii[:-1]])    # Shape: (n_stellar_mus,)
    r_outer_edges = stellar_radii

    # Vectorize over chord positions and annuli
    r_ps_grid = r_ps[:, None]           # Shape: (N_chords, 1)
    r_inner_grid = r_inner_edges[None, :]  # Shape: (1, n_stellar_mus)
    r_outer_grid = r_outer_edges[None, :]  # Shape: (1, n_stellar_mus)

    # Calculate all overlaps at once
    overlap_areas = calculate_annulus_overlap(
        r_ps_grid, p, r_inner_grid, r_outer_grid
    )  # Shape: (N_chords, n_stellar_mus)
        
    # Calculate the occulted intensity spectrum by doing a weighted sum over the occulted annuli 
    # and the weights are the % of planet-occulted area covered by each annulus
    total_planet_area = jnp.pi * p**2
    weights = overlap_areas / total_planet_area  # Shape: (N_chords, n_stellar_mus)

    # Weighted sum: intensity_spectra @ weights.T -> (n_wavelengths, n_stellar_mus) @ (n_stellar_mus, N_chords)
    # We want (n_wavelengths, N_chords)
    occulted_intensity_spectra = intensity_spectra @ weights.T
    # Shape: (n_wavelengths, N_chords)
        
    return occulted_intensity_spectra

# Vectorize over b and p
chord_intensity_vectorized = jit(vmap(
    vmap(chord_intensity, in_axes=(None, 0, None, None)),
    in_axes=(0, None, None, None)
))

################################
########## Code block ##########
################################
if not os.path.exists(orig_save_data_path):os.makedirs(orig_save_data_path)

# Instantiate dictionary to store information 
gen_dict = {}

# gen_dict['stellar_wavelengths'] = {model : np.zeros(lambda_resolution[model], dtype=float) for model in models}

# gen_dict['stellar_mus']={model : np.zeros(mu_resolution[model], dtype=float) for model in models}

gen_dict['local_rps']={model : np.zeros((N_star, N_star, N_star, N_bs_ps, N_bs_ps, N_chords), dtype=float) for model in models}

# gen_dict['global_intensity_profiles']={model : np.zeros((N_star, N_star, N_star, mu_resolution[model]), dtype=float) for model in models}

gen_dict['local_intensity_profiles']={model : np.zeros((N_star, N_star, N_star, N_bs_ps, N_bs_ps, lambda_resolution[model], N_chords), dtype=float) for model in models}

gen_dict['intensity_profiles_mask']={model : np.zeros((N_star, N_star, N_star, N_bs_ps, N_bs_ps, lambda_resolution[model]), dtype=bool) for model in models}

#Iterate over all the stellar models available 
for model in models:
    
    # Create save path for each model
    save_data_path = orig_save_data_path + f'{model}/'
    if not os.path.exists(save_data_path):os.makedirs(save_data_path)

    #Build intensity profiles grid
    if mode == 'build':
        #Iterate over the three stellar parameters and retrieve intensity profiles
        #Temperature
        for i, T in enumerate(jnp.linspace(Teffs[model][0], Teffs[model][1], N_star)):
            #Surface gravity
            for j, g in enumerate(jnp.linspace(loggs[model][0], loggs[model][1], N_star)):
                #Metallicity
                for k, m in enumerate(jnp.linspace(metallicitys[model][0], metallicitys[model][1], N_star)):
                    
                    #Calculate stellar spectrum - across wavelength and viewing angle
                    print('GENERATING Teff =', T, 'logg =', g, 'metallicty =', m, 'for model', model)
                    sld = el.StellarLimbDarkening(M_H=m, Teff=T, logg=g,
                                ld_model=model,
                                ld_data_path=LD_data_path,
                                interpolate_type="nearest")
                    
                    #Store the wavelength and mu arrays
                    if (i == 0) and (j == 0) and (k == 0):
                        stellar_mus = jnp.copy(sld.mus)
                        # stellar_wavelengths = jnp.copy(sld.stellar_wavelengths)

                    #Store the global stellar intensity spectrum
                    global_stellar_intensities = jnp.copy(sld.stellar_intensities)
                    del sld  # Free memory from large StellarLimbDarkening object
                
                    # Integrate stellar spectrum over wavelength
                    # global_intensity_profile = jnp.trapezoid(global_stellar_intensities, stellar_wavelengths, axis=0) #shape : (n_mus,)

                    # Normalize and store the global intensity profile
                    # gen_dict['global_intensity_profiles'][model][i, j, k] = global_intensity_profile/global_intensity_profile[0]

                    ##############################################################################
                    ########## Extract intensity profile for each transit chord ##################
                    ##############################################################################

                    # Define the annuli edges - the models define intensity spectra at a specific 
                    # mu values so this spreads out these predictions over a band
                    annuli_mus = jnp.append(
                        stellar_mus[:-1] + jnp.diff(stellar_mus)/2, 
                        stellar_mus[-1] + (jnp.diff(stellar_mus)[-1]/2)
                    )

                    # Compute for all (b, p) combinations at once
                    local_stellar_intensities = chord_intensity_vectorized(
                        bs, ps, global_stellar_intensities, jnp.sqrt(1 - annuli_mus**2)
                    ) #shape : (n_bs, n_ps, n_wavelengths, N_chords)

                    #Define the grid of mu values for each ps and bs considered
                    x_max = jnp.sqrt((1 + ps[None, :])**2 - bs[:, None]**2)  # Shape: (n_bs, n_ps)
                    t = jnp.linspace(0.0, 1.0, N_chords)
                    x_vals = x_max[:, :, None] * t[None, None, :]  # Shape: (n_bs, n_ps, N_chords)
                    r_ps = jnp.sqrt(bs[:, None, None]**2 + x_vals**2)  # Shape: (n_bs, n_ps, N_chords)
                    
                    # Normalize the profiles 
                    normalized_profiles = (local_stellar_intensities / local_stellar_intensities[:,:,:,0:1])

                    # Filter out profiles full of zeroes
                    gen_dict['intensity_profiles_mask'][model][i, j, k, :, :, :] |= jnp.all(jnp.isfinite(normalized_profiles), axis=-1)
                                
                    #Normalize and store this local intensity profile
                    gen_dict['local_intensity_profiles'][model][i, j, k, :, :, :, :] = normalized_profiles
                    gen_dict['local_rps'][model][i, j, k, :, :, :] = r_ps

                    #Garbage collection
                    del local_stellar_intensities, global_stellar_intensities, normalized_profiles, annuli_mus, r_ps, x_max, x_vals, t
                    gc.collect()

        #Update the storage 
        # Boolean mask shape: (N_star, N_star, N_star, N_bs_ps, N_bs_ps, lambda_resolution)
        # Applied to array shape: (N_star, N_star, N_star, N_bs_ps, N_bs_ps, lambda_resolution, N_chords)
        # Boolean indexing on the first 6 dims → output shape: (n_valid, N_chords) ✓
        gen_dict['local_intensity_profiles'][model] = gen_dict['local_intensity_profiles'][model][gen_dict['intensity_profiles_mask'][model]]
        gen_dict['local_rps'][model] = gen_dict['local_rps'][model][gen_dict['intensity_profiles_mask'][model]]

        #Remove the mask
        del gen_dict['intensity_profiles_mask'], stellar_mus
        
        # Diagnostic print
        n_total = N_star*N_star*N_star * N_bs_ps*N_bs_ps * lambda_resolution[model]
        n_valid = gen_dict['local_intensity_profiles'][model].shape[0]
        print(f"  Valid profiles: {100 * n_valid / n_total:.1f} %")

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
    print('PCA ANALYSIS')
    # Retrieve the grid of mu values
    xs = jnp.copy(gen_dict['local_rps'][model]) #shape : (n_valid, N_chords)

    # Reshaping intensity profiles for PCA
    pca_int_profile = gen_dict['local_intensity_profiles'][model] #shape : (n_valid, N_chords)

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
    for x, prof in zip([xs, pca_int_profile]):
        ax1.plot(xs, prof, alpha=0.3, color='gray', linewidth=0.5)
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
        ax.plot(xs[i_plot], eigen_profiles[i_plot], color=colors[i_plot], linewidth=2)
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
    for x, prof in zip([xs, pca_int_profile]):
        ax.plot(x, prof, alpha=0.3, color='gray', linewidth=0.5)
    ax.plot(xs, typical_profile, 'b-', linewidth=2, label='Typical (Mode)', zorder=10)
    ax.set_xlabel('μ = cos(θ)')
    ax.set_ylabel('Normalized Intensity')
    ax.set_title('Most Typical Intensity Profile')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot outlier profiles
    for i_plot, outlier_idx in enumerate(outlier_indices):
        ax = fig2.add_subplot(gs[0, i_plot+1])
        for x, prof in zip([xs, pca_int_profile]):
            ax.plot(x, prof, alpha=0.3, color='gray', linewidth=0.5)
        ax.plot(xs[outlier_idx], pca_int_profile[outlier_idx], 'r-', linewidth=2, 
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
        ax_top.plot(xs[test_profile_idx], original, 'k-', linewidth=2, label='Original', alpha=0.7)
        ax_top.plot(xs[test_profile_idx], reconstructed, 'r--', linewidth=2, label='Reconstructed')
        ax_top.set_ylabel('Normalized Intensity')
        ax_top.set_title(f'{n_comp_plot} Components (RMSE={rmse:.4f})')
        ax_top.legend(loc='best')
        ax_top.grid(True, alpha=0.3)
        ax_top.set_xticklabels([])  # Remove x-axis labels for top subplot
        
        # Bottom part: Residuals (1/3 of height)
        ax_bottom = fig2.add_subplot(gs[2, col_idx], sharex=ax_top)
        ax_bottom.plot(xs[test_profile_idx], 100 * residual / original, 'g-', linewidth=1.5, label='Rel. Diff.')
        ax_bottom.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax_bottom.set_xlabel('μ = cos(θ)')
        ax_bottom.set_ylabel('Relative Diff. (%)')
        ax_bottom.legend(loc='best')
        ax_bottom.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_data_path + 'Mode&Outliers.png', dpi=150, bbox_inches='tight')

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

    print(f"\nSaved profiles to {save_data_path}")


    # #Fitting the mode and outlier profiles with a 4-th order non-linear limb-darkening law
    # for j_fit, special_profile in enumerate([typical_profile] + [pca_int_profile[idx] for idx in outlier_indices]):
        
    #     # #Interpolate intensity profile from its grid to a grid of 100 mu values going from 0.01 to 1.0 with increments of 0.01 with cubic spline (Claret & Bloemen 2011)
    #     # new_mus = jnp.linspace(0.01, 1.0, 100)
    #     # inter_special_profile = CubicSpline(mus[::-1], special_profile[::-1])(new_mus)

    #     #Define 4-th order non-linear LD law
    #     def fourNLLD(x, coeffs):
    #         return 1 - coeffs[0] * (1 - x**(1/2)) - coeffs[1] * (1 - x) - coeffs[2] * (1 - x**(3/2)) - coeffs[3] * (1 - x**2)
        
    #     #Define residual function to minimize
    #     def residual(params, x, base_prof):
    #         return fourNLLD(x, [params[f'c{i_coeff+1}'].value for i_coeff in range(4)]) - base_prof
    
    #     #Define lmfit parameters
    #     params = Parameters()
    #     for i_param in range(4):
    #         params.add(f'c{i_param+1}', value=np.random.uniform(0, 1))

    #     #Perform the minimization
    #     result = minimize(residual, params, args=(new_mus, inter_special_profile))

    #     #Plot base profile, interpolated profile, and best-fit profile
    #     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    #     ax1.plot(mus, special_profile, 'bo', label='Original Profile', alpha=0.5)
    #     ax1.plot(new_mus, inter_special_profile, 'g-', label='Interpolated Profile', alpha=0.7)
    #     ax1.plot(new_mus, fourNLLD(new_mus, [result.params[f'c{i_coeff+1}'].value for i_coeff in range(4)]), 'r--', label='Best-fit 4th Order NLLD', linewidth=2)
    #     ax2.plot(new_mus, 100 * (inter_special_profile - fourNLLD(new_mus, [result.params[f'c{i_coeff+1}'].value for i_coeff in range(4)]))/inter_special_profile, 'r--', linewidth=2)
    #     ax2.set_xlabel('μ = cos(θ)')
    #     ax1.set_ylabel('Normalized Intensity')
    #     ax2.set_ylabel('Relative Difference (%)')
    #     ax1.set_title('4th Order Non-Linear Limb-Darkening Fit - ' + ('Typical Profile' if j_fit==0 else f'Outlier Profile {j_fit}'))
    #     ax1.legend()
    #     ax1.grid(True, alpha=0.3)
    #     ax2.grid(True, alpha=0.3)
    #     plt.savefig(save_data_path + f'4thOrderNLLD_Fit_Profile_{"mode" if j_fit==0 else f"outlier{j_fit}"}_{model}.png', dpi=150, bbox_inches='tight')
