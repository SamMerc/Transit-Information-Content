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
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import exotic_ld as el
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns


######################################
########## Hyper-parameters ##########
######################################

LD_data_path = '/Volumes/Pandora/Work/PhD/Research/TIC/LD simulation'
save_data_path = '/Users/samsonmercier/Desktop/Work/PhD/Research/TIC/Fig2_helper_Storage/'

models = ['phoenix']#,'kurucz', 'stagger', 'mps1', 'mps2']

mu_values = np.linspace(0, 1, 1000)

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

N = 60

n_components = 5

################################
########## Code block ##########
################################

#Instantiate model to store intensity profiles
intensity_profiles={model : np.zeros((N, N, N), dtype=object) for model in models}

#Iterate over all the stellar models available 
for model in models:
    
    #Instantiate 
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
                
                # Integrate stellar spectrum over wavelength
                intensity_profile = np.trapezoid(sld.stellar_intensities, sld.stellar_wavelengths, axis=0)

                # Normalize
                intensity_profile /= intensity_profile[0]
                intensity_profiles[model][i, j, k] = intensity_profile

    # Retrieve the grid of mu values
    mus = jnp.copy(sld.mus)
    rs = jnp.sqrt(1 - mus**2)

    # Perform PCA analysis 
    pca = PCA(n_components=n_components)
    profiles_pca = pca.fit_transform(intensity_profiles[model])

    # Extract eigen intensity profile
    eigen_profiles = pca.components_  # Shape: (n_components, n_mu_points)

    # Clustering in PCA space
    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(profiles_pca[:, :3])  # Use first 3 PCs

    # Visualization
    fig = plt.figure(figsize=(16, 12))

    # Plot 1: All original intensity profiles
    ax1 = plt.subplot(3, 3, 1)
    for i in range(min(50, len(intensity_profiles[model]))):
        ax1.plot(mu_values, intensity_profiles[model][i], alpha=0.3, color='gray', linewidth=0.5)
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
    ax3.plot(range(1, n_components + 1), np.cumsum(pca.explained_variance_ratio_), 'ro-', linewidth=2)
    ax3.axhline(y=0.95, color='g', linestyle='--', label='95% variance')
    ax3.set_xlabel('Number of Components')
    ax3.set_ylabel('Cumulative Explained Variance')
    ax3.set_title('Cumulative Variance Explained')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4-8: First 5 Eigen-intensity profiles
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    for i in range(n_components):
        ax = plt.subplot(3, 3, 4 + i)
        ax.plot(mu_values, eigen_profiles[i], color=colors[i], linewidth=2)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('μ = cos(θ)')
        ax.set_ylabel('Component Value')
        ax.set_title(f'Eigen-profile {i+1} ({pca.explained_variance_ratio_[i]*100:.1f}%)')
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

    plt.tight_layout()
    # plt.savefig('pca_intensity_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

                # # Retrieve r values
                # r = jnp.sqrt(1 - sld.mus**2)

                # # Perform spherical model correction
                # if model == 'phoenix':

                #     #Calculate intensity profile derivative
                #     dIdmu = jnp.gradient(intensity_profile, sld.mus)

                #     #Find index where derivative is maximum
                #     max_dIdmu_index = jnp.argmax(dIdmu)

                #     #Renormalize radial profile at this index
                #     norm_r = r / r[max_dIdmu_index]
                #     norm_mu = jnp.sqrt(1 - norm_r**2)