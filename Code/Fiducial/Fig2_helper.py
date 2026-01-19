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


######################################
########## Hyper-parameters ##########
######################################

LD_data_path = '/Volumes/Pandora/Work/PhD/Research/TIC/LD simulation'
save_data_path = '/Users/samsonmercier/Desktop/Work/PhD/Research/TIC/Fig2_helper_Storage/'

models = ['phoenix','kurucz', 'stagger', 'mps1', 'mps2']

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

wavelength_range = 'JWST_NIRSpec_Prism'

################################
########## Code block ##########
################################

#Instantiate model to store intensity profiles
intensity_profiles={model : np.zeros((N, N, N), dtype=object) for model in models}

#Iterate over all the stellar models available 
for model in models:
    
    #Instantiate 
    #Iterate over the three stellar parameters
    for i, T in enumerate(jnp.linspace(Teffs[model][0], Teffs[model][1], N)):
        for j, g in enumerate(jnp.linspace(loggs[model][0], loggs[model][1], N)):
            for k, m in enumerate(jnp.linspace(metallicitys[model][0], metallicitys[model][1], N)):
                
                #Calculate stellar spectrum - across wavelength and viewing angle
                print('GENERATING Teff =', T, 'logg =', g, 'metallicty =', m, 'for model', model)
                sld = el.StellarLimbDarkening(M_H=m, Teff=T, logg=g,
                            ld_model=model,
                            ld_data_path=LD_data_path,
                            interpolate_type="nearest")
                
                # Integrate stellar spectrum over wavelength
                intensity_profile = np.trapezoid(sld.stellar_intensities, sld.stellar_wavelengths, axis=0)
                intensity_profile /= intensity_profile[0]
                intensity_profiles[model][i, j, k] = intensity_profile

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