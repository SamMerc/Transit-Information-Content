#############################
########## Purpose ##########
#############################

# Figure 2 showcases the perturbation on transit shape incurred by modifying the input parameters.
# Modifying the input parameters requires figuring out by how much to modify each parameter by and to do this need to use an MCMC.
# In this file the results from the MCMC and retrieved and used to generate Figure 2.



######################################
########## Import libraries ##########
######################################
from jax import random
import paths
import jax
print(f"JAX devices: {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jaxoplanet.orbits.keplerian import System, Central
import astropy.units as u
from astropy.constants import G
from jaxoplanet.light_curves import limb_dark_light_curve
from squishyplanet.limb_darkening_laws import nonlinear_4param_ld_law
import numpy as np

# For 64-bit precision since JAX defaults to 32-bit
jax.config.update("jax_enable_x64", True)

# Set random seed
jaxnoise_key = jax.random.PRNGKey(0)

#####################################
######## Add temp. plots ############
#####################################
# Generate some data
random_numbers = np.random.randn(100, 10)

# Plot and save
fig = plt.figure(figsize=(7, 6))
plt.plot(random_numbers)
plt.xlabel("x")
plt.ylabel("y")
fig.savefig(paths.figures / "Fig2.pdf", bbox_inches="tight", dpi=300)

#############################################
########## Define hyper-parameters ##########
#############################################
#%% Mock light curve

#%%%% Define G in units needed now to avoid JAX tracing issues
G_solar_units = G.to(u.Rsun**3 / (u.Msun * u.day**2)).value
G_cgday = G.to(u.cm**3 / (u.g * u.day**2)).value
R_star = (1.0 * u.R_sun).value
#%%%% Mock system - fiducial
init_state_dic = {}
init_state_dic['period'] = 1.                                 #days
a_meters = ( (G.value * (1.0 * u.M_sun).to(u.kg).value * (init_state_dic['period'] * 24 * 3600)**2)/(4 * jnp.pi**2) )**(1/3)  
init_state_dic['a'] = a_meters / (1.0 * u.R_sun).to(u.m).value  #stellar radius
init_state_dic['r'] = 0.1                                     #stellar radius
init_state_dic['i'] = jnp.deg2rad(90)                         #radians
init_state_dic['omega'] = 0.0                                 #radians
init_state_dic['e'] = 0.0                                     #unitless
init_state_dic['t0'] = 0.0                                    #days

#Setting base LDCs
init_NLLD_coeffs = nonlinear_4param_ld_law(u1=0.1, u2=0.2, u3=0.4, u4=0.3)

#Updating initial state dictionary
for iLD, LD_coeff in enumerate(init_NLLD_coeffs):
    init_state_dic[f'LD_u{iLD+1}'] = LD_coeff

#Get starting points for the LD coefficients
init_LD_prop = nonlinear_4param_ld_law(u1=0.1, u2=0.2, u3=0.4, u4=0.3, order=3)

#%%%% Calculate transit duration
# Convert angles to radians
# Impact parameter (eccentricity-corrected)
b = (
    (init_state_dic['a'] * jnp.cos(init_state_dic['i'])) / R_star
    * (1 - init_state_dic['e']**2) / (1 + init_state_dic['e'] * jnp.sin(init_state_dic['omega']))
)
# Argument inside arcsin
arg = (
    (1/init_state_dic['a'])
    * jnp.sqrt((1 + init_state_dic['r'])**2 - b**2)
    / jnp.sin(init_state_dic['i'])
)
# Numerical safety
arg = np.clip(arg, -1.0, 1.0)
# Transit duration
T_dur = (
    (init_state_dic['period'] / jnp.pi)
    * jnp.sqrt(1 - init_state_dic['e']**2) / (1 + init_state_dic['e'] * jnp.sin(init_state_dic['omega']))
    * jnp.arcsin(arg)
)

#%%%% Model time - ensure pre- and post-transit are same duration as transit
low_t = -1.5*T_dur                                                      #days
high_t = 1.5*T_dur                                                      #days
exposure_time = 5                                                       #seconds
num_t = jnp.floor((((high_t - low_t) * 24 * 3600)/exposure_time))       #number of points
init_state_dic['times'] = jnp.linspace(low_t, high_t, int(num_t))       #days


#%% Storing outputs of nested sampling and plots
raw_save_dir = '/Users/samsonmercier/Desktop/Work/PhD/Research/TIC/Fig2_Storage/'

#%% Model parameters
mod_prop = {
    'r'         : {'vary':True, 'guess':0.11, 'bounds':[0.07, 0.15]},
    'i'         : {'vary':True, 'guess':jnp.deg2rad(88.5), 'bounds':[jnp.deg2rad(88.), jnp.deg2rad(92.)]},
    'a'         : {'vary':True, 'guess':init_state_dic['a']-1, 'bounds':[init_state_dic['a']-2, init_state_dic['a']+2]},
    'LD_u1'     : {'vary':True, 'guess':init_LD_prop[0], 'bounds':[init_LD_prop[0] - 0.5, init_LD_prop[0] + 0.5]},
    'LD_u2'     : {'vary':True, 'guess':init_LD_prop[1], 'bounds':[init_LD_prop[1] - 0.5, init_LD_prop[1] + 0.5]},
    'LD_u3'     : {'vary':True, 'guess':init_LD_prop[2], 'bounds':[init_LD_prop[2] - 0.5, init_LD_prop[2] + 0.5]},
    'period'    : {'vary':True, 'guess':1., 'bounds':[0.9995, 1.0005]}, #Gaussian prior
    'sqrtecosw' : {'vary':True, 'guess':0.1, 'bounds':[-0.2, 0.2]},
    'sqrtesinw' : {'vary':True, 'guess':0.1, 'bounds':[-0.2, 0.2]},
    't0'        : {'vary':False, 'guess':0., 'bounds':[-100,100]},
}

#%% Defining important lists
var_param_list = []
fix_param_list = []
fix_param_val = []
for key in mod_prop:
    if mod_prop[key]['vary']:
        var_param_list.append(str(key))
    else:
        fix_param_list.append(str(key))
        fix_param_val.append(mod_prop[key]['guess'])

#%% Defining dictionary to store additional info. needed for the model
fixed_args={}
fixed_args['var_param_list']=var_param_list
fixed_args['fix_param_list']=fix_param_list
fixed_args['fix_param_val']=fix_param_val

#%% Number of burn-in steps
fixed_args['nburn'] = 700000

#%% Model scatter and seed to use for the plot
model_scatter =  16.68100537200059 
seed = 80

##############################
##### Relevant functions #####
##############################

def create_jaxoplanet_model(x, p):
    
    #Retrieving ecc and w
    if ('sqrtecosw' in p) and ('sqrtesinw' in p):
        ecc = p['sqrtecosw']**2 + p['sqrtesinw']**2
        w = jnp.arccos(p['sqrtecosw']/jnp.sqrt(ecc))
    elif ('e' in p) and ('omega' in p):
        ecc = p['e']
        w = p['omega']
    else:
        ecc = 0.
        w = 0.

    #Retrieving inclination
    if 'cosi' in p:
        inc = jnp.arccos(p['cosi'])
    else:
        inc = p['i']

    #Define star
    stellar_rho =  (3 * jnp.pi * p['a']**3)/ ( p['period']**2 * G_solar_units )
    star = Central(density=stellar_rho)

    #Define planet
    planet = System(star).add_body(
        time_transit = p['t0'],
        period = p['period'],
        inclination = inc,
        eccentricity = ecc,
        omega_peri = w, 
        radius = (p['r'] * R_star),
    )

    #Apply limb-darkening
    max_coeff = len([param for param in p if 'LD' in param])
    ld_u_coeffs = jnp.array([p[f"LD_u{i}"] for i in range(1, max_coeff+1)])

    jaxo_lc = 1.0 + limb_dark_light_curve(planet, ld_u_coeffs)(x)
    return jaxo_lc.reshape((-1))

#############################################
################ Running code ###############
#############################################

#############################
####### Generate data #######
#############################
print('GENERATING DATA')

#Pure data
true_lc = create_jaxoplanet_model(init_state_dic['times'], init_state_dic)

#Build noisy data
std = model_scatter * 1e-6
noisy_LC = true_lc + std * random.normal(jax.random.PRNGKey(seed), shape=true_lc.shape)
noisy_std = std * jnp.ones(true_lc.shape, dtype=float)

##################
#### Plotting ####
##################

#Load MCMC results
raw_chain = jnp.load(raw_save_dir+f"{jnp.floor(model_scatter)}ppm/Seed{seed}/chains.npy")
logprob = jnp.load(raw_save_dir+f"{jnp.floor(model_scatter)}ppm/Seed{seed}/logprob.npy")

# Get highest log probability parameters
max_walker, max_step = jnp.unravel_index(jnp.argmax(logprob), logprob.shape)
best_params = {}
for i, param in enumerate(fixed_args['var_param_list']):
    best_params[param] = raw_chain[max_walker, max_step, i]
for idx, param in enumerate(fixed_args['fix_param_list']):
    best_params[param] = fixed_args['fix_param_val'][idx]

# Compute bestfit model
bestfit_lc = create_jaxoplanet_model(init_state_dic['times'], best_params)
bestfit_RMS = jnp.sqrt(jnp.average((noisy_LC - bestfit_lc)**2))

# Figure 2
print('FIGURE 2')

#Calculate bestfit stellar density
bestfit_stellar_rho = ( ( 3 * jnp.pi ) / ( G_cgday * best_params['period']**2 ) ) * best_params['a']**3

#Initializing figure 
# Define the layout
layout = [
    ['A', 'B'],  # Row 1
    ['C', 'C'],  # Row 2
]
fig, axes = plt.subplot_mosaic(layout, figsize=(6, 6))

plot_phase = init_state_dic['times'] / init_state_dic['period']
marksize = 4
elem_size = 10
#Looping over each parameter to perturb it
for param, paramlabel, perturbation_sigma, color, shape in zip(
                                            ['LD_u1', 'LD_u2', 'LD_u3', 'i', 'stellar_rho', 'period', 'sqrtecosw', 'sqrtesinw'],
                                            [r'u$_1$', r'u$_2$', r'u$_3$', 'i', r'$\rho_{\star}$', r'$P$', r'$\sqrt{e}$cos($\omega$)', r'$\sqrt{e}$sin($\omega$)'],
                                            [1, 1, 1, 1, 1, 1, 1, 1],
                                            plt.get_cmap('coolwarm')(jnp.linspace(0., 1, 8)),
                                            ['.','.','.','.','.','.','.','.']):

    #Get the perturbation value
    if param=='stellar_rho':
        period_chain = raw_chain[:, fixed_args['nburn']:, fixed_args['var_param_list'].index('period')]
        a_chain = raw_chain[:, fixed_args['nburn']:, fixed_args['var_param_list'].index('a')]
        stellar_rho_chain = ( ( 3 * jnp.pi ) / ( G_cgday * period_chain**2 ) ) * a_chain**3
        perturbation = perturbation_sigma * jnp.std(stellar_rho_chain)
    else:
        param_chain = raw_chain[:, fixed_args['nburn']:, fixed_args['var_param_list'].index(param)]
        perturbation = perturbation_sigma * jnp.std(param_chain)

    
    #Adust the value of the corresponding parameter
    perturbed_state_dic = best_params.copy()
    if param=='stellar_rho':
        perturbed_state_dic['a'] = ( ( perturbed_state_dic['period']**2 * (bestfit_stellar_rho + perturbation) * G_cgday) / (3 * jnp.pi))**(1/3) 
    else:perturbed_state_dic[param] += perturbation

    #Compute the perturbed model
    tr_perturbed_LC = create_jaxoplanet_model(init_state_dic['times'], perturbed_state_dic)
        
    #Plot the difference between nominal and perturbed model
    axes['A'].plot(plot_phase, (bestfit_lc - tr_perturbed_LC)*1e6, shape, color=color, markersize=marksize)
    axes['A'].set_xlim([-0.0425, -0.033])
    axes['C'].plot(plot_phase, (bestfit_lc - tr_perturbed_LC)*1e6, shape, color=color, label=paramlabel, markersize=marksize)
    axes['C'].set_xlim([-0.02, 0.02])
    axes['C'].set_ylim([-55, 40])

#Plotting contours for stellar density vs i in top right panel 
# Build flattened sample vectors (ensure 1D arrays) before feeding KDE
rho_chain = np.asarray(stellar_rho_chain).ravel()
sqrtesinw_chain = np.asarray(raw_chain[:, fixed_args['nburn']:, fixed_args['var_param_list'].index('sqrtesinw')]).ravel()

# Create a 2D kernel density estimate on a grid and draw filled contours in axes['B']
ax = axes['B']

# # Choose contour probability fractions
frac_levels = [0.1, 0.3, 0.6, 0.9]

# 2D histogram
H, xedges, yedges = np.histogram2d(rho_chain, sqrtesinw_chain, bins=25)

levels = np.max(H) * np.array(frac_levels)

# Extract a contour (e.g., 30% of max density)
Xc, Yc = np.meshgrid(0.5*(xedges[:-1]+xedges[1:]),
                    0.5*(yedges[:-1]+yedges[1:]))
cs = ax.contour(Xc, Yc, H.T, levels=levels, colors=plt.cm.coolwarm(np.linspace(0, 1, len(frac_levels))), linewidths=2)

# A bit of labeling and limits
ax.set_xlabel(r"$\rho_{\star}$ (g/cm$^{3}$)", fontsize=elem_size)
ax.set_ylabel(r"$\sqrt{e}\sin{\omega}$", fontsize=elem_size)
ax.grid(True)
ax.tick_params(labelsize=elem_size)

# === Aesthetic adjustments === #
#Shared axes
axes['B'].yaxis.set_ticks_position("right")      # Moves tick marks to the top
axes['B'].yaxis.set_label_position("right")      # Moves x-axis label to the top

# Reduce vertical space and horizontal space
fig.subplots_adjust(hspace=0.31, wspace=0.05)

#Labels
# Shared y-label across both rows (centered vertically on the figure)
axes['A'].set_title('Ingress/Egress', fontsize=elem_size)
axes['B'].set_title(r'$\rho_{\star}$-i-e-$\omega$ Correlation', fontsize=elem_size)
axes['C'].set_title('Transit Bottom', fontsize=elem_size)
axes['A'].set_ylabel('Flux difference (ppm)', fontsize=elem_size)
axes['C'].set_ylabel('Flux difference (ppm)', fontsize=elem_size)
axes['C'].set_xlabel('Orbital Phase', fontsize=elem_size)
axes['A'].set_xlabel('Orbital Phase', fontsize=elem_size)
axes['A'].tick_params(axis='x', labelsize=elem_size)
axes['B'].tick_params(axis='both', labelsize=elem_size)
axes['C'].tick_params(axis='both', labelsize=elem_size)
axes['A'].set_xticks([-0.041, -0.038, -0.035], [-0.041, -0.038, -0.035])
axes['C'].set_xticks([-0.015, -0.005, 0.005, 0.015], [-0.015, -0.005, 0.005, 0.015])
axes['C'].legend(handletextpad=0.01, loc='lower center', frameon=True, ncol=4)
for mosaic_elem in ['A','C']:axes[mosaic_elem].grid(True)
plt.savefig(raw_save_dir+'/Fig2.pdf', bbox_inches="tight")
plt.close()