#############################
########## Purpose ##########
#############################

# Figure 3 is the sensitivity analysis, showcasing the family of correlations that exist within the transit parameters.
# To generate this figure we must explore the chi-squared space of pair of parameters. To do this, we need perform an MCMC
# to identify the parts of parameter space that are most relevant to explore i.e. in the locality of the best-fit solution.
# The same parameters for the injection and retrieval are used in Figure 2 and 3 so we can just use the results of Figure 2's MCMC
# for this first step.
# Once these chi-squared maps are evaluated, a correlation metric is calculated from their shape, and these correlation values
# are plotted in a colourful matrix.
# This file generates the chi-squared spaces for pairs of parameters.

######################################
########## Import libraries ##########
######################################
from jax import random
import jax

import paths
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
import os

# For 64-bit precision since JAX defaults to 32-bit
jax.config.update("jax_enable_x64", True)


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
# Overall LDCs : c1 = 0.7219 c2 = -0.8824 c3 = 0.7763 c4 = -0.2683
init_NLLD_coeffs = nonlinear_4param_ld_law(u1=0.7219, u2=-0.8824, u3=0.7763, u4=-0.2683)

#Updating initial state dictionary
for iLD, LD_coeff in enumerate(init_NLLD_coeffs):
    init_state_dic[f'LD_u{iLD+1}'] = LD_coeff

#Get starting points for the LD coefficients
init_LD_prop = nonlinear_4param_ld_law(u1=0.7219, u2=-0.8824, u3=0.7763, u4=-0.2683, order=3)

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


#%% Input and outputs directories
input_dir = str(paths.data / "Fig2_Storage") + "/"
output_dir = str(paths.data / "Fig3_Storage") + "/"

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
fixed_args['all_param_list'] = var_param_list + fix_param_list
fixed_args['labels'] = [r'R$_p$/R$_{\star}$',r'i (rad)',r'$\rho_{\star}$ (g/cm$^{3}$)',r'u$_1$', r'u$_2$',r'u$_3$',r'P (days)',r'$\sqrt{e}$cos($\omega$)',r'$\sqrt{e}$sin($\omega$)']

#% Define number of points to sample the parameter space with
fixed_args['sample_pts'] = 100

#%% Number of burn-in steps used in MCMC
fixed_args['nburn'] = 70000

#%% Model scatter and seed to use for the plot
model_scatter =  16.68100537200059 
seed = 70

# Filtering parameters
THRESHOLDS = [5, 4, 3]  # Number of IQRs for outlier detection (5 is conservative)
ROUNDS = 3
verbose = True

##############################
##### Relevant functions #####
##############################
def load_result(args):
    """
    Optimized file loading with iterative 2D sigma clipping.
    
    FILTERING STRATEGY:
    - Maintains a 2D mask of shape (n_walkers, n_steps_post) throughout
    - At each round, sigma clipping is computed over all currently-surviving
      (walker, step) pairs for chi2 and each parameter independently
    - Individual (walker, step) pairs are removed without discarding the
      entire walker chain
    - Returns good_steps_mask (n_walkers, n_steps_post) for fine-grained use
    """
    raw_save_dir, model_scatter, seed, return_full = args
    print(f"  Processing scatter{model_scatter:.3f}, seed{seed}...")
    try:
        path_base = f'{raw_save_dir}/{jnp.floor(model_scatter)}ppm/Seed{seed}/'

        # Load with memory mapping
        raw_chain = np.load(path_base + 'chains.npy', mmap_mode='r')
        logprob   = np.load(path_base + 'logprob.npy', mmap_mode='r')
        chi2      = np.load(path_base + 'chi2_chain.npy', mmap_mode='r')

        n_walkers, n_steps, n_params = raw_chain.shape
        n_steps_post = n_steps - fixed_args['nburn']

        # ======================================================================
        # BURN: slice to post-burnin only
        # All arrays are (n_walkers, n_steps_post) or (n_walkers, n_steps_post, n_params)
        # ======================================================================
        burnt_chain   = np.array(raw_chain[:, fixed_args['nburn']:, :])   # (n_walkers, n_steps_post, n_params)
        burnt_chi2    = np.array(chi2[:,    fixed_args['nburn']:])         # (n_walkers, n_steps_post)
        burnt_logprob = np.array(logprob[:, fixed_args['nburn']:])         # (n_walkers, n_steps_post)

        # ======================================================================
        # 2D MASK: True = this (walker, step) pair is still alive
        # ======================================================================
        good_steps_mask = np.ones((n_walkers, n_steps_post), dtype=bool)  # (n_walkers, n_steps_post)

        # ======================================================================
        # ITERATIVE SIGMA CLIPPING - operates on surviving pairs each round
        # ======================================================================
        for round_idx in range(ROUNDS):

            THRESHOLD  = THRESHOLDS[round_idx]
            n_alive    = np.sum(good_steps_mask)
            if verbose:print(f'    ROUND {round_idx+1}/{ROUNDS} (threshold={THRESHOLD}σ, {n_alive} pairs alive)')

            # ------------------------------------------------------------------
            # FILTER 1: CHI2
            # Extract the chi2 values of currently-alive (walker, step) pairs
            # ------------------------------------------------------------------
            alive_chi2 = burnt_chi2[good_steps_mask]                      # (n_alive,)

            quartiles  = np.percentile(alive_chi2, [25, 50, 75])
            mu, iqr    = quartiles[1], quartiles[2] - quartiles[0]

            # Build a 2D bad mask: False everywhere, then flag outliers among alive pairs
            chi2_bad_2d                  = np.zeros((n_walkers, n_steps_post), dtype=bool)
            chi2_bad_2d[good_steps_mask] = (alive_chi2 < mu - THRESHOLD * iqr) | (alive_chi2 > mu + THRESHOLD * iqr)

            if verbose:print(f"      Chi2:  flagged {np.sum(chi2_bad_2d)} / {n_alive} ({100*np.sum(chi2_bad_2d)/n_alive:.1f}%)")

            # ------------------------------------------------------------------
            # FILTER 2: PARAMETERS
            # Same pattern: extract alive values per parameter, flag outliers
            # ------------------------------------------------------------------
            param_bad_2d = np.zeros((n_walkers, n_steps_post), dtype=bool)

            for param_idx in range(n_params):
                alive_param = burnt_chain[:, :, param_idx][good_steps_mask]  # (n_alive,)

                quartiles = np.percentile(alive_param, [25, 50, 75])
                mu, iqr   = quartiles[1], quartiles[2] - quartiles[0]

                outliers_flat = (alive_param < mu - THRESHOLD * iqr) | (alive_param > mu + THRESHOLD * iqr)
                param_bad_2d[good_steps_mask] |= outliers_flat

                param_name = fixed_args['var_param_list'][param_idx] if param_idx < len(fixed_args['var_param_list']) else f'param_{param_idx}'
                if verbose:print(f"      {param_name}: flagged {np.sum(outliers_flat)} / {n_alive} ({100*np.sum(outliers_flat)/n_alive:.1f}%)")

            # ------------------------------------------------------------------
            # UPDATE 2D MASK
            # ------------------------------------------------------------------
            round_bad_2d  = chi2_bad_2d | param_bad_2d
            good_steps_mask &= ~round_bad_2d

            if verbose:print(f"      Round removed {np.sum(round_bad_2d)} / {n_alive} ({100*np.sum(round_bad_2d)/n_alive:.1f}%)")

        if verbose:print(f"    Final: {np.sum(good_steps_mask)} / {n_walkers * n_steps_post} (walker, step) pairs survived ({100 * np.sum(good_steps_mask)/(n_walkers * n_steps_post)} %)")

        # ======================================================================
        # EXTRACT DATA
        # ======================================================================

        # Best-fit: find the best logprob among surviving pairs
        masked_logprob         = np.where(good_steps_mask, burnt_logprob, -np.inf)
        best_walker, best_step = np.unravel_index(np.argmax(masked_logprob), masked_logprob.shape)
        bestfit_r              = float(burnt_chain[best_walker, best_step, 0])

        # r chain: collect parameter 0 from all surviving (walker, step) pairs
        r_chain_post_burnin = burnt_chain[:, :, 0][good_steps_mask]       # (n_surviving_pairs,)

        if return_full:
            full_chain   = np.array(raw_chain)
            full_logprob = np.array(logprob)
            full_chi2    = np.array(chi2)

            return (model_scatter, seed, r_chain_post_burnin, bestfit_r,
                    True, full_chain, full_logprob, full_chi2, good_steps_mask) 
        else:
            return (model_scatter, seed, r_chain_post_burnin, bestfit_r,
                    False, None, None, None, None)

    except Exception as e:
        print(f"Error loading scatter{model_scatter:.3f}, seed{seed}: {e}")
        import traceback
        traceback.print_exc()
        return None

#Helper function to check the existence of directories
def check_dir(dir_name):
    if not os.path.isdir(dir_name):os.makedirs(dir_name)
    return dir_name

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

#Check directories exist
check_dir(input_dir)
fixed_args['save_loc'] = check_dir(output_dir)


#Check model scatter directory exists
scatter_dir = check_dir(input_dir+f'{jnp.floor(model_scatter)}ppm/')
print(f"MODEL SCATTER = {model_scatter:.2f}")

#Check seed directory exists
seed_dir = check_dir(scatter_dir+f'Seed{seed}/')
print(f"SEED = {seed}")

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

# evaluate this likelihood
print(f"initial chi2: {jnp.sum( (true_lc - noisy_LC)**2/noisy_std**2 )}, initial chi2: {-0.5* ( jnp.sum( (true_lc - noisy_LC)**2/noisy_std**2 ) + jnp.sum(jnp.log(2*jnp.pi*noisy_std**2)) ) }")

#Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=[10, 6], sharex=True, gridspec_kw={'height_ratios': [3, 1]})
ax1.errorbar(init_state_dic['times'], noisy_LC, yerr=noisy_std, fmt='.', zorder=1)
ax1.plot(init_state_dic['times'], true_lc, color='red', zorder=2)
ax2.errorbar(init_state_dic['times'], 1e6*(noisy_LC - true_lc), yerr=noisy_std, fmt='r.', zorder=1)
for ax in [ax1, ax2]:
    ax.axvline(-0.5 * T_dur, color='black', linestyle='dashed')
    ax.axvline(0.5 * T_dur, color='black', linestyle='dashed')
    ax.axvline(-1.5 * T_dur, color='black', linestyle='dotted')
    ax.axvline(1.5 * T_dur, color='black', linestyle='dotted')
ax1.set_title('Model LC with %.f ppm scatter'%model_scatter)
ax2.set_xlabel('Time (BJD)')
ax1.set_ylabel('Flux')
ax2.set_ylabel('Difference (ppm)')
fig.tight_layout()
plt.savefig(fixed_args['save_loc']+'init_guess.pdf')
plt.close()

#Loading the MCMC results
print(f'Retrieving MCMC')
raw_chain = jnp.load(seed_dir+"chains.npy")
logprob = jnp.load(seed_dir+"logprob.npy")
_, _, _, _, _, _, _, _, good_steps_mask = load_result((input_dir, model_scatter, seed, True))

#Finding the index of max log-probability
max_step, max_walker = jnp.unravel_index(jnp.argmax(logprob), logprob.shape)

# Get highest log probability parameters
best_params = {}
for i, param in enumerate(fixed_args['var_param_list']):
    best_params[param] = raw_chain[max_walker, max_step, i]
for idx, param in enumerate(fixed_args['fix_param_list']):
    best_params[param] = fixed_args['fix_param_val'][idx]

#Generate bestfit vector 
theta_best = jnp.array([best_params[p] for p in fixed_args['all_param_list']])

#Calculate the bestfit LC to use for comparison
bestfit_LC = create_jaxoplanet_model(init_state_dic['times'], best_params)



#######################
### More functions ####
#######################

#Helper function to unpack array into a dictionary
def unpack_params(theta):
    p = {}
    for i, name in enumerate(fixed_args['all_param_list']):
        p[name] = theta[i]
    return p

def chi2_from_theta(theta, x, y, yerr):
    #Build dictionary from vector
    p = unpack_params(theta)
    #Create the light curve with jaxoplanet function
    y_pred = create_jaxoplanet_model(x, p)
    #Calculate chi2
    return jnp.sum((y_pred - y)**2 / yerr**2)

def chi2_1d(param_idx, param_vals):
    
    def eval_one(val):
        theta = theta_best.at[param_idx].set(val)
        return chi2_from_theta(theta, init_state_dic['times'], bestfit_LC, noisy_std)

    return jax.vmap(eval_one)(param_vals)

def chi2_2d(idx1, idx2, vals1, vals2):
    V1, V2 = jnp.meshgrid(vals1, vals2, indexing="ij")
    flat_v1 = V1.ravel()
    flat_v2 = V2.ravel()

    def eval_one(v1, v2):
        theta = theta_best.at[idx1].set(v1)
        theta = theta.at[idx2].set(v2)
        return chi2_from_theta(theta, init_state_dic['times'], bestfit_LC, noisy_std)

    chi2_flat = jax.vmap(eval_one)(flat_v1, flat_v2)
    return chi2_flat.reshape((vals1.size, vals2.size))

#######################################
##### Chi-squared map calculation #####
#######################################

# Replace the chi2_dic dictionary approach with individual file saves
for i, param1 in enumerate(fixed_args['var_param_list']):
    for j, param2 in enumerate(fixed_args['var_param_list']):
        if j < i:
            continue

        print(f'CHI2 RETRIEVAL: {param1} vs {param2}')

        if param1 == param2:
            #Generate chi2 range
            param_vals = jnp.linspace(
                raw_chain[max_walker, max_step, i] - jnp.std(raw_chain[:, fixed_args['nburn']:, i][good_steps_mask]),
                raw_chain[max_walker, max_step, i] + jnp.std(raw_chain[:, fixed_args['nburn']:, i][good_steps_mask]),
                fixed_args['sample_pts']
            )
            #Calculate chi2 values
            chi2_vals = chi2_1d(i, param_vals)
            chi2_vals -= jnp.min(chi2_vals)
            
            #Save values
            jnp.save(fixed_args['save_loc'] + f"chi2_{param1}_{param1}.npy", chi2_vals)
            
            #Delete variables (frees memory)
            del chi2_vals, param_vals
            
        else:
            #Generate 2D chi2 range
            param1_vals = jnp.linspace(
                raw_chain[max_walker, max_step, i] - jnp.std(raw_chain[:, fixed_args['nburn']:, i][good_steps_mask]),
                raw_chain[max_walker, max_step, i] + jnp.std(raw_chain[:, fixed_args['nburn']:, i][good_steps_mask]),
                fixed_args['sample_pts']
            )
            param2_vals = jnp.linspace(
                raw_chain[max_walker, max_step, j] - jnp.std(raw_chain[:, fixed_args['nburn']:, j][good_steps_mask]),
                raw_chain[max_walker, max_step, j] + jnp.std(raw_chain[:, fixed_args['nburn']:, j][good_steps_mask]),
                fixed_args['sample_pts']
            )
            #Calculate chi2 values
            chi2_map = chi2_2d(i, j, param1_vals, param2_vals)
            chi2_map -= jnp.min(chi2_map)
            
            #Save values
            jnp.save(fixed_args['save_loc'] + f"chi2_{param1}_{param2}.npy", chi2_map)
            
            #Delete variables (frees memory)
            del chi2_map, param1_vals, param2_vals
        