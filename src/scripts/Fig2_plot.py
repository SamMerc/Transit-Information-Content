#############################
########## Purpose ##########
#############################

# Figure 2 showcases the perturbation on transit shape incurred by modifying the input parameters.
# Modifying the input parameters requires figuring out by how much to modify each parameter by and to do this need to use an MCMC.
# In this file the results from the MCMC and retrieved and used to generate Figure 2.



######################################
########## Import libraries ##########
######################################
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
# Global LDCs : c1 = 0.6245 c2 = -0.1898 c3 = 0.1473 c4 = -0.0634           
init_NLLD_coeffs = nonlinear_4param_ld_law(u1=0.6245, u2=-0.1898, u3=0.1473, u4=-0.0634)

#Updating initial state dictionary
for iLD, LD_coeff in enumerate(init_NLLD_coeffs):
    init_state_dic[f'LD_u{iLD+1}'] = LD_coeff

#Get starting points for the LD coefficients
init_LD_prop = nonlinear_4param_ld_law(u1=0.6245, u2=-0.1898, u3=0.1473, u4=-0.0634, order=3)

#%%%% Calculate transit duration
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


#%% Storing outputs
raw_save_dir = str(paths.data / "Fig2_Storage") + "/"

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
fixed_args['nburn'] = 70000

#%% Model scatter and seeds to use for the plot
model_scatter =  16.68100537200059
seeds = [40, 50, 60, 70, 80, 90, 100, 110, 120, 130]

#Seed used for the single-run panel B joint-density contour
plot_seed = 70

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

##################
#### Plotting ####
##################

#Parameters perturbed to build panels A and C, and their plot labels
perturb_params = ['LD_u1', 'LD_u2', 'LD_u3', 'i', 'stellar_rho', 'period', 'sqrtecosw', 'sqrtesinw']
perturb_labels = [r'u$_1$', r'u$_2$', r'u$_3$', 'i', r'$\rho_{\star}$', r'$P$', r'$\sqrt{e}$cos($\omega$)', r'$\sqrt{e}$sin($\omega$)']

#Collecting the per-seed perturbation curves so we can plot their mean and std across seeds
perturbation_curves = {param: [] for param in perturb_params}

#Loop over each of the 10 MCMC runs (different noise seeds)
for seed in seeds:
    print(f'LOADING SEED {seed}')

    #Load MCMC results
    raw_chain_s = jnp.load(raw_save_dir+f"{jnp.floor(model_scatter)}ppm/Seed{seed}/chains.npy")
    logprob_s = jnp.load(raw_save_dir+f"{jnp.floor(model_scatter)}ppm/Seed{seed}/logprob.npy")

    # Get highest log probability parameters
    max_walker_s, max_step_s = jnp.unravel_index(jnp.argmax(logprob_s), logprob_s.shape)
    best_params_s = {}
    for i, param in enumerate(fixed_args['var_param_list']):
        best_params_s[param] = raw_chain_s[max_walker_s, max_step_s, i]
    for idx, param in enumerate(fixed_args['fix_param_list']):
        best_params_s[param] = fixed_args['fix_param_val'][idx]

    _, _, _, _, _, _, _, _, good_steps_mask_s = load_result((raw_save_dir, model_scatter, seed, True))

    # Compute bestfit model
    bestfit_lc_s = create_jaxoplanet_model(init_state_dic['times'], best_params_s)

    #Calculate bestfit stellar density
    bestfit_stellar_rho_s = ( ( 3 * jnp.pi ) / ( G_cgday * best_params_s['period']**2 ) ) * best_params_s['a']**3

    #Looping over each parameter to perturb it
    for param in perturb_params:

        #Get the perturbation value
        if param=='stellar_rho':
            period_chain = raw_chain_s[:, fixed_args['nburn']:, fixed_args['var_param_list'].index('period')][good_steps_mask_s]
            a_chain = raw_chain_s[:, fixed_args['nburn']:, fixed_args['var_param_list'].index('a')][good_steps_mask_s]
            stellar_rho_chain_s = ( ( 3 * jnp.pi ) / ( G_cgday * period_chain**2 ) ) * a_chain**3
            perturbation = jnp.std(stellar_rho_chain_s)
        else:
            param_chain = raw_chain_s[:, fixed_args['nburn']:, fixed_args['var_param_list'].index(param)][good_steps_mask_s]
            perturbation = jnp.std(param_chain)

        #Adjust the value of the corresponding parameter
        perturbed_state_dic = best_params_s.copy()
        if param=='stellar_rho':
            perturbed_state_dic['a'] = ( ( perturbed_state_dic['period']**2 * (bestfit_stellar_rho_s + perturbation) * G_cgday) / (3 * jnp.pi))**(1/3)
        else:perturbed_state_dic[param] += perturbation

        #Compute the perturbed model and store the flux difference for this seed
        tr_perturbed_LC_s = create_jaxoplanet_model(init_state_dic['times'], perturbed_state_dic)
        perturbation_curves[param].append((bestfit_lc_s - tr_perturbed_LC_s)*1e6)

    #Keep the representative seed's results around for panel B (joint-density contour)
    if seed == plot_seed:
        raw_chain = raw_chain_s
        good_steps_mask = good_steps_mask_s
        best_params = best_params_s
        bestfit_lc = bestfit_lc_s
        bestfit_stellar_rho = bestfit_stellar_rho_s


#Compute the mean and standard deviation of each parameter's perturbation curve across the 10 seeds
perturbation_mean = {param: jnp.mean(jnp.stack(curves), axis=0) for param, curves in perturbation_curves.items()}
perturbation_std  = {param: jnp.std(jnp.stack(curves), axis=0) for param, curves in perturbation_curves.items()}

# Figure 2
print('FIGURE 2')

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
#Looping over each parameter to plot the mean (and across-seed std) of its perturbation curve
for param, paramlabel, color, shape in zip(
                                            perturb_params,
                                            perturb_labels,
                                            plt.get_cmap('coolwarm')(jnp.linspace(0., 1, 8)),
                                            ['.','.','.','.','.','.','.','.']):

    mean_curve = perturbation_mean[param]
    std_curve = perturbation_std[param]

    #Plot the mean difference between nominal and perturbed model, shaded by std across seeds
    axes['A'].plot(plot_phase, mean_curve, shape, color=color, markersize=marksize)
    axes['A'].fill_between(plot_phase, mean_curve - std_curve, mean_curve + std_curve, color=color, alpha=0.3, linewidth=0)
    axes['A'].set_xlim([-0.0425, -0.033])
    axes['C'].plot(plot_phase, mean_curve, shape, color=color, label=paramlabel, markersize=marksize)
    axes['C'].fill_between(plot_phase, mean_curve - std_curve, mean_curve + std_curve, color=color, alpha=0.3, linewidth=0)
    axes['C'].set_xlim([-0.02, 0.02])
    axes['C'].set_ylim([-10, 20])

#Plotting contours for stellar density vs i in top right panel (representative seed only)
# Build flattened sample vectors (ensure 1D arrays) before feeding KDE
period_chain = raw_chain[:, fixed_args['nburn']:, fixed_args['var_param_list'].index('period')][good_steps_mask]
a_chain = raw_chain[:, fixed_args['nburn']:, fixed_args['var_param_list'].index('a')][good_steps_mask]
stellar_rho_chain = ( ( 3 * jnp.pi ) / ( G_cgday * period_chain**2 ) ) * a_chain**3
rho_chain = np.asarray(stellar_rho_chain).ravel()
sqrtesinw_chain = np.asarray(raw_chain[:, fixed_args['nburn']:, fixed_args['var_param_list'].index('sqrtesinw')])[good_steps_mask]

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
plt.savefig(paths.figures / "Fig2.pdf", bbox_inches="tight")
plt.close()