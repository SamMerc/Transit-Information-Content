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
# This file retrieves the chi-squared spaces for pairs of parameters and uses it to Appendix 1, which plots a corner plot of correlations.

######################################
########## Import libraries ##########
######################################
import jax
print(f"JAX devices: {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")
import jax.numpy as jnp
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.constants import G
from squishyplanet.limb_darkening_laws import nonlinear_4param_ld_law
import numpy as np
import os
import paths

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
# Overall LDCs : c1 = 0.5586 c2 = -0.0382 c3 = -0.0852 c4 = 0.0336
init_NLLD_coeffs = nonlinear_4param_ld_law(u1=0.5586, u2=-0.0382, u3=-0.0852, u4=0.0336)

#Updating initial state dictionary
for iLD, LD_coeff in enumerate(init_NLLD_coeffs):
    init_state_dic[f'LD_u{iLD+1}'] = LD_coeff

#Get starting points for the LD coefficients
init_LD_prop = nonlinear_4param_ld_law(u1=0.5586, u2=-0.0382, u3=-0.0852, u4=0.0336, order=3)

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
for key in mod_prop:
    if mod_prop[key]['vary']:
        var_param_list.append(str(key))

#%% Defining dictionary to store additional info. needed for the model
fixed_args={}
fixed_args['var_param_list']=var_param_list
fixed_args['labels'] = [r'R$_p$/R$_{\star}$',r'i (rad)',r'$\rho_{\star}$ (g/cm$^{3}$)',r'u$_1$', r'u$_2$',r'u$_3$',r'P (days)',r'$\sqrt{e}$cos($\omega$)',r'$\sqrt{e}$sin($\omega$)']

#% Define number of points to sample the parameter space with
fixed_args['sample_pts'] = 100

#% Delat chi2 thresholds to include in heatmaps
fixed_args['delta_chi2_thresh'] = 1.0

#% Contour levels 
lvls = np.logspace(np.log10(1), np.log10(1000), base=10, num=10)

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
    
#Helper function to fit an ellipse given points
def fit_ellipse_conic(x, y):
    # Build design matrix
    D = jnp.vstack([x**2, x*y, y**2, x, y, jnp.ones_like(x)]).T
    # Solve normal equations: minimize ||D @ p||²
    _, _, V = jnp.linalg.svd(D)
    p = V[-1]  # solution is last row of V
    return p  # [A, B, C, D, E, F]

#Helper function to calculate ellipse properties given its fit
def ellipse_parameters_from_conic(p):
    a, B, c, D, E, g = p
    b = B/2.
    d = D/2.
    f = E/2.

    #Get center values
    x0 = ( c*d - b*f )/( b*b - a*c )
    y0 = ( a*f - b*d )/( b*b - a*c )

    #Get semi-major and semi-minor axes
    up = 2.0 * (a*f*f + c*d*d + g*b*b - 2.0*b*d*f - a*c*g)
    down1 = (b*b - a*c) * (jnp.sqrt((a - c)*(a - c) + 4.0*b*b) - (a + c))                 
    down2 = (b*b - a*c) * (-jnp.sqrt((a - c)*(a - c) + 4.0*b*b) - (a + c))
    am = jnp.sqrt( up/down1 )
    bm = jnp.sqrt( up/down2 )
    
#Get angle
    if (a == c):  # Circle case
        phi = 0.0
    elif (b == 0):
        if (a < c):
            phi = 0.0
        else:  # a > c
            phi = 0.5 * jnp.pi
    else:  # b != 0 and a != c
        if (a < c):
            phi = 0.5 * jnp.arctan((2.0 * b) / (a - c))
        else:  # a > c
            phi = 0.5 * jnp.pi + 0.5 * jnp.arctan((2.0 * b) / (a - c))

    #Get cardinal points
    # x = xmid
    y_card_low = y0 - jnp.sqrt((am*am*bm*bm) / (bm*bm*jnp.sin(phi)**2 + am*am*jnp.cos(phi)**2))
    y_card_high = y0 + jnp.sqrt((am*am*bm*bm) / (bm*bm*jnp.sin(phi)**2 + am*am*jnp.cos(phi)**2))
    # y = ymid
    x_card_low = x0 - jnp.sqrt((am*am*bm*bm) / (bm*bm*jnp.cos(phi)**2 + am*am*jnp.sin(phi)**2))
    x_card_high = x0 + jnp.sqrt((am*am*bm*bm) / (bm*bm*jnp.cos(phi)**2 + am*am*jnp.sin(phi)**2))

    return {
        "center": (x0, y0),
        "angle_rad": phi,
        "angle_deg": jnp.degrees(phi),
        "semi_major": am,
        "semi_minor": bm,
        'y_card' : (y_card_low, y_card_high),
        'x_card' : (x_card_low, x_card_high),
    }


#############################################
################ Running code ###############
#############################################

fixed_args['save_loc'] = output_dir

print(f"MODEL SCATTER = {model_scatter:.2f}")
print(f"SEED = {seed}")
seed_dir = input_dir+f'{jnp.floor(model_scatter)}ppm/Seed{seed}/'


####################################
##### Chi-squared map plotting #####
####################################

def load_chi2_data(param1, param2, save_loc):
    """
    Load chi-squared data for a parameter pair from individual files.
    Handles both same-parameter (1D) and different-parameter (2D) cases.
    """
    # Try both possible file name combinations
    file1 = os.path.join(save_loc, f"chi2_{param1}_{param2}.npy")
    file2 = os.path.join(save_loc, f"chi2_{param2}_{param1}.npy")
    
    if os.path.exists(file1):
        return jnp.load(file1)
    elif os.path.exists(file2):
        data = jnp.load(file2)
        # If it's a 2D array and params are different, transpose it
        if param1 != param2 and data.ndim == 2:
            return data.T
        return data
    else:
        raise FileNotFoundError(f"Could not find chi2 file for {param1} vs {param2}")

#Loading the MCMC results
print(f'Retrieving MCMC')
raw_chain = jnp.load(seed_dir+"chains.npy")
logprob = jnp.load(seed_dir+"logprob.npy")
_, _, _, _, _, _, _, _, good_steps_mask = load_result((input_dir, model_scatter, seed, True))

#Finding the index of max log-probability
max_step, max_walker = jnp.unravel_index(jnp.argmax(logprob), logprob.shape)

n_params = len(fixed_args['var_param_list'])

#Initializing correlation matrix
corr_matrix = np.zeros((n_params, n_params))

#Initializing figure
fig, axes = plt.subplots(n_params, n_params, figsize=(2.5 * n_params, 2.5 * n_params))

for i, param1 in enumerate(fixed_args['var_param_list']):
    for j, param2 in enumerate(fixed_args['var_param_list']):
        ax = axes[i, j]
        ax.tick_params(labelsize=14)
        print(param1, ' vs ', param2)

        # remove top/right spines for nicer look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if j > i:
            ax.axis('off')  # upper triangle: turn off
            continue

        # Diagonal: 1D chi2 line plot
        if i == j:

            # Load chi2 data from file
            chi2_vals = load_chi2_data(param1, param1, fixed_args['save_loc'])

            param_vals = jnp.linspace(
                raw_chain[max_walker, max_step, i] - jnp.std(raw_chain[:, fixed_args['nburn']:, i][good_steps_mask]),
                raw_chain[max_walker, max_step, i] + jnp.std(raw_chain[:, fixed_args['nburn']:, i][good_steps_mask]),
                fixed_args['sample_pts']
            )

            ax.plot(param_vals, chi2_vals, color='black')
            ax.axvline(raw_chain[max_walker, max_step, i], color='red', linestyle='--', lw=1)
            ax.set_yticklabels([])
            ax.set_xticklabels([])

        # Lower triangle: 2D chi2 contours (filled)
        else:
            # Use param2 on x-axis, param1 on y-axis (lower triangle convention)
            param2_vals = jnp.linspace(
                raw_chain[max_walker, max_step, j] - jnp.std(raw_chain[:, fixed_args['nburn']:, j][good_steps_mask]),
                raw_chain[max_walker, max_step, j] + jnp.std(raw_chain[:, fixed_args['nburn']:, j][good_steps_mask]),
                fixed_args['sample_pts']
            )
            param1_vals = jnp.linspace(
                raw_chain[max_walker, max_step, i] - jnp.std(raw_chain[:, fixed_args['nburn']:, i][good_steps_mask]),
                raw_chain[max_walker, max_step, i] + jnp.std(raw_chain[:, fixed_args['nburn']:, i][good_steps_mask]),
                fixed_args['sample_pts']
            )

            #Get normalized arrays
            norm_param2_vals = jnp.linspace(
                - 1,
                + 1,
                fixed_args['sample_pts']
            )
            norm_param1_vals = jnp.linspace(
                - 1,
                + 1,
                fixed_args['sample_pts']
            )

            key1 = f'{param1}_{param2}'
            key2 = f'{param2}_{param1}'

            # Load chi2 data from file
            chi2_grid = load_chi2_data(param1, param2, fixed_args['save_loc'])
            
            A, B = np.meshgrid(param2_vals, param1_vals)  # (x=param2, y=param1)
            norm_A, norm_B = np.meshgrid(norm_param2_vals, norm_param1_vals)  # (x=param2, y=param1)

            # Define three nicely spaced contour levels (multiples of the delta threshold)
            contour_levels = []
            for lvl in lvls:
                lv = lvl * fixed_args['delta_chi2_thresh']
                contour_levels.append(lv)

            # Filled contours
            fill_levels = [0.0] + list(lvls)
            fill_colors = plt.get_cmap('Greens')(jnp.linspace(0., 1, len(lvls)))  # light -> dark green
            ax.contourf(A, B, chi2_grid, levels=fill_levels, colors=fill_colors, alpha=0.5, antialiased=True)

            # Outline the three contours with black lines
            cs = ax.contour(A, B, chi2_grid, levels=contour_levels, colors=['black']*len(contour_levels), linewidths=0.8)

            # Also compute normalized contours for later geometry fitting (do not rely on plotted result)
            norm_cs = ax.contour(norm_A, norm_B, chi2_grid, levels=contour_levels, colors=['none']*len(contour_levels), linewidths=0)

            ax.axvline(raw_chain[max_walker, max_step, j], color='black', linestyle='--', lw=1)
            ax.axhline(raw_chain[max_walker, max_step, i], color='black', linestyle='--', lw=1)

            # Find largest valid contour fully within bounds using the computed contour objects
            valid_contours = []

            # Get axis bounds
            x_bounds = (param2_vals[0], param2_vals[-1])
            y_bounds = (param1_vals[0], param1_vals[-1])

            for norm_contour_level, contour_level in zip(norm_cs.allsegs, cs.allsegs):
                for norm_segment, segment in zip(norm_contour_level, contour_level):
                    if len(segment) < 15:
                        continue
                    x, y = segment[:, 0], segment[:, 1]

                    # Check if entire segment lies within bounds
                    if (jnp.all((x >= x_bounds[0]) & (x <= x_bounds[1])) and
                        jnp.all((y >= y_bounds[0]) & (y <= y_bounds[1]))):
                        valid_contours.append((segment, norm_segment))

            # Select the longest valid contour
            if valid_contours:
                selected, norm_selected = max(valid_contours, key=lambda seg_pair: len(seg_pair[0]))
            else:
                # If no valid contour found, fall back to the longest available contour in cs
                all_segments = []
                for level_segs in cs.allsegs:
                    for seg in level_segs:
                        all_segments.append((seg, None))
                if len(all_segments) == 0:
                    raise KeyError('Need bigger contours')
                selected, norm_selected = max(all_segments, key=lambda seg_pair: len(seg_pair[0]))

            #Retrieving contours
            contour_x, contour_y = selected[:, 0], selected[:, 1]
            if norm_selected is None:
                # build a dummy normalized contour by scaling to unit stds if we don't have normalized seg
                norm_contour_x = (contour_x - raw_chain[max_walker, max_step, j]) / jnp.std(raw_chain[max_walker, fixed_args['nburn']:, j][good_steps_mask])
                norm_contour_y = (contour_y - raw_chain[max_walker, max_step, i]) / jnp.std(raw_chain[max_walker, fixed_args['nburn']:, i][good_steps_mask])
            else:
                norm_contour_x, norm_contour_y = norm_selected[:, 0], norm_selected[:, 1]

            #Fit the normalized contour to retrieve slope and correlation value
            ##Fit ellipse
            bestfit_norm_ellipse = fit_ellipse_conic(norm_contour_x, norm_contour_y)
            ##Retrieve properties
            ellipse_norm_params = ellipse_parameters_from_conic(bestfit_norm_ellipse)
            ##Plot semi-major axis
            norm_orig_a = ellipse_norm_params['semi_major']
            norm_orig_b = ellipse_norm_params['semi_minor']
            norm_xmid, norm_ymid = ellipse_norm_params['center']
            norm_theta = ellipse_norm_params['angle_rad']
            ##Swap axes if necessary
            if norm_orig_a < norm_orig_b:
                norm_a = norm_orig_b
                norm_theta += jnp.pi/2
            else:norm_a = norm_orig_a
            ##Plot the slope
            norm_x0, norm_x1 = norm_xmid - 1 * (norm_a*jnp.cos(norm_theta)), norm_xmid + 1 * (norm_a*jnp.cos(norm_theta))
            norm_y0, norm_y1 = norm_ymid - 1 * (norm_a*jnp.sin(norm_theta)), norm_ymid + 1 * (norm_a*jnp.sin(norm_theta))
            ##Getting correlation value
            norm_slope = (norm_y1-norm_y0)/(norm_x1-norm_x0)   
            corr_value = jnp.abs( jnp.sin(2 * jnp.arctan(norm_slope)) )     

            if jnp.isnan(corr_value):
                # fallback: fit line to all points in largest contour
                largest_idx = 0
                largest_size = 0
                for seg_idx, seg in enumerate(cs.allsegs):
                    if len(seg[0][:,0]) > largest_size:
                        largest_size = len(seg[0][:,0])
                        largest_idx = seg_idx

                x, y = cs.allsegs[largest_idx][0][:, 0], cs.allsegs[largest_idx][0][:, 1]
                slope, intercept = jnp.polyfit(x, y, 1)
                corr_value = jnp.abs(jnp.sin(2 * jnp.arctan(slope)))

            #Populate correlation matrix
            corr_matrix[i, j] = corr_value

            if i == n_params - 1:
                ax.set_xlabel(fixed_args['labels'][j],fontsize=14)
                ax.tick_params(axis="x", labelsize=12, rotation=45)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(fixed_args['labels'][i],fontsize=14)
                ax.tick_params(axis="y", labelsize=12, rotation=45)
            else:
                ax.set_yticklabels([])

            #Force the limits of the plot
            ax.set_xlim([param2_vals[0], param2_vals[-1]])
            ax.set_ylim([param1_vals[0], param1_vals[-1]])

fig.tight_layout()
plt.savefig(paths.figures / "Appendix1.pdf", bbox_inches="tight")
plt.close(fig)
