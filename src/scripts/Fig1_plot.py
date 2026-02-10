#############################
########## Purpose ##########
#############################

# Figure 1 showcases the amplification factor change across model scatters and limb-darkening law used (aka the dimensionality).
# The goal of this file is to retrieve the results from the injection-retrievals computed previously and compile them into Figure 1.


######################################
########## Import libraries ##########
######################################
from jax import random, jit, vmap
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
import numpy as np
import os
from scipy.stats import norm
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
import time
from tqdm import tqdm
import pickle
from multiprocessing import Pool, cpu_count
import gc
import corner

# For 64-bit precision since JAX defaults to 32-bit
jax.config.update("jax_enable_x64", True)

#############################################
########## Define hyper-parameters ##########
#############################################
#%% Mock light curve

#%%%% Define G in units needed now to avoid JAX tracing issues
G_solar_units = G.to(u.Rsun**3 / (u.Msun * u.day**2)).value
R_star = (1.0 * u.R_sun).value
#%%%% Mock system - fiducial
init_state_dic = {}
init_state_dic['period'] = 1.                                 #days
a_meters = ( (G.value * (1.0 * u.M_sun).to(u.kg).value * (init_state_dic['period'] * 24 * 3600)**2)/(4 * jnp.pi**2) )**(1/3)  
init_state_dic['a'] = a_meters / (1.0 * u.R_sun).to(u.m).value  #stellar radius
init_state_dic['r'] = 0.1                                     #stellar radius
init_state_dic['i'] = jnp.deg2rad(90)                         #radians
init_state_dic['omega'] = 0.0                                 #radians
init_state_dic['e'] = 0.                                      #unitless
init_state_dic['t0'] = 0.0                                    #days


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


# Calculate IT points once
num_IT_pts = jnp.sum(((init_state_dic['times'] > init_state_dic['t0'] - T_dur/2) & 
                        (init_state_dic['times'] < init_state_dic['t0'] + T_dur/2)))

#%% Location of MCMC results and where plot will be output
raw_save_dir = '/Volumes/Pandora/Work/PhD/Research/TIC/Gen_Storage/Fig1_Storage/'

#%% Model parameters
mod_prop = {
    'r'         : {'vary':True, 'guess':0.11, 'bounds':[0.07, 0.15]},
    'i'         : {'vary':True, 'guess':jnp.deg2rad(88.5), 'bounds':[jnp.deg2rad(88.), jnp.deg2rad(92.)]},
    'a'         : {'vary':True, 'guess':init_state_dic['a']-1, 'bounds':[init_state_dic['a']-2, init_state_dic['a']+2]},
    'period'    : {'vary':True, 'guess':1., 'bounds':[0.9995, 1.0005]},
    'sqrtecosw' : {'vary':True, 'guess': 0., 'bounds': [-0.2, 0.2]},
    'sqrtesinw' : {'vary':True, 'guess': 0., 'bounds': [-0.2, 0.2]},
    't0'        : {'vary':False, 'guess':0., 'bounds':[-100,100]},
    'LD_u1'     : {'vary':True, 'guess':0., 'bounds':[-0.4, 0.7]},
    'LD_u2'     : {'vary':True, 'guess':0.1, 'bounds':[-0.3, 0.7]},
    'LD_u3'     : {'vary':True, 'guess':0.3, 'bounds':[-0.1, 0.9]},
}

#%% Fitting mode
fixed_args={}

#%% Number of burn-in steps
fixed_args['nburn'] = 400000

# Optimization
# Set number of cpus to use
num_workers = int(0.5 * cpu_count())
# Number of files in each chunk
CHUNK_SIZE = 2

# Filtering parameters
THRESHOLD = 4.0  # Number of IQRs for outlier detection (5 is conservative)
step_threshold = 0.1
ROUNDS = 3

# Diagnostic plotting parameters
ENABLE_DIAGNOSTICS = True  # Set to False to skip diagnostic plots

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
fixed_args['var_param_list']=var_param_list
fixed_args['fix_param_list']=fix_param_list
fixed_args['fix_param_val']=fix_param_val

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

def load_result(args):
    """
    Optimized file loading with step-by-step sigma clipping
    
    NEW FILTERING STRATEGY:
    - At EACH step, calculate median and sigma across all walkers
    - Flag walkers that are outliers at that step
    
    This catches walkers that diverge at any point, not just those with
    different overall medians.
    """
    raw_save_dir, PLD_order, model_scatter, seed, return_full = args
    print(f"  Processing PLD{PLD_order}, scatter{np.floor(model_scatter)}, seed{seed}...")
    try:
        path_base = f'{raw_save_dir}PLD_{PLD_order}/{np.floor(model_scatter)}ppm/Seed{seed}/'
        
        # CRITICAL: Use memory mapping instead of loading entire file
        raw_chain = np.load(path_base + 'chains.npy', mmap_mode='r')
        logprob = np.load(path_base + 'logprob.npy', mmap_mode='r')
        chi2 = np.load(path_base + 'chi2_chain.npy', mmap_mode='r')
        n_walkers, n_steps, n_params = raw_chain.shape
        n_steps_post = n_steps - fixed_args['nburn']

        # Initialize: START tracking which ORIGINAL walkers are good
        good_walker_mask = np.ones(n_walkers, dtype=bool)

        # Iterate over the number of sigma clipping rounds desired:
        for round in range(ROUNDS):
            
            print(f'ROUND {round+1}')
            round_chi2 = chi2[good_walker_mask, :]
            round_raw_chain = raw_chain[good_walker_mask, :, :]
            
            # Get the number of good walkers in THIS round
            n_round_walkers = round_chi2.shape[0]

            # ======================================================================
            # FILTER 1: CHI2-BASED STEP-BY-STEP SIGMA CLIPPING
            # ======================================================================            
            # Process post-burn-in steps only
            chi2_post_burnin = round_chi2[:, fixed_args['nburn']:]
            
            # For each step, identify outliers
            chi2_step_outlier_mask = np.zeros((n_round_walkers, n_steps_post), dtype=bool)
            
            for step_idx in range(n_steps_post):
                chi2_at_step = chi2_post_burnin[:, step_idx]  # (round_walkers,)
                
                # Calculate median and sigma at this step
                quartiles = np.percentile(chi2_at_step, [25, 50, 75])
                mu, sig = quartiles[1], 0.74 * (quartiles[2] - quartiles[0])
                
                # Identify outliers at this step
                lower_bound = mu - THRESHOLD * sig
                upper_bound = mu + THRESHOLD * sig
                
                chi2_step_outlier_mask[:, step_idx] |= ((chi2_at_step < lower_bound) | (chi2_at_step > upper_bound))

            # Only keep the masked steps if more than X% of the steps are bad. X is set by the step threshold
            chi2_outlier_mask = np.sum(chi2_step_outlier_mask, axis=1) > (step_threshold * n_steps_post)

            print(f"    Chi2 filter: removed {np.sum(chi2_outlier_mask)} walker(s) ({100*(np.sum(chi2_outlier_mask)/n_round_walkers):.1f}%)")

            # ======================================================================
            # FILTER 2: PARAMETER SPACE STEP-BY-STEP SIGMA CLIPPING
            # ======================================================================
            print(f"  Processing parameters...")
            
            # Load post-burn-in samples
            post_burnin_samples = np.array(round_raw_chain[:, fixed_args['nburn']:, :])
            
            # Check each parameter independently
            for param_idx in range(n_params):
                param_chain = post_burnin_samples[:, :, param_idx]  # (round_walkers, n_steps_post)
                
                # For each step, identify outliers
                param_step_outlier_mask = np.zeros((n_round_walkers, n_steps_post), dtype=bool)
                
                for step_idx in range(n_steps_post):
                    param_at_step = param_chain[:, step_idx]  # (round_walkers,)
                    
                    # Calculate median and sigma at this step
                    quartiles = np.percentile(param_at_step, [25, 50, 75])
                    mu, sig = quartiles[1], 0.74 * (quartiles[2] - quartiles[0])
                    
                    # Identify outliers at this step
                    lower_bound = mu - THRESHOLD * sig
                    upper_bound = mu + THRESHOLD * sig
                    
                    param_step_outlier_mask[:, step_idx] |= (param_at_step < lower_bound) | (param_at_step > upper_bound)
                
                # Only keep the masked steps if more than X% of the steps are bad. X is set by the step threshold
                param_outlier_mask = np.sum(param_step_outlier_mask, axis=1) > (step_threshold * n_steps_post)

                print(f"    {fixed_args['var_param_list'][param_idx]} filter: removed {np.sum(param_outlier_mask)} walker(s) ({100*(np.sum(param_outlier_mask)/n_round_walkers):.1f}%)")

            # ======================================================================
            # BUILD GOOD WALKERS - MAINTAIN ORIGINAL WALKER INDICES
            # ======================================================================
            round_bad_walkers = (param_outlier_mask | chi2_outlier_mask)
            round_good_indices = np.where(~round_bad_walkers)[0]
            
            # Get the ORIGINAL indices of good walkers
            original_good_indices = np.where(good_walker_mask)[0]
            original_good_indices_to_keep = original_good_indices[round_good_indices]
            
            # Update good_walker_mask with only the walkers that passed this round
            good_walker_mask = np.zeros(n_walkers, dtype=bool)
            good_walker_mask[original_good_indices_to_keep] = True
            
            print(f"    Total filter: removed {np.sum(round_bad_walkers)} walker(s) ({100*(np.sum(round_bad_walkers)/n_round_walkers):.1f}%)")

        # ===================================================================
        # EXTRACT DATA FROM GOOD WALKERS
        # ===================================================================
        
        # Find best fit from good walkers only
        good_logprob = logprob[good_walker_mask, :]
            
        max_walker_idx, max_step = np.unravel_index(np.argmax(good_logprob), good_logprob.shape)

        # Map back to original walker index
        good_walker_indices = np.where(good_walker_mask)[0]
        max_walker = good_walker_indices[max_walker_idx]

        # Only load post burn-in data for parameter 0
        r_chain_post_burnin = np.array(raw_chain[good_walker_mask, fixed_args['nburn']:, 0])
        bestfit_r = float(raw_chain[max_walker, max_step, 0])
        
        if return_full:
            full_chain = np.array(raw_chain)
            full_logprob = np.array(logprob)
            full_chi2 = np.array(chi2)
            
            return (PLD_order, model_scatter, seed, r_chain_post_burnin, bestfit_r, 
                   True, full_chain, full_logprob, full_chi2, good_walker_mask)
        else:
            return (PLD_order, model_scatter, seed, r_chain_post_burnin, bestfit_r, 
                   False, None, None, None, None)
    
    except Exception as e:
        print(f"Error loading PLD{PLD_order}, scatter{model_scatter}, seed{seed}: {e}")
        import traceback
        traceback.print_exc()
        return None


def plot_diagnostics(full_chain, full_logprob, full_chi2, good_walkers, 
                     PLD_order, model_scatter, seed):
    """
    Create comprehensive diagnostic plots after each chunk
    
    Plots:
    1. Trace plots (all parameters) - pre-filtering in red, post-filtering in blue
    2. Chi2 evolution - pre-filtering in red, post-filtering in blue
    3. Corner plot (post burn-in) - BLACK: pre-filtering, RED: post-filtering
       with amplification factors displayed
    
    Parameters:
    -----------
    full_chain : ndarray
        Full MCMC chains (all walkers, all steps, all params)
    full_logprob : ndarray
        Log probability for all walkers
    full_chi2 : ndarray
        Chi-squared for all walkers
    good_walkers : ndarray of bool or indices
        Walkers that passed filtering
    """
    n_walkers, n_steps, n_params = full_chain.shape
    nburn = fixed_args['nburn']
    
    # Create output directory for diagnostics
    diag_dir = f'{raw_save_dir}PLD_{PLD_order}/{np.floor(model_scatter)}ppm/Seed{seed}/diagnostics/'
    os.makedirs(diag_dir, exist_ok=True)
    
    print(f"    Generating diagnostics for PLD{PLD_order}, scatter{model_scatter}, seed{seed}")
    
    n_good = np.sum(good_walkers)
    n_bad = n_walkers - n_good

    # =========================================================================
    # PLOT 1: Trace plots for all parameters with histograms
    # =========================================================================
    param_names = ['r', 'i', 'a', 'period', 'sqrtecosw', 'sqrtesinw'] + [f'LD_u{i}' for i in range(1, PLD_order+1)]

    # Create figure with 2 columns: traces and histograms
    fig_trace = plt.figure(figsize=(16, 2*n_params))
    gs = fig_trace.add_gridspec(n_params, 3, width_ratios=[1,1,1], hspace=0.05, wspace=0.05)

    walker_param_medians = np.median(full_chain[:, nburn:, :], axis=1)

    for i, param_name in enumerate(param_names):
        # Left column: Trace plot
        ax_trace = fig_trace.add_subplot(gs[i, 0])
        
        # Plot good walkers: burn-in in orange, post-burn-in in blue
        for walker in range(n_walkers):
            if not good_walkers[walker]:
                ax_trace.plot(full_chain[walker, :, i], color='red', alpha=0.2, linewidth=0.2)
            else:
                ax_trace.plot(np.arange(nburn), full_chain[walker, :nburn, i], 
                            color='red', alpha=0.2, linewidth=0.2, label=f'Burn-in {n_good} & Filtered({n_bad})' if i==0 else '')
                ax_trace.plot(np.arange(nburn, n_steps), full_chain[walker, nburn:, i], 
                            color='blue', alpha=0.2, linewidth=0.2, label=f'Post burn-in ({n_good})')
        
        ax_trace.axvline(nburn, color='black', linestyle='--', linewidth=0.2, alpha=0.2, label='Burn-in cutoff' if i==0 else '')
        ax_trace.set_ylabel(param_name, fontsize=10)
        ax_trace.grid(True, alpha=0.3)
        
        if i == 0:ax_trace.legend(loc='upper right', fontsize=8)
        if i < n_params - 1:
            ax_trace.set_xticklabels([])
        else:
            ax_trace.set_xlabel('Step', fontsize=10)
        
        # Middle column: Trace plot (post-burn)
        ax_burn = fig_trace.add_subplot(gs[i, 1])
        
        # Plot walkers
        for walker in range(n_walkers):
            if not good_walkers[walker]:
                ax_burn.plot(full_chain[walker, nburn:, i], color='red', alpha=0.2)
            else:
                ax_burn.plot(full_chain[walker, nburn:, i], color='blue', alpha=0.2)

        # ax_burn.tick_params(axis='y', labelleft=False)
        ax_burn.set_ylabel(param_name, fontsize=10)
        ax_burn.grid(True, alpha=0.3)

        # Right column: Horizontal histogram
        ax_hist = fig_trace.add_subplot(gs[i, 2], sharey=ax_burn)
        
        # Plot horizontal histogram
        ax_hist.hist(full_chain[:, nburn:, i].flatten(), bins=30, orientation='horizontal', 
                    density=True, histtype='stepfilled', color='red', alpha=0.5, edgecolor='red', linewidth=0.5)
        ax_hist.hist(full_chain[good_walkers, nburn:, i].flatten(), bins=30, orientation='horizontal', 
                    density=True, histtype='stepfilled', color='blue', alpha=0.5, edgecolor='blue', linewidth=0.5)

        ax_hist.set_xlabel('Count', fontsize=8)
        # ax_hist.tick_params(axis='y', labelleft=False)
        ax_hist.grid(True, alpha=0.3, axis='x')

    fig_trace.suptitle(f'Trace Plots - PLD{PLD_order}, scatter={model_scatter:.1f}, seed={seed}', 
                    fontsize=12)
    plt.savefig(os.path.join(diag_dir, f'trace.pdf'), 
                dpi=150, bbox_inches='tight')
    plt.close(fig_trace)
    
    # =========================================================================
    # PLOT 2: Chi2 evolution
    # =========================================================================
    fig_chi2, ax_chi2 = plt.subplots(1, 1, figsize=(12, 4))
    
    # Plot filtered-out walkers in red
    for walker in range(n_walkers):
        if not good_walkers[walker]:
            ax_chi2.loglog(full_chi2[walker, :], color='red', alpha=0.5, linewidth=0.8)
        else:
            ax_chi2.loglog(np.arange(nburn), full_chi2[walker, :nburn], 
                        color='red', alpha=0.3, linewidth=0.5)
            ax_chi2.loglog(np.arange(nburn, n_steps), full_chi2[walker, nburn:], 
                        color='blue', alpha=0.3, linewidth=0.5)
    
    ax_chi2.axvline(nburn, color='black', linestyle='--', linewidth=2, alpha=0.5, label='Burn-in')
    ax_chi2.set_xlabel('Step', fontsize=10)
    ax_chi2.set_ylabel('Chi-squared', fontsize=10)
    ax_chi2.set_yscale('log')
    ax_chi2.grid(True, alpha=0.3)
    ax_chi2.legend(loc='upper right')
    ax_chi2.set_title(f'Chi2 Evolution - PLD{PLD_order}, scatter={model_scatter:.1f}, seed={seed}', 
                     fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(diag_dir, f'chi2.pdf'), 
               dpi=150, bbox_inches='tight')
    plt.close(fig_chi2)
    
    # =========================================================================
    # PLOT 3: Corner plot with pre/post filtering comparison
    # =========================================================================
    
    # Calculate amplification factors for both cases
    
    # PRE-FILTERING: Use all walkers
    all_chains_post_burnin = full_chain[:, nburn:, :]
    samples_pre_filter = all_chains_post_burnin.reshape(-1, n_params)
    
    # Calculate pre-filtering amplification factor
    # Extract r parameter (assumed to be index 0)
    r_chain_pre = full_chain[:, nburn:, 0].flatten()
    max_idx_pre = np.unravel_index(np.argmax(full_logprob), full_logprob.shape)
    bestfit_r_pre = full_chain[max_idx_pre[0], max_idx_pre[1], 0]
    
    std_r_pre = np.std(r_chain_pre)
    bestfit_r_error_pre = 2 * std_r_pre * bestfit_r_pre
    scatter_in_bin = (model_scatter * 1e-6) / np.sqrt(num_IT_pts)
    amp_factor_pre = bestfit_r_error_pre / scatter_in_bin
    
    # POST-FILTERING: Use only good walkers
    good_chains_post_burnin = full_chain[good_walkers, nburn:, :]
    samples_post_filter = good_chains_post_burnin.reshape(-1, n_params)
    
    # Calculate post-filtering amplification factor
    r_chain_post = full_chain[good_walkers, nburn:, 0].flatten()
    good_logprob = full_logprob[good_walkers, :]
    max_idx_post = np.unravel_index(np.argmax(good_logprob), good_logprob.shape)
    bestfit_r_post = full_chain[good_walkers, :, :][max_idx_post[0], max_idx_post[1], 0]
    
    std_r_post = np.std(r_chain_post)
    bestfit_r_error_post = 2 * std_r_post * bestfit_r_post
    amp_factor_post = bestfit_r_error_post / scatter_in_bin
    
    # Create corner plot with both distributions
    fig_corner = corner.corner(
        samples_pre_filter, 
        labels=param_names,
        color='black',
        quantiles=[0.16, 0.5, 0.84],
        show_titles=False,
        plot_datapoints=False,
        plot_density=True,
        hist_kwargs={'alpha': 0.6, 'linewidth': 2},
        contour_kwargs={'linewidths': 1.5, 'alpha': 0.6}
    )
    
    # Overlay post-filtering distribution in red
    corner.corner(
        samples_post_filter,
        fig=fig_corner,
        labels=param_names,
        color='red',
        quantiles=[0.16, 0.5, 0.84],
        show_titles=False,
        plot_datapoints=False,
        plot_density=True,
        hist_kwargs={'alpha': 0.8, 'linewidth': 2},
        contour_kwargs={'linewidths': 2, 'alpha': 0.8},
        reverse=True
    )
    
    # Add title with amplification factors
    title_text = (
        f'Corner Plot - PLD{PLD_order}, scatter={model_scatter:.1f} ppm, seed={seed}\n'
        f'BLACK: Pre-filtering (all {n_walkers} walkers) - A = {amp_factor_pre:.2f}\n'
        f'RED: Post-filtering ({n_good} walkers) - A = {amp_factor_post:.2f}\n'
        f'Improvement: {((amp_factor_pre - amp_factor_post)/amp_factor_pre * 100):.1f}% reduction in A'
    )
    fig_corner.suptitle(title_text, fontsize=11, y=1.0)
    
    # Add custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='black', alpha=0.6, label=f'Pre-filter: A={amp_factor_pre:.2f}'),
        Patch(facecolor='red', alpha=0.8, label=f'Post-filter: A={amp_factor_post:.2f}')
    ]
    fig_corner.legend(handles=legend_elements, loc='upper right', fontsize=10, 
                     bbox_to_anchor=(0.95, 0.95))
    
    plt.savefig(os.path.join(diag_dir, f'corner.pdf'), 
               dpi=150, bbox_inches='tight')
    plt.close(fig_corner)

# OPTIMIZATION 2: JAX-optimized computation
@jit
def compute_amplification_factor_jax(r_chain_flat, bestfit_r, model_scatter, num_IT_pts):
    """JIT-compiled amplification factor calculation"""
    std_r = jnp.std(r_chain_flat)
    bestfit_r_error = 2 * std_r * bestfit_r
    scatter_in_bin = (model_scatter * 1e-6) / jnp.sqrt(num_IT_pts)
    return bestfit_r_error / scatter_in_bin


def batch_compute_amplification_factors(results_batch, num_IT_pts):
    """
    Process multiple results in batch using JAX vectorization
    
    This processes all chains for a given (PLD_order, model_scatter) at once
    """
    amp_factors = []
    
    for _, _, _, r_chain_post_burnin, bestfit_r in results_batch:
    
        # Get model_scatter from first result
        model_scatter = results_batch[0][1]
    
        # Flatten the entire filtered chain for this seed
        # r_chain_post_burnin shape: (n_good_walkers, n_steps_post_burnin)
        r_chain_flat = r_chain_post_burnin.flatten()

        # Convert to JAX arrays and compute
        r_chain_jax = jnp.array(r_chain_flat)
        amp_factor = compute_amplification_factor_jax(
            r_chain_jax, bestfit_r, model_scatter, num_IT_pts
        )
        amp_factors.append(amp_factor)

    return amp_factors

#############################################
################ Running code ###############
#############################################

if __name__ == '__main__':

    #####################
    #### Optimization ###
    #####################
    model_scatters = [0.1, 1, 10, 16.68100537200059, 27.825594022071243, 46.41588833612777, 77.4263682681127,
                    129.1549665014884, 215.44346900318823, 359.38136638046257, 599.4842503189409, 1000.0, 3000.0, 10000.0]
    seeds = [40, 50, 60, 70, 80, 90, 100, 110, 120, 130]
    PLD_orders = [2, 3, 4]

    #Create cache file to avoid reloading data
    cache_file = raw_save_dir + 'processed_data_cache.pkl'
    t_start = time.time()

    if os.path.exists(cache_file):
        print("Loading cached processed data...")
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
        print(f"Cache loaded in {time.time() - t_start:.2f} seconds")
    else:
        print("No cache found. Loading and processing all data...")
        
        # Prepare all loading tasks
        loading_tasks = []
        for PLD_order in PLD_orders:
            for model_scatter in model_scatters:
                for seed in seeds:
                    loading_tasks.append((raw_save_dir, PLD_order, model_scatter, seed))
        
        total_files = len(loading_tasks)
        print(f"Loading {total_files} files ...")
        
        # Process in chunks to reduce memory pressure
        num_chunks = (total_files + CHUNK_SIZE - 1) // CHUNK_SIZE

        print(f"Using {num_workers} CPU cores")

        # Process all chunks
        results = []
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * CHUNK_SIZE
            chunk_end = min((chunk_idx + 1) * CHUNK_SIZE, total_files)
            chunk_tasks = loading_tasks[chunk_start:chunk_end]
            
            # Do full data extraction for diagnostics
            chunk_tasks_enhanced = [(*task, ENABLE_DIAGNOSTICS) for task in chunk_tasks]

            print(f"\nProcessing chunk {chunk_idx+1}/{num_chunks} ({len(chunk_tasks)} files)...")
            chunk_start_time = time.time()
            
            # Load chunk with progress bar            
            with Pool(processes=num_workers) as pool:
                chunk_results = []
                for result in tqdm(pool.imap_unordered(load_result, chunk_tasks_enhanced),
                                 total=len(chunk_tasks),
                                 desc=f"  Loading files",
                                 ncols=80,
                                 unit="file"):
                    if result is not None:
                        chunk_results.append(result)
            
            results.extend(chunk_results)
            
            chunk_time = time.time() - chunk_start_time
            files_per_sec = len(chunk_tasks) / chunk_time
            print(f"  Chunk processed in {chunk_time:.1f}s ({files_per_sec:.1f} files/sec)")
            
            # Generate diagnostic plots for first file in chunk
            if ENABLE_DIAGNOSTICS:
                for r in chunk_results:
                    PLD, scatter, sd, _, _, _, full_chain, full_logp, full_chi2, good_walkers = r
                    
                    plot_diagnostics(full_chain, full_logp, full_chi2, good_walkers,
                                    PLD, scatter, sd)

            # Force garbage collection between chunks
            gc.collect()

        loading_time = time.time() - t_start
        print(f"\n{'='*70}")
        print(f"All files loaded in {loading_time:.1f}s ({loading_time/60:.2f} min)")
        print(f"Successfully loaded {len(results)}/{total_files} files")
        print(f"{'='*70}\n")
        
        # Organize and compute amplification factors with JAX
        print("Computing amplification factors with JAX vectorization...")
        computation_start = time.time()

        cached_data = {}
        for PLD_order in PLD_orders:
            cached_data[PLD_order] = {}
            print(f"\nProcessing PLD order {PLD_order}")
            for model_scatter in model_scatters:
                print(f"\nProcessing Model scatter {model_scatter:.0f}")
                batch_data = []
                # Filter results for this specific PLD_order and model_scatter
                for result in results:
                    result_pld, result_scatter, seed, r_chain_post_burnin, bestfit_r = result
                    if result_pld == PLD_order and result_scatter == model_scatter:
                        batch_data.append(result)

                if len(batch_data) > 0:
                    # Use batch processing with JAX
                    amp_factors = batch_compute_amplification_factors(
                        batch_data, num_IT_pts
                    )
                    cached_data[PLD_order][model_scatter] = np.array(amp_factors)
                else:
                    cached_data[PLD_order][model_scatter] = np.array([])
        
        computation_time = time.time() - computation_start
        total_time = time.time() - t_start
        
        print(f"\n{'='*70}")
        print("TIMING SUMMARY")
        print(f"{'='*70}")
        print(f"  Data loading:      {loading_time:8.1f}s ({loading_time/60:6.2f} min)")
        print(f"  JAX computation:   {computation_time:8.1f}s ({computation_time/60:6.2f} min)")
        print(f"  {'─'*68}")
        print(f"  Total time:        {total_time:8.1f}s ({total_time/60:6.2f} min)")
        print(f"{'='*70}")

        # Save cache for future runs
        print("Saving cache...")
        with open(cache_file, 'wb') as f:
            pickle.dump(cached_data, f)
        print("Cache saved!")

    ##################
    #### Plotting ####
    ##################
    t_plot_start = time.time()

    PLD_order = 3
    model_scatter = 359.38136638046257
    seed = 100
    #Setting base LDCs
    init_PLD_coeffs = [0.1, 0.2, 0.4]

    #Updating initial state dictionary
    for iLD, LD_coeff in enumerate(init_PLD_coeffs):
        init_state_dic[f'LD_u{iLD+1}'] = LD_coeff
        
    #Pure data
    true_lc = create_jaxoplanet_model(init_state_dic['times'], init_state_dic)

    #Build noisy data
    std = model_scatter * 1e-6
    noisy_LC = true_lc + std * random.normal(jax.random.PRNGKey(seed), shape=true_lc.shape)
    noisy_std = std * jnp.ones(true_lc.shape, dtype=float)

    #Loading MCMC results
    raw_chain = jnp.load(raw_save_dir+f'PLD_{PLD_order}/{jnp.floor(model_scatter)}ppm/Seed{seed}/chains.npy')

    #Burning chains
    burnt_chains = jnp.copy(raw_chain[:, fixed_args['nburn']:, :])

    #Starting to plot
    fig = plt.figure(figsize=(17, 7))
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[1, 2.2],
                        wspace=0.1, hspace=0.2)

    ax_left = fig.add_subplot(gs[0, 0])      # top-left: time-series / sampled lightcurves
    ax_right = fig.add_subplot(gs[0, 1])     # right: full-height plot (spans both rows)

    # --- Left-hand plot ---
    print('LEFT-HAND PLOT')

    ax_left.errorbar(init_state_dic['times'][::52], noisy_LC[::52], yerr=noisy_std[::52], fmt='.', color='black', alpha=0.4, markersize=12, zorder=2)

    #Retrieving samples
    param_flatten_chain = np.reshape(burnt_chains, (burnt_chains.shape[0]*burnt_chains.shape[1], burnt_chains.shape[2]))
    n_total = param_flatten_chain.shape[0]
    chosen_indices = np.random.choice(n_total, size=25, replace=False)
    param_samples = param_flatten_chain[chosen_indices]

    #Calculating for each r sample a light curve and plotting it
    for param_sample in param_samples:
        sample_dic={}
        for param in fixed_args['var_param_list']:
            sample_dic.update({param : param_sample[fixed_args['var_param_list'].index(param)]})
        for ipar, param in enumerate(fixed_args['fix_param_list']):
            sample_dic.update({param:fixed_args['fix_param_val'][ipar]})
        
        sample_lc = create_jaxoplanet_model(init_state_dic['times'], sample_dic)

        ax_left.plot(init_state_dic['times'], sample_lc, color='blue', linewidth=1, alpha=0.1, zorder=1)

    ax_left.spines[['right', 'top', 'left', 'bottom']].set_visible(False)
    ax_left.set_xticks([])
    ax_left.set_yticks([])
    ax_left.set_xticklabels([])
    ax_left.set_yticklabels([])



    # --- Right-hand plot ---
    print('RIGHT-HAND PLOT')

    fit_colors = ['blue', 'green', 'salmon']
    fit_labels = ['2nd order LD', '3rd order LD', '4th order LD']

    plt.rcParams["font.family"] = "Arial"

    #Loop over LD models
    for PLD_order, fit_color in zip(PLD_orders, fit_colors):
        print('    PROCESSING PLD ORDER:', PLD_order)
    
        #Loop over model scatters
        for model_scatter in model_scatters:
            print('        PROCESSING MODEL SCATTER:', model_scatter)
            
            #Initialize array to store amplification factors for all seeds
            amp_factors = cached_data[PLD_order][model_scatter]
            
            #Make box-plot for this PLD-model scatter-seed combination
            ax_right.boxplot(amp_factors, positions=[model_scatter], patch_artist=True,
                    boxprops=dict(facecolor=f'light{fit_color}', color=fit_color), widths=[model_scatter * 0.15],
                    medianprops=dict(color='black', linewidth=2),
                    whiskerprops=dict(color=fit_color, linewidth=1.5),
                    capprops=dict(color=fit_color, linewidth=1.5),
                    flierprops=dict(marker='o', color=fit_color, markersize=5, alpha=0.5),
                    showfliers=False)
    # Styling
    ax_right.set_xscale('log')
    ax_right.set_yscale('log')
    ax_right.set_xlabel(r'Baseline Scatter, $\sigma_{\rm OOT}$ (ppm)', fontsize=12)
    ax_right.set_ylabel(r'Amplification Factor ($A$)', fontsize=12)
    ax_right.tick_params(axis='x', labelsize=12)
    ax_right.tick_params(axis='y', labelsize=12)
    ax_right.set_xticks([0.1, 1, 10, 100, 1000, 10000], labels = [0.1, 1, 10, 100, 1000, 10000])
    ax_right.set_xlim([0.08, 30000])
    ax_right.set_yticks([1, 10], labels = [1, 10])
    ax_right.grid(which='both', linestyle='--', alpha=0.5)
    ax_right.set_ylim([0.85, 55])

    # --- Gradient Transition Region with log-aware fading ---
    x_min, x_max = 2, 10000
    y_min, y_max = ax_right.get_ylim()

    # resolution in x
    n = 500
    x_vals = np.logspace(np.log10(x_min), np.log10(x_max), n)

    # Build horizontal gradient: fade from white at edges → gray at center
    grad = np.ones((n, 1, 4))  # (n,1,RGBA)
    x = np.linspace(-3, 3, n)  # -3σ to +3σ
    alpha_profile = norm.pdf(x, 0, 1.4)  # Gaussian curve
    alpha_profile /= alpha_profile.max()  # normalize to [0,1]
    grad[:,0,0:3] = 0.2  # lightgray base color
    grad[:,0,3] = alpha_profile * 0.9  # opacity max in middle

    # Plot as pcolormesh in log-space
    X, Y = np.meshgrid(x_vals, [y_min, y_max])
    # Broadcast alpha to shape (2, 500)
    alpha_2d = np.vstack([grad[:,0,3], grad[:,0,3]])
    ax_right.pcolormesh(X, Y, np.zeros_like(alpha_2d), color=(0.8,0.8,0.8,1),
                shading='auto', cmap='Greys', alpha=alpha_2d, zorder=0)

    ax_right.axhline(np.sqrt(3), color='k', linestyle='dashed')
    ax_right.text(0.13, np.sqrt(3) + 0.3, r'Theoretical limit @ $\sqrt{3}$', fontsize=12, color='black')
    ax_right.text(30, 48 - 0.5, 'Transition region', fontsize=12, color='black')
    ax_right.text(4400, 48 - 0.5, 'Noise limited', fontsize=12, color='black')
    ax_right.text(0.13, 48 - 0.5, 'Model (i.e. degeneracy) limited', fontsize=12, color='black')

    # Add arrows from "Transition Region" to the other two regions
    ax_right.annotate("",
                xy=(3200, 49), xycoords="data",   # Noise Limited
                xytext=(250, 49), textcoords="data",  # Transition Region
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

    ax_right.annotate("",
                xy=(5, 49), xycoords="data",   # Degeneracy Limited
                xytext=(21, 49), textcoords="data",  # Transition Region
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

    # --- Asymptote lines with fading into transition region ---
    for y_asym, bias_asym, x_fade_min, x_fade_max, asym_color in zip([3.4760264633464213, 10.614606725662671, 32.00651136170043], [10, 5, 1], [10, 1, 0.5], [130, 20, 8], fit_colors):

        # log-spaced x for the line
        x_line = np.logspace(np.log10(0.08), np.log10(x_fade_max), 500)
        y_line = np.full_like(x_line, y_asym)

        # Compute alpha: 1 (solid) on left, fade to 0 in transition region
        alphas = np.ones_like(x_line)
        fade_mask = (x_line >= x_fade_min) & (x_line <= x_fade_max)
        fade_x = x_line[fade_mask]
        if fade_x.size > 0:
            # Smooth fade: Gaussian or linear
            fade_profile = np.linspace(1, 0, fade_x.size)
            alphas[fade_mask] = fade_profile

        # Build line segments with alpha fading
        points = np.array([x_line, y_line]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        base_color = mcolors.to_rgba(asym_color)  # convert to RGBA
        lc = LineCollection(segments, colors=[(base_color[0], base_color[1], base_color[2], a) for a in alphas[:-1]], linewidths=2, zorder=1)
        ax_right.add_collection(lc)
        ax_right.text(0.13, y_asym + (y_asym/6), f'asymptote @ A = {y_asym:.0f}, Bias = {bias_asym:.0f}', fontsize=12, color=asym_color)

    print(f"\nPlotting complete in {time.time() - t_plot_start:.2f} seconds")

    plt.savefig(raw_save_dir+'Fig1_opt.pdf')
    # fig.savefig(paths.figures / "Fig1.pdf", bbox_inches="tight", dpi=300)
    plt.show()