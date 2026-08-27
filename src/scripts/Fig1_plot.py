#############################
########## Purpose ##########
#############################

# Figure 1 showcases the amplification factor change across model scatters and limb-darkening law used (aka the dimensionality).
# The goal of this file is to retrieve the results from the injection-retrievals computed previously and compile them into Figure 1.


######################################
########## Import libraries ##########
######################################
from jax import random, jit
import jax
import paths
print(f"JAX devices: {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")
import jax.numpy as jnp
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"
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
from multiprocessing import Pool
import gc
import corner
from matplotlib.patches import Patch
from scipy.optimize import curve_fit

# For 64-bit precision since JAX defaults to 32-bit
jax.config.update("jax_enable_x64", True)
from matplotlib.patches import Patch

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
exposure_time = 5                                                       #secondso
num_t = jnp.floor((((high_t - low_t) * 24 * 3600)/exposure_time))       #number of points
init_state_dic['times'] = jnp.linspace(low_t, high_t, int(num_t))       #days


# Calculate IT points once
num_IT_pts = jnp.sum(((init_state_dic['times'] > init_state_dic['t0'] - T_dur/2) & 
                        (init_state_dic['times'] < init_state_dic['t0'] + T_dur/2)))

#%% Location of MCMC results and where plot will be output
raw_save_dir = str(paths.data / "Fig1_Storage") + "/"

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
num_workers = 1
# Number of files in each chunk
CHUNK_SIZE = 30

# Filtering parameters
THRESHOLDS = [5, 4, 3]  # Number of IQRs for outlier detection (5 is conservative)
ROUNDS = 3

# Diagnostic plotting parameters
ENABLE_DIAGNOSTICS = False  # Set to False to skip diagnostic plots
verbose = False

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
    Optimized file loading with iterative 2D sigma clipping.
    
    FILTERING STRATEGY:
    - Maintains a 2D mask of shape (n_walkers, n_steps_post) throughout
    - At each round, sigma clipping is computed over all currently-surviving
      (walker, step) pairs for chi2 and each parameter independently
    - Individual (walker, step) pairs are removed without discarding the
      entire walker chain
    - Returns good_steps_mask (n_walkers, n_steps_post) for fine-grained use
    """
    raw_save_dir, PLD_order, model_scatter, seed, return_full = args
    print(f"  Processing PLD{PLD_order}, scatter{model_scatter:.3f}, seed{seed}...")
    try:
        path_base = f'{raw_save_dir}PLD_{PLD_order}/{model_scatter:.3f}ppm/Seed{seed}/'

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

            return (PLD_order, model_scatter, seed, r_chain_post_burnin, bestfit_r,
                    True, full_chain, full_logprob, full_chi2, good_steps_mask) 
        else:
            return (PLD_order, model_scatter, seed, r_chain_post_burnin, bestfit_r,
                    False, None, None, None, None)

    except Exception as e:
        print(f"Error loading PLD{PLD_order}, scatter{model_scatter:.3f}, seed{seed}: {e}")
        import traceback
        traceback.print_exc()
        return None


def plot_diagnostics(full_chain, full_logprob, full_chi2, good_steps_mask,
                     PLD_order, model_scatter, seed):
    """
    Create comprehensive diagnostic plots after each chunk.

    Plots:
    1. Trace plots (all parameters) - removed walkers in red, surviving in blue,
       with masked-out steps shown as gaps
    2. Chi2 evolution
    3. Corner plot (post burn-in) - BLACK: pre-filtering, RED: post-filtering

    Parameters:
    -----------
    full_chain      : ndarray (n_walkers, n_steps, n_params)
    full_logprob    : ndarray (n_walkers, n_steps)
    full_chi2       : ndarray (n_walkers, n_steps)
    good_steps_mask : ndarray (n_walkers, n_steps_post) — 2D boolean mask from load_result
    """
    n_walkers, n_steps, n_params = full_chain.shape
    nburn        = fixed_args['nburn']
    n_steps_post = n_steps - nburn

    # Create output directory
    diag_dir = f'{raw_save_dir}PLD_{PLD_order}/{model_scatter:.3f}ppm/Seed{seed}/diagnostics/'
    os.makedirs(diag_dir, exist_ok=True)
    print(f"    Generating diagnostics for PLD{PLD_order}, scatter{model_scatter:.3f}, seed{seed}")

    # =========================================================================
    # DERIVE PER-WALKER SUMMARY FROM 2D MASK (for colouring traces)
    # fully_good  : all steps survived  → blue
    # partially   : some steps survived → orange
    # fully_bad   : no steps survived   → red
    # =========================================================================
    fully_good_mask   = good_steps_mask.all(axis=1)                       # (n_walkers,)
    fully_bad_mask    = ~good_steps_mask.any(axis=1)                      # (n_walkers,)
    partial_mask      = ~fully_good_mask & ~fully_bad_mask                # (n_walkers,)

    n_fully_good  = np.sum(fully_good_mask)
    n_partial     = np.sum(partial_mask)
    n_fully_bad   = np.sum(fully_bad_mask)
    n_good_pairs  = np.sum(good_steps_mask)

    print(f"      Walkers fully survived : {n_fully_good}")
    print(f"      Walkers partially survived : {n_partial}")
    print(f"      Walkers fully removed  : {n_fully_bad}")
    print(f"      Total surviving (walker, step) pairs: {n_good_pairs} / {n_walkers * n_steps_post}")

    # Post-burnin arrays
    burnt_chain   = full_chain[:, nburn:, :]                              # (n_walkers, n_steps_post, n_params)
    burnt_logprob = full_logprob[:, nburn:]                               # (n_walkers, n_steps_post)

    # Surviving samples as flat arrays (used for corner plot and amp factor)
    # Shape: (n_good_pairs, n_params) and (n_good_pairs,)
    samples_post  = burnt_chain[good_steps_mask, :]                       # (n_good_pairs, n_params)
    samples_pre   = burnt_chain.reshape(-1, n_params)                     # (n_walkers * n_steps_post, n_params)

    param_names = ['r', 'i', 'a', 'period', 'sqrtecosw', 'sqrtesinw'] + \
                  [f'LD_u{i}' for i in range(1, PLD_order + 1)]

    # =========================================================================
    # PLOT 1: Trace plots
    # =========================================================================
    fig_trace = plt.figure(figsize=(16, 2 * n_params))
    gs = fig_trace.add_gridspec(n_params, 3, width_ratios=[1, 1, 1], hspace=0.05, wspace=0.05)

    for i, param_name in enumerate(param_names):

        ax_trace = fig_trace.add_subplot(gs[i, 0])
        ax_burn  = fig_trace.add_subplot(gs[i, 1])
        ax_hist  = fig_trace.add_subplot(gs[i, 2], sharey=ax_burn)

        # ----- Left: full trace (burn-in + post) -----
        for walker in range(n_walkers):
            color = 'red' if fully_bad_mask[walker] else ('orange' if partial_mask[walker] else 'blue')
            ax_trace.plot(np.arange(nburn), full_chain[walker, :nburn, i],
                          color='red', alpha=0.2, linewidth=0.2)
            ax_trace.plot(np.arange(nburn, n_steps), full_chain[walker, nburn:, i],
                          color=color, alpha=0.2, linewidth=0.2)

        ax_trace.axvline(nburn, color='black', linestyle='--', linewidth=0.8, alpha=0.5,
                         label='Burn-in cutoff' if i == 0 else '')
        ax_trace.set_ylabel(param_name, fontsize=10)
        ax_trace.grid(True, alpha=0.3)
        if i == 0:
            from matplotlib.lines import Line2D
            ax_trace.legend(handles=[
                Line2D([0], [0], color='blue',   label=f'Fully good ({n_fully_good})'),
                Line2D([0], [0], color='orange', label=f'Partial ({n_partial})'),
                Line2D([0], [0], color='red',    label=f'Fully removed ({n_fully_bad})'),
            ], fontsize=7, loc='upper right')
        if i < n_params - 1:
            ax_trace.set_xticklabels([])
        else:
            ax_trace.set_xlabel('Step', fontsize=10)

        # ----- Middle: post-burnin trace with masked steps shown as NaN gaps -----
        for walker in range(n_walkers):
            # Mask out steps that were removed for this walker
            trace = full_chain[walker, nburn:, i].copy().astype(float)
            trace[~good_steps_mask[walker]] = np.nan          # gaps where steps were removed

            color = 'red' if fully_bad_mask[walker] else ('orange' if partial_mask[walker] else 'blue')
            ax_burn.plot(trace, color=color, alpha=0.4, linewidth=0.5)

        ax_burn.set_ylabel(param_name, fontsize=10)
        ax_burn.grid(True, alpha=0.3)
        if i < n_params - 1:
            ax_burn.set_xticklabels([])
        else:
            ax_burn.set_xlabel('Post-burnin step', fontsize=10)

        # ----- Right: histogram pre (red) vs post (blue) -----
        ax_hist.hist(samples_pre[:, i], bins=30, orientation='horizontal',
                     density=True, histtype='stepfilled',
                     color='red', alpha=0.4, edgecolor='red', linewidth=0.5,
                     label='Pre-filter' if i == 0 else '')
        ax_hist.hist(samples_post[:, i], bins=30, orientation='horizontal',
                     density=True, histtype='stepfilled',
                     color='blue', alpha=0.5, edgecolor='blue', linewidth=0.5,
                     label='Post-filter' if i == 0 else '')
        ax_hist.set_xlabel('Density', fontsize=8)
        ax_hist.tick_params(axis='y', labelleft=False)
        ax_hist.grid(True, alpha=0.3, axis='x')
        if i == 0:
            ax_hist.legend(fontsize=7)

    fig_trace.suptitle(f'Trace Plots - PLD{PLD_order}, scatter={model_scatter:.3f}, seed={seed}',
                       fontsize=12)
    plt.savefig(os.path.join(diag_dir, 'trace.pdf'), dpi=150, bbox_inches='tight')
    plt.close(fig_trace)

    # =========================================================================
    # PLOT 2: Chi2 evolution
    # =========================================================================
    fig_chi2, ax_chi2 = plt.subplots(1, 1, figsize=(12, 4))

    for walker in range(n_walkers):
        color = 'red' if fully_bad_mask[walker] else ('orange' if partial_mask[walker] else 'blue')
        ax_chi2.loglog(np.arange(nburn), full_chi2[walker, :nburn],
                         color='red', alpha=0.2, linewidth=0.4)

        # Show post-burnin with gaps for masked steps
        chi2_trace = full_chi2[walker, nburn:].copy().astype(float)
        chi2_trace[~good_steps_mask[walker]] = np.nan
        ax_chi2.loglog(np.arange(nburn, n_steps), chi2_trace,
                         color=color, alpha=0.4, linewidth=0.5)

    ax_chi2.axvline(nburn, color='black', linestyle='--', linewidth=1.5, alpha=0.5, label='Burn-in')
    ax_chi2.set_xlabel('Step', fontsize=10)
    ax_chi2.set_ylabel('Chi-squared', fontsize=10)
    ax_chi2.grid(True, alpha=0.3)
    ax_chi2.legend(loc='upper right')
    ax_chi2.set_title(f'Chi2 Evolution - PLD{PLD_order}, scatter={model_scatter:.3f}, seed={seed}',
                      fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(diag_dir, 'chi2.pdf'), dpi=150, bbox_inches='tight')
    plt.close(fig_chi2)

    # =========================================================================
    # PLOT 3: Corner plot
    # =========================================================================

    # Pre-filtering amplification factor
    r_chain_pre   = samples_pre[:, 0]
    max_idx_pre   = np.unravel_index(np.argmax(full_logprob), full_logprob.shape)
    bestfit_r_pre = full_chain[max_idx_pre[0], max_idx_pre[1], 0]
    amp_factor_pre = (2 * np.std(r_chain_pre) * bestfit_r_pre) / ((model_scatter * 1e-6) / np.sqrt(num_IT_pts))

    # Post-filtering amplification factor
    r_chain_post   = samples_post[:, 0]
    masked_logprob = np.where(good_steps_mask, burnt_logprob, -np.inf)
    best_w, best_s = np.unravel_index(np.argmax(masked_logprob), masked_logprob.shape)
    bestfit_r_post = burnt_chain[best_w, best_s, 0]
    amp_factor_post = (2 * np.std(r_chain_post) * bestfit_r_post) / ((model_scatter * 1e-6) / np.sqrt(num_IT_pts))

    fig_corner = corner.corner(
        samples_pre, labels=param_names, color='black',
        quantiles=[0.16, 0.5, 0.84], show_titles=False,
        plot_datapoints=False, plot_density=True,
        hist_kwargs={'alpha': 0.6, 'linewidth': 2},
        contour_kwargs={'linewidths': 1.5, 'alpha': 0.6}
    )
    corner.corner(
        samples_post, fig=fig_corner, labels=param_names, color='red',
        quantiles=[0.16, 0.5, 0.84], show_titles=False,
        plot_datapoints=False, plot_density=True,
        hist_kwargs={'alpha': 0.8, 'linewidth': 2},
        contour_kwargs={'linewidths': 2, 'alpha': 0.8},
        reverse=True
    )

    # Diagonal: mirrored histogram
    axes = np.array(fig_corner.axes).reshape((n_params, n_params))
    for i in range(n_params):
        ax_bottom = axes[i, i]
        ax_bottom.cla()
        ax_bottom.set_visible(False)  # Hide the original axis — we'll draw two fresh ones on top

        # Get the position of the diagonal cell in figure coordinates
        pos = ax_bottom.get_position()  # Bbox in figure fraction: x0, y0, width, height

        # =========================
        # PRE axis — bottom half, bars grow UPWARD
        # =========================
        ax_pre = fig_corner.add_axes(
            [pos.x0, pos.y0, pos.width, pos.height / 2],
            label=f"pre_{i}"
        )
        bins_pre = np.linspace(samples_pre[:, i].min(), samples_pre[:, i].max(), 30)
        counts_pre, edges_pre = np.histogram(samples_pre[:, i], bins=bins_pre, density=True)
        ax_pre.bar(
            edges_pre[:-1],
            counts_pre,
            width=np.diff(edges_pre),
            align="edge",
            color="black",
            alpha=0.6,
            linewidth=0,
        )
        ax_pre.set_xlim(samples_pre[:, i].min(), samples_pre[:, i].max())
        ax_pre.set_ylim(0, counts_pre.max() * 1.05)  # Normal upward orientation
        ax_pre.yaxis.set_visible(False)
        ax_pre.spines["top"].set_visible(False)
        ax_pre.spines["right"].set_visible(False)
        ax_pre.spines["left"].set_visible(False)
        ax_pre.tick_params(axis="x", labelsize=8)

        # =========================
        # POST axis — top half, bars grow DOWNWARD (y-axis inverted)
        # =========================
        ax_post = fig_corner.add_axes(
            [pos.x0, pos.y0 + pos.height / 2, pos.width, pos.height / 2],
            label=f"post_{i}"
        )
        bins_post = np.linspace(samples_post[:, i].min(), samples_post[:, i].max(), 30)
        counts_post, edges_post = np.histogram(samples_post[:, i], bins=bins_post, density=True)
        ax_post.bar(
            edges_post[:-1],
            counts_post,
            width=np.diff(edges_post),
            align="edge",
            color="red",
            alpha=0.8,
            linewidth=0,
        )
        ax_post.set_xlim(samples_post[:, i].min(), samples_post[:, i].max())
        ax_post.set_ylim(0, counts_post.max() * 1.05)
        ax_post.invert_yaxis()  # Flip so bars hang DOWN from the top spine
        ax_post.yaxis.set_visible(False)
        ax_post.xaxis.set_visible(False)  # Hide x-axis on top panel (shown on bottom)
        ax_post.spines["bottom"].set_visible(False)
        ax_post.spines["right"].set_visible(False)
        ax_post.spines["left"].set_visible(False)

    title_text = (
        f'Corner Plot - PLD{PLD_order}, scatter={model_scatter:.3f} ppm, seed={seed}\n'
        f'BLACK: Pre-filtering ({n_walkers} walkers, {len(samples_pre)} pairs) - A = {amp_factor_pre:.2f}\n'
        f'RED: Post-filtering ({n_good_pairs} pairs, {n_fully_good} full + {n_partial} partial walkers) - A = {amp_factor_post:.2f}\n'
        f'Improvement: {((amp_factor_pre - amp_factor_post) / amp_factor_pre * 100):.1f}% reduction in A'
    )
    fig_corner.suptitle(title_text, fontsize=11, y=1.03)

    fig_corner.legend(handles=[
        Patch(facecolor='black', alpha=0.6, label=f'Pre-filter:  A={amp_factor_pre:.2f}'),
        Patch(facecolor='red',   alpha=0.8, label=f'Post-filter: A={amp_factor_post:.2f}'),
    ], loc='upper right', fontsize=10, bbox_to_anchor=(0.95, 0.95))

    plt.savefig(os.path.join(diag_dir, 'corner.pdf'), dpi=150, bbox_inches='tight')
    plt.close(fig_corner)

# OPTIMIZATION 2: JAX-optimized computation
@jit
def compute_bias_jax(true_r, bestfit_r):
    """JIT-compiled bias calculation"""
    return jnp.abs((bestfit_r - true_r)/true_r)


def batch_compute_biases(results_batch):
    """
    Process multiple results in batch using JAX vectorization
    
    This processes all chains for a given (PLD_order, model_scatter) at once
    """
    biases = []
    
    for _, _, _, _, bestfit_r, _, _, _, _, _ in results_batch:
    
        bias = compute_bias_jax(
            init_state_dic['r'], bestfit_r 
        )
        biases.append(bias)

    return biases

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
    
    for _, model_scatter, _, r_chain_post_burnin, bestfit_r, _, _, _, _, _ in results_batch:
    
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
    model_scatters = [0.01, 0.03162277660168379, 0.09999999999999995, 0.31622776601683783,\
                      0.9999999999999994, 3.162277660168377, 9.999999999999995,\
                      99.9999999999999, 999.999999999998, 3162.2776601683763,
                      9999.99999999998, 31622.77660168373] + [16.68100537200059, 27.825594022071243,\
                      46.41588833612777, 77.4263682681127, 129.1549665014884, 215.44346900318823,\
                      359.38136638046257, 599.4842503189409]
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

        cached_data = {'amp_factors':{},'biases':{}}
        for PLD_order in PLD_orders:
            cached_data['amp_factors'][PLD_order] = {}
            cached_data['biases'][PLD_order] = {}
            print(f"\nProcessing PLD order {PLD_order}")
            for model_scatter in model_scatters:
                print(f"\nProcessing Model scatter {model_scatter:.0f}")
                batch_data = []
                # Filter results for this specific PLD_order and model_scatter
                for result in results:
                    result_pld, result_scatter, seed, r_chain_post_burnin, bestfit_r, _, _, _, _, _ = result
                    if result_pld == PLD_order and result_scatter == model_scatter:
                        batch_data.append(result)

                if len(batch_data) > 0:
                    # Use batch processing with JAX
                    ## Amplification factor
                    amp_factors = batch_compute_amplification_factors(
                        batch_data, num_IT_pts
                    )
                    ## Bias (in sigma)
                    biases = batch_compute_biases(
                        batch_data,
                    )
                    cached_data['amp_factors'][PLD_order][model_scatter] = np.array(amp_factors)
                    cached_data['biases'][PLD_order][model_scatter] = np.array(biases)
                else:
                    cached_data['amp_factors'][PLD_order][model_scatter] = np.array([])
                    cached_data['biases'][PLD_order][model_scatter] = np.array([])
        
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
    model_scatter = 359.0
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
    raw_chain = jnp.load(raw_save_dir+f'PLD_{PLD_order}/{model_scatter:.3f}ppm/Seed{seed}/chains.npy')

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

    # --- Left panel annotations ---
    # Get axis limits for positioning
    xlim = ax_left.get_xlim()
    ylim = ax_left.get_ylim()

    x_left  = xlim[0]
    x_right = xlim[1]
    x_range = x_right - x_left
    y_range = ylim[1] - ylim[0]

    # --- σ_OOT: vertical double-headed arrow on the right, in the out-of-transit region ---
    # Place it just to the right of the transit, at the out-of-transit flux level
    x_arrow_oot = x_left + x_range * 0.37   # right of transit egress
    y_oot_center = float(true_lc.max())-float(noisy_std.mean())    # out-of-transit flux ~ 1.0
    y_oot_top    = y_oot_center + float(noisy_std.mean())
    y_oot_bot    = y_oot_center - 2*float(noisy_std.mean())

    ax_left.annotate("", xy=(x_arrow_oot, y_oot_top), xytext=(x_arrow_oot, y_oot_bot),
                    arrowprops=dict(arrowstyle="<->", color="black", lw=1.2))
    ax_left.text(x_arrow_oot + x_range * 0.02, y_oot_center - 0.5*float(noisy_std.mean()),
                r'$\sigma_{\rm OOT}$', fontsize=11, va='center', ha='left')

    # --- σ_ref label on the left side ---
    ax_left.text(x_left + x_range * 0.3, y_oot_center - y_range * 0.15,
                r'$\sigma_{\rm ref} = \dfrac{\sigma_{\rm OOT}}{\sqrt{N}}$',
                fontsize=10, va='center', ha='right')

    # --- Transit Depth D: vertical double-headed arrow on the left side ---
    y_bottom = float(true_lc.min())   # bottom of transit
    y_top    = float(true_lc.max())   # out-of-transit level

    x_arrow_D = x_left + x_range * 0.5   # slightly inside left edge

    ax_left.annotate("", xy=(x_arrow_D, y_top), xytext=(x_arrow_D, y_bottom),
                    arrowprops=dict(arrowstyle="<->", color="black", lw=1.2))
    ax_left.text(x_arrow_D - x_range * 0.03, (y_top + y_bottom) / 2,
                r'Transit Depth, $D$', fontsize=10, va='center', ha='right',
                rotation=90)

    # --- A definition: placed in the middle-right of the left panel ---
    x_def = x_right * 0.35
    y_def = y_bottom - y_range * 0.064
    ax_left.text(x_def, y_def,
                r'$A := \dfrac{\sigma_D}{\sigma_{\rm ref}}$' + '\n' +
                r'$= \sqrt{3/2}$, in theory',
                fontsize=10, va='bottom', ha='left')

    # Arrow from "= sqrt(3/2), in theory" pointing toward the dashed line on right panel
    # (This is a cross-axes annotation; we use figure coordinates)
    # Simpler: just add a small rightward arrow from the text
    y_def = y_bottom - y_range * 0.066
    ax_left.annotate("",
                    xy=(x_right * 0.94, y_def + y_range * 0.02),
                    xytext=(x_def + x_range * 0.48, y_def + y_range * 0.02),
                    arrowprops=dict(arrowstyle="<-", color="black", lw=1.0))

    # --- N in-transit points: horizontal arrow along the bottom ---
    # Find the transit duration extent in time
    in_transit_mask = (init_state_dic['times'] > init_state_dic['t0'] - T_dur/2) & \
                    (init_state_dic['times'] < init_state_dic['t0'] + T_dur/2)
    t_ingress = float(init_state_dic['times'][in_transit_mask][0])
    t_egress  = float(init_state_dic['times'][in_transit_mask][-1])

    y_N = y_bottom - y_range * 0.09   # just below the transit bottom

    ax_left.annotate("", xy=(t_egress, y_N), xytext=(t_ingress, y_N),
                    arrowprops=dict(arrowstyle="|-|", color="black", lw=1.2))
    ax_left.text(0.0, y_N - y_range * 0.04,
                r'$N$ in-transit points', fontsize=10, va='top', ha='center')


    # --- Right-hand plot ---
    print('RIGHT-HAND PLOT')

    fit_colors = ['blue', 'green', 'salmon']
    fit_labels = ['2nd order LD', '3rd order LD', '4th order LD']

    def piecewise_log_linear(log_x, log_x_break, slope, log_A_asym):
        """
        Piecewise linear model in log-log space with one flat (slope=0) piece.

        Left piece  (log_x <= log_x_break): flat at log_A_asym
        Right piece (log_x >  log_x_break): slope * (log_x - log_x_break) + log_A_asym

        Continuity is enforced by construction at the breakpoint.
        """
        return np.where(
            log_x <= log_x_break,
            log_A_asym,
            slope * (log_x - log_x_break) + log_A_asym
        )

    # --- Fit asymptotes automatically for each PLD order ---
    asymp_amp_factor_left = np.zeros(len(PLD_orders), dtype=float)
    asymp_amp_factor_right = np.zeros(len(PLD_orders), dtype=float)
    asymp_bias = np.zeros(len(PLD_orders), dtype=float)
    # Define the scatter thresholds to split the data sets for the two fits
    scatter_splits = [2, 2, 2]

    for ipld, (PLD_order, scatter_split) in enumerate(zip(PLD_orders, scatter_splits)):

        # Collect per-scatter medians
        scatter_vals = []
        median_amps  = []

        for model_scatter in model_scatters:
            amp_arr = cached_data['amp_factors'][PLD_order][model_scatter]
            if len(amp_arr) > 0:
                scatter_vals.append(model_scatter)
                median_amps.append(np.median(amp_arr))

        scatter_vals = np.array(scatter_vals)
        median_amps  = np.array(median_amps)

        #Left side of the break : define a constant from the median
        left_mask = (scatter_vals < scatter_split)
        asymp_amp_factor_left[ipld] = np.median(median_amps[left_mask])

        #Right side of the break : fit a piecewise log linear function
        right_mask = (scatter_vals > scatter_split)
        log_scatter = np.log10(scatter_vals)[right_mask]
        log_amps    = np.log10(median_amps)[right_mask]

        # Initial guesses:
        #   - break near the middle of the scatter range
        #   - negative slope for the right (noise-limited) piece
        #   - asymptote at the maximum observed amp factor (leftmost plateau)
        p0     = [np.median(log_scatter), -1.0, np.max(log_amps)]
        # Constrain slope <= 0 so the right piece is non-increasing
        bounds = (
            [log_scatter.min(), -np.inf, -np.inf],
            [log_scatter.max(),  0.0,     np.inf]
        )

        try:
            popt, _ = curve_fit(
                piecewise_log_linear, log_scatter, log_amps,
                p0=p0, bounds=bounds, maxfev=10000
            )
            _, _, log_A_asym = popt
            asymp_amp_factor_right[ipld] = 10**log_A_asym

            print(f"PLD{PLD_order}: A_asym_left = {asymp_amp_factor_left[ipld]:.3f}, A_asym_right = {asymp_amp_factor_right[ipld]:.3f}")

        except RuntimeError as e:
            print(f"  WARNING — fit failed for PLD{PLD_order}: {e}. Using low-scatter fallback.")
            n_low = max(1, len(scatter_vals) // 3)
            asymp_amp_factor_right[ipld] = np.median(median_amps[:n_low])

    #Loop over LD models
    for ipld, (PLD_order, fit_color) in enumerate(zip(PLD_orders, fit_colors)):
        print('    PROCESSING PLD ORDER:', PLD_order)
    
        #Loop over model scatters
        for model_scatter in model_scatters:
            print('        PROCESSING MODEL SCATTER:', model_scatter)
            
            #Initialize array to store amplification factors for all seeds
            amp_factors = cached_data['amp_factors'][PLD_order][model_scatter]

            #Initialize array to store amplification factors for all seeds
            biases = cached_data['biases'][PLD_order][model_scatter]
            
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
    ax_right.set_xticks([0.01, 0.1, 1, 10, 100, 1000, 10000], labels = [0.01, 0.1, 1, 10, 100, 1000, 10000])
    ax_right.set_xlim([0.008, 100000])
    ax_right.set_yticks([1, 10], labels = [1, 10])
    ax_right.grid(which='both', linestyle='--', alpha=0.5)
    ax_right.set_ylim([0.85, 60])

    # --- Gradient Transition Region with log-aware fading ---
    x_min, x_max = 50, 20000
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

    ax_right.axhline(np.sqrt(3/2), color='k', linestyle='dashed')
    ax_right.text(0.009, np.sqrt(3/2) - 0.3, r'Theoretical limit @ $\sqrt{3/2}$', fontsize=12, color='black')
    ax_right.text(200, 50, 'Transition region', fontsize=10.5, color='black')
    ax_right.text(12000, 50, 'Noise limited', fontsize=10.5, color='black')
    ax_right.text(5.5, 50, 'Model limited', fontsize=10.5, color='black')
    ax_right.text(0.03, 50, 'Sampler limited', fontsize=10.5, color='black')

    # Add arrows from "Transition Region" to the other two regions
    ax_right.annotate("",
                xy=(10000, 52), xycoords="data",   # Noise Limited
                xytext=(2100, 52), textcoords="data",  # Transition Region
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

    ax_right.annotate("",
                xy=(40, 52), xycoords="data",   # Model Limited
                xytext=(180, 52), textcoords="data",  # Transition Region
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

    ax_right.annotate("",
                xy=(0.3, 52), xycoords="data",   # Sampler Limited
                xytext=(4, 52), textcoords="data",  # Model Limited
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
    
    transition_buffers = [0.3, 0.6, 0.6]
    # --- Asymptote lines with fading into transition region ---
    for idx, (y_asym_left, y_asym_right, x_fade_min, x_fade_max, asym_color, text_loc, transition_buffer) in enumerate(zip(asymp_amp_factor_left, asymp_amp_factor_right, [10, 1, 0.5], [10000, 1000, 100], fit_colors, [5., 4., 2.], transition_buffers)):

        log_split = np.log10(scatter_splits[idx])

        # log-spaced x for the line
        x_line = np.logspace(np.log10(0.008), np.log10(x_fade_max), 500)

        # Compute alpha: 1 (solid) on left, fade to 0 in transition region
        alphas = np.ones_like(x_line)
        fade_mask = (x_line >= x_fade_min) & (x_line <= x_fade_max)
        fade_x = x_line[fade_mask]
        if fade_x.size > 0:
            fade_profile = np.linspace(1, 0, fade_x.size)
            alphas[fade_mask] = fade_profile

        # --- Upper edge: interpolates from y_asym_right → y_asym_left (in log space) ---
        log_y_bottom = np.log10(y_asym_right)
        log_y_top_max = np.log10(y_asym_left)

        # Width factor: 0 to the right of split, ramps to 1 to the left
        width_factor = np.clip(
            (log_split - np.log10(x_line)) / transition_buffer, 0.0, 1.0
        )

        # Upper edge grows upward only; bottom edge stays fixed at y_asym_right
        log_y_upper = log_y_bottom + width_factor * (log_y_top_max - log_y_bottom)
        y_upper = 10 ** log_y_upper
        y_lower = np.full_like(x_line, y_asym_right)

        # Build per-segment fill_between patches
        # - Right of split: horizontal alpha fading (as before)
        # - Left of split: vertical alpha fading (bottom opaque, top transparent)
        base_color = mcolors.to_rgba(asym_color)
        n_segs = len(x_line) - 1

        for i in range(n_segs):
            x_seg = x_line[i:i+2]
            y_seg_lower = y_lower[i:i+2]
            y_seg_upper = y_upper[i:i+2]

            if np.log10(x_line[i]) < log_split:
                # --- Vertical fading: stack thin horizontal slices ---
                n_slices = 30
                log_y_bot = np.log10(y_seg_lower[0])
                log_y_top = np.log10(y_seg_upper[0])
                if log_y_top <= log_y_bot:
                    continue
                log_y_edges = np.linspace(log_y_bot, log_y_top, n_slices + 1)
                y_edges = 10 ** log_y_edges

                for j in range(n_slices):
                    # 0 at bottom (y_asym_right), 1 at top (y_asym_left)
                    t = j / n_slices
                    slice_alpha = (1.0 - t) * base_color[3]
                    ax_right.fill_between(
                        x_seg,
                        [y_edges[j], y_edges[j]],
                        [y_edges[j+1], y_edges[j+1]],
                        color=(base_color[0], base_color[1], base_color[2], slice_alpha),
                        linewidth=0,
                        zorder=1
                    )
            else:
                # --- Horizontal fading (right of split): as before ---
                a = alphas[i]
                if a < 0.01:
                    continue
                ax_right.fill_between(
                    x_seg,
                    y_seg_lower,
                    y_seg_upper,
                    color=(base_color[0], base_color[1], base_color[2], a),
                    linewidth=0,
                    zorder=1
                )

        # Draw a thin solid baseline at y_asym_right with alpha fading
        points = np.array([x_line, y_lower]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(
            segments,
            colors=[(base_color[0], base_color[1], base_color[2], a) for a in alphas[:-1]],
            linewidths=2.0,
            zorder=2
        )
        ax_right.add_collection(lc)

        if idx == 0:
            ax_right.text(text_loc, y_asym_right + (y_asym_right/6), f'asymptote @ A = {y_asym_right:.0f}', fontsize=12, color=asym_color)
        elif idx == 1:
            ax_right.text(text_loc, y_asym_right - (y_asym_right/5), f'asymptote @ A = {y_asym_right:.0f}', fontsize=12, color=asym_color)
        else:
            ax_right.text(text_loc, y_asym_right + (y_asym_right/5), f'asymptote @ A = {y_asym_right:.0f}', fontsize=12, color=asym_color)

    # --- Convergence to theoretical limit text ---
    ax_right.text(9000, 15, 
                  'Convergence to\ntheoretical limit', 
                  fontsize=10.5, 
                  color='black',
                  bbox=dict(facecolor='white', alpha=0.7, edgecolor='silver', boxstyle='round,pad=0.3', lw=1.5))

    ax_right.annotate("",
                xy=(31622, 46), xycoords="data",   # Noise Limited
                xytext=(31622, 22), textcoords="data",  # Convergence Text
                arrowprops=dict(arrowstyle="<-", color="black", lw=1.5))
    
    ax_right.annotate("",
                xy=(31622, 12), xycoords="data",   # Convergence Text
                xytext=(31622, 2.4), textcoords="data",  # 30,000ppm points
                arrowprops=dict(arrowstyle="<-", color="black", lw=1.5))
    
    # --- Right panel legend: Injection/Retrieval LD law ---
    legend_elements = [
        Patch(facecolor='lightblue',   edgecolor='blue',   label=r'$2^{\rm nd}$ order poly.'),
        Patch(facecolor='lightgreen',  edgecolor='green',  label=r'$3^{\rm rd}$ order poly.'),
        Patch(facecolor='lightsalmon', edgecolor='salmon',  label=r'$4^{\rm th}$ order poly.'),
    ]

    legend = ax_right.legend(
        handles=legend_elements,
        title=r'Injection/Retrieval LD law :',
        title_fontsize=11,
        fontsize=11,
        loc='lower center',
        framealpha=0.85,
        edgecolor='gray',
        fancybox=False,
        bbox_to_anchor=[0.55, 0.],
        ncol=4,                        
        handlelength=1.5,
        handleheight=1.5,
        borderpad=0.8,
    )

    print(f"\nPlotting complete in {time.time() - t_plot_start:.2f} seconds")

    # plt.savefig(raw_save_dir+'Fig1_opt.pdf', bbox_inches="tight")
    plt.savefig(paths.figures / "Fig1.pdf", bbox_inches="tight")