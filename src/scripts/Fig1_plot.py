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

PLD_order = 3
model_scatter = 599.4842503189409
seed = 70

# Optimization
# Set number of cpus to use
num_workers = int(0.5 * cpu_count())
# Number of files in each chunk
CHUNK_SIZE = 60  

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
    Optimized file loading with memory mapping and selective loading
    
    Key optimizations:
    1. Use mmap_mode='r' to memory-map files instead of loading into RAM
    2. Only load the data we actually need (post burn-in, parameter 0)
    3. Extract bestfit info without loading full chain
    4. Return minimal data needed for computation
    """
    raw_save_dir, PLD_order, model_scatter, seed = args
    try:
        path_base = f'{raw_save_dir}PLD_{PLD_order}/{np.floor(model_scatter)}ppm/Seed{seed}/'
        
        # CRITICAL: Use memory mapping instead of loading entire file
        # This keeps data on disk and only loads what's needed
        raw_chain = np.load(path_base + 'chains.npy', mmap_mode='r')
        logprob = np.load(path_base + 'logprob.npy', mmap_mode='r')
        
        # Extract only what we need BEFORE returning
        # This significantly reduces memory usage
        max_walker, max_step = np.unravel_index(np.argmax(logprob), logprob.shape)
        
        # Only load post burn-in data for parameter 0
        r_chain_post_burnin = np.array(raw_chain[:, fixed_args['nburn']:, 0])  # Force load only this slice
        bestfit_r = float(raw_chain[max_walker, max_step, 0])
        
        # Return minimal data (not entire chains)
        return (PLD_order, model_scatter, seed, r_chain_post_burnin, bestfit_r)
    except Exception as e:
        print(f"Error loading PLD{PLD_order}, scatter{model_scatter}, seed{seed}: {e}")
        return None
    
# OPTIMIZATION 2: JAX-optimized computation
@jit
def compute_amplification_factor_jax(r_chain_flat, bestfit_r, model_scatter, num_IT_pts):
    """JIT-compiled amplification factor calculation"""
    std_r = jnp.std(r_chain_flat)
    bestfit_r_error = 2 * std_r * bestfit_r
    scatter_in_bin = (model_scatter * 1e-6) / jnp.sqrt(num_IT_pts)
    return bestfit_r_error / scatter_in_bin


# OPTIMIZATION 3: Vectorized batch processing
compute_amp_factors_batch = vmap(
    compute_amplification_factor_jax, 
    in_axes=(0, 0, None, None)
)


def batch_compute_amplification_factors(results_batch, num_IT_pts):
    """
    Process multiple results in batch using JAX vectorization
    
    This processes all chains for a given (PLD_order, model_scatter) at once
    """
    r_chains_flat = []
    bestfit_rs = []
    model_scatter = None
    
    for _, _, _, r_chain_post_burnin, bestfit_r in results_batch:
        r_chains_flat.append(r_chain_post_burnin.flatten())
        bestfit_rs.append(bestfit_r)
    
    # Get model_scatter from first result (all same in batch)
    model_scatter = results_batch[0][1]
    
    # Convert to JAX arrays and compute
    r_chains_jax = jnp.array(r_chains_flat)
    bestfit_rs_jax = jnp.array(bestfit_rs)
    
    amp_factors = compute_amp_factors_batch(
        r_chains_jax, bestfit_rs_jax, model_scatter, num_IT_pts
    )
    
    return [float(x) for x in amp_factors]

#############################################
################ Running code ###############
#############################################

if __name__ == '__main__':
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

    # Calculate IT points once
    num_IT_pts = jnp.sum(((init_state_dic['times'] > init_state_dic['t0'] - T_dur/2) & 
                            (init_state_dic['times'] < init_state_dic['t0'] + T_dur/2)))

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
            
            print(f"\nProcessing chunk {chunk_idx+1}/{num_chunks} ({len(chunk_tasks)} files)...")
            chunk_start_time = time.time()
            
            # Load chunk with progress bar            
            with Pool(processes=num_workers) as pool:
                chunk_results = []
                for result in tqdm(pool.imap_unordered(load_result, chunk_tasks),
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