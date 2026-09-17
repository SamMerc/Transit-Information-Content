#############################
########## Purpose ##########
#############################

# Companion to Fig5_plot.py. Instead of reporting the transit depth bias separately
# for each (limb-darkening law, prior strength) combination, this script computes a
# Bayesian Model Average (BMA) transit depth for each (prior strength, noise seed)
# pair -- averaging across all the limb-darkening laws fit to that realization of
# the data -- and compares the BMA depth to the true injected depth. This collapses
# the LDL axis, leaving 5 prior strengths x 10 noise seeds = 50 points per stellar
# LDC set (C label).
#
# The BMA follows Hoeting et al. (1999), "Bayesian Model Averaging: A Tutorial",
# eq (1)-(3):
#   pr(Delta|D)  = sum_k pr(Mk|D) * pr(Delta|D,Mk)                              (1)
#   pr(Mk|D)     = pr(D|Mk) pr(Mk) / sum_l pr(D|Ml) pr(Ml)                      (2)
#   pr(D|Mk)     = integral pr(D|theta_k,Mk) pr(theta_k|Mk) d(theta_k)          (3)
# where Delta is the transit depth (r^2). The MCMC pipeline does not compute the
# marginal likelihood pr(D|Mk) in (3) (no nested sampling / thermodynamic
# integration is run), so pr(Mk|D) is approximated with BIC-based Bayes factors
# (Schwarz 1978) -- see compute_bma_bias(). Eq (1) itself, however, is built
# literally: for each (prior strength, seed) the mixture is realised by Monte
# Carlo, drawing a model according to its BIC weight and then a posterior sample
# of that model's depth from its own (sigma-clipped) MCMC chain -- rather than
# collapsing each model's posterior pr(Delta|D,Mk) to a single point + Gaussian
# error beforehand.
#
# Two versions of the figure are produced:
#   - Fig5_BMA_all.pdf:     BMA computed across all 7 fitted LDLs (PLD_2-PLD_6, PLD_9, 4NLLD)
#   - Fig5_BMA_reduced.pdf: BMA computed across only the quadratic (PLD_2), 3rd-order
#                           polynomial (PLD_3), and 4th-order non-linear (4NLLD) laws.


######################################
########## Import libraries ##########
######################################
import paths
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"
import astropy.units as u
from astropy.constants import G
import numpy as np
import jax
import jax.numpy as jnp
import os
import time

# For 64-bit precision since JAX defaults to 32-bit (must match Fig5_run.py / Fig5_plot.py)
jax.config.update("jax_enable_x64", True)
import pickle
from tqdm import tqdm
from multiprocessing import Pool
import gc
import matplotlib.gridspec as gridspec


#############################################
########## Define hyper-parameters ##########
#############################################

#%% System parameters (must match Fig5_run.py / Fig5_plot.py)
G_solar_units = G.to(u.Rsun**3 / (u.Msun * u.day**2)).value
R_star = (1.0 * u.R_sun).value

init_state_dic = {}
init_state_dic['period'] = 1.                                  # days
a_meters = ((G.value * (1.0 * u.M_sun).to(u.kg).value * (init_state_dic['period'] * 24 * 3600)**2)
            / (4 * jnp.pi**2))**(1/3)
init_state_dic['a']     = a_meters / (1.0 * u.R_sun).to(u.m).value  # stellar radii
init_state_dic['r']     = 0.1                                  # stellar radii
init_state_dic['i']     = jnp.deg2rad(90)                      # radians
init_state_dic['omega'] = 0.0                                  # radians
init_state_dic['e']     = 0.                                   # unitless
init_state_dic['t0']    = 0.0                                  # days

b   = ((init_state_dic['a'] * jnp.cos(init_state_dic['i'])) / R_star
       * (1 - init_state_dic['e']**2) / (1 + init_state_dic['e'] * jnp.sin(init_state_dic['omega'])))
arg = ((1 / init_state_dic['a'])
       * jnp.sqrt((1 + init_state_dic['r'])**2 - b**2)
       / jnp.sin(init_state_dic['i']))
arg = np.clip(arg, -1.0, 1.0)
T_dur = ((init_state_dic['period'] / jnp.pi)
         * jnp.sqrt(1 - init_state_dic['e']**2) / (1 + init_state_dic['e'] * jnp.sin(init_state_dic['omega']))
         * jnp.arcsin(arg))

low_t  = -1.5 * T_dur
high_t =  1.5 * T_dur
exposure_time = 5                                              # seconds
num_t  = jnp.floor((((high_t - low_t) * 24 * 3600) / exposure_time))
init_state_dic['times'] = jnp.linspace(low_t, high_t, int(num_t))

N_TOTAL_PTS = int(init_state_dic['times'].shape[0])   # total LC points used in the BIC penalty term
TRUE_DEPTH  = init_state_dic['r']**2

#%% Grid parameters
RAW_BASE_DIR = '/Volumes/Ajax/Work/PhD/Research/Transit-Information-Content/Fig5_Storage/'

LDLs              = ['PLD_2', 'PLD_3', 'PLD_4', 'PLD_5', 'PLD_6', 'PLD_9', '4NLLD']
REDUCED_LDLs      = ['PLD_2', 'PLD_3', '4NLLD']   # quadratic, 3rd-order polynomial, 4th-order non-linear law
prior_strengths   = ['uniform', 'gauss_20', 'gauss_10', 'gauss_5', 'gauss_1']
prior_strengths_labels = ['Uniform', r'$20\%$ Gaussian', r'$10\%$ Gaussian',
                          r'$5\%$ Gaussian', r'$1\%$ Gaussian']
seeds             = [40, 50, 60, 70, 80, 90, 100, 110, 120, 130]

BASE_NDIM = 6   # r, i, a, period, sqrtecosw, sqrtesinw (t0 is fixed)

C_LABELS      = ['C0', 'C2', 'C3', 'C4', 'C7']
C_LABEL_NAMES = {'C0': 'Cluster 0 - M/K type, metal poor, near-infrared',
                 'C2': 'Cluster 2 - M/K type, metal rich, optical',
                 'C3': 'Cluster 3 - M/K type, solar metallicity, mid-infrared',
                 'C4': 'Cluster 4 - K/G type, solar metallicity, mid-infrared',
                 'C7': 'Cluster 7 - G type, solar metallicity, near-infrared'}

# Sequential (ordinal-safe) blue ramp for the 5 prior strengths, light->dark
# (uniform -> most informative). Steps 250/350/450/550/650 of the palette's
# blue sequential ramp; validated with scripts/validate_palette.js --ordinal.
PRIOR_COLORS = ['#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#104281']

#%% Filtering / processing parameters (same as Fig5_plot.py)
NBURN      = 70000
THRESHOLDS = [5, 4, 3]
ROUNDS     = 3
num_workers = 1
CHUNK_SIZE  = 30
verbose     = False

#%% BMA-specific parameters
N_SUB       = 2000    # posterior draws of r kept per (LDL, prior_strength, seed) in the cache
N_MIX_DRAWS = 20000    # Monte Carlo draws used to realise the BMA mixture in compute_bma_bias


##############################
##### Relevant functions #####
##############################

def get_n_LD_params(LDL):
    """Number of free limb-darkening coefficients for a given LDL."""
    if 'PLD' in LDL:
        return int(LDL.split('_')[1])
    elif LDL == '4NLLD':
        return 4
    else:
        raise KeyError(f'Unknown limb-darkening law: {LDL}')


def load_result_bma(args):
    """
    Load one (LDL, prior_strength, seed) run and apply the same iterative 2D sigma
    clipping as Fig5_plot.py's load_result. Returns:
      - a deterministic, evenly-spaced subsample of the sigma-clipped posterior
        chain of r (the actual pr(theta_k|D,Mk) for this model, used to realise
        pr(Delta|D,Mk) in eq 1), and
      - the chi-squared value at the surviving best-fit step (used to build the
        BIC-based approximation to pr(Mk|D) in eq 2).
    """
    raw_save_dir, LDL, prior_strength, seed = args
    if verbose:
        print(f"  Processing {LDL}, {prior_strength}, seed{seed}...")
    try:
        path_base = f'{raw_save_dir}{LDL}/{prior_strength}/Seed{seed}/'

        raw_chain = np.load(path_base + 'chains.npy',     mmap_mode='r')
        logprob   = np.load(path_base + 'logprob.npy',    mmap_mode='r')
        chi2      = np.load(path_base + 'chi2_chain.npy', mmap_mode='r')

        n_walkers, n_steps, n_params = raw_chain.shape
        n_steps_post = n_steps - NBURN

        burnt_chain   = np.array(raw_chain[:, NBURN:, :])
        burnt_chi2    = np.array(chi2[:,    NBURN:])
        burnt_logprob = np.array(logprob[:, NBURN:])

        good_steps_mask = np.ones((n_walkers, n_steps_post), dtype=bool)

        for round_idx in range(ROUNDS):
            THRESHOLD = THRESHOLDS[round_idx]

            alive_chi2 = burnt_chi2[good_steps_mask]
            q          = np.percentile(alive_chi2, [25, 50, 75])
            mu, iqr    = q[1], q[2] - q[0]

            chi2_bad_2d = np.zeros((n_walkers, n_steps_post), dtype=bool)
            chi2_bad_2d[good_steps_mask] = (
                (alive_chi2 < mu - THRESHOLD * iqr) | (alive_chi2 > mu + THRESHOLD * iqr)
            )

            param_bad_2d = np.zeros((n_walkers, n_steps_post), dtype=bool)
            for param_idx in range(n_params):
                alive_param = burnt_chain[:, :, param_idx][good_steps_mask]
                q       = np.percentile(alive_param, [25, 50, 75])
                mu, iqr = q[1], q[2] - q[0]
                outliers = (alive_param < mu - THRESHOLD * iqr) | (alive_param > mu + THRESHOLD * iqr)
                param_bad_2d[good_steps_mask] |= outliers

            good_steps_mask &= ~(chi2_bad_2d | param_bad_2d)

        masked_logprob         = np.where(good_steps_mask, burnt_logprob, -np.inf)
        best_walker, best_step = np.unravel_index(np.argmax(masked_logprob), masked_logprob.shape)
        chi2_best               = float(burnt_chi2[best_walker, best_step])

        r_chain_post_burnin = burnt_chain[:, :, 0][good_steps_mask]   # sigma-clipped posterior of r

        # Deterministic, evenly-spaced subsample so the cached posterior representation
        # (used to realise pr(Delta|D,Mk)) stays small while still spanning the full chain.
        n_avail = r_chain_post_burnin.shape[0]
        n_sub   = min(N_SUB, n_avail)
        sub_idx = np.linspace(0, n_avail - 1, n_sub).astype(int)
        r_samples = np.asarray(r_chain_post_burnin)[sub_idx]

        ndim = BASE_NDIM + get_n_LD_params(LDL)

        return (LDL, prior_strength, seed, r_samples, chi2_best, ndim)

    except Exception as e:
        print(f"Error loading {LDL}, {prior_strength}, seed{seed}: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_c_label_bma(c_label):
    """
    Load (or retrieve from cache) bestfit_r / std_r / chi2_best / ndim for one C label.
    Returns a dict indexed by LDL then prior_strength, holding:
      - 'r_samples': list (length len(seeds)) of subsampled sigma-clipped r chains
      - 'chi2_best': 1-D array over seeds of the chi2 at the surviving best-fit step
      - 'ndim':      scalar, number of free parameters for that LDL
    Returns None if the directory does not exist.
    """
    raw_save_dir = RAW_BASE_DIR + c_label + '/'
    if not os.path.exists(raw_save_dir):
        print(f"Skipping {c_label}: directory not found at {raw_save_dir}")
        return None

    cache_file = raw_save_dir + 'processed_data_cache_bma.pkl'
    t_start    = time.time()

    if os.path.exists(cache_file):
        print(f"Loading cached BMA data for {c_label}...")
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
        print(f"Cache loaded in {time.time() - t_start:.2f}s")
        return cached_data

    print(f"No BMA cache found for {c_label}. Loading and processing all data...")

    loading_tasks = []
    for LDL in LDLs:
        for prior_strength in prior_strengths:
            for seed in seeds:
                loading_tasks.append((raw_save_dir, LDL, prior_strength, seed))

    total_files = len(loading_tasks)
    num_chunks  = (total_files + CHUNK_SIZE - 1) // CHUNK_SIZE
    print(f"Loading {total_files} files in {num_chunks} chunks using {num_workers} worker(s)...")

    results = []
    for chunk_idx in range(num_chunks):
        chunk_tasks = loading_tasks[chunk_idx * CHUNK_SIZE: (chunk_idx + 1) * CHUNK_SIZE]

        print(f"\nProcessing chunk {chunk_idx+1}/{num_chunks} ({len(chunk_tasks)} files)...")
        t_chunk = time.time()

        with Pool(processes=num_workers) as pool:
            chunk_results = []
            for result in tqdm(pool.imap_unordered(load_result_bma, chunk_tasks),
                               total=len(chunk_tasks), desc="  Loading files",
                               ncols=80, unit="file"):
                if result is not None:
                    chunk_results.append(result)

        results.extend(chunk_results)
        elapsed = time.time() - t_chunk
        print(f"  Chunk done in {elapsed:.1f}s ({len(chunk_tasks)/elapsed:.1f} files/sec)")
        gc.collect()

    loading_time = time.time() - t_start
    print(f"\n{c_label}: loaded {len(results)}/{total_files} files in {loading_time:.1f}s")

    cached_data = {}
    for LDL in LDLs:
        cached_data[LDL] = {}
        for prior_strength in prior_strengths:
            batch = [r for r in results if r[0] == LDL and r[1] == prior_strength]
            batch.sort(key=lambda r: seeds.index(r[2]))

            cached_data[LDL][prior_strength] = {
                'r_samples': [r[3] for r in batch],
                'chi2_best': np.array([r[4] for r in batch]),
                'ndim':      batch[0][5] if batch else BASE_NDIM + get_n_LD_params(LDL),
            }

    print(f"Saving BMA cache for {c_label}...")
    with open(cache_file, 'wb') as f:
        pickle.dump(cached_data, f)
    print("Cache saved!")

    return cached_data


def compute_bma_bias(cached_data, models_to_use):
    """
    For each (prior_strength, seed), compute the Bayesian-model-averaged transit
    depth across `models_to_use` and its associated bias (in sigma) relative to
    the true injected depth.

    Model weights are BIC-based Bayes-factor approximations:
        BIC_k    = chi2_best_k + ndim_k * ln(N_TOTAL_PTS)
        w_k     ~ exp(-0.5 * (BIC_k - min_j BIC_j)),  normalised to sum to 1

    Eq (1), pr(Delta|D) = sum_k w_k * pr(Delta|D,Mk), is built literally rather
    than approximated by a per-model point + Gaussian error: for each of
    N_MIX_DRAWS Monte Carlo draws, a model Mk is picked with probability w_k and
    a posterior depth sample (r^2) is drawn (with replacement) from that model's
    own cached, sigma-clipped chain. The bias is then the offset of the resulting
    pooled mixture's mean from the truth, in units of the mixture's own std.

    Returns a dict {prior_strength: np.array of biases over seeds}.
    """
    biases_by_prior = {}

    for ips, prior_strength in enumerate(prior_strengths):
        biases = []
        for iseed, seed in enumerate(seeds):
            chi2_best = np.array([cached_data[LDL][prior_strength]['chi2_best'][iseed] for LDL in models_to_use])
            ndim      = np.array([cached_data[LDL][prior_strength]['ndim']             for LDL in models_to_use])
            r_samples_list = [cached_data[LDL][prior_strength]['r_samples'][iseed] for LDL in models_to_use]

            BIC = chi2_best + ndim * np.log(N_TOTAL_PTS)
            w   = np.exp(-0.5 * (BIC - BIC.min()))
            w  /= w.sum()

            # Deterministic per-(prior, seed) RNG so the mixture realisation is reproducible.
            rng          = np.random.default_rng(seed * 100 + ips)
            model_draws  = rng.choice(len(models_to_use), size=N_MIX_DRAWS, p=w)
            pooled_depth = np.empty(N_MIX_DRAWS)
            for k in range(len(models_to_use)):
                mask = model_draws == k
                n_k  = int(mask.sum())
                if n_k > 0:
                    pooled_depth[mask] = rng.choice(r_samples_list[k], size=n_k, replace=True) ** 2

            bma_mean = pooled_depth.mean()
            bma_std  = pooled_depth.std()
            biases.append(np.abs(bma_mean - TRUE_DEPTH) / bma_std)

        biases_by_prior[prior_strength] = np.array(biases)

    return biases_by_prior


def plot_bma_row(ax, biases_by_prior, c_label, is_bottom_row):
    """Draw one C-label's row: a boxplot of BMA bias (10 seeds) per prior strength."""
    positions = np.arange(1, len(prior_strengths) + 1)

    for ips, (prior_strength, prior_label, color) in enumerate(
        zip(prior_strengths, prior_strengths_labels, PRIOR_COLORS)
    ):
        data = biases_by_prior[prior_strength]
        ax.boxplot(
            data, positions=[positions[ips]], patch_artist=True,
            boxprops=dict(facecolor=color, color=color, alpha=0.85),
            widths=[0.5],
            medianprops=dict(color='gold', linewidth=1.5),
            whiskerprops=dict(color=color, linewidth=1.5),
            capprops=dict(color=color, linewidth=1.5),
            flierprops=dict(marker='o', color=color, markersize=5),
            showfliers=False,
        )
        # Overlay individual seeds so all 10 points per prior strength are visible.
        jitter = (np.random.default_rng(0).uniform(-0.08, 0.08, size=len(data)))
        ax.scatter(positions[ips] + jitter, data, color='0.2', s=10, zorder=3, alpha=0.7)

    ax.set_yscale('log')
    ax.set_ylim([0.03, 120])
    ax.set_xlim([0.4, len(prior_strengths) + 0.6])

    ax.set_xticks(positions)
    if is_bottom_row:
        ax.set_xticklabels(prior_strengths_labels, fontsize=10)
    else:
        ax.set_xticklabels([])

    ax.set_ylabel(r'BMA Transit Depth Bias ($\sigma$)', fontsize=11)
    ax.set_title(C_LABEL_NAMES[c_label], fontsize=12, fontweight='bold', loc='left')

    band_kwargs = dict(facecolor='green', alpha=0.2, edgecolor='none', zorder=-1)
    ax.axhspan(0.1, 2.0, **band_kwargs)
    if is_bottom_row:
        ax.text(0.5, 2.2, r'No bias', fontsize=10, color='seagreen')

    grid_color = '0.85'
    for pos in positions:
        ax.axvline(pos, color=grid_color, zorder=0)
    for val in [0.1, 1, 10, 100]:
        ax.axhline(val, color=grid_color, zorder=0)


def build_and_save_bma_figure(all_biases, out_name):
    """Build the stacked per-C-label BMA bias figure and save it to paths.figures."""
    available_c_labels = [c_label for c_label in C_LABELS if c_label in all_biases]
    if not available_c_labels:
        print(f"Skipping {out_name}: no data available.")
        return

    n_panels   = len(available_c_labels)
    fig_height = n_panels * 3.2
    fig, axes  = plt.subplots(n_panels, 1, figsize=(8, fig_height), sharex=True)
    if n_panels == 1:
        axes = [axes]

    for ic, c_label in enumerate(available_c_labels):
        is_bottom = (ic == n_panels - 1)
        plot_bma_row(axes[ic], all_biases[c_label], c_label, is_bottom)

    fig.tight_layout()
    plt.savefig(paths.figures / f"{out_name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"{out_name}.pdf saved.")


#############################################
################ Running code ###############
#############################################

if __name__ == '__main__':

    # ── Load / build cache for each C label ──────────────────────────────
    all_cached_data = {}
    for c_label in C_LABELS:
        data = process_c_label_bma(c_label)
        if data is not None:
            all_cached_data[c_label] = data

    if not all_cached_data:
        raise RuntimeError("No C-label data found. Check RAW_BASE_DIR.")

    # ── Version 1: BMA across all fitted LDLs ────────────────────────────
    all_biases = {
        c_label: compute_bma_bias(cached_data, LDLs)
        for c_label, cached_data in all_cached_data.items()
    }
    build_and_save_bma_figure(all_biases, "Fig5_BMA_all")

    # ── Version 2: BMA across quadratic / 3rd-order / 4th-order NL laws only ──
    reduced_biases = {
        c_label: compute_bma_bias(cached_data, REDUCED_LDLs)
        for c_label, cached_data in all_cached_data.items()
    }
    build_and_save_bma_figure(reduced_biases, "Fig5_BMA_reduced")
