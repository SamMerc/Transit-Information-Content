#############################
########## Purpose ##########
#############################

# Figure 4 showcases the amplification factor change across limb-darkening laws and priors used.
# The goal of this file is to retrieve the results from the injection-retrievals computed previously
# and compile them into Figure 4.  Three sub-panels show results for three different sets of
# injected 4th-order non-linear LDCs (C1, C3, C4).


######################################
########## Import libraries ##########
######################################
from jax import jit
import jax
import paths
print(f"JAX devices: {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")
import jax.numpy as jnp
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"
import astropy.units as u
from astropy.constants import G
import numpy as np
import os
import time
from tqdm import tqdm
import pickle
from multiprocessing import Pool
import gc
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# For 64-bit precision since JAX defaults to 32-bit
jax.config.update("jax_enable_x64", True)


#############################################
########## Define hyper-parameters ##########
#############################################

#%% System parameters (must match Fig4_run.py)
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

num_IT_pts = jnp.sum(
    (init_state_dic['times'] > init_state_dic['t0'] - T_dur / 2) &
    (init_state_dic['times'] < init_state_dic['t0'] + T_dur / 2)
)

#%% Grid parameters
RAW_BASE_DIR = '/Volumes/Ajax/Work/PhD/Research/Transit-Information-Content/Fig4_Storage/' #str(paths.data / 'Fig4_Storage') + '/'
model_scatter = 16.68100537200059

LDLs              = ['PLD_2', 'PLD_3', 'PLD_4', 'PLD_5', 'PLD_6', 'PLD_9', '4NLLD']
prior_strengths   = ['uniform', 'gauss_20', 'gauss_10', 'gauss_5', 'gauss_1']
prior_strengths_labels = ['Uniform', r'$20\%$ Gaussian', r'$10\%$ Gaussian',
                          r'$5\%$ Gaussian', r'$1\%$ Gaussian']
seeds             = [40, 50, 60, 70, 80, 90, 100, 110, 120, 130]

# C labels correspond to the three stellar LDC sets in Fig4_run.py
#C0 : 0.5597   -0.1310    0.4556   -0.2398
#C2 : 0.7617   -0.8502    1.4229   -0.4899
#C3 : 0.6666   -0.8767    0.8343   -0.2968
#C4 : 0.5172   -0.1913    0.0819   -0.0316
#C7 : 0.6383   -0.0511   -0.2722    0.1455
C_LABELS      = ['C0', 'C2', 'C3']
C_LABEL_NAMES = {'C0': 'Cluster 0 - M/K type, metal poor, near-infrared', 
                 'C2': 'Cluster 2 - M/K type, metal rich, optical', 
                 'C3': 'Cluster 3 - M/K type, solar metallicity, mid-infrared',
                 'C4': 'Cluster 4 - K/G type, solar metallicity, mid-infrared',
                 'C7': 'Cluster 7 - G type, solar metallicity, near-infrared'}

#%% Filtering / processing parameters (same as Fig1_plot.py)
NBURN      = 70000
THRESHOLDS = [5, 4, 3]
ROUNDS     = 3
num_workers = 1
CHUNK_SIZE  = 30
verbose     = False


##############################
##### Relevant functions #####
##############################

def load_result(args):
    """
    Load one (LDL, prior_strength, seed) run and apply iterative 2D sigma clipping.
    Adapted from Fig1_plot.py's load_result.
    """
    raw_save_dir, LDL, prior_strength, seed, return_full = args
    if verbose:
        print(f"  Processing {LDL}, {prior_strength}, seed{seed}...")
    try:
        path_base = f'{raw_save_dir}{LDL}/{prior_strength}/Seed{seed}/'

        raw_chain = np.load(path_base + 'chains.npy',     mmap_mode='r')
        logprob   = np.load(path_base + 'logprob.npy',    mmap_mode='r')
        chi2      = np.load(path_base + 'chi2_chain.npy', mmap_mode='r')

        n_walkers, n_steps, n_params = raw_chain.shape
        n_steps_post = n_steps - NBURN

        burnt_chain   = np.array(raw_chain[:, NBURN:, :])   # (n_walkers, n_steps_post, n_params)
        burnt_chi2    = np.array(chi2[:,    NBURN:])         # (n_walkers, n_steps_post)
        burnt_logprob = np.array(logprob[:, NBURN:])         # (n_walkers, n_steps_post)

        good_steps_mask = np.ones((n_walkers, n_steps_post), dtype=bool)

        for round_idx in range(ROUNDS):
            THRESHOLD = THRESHOLDS[round_idx]

            # --- Filter on chi2 ---
            alive_chi2 = burnt_chi2[good_steps_mask]
            q          = np.percentile(alive_chi2, [25, 50, 75])
            mu, iqr    = q[1], q[2] - q[0]

            chi2_bad_2d = np.zeros((n_walkers, n_steps_post), dtype=bool)
            chi2_bad_2d[good_steps_mask] = (
                (alive_chi2 < mu - THRESHOLD * iqr) | (alive_chi2 > mu + THRESHOLD * iqr)
            )

            # --- Filter on parameters ---
            param_bad_2d = np.zeros((n_walkers, n_steps_post), dtype=bool)
            for param_idx in range(n_params):
                alive_param = burnt_chain[:, :, param_idx][good_steps_mask]
                q     = np.percentile(alive_param, [25, 50, 75])
                mu, iqr = q[1], q[2] - q[0]
                outliers = (alive_param < mu - THRESHOLD * iqr) | (alive_param > mu + THRESHOLD * iqr)
                param_bad_2d[good_steps_mask] |= outliers

            good_steps_mask &= ~(chi2_bad_2d | param_bad_2d)

        # Best-fit: highest logprob among surviving pairs
        masked_logprob         = np.where(good_steps_mask, burnt_logprob, -np.inf)
        best_walker, best_step = np.unravel_index(np.argmax(masked_logprob), masked_logprob.shape)
        bestfit_r              = float(burnt_chain[best_walker, best_step, 0])

        # r chain: all surviving (walker, step) pairs
        r_chain_post_burnin = burnt_chain[:, :, 0][good_steps_mask]   # (n_surviving,)

        if return_full:
            return (LDL, prior_strength, seed, r_chain_post_burnin, bestfit_r,
                    True, np.array(raw_chain), np.array(logprob), np.array(chi2), good_steps_mask)
        else:
            return (LDL, prior_strength, seed, r_chain_post_burnin, bestfit_r,
                    False, None, None, None, None)

    except Exception as e:
        print(f"Error loading {LDL}, {prior_strength}, seed{seed}: {e}")
        import traceback
        traceback.print_exc()
        return None


@jit
def compute_amplification_factor_jax(r_chain_flat, bestfit_r, model_scatter, num_IT_pts):
    std_r              = jnp.std(r_chain_flat)
    bestfit_r_error    = 2 * std_r * bestfit_r
    scatter_in_bin     = (model_scatter * 1e-6) / jnp.sqrt(num_IT_pts)
    return bestfit_r_error / scatter_in_bin


@jit
def compute_sigma_bias_jax(bestfit_r, r_chain_flat, true_r):
    std_r           = jnp.std(r_chain_flat)
    bestfit_r_error = 2 * std_r * bestfit_r
    return jnp.abs((bestfit_r**2 - true_r**2) / bestfit_r_error)


def process_c_label(c_label):
    """
    Load (or retrieve from cache) processed amp factors and biases for one C label.
    Returns a dict with keys 'amp_factors' and 'biases', each a nested dict
    indexed by LDL string then prior_strength string, holding a 1-D array over seeds.
    Returns None if the directory does not exist.
    """
    raw_save_dir = RAW_BASE_DIR + c_label + '/'
    if not os.path.exists(raw_save_dir):
        print(f"Skipping {c_label}: directory not found at {raw_save_dir}")
        return None

    cache_file = raw_save_dir + 'processed_data_cache.pkl'
    t_start    = time.time()

    if os.path.exists(cache_file):
        print(f"Loading cached data for {c_label}...")
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
        print(f"Cache loaded in {time.time() - t_start:.2f}s")
        return cached_data

    print(f"No cache found for {c_label}. Loading and processing all data...")

    # Build task list
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
        chunk_tasks = loading_tasks[chunk_idx * CHUNK_SIZE : (chunk_idx + 1) * CHUNK_SIZE]
        chunk_tasks_enhanced = [(*task, False) for task in chunk_tasks]

        print(f"\nProcessing chunk {chunk_idx+1}/{num_chunks} ({len(chunk_tasks)} files)...")
        t_chunk = time.time()

        with Pool(processes=num_workers) as pool:
            chunk_results = []
            for result in tqdm(pool.imap_unordered(load_result, chunk_tasks_enhanced),
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

    # Build cached_data dict indexed by LDL key then prior_strength key
    print("Computing amplification factors and biases...")
    cached_data = {'amp_factors': {}, 'biases': {}}

    for LDL in LDLs:
        cached_data['amp_factors'][LDL] = {}
        cached_data['biases'][LDL] = {}
        for prior_strength in prior_strengths:
            batch = [r for r in results if r[0] == LDL and r[1] == prior_strength]

            amp_factors = []
            biases      = []
            for _, _, _, r_chain_post_burnin, bestfit_r, _, _, _, _, _ in batch:
                r_jax = jnp.array(r_chain_post_burnin.flatten())
                amp_factors.append(float(compute_amplification_factor_jax(
                    r_jax, bestfit_r, model_scatter, num_IT_pts
                )))
                biases.append(float(compute_sigma_bias_jax(
                    bestfit_r, r_jax, init_state_dic['r']
                )))

            cached_data['amp_factors'][LDL][prior_strength] = np.array(amp_factors)
            cached_data['biases'][LDL][prior_strength]      = np.array(biases)

    print(f"Saving cache for {c_label}...")
    with open(cache_file, 'wb') as f:
        pickle.dump(cached_data, f)
    print("Cache saved!")

    return cached_data


def draw_matched_breaks(fig, left_ax, right_ax, size=0.006, lw=1.0, color='k'):
    """
    Draw diagonal "/" break marks at the interior spine boundary between two axes,
    placed in figure coordinates so the marks scale consistently across panels.
    """
    lb = left_ax.get_position()
    rb = right_ax.get_position()
    y_top = 0.5 * (lb.y1 + rb.y1)
    y_bot = 0.5 * (lb.y0 + rb.y0)
    for xc in [lb.x1, rb.x0]:
        for yc in [y_top, y_bot]:
            fig.add_artist(Line2D([xc - size, xc + size], [yc - size, yc + size],
                                  transform=fig.transFigure, color=color, lw=lw, clip_on=False))


def plot_two_rows(fig, outer_gs_cell, cached_data, c_label, is_bottom_row):
    """
    Render amp-factor (top) and bias (bottom) for one C label using a 2x3 broken-axis
    layout: left panel covers PLD_2-PLD_6, middle panel PLD_9, right panel 4NLLD.
    Interior spines are hidden; break marks are drawn by the caller via draw_matched_breaks.
    Returns (ax1, ax1m, ax1r, ax2, ax2m, ax2r).
    """
    inner_gs = gridspec.GridSpecFromSubplotSpec(
        2, 3, subplot_spec=outer_gs_cell,
        width_ratios=[3.5, 1, 1], wspace=0.05, hspace=0.08
    )
    ax1  = fig.add_subplot(inner_gs[0, 0])   # amp,  left
    ax1m = fig.add_subplot(inner_gs[0, 1])   # amp,  middle
    ax1r = fig.add_subplot(inner_gs[0, 2])   # amp,  right
    ax2  = fig.add_subplot(inner_gs[1, 0])   # bias, left
    ax2m = fig.add_subplot(inner_gs[1, 1])   # bias, middle
    ax2r = fig.add_subplot(inner_gs[1, 2])   # bias, right

    left_LDLs   = ['PLD_2', 'PLD_3', 'PLD_4', 'PLD_5', 'PLD_6']
    middle_LDLs = ['PLD_9']
    right_LDLs  = ['4NLLD']
    positions_left   = np.array([2, 3, 4, 5, 6])
    positions_middle = np.array([9])
    positions_right  = np.array([10])

    # ── Box plots ──────────────────────────────────────────────────────────
    for ips, (prior_strength, prior_label) in enumerate(zip(prior_strengths, prior_strengths_labels)):
        alpha = 1.0 - 0.2 * ips
        if prior_strength == 'gauss_20':
            boxcolor = 'blue'
        elif prior_strength == 'gauss_10':
            boxcolor = 'royalblue'
        else:
            boxcolor = 'black'

        for ax_row, content_key in zip(
            [[ax1, ax1m, ax1r], [ax2, ax2m, ax2r]],
            ['amp_factors', 'biases']
        ):
            for panel_ax, xpositions, panel_LDLs in zip(
                ax_row,
                [positions_left, positions_middle, positions_right],
                [left_LDLs, middle_LDLs, right_LDLs]
            ):
                for iLDL, LDL in enumerate(panel_LDLs):
                    pos  = xpositions[iLDL] - 0.16 + 0.08 * ips
                    data = cached_data[content_key][LDL][prior_strength]
                    panel_ax.boxplot(
                        data, positions=[pos], patch_artist=True,
                        boxprops=dict(facecolor=boxcolor, color=boxcolor, alpha=alpha),
                        widths=[0.1],
                        medianprops=dict(color='gold', linewidth=1, alpha=alpha),
                        whiskerprops=dict(color=boxcolor, linewidth=1.5, alpha=alpha),
                        capprops=dict(color=boxcolor, linewidth=1.5, alpha=alpha),
                        flierprops=dict(marker='o', color=boxcolor, markersize=5, alpha=alpha),
                        showfliers=False,
                        label=prior_label if iLDL == 0 else '',
                    )

    # ── Scales and limits ──────────────────────────────────────────────────
    for ax in [ax1, ax1m, ax1r, ax2, ax2m, ax2r]:
        ax.set_yscale('log')
    for ax in [ax1, ax1m, ax1r]:
        ax.set_ylim([1., 120])
    for ax in [ax2, ax2m, ax2r]:
        ax.set_ylim([0.1, 120])
    ax1.set_xlim([1.5, 6.5]);   ax2.set_xlim([1.5, 6.5])
    ax1m.set_xlim([8.5, 9.5]);  ax2m.set_xlim([8.5, 9.5])
    ax1r.set_xlim([9.5, 10.5]); ax2r.set_xlim([9.5, 10.5])

    # ── Hide interior spines ───────────────────────────────────────────────
    ax1.spines['right'].set_visible(False);  ax2.spines['right'].set_visible(False)
    ax1m.spines['left'].set_visible(False);  ax2m.spines['left'].set_visible(False)
    ax1m.spines['right'].set_visible(False); ax2m.spines['right'].set_visible(False)
    ax1r.spines['left'].set_visible(False);  ax2r.spines['left'].set_visible(False)

    # ── y ticks (only on left panels) ─────────────────────────────────────
    ax1.set_yticks([10, 100]);            ax1.set_yticklabels([10, 100])
    ax2.set_yticks([0.1, 1, 10, 100]);   ax2.set_yticklabels([0.1, 1, 10, 100])
    for ax in [ax1m, ax2m, ax1r, ax2r]:
        ax.set_yticks([])
        ax.tick_params(axis='y', which='both', left=False, labelleft=False)

    # ── x ticks ────────────────────────────────────────────────────────────
    ax1.set_xticks(positions_left);   ax1.set_xticklabels([])
    ax1m.set_xticks(positions_middle); ax1m.set_xticklabels([])
    ax1r.set_xticks(positions_right);  ax1r.set_xticklabels([])

    if is_bottom_row:
        ax2.set_xticks(positions_left)
        ax2.set_xticklabels([2, 3, 4, 5, 6], color='red')
        ax2.get_xticklabels()[1].set_color('green')   # PLD_3
        ax2m.set_xticks(positions_middle); ax2m.set_xticklabels([9], color='red')
        ax2r.set_xticks(positions_right);  ax2r.set_xticklabels(['4NLLD'], color='green')
    else:
        ax2.set_xticks(positions_left);    ax2.set_xticklabels([])
        ax2m.set_xticks(positions_middle); ax2m.set_xticklabels([])
        ax2r.set_xticks(positions_right);  ax2r.set_xticklabels([])

    # ── Labels ─────────────────────────────────────────────────────────────
    ax1.set_ylabel(r'Amplification Factor ($A$)', fontsize=12)
    ax2.set_ylabel(r'Transit Depth Bias ($\sigma$)', fontsize=12)

    # ── Row label ──────────────────────────────────────────────────────────
    ax1.set_title(C_LABEL_NAMES[c_label], fontsize=13, fontweight='bold', loc='left')

    # ── Legend ─────────────────────────────────────────────────────────────
    ax1.legend(fontsize=10, loc='upper left', ncols=2)

    # ── Annotations ────────────────────────────────────────────────────────
    band_kwargs = dict(facecolor='green', alpha=0.2, edgecolor='none', zorder=-1)
    for ax in [ax1, ax1m, ax1r]: ax.axhspan(1., 6., **band_kwargs)
    for ax in [ax2, ax2m, ax2r]: ax.axhspan(0.1, 2.0, **band_kwargs)
    for ax in [ax1, ax1m, ax1r]: ax.axhline(np.sqrt(3/2), linestyle='dashed', color='black')
    ax1.text(2.4, np.sqrt(3/2) + 0.2, r'Theoretical limit @ $\sqrt{3/2}$', fontsize=10, color='black')
    ax1.text(1.6, 6.4,   r'Acceptable $A$', fontsize=10, color='seagreen')
    ax2.text(1.6, 2.1, r'No bias',        fontsize=10, color='seagreen')

    # ── Grid lines ─────────────────────────────────────────────────────────
    grid_color = '0.8'
    for pos in positions_left:
        ax1.axvline(pos,  color=grid_color, zorder=0)
        ax2.axvline(pos,  color=grid_color, zorder=0)
    for pos in positions_middle:
        ax1m.axvline(pos, color=grid_color, zorder=0)
        ax2m.axvline(pos, color=grid_color, zorder=0)
    for pos in positions_right:
        ax1r.axvline(pos, color=grid_color, zorder=0)
        ax2r.axvline(pos, color=grid_color, zorder=0)
    for val in [10, 100]:
        for ax in [ax1, ax1m, ax1r]: ax.axhline(val, color=grid_color, zorder=0)
    for val in [0.1, 1, 10, 100]:
        for ax in [ax2, ax2m, ax2r]: ax.axhline(val, color=grid_color, zorder=0)

    return ax1, ax1m, ax1r, ax2, ax2m, ax2r


#############################################
################ Running code ###############
#############################################

if __name__ == '__main__':

    # ── Load / build cache for each C label ──────────────────────────────
    all_cached_data    = {}
    available_c_labels = []

    for c_label in C_LABELS:
        data = process_c_label(c_label)
        if data is not None:
            all_cached_data[c_label] = data
            available_c_labels.append(c_label)

    if not available_c_labels:
        raise RuntimeError("No C-label data found. Check RAW_BASE_DIR.")

    # ── Build figure ──────────────────────────────────────────────────────
    n_panels   = len(available_c_labels)
    fig_height = n_panels * 6
    fig = plt.figure(figsize=(13, fig_height))

    outer_gs = gridspec.GridSpec(n_panels, 1, hspace=0.1, figure=fig)

    all_axes = []   # list of (ax1, ax1m, ax1r, ax2, ax2m, ax2r) per C label

    for ic, c_label in enumerate(available_c_labels):
        is_bottom = (ic == n_panels - 1)
        axes_tuple = plot_two_rows(
            fig, outer_gs[ic], all_cached_data[c_label], c_label, is_bottom
        )
        all_axes.append(axes_tuple)

    # Draw break marks after all axes exist (needs finalised positions)
    for ax1, ax1m, ax1r, ax2, ax2m, ax2r in all_axes:
        draw_matched_breaks(fig, ax1,  ax1m, size=0.006)
        draw_matched_breaks(fig, ax1m, ax1r, size=0.006)
        draw_matched_breaks(fig, ax2,  ax2m, size=0.006)
        draw_matched_breaks(fig, ax2m, ax2r, size=0.006)

    # x-axis label centred across the full bottom row
    if all_axes:
        ax1_last, _, ax1r_last, ax2_last, _, _ = all_axes[-1]
        left_box  = ax2_last.get_position()
        right_box = ax1r_last.get_position()
        x_center  = 0.5 * (left_box.x0 + right_box.x1)
        y_label   = left_box.y0 - 0.02
        fig.text(x_center, y_label, 'Number of LDCs',
                 ha='center', va='top', fontsize=12)

    plt.savefig(paths.figures / "Fig4.pdf", bbox_inches="tight")
    plt.close()
    print("Fig4.pdf saved.")
