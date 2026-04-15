#############################
########## Purpose ##########
#############################

# Restricted analysis for a single (b, p) pair drawn from the intensity profile
# grid produced by Fig2_side_helper.py.  For a user-specified impact parameter b
# and planet-to-star radius ratio p, this script:
#   1. Loads the pre-built intensity profile grid (gen_dict) from disk.
#   2. Collects every valid profile at that (b, p).
#   3. Runs PCA and hierarchical clustering to identify structure.
#   4. Fits each profile with a 4th-order non-linear limb-darkening law,
#      using rs_mask = r <= 1 - p (consistent with Fig2_side_helper.py).
#   5. Produces corner plots coloured by wavelength, Teff, logg, [M/H] and
#      PCA cluster, as well as representative profile plots with NLLD fits.
#   6. Saves all outputs — arrays, figures, and a summary pickle — to a
#      dedicated sub-directory.
#
# This analysis was separated from the full pipeline because it targets a
# specific fiducial system (b=0, p=0.1) needed for the injection-retrieval
# simulations in Paper 1, while the rest of the pipeline feeds Paper 2.


######################################
########## Import libraries ##########
######################################

import numpy as np
import matplotlib
import os
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle
from sklearn.decomposition import PCA
from lmfit import minimize, Parameters
import gc
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import corner
from tqdm import tqdm
from scipy.cluster.hierarchy import (linkage, fcluster,
                                     dendrogram as scipy_dendrogram,
                                     optimal_leaf_ordering)
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

######################################
########## Hyper-parameters ##########
######################################

orig_load_data_path = '/Volumes/Ajax/Work/PhD/Research/Transit-Information-Content/Fig2_sidehelper_Storage/'
orig_save_data_path = '/Volumes/Ajax/Work/PhD/Research/Transit-Information-Content/Fig2_helper_Storage/'

models = ['mps1']  # must match the models run in Fig2_side_helper.py

Teffs = {
    'phoenix': [2300, 15000],
    'kurucz':  [3500,  6500],
    'stagger': [4000,  7000],
    'mps2':    [3500,  9000],
    'mps1':    [3500,  9000],
}

loggs = {
    'phoenix': [0.0, 6.0],
    'kurucz':  [4.0, 5.0],
    'stagger': [1.5, 5.0],
    'mps1':    [3.0, 5.0],
    'mps2':    [3.0, 5.0],
}

metallicitys = {
    'phoenix': [-1.5,  1.0],
    'kurucz':  [-5.0,  1.0],
    'stagger': [-3.0,  0.0],
    'mps1':    [-5.0,  1.5],
    'mps2':    [-5.0,  1.5],
}

N_star   = 5
N_chords = 100

N_bs_ps = 5
# Must match the bs / ps used in Fig2_side_helper.py
import jax.numpy as jnp
bs = jnp.linspace(0, 1, N_bs_ps)
ps = jnp.logspace(-3, -1, N_bs_ps)

n_components = 4
cmap   = plt.cm.coolwarm
colors = cmap(np.linspace(0, 1, n_components))

# ── Profile subsampling ───────────────────────────────────────────────────────
subsample_profiles   = False
n_subsample_profiles = 50000
subsample_seed       = 42

excluded_bp_pairs = [
    # (N_bs_ps - 1, 0),
]

plot_dendogram = False

# ── Target (b, p) for this analysis ──────────────────────────────────────────
target_b_val = 0.0
target_p_val = 1e-1


############################
###### Function block ######
############################

def hierarchical_clustering(
    data,
    label,
    save_path,
    feature_labels    = None,
    cutoff            = None,
    method            = 'complete',
    max_display       = 60,
    n_subsample       = 30_000,
    external_labels   = None,
    clustering_metric = 'euclidean',
):
    """
    Hierarchical clustering with scipy pairwise distance computation.

    Parameters
    ----------
    data            : np.ndarray (N, D)
    label           : str
    save_path       : str
    feature_labels  : list of str, optional
    cutoff          : float or None
    method          : str            linkage method, default 'complete', Can use 'single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward'
    max_display     : int            max dendrogram leaves shown
    n_subsample     : int            scatter plot cap per cluster
    external_labels : np.ndarray or None   pre-computed labels (skips clustering)
    clustering_metric: str           distance metric for pdist, default 'euclidean', can use any metric supported by scipy's cdist ('braycurtis', 'canberra', 
                                    'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon', 'mahalanobis',
                                    'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalsneath', 'sqeuclidean', 'yule')
    Returns
    -------
    labels      : np.ndarray (N,)   1-indexed cluster labels
    cutoff_used : float
    Z           : np.ndarray (N-1, 3) or None
    """
    N, D = data.shape
    if feature_labels is None:
        feature_labels = [f'Feature {d}' for d in range(D)]

    # ─────────────────────────────────────────────────────────────────────────
    # External-labels path — skip clustering entirely, go straight to plotting
    # ─────────────────────────────────────────────────────────────────────────
    if external_labels is not None:
        valid_ext  = external_labels >= 0
        data       = data[valid_ext]
        labels     = (external_labels[valid_ext] + 1).astype(int)
        unique_cl  = np.unique(labels)
        n_cls      = len(unique_cl)
        cutoff_used = np.nan
        Z           = None
        print(f'  [{label}] External labels: {n_cls} modes, '
              f'{valid_ext.sum():,} valid profiles')

        # ── Colour palette ─────────────────────────────
        cls_colors = plt.cm.tab10(np.linspace(0, 1, min(n_cls, 10)))
        if n_cls > 10:
            cls_colors = plt.cm.hsv(np.linspace(0, 0.9, n_cls))

    else:
        # ── 1. Standardise ────────────────────────────────────────────────────
        scaler      = StandardScaler()
        data_scaled = scaler.fit_transform(data).astype(np.float32)

        # ── 2. Scipy cdist in row-chunks with tqdm progress ───────────────────
        n_pairs  = N * (N - 1) // 2
        dist_vec = np.empty(n_pairs, dtype=np.float32)
        CHUNK    = max(1, N // 50)

        cdist_kwargs = {}
        if clustering_metric == 'mahalanobis':
            cov = np.cov(data_scaled, rowvar=False)
            VI  = np.linalg.inv(cov).astype(np.float64)
            cdist_kwargs['VI'] = VI

        ptr = 0
        print(f'  [{label}] Computing {N:,}x{N:,} distance matrix '
              f'({n_pairs:,} pairs) with scipy cdist in row-chunks of {CHUNK} ...')

        with tqdm(
            total        = N,
            desc         = f'  [{label}] pdist',
            unit         = ' rows',
            dynamic_ncols= True,
            bar_format   = ('{l_bar}{bar}| {n_fmt}/{total_fmt} rows '
                            '[{elapsed}<{remaining}, {rate_fmt}]'),
        ) as pbar:
            for chunk_start in range(0, N, CHUNK):
                chunk_end  = min(chunk_start + CHUNK, N)
                query_rows = data_scaled[chunk_start:chunk_end]

                # Only compute distances to columns strictly right of the diagonal
                # to fill the condensed vector in the correct scipy order.
                for local_i, global_i in enumerate(range(chunk_start, chunk_end)):
                    if global_i == N - 1:
                        break
                    right_cols = data_scaled[global_i + 1:]
                    row_dists  = cdist(
                        query_rows[local_i:local_i + 1],
                        right_cols,
                        metric=clustering_metric,
                        **cdist_kwargs,
                    )[0]
                    n_vals = len(row_dists)
                    if clustering_metric == 'cityblock':
                        dist_vec[ptr:ptr + n_vals] = row_dists / D
                    else:
                        dist_vec[ptr:ptr + n_vals] = row_dists
                    ptr += n_vals

                pbar.update(chunk_end - chunk_start)
                pbar.set_postfix(
                    pairs_filled=f'{ptr:,}/{n_pairs:,}',
                    pct=f'{100 * ptr / n_pairs:.1f}%',
                    refresh=False,
                )

        assert ptr == n_pairs, f'Distance vector incomplete: {ptr} / {n_pairs}'

        # ── 3. Linkage tree ───────────────────────────────────────────────────
        print(f'  [{label}] Building linkage tree ...')
        with tqdm(total=1, desc=f'  [{label}] linkage',
                  bar_format='{l_bar}{bar}| {elapsed}') as pbar:
            Z = linkage(dist_vec.astype(np.float64), method=method)
            pbar.update(1)

        # ── 4. Auto-select cutoff ─────────────────────────────────────────────
        if cutoff is None:
            merge_dists = Z[:, 2]
            gaps        = np.diff(merge_dists)
            elbow       = np.argmax(gaps)
            cutoff      = float((merge_dists[elbow] + merge_dists[elbow + 1]) / 2)
            print(f'  [{label}] Auto cutoff : {cutoff:.4f}  '
                  f'(gap {merge_dists[elbow]:.4f} → {merge_dists[elbow+1]:.4f})')
        else:
            print(f'  [{label}] Using cutoff: {cutoff:.4f}')
        cutoff_used = cutoff

        # ── 5. Cut tree ───────────────────────────────────────────────────────
        labels    = fcluster(Z, t=cutoff, criterion='distance')
        unique_cl = np.unique(labels)
        n_cls     = len(unique_cl)
        print(f'  [{label}] {n_cls} clusters found')
        for cl in unique_cl:
            print(f'    Cluster {cl}: {np.sum(labels == cl):6d} profiles')

        # ── 6. Reassign singleton clusters ────────────────────────────────────
        unique_cl         = np.unique(labels)
        singleton_cls     = [cl for cl in unique_cl if np.sum(labels == cl) == 1]
        non_singleton_cls = [cl for cl in unique_cl if np.sum(labels == cl) > 1]

        if len(singleton_cls) > 0 and len(non_singleton_cls) > 0:
            print(f'  [{label}] Reassigning {len(singleton_cls)} singleton cluster(s)')

            centroids = np.array([
                data_scaled[labels == cl].mean(axis=0)
                for cl in non_singleton_cls
            ])

            for scl in singleton_cls:
                idx_singleton = np.where(labels == scl)[0][0]
                point  = data_scaled[idx_singleton:idx_singleton + 1]
                dists  = cdist(point, centroids, metric='euclidean')[0]
                nearest = non_singleton_cls[np.argmin(dists)]
                print(f'    Singleton cluster {scl} (idx={idx_singleton}) '
                      f'→ cluster {nearest}')
                labels[idx_singleton] = nearest

            unique_cl = np.unique(labels)
            n_cls     = len(unique_cl)
            print(f'  [{label}] After reassignment: {n_cls} clusters')
            for cl in unique_cl:
                print(f'    Cluster {cl}: {np.sum(labels == cl):6d} profiles')

        cls_colors = plt.cm.tab10(np.linspace(0, 1, min(n_cls, 10)))
        if n_cls > 10:
            cls_colors = plt.cm.hsv(np.linspace(0, 0.9, n_cls))

        # ── 7. Dendrogram ─────────────────────────────────────────────────────
        if plot_dendogram:
            print(f'  [{label}] Plotting dendrogram ...')
            fig_d, axes_d = plt.subplots(1, 2, figsize=(16, 6),
                                         gridspec_kw={'width_ratios': [3, 1]})
            fig_d.suptitle(f'Hierarchical clustering — {label}', fontsize=12)

            Z_ol = optimal_leaf_ordering(Z, dist_vec.astype(np.float64))
            scipy_dendrogram(
                Z_ol,
                ax=axes_d[0],
                truncate_mode='lastp',
                p=max_display,
                color_threshold=cutoff,
                above_threshold_color='gray',
                no_labels=True,
            )
            axes_d[0].axhline(cutoff, color='red', linestyle='--',
                              linewidth=1.5, label=f'cutoff = {cutoff:.3f}')
            axes_d[0].set_xlabel('Profiles (truncated to last merges)')
            axes_d[0].set_ylabel('Merge distance')
            axes_d[0].set_title('Dendrogram')
            axes_d[0].legend(fontsize=8)
            axes_d[0].grid(True, alpha=0.2)

            counts = [np.sum(labels == cl) for cl in unique_cl]
            axes_d[1].barh(unique_cl, counts, color=cls_colors[:n_cls],
                           edgecolor='k', linewidth=0.5)
            axes_d[1].set_xlabel('Profiles per cluster')
            axes_d[1].set_ylabel('Cluster label')
            axes_d[1].set_title('Cluster sizes')
            axes_d[1].set_yticks(unique_cl)
            axes_d[1].grid(True, alpha=0.2, axis='x')

            fig_d.tight_layout()
            path_d = os.path.join(save_path, f'HC_Dendrogram_{label}.pdf')
            fig_d.savefig(path_d, dpi=150, bbox_inches='tight')
            plt.close(fig_d)
            print(f'  [{label}] Saved dendrogram → {path_d}')

    # ── 8. Corner scatter ─────────────────────────────────────────────────────
    print(f'  [{label}] Plotting corner scatter ...')
    fig_c, axes_c = plt.subplots(D, D, figsize=(3 * D, 3 * D))
    if D == 1:
        axes_c = np.array([[axes_c]])
    elif D > 1:
        axes_c = np.atleast_2d(axes_c)
    fig_c.suptitle(f'Corner plot — {label}', fontsize=12, y=1.01)

    for row in range(D):
        for col in range(D):
            ax = axes_c[row, col]
            if row == col:
                for ci, cl in enumerate(unique_cl):
                    mask = labels == cl
                    ax.hist(data[mask, row], bins=40, alpha=0.55,
                            color=cls_colors[ci % len(cls_colors)],
                            density=True, histtype='stepfilled', edgecolor='none')
                ax.set_xlabel(feature_labels[row], fontsize=8)
                for spine in ['top', 'left', 'right']:
                    ax.spines[spine].set_visible(False)
                ax.set_yticks([])
                ax.tick_params(labelsize=7)
            elif row > col:
                step = max(1, N // n_subsample)
                for ci, cl in enumerate(unique_cl):
                    mask = labels == cl
                    idx  = np.where(mask)[0][::step]
                    ax.scatter(data[idx, col], data[idx, row],
                               color=cls_colors[ci % len(cls_colors)],
                               s=4, alpha=0.3, linewidths=0, rasterized=True)
                if col == 0:
                    ax.set_ylabel(feature_labels[row], fontsize=8)
                if row == D - 1:
                    ax.set_xlabel(feature_labels[col], fontsize=8)
                ax.tick_params(labelsize=6)
                ax.grid(True, alpha=0.15)
            else:
                ax.set_visible(False)

    handles_c = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=cls_colors[ci % len(cls_colors)],
                   markersize=7, label=f'Cluster {cl}')
        for ci, cl in enumerate(unique_cl)
    ]
    fig_c.legend(handles=handles_c, loc='upper right', fontsize=8,
                 framealpha=0.85, title='HC cluster', title_fontsize=8)
    fig_c.tight_layout()
    path_c = os.path.join(save_path, f'HC_Corner_{label}.pdf')
    fig_c.savefig(path_c, dpi=150, bbox_inches='tight')
    plt.close(fig_c)
    print(f'  [{label}] Saved corner plot → {path_c}')

    return labels, cutoff_used, Z


def fourNLLD(x, coeffs):
    """4th-order non-linear limb-darkening law."""
    return (1
            - coeffs[0] * (1 - x ** 0.5)
            - coeffs[1] * (1 - x)
            - coeffs[2] * (1 - x ** 1.5)
            - coeffs[3] * (1 - x ** 2))


def residual_fn(params, x, base_prof):
    """Residual function for lmfit minimisation of NLLD coefficients."""
    return fourNLLD(x, [params[f'c{ic+1}'].value for ic in range(4)]) - base_prof


def rs_to_mu(rs, p):
    """
    Convert r/R_star values to mu = cos(theta), masking to r <= 1-p.
    Returns (mask, mu_values) where mu is clipped to [0, 1] for r > 1.
    """
    mask = rs <= (1.0 - float(p))
    mu   = np.sqrt(np.clip(1.0 - rs[mask] ** 2, 0.0, None))
    return mask, mu


################################
########## Code block ##########
################################

for model in models:

    save_data_path = orig_save_data_path + f'{model}/'
    load_data_path = orig_load_data_path + f'{model}/'

    if not os.path.exists(save_data_path):
        os.makedirs(save_data_path)

    if not os.path.exists(load_data_path):
        raise FileNotFoundError(
            f'Storage directory not found: {load_data_path}\n'
            f'Run Fig2_side_helper.py with intr_prof_mode="build" first.'
        )

    # ── Load pre-built intensity profile grid ─────────────────────────────────
    print(f'Loading gen_dict for model {model} ...')
    with open(load_data_path + 'data.pkl', 'rb') as f:
        gen_dict = pickle.load(f)
    print(f'  Loaded from {load_data_path}data.pkl')

    # ═════════════════════════════════════════════════════════════════════════════
    # RESTRICTED ANALYSIS: target_b_val, target_p_val
    # ═════════════════════════════════════════════════════════════════════════════
    print('\n' + '=' * 80)
    print(f'  RESTRICTED ANALYSIS: b = {target_b_val}, p = {target_p_val}  '
          f'[model: {model}]')
    print('=' * 80)

    # ── 0. Identify the target indices ───────────────────────────────────────
    target_ib = int(np.argmin(np.abs(np.array(bs) - target_b_val)))
    target_ip = int(np.argmin(np.abs(np.array(ps) - target_p_val)))
    target_p  = float(ps[target_ip])   # actual p value used in the grid

    print(f'  Target b = {target_b_val}  →  ib = {target_ib}  '
          f'(actual b = {float(bs[target_ib]):.4f})')
    print(f'  Target p = {target_p_val}  →  ip = {target_ip}  '
          f'(actual p = {target_p:.6f})')

    bp_save_path = os.path.join(save_data_path, 'b0_p0.1_analysis/')
    if not os.path.exists(bp_save_path):
        os.makedirs(bp_save_path)

    # ── 1. Collect every valid intensity profile for this (b, p) pair ────────
    # bp_rs stores r/R_star values (NOT reversed, NOT converted to mu here).
    # The rs_to_mu() helper is called at each use site so that the mask
    # r <= 1 - p is applied consistently everywhere.
    print('  Collecting profiles …')

    T_vals_bp   = np.linspace(Teffs[model][0],       Teffs[model][1],       N_star)
    g_vals_bp   = np.linspace(loggs[model][0],        loggs[model][1],        N_star)
    m_vals_bp   = np.linspace(metallicitys[model][0], metallicitys[model][1], N_star)
    wavs_ref_bp = np.array(gen_dict['stellar_wavelengths'][model][0, 0, 0])

    bp_profiles_list = []
    bp_rs_list       = []   # r/R_star values, shape (N_chords,) per profile
    bp_meta_list     = []   # columns: [i_Teff, j_logg, k_met, iw]

    for i in range(N_star):
        for j in range(N_star):
            for k in range(N_star):
                entry      = np.array(gen_dict['local_intensity_profiles'][model][i, j, k])
                mask_entry = gen_dict['intensity_profiles_mask'][model][i, j, k]

                for ib_excl, ip_excl in excluded_bp_pairs:
                    mask_entry[ib_excl, ip_excl, :] = False

                profs_ijk = entry[target_ib, target_ip]      # (n_wav, N_chords)
                mask_ijk  = mask_entry[target_ib, target_ip] # (n_wav,)

                n_valid_ijk = int(np.sum(mask_ijk))
                if n_valid_ijk == 0:
                    continue

                bp_profiles_list.append(profs_ijk[mask_ijk])

                rps_row = np.array(
                    gen_dict['local_rps'][model][i, j, k, target_ib, target_ip]
                )  # shape (N_chords,) — do NOT reverse here
                bp_rs_list.append(np.tile(rps_row, (n_valid_ijk, 1)))

                wav_idx    = np.where(mask_ijk)[0].astype(np.int32)
                meta_block = np.column_stack([
                    np.full(n_valid_ijk, i, dtype=np.int32),
                    np.full(n_valid_ijk, j, dtype=np.int32),
                    np.full(n_valid_ijk, k, dtype=np.int32),
                    wav_idx,
                ])
                bp_meta_list.append(meta_block)

    bp_profiles = np.vstack(bp_profiles_list).astype(np.float64)
    bp_rs       = np.vstack(bp_rs_list).astype(np.float64)
    bp_meta     = np.vstack(bp_meta_list)
    N_bp        = bp_profiles.shape[0]

    bp_Teff = T_vals_bp[bp_meta[:, 0]]
    bp_logg = g_vals_bp[bp_meta[:, 1]]
    bp_met  = m_vals_bp[bp_meta[:, 2]]
    bp_wav  = wavs_ref_bp[bp_meta[:, 3]] / 1e4   # → µm

    print(f'  Collected {N_bp} valid profiles')
    print(f'    Teff  : [{bp_Teff.min():.0f}, {bp_Teff.max():.0f}] K')
    print(f'    logg  : [{bp_logg.min():.2f}, {bp_logg.max():.2f}]')
    print(f'    [M/H] : [{bp_met.min():.2f}, {bp_met.max():.2f}]')
    print(f'    λ     : [{bp_wav.min():.2f}, {bp_wav.max():.2f}] µm')

    # ── 1b. Optional sub-sampling ────────────────────────────────────────────
    n_subsample_bp = min(n_subsample_profiles, N_bp) if subsample_profiles else N_bp

    if n_subsample_bp < N_bp:
        rng_bp  = np.random.default_rng(subsample_seed)
        idx_bp  = np.sort(rng_bp.choice(N_bp, size=n_subsample_bp, replace=False))
        print(f'  Sub-sampled {N_bp} → {n_subsample_bp} profiles')
    else:
        idx_bp = np.arange(N_bp)

    bp_profiles = bp_profiles[idx_bp]
    bp_rs       = bp_rs[idx_bp]
    bp_meta     = bp_meta[idx_bp]
    bp_Teff     = bp_Teff[idx_bp]
    bp_logg     = bp_logg[idx_bp]
    bp_met      = bp_met[idx_bp]
    bp_wav      = bp_wav[idx_bp]
    N_bp        = len(idx_bp)

    # ── Pre-compute mu arrays for every profile (applying r <= 1-p mask) ─────
    # bp_mus[i] and bp_profs_masked[i] are 1-D arrays of varying length
    # because the mask r <= 1-p can cut a different number of chord points
    # per profile (all identical here since p is fixed, but kept general).
    bp_mus          = []
    bp_profs_masked = []
    for idx in range(N_bp):
        mask_i, mu_i = rs_to_mu(bp_rs[idx], target_p)
        bp_mus.append(mu_i)
        bp_profs_masked.append(bp_profiles[idx][mask_i])

    # ── 2. PCA — fit on the full N_chords vectors (before masking) ───────────
    # PCA is performed on the full chord to preserve consistent dimensionality.
    print(f'  Running PCA ({n_components} components) on {N_bp} profiles …')
    pca_bp    = PCA(n_components=n_components)
    bp_scores = pca_bp.fit_transform(bp_profiles)
    bp_eigen  = pca_bp.components_
    bp_evr    = pca_bp.explained_variance_ratio_

    print(f'    Variance explained per PC : '
          f'{", ".join(f"{v*100:.2f}%" for v in bp_evr)}')
    print(f'    Total variance captured   : {bp_evr.sum()*100:.2f}%')

    # ── 2a. PCA summary figure ───────────────────────────────────────────────
    # Eigen profiles are plotted against mu using the r<=1-p mask on bp_rs[0]
    ncols_pca_bp = 2 + n_components
    fig_pca_bp, ax_pca_bp = plt.subplots(1, ncols_pca_bp, figsize=(4 * ncols_pca_bp, 4))

    ax_pca_bp[0].plot(range(1, n_components + 1), bp_evr, 'o-', color='teal', linewidth=2)
    ax_pca_bp[0].set_title('Scree')
    ax_pca_bp[0].set_xlabel('PC')
    ax_pca_bp[0].set_ylabel('Expl. var. ratio')
    ax_pca_bp[0].grid(True, alpha=0.3)

    ax_pca_bp[1].plot(range(1, n_components + 1), np.cumsum(bp_evr), 'o-', color='teal', linewidth=2)
    ax_pca_bp[1].axhline(0.95, color='g', linestyle='--', label='95 %')
    ax_pca_bp[1].set_title('Cumul. variance')
    ax_pca_bp[1].set_xlabel('PC')
    ax_pca_bp[1].set_ylabel('Cumul. expl. var.')
    ax_pca_bp[1].legend(fontsize=7)
    ax_pca_bp[1].grid(True, alpha=0.3)

    # Use profile 0's mu grid as the reference x-axis for eigen profiles
    mask_ref, mu_ref = rs_to_mu(bp_rs[0], target_p)
    for ic in range(n_components):
        ax = ax_pca_bp[2 + ic]
        ax.plot(mu_ref, bp_eigen[ic][mask_ref], color=colors[ic], linewidth=1.5)
        ax.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax.set_title(f'PC{ic+1}  ({bp_evr[ic]*100:.1f} %)')
        ax.set_xlabel(r'$\mu = \cos(\theta)$')
        ax.set_ylabel('Component value')
        ax.grid(True, alpha=0.3)

    fig_pca_bp.suptitle(
        f'PCA — b = {target_b_val}, p = {target_p_val}  ({model})',
        fontsize=12, y=1.03)
    fig_pca_bp.tight_layout()
    fig_pca_bp.savefig(bp_save_path + f'PCA_summary_b0_p0.1_{model}.png',
                       dpi=150, bbox_inches='tight')
    plt.close(fig_pca_bp)
    print('    Saved PCA summary figure')

    # ── 3. Hierarchical clustering on PCA scores ──────────────────────────────
    print('  Clustering on PCA scores …')

    bp_cl_labels, bp_cl_cutoff, bp_cl_Z = hierarchical_clustering(
        data=bp_scores,
        label=f'b0_p0.1_PCA_{model}',
        save_path=bp_save_path,
        feature_labels=[f'PC{k+1}' for k in range(n_components)],
        clustering_metric='mahalanobis',
        method='single',
        cutoff=7,
    )

    bp_cl_0idx   = bp_cl_labels - 1
    bp_unique_cl = np.unique(bp_cl_0idx)
    n_cl_bp      = len(bp_unique_cl)

    bp_typical_idx  = None
    bp_min_dist     = np.inf
    bp_outlier_idxs = []

    for cl in bp_unique_cl:
        mask_cl  = bp_cl_0idx == cl
        members  = bp_scores[mask_cl]
        centroid = members.mean(axis=0)
        dists    = np.linalg.norm(members - centroid, axis=1)

        closest_global = int(np.where(mask_cl)[0][np.argmin(dists)])
        if dists.min() < bp_min_dist:
            bp_min_dist    = dists.min()
            bp_typical_idx = closest_global

        bp_outlier_idxs.append(int(np.where(mask_cl)[0][np.argmax(dists)]))

    print(f'    Typical profile index : {bp_typical_idx}')
    print(f'    Outlier indices       : {bp_outlier_idxs}')

    # ── 3a. PCA corner scatter coloured by cluster ────────────────────────────
    bp_cluster_cmap   = plt.cm.get_cmap('tab10', n_cl_bp)
    bp_cluster_colors = [bp_cluster_cmap(c) for c in range(n_cl_bp)]

    fig_pca_corner_bp, axes_pcc = plt.subplots(
        n_components, n_components,
        figsize=(3 * n_components, 3 * n_components))
    fig_pca_corner_bp.suptitle(
        f'PCA cluster structure — b = {target_b_val}, p = {target_p_val}  ({model})',
        fontsize=12, y=1.01)

    for row in range(n_components):
        for col in range(n_components):
            ax = axes_pcc[row, col]
            if row == col:
                for ci, cl in enumerate(bp_unique_cl):
                    cmask = bp_cl_0idx == cl
                    ax.hist(bp_scores[cmask, row], bins=30, alpha=0.5,
                            color=bp_cluster_colors[ci], density=True, edgecolor='none')
                ax.set_xlabel(f'PC{row+1}', fontsize=8)
                for sp in ['top', 'left', 'right']:
                    ax.spines[sp].set_visible(False)
                ax.set_yticks([])
            elif row > col:
                for ci, cl in enumerate(bp_unique_cl):
                    cmask = bp_cl_0idx == cl
                    ax.scatter(bp_scores[cmask, col], bp_scores[cmask, row],
                               color=bp_cluster_colors[ci], s=6, alpha=0.35,
                               linewidths=0, rasterized=True)
                ax.scatter(bp_scores[bp_typical_idx, col],
                           bp_scores[bp_typical_idx, row],
                           color='blue', s=80, marker='*',
                           edgecolors='k', linewidths=0.5, zorder=10,
                           label='Typical')
                for oi, oidx in enumerate(bp_outlier_idxs):
                    ax.scatter(bp_scores[oidx, col], bp_scores[oidx, row],
                               color='red', s=60, marker='D',
                               edgecolors='k', linewidths=0.5, zorder=10,
                               label=f'Outlier {oi}' if col == 0 and row == 1 else '')
                if col == 0:
                    ax.set_ylabel(f'PC{row+1}', fontsize=8)
                if row == n_components - 1:
                    ax.set_xlabel(f'PC{col+1}', fontsize=8)
                ax.tick_params(labelsize=6)
                ax.grid(True, alpha=0.2)
            else:
                ax.set_visible(False)

    h_leg = (
        [plt.Line2D([0], [0], marker='o', color='w',
                    markerfacecolor=bp_cluster_colors[ci], markersize=7,
                    label=f'Cluster {cl}')
         for ci, cl in enumerate(bp_unique_cl)]
        + [plt.Line2D([0], [0], marker='*', color='w',
                      markerfacecolor='blue', markersize=9, label='Typical'),
           plt.Line2D([0], [0], marker='D', color='w',
                      markerfacecolor='red', markersize=7, label='Outlier')]
    )
    fig_pca_corner_bp.legend(handles=h_leg, loc='upper right', fontsize=8, framealpha=0.85)
    fig_pca_corner_bp.tight_layout()
    fig_pca_corner_bp.savefig(bp_save_path + f'PCA_Corner_b0_p0.1_{model}.png',
                              dpi=150, bbox_inches='tight')
    plt.close(fig_pca_corner_bp)
    print('    Saved PCA corner scatter')

    # ── 4. Fit every profile with 4th-order NLLD ─────────────────────────────
    # Each fit uses the profile values masked to r <= 1-p, with mu = sqrt(clip(1-r^2))
    print(f'  Fitting {N_bp} profiles with 4th-order NLLD …')

    bp_coeffs = np.zeros((N_bp, 4), dtype=np.float64)

    for idx_fit in tqdm(range(N_bp), desc='  NLLD fits'):
        mu_fit   = bp_mus[idx_fit]           # already masked to r <= 1-p
        prof_fit = bp_profs_masked[idx_fit]  # matching profile values

        if np.all(np.abs(prof_fit) < 1e-10):
            bp_coeffs[idx_fit] = np.nan
            continue

        params_fit = Parameters()
        for ipc in range(4):
            params_fit.add(f'c{ipc+1}', value=np.random.uniform(0, 1))
        try:
            res_fit = minimize(residual_fn, params_fit, args=(mu_fit, prof_fit))
            bp_coeffs[idx_fit] = [res_fit.params[f'c{ic+1}'].value for ic in range(4)]
        except Exception:
            bp_coeffs[idx_fit] = np.nan

    valid_fit   = ~np.any(np.isnan(bp_coeffs), axis=1)
    n_valid_fit = int(valid_fit.sum())
    print(f'    {n_valid_fit} / {N_bp} fits converged')

    # ── 4a. Save coefficients + metadata ─────────────────────────────────────
    bp_save_array = np.column_stack([
        bp_coeffs, bp_Teff, bp_logg, bp_met, bp_wav, bp_cl_0idx,
    ])   # (N_bp, 9)  columns: c1, c2, c3, c4, Teff, logg, [M/H], λ(µm), cluster

    npy_bp_path = bp_save_path + f'coeffs_b0_p0.1_full_{model}.npy'
    np.save(npy_bp_path, bp_save_array)
    print(f'    Saved {bp_save_array.shape} array → {npy_bp_path}')
    print(f'    Columns: c1, c2, c3, c4, Teff, logg, [M/H], λ(µm), cluster')

    # ── 5. Figure: typical + outlier profiles with NLLD fits ─────────────────
    print('  Plotting typical & outlier profiles with NLLD fits …')

    special_bp = (
        [('Typical', bp_typical_idx, 'blue')]
        + [(f'Outlier {oi}', oidx, 'red')
           for oi, oidx in enumerate(bp_outlier_idxs)]
    )

    fig_prof_bp, axes_prof_bp = plt.subplots(
        len(special_bp), 2,
        figsize=(14, 4.5 * len(special_bp)),
        gridspec_kw={'width_ratios': [3, 1]})
    if len(special_bp) == 1:
        axes_prof_bp = axes_prof_bp[np.newaxis, :]

    fig_prof_bp.suptitle(
        f'4th-order NLLD fits — b = {target_b_val}, p = {target_p_val}  ({model})',
        fontsize=13)

    for isp, (sp_label, sp_idx, sp_color) in enumerate(special_bp):
        mu_sp   = bp_mus[sp_idx]            # masked mu values
        prof_sp = bp_profs_masked[sp_idx]   # matching masked profile
        c_sp    = bp_coeffs[sp_idx]
        fit_sp  = fourNLLD(mu_sp, c_sp)

        ax_l = axes_prof_bp[isp, 0]
        ax_r = axes_prof_bp[isp, 1]

        # Background: draw a subsample of all profiles (also masked)
        step_bg = max(1, N_bp // 300)
        for ibg in range(0, N_bp, step_bg):
            ax_l.plot(bp_mus[ibg], bp_profs_masked[ibg],
                      alpha=0.1, color='gray', linewidth=0.3)

        ax_l.plot(mu_sp, prof_sp, 'o', color=sp_color, markersize=3,
                  alpha=0.7, label=f'{sp_label} profile')
        ax_l.plot(mu_sp, fit_sp, '--', color='black', linewidth=2,
                  label='4th-order NLLD')
        ax_l.set_ylabel('Norm. Intensity')
        ax_l.set_title(
            f'{sp_label}   '
            f'c=[{c_sp[0]:.4f}, {c_sp[1]:.4f}, {c_sp[2]:.4f}, {c_sp[3]:.4f}]',
            fontsize=9)
        ax_l.legend(fontsize=7)
        ax_l.grid(True, alpha=0.3)
        if isp < len(special_bp) - 1:
            ax_l.tick_params(labelbottom=False)
        else:
            ax_l.set_xlabel(r'$\mu = \cos(\theta)$')

        safe_sp = np.where(np.abs(prof_sp) < 1e-10, np.nan, prof_sp)
        ax_r.plot(mu_sp, 100 * (prof_sp - fit_sp) / safe_sp,
                  '--', color=sp_color, linewidth=1.5)
        ax_r.axhline(0, color='k', linestyle='--', alpha=0.5)
        ax_r.set_ylabel('Rel. diff. (%)')
        ax_r.grid(True, alpha=0.3)
        if isp < len(special_bp) - 1:
            ax_r.tick_params(labelbottom=False)
        else:
            ax_r.set_xlabel(r'$\mu = \cos(\theta)$')

        redchi_sp = np.nan
        try:
            params_sp = Parameters()
            for ipc in range(4):
                params_sp.add(f'c{ipc+1}', value=c_sp[ipc])
            res_sp    = minimize(residual_fn, params_sp, args=(mu_sp, prof_sp))
            redchi_sp = res_sp.redchi
        except Exception:
            pass

        print(f'    {sp_label:12s}  '
              f'c=[{c_sp[0]:+.4f}, {c_sp[1]:+.4f}, {c_sp[2]:+.4f}, {c_sp[3]:+.4f}]  '
              f'red-χ²={redchi_sp:.4e}  '
              f'Teff={bp_Teff[sp_idx]:.0f} K  '
              f'logg={bp_logg[sp_idx]:.2f}  '
              f'[M/H]={bp_met[sp_idx]:.2f}  '
              f'λ={bp_wav[sp_idx]:.2f} µm')

    fig_prof_bp.tight_layout()
    fig_prof_bp.savefig(bp_save_path + f'Profiles_NLLD_b0_p0.1_{model}.png',
                        dpi=150, bbox_inches='tight')
    plt.close(fig_prof_bp)
    print('    Saved typical & outlier profile figure')

    # ── 6. Per-cluster representative profiles with NLLD fits ────────────────
    print('  Plotting per-cluster representative profiles …')

    fig_cl_bp, axes_cl_bp = plt.subplots(
        n_cl_bp, 2,
        figsize=(14, 4.5 * n_cl_bp),
        gridspec_kw={'width_ratios': [3, 1]})
    if n_cl_bp == 1:
        axes_cl_bp = axes_cl_bp[np.newaxis, :]

    fig_cl_bp.suptitle(
        f'Cluster representatives with NLLD fits — b = {target_b_val}, '
        f'p = {target_p_val}  ({model})',
        fontsize=13)

    for ci, cl in enumerate(bp_unique_cl):
        mask_cl  = bp_cl_0idx == cl
        members  = bp_scores[mask_cl]
        centroid = members.mean(axis=0)
        dists_cl = np.linalg.norm(members - centroid, axis=1)
        rep_idx  = int(np.where(mask_cl)[0][np.argmin(dists_cl)])

        mu_rep   = bp_mus[rep_idx]
        prof_rep = bp_profs_masked[rep_idx]
        c_rep    = bp_coeffs[rep_idx]
        fit_rep  = fourNLLD(mu_rep, c_rep)

        ax_l = axes_cl_bp[ci, 0]
        ax_r = axes_cl_bp[ci, 1]

        cl_indices = np.where(mask_cl)[0]
        step_cl    = max(1, len(cl_indices) // 200)
        for ii in cl_indices[::step_cl]:
            ax_l.plot(bp_mus[ii], bp_profs_masked[ii],
                      alpha=0.15, color=bp_cluster_colors[ci], linewidth=0.3)

        ax_l.plot(mu_rep, prof_rep, 'o',
                  color=bp_cluster_colors[ci], markersize=3, alpha=0.8,
                  label=f'Cluster {cl} representative')
        ax_l.plot(mu_rep, fit_rep, '--', color='black', linewidth=2,
                  label='4th-order NLLD')
        ax_l.set_ylabel('Norm. Intensity')
        ax_l.set_title(
            f'Cluster {cl}  (n={mask_cl.sum()})   '
            f'c=[{c_rep[0]:.4f}, {c_rep[1]:.4f}, {c_rep[2]:.4f}, {c_rep[3]:.4f}]',
            fontsize=9)
        ax_l.legend(fontsize=7)
        ax_l.grid(True, alpha=0.3)
        if ci < n_cl_bp - 1:
            ax_l.tick_params(labelbottom=False)
        else:
            ax_l.set_xlabel(r'$\mu = \cos(\theta)$')

        safe_rep = np.where(np.abs(prof_rep) < 1e-10, np.nan, prof_rep)
        ax_r.plot(mu_rep, 100 * (prof_rep - fit_rep) / safe_rep,
                  '--', color=bp_cluster_colors[ci], linewidth=1.5)
        ax_r.axhline(0, color='k', linestyle='--', alpha=0.5)
        ax_r.set_ylabel('Rel. diff. (%)')
        ax_r.grid(True, alpha=0.3)
        if ci < n_cl_bp - 1:
            ax_r.tick_params(labelbottom=False)
        else:
            ax_r.set_xlabel(r'$\mu = \cos(\theta)$')

        print(f'    Cluster {cl:2d}  n={mask_cl.sum():5d}  '
              f'c=[{c_rep[0]:+.4f}, {c_rep[1]:+.4f}, {c_rep[2]:+.4f}, {c_rep[3]:+.4f}]  '
              f'Teff={bp_Teff[rep_idx]:.0f}  logg={bp_logg[rep_idx]:.2f}  '
              f'[M/H]={bp_met[rep_idx]:.2f}  λ={bp_wav[rep_idx]:.2f} µm')

    fig_cl_bp.tight_layout()
    fig_cl_bp.savefig(bp_save_path + f'ClusterReps_NLLD_b0_p0.1_{model}.png',
                      dpi=150, bbox_inches='tight')
    plt.close(fig_cl_bp)
    print('    Saved per-cluster representative figure')

    # ── 7. Corner plots of NLLD coefficients ─────────────────────────────────
    bp_coeffs_valid = bp_coeffs[valid_fit]
    bp_cl_valid     = bp_cl_0idx[valid_fit]
    bp_wav_valid    = bp_wav[valid_fit]
    bp_Teff_valid   = bp_Teff[valid_fit]
    bp_logg_valid   = bp_logg[valid_fit]
    bp_met_valid    = bp_met[valid_fit]

    labels_bp_corner = [r'$c_1$', r'$c_2$', r'$c_3$', r'$c_4$']
    ranges_bp_corner = [(np.percentile(bp_coeffs_valid[:, ic], 1),
                         np.percentile(bp_coeffs_valid[:, ic], 99))
                        for ic in range(4)]
    ndim_bp = 4

    # Helper: subsample index for scatter
    max_scatter_bp = 40_000
    rng_cc     = np.random.default_rng(42)
    idx_cc_sub = (np.sort(rng_cc.choice(n_valid_fit, size=max_scatter_bp, replace=False))
                  if n_valid_fit > max_scatter_bp else np.arange(n_valid_fit))

    # ── 7a. Coloured by PCA cluster ──────────────────────────────────────────
    print('  Corner plot of coefficients coloured by PCA cluster …')

    fig_cc_bp = corner.corner(
        bp_coeffs_valid, labels=labels_bp_corner, range=ranges_bp_corner,
        bins=50, smooth1d=1.0,
        plot_datapoints=False, plot_density=False, fill_contours=False, no_fill_contours=True,
        levels=(0.5, 0.68, 0.95, 0.99),
        hist_kwargs={'color': 'gray', 'linewidth': 1.2, 'alpha': 0.5},
        label_kwargs={'fontsize': 13}, show_titles=False,
        contour_kwargs={'colors': 'none'},
    )
    axes_cc_bp = np.array(fig_cc_bp.axes).reshape(ndim_bp, ndim_bp)

    for row in range(ndim_bp):
        for col in range(row):
            ax = axes_cc_bp[row, col]
            for ci, cl in enumerate(bp_unique_cl):
                mask_cl_v = bp_cl_valid[idx_cc_sub] == cl
                if mask_cl_v.sum() == 0:
                    continue
                ax.scatter(bp_coeffs_valid[idx_cc_sub[mask_cl_v], col],
                           bp_coeffs_valid[idx_cc_sub[mask_cl_v], row],
                           color=bp_cluster_colors[ci], s=2, alpha=0.35,
                           linewidths=0, rasterized=True)

    for d in range(ndim_bp):
        ax = axes_cc_bp[d, d]
        ax.clear()
        for ci, cl in enumerate(bp_unique_cl):
            mask_cl_v = bp_cl_valid == cl
            if mask_cl_v.sum() == 0:
                continue
            ax.hist(bp_coeffs_valid[mask_cl_v, d], bins=50, range=ranges_bp_corner[d],
                    alpha=0.55, color=bp_cluster_colors[ci], density=True,
                    histtype='stepfilled', edgecolor='none')
        ax.set_xlim(ranges_bp_corner[d])
        ax.set_yticks([])
        for sp in ['top', 'left', 'right']:
            ax.spines[sp].set_visible(False)
        if d == ndim_bp - 1:
            ax.set_xlabel(labels_bp_corner[d], fontsize=13)

    for row in range(ndim_bp):
        for col in range(row):
            ax = axes_cc_bp[row, col]
            if col == 0:
                ax.set_ylabel(labels_bp_corner[row], fontsize=13)
            if row == ndim_bp - 1:
                ax.set_xlabel(labels_bp_corner[col], fontsize=13)
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.15)

    h_cc = [plt.Line2D([0], [0], marker='o', color='w',
                        markerfacecolor=bp_cluster_colors[ci], markersize=7,
                        label=f'Cluster {cl}')
            for ci, cl in enumerate(bp_unique_cl)]
    fig_cc_bp.legend(handles=h_cc, loc='upper right', fontsize=8,
                     framealpha=0.85, title='PCA cluster')
    fig_cc_bp.text(0.72, 0.55,
                   f'$b = {target_b_val:.1f}$,  $p = {target_p_val}$\n'
                   f'$N = {n_valid_fit}$ converged fits\n{n_cl_bp} clusters',
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.7))
    fig_cc_bp.suptitle(
        f'NLLD coefficients by PCA cluster — b={target_b_val}, '
        f'p={target_p_val}  ({model})',
        fontsize=13, y=1.02)
    fig_cc_bp.savefig(bp_save_path + f'Corner_coeffs_byCluster_b0_p0.1_{model}.png',
                      dpi=150, bbox_inches='tight')
    plt.close(fig_cc_bp)
    print('    Saved cluster-coloured coefficient corner plot')

    # ── 7b. Coloured by wavelength ────────────────────────────────────────────
    print('  Corner plot of coefficients coloured by wavelength …')

    wav_cmap_bp = plt.cm.turbo
    wav_norm_bp = mcolors.Normalize(vmin=bp_wav_valid.min(), vmax=bp_wav_valid.max())

    fig_cw_bp = corner.corner(
        bp_coeffs_valid, labels=labels_bp_corner, range=ranges_bp_corner,
        bins=50, smooth1d=1.0,
        plot_datapoints=False, plot_density=False, fill_contours=False, no_fill_contours=True,
        levels=(0.5, 0.68, 0.95, 0.99),
        hist_kwargs={'color': 'gray', 'linewidth': 1.2, 'alpha': 0.5},
        label_kwargs={'fontsize': 13}, show_titles=False,
        contour_kwargs={'colors': 'none'},
    )
    axes_cw_bp = np.array(fig_cw_bp.axes).reshape(ndim_bp, ndim_bp)

    idx_wav_sub = idx_cc_sub[np.argsort(bp_wav_valid[idx_cc_sub])]

    for row in range(ndim_bp):
        for col in range(row):
            axes_cw_bp[row, col].scatter(
                bp_coeffs_valid[idx_wav_sub, col], bp_coeffs_valid[idx_wav_sub, row],
                c=bp_wav_valid[idx_wav_sub], cmap=wav_cmap_bp, norm=wav_norm_bp,
                s=2, alpha=0.35, linewidths=0, rasterized=True)

    n_wav_hist_bp  = 10
    wav_edges_bp   = np.linspace(bp_wav_valid.min(), bp_wav_valid.max(), n_wav_hist_bp + 1)
    wav_centres_bp = 0.5 * (wav_edges_bp[:-1] + wav_edges_bp[1:])

    for d in range(ndim_bp):
        ax = axes_cw_bp[d, d]
        ax.clear()
        for ibin, centre in enumerate(wav_centres_bp):
            mask_bin = ((bp_wav_valid >= wav_edges_bp[ibin]) &
                        (bp_wav_valid < wav_edges_bp[ibin + 1]))
            if ibin == n_wav_hist_bp - 1:
                mask_bin |= (bp_wav_valid == wav_edges_bp[ibin + 1])
            if mask_bin.sum() == 0:
                continue
            ax.hist(bp_coeffs_valid[mask_bin, d], bins=50, range=ranges_bp_corner[d],
                    alpha=0.55, color=wav_cmap_bp(wav_norm_bp(centre)),
                    density=True, histtype='stepfilled', edgecolor='none')
        ax.set_xlim(ranges_bp_corner[d])
        ax.set_yticks([])
        for sp in ['top', 'left', 'right']:
            ax.spines[sp].set_visible(False)
        if d == ndim_bp - 1:
            ax.set_xlabel(labels_bp_corner[d], fontsize=13)

    for row in range(ndim_bp):
        for col in range(row):
            ax = axes_cw_bp[row, col]
            if col == 0:
                ax.set_ylabel(labels_bp_corner[row], fontsize=13)
            if row == ndim_bp - 1:
                ax.set_xlabel(labels_bp_corner[col], fontsize=13)
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.15)

    cbar_ax_cw = fig_cw_bp.add_axes([0.72, 0.72, 0.025, 0.20])
    sm_cw      = cm.ScalarMappable(cmap=wav_cmap_bp, norm=wav_norm_bp)
    sm_cw.set_array([])
    cbar_cw    = fig_cw_bp.colorbar(sm_cw, cax=cbar_ax_cw)
    cbar_cw.set_label(r'Wavelength ($\mu$m)', fontsize=13)
    cbar_cw.ax.tick_params(labelsize=10)
    wav_tick_bp  = 0.5
    wav_ticks_cw = np.arange(
        np.ceil(bp_wav_valid.min() / wav_tick_bp) * wav_tick_bp,
        bp_wav_valid.max() + wav_tick_bp / 2,
        wav_tick_bp)
    cbar_cw.set_ticks(wav_ticks_cw)
    cbar_cw.set_ticklabels([f'{t:.1f}' for t in wav_ticks_cw])

    fig_cw_bp.suptitle(
        f'NLLD coefficients by wavelength — b={target_b_val}, '
        f'p={target_p_val}  ({model})',
        fontsize=13, y=1.02)
    fig_cw_bp.savefig(bp_save_path + f'Corner_coeffs_byWav_b0_p0.1_{model}.png',
                      dpi=150, bbox_inches='tight')
    plt.close(fig_cw_bp)
    print('    Saved wavelength-coloured coefficient corner plot')

    # ── 7c. Coloured by Teff, logg, metallicity ───────────────────────────────
    bp_meta_sources = [
        (r'$T_{\rm eff}$ (K)', 'Teff', bp_Teff_valid, plt.cm.inferno),
        (r'$\log\,g$',         'logg', bp_logg_valid, plt.cm.cividis),
        (r'[M/H]',             'MH',   bp_met_valid,  plt.cm.coolwarm),
    ]

    for meta_label, meta_key, meta_vals, meta_cmap in bp_meta_sources:
        print(f'  Corner plot of coefficients coloured by {meta_key} …')

        meta_norm = mcolors.Normalize(vmin=meta_vals.min(), vmax=meta_vals.max())

        fig_cm_bp = corner.corner(
            bp_coeffs_valid, labels=labels_bp_corner, range=ranges_bp_corner,
            bins=50, smooth1d=1.0,
            plot_datapoints=False, plot_density=False, fill_contours=False, no_fill_contours=True,
            levels=(0.5, 0.68, 0.95, 0.99),
            hist_kwargs={'color': 'gray', 'linewidth': 1.2, 'alpha': 0.5},
            label_kwargs={'fontsize': 13}, show_titles=False,
            contour_kwargs={'colors': 'none'},
        )
        axes_cm_bp = np.array(fig_cm_bp.axes).reshape(ndim_bp, ndim_bp)

        idx_meta_sub = idx_cc_sub[np.argsort(meta_vals[idx_cc_sub])]

        for row in range(ndim_bp):
            for col in range(row):
                axes_cm_bp[row, col].scatter(
                    bp_coeffs_valid[idx_meta_sub, col],
                    bp_coeffs_valid[idx_meta_sub, row],
                    c=meta_vals[idx_meta_sub], cmap=meta_cmap, norm=meta_norm,
                    s=2, alpha=0.35, linewidths=0, rasterized=True)

        unique_meta = np.unique(meta_vals)
        for d in range(ndim_bp):
            ax = axes_cm_bp[d, d]
            ax.clear()
            if len(unique_meta) <= 20:
                for uv in unique_meta:
                    mask_uv = meta_vals == uv
                    if mask_uv.sum() == 0:
                        continue
                    ax.hist(bp_coeffs_valid[mask_uv, d], bins=50, range=ranges_bp_corner[d],
                            alpha=0.55, color=meta_cmap(meta_norm(uv)),
                            density=True, histtype='stepfilled', edgecolor='none')
            else:
                n_bins_meta = 8
                meta_edges  = np.linspace(meta_vals.min(), meta_vals.max(), n_bins_meta + 1)
                meta_ctrs   = 0.5 * (meta_edges[:-1] + meta_edges[1:])
                for ib_m, ctr in enumerate(meta_ctrs):
                    mask_bm = ((meta_vals >= meta_edges[ib_m]) &
                               (meta_vals < meta_edges[ib_m + 1]))
                    if ib_m == len(meta_ctrs) - 1:
                        mask_bm |= (meta_vals == meta_edges[ib_m + 1])
                    if mask_bm.sum() == 0:
                        continue
                    ax.hist(bp_coeffs_valid[mask_bm, d], bins=50, range=ranges_bp_corner[d],
                            alpha=0.55, color=meta_cmap(meta_norm(ctr)),
                            density=True, histtype='stepfilled', edgecolor='none')
            ax.set_xlim(ranges_bp_corner[d])
            ax.set_yticks([])
            for sp in ['top', 'left', 'right']:
                ax.spines[sp].set_visible(False)
            if d == ndim_bp - 1:
                ax.set_xlabel(labels_bp_corner[d], fontsize=13)

        for row in range(ndim_bp):
            for col in range(row):
                ax = axes_cm_bp[row, col]
                if col == 0:
                    ax.set_ylabel(labels_bp_corner[row], fontsize=13)
                if row == ndim_bp - 1:
                    ax.set_xlabel(labels_bp_corner[col], fontsize=13)
                ax.tick_params(labelsize=8)
                ax.grid(True, alpha=0.15)

        cbar_ax_m = fig_cm_bp.add_axes([0.72, 0.72, 0.025, 0.20])
        sm_m      = cm.ScalarMappable(cmap=meta_cmap, norm=meta_norm)
        sm_m.set_array([])
        cbar_m    = fig_cm_bp.colorbar(sm_m, cax=cbar_ax_m)
        cbar_m.set_label(meta_label, fontsize=13)
        cbar_m.ax.tick_params(labelsize=10)
        if len(unique_meta) <= 15:
            cbar_m.set_ticks(unique_meta)
            cbar_m.set_ticklabels(
                [f'{v:.0f}' for v in unique_meta] if meta_key == 'Teff'
                else [f'{v:.2f}' for v in unique_meta])

        fig_cm_bp.suptitle(
            f'NLLD coefficients by {meta_label} — b={target_b_val}, '
            f'p={target_p_val}  ({model})',
            fontsize=13, y=1.02)
        fig_cm_bp.savefig(
            bp_save_path + f'Corner_coeffs_by{meta_key}_b0_p0.1_{model}.png',
            dpi=150, bbox_inches='tight')
        plt.close(fig_cm_bp)
        print(f'    Saved {meta_key}-coloured coefficient corner plot')

    # ── 8. Reconstruction quality for typical profile ─────────────────────────
    print('  Reconstruction quality for typical profile …')

    n_comp_check = [1, max(2, n_components // 2), n_components]
    fig_recon_bp, axes_recon_bp = plt.subplots(
        len(n_comp_check), 2, figsize=(12, 4 * len(n_comp_check)),
        gridspec_kw={'width_ratios': [3, 1]})

    fig_recon_bp.suptitle(
        f'PCA reconstruction — typical profile, b={target_b_val}, '
        f'p={target_p_val}  ({model})',
        fontsize=13)

    # Reconstruction operates on the full N_chords vector; masking applied for display
    original_bp_full = bp_profiles[bp_typical_idx]
    mask_typical, mu_typical    = rs_to_mu(bp_rs[bp_typical_idx], target_p)   # for x-axis

    for ri, nc in enumerate(n_comp_check):
        pca_temp_bp = PCA(n_components=nc)
        pca_temp_bp.fit(bp_profiles)
        recon_bp_full = pca_temp_bp.inverse_transform(
            pca_temp_bp.transform(original_bp_full.reshape(1, -1)))[0]

        original_bp = original_bp_full[mask_typical]
        recon_bp    = recon_bp_full[mask_typical]
        resid_bp    = original_bp - recon_bp
        rmse_bp     = np.sqrt(np.mean(resid_bp ** 2))

        ax_t = axes_recon_bp[ri, 0]
        ax_b = axes_recon_bp[ri, 1]

        ax_t.plot(mu_typical, original_bp,  color='teal', linewidth=2, label='Original')
        ax_t.plot(mu_typical, recon_bp, color='teal', linewidth=1.2,
                  linestyle='--', label='Reconstructed')
        ax_t.set_title(f'{nc} PC   RMSE = {rmse_bp:.6f}', fontsize=9)
        ax_t.set_ylabel('Norm. Intensity')
        ax_t.legend(fontsize=7)
        ax_t.grid(True, alpha=0.3)

        safe_bp = np.where(np.abs(original_bp) < 1e-10, np.nan, original_bp)
        ax_b.plot(mu_typical, 100 * resid_bp / safe_bp, color='teal', linewidth=1.0)
        ax_b.axhline(0, color='k', linestyle='--', alpha=0.5)
        ax_b.set_ylabel('Rel. diff. (%)')
        ax_b.grid(True, alpha=0.3)
        if ri == len(n_comp_check) - 1:
            ax_t.set_xlabel(r'$\mu = \cos(\theta)$')
            ax_b.set_xlabel(r'$\mu = \cos(\theta)$')

    fig_recon_bp.tight_layout()
    fig_recon_bp.savefig(bp_save_path + f'ReconQuality_b0_p0.1_{model}.png',
                         dpi=150, bbox_inches='tight')
    plt.close(fig_recon_bp)
    print('    Saved reconstruction quality figure')

    # ── 9. Final summary ──────────────────────────────────────────────────────
    print(f'\n  ═══ b={target_b_val}, p={target_p_val} analysis complete for {model} ═══')
    print(f'  Profiles collected      : {N_bp}')
    print(f'  PCA components          : {n_components}')
    print(f'  Variance captured       : {bp_evr.sum()*100:.2f} %')
    print(f'  Clusters found          : {n_cl_bp}')
    print(f'  Converged NLLD fits     : {n_valid_fit} / {N_bp}')
    print(f'  All outputs saved to    : {bp_save_path}')

    print(f'\n  ── Coefficient summary (converged fits) ──')
    for ic in range(4):
        vals = bp_coeffs_valid[:, ic]
        print(f'    c{ic+1}:  mean={vals.mean():+.4f}  '
              f'median={np.median(vals):+.4f}  '
              f'std={vals.std():.4f}  '
              f'[{vals.min():+.4f}, {vals.max():+.4f}]')

    print(f'\n  ── Per-cluster coefficient means ──')
    for ci, cl in enumerate(bp_unique_cl):
        mask_cl_v = bp_cl_valid == cl
        if mask_cl_v.sum() == 0:
            continue
        means_cl = bp_coeffs_valid[mask_cl_v].mean(axis=0)
        stds_cl  = bp_coeffs_valid[mask_cl_v].std(axis=0)
        print(f'    Cluster {cl} (n={mask_cl_v.sum():5d}):  '
              f'c=[{means_cl[0]:+.4f}±{stds_cl[0]:.4f}, '
              f'{means_cl[1]:+.4f}±{stds_cl[1]:.4f}, '
              f'{means_cl[2]:+.4f}±{stds_cl[2]:.4f}, '
              f'{means_cl[3]:+.4f}±{stds_cl[3]:.4f}]')

    # ── 10. Summary pickle ────────────────────────────────────────────────────
    bp_summary = {
        'model'             : model,
        'target_b'          : target_b_val,
        'target_p'          : target_p_val,
        'actual_p'          : target_p,
        'ib'                : target_ib,
        'ip'                : target_ip,
        'N_profiles'        : N_bp,
        'n_components'      : n_components,
        'explained_variance': bp_evr,
        'pca_object'        : pca_bp,
        'pca_scores'        : bp_scores,
        'pca_eigen'         : bp_eigen,
        'profiles'          : bp_profiles,        # full N_chords vectors
        'rs'                : bp_rs,              # r/R_star values (N_bp, N_chords)
        'mus'               : bp_mus,             # list of masked mu arrays
        'profs_masked'      : bp_profs_masked,    # list of masked profile arrays
        'coeffs'            : bp_coeffs,
        'valid_fit_mask'    : valid_fit,
        'cluster_labels'    : bp_cl_0idx,
        'n_clusters'        : n_cl_bp,
        'cluster_cutoff'    : bp_cl_cutoff,
        'typical_idx'       : bp_typical_idx,
        'outlier_idxs'      : bp_outlier_idxs,
        'meta_indices'      : bp_meta,
        'Teff'              : bp_Teff,
        'logg'              : bp_logg,
        'metallicity'       : bp_met,
        'wavelength_um'     : bp_wav,
        'stellar_params'    : {'T_vals': T_vals_bp, 'g_vals': g_vals_bp, 'm_vals': m_vals_bp},
        'wavs_ref_angstrom' : wavs_ref_bp,
    }

    pkl_path = bp_save_path + f'summary_b0_p0.1_{model}.pkl'
    with open(pkl_path, 'wb') as f:
        pickle.dump(bp_summary, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'\n  Saved summary pickle → {pkl_path}')

    # ── 11. Plain-text coefficient table ──────────────────────────────────────
    txt_path = bp_save_path + f'special_coeffs_b0_p0.1_{model}.txt'
    with open(txt_path, 'w') as ftxt:
        ftxt.write(f'# 4th-order NLLD coefficients for b={target_b_val}, '
                   f'p={target_p_val}, model={model}\n')
        ftxt.write('# Format: label  c1  c2  c3  c4  Teff  logg  [M/H]  wavelength_um  cluster\n#\n')

        c_typ = bp_coeffs[bp_typical_idx]
        ftxt.write(f'typical  {c_typ[0]:+.6f}  {c_typ[1]:+.6f}  {c_typ[2]:+.6f}  {c_typ[3]:+.6f}  '
                   f'{bp_Teff[bp_typical_idx]:.0f}  {bp_logg[bp_typical_idx]:.2f}  '
                   f'{bp_met[bp_typical_idx]:.2f}  {bp_wav[bp_typical_idx]:.4f}  '
                   f'{bp_cl_0idx[bp_typical_idx]}\n')

        for oi, oidx in enumerate(bp_outlier_idxs):
            c_out = bp_coeffs[oidx]
            ftxt.write(f'outlier{oi}  {c_out[0]:+.6f}  {c_out[1]:+.6f}  {c_out[2]:+.6f}  {c_out[3]:+.6f}  '
                       f'{bp_Teff[oidx]:.0f}  {bp_logg[oidx]:.2f}  {bp_met[oidx]:.2f}  '
                       f'{bp_wav[oidx]:.4f}  {bp_cl_0idx[oidx]}\n')

        ftxt.write('#\n# Per-cluster representatives (closest to centroid)\n')
        for ci, cl in enumerate(bp_unique_cl):
            mask_cl  = bp_cl_0idx == cl
            members  = bp_scores[mask_cl]
            centroid = members.mean(axis=0)
            dists_cl = np.linalg.norm(members - centroid, axis=1)
            rep_idx  = int(np.where(mask_cl)[0][np.argmin(dists_cl)])
            c_rep    = bp_coeffs[rep_idx]
            ftxt.write(f'cluster{cl}_rep  {c_rep[0]:+.6f}  {c_rep[1]:+.6f}  '
                       f'{c_rep[2]:+.6f}  {c_rep[3]:+.6f}  '
                       f'{bp_Teff[rep_idx]:.0f}  {bp_logg[rep_idx]:.2f}  '
                       f'{bp_met[rep_idx]:.2f}  {bp_wav[rep_idx]:.4f}  {cl}\n')

        ftxt.write('#\n# Per-cluster mean coefficients (± std)\n')
        for ci, cl in enumerate(bp_unique_cl):
            mask_cl_v = bp_cl_valid == cl
            if mask_cl_v.sum() == 0:
                continue
            means = bp_coeffs_valid[mask_cl_v].mean(axis=0)
            stds  = bp_coeffs_valid[mask_cl_v].std(axis=0)
            ftxt.write(f'cluster{cl}_mean  '
                       f'{means[0]:+.6f}±{stds[0]:.6f}  {means[1]:+.6f}±{stds[1]:.6f}  '
                       f'{means[2]:+.6f}±{stds[2]:.6f}  {means[3]:+.6f}±{stds[3]:.6f}  '
                       f'n={mask_cl_v.sum()}\n')

    print(f'  Saved special coefficients → {txt_path}')

    # ── Clean up ──────────────────────────────────────────────────────────────
    del (bp_profiles_list, bp_rs_list, bp_meta_list,
         bp_profiles, bp_rs, bp_meta, bp_scores, bp_coeffs,
         bp_coeffs_valid, bp_cl_valid, bp_wav_valid,
         bp_Teff_valid, bp_logg_valid, bp_met_valid,
         bp_mus, bp_profs_masked)
    gc.collect()

    print('\n' + '=' * 80)
    print(f'  RESTRICTED ANALYSIS COMPLETE — b={target_b_val}, '
          f'p={target_p_val}, model={model}')
    print('=' * 80 + '\n')