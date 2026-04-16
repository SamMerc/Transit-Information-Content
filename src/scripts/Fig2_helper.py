#############################
########## Purpose ##########
#############################

# Figures 2, 3, and 4 require a 4-th order non-linear limb-darkening law for the injection / simulation of the LC.
# Given that we are working with a made up fiducial system, we need to identify the limb-darkening values to use for this.
# In order to do this, we explore all available intensity profiles for a given grid of stellar models, and perform a PCA
# analysis to identify both the median/mode and an outlier intensity profile which can be used in our analyses.
# We perform such decomposition on each individual grid of stellar models, and in doing so this allows us to highlight
# the choice of 1. stellar model and 2. limb-darkening prescription on the transit depth amplification factor and bias.
#
# This version works exclusively with global (disc-integrated) stellar intensity profiles — no transit
# chord / impact-parameter / planet-size dependence. For that see the Paper2 code.


######################################
########## Import libraries ##########
######################################

import numpy as np
import matplotlib
import os
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import exotic_ld as el
import pickle
from sklearn.decomposition import PCA
from lmfit import minimize, Parameters
import gc
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from scipy.interpolate import interp1d
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

LD_data_path        = '/Volumes/Ajax/Work/PhD/Research/Transit-Information-Content/LD_simulation'
orig_save_data_path = '/Volumes/Ajax/Work/PhD/Research/Transit-Information-Content/Fig2_helper_Storage/'

models = ['mps1']  # ['phoenix','kurucz', 'stagger', 'mps1', 'mps2']

# The range of Teff values explorable for each model.
Teffs = {
    'phoenix': [2300, 15000],
    'kurucz':  [3500, 6500],
    'stagger': [4000, 7000],
    'mps2':    [3500, 9000],
    'mps1':    [3500, 9000],
}
# The range of logg values explorable for each model.
loggs = {
    'phoenix': [0.0, 6.0],
    'kurucz':  [4.0, 5.0],
    'stagger': [1.5, 5.0],
    'mps1':    [3.0, 5.0],
    'mps2':    [3.0, 5.0],
}
# The range of metallicity values explorable for each model.
metallicitys = {
    'phoenix': [-1.5,  1.0],
    'kurucz':  [-5.0,  1.0],
    'stagger': [-3.0,  0.0],
    'mps1':    [-5.0,  1.5],
    'mps2':    [-5.0,  1.5],
}
# The resolution of mu values for each model.
mu_resolution = {
    'phoenix': 78,
    'kurucz':  17,
    'stagger': 10,
    'mps1':    24,
    'mps2':    24,
}
# The total number of lambda values for each model.
lambda_resolution = {
    'phoenix': 54500,
    'kurucz':  1221,
    'stagger': 105767,
    'mps1':    1221,
    'mps2':    1221,
}

# Number of points in the grid of stellar parameters (Teff, logg, metallicity).
N_star = 10

# Number of mu values to interpolate to — set to EJ16 value.
n_mu_fine = 100

# Number of principal components to use in the PCA.
n_components = 4
cmap   = plt.cm.coolwarm
colors = cmap(np.linspace(0, 1, n_components))

wav_region = [6000, 53000]  # 0.6 – 5.3 micron

intr_prof_mode = 'build'  # 'build' or 'load'
PCA_mode       = 'build'  # 'build' or 'load'
All_Corner     = 'build'  # 'build' or 'load'

# ── Profile subsampling ───────────────────────────────────────────────────────
# If True, randomly draw n_subsample_profiles from the valid profiles before
# running PCA, clustering, fitting, etc.  Set to False to use all valid profiles.
subsample_profiles   = True
n_subsample_profiles = 50000
subsample_seed       = 42  # for reproducibility

# Whether to plot the dendrogram of the hierarchical clustering step.
plot_dendogram = False


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
    plot_corner       = False,
):
    """
    Hierarchical clustering with scipy pairwise distance computation.

    Parameters
    ----------
    data             : np.ndarray (N, D)
    label            : str
    save_path        : str
    feature_labels   : list of str, optional
    cutoff           : float or None
    method           : str   linkage method  ('single', 'complete', 'average',
                             'weighted', 'centroid', 'median', 'ward')
    max_display      : int   max dendrogram leaves shown
    n_subsample      : int   scatter plot cap per cluster
    external_labels  : np.ndarray or None   pre-computed labels (skips clustering)
    clustering_metric: str   distance metric for cdist

    Returns
    -------
    labels       : np.ndarray (N,)   1-indexed cluster labels
    cutoff_used  : float
    Z            : np.ndarray (N-1, 3) or None
    """
    N, D = data.shape
    if feature_labels is None:
        feature_labels = [f'Feature {d}' for d in range(D)]

    # ── External-labels path — skip clustering, go straight to plotting ───────
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

        cls_colors = plt.cm.tab10(np.linspace(0, 1, min(n_cls, 10)))
        if n_cls > 10:
            cls_colors = plt.cm.hsv(np.linspace(0, 0.9, n_cls))

    else:
        # ── 1. Standardise ────────────────────────────────────────────────────
        scaler      = StandardScaler()
        data_scaled = scaler.fit_transform(data).astype(np.float32)

        # ── 2. Chunked scipy cdist ────────────────────────────────────────────
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
            total         = N,
            desc          = f'  [{label}] pdist',
            unit          = ' rows',
            dynamic_ncols = True,
            bar_format    = ('{l_bar}{bar}| {n_fmt}/{total_fmt} rows '
                             '[{elapsed}<{remaining}, {rate_fmt}]'),
        ) as pbar:
            for chunk_start in range(0, N, CHUNK):
                chunk_end  = min(chunk_start + CHUNK, N)
                query_rows = data_scaled[chunk_start:chunk_end]

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
        unique_cl      = np.unique(labels)
        singleton_cls  = [cl for cl in unique_cl if np.sum(labels == cl) == 1]
        non_singleton_cls = [cl for cl in unique_cl if np.sum(labels == cl) > 1]

        if len(singleton_cls) > 0 and len(non_singleton_cls) > 0:
            print(f'  [{label}] Reassigning {len(singleton_cls)} singleton cluster(s)')
            centroids = np.array([
                data_scaled[labels == cl].mean(axis=0)
                for cl in non_singleton_cls
            ])
            for scl in singleton_cls:
                idx_singleton = np.where(labels == scl)[0][0]
                point   = data_scaled[idx_singleton:idx_singleton + 1]
                dists   = cdist(point, centroids, metric='euclidean')[0]
                nearest = non_singleton_cls[np.argmin(dists)]
                print(f'    Singleton cluster {scl} (idx={idx_singleton}) → cluster {nearest}')
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
    if plot_corner:
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


################################
########## Code block ##########
################################

if not os.path.exists(orig_save_data_path):
    os.makedirs(orig_save_data_path)

# Instantiate dictionary to store information.
# Each entry is indexed [i_Teff, j_logg, k_met] over the stellar parameter grid.
gen_dict = {}
gen_dict['stellar_wavelengths']      = {model: np.empty((N_star, N_star, N_star), dtype=object) for model in models}
gen_dict['global_intensity_profiles']= {model: np.empty((N_star, N_star, N_star), dtype=object) for model in models}
gen_dict['global_mus']               = {model: np.empty((N_star, N_star, N_star), dtype=object) for model in models}


# ─────────────────────────────────────────────────────────────────────────────
# Iterate over all stellar models
# ─────────────────────────────────────────────────────────────────────────────
for model in models:

    # Convert physical indices to values for labels
    T_vals_arr = np.linspace(Teffs[model][0],       Teffs[model][1],       N_star)
    g_vals_arr = np.linspace(loggs[model][0],        loggs[model][1],        N_star)
    m_vals_arr = np.linspace(metallicitys[model][0], metallicitys[model][1], N_star)

    # Create save path for each model
    save_data_path = orig_save_data_path + f'{model}/'
    if not os.path.exists(save_data_path):
        os.makedirs(save_data_path)

    # ── Build / load global intensity profiles ────────────────────────────────
    if intr_prof_mode == 'build':

        for i, T in enumerate(T_vals_arr):
            for j, g in enumerate(g_vals_arr):
                for k, m in enumerate(m_vals_arr):

                    print('GENERATING Teff =', T, 'logg =', g, 'metallicity =', m,
                          'for model', model)
                    sld = el.StellarLimbDarkening(
                        M_H=m, Teff=T, logg=g,
                        ld_model=model,
                        ld_data_path=LD_data_path,
                        interpolate_type='nearest',
                    )

                    #Store the wavelength, mu arrays and global intensity spectrum
                    stellar_wavelengths        = np.array(sld.stellar_wavelengths)  # (n_wav,)
                    stellar_mus                = np.array(sld.mus)                  # (n_mu,)
                    global_stellar_intensities = np.array(sld.stellar_intensities)  # (n_wav, n_mu)
                    del sld

                    # Filter wavelength range
                    cond = ((stellar_wavelengths > wav_region[0]) &
                            (stellar_wavelengths < wav_region[1]))
                    print(f'    Removing '
                          f'{100 * (len(stellar_wavelengths) - np.sum(cond)) / len(stellar_wavelengths):.2f}'
                          f' % of the wavelength range')
                    global_stellar_intensities = global_stellar_intensities[cond, :]
                    stellar_wavelengths        = stellar_wavelengths[cond]
                    gen_dict['stellar_wavelengths'][model][i, j, k] = stellar_wavelengths

                    # Interpolate onto fine, uniform mu grid
                    stellar_mus_fine = np.linspace(stellar_mus[-1], stellar_mus[0], n_mu_fine)
                    interp_func = interp1d(
                        stellar_mus[::-1],
                        global_stellar_intensities[:, ::-1],
                        kind='cubic', axis=1, bounds_error=False,
                    )
                    global_stellar_intensities_fine = interp_func(stellar_mus_fine)  # (n_wav, n_mu_fine)

                    # Restore descending-mu order (centre to limb: mu=1 → mu≈0)
                    stellar_mus_fine                = stellar_mus_fine[::-1]
                    global_stellar_intensities_fine = global_stellar_intensities_fine[:, ::-1]

                    gen_dict['global_intensity_profiles'][model][i, j, k] = global_stellar_intensities_fine
                    gen_dict['global_mus'][model][i, j, k]                = stellar_mus_fine

                    del (global_stellar_intensities, stellar_wavelengths,
                         stellar_mus, interp_func, stellar_mus_fine,
                         global_stellar_intensities_fine, cond)
                    gc.collect()

        with open(save_data_path + 'data.pkl', 'wb') as f:
            pickle.dump(gen_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    elif intr_prof_mode == 'load':
        with open(save_data_path + 'data.pkl', 'rb') as f:
            gen_dict = pickle.load(f)

    else:
        raise ValueError('intr_prof_mode not recognized.')

    # ── PCA analysis ──────────────────────────────────────────────────────────
    outlier_profiles = []
    outlier_mus      = []

    if PCA_mode == 'build':

        print('MASKING')

        # ── Count valid profiles per wavelength bin ───────────────────────────
        n_wav_bins      = 10
        wavs_ref        = np.array(gen_dict['stellar_wavelengths'][model][0, 0, 0])
        wav_bin_slices  = np.array_split(np.arange(len(wavs_ref)), n_wav_bins)
        wav_bin_labels  = [f'{wavs_ref[s[0]]/1e4:.2f}-{wavs_ref[s[-1]]/1e4:.2f} μm'
                           for s in wav_bin_slices]

        n_valid_per_wav = np.zeros(n_wav_bins, dtype=int)
        n_considered    = 0
        n_valid         = 0

        # Each entry: profile (n_wav, n_mu_fine), normalized at mu=1 (centre).
        # A profile/wavelength slice is valid when the intensity is monotonically
        # non-increasing from centre to limb (no upward steps).
        for i in range(N_star):
            for j in range(N_star):
                for k in range(N_star):
                    prof = gen_dict['global_intensity_profiles'][model][i, j, k]  # (n_wav, n_mu_fine)
                    # Normalise each wavelength slice to its centre value
                    center_vals = prof[:, 0:1]  # (n_wav, 1)  — mu=1 is the first column
                    safe_center = np.where(np.abs(center_vals) < 1e-10, 1.0, center_vals)
                    norm_prof   = prof / safe_center  # (n_wav, n_mu_fine)

                    # Valid = monotonically non-increasing (no positive differences)
                    mask = ~np.any(np.diff(norm_prof, axis=1) > 0.0, axis=1)  # (n_wav,)

                    n_considered += mask.size
                    n_valid      += int(np.sum(mask))
                    for iw_bin, idx in enumerate(wav_bin_slices):
                        n_valid_per_wav[iw_bin] += int(np.sum(mask[idx]))

        print(f"=== Profile filtering summary ===")
        print(f"Total profiles considered : {n_considered}")
        print(f"Total valid               : {n_valid} ({100 * n_valid / n_considered:.1f} %)")
        print(f"Per wavelength region:")
        n_total_per_wav = np.array([len(s) * N_star**3 for s in wav_bin_slices])
        for iw_bin in range(n_wav_bins):
            print(f"  {wav_bin_labels[iw_bin]} : "
                  f"{n_valid_per_wav[iw_bin]}/{n_total_per_wav[iw_bin]} valid "
                  f"({100 * n_valid_per_wav[iw_bin] / n_total_per_wav[iw_bin]:.1f} %)")

        # ── Pass 2 — pre-allocate and fill profile matrix ─────────────────────
        # Metadata columns: [i_Teff, j_logg, k_met, i_wav]
        pca_int_profile = np.empty((n_valid, n_mu_fine), dtype=np.float32)
        mus_array       = np.empty((n_valid, n_mu_fine), dtype=np.float32)
        meta_array      = np.empty((n_valid, 4),         dtype=np.int32)
        ptr             = 0

        for i in range(N_star):
            for j in range(N_star):
                for k in range(N_star):
                    prof = gen_dict['global_intensity_profiles'][model][i, j, k]  # (n_wav, n_mu_fine)
                    mus  = gen_dict['global_mus'][model][i, j, k]                 # (n_mu_fine,)

                    center_vals = prof[:, 0:1]
                    safe_center = np.where(np.abs(center_vals) < 1e-10, 1.0, center_vals)
                    norm_prof   = prof / safe_center

                    mask = ~np.any(np.diff(norm_prof, axis=1) > 0.0, axis=1)  # (n_wav,)
                    n_ok = int(np.sum(mask))
                    if n_ok == 0:
                        continue

                    iw_indices  = np.where(mask)[0]
                    pca_int_profile[ptr:ptr + n_ok] = norm_prof[mask].astype(np.float32)
                    mus_array[ptr:ptr + n_ok]       = np.broadcast_to(mus, (n_ok, n_mu_fine)).astype(np.float32)

                    meta_block = np.column_stack([
                        np.full(n_ok, i,  dtype=np.int32),
                        np.full(n_ok, j,  dtype=np.int32),
                        np.full(n_ok, k,  dtype=np.int32),
                        iw_indices.astype(np.int32),
                    ])
                    meta_array[ptr:ptr + n_ok] = meta_block
                    ptr += n_ok

        assert ptr == n_valid, f'Pointer mismatch: {ptr} vs {n_valid}'

        # ── Optional subsampling ──────────────────────────────────────────────
        if subsample_profiles:
            rng    = np.random.default_rng(subsample_seed)
            n_draw = min(n_subsample_profiles, n_valid)
            if n_draw < n_valid:
                idx = np.sort(rng.choice(n_valid, size=n_draw, replace=False))
                print(f'\nSUBSAMPLING: {n_valid} → {n_draw} profiles (seed={subsample_seed})')
            else:
                idx = np.arange(n_valid)
                print(f'\nSUBSAMPLING: {n_valid} profiles (fewer than '
                      f'{n_subsample_profiles}, keeping all)')
            pca_int_profile = pca_int_profile[idx]
            mus_array       = mus_array[idx]
            meta_array      = meta_array[idx]
            n_valid         = n_draw

        # ── Residual plots coloured by parameter ──────────────────────────────
        print('RESIDUAL PLOTTING')
        med   = np.median(pca_int_profile, axis=0)
        resid = pca_int_profile - med

        wav_bin_of_profile = np.empty(len(meta_array), dtype=np.int32)
        for iw_bin, idx_slice in enumerate(wav_bin_slices):
            in_bin = np.isin(meta_array[:, 3], idx_slice)
            wav_bin_of_profile[in_bin] = iw_bin

        color_sources = {
            '$T_{eff}$ (K)':     T_vals_arr[meta_array[:, 0]],
            '$\\log g$':         g_vals_arr[meta_array[:, 1]],
            'Metallicity [M/H]': m_vals_arr[meta_array[:, 2]],
            'Wavelength bin':    wav_bin_of_profile.astype(float),
        }

        step    = max(1, len(resid) // 20000)
        fig_g, axes_g = plt.subplots(1, 4, figsize=(24, 5), sharex=True, sharey=True)
        fig_g.suptitle(f'Global intensity profile residuals coloured by parameter',
                       fontsize=12)

        for ax, (clabel, cvals) in zip(axes_g, color_sources.items()):
            cmap_g = cm.get_cmap('coolwarm')
            norm_g = mcolors.Normalize(vmin=np.min(cvals), vmax=np.max(cvals))
            for idx in range(0, len(resid), step):
                ax.plot(mus_array[idx], resid[idx],
                        alpha=0.15, linewidth=0.4,
                        color=cmap_g(norm_g(cvals[idx])))
            sm = cm.ScalarMappable(cmap=cmap_g, norm=norm_g)
            sm.set_array([])
            plt.colorbar(sm, ax=ax, label=clabel, fraction=0.046, pad=0.04)
            ax.set_xlabel('$\\mu$')
            ax.set_ylabel('Residual intensity' if ax is axes_g[0] else '')
            ax.set_title(clabel)
            ax.grid(True, alpha=0.3)

        fig_g.tight_layout()
        fig_g.savefig(save_data_path + f'Global_profiles_residuals_{model}.pdf',
                      dpi=150, bbox_inches='tight')
        plt.close(fig_g)

        # ── PCA ───────────────────────────────────────────────────────────────
        print('PCA ANALYSIS')
        print(f'  Fitting PCA on {n_valid} global intensity profiles '
              f'({n_mu_fine} mu points, {n_components} components) ...')
        pca_model  = PCA(n_components=n_components)
        scores     = pca_model.fit_transform(pca_int_profile)  # (n_valid, n_components)
        print(f'  Variance captured: {np.sum(pca_model.explained_variance_ratio_)*100:.1f}%')

        # ── Hierarchical clustering ───────────────────────────────────────────
        print('CLUSTERING')
        hc_labels, _, _ = hierarchical_clustering(
            data=scores,
            label=f'PCA_clustering_{model}',
            save_path=save_data_path,
            feature_labels=[f'PC{k+1}' for k in range(n_components)],
            clustering_metric='mahalanobis',
            method='single',
            cutoff=1.2,
        )

        cluster_labels = hc_labels - 1  # 0-indexed
        unique_cl      = np.unique(cluster_labels)

        # Typical profile: closest to its cluster's centroid (globally nearest centroid)
        typical_idx = None
        min_dist    = np.inf
        for cl in unique_cl:
            mask     = cluster_labels == cl
            members  = scores[mask]
            centroid = members.mean(axis=0)
            dists    = np.linalg.norm(members - centroid, axis=1)
            closest  = int(np.where(mask)[0][np.argmin(dists)])
            if dists.min() < min_dist:
                min_dist    = dists.min()
                typical_idx = closest

        # Outlier profiles: furthest from centroid per cluster
        outlier_indices = []
        for cl in unique_cl:
            mask     = cluster_labels == cl
            members  = scores[mask]
            centroid = members.mean(axis=0)
            dists    = np.linalg.norm(members - centroid, axis=1)
            outlier_indices.append(int(np.where(mask)[0][np.argmax(dists)]))

        # ── Figure 1: scree, cumulative variance, eigen profiles ──────────────
        print('PLOTTING')
        print('    FIGURE 1')

        ncols_f1 = 2 + n_components
        fig1, axes1 = plt.subplots(1, ncols_f1, figsize=(4 * ncols_f1, 4))
        evr  = pca_model.explained_variance_ratio_
        eigen = pca_model.components_  # (n_components, n_mu_fine)
        ref_mus = mus_array[0]

        axes1[0].plot(range(1, n_components + 1), evr, 'o-', linewidth=2)
        axes1[0].set_title('Scree')
        axes1[0].set_xlabel('PC')
        axes1[0].set_ylabel('Expl. var. ratio')
        axes1[0].grid(True, alpha=0.3)

        axes1[1].plot(range(1, n_components + 1), np.cumsum(evr), 'o-', linewidth=2)
        axes1[1].axhline(0.95, color='g', linestyle='--', label='95%')
        axes1[1].set_title('Cumul. var.')
        axes1[1].set_xlabel('PC')
        axes1[1].set_ylabel('Cumul. expl. var.')
        axes1[1].legend(fontsize=7)
        axes1[1].grid(True, alpha=0.3)

        for i_plot in range(n_components):
            axes1[2 + i_plot].plot(ref_mus, eigen[i_plot], color=colors[i_plot], linewidth=1.5)
            axes1[2 + i_plot].axhline(0, color='k', linestyle='--', alpha=0.3)
            axes1[2 + i_plot].set_title(f'PC{i_plot+1} ({evr[i_plot]*100:.1f}%)')
            axes1[2 + i_plot].set_xlabel('$\\mu$')
            axes1[2 + i_plot].set_ylabel('Component value')
            axes1[2 + i_plot].grid(True, alpha=0.3)

        fig1.tight_layout()
        fig1.savefig(save_data_path + 'PCA_Analysis.png', dpi=150, bbox_inches='tight')
        plt.close(fig1)

        # ── Figure 2a: PCA corner scatter coloured by cluster ─────────────────
        print('    FIGURE 2a - PCA corner scatter')
        n_cl = len(unique_cl)
        cluster_cmap   = plt.cm.get_cmap('tab10', n_cl)
        cluster_colors = [cluster_cmap(c) for c in range(n_cl)]

        special_indices = [typical_idx] + outlier_indices
        special_labels  = ['Typical'] + [f'Outlier {c}' for c in range(len(outlier_indices))]
        special_colors  = ['blue'] + ['red'] * len(outlier_indices)

        fig2a, axes2a = plt.subplots(n_components, n_components,
                                     figsize=(3 * n_components, 3 * n_components))
        fig2a.suptitle(f'PCA cluster structure — {model}', fontsize=12, y=1.01)

        for row in range(n_components):
            for col in range(n_components):
                ax = axes2a[row, col]
                if row == col:
                    for ci, cl in enumerate(unique_cl):
                        cmask = cluster_labels == cl
                        ax.hist(scores[cmask, row], bins=30, alpha=0.5,
                                color=cluster_colors[ci], label=f'C{cl}',
                                density=True, linewidth=0.8, edgecolor='none')
                    ax.set_xlabel(f'PC{row+1}', fontsize=8)
                    ax.tick_params(labelsize=7)
                    for spine in ['top', 'left', 'right']:
                        ax.spines[spine].set_visible(False)
                    ax.set_yticks([])
                elif row > col:
                    for ci, cl in enumerate(unique_cl):
                        cmask = cluster_labels == cl
                        ax.scatter(scores[cmask, col], scores[cmask, row],
                                   color=cluster_colors[ci], s=6, alpha=0.35,
                                   linewidths=0, label=f'C{cl}', rasterized=True)
                    for cidx, slabel, scol in zip(special_indices, special_labels, special_colors):
                        ax.scatter(scores[cidx, col], scores[cidx, row],
                                   color=scol, s=60, zorder=10,
                                   marker='*' if slabel == 'Typical' else 'D',
                                   edgecolors='k', linewidths=0.5, label=slabel)
                    if col == 0:
                        ax.set_ylabel(f'PC{row+1}', fontsize=8)
                    if row == n_components - 1:
                        ax.set_xlabel(f'PC{col+1}', fontsize=8)
                    ax.tick_params(labelsize=6)
                    ax.grid(True, alpha=0.2)
                else:
                    ax.set_visible(False)

        last_ax  = axes2a[n_components - 1, n_components - 2]
        handles, lbls = last_ax.get_legend_handles_labels()
        seen = {}
        for h, l in zip(handles, lbls):
            if l not in seen:
                seen[l] = h
        fig2a.legend(seen.values(), seen.keys(), loc='upper right',
                     fontsize=8, markerscale=1.5, framealpha=0.8)
        fig2a.tight_layout()
        fig2a.savefig(save_data_path + f'PCA_Corner_Scatter_{model}.png',
                      dpi=150, bbox_inches='tight')
        plt.close(fig2a)

        # ── Figure 2b: PCA corner scatter coloured by physical parameters ─────
        print('    FIGURE 2b - PCA corner scatter by multiple parameters')

        wav_bins_ref    = np.array(gen_dict['stellar_wavelengths'][model][0, 0, 0])
        color_schemes = {
            'Teff': {'name': '$T_{eff}$ (K)',        'cmap': plt.cm.inferno,  'unit': 'K'},
            'logg': {'name': '$\\log g$',            'cmap': plt.cm.cividis,  'unit': 'dex'},
            'MH':   {'name': '[M/H]',                'cmap': plt.cm.coolwarm, 'unit': 'dex'},
            'wav':  {'name': 'Wavelength ($\\mu$m)', 'cmap': plt.cm.turbo,    'unit': 'μm'},
        }
        phys_dict = {
            'Teff': T_vals_arr[meta_array[:, 0]],
            'logg': g_vals_arr[meta_array[:, 1]],
            'MH':   m_vals_arr[meta_array[:, 2]],
            'wav':  wav_bins_ref[meta_array[:, 3]] / 1e4,
        }

        for scheme_key, scheme_info in color_schemes.items():
            col_vals  = phys_dict[scheme_key]
            col_norm  = mcolors.Normalize(vmin=col_vals.min(), vmax=col_vals.max())
            col_cmap  = scheme_info['cmap']
            col_label = scheme_info['name']

            print(f'      PCA corner for {model} coloured by {scheme_key}')

            fig2b, axes2b = plt.subplots(n_components, n_components,
                                         figsize=(3 * n_components, 3 * n_components))
            fig2b.suptitle(f'PCA space — {model}, coloured by {col_label}',
                           fontsize=12, y=1.01)

            max_scatter = 20_000
            rng_2b = np.random.default_rng(42)
            if len(scores) > max_scatter:
                idx_sub = np.sort(rng_2b.choice(len(scores), size=max_scatter, replace=False))
            else:
                idx_sub = np.arange(len(scores))
            idx_sub = idx_sub[np.argsort(col_vals[idx_sub])]

            for row in range(n_components):
                for col in range(n_components):
                    ax = axes2b[row, col]
                    if row == col:
                        n_hist_bins = 8 if len(np.unique(col_vals)) > 50 else len(np.unique(col_vals))
                        if n_hist_bins > 1:
                            edges   = np.linspace(col_vals.min(), col_vals.max(), n_hist_bins + 1)
                            centres = 0.5 * (edges[:-1] + edges[1:])
                            for ib_m, centre in enumerate(centres):
                                mask_bin = (col_vals >= edges[ib_m]) & (col_vals < edges[ib_m + 1])
                                if ib_m == len(centres) - 1:
                                    mask_bin |= (col_vals == edges[ib_m + 1])
                                if mask_bin.sum() == 0:
                                    continue
                                ax.hist(scores[mask_bin, row], bins=30, alpha=0.55,
                                        color=col_cmap(col_norm(centre)), density=True,
                                        histtype='stepfilled', edgecolor='none')
                        else:
                            ax.hist(scores[:, row], bins=30, alpha=0.55,
                                    color=col_cmap(col_norm(col_vals.mean())), density=True,
                                    histtype='stepfilled', edgecolor='none')
                        ax.set_xlabel(f'PC{row+1}', fontsize=8)
                        ax.tick_params(labelsize=7)
                        for spine in ['top', 'left', 'right']:
                            ax.spines[spine].set_visible(False)
                        ax.set_yticks([])
                    elif row > col:
                        ax.scatter(scores[idx_sub, col], scores[idx_sub, row],
                                   c=col_vals[idx_sub], cmap=col_cmap, norm=col_norm,
                                   s=8, alpha=0.4, linewidths=0, rasterized=True)
                        for cidx, slabel, scol in zip(special_indices, special_labels, special_colors):
                            ax.scatter(scores[cidx, col], scores[cidx, row],
                                       color=scol, s=60, zorder=10,
                                       marker='*' if slabel == 'Typical' else 'D',
                                       edgecolors='k', linewidths=0.5, label=slabel)
                        if col == 0:
                            ax.set_ylabel(f'PC{row+1}', fontsize=8)
                        if row == n_components - 1:
                            ax.set_xlabel(f'PC{col+1}', fontsize=8)
                        ax.tick_params(labelsize=6)
                        ax.grid(True, alpha=0.2)
                    else:
                        ax.set_visible(False)

            cbar_ax = fig2b.add_axes([0.92, 0.15, 0.02, 0.7])
            sm = cm.ScalarMappable(cmap=col_cmap, norm=col_norm)
            sm.set_array([])
            cbar = fig2b.colorbar(sm, cax=cbar_ax)
            cbar.set_label(col_label, fontsize=11)
            cbar.ax.tick_params(labelsize=9)
            unique_vals = np.unique(col_vals)
            if len(unique_vals) <= 12:
                cbar.set_ticks(unique_vals)
                if scheme_key == 'Teff':
                    cbar.set_ticklabels([f'{v:.0f}' for v in unique_vals], fontsize=8)
                elif scheme_key == 'wav':
                    cbar.set_ticklabels([f'{v:.2f}' for v in unique_vals], fontsize=8)
                else:
                    cbar.set_ticklabels([f'{v:.2f}' for v in unique_vals], fontsize=8)

            fig2b.tight_layout(rect=[0, 0, 0.90, 1])
            fig2b.savefig(save_data_path + f'PCA_Corner_Scatter_{model}_by{scheme_key}.png',
                          dpi=150, bbox_inches='tight')
            plt.close(fig2b)
            print(f'        Saved: by{scheme_key}')

        # ── Figure 2c: typical and outlier profiles ───────────────────────────
        print('    FIGURE 2c - Mode and outlier profiles')
        n_specials = 1 + len(outlier_indices)
        fig2c, axes2c = plt.subplots(1, n_specials,
                                     figsize=(5 * n_specials, 5),
                                     sharey=True)
        if n_specials == 1:
            axes2c = np.array([axes2c])

        for col, (cidx, slabel, scol) in enumerate(
                zip(special_indices, special_labels, special_colors)):
            ax = axes2c[col]
            step = max(1, n_valid // 200)
            for nval in range(0, n_valid, step):
                ax.plot(mus_array[nval], pca_int_profile[nval],
                        alpha=0.15, color='gray', linewidth=0.3)
            ax.plot(mus_array[cidx], pca_int_profile[cidx],
                    color=scol, linewidth=2, label=slabel, zorder=10)
            ax.set_xlabel('$\\mu$')
            ax.set_ylabel('Norm. Intensity')
            ax.set_title(f'{slabel}')
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        fig2c.suptitle(f'Mode and outlier profiles — {model}', fontsize=12)
        fig2c.tight_layout()
        fig2c.savefig(save_data_path + 'Mode_and_Outliers.png', dpi=150, bbox_inches='tight')
        plt.close(fig2c)

        # ── Figure 3: reconstruction quality for the typical profile ──────────
        print('    FIGURE 3')
        n_comp_list = [1, max(2, n_components // 2), n_components]
        fig3 = plt.figure(figsize=(5 * len(n_comp_list), 6 * 1))
        gs   = GridSpec(2, len(n_comp_list), figure=fig3, hspace=0.05, wspace=0.3)

        original = pca_int_profile[typical_idx]
        mu_orig  = mus_array[typical_idx]

        for col_idx, n_comp_plot in enumerate(n_comp_list):
            pca_temp      = PCA(n_components=n_comp_plot)
            pca_temp.fit(pca_int_profile)
            reconstructed = pca_temp.inverse_transform(
                                pca_temp.transform(original.reshape(1, -1)))[0]
            residual      = original - reconstructed
            rmse          = np.sqrt(np.mean(residual ** 2))

            ax_top = fig3.add_subplot(gs[0, col_idx])
            ax_bot = fig3.add_subplot(gs[1, col_idx], sharex=ax_top)

            ax_top.plot(mu_orig, original,      linewidth=2,   label='Original')
            ax_top.plot(mu_orig, reconstructed, linewidth=1.2, linestyle='--', label='Recon.')
            ax_top.set_title(f'{n_comp_plot}PC  RMSE={rmse:.4f}', fontsize=8)
            ax_top.set_ylabel('Norm. Intensity', fontsize=7)
            ax_top.legend(fontsize=6)
            ax_top.grid(True, alpha=0.3)
            ax_top.tick_params(labelbottom=False)

            safe = np.where(np.abs(original) < 1e-10, np.nan, original)
            ax_bot.plot(mu_orig, 100 * (residual / safe), linewidth=1.0)
            ax_bot.axhline(0, color='k', linestyle='--', alpha=0.5)
            ax_bot.set_xlabel('$\\mu$', fontsize=7)
            ax_bot.set_ylabel('Rel. diff. (%)', fontsize=7)
            ax_bot.grid(True, alpha=0.3)

        fig3.savefig(save_data_path + 'Reconstruction_Quality.png', dpi=150, bbox_inches='tight')
        plt.close(fig3)

        # ── Save profiles and metadata ────────────────────────────────────────
        np.save(save_data_path + f'mode_intensity_profile_{model}.npy',
                pca_int_profile[typical_idx])
        np.save(save_data_path + f'mode_mus_{model}.npy',
                mus_array[typical_idx])
        for i_save, cidx in enumerate(outlier_indices):
            np.save(save_data_path + f'outlier{i_save+1}_intensity_profile_{model}.npy',
                    pca_int_profile[cidx])
            np.save(save_data_path + f'outlier{i_save+1}_mus_{model}.npy',
                    mus_array[cidx])
        np.save(save_data_path + f'meta_{model}.npy', meta_array)

        typical_profile  = pca_int_profile[typical_idx]
        typical_mus      = mus_array[typical_idx]
        outlier_profiles = [pca_int_profile[cidx] for cidx in outlier_indices]
        outlier_mus      = [mus_array[cidx]        for cidx in outlier_indices]
        print(f"Saved profiles to {save_data_path}")

    elif PCA_mode == 'load':
        typical_profile = np.load(save_data_path + f'mode_intensity_profile_{model}.npy')
        typical_mus     = np.load(save_data_path + f'mode_mus_{model}.npy')
        meta_array      = np.load(save_data_path + f'meta_{model}.npy')

        outlier_profiles, outlier_mus = [], []
        i_save = 0
        while True:
            p_path = save_data_path + f'outlier{i_save+1}_intensity_profile_{model}.npy'
            if not os.path.exists(p_path):
                break
            outlier_profiles.append(np.load(p_path))
            outlier_mus.append(np.load(save_data_path + f'outlier{i_save+1}_mus_{model}.npy'))
            i_save += 1
        print(f"Loaded profiles from {save_data_path}")

    else:
        raise ValueError('PCA_mode not recognized.')

    # ── Figure 4: 4th-order NLLD fit for mode and outlier profiles ────────────
    print('    FIGURE 4 - NLLD fits for mode and outliers')

    specials = [('mode', typical_profile, typical_mus)]
    for i_out, (op, om) in enumerate(zip(outlier_profiles, outlier_mus)):
        specials.append((f'outlier{i_out+1}', op, om))

    for label, prof, mus_sp in specials:
        params = Parameters()
        for ip in range(4):
            params.add(f'c{ip+1}', value=np.random.uniform(0, 1))
        result = minimize(residual_fn, params, args=(mus_sp, prof))
        coeffs = [result.params[f'c{ic+1}'].value for ic in range(4)]
        fit    = fourNLLD(mus_sp, coeffs)

        fig4, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(8, 6), sharex=True,
            gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05},
        )
        fig4.suptitle(f'4th Order NLLD Fit — {label} — {model}', fontsize=13)

        ax1.plot(mus_sp, prof, 'o', markersize=3, alpha=0.6, label='Profile')
        ax1.plot(mus_sp, fit,  'r--', linewidth=2, label='4th order NLLD')
        ax1.set_ylabel('Norm. Intensity')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.text(0.98, 0.05,
                 f"$c$=[{coeffs[0]:.3f}, {coeffs[1]:.3f}, {coeffs[2]:.3f}, {coeffs[3]:.3f}]",
                 transform=ax1.transAxes, fontsize=8, ha='right', va='bottom',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        safe = np.where(np.abs(prof) < 1e-10, np.nan, prof)
        ax2.plot(mus_sp, 100 * (prof - fit) / safe, '--', linewidth=1.5)
        ax2.axhline(0, color='k', linestyle='--', alpha=0.5)
        ax2.set_xlabel('$\\mu$ = cos($\\theta$)')
        ax2.set_ylabel('Rel. diff. (%)')
        ax2.grid(True, alpha=0.3)

        print(f"  {label}  c=[{', '.join(f'{c:.4f}' for c in coeffs)}]  "
              f"redchi={result.redchi:.4e}")

        fig4.savefig(save_data_path + f'4thOrderNLLD_Fit_Profile_{label}_{model}.png',
                     dpi=150, bbox_inches='tight')
        plt.close(fig4)

    # ── Fit ALL profiles with 4th-order NLLD ──────────────────────────────────
    print('Fitting ALL profiles with 4th-order NLLD')

    if All_Corner == 'build':
        n_profs  = pca_int_profile.shape[0]
        all_coeffs = np.zeros((n_profs, 4), dtype=np.float64)
        print(f'  Fitting {n_profs} global intensity profiles ...')

        for idx in tqdm(range(n_profs)):
            prof_idx = pca_int_profile[idx]
            mus_idx  = mus_array[idx]

            if np.all(np.abs(prof_idx) < 1e-10):
                all_coeffs[idx] = np.nan
                continue

            params = Parameters()
            for ip in range(4):
                params.add(f'c{ip+1}', value=np.random.uniform(0, 1))
            try:
                result = minimize(residual_fn, params, args=(mus_idx, prof_idx))
                all_coeffs[idx] = [result.params[f'c{ic+1}'].value for ic in range(4)]
            except Exception:
                all_coeffs[idx] = np.nan

        np.save(save_data_path + f'all_coeffs_{model}.npy', all_coeffs)

    elif All_Corner == 'load':
        all_coeffs = np.load(save_data_path + f'all_coeffs_{model}.npy')

    else:
        raise ValueError('All_Corner mode not recognized.')

    print('  Done fitting. Now making corner plots...')

    # ── Corner data: [c1, c2, c3, c4] ────────────────────────────────────────
    valid_mask  = ~np.any(np.isnan(all_coeffs), axis=1)
    corner_data = all_coeffs[valid_mask]
    print(f'  Corner plot for {corner_data.shape[0]} valid profiles')

    labels_4d = [r'$c_1$', r'$c_2$', r'$c_3$', r'$c_4$']
    ranges_4d = [
        (np.percentile(corner_data[:, ic], 1),
         np.percentile(corner_data[:, ic], 99))
        for ic in range(4)
    ]
    ndim_4d = 4

    # ── Figure 5a: density corner ─────────────────────────────────────────────
    print('    FIGURE 5a - All coefficients corner plot')
    fig_corner = corner.corner(
        corner_data,
        labels=labels_4d, range=ranges_4d, bins=50,
        smooth=1.0, smooth1d=1.0,
        plot_datapoints=True, plot_density=False, fill_contours=False,
        levels=(0.5, 0.68, 0.95, 0.99),
        hist_kwargs={'color': 'black', 'linewidth': 1.5},
        label_kwargs={'fontsize': 13}, title_kwargs={'fontsize': 11},
        show_titles=False,
        data_kwargs={'alpha': 0.2, 'ms': 1.5, 'color': 'black'},
        contourf_kwargs={'alpha': 0.5},
        contour_kwargs={'colors': ['brown', 'red', 'orange', 'yellow']},
    )
    for i in range(ndim_4d):
        ax = fig_corner.axes[i * ndim_4d + i]
        for spine in ['top', 'left', 'right']:
            ax.spines[spine].set_visible(False)
    fig_corner.suptitle(f'4th-order NLLD coefficients — {model}', fontsize=13, y=1.02)
    fig_corner.savefig(save_data_path + f'Corner_NLLD_{model}.png', dpi=150, bbox_inches='tight')
    plt.close(fig_corner)

    # ── Build physical parameter arrays aligned with corner_data ─────────────
    corner_meta    = meta_array[valid_mask]
    wavs_ref       = np.array(gen_dict['stellar_wavelengths'][model][0, 0, 0])
    corner_wav_um  = wavs_ref[corner_meta[:, 3]] / 1e4

    meta_col_names = [r'$T_{\rm eff}$ (K)', r'$\log\,g$', r'[M/H]']
    meta_col_keys  = ['Teff', 'logg', 'MH']
    meta_col_vals  = [T_vals_arr[corner_meta[:, 0]],
                      g_vals_arr[corner_meta[:, 1]],
                      m_vals_arr[corner_meta[:, 2]]]
    meta_cmaps     = [plt.cm.inferno, plt.cm.cividis, plt.cm.coolwarm]

    # ── Figures 5b–5d: coloured by physical parameters ────────────────────────
    for imeta in range(3):
        col_vals  = meta_col_vals[imeta]
        col_label = meta_col_names[imeta]
        col_key   = meta_col_keys[imeta]
        col_cmap  = meta_cmaps[imeta]
        col_norm  = mcolors.Normalize(vmin=col_vals.min(), vmax=col_vals.max())

        print(f'    FIGURE 5{chr(98 + imeta)} - Corner plot coloured by {col_key}')

        fig_cm = corner.corner(
            corner_data, labels=labels_4d, range=ranges_4d, bins=50, smooth1d=1.0,
            plot_datapoints=False, plot_density=False, fill_contours=False, no_fill_contours=True,
            levels=(0.5, 0.68, 0.95, 0.99),
            hist_kwargs={'color': 'gray', 'linewidth': 1.2, 'alpha': 0.5},
            label_kwargs={'fontsize': 13}, show_titles=False,
            contour_kwargs={'colors': 'none'},
        )
        axes_cm = np.array(fig_cm.axes).reshape(ndim_4d, ndim_4d)

        rng_cm  = np.random.default_rng(42)
        idx_sub = (np.sort(rng_cm.choice(len(corner_data), size=40_000, replace=False))
                   if len(corner_data) > 40_000 else np.arange(len(corner_data)))
        idx_sub = idx_sub[np.argsort(col_vals[idx_sub])]

        for row in range(ndim_4d):
            for col in range(row):
                axes_cm[row, col].scatter(
                    corner_data[idx_sub, col], corner_data[idx_sub, row],
                    c=col_vals[idx_sub], cmap=col_cmap, norm=col_norm,
                    s=1.5, alpha=0.35, linewidths=0, rasterized=True)

        unique_vals = np.unique(col_vals)
        for d in range(ndim_4d):
            ax = axes_cm[d, d]
            ax.clear()
            if len(unique_vals) <= 20:
                for uv in unique_vals:
                    mask_uv = col_vals == uv
                    if mask_uv.sum() == 0:
                        continue
                    ax.hist(corner_data[mask_uv, d], bins=50, range=ranges_4d[d],
                            alpha=0.55, color=col_cmap(col_norm(uv)), density=True,
                            histtype='stepfilled', edgecolor='none')
            else:
                n_bins = 8
                edges  = np.linspace(col_vals.min(), col_vals.max(), n_bins + 1)
                ctrs   = 0.5 * (edges[:-1] + edges[1:])
                for ib_m, ctr in enumerate(ctrs):
                    mask_bm = ((col_vals >= edges[ib_m]) &
                               (col_vals < edges[ib_m + 1]))
                    if ib_m == len(ctrs) - 1:
                        mask_bm |= (col_vals == edges[ib_m + 1])
                    if mask_bm.sum() == 0:
                        continue
                    ax.hist(corner_data[mask_bm, d], bins=50, range=ranges_4d[d],
                            alpha=0.55, color=col_cmap(col_norm(ctr)), density=True,
                            histtype='stepfilled', edgecolor='none')
            ax.set_xlim(ranges_4d[d])
            ax.set_yticks([])
            for spine in ['top', 'left', 'right']:
                ax.spines[spine].set_visible(False)
            if d == ndim_4d - 1:
                ax.set_xlabel(labels_4d[d], fontsize=13)

        for row in range(ndim_4d):
            for col in range(row):
                ax = axes_cm[row, col]
                if col == 0:
                    ax.set_ylabel(labels_4d[row], fontsize=13)
                if row == ndim_4d - 1:
                    ax.set_xlabel(labels_4d[col], fontsize=13)
                ax.tick_params(labelsize=8)
                ax.grid(True, alpha=0.15)

        cbar_ax = fig_cm.add_axes([0.72, 0.72, 0.025, 0.20])
        sm_cm   = cm.ScalarMappable(cmap=col_cmap, norm=col_norm)
        sm_cm.set_array([])
        cbar_cm = fig_cm.colorbar(sm_cm, cax=cbar_ax)
        cbar_cm.set_label(col_label, fontsize=13)
        cbar_cm.ax.tick_params(labelsize=10)
        if len(unique_vals) <= 15:
            cbar_cm.set_ticks(unique_vals)
            cbar_cm.set_ticklabels(
                [f'{v:.0f}' for v in unique_vals] if col_key == 'Teff'
                else [f'{v:.2f}' for v in unique_vals])

        fig_cm.suptitle(f'4th-order NLLD coefficients coloured by {col_label} — {model}',
                        fontsize=13, y=1.02)
        fig_cm.savefig(save_data_path + f'Corner_NLLD_by{col_key}_{model}.png',
                       dpi=150, bbox_inches='tight')
        plt.close(fig_cm)

    # ── Figure 5e: coloured by wavelength ─────────────────────────────────────
    print('    FIGURE 5e - Corner plot coloured by wavelength')
    wav_cmap = plt.cm.turbo
    wav_norm = mcolors.Normalize(vmin=corner_wav_um.min(), vmax=corner_wav_um.max())

    fig_wav = corner.corner(
        corner_data, labels=labels_4d, range=ranges_4d, bins=50, smooth1d=1.0,
        plot_datapoints=False, plot_density=False, fill_contours=False, no_fill_contours=True,
        levels=(0.5, 0.68, 0.95, 0.99),
        hist_kwargs={'color': 'gray', 'linewidth': 1.2, 'alpha': 0.5},
        label_kwargs={'fontsize': 13}, show_titles=False,
        contour_kwargs={'colors': 'none'},
    )
    axes_wav = np.array(fig_wav.axes).reshape(ndim_4d, ndim_4d)

    rng_wav = np.random.default_rng(42)
    idx_sub_wav = (np.sort(rng_wav.choice(len(corner_data), size=40_000, replace=False))
                   if len(corner_data) > 40_000 else np.arange(len(corner_data)))
    idx_sub_wav = idx_sub_wav[np.argsort(corner_wav_um[idx_sub_wav])]

    for row in range(ndim_4d):
        for col in range(row):
            axes_wav[row, col].scatter(
                corner_data[idx_sub_wav, col], corner_data[idx_sub_wav, row],
                c=corner_wav_um[idx_sub_wav], cmap=wav_cmap, norm=wav_norm,
                s=1.5, alpha=0.35, linewidths=0, rasterized=True)

    n_wav_hist_bins = 10
    wav_edges   = np.linspace(corner_wav_um.min(), corner_wav_um.max(), n_wav_hist_bins + 1)
    wav_centres = 0.5 * (wav_edges[:-1] + wav_edges[1:])

    for d in range(ndim_4d):
        ax = axes_wav[d, d]
        ax.clear()
        for ibin, centre in enumerate(wav_centres):
            mask_bin = ((corner_wav_um >= wav_edges[ibin]) &
                        (corner_wav_um <  wav_edges[ibin + 1]))
            if ibin == n_wav_hist_bins - 1:
                mask_bin |= (corner_wav_um == wav_edges[ibin + 1])
            if mask_bin.sum() == 0:
                continue
            ax.hist(corner_data[mask_bin, d], bins=50, range=ranges_4d[d],
                    alpha=0.55, color=wav_cmap(wav_norm(centre)), density=True,
                    histtype='stepfilled', edgecolor='none')
        ax.set_xlim(ranges_4d[d])
        ax.set_yticks([])
        for spine in ['top', 'left', 'right']:
            ax.spines[spine].set_visible(False)
        if d == ndim_4d - 1:
            ax.set_xlabel(labels_4d[d], fontsize=13)

    for row in range(ndim_4d):
        for col in range(row):
            ax = axes_wav[row, col]
            if col == 0:
                ax.set_ylabel(labels_4d[row], fontsize=13)
            if row == ndim_4d - 1:
                ax.set_xlabel(labels_4d[col], fontsize=13)
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.15)

    cbar_ax_wav = fig_wav.add_axes([0.72, 0.72, 0.025, 0.20])
    sm_wav      = cm.ScalarMappable(cmap=wav_cmap, norm=wav_norm)
    sm_wav.set_array([])
    cbar_wav    = fig_wav.colorbar(sm_wav, cax=cbar_ax_wav)
    cbar_wav.set_label(r'Wavelength ($\mu$m)', fontsize=13)
    cbar_wav.ax.tick_params(labelsize=10)
    wav_tick_step = 0.5
    wav_ticks     = np.arange(
        np.ceil(corner_wav_um.min() / wav_tick_step) * wav_tick_step,
        corner_wav_um.max() + wav_tick_step / 2,
        wav_tick_step)
    cbar_wav.set_ticks(wav_ticks)
    cbar_wav.set_ticklabels([f'{t:.1f}' for t in wav_ticks])

    fig_wav.suptitle(f'4th-order NLLD coefficients coloured by wavelength — {model}',
                     fontsize=13, y=1.02)
    fig_wav.savefig(save_data_path + f'Corner_NLLD_byWav_{model}.png',
                    dpi=150, bbox_inches='tight')
    plt.close(fig_wav)

    # ── Figure 5f: coloured by PCA-space cluster labels ───────────────────────
    print('    FIGURE 5f - Coefficient corner plot coloured by PCA clusters')

    pca_cl_corner  = cluster_labels[valid_mask]
    unique_pca_cl  = np.unique(pca_cl_corner)
    n_pca_clusters = len(unique_pca_cl)

    pca_cl_cmap   = (plt.cm.tab10 if n_pca_clusters <= 10
                     else plt.cm.tab20 if n_pca_clusters <= 20
                     else plt.cm.hsv)
    pca_cl_colors = {cl: pca_cl_cmap(i / max(n_pca_clusters - 1, 1))
                     for i, cl in enumerate(unique_pca_cl)}

    fig_pca_on_coeff = corner.corner(
        corner_data, labels=labels_4d, range=ranges_4d, bins=50, smooth1d=1.0,
        plot_datapoints=False, plot_density=False, fill_contours=False, no_fill_contours=True,
        levels=(0.5, 0.68, 0.95, 0.99),
        hist_kwargs={'color': 'gray', 'linewidth': 1.2, 'alpha': 0.5},
        label_kwargs={'fontsize': 13}, show_titles=False,
        contour_kwargs={'colors': 'none'},
    )
    axes_pc = np.array(fig_pca_on_coeff.axes).reshape(ndim_4d, ndim_4d)

    rng_pc     = np.random.default_rng(42)
    idx_sub_pc = (np.sort(rng_pc.choice(len(corner_data), size=40_000, replace=False))
                  if len(corner_data) > 40_000 else np.arange(len(corner_data)))

    for row in range(ndim_4d):
        for col in range(row):
            ax = axes_pc[row, col]
            for cl in unique_pca_cl:
                mask_cl = pca_cl_corner[idx_sub_pc] == cl
                if mask_cl.sum() == 0:
                    continue
                ax.scatter(corner_data[idx_sub_pc[mask_cl], col],
                           corner_data[idx_sub_pc[mask_cl], row],
                           color=pca_cl_colors[cl], s=1.5, alpha=0.35,
                           linewidths=0, rasterized=True)

    for d in range(ndim_4d):
        ax = axes_pc[d, d]
        ax.clear()
        for cl in unique_pca_cl:
            mask_cl = pca_cl_corner == cl
            if mask_cl.sum() == 0:
                continue
            ax.hist(corner_data[mask_cl, d], bins=50, range=ranges_4d[d],
                    alpha=0.55, color=pca_cl_colors[cl], density=True,
                    histtype='stepfilled', edgecolor='none')
        ax.set_xlim(ranges_4d[d])
        ax.set_yticks([])
        for spine in ['top', 'left', 'right']:
            ax.spines[spine].set_visible(False)
        if d == ndim_4d - 1:
            ax.set_xlabel(labels_4d[d], fontsize=13)

    for row in range(ndim_4d):
        for col in range(row):
            ax = axes_pc[row, col]
            if col == 0:
                ax.set_ylabel(labels_4d[row], fontsize=13)
            if row == ndim_4d - 1:
                ax.set_xlabel(labels_4d[col], fontsize=13)
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.15)

    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=pca_cl_colors[cl], markersize=7,
                   label=f'PCA cluster {cl}')
        for cl in unique_pca_cl
    ]
    fig_pca_on_coeff.legend(handles=legend_handles, loc='upper right', fontsize=7,
                            ncol=2 if n_pca_clusters > 8 else 1, framealpha=0.85,
                            title='PCA clusters', title_fontsize=8,
                            bbox_to_anchor=(0.98, 0.98))
    fig_pca_on_coeff.suptitle(f'NLLD coefficients coloured by PCA clusters — {model}',
                              fontsize=13, y=1.02)
    fig_pca_on_coeff.savefig(save_data_path + f'Corner_NLLD_coloured_byPCA_clustering_{model}.png',
                             dpi=150, bbox_inches='tight')
    plt.close(fig_pca_on_coeff)

    # ── Mode identification in coefficient space ──────────────────────────────
    print('\n=== MODE IDENTIFICATION IN COEFFICIENT SPACE ===')

    mode_labels_corner, _, _ = hierarchical_clustering(
        data=corner_data,
        label=f'LDC_clustering_{model}',
        save_path=save_data_path,
        feature_labels=[r'$c_1$', r'$c_2$', r'$c_3$', r'$c_4$'],
        cutoff=80,
        clustering_metric='mahalanobis',
        method='ward',
    )

    cluster_labels = mode_labels_corner - 1  # 0-indexed
    unique_cl      = np.unique(cluster_labels)

    for m in unique_cl:
        mask_m = mode_labels_corner == m
        print(
            f'  Mode {m}: n={mask_m.sum():5d}  '
            f'c=[{corner_data[mask_m, 0].mean():.3f}, '
            f'{corner_data[mask_m, 1].mean():.3f}, '
            f'{corner_data[mask_m, 2].mean():.3f}, '
            f'{corner_data[mask_m, 3].mean():.3f}]'
        )

    # Typical profile: closest to its cluster's centroid (globally nearest centroid)
    typical_idx = None
    min_dist    = np.inf
    for cl in unique_cl:
        mask     = cluster_labels == cl
        members  = corner_data[mask]
        centroid = members.mean(axis=0)
        dists    = np.linalg.norm(members - centroid, axis=1)
        closest  = int(np.where(mask)[0][np.argmin(dists)])
        if dists.min() < min_dist:
            min_dist    = dists.min()
            typical_idx = closest

    # Outlier profiles: furthest from centroid per cluster
    outlier_indices = []
    for cl in unique_cl:
        mask     = cluster_labels == cl
        members  = corner_data[mask]
        centroid = members.mean(axis=0)
        dists    = np.linalg.norm(members - centroid, axis=1)
        outlier_indices.append(int(np.where(mask)[0][np.argmax(dists)]))

    # ── Figure 6a: PCA corner scatter coloured by cluster ─────────────────
    print('    FIGURE 6a - NLLD corner scatter')
    n_cl = len(unique_cl)
    cluster_cmap   = plt.cm.get_cmap('tab10', n_cl)
    cluster_colors = [cluster_cmap(c) for c in range(n_cl)]

    special_indices = [typical_idx] + outlier_indices
    special_labels  = ['Typical'] + [f'Outlier {c}' for c in range(len(outlier_indices))]
    special_colors  = ['blue'] + ['red'] * len(outlier_indices)

    fig6a, axes6a = plt.subplots(4, 4,
                                    figsize=(3 * 4, 3 * 4))
    fig6a.suptitle(f'NLLD cluster structure — {model}', fontsize=12, y=1.01)

    for row in range(4):
        for col in range(4):
            ax = axes6a[row, col]
            if row == col:
                for ci, cl in enumerate(unique_cl):
                    cmask = cluster_labels == cl
                    ax.hist(corner_data[cmask, row], bins=30, alpha=0.5,
                            color=cluster_colors[ci], label=f'C{cl}',
                            density=True, linewidth=0.8, edgecolor='none')
                ax.set_xlabel(f'PC{row+1}', fontsize=8)
                ax.tick_params(labelsize=7)
                for spine in ['top', 'left', 'right']:
                    ax.spines[spine].set_visible(False)
                ax.set_yticks([])
            elif row > col:
                for ci, cl in enumerate(unique_cl):
                    cmask = cluster_labels == cl
                    ax.scatter(corner_data[cmask, col], corner_data[cmask, row],
                                color=cluster_colors[ci], s=6, alpha=0.35,
                                linewidths=0, label=f'C{cl}', rasterized=True)
                for cidx, slabel, scol in zip(special_indices, special_labels, special_colors):
                    ax.scatter(corner_data[cidx, col], corner_data[cidx, row],
                                color=scol, s=60, zorder=10,
                                marker='*' if slabel == 'Typical' else 'D',
                                edgecolors='k', linewidths=0.5, label=slabel)
                if col == 0:
                    ax.set_ylabel(f'PC{row+1}', fontsize=8)
                if row == 3:
                    ax.set_xlabel(f'PC{col+1}', fontsize=8)
                ax.tick_params(labelsize=6)
                ax.grid(True, alpha=0.2)
            else:
                ax.set_visible(False)

    last_ax  = axes6a[3, 2]
    handles, lbls = last_ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, lbls):
        if l not in seen:
            seen[l] = h
    fig6a.legend(seen.values(), seen.keys(), loc='upper right',
                    fontsize=8, markerscale=1.5, framealpha=0.8)
    fig6a.tight_layout()
    fig6a.savefig(save_data_path + f'NLLD_Corner_Scatter_{model}.png',
                    dpi=150, bbox_inches='tight')
    plt.close(fig6a)



    # Back-project coefficient-space mode labels onto PCA score space
    valid_rows = np.where(valid_mask)[0]
    mode_per_profile = np.full(len(all_coeffs), -1, dtype=int)
    mode_per_profile[valid_rows] = mode_labels_corner

    # PCA space coloured by coefficient-space mode
    hierarchical_clustering(
        data=scores,
        label=f'PCA_coloured_by_LDC_clustering_{model}',
        save_path=save_data_path,
        feature_labels=[f'PC{k+1}' for k in range(n_components)],
        cutoff=75,
        external_labels=mode_per_profile,
        clustering_metric='mahalanobis',
        method='ward',
        plot_corner=True,
    )

    np.save(save_data_path + f'mode_labels_corner_{model}.npy', mode_labels_corner)
    np.save(save_data_path + f'mode_labels_pca_{model}.npy', mode_per_profile)
    print(f'  Saved mode label arrays to {save_data_path}')