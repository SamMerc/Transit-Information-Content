#############################
########## Purpose ##########
#############################

# Figures 2, 3, 4, and 5 require a 4-th order non-linear limb-darkening law for the injection / simulation of the LC.
# Given that we are working with a made up fiducial system, we need to identify the limb-darkening values to use for this.
# In order to do this, we explore all available intensity profiles for a given grid of stellar properties, fit these profiles
# with a 4th order NLLD and perform a clustering algorithm to identify clusters, the overall mode, and the cluster modes. This
# file shows how to the intensity profiles needed for these figures.
#
# This version works exclusively with global (disc-integrated) stellar intensity profiles — no transit
# chord / impact-parameter / planet-size dependence. For that see the Paper2 related files.


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
from lmfit import minimize, Parameters
import gc
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.interpolate import interp1d
import corner
from tqdm import tqdm
from scipy.cluster.hierarchy import (linkage, fcluster,
                                     dendrogram as scipy_dendrogram,
                                     optimal_leaf_ordering)
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import paths


######################################
########## Hyper-parameters ##########
######################################

LD_data_path        = '/Volumes/Ajax/Work/PhD/Research/Transit-Information-Content/LD_simulation'
orig_save_data_path = str(paths.data / "Fig5_Storage") + "/"

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

# ── Profile subsampling ───────────────────────────────────────────────────────
# If True, randomly draw n_subsample_profiles from the valid profiles before
# running PCA, clustering, fitting, etc.  Set to False to use all valid profiles.
subsample_profiles   = True
n_subsample_profiles = 50000
subsample_seed       = 42  # for reproducibility

# If True, the subsampling draw above is importance-weighted by the joint KDE of
# (Teff, logg, [M/H]) fitted to confirmed exoplanet hosts (see HostStar_extract.py),
# so the profiles that feed the clustering reflect the real host-star population
# rather than the uniformly-spaced simulation grid. Set to False to recover the
# original uniform-grid sampling.
weight_by_host_star_kde = True
host_star_kde_path      = str(paths.data / "HostStar_Storage" / "host_star_kde.pkl")

# Seed for the random initial guesses used when fitting each profile's 4th-order NLLD
# coefficients (see the fitting loop below). This determines all_coeffs / corner_data and
# therefore the clustering and chosen modes, so it needs to be fixed for the results to be
# reproducible if this fitting step is ever run standalone in another file.
fit_init_seed = 42

# Whether to plot the dendrogram of the hierarchical clustering step.
plot_dendogram = False

# Distance threshold for cutting the hierarchical clustering tree
CLUSTER_CUTOFF = 100.0  


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

    print('MASKING')

    # ── Count valid profiles per wavelength bin ───────────────────────────
    n_wav_bins      = 10
    wavs_ref        = np.array(gen_dict['stellar_wavelengths'][model][0, 0, 0])
    wav_bin_slices  = np.array_split(np.arange(len(wavs_ref)), n_wav_bins)
    wav_bin_labels  = [f'{wavs_ref[s[0]]/1e4:.2f}-{wavs_ref[s[-1]]/1e4:.2f} μm'
                        for s in wav_bin_slices]

    n_valid_per_wav  = np.zeros(n_wav_bins, dtype=int)
    n_valid_per_Teff = np.zeros(N_star,    dtype=int)
    n_valid_per_logg = np.zeros(N_star,    dtype=int)
    n_valid_per_met  = np.zeros(N_star,    dtype=int)
    n_considered     = 0
    n_valid          = 0

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

                n_valid_this = int(np.sum(mask))
                n_considered += mask.size
                n_valid      += n_valid_this
                for iw_bin, idx in enumerate(wav_bin_slices):
                    n_valid_per_wav[iw_bin] += int(np.sum(mask[idx]))
                n_valid_per_Teff[i] += n_valid_this
                n_valid_per_logg[j] += n_valid_this
                n_valid_per_met[k]  += n_valid_this

    # Total possible per stellar-parameter value:
    # each value appears in N_star² (other two params) × n_wav wavelength slices
    n_wav_total      = len(wavs_ref)
    n_total_per_star = N_star**2 * n_wav_total

    print(f"=== Profile filtering summary ===")
    print(f"Total profiles considered : {n_considered}")
    print(f"Total valid               : {n_valid} ({100 * n_valid / n_considered:.1f} %)")

    print(f"\nPer wavelength region:")
    n_total_per_wav = np.array([len(s) * N_star**3 for s in wav_bin_slices])
    for iw_bin in range(n_wav_bins):
        print(f"  {wav_bin_labels[iw_bin]} : "
                f"{n_valid_per_wav[iw_bin]}/{n_total_per_wav[iw_bin]} valid "
                f"({100 * n_valid_per_wav[iw_bin] / n_total_per_wav[iw_bin]:.1f} %)")

    print(f"\nPer Teff value:")
    for i in range(N_star):
        print(f"  Teff = {T_vals_arr[i]:7.0f} K : "
              f"{n_valid_per_Teff[i]}/{n_total_per_star} valid "
              f"({100 * n_valid_per_Teff[i] / n_total_per_star:.1f} %)")

    print(f"\nPer logg value:")
    for j in range(N_star):
        print(f"  logg = {g_vals_arr[j]:5.2f}    : "
              f"{n_valid_per_logg[j]}/{n_total_per_star} valid "
              f"({100 * n_valid_per_logg[j] / n_total_per_star:.1f} %)")

    print(f"\nPer metallicity value:")
    for k in range(N_star):
        print(f"  [M/H] = {m_vals_arr[k]:+6.2f}  : "
              f"{n_valid_per_met[k]}/{n_total_per_star} valid "
              f"({100 * n_valid_per_met[k] / n_total_per_star:.1f} %)")

    # ── Pass 2 — pre-allocate and fill profile matrix ─────────────────────
    # Metadata columns: [i_Teff, j_logg, k_met, i_wav]
    int_profile = np.empty((n_valid, n_mu_fine), dtype=np.float32)
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
                int_profile[ptr:ptr + n_ok] = norm_prof[mask].astype(np.float32)
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

    # Keep the full, un-subsampled population around so the impact of the KDE
    # weighting on the physical-parameter distribution can be visualised later (Figure 3b).
    meta_array_full = meta_array.copy()

    # ── Optional subsampling ──────────────────────────────────────────────
    sample_weights = None
    if subsample_profiles:
        rng    = np.random.default_rng(subsample_seed)
        n_draw = min(n_subsample_profiles, n_valid)

        if weight_by_host_star_kde:
            with open(host_star_kde_path, 'rb') as f:
                host_star_kde = pickle.load(f)
            Tg, Gg, Mg     = np.meshgrid(T_vals_arr, g_vals_arr, m_vals_arr, indexing='ij')
            grid_weights   = host_star_kde(np.vstack([Tg.ravel(), Gg.ravel(), Mg.ravel()]))
            grid_weights   = grid_weights.reshape(N_star, N_star, N_star)
            sample_weights = grid_weights[meta_array[:, 0], meta_array[:, 1], meta_array[:, 2]]
            sample_weights = sample_weights / sample_weights.sum()
            ess = 1.0 / np.sum(sample_weights ** 2)
            print(f'  KDE weighting: effective sample size {ess:,.0f} '
                  f'(of {n_valid:,} candidate profiles)')

        if n_draw < n_valid:
            idx = np.sort(rng.choice(n_valid, size=n_draw, replace=False, p=sample_weights))
            weight_str = 'KDE-weighted' if weight_by_host_star_kde else 'uniform'
            print(f'\nSUBSAMPLING ({weight_str}): {n_valid} → {n_draw} profiles (seed={subsample_seed})')
        else:
            idx = np.arange(n_valid)
            print(f'\nSUBSAMPLING: {n_valid} profiles (fewer than '
                    f'{n_subsample_profiles}, keeping all)')

        int_profile = int_profile[idx]
        mus_array       = mus_array[idx]
        meta_array      = meta_array[idx]
        n_valid         = n_draw
        if sample_weights is not None:
            sample_weights = sample_weights[idx]

    # ── Figure 1 : Residual plots coloured by parameter ──────────────────────────────
    print('FIGURE 1 - RESIDUAL PLOTTING')
    med   = np.median(int_profile, axis=0)
    resid = int_profile - med

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

    # ── Fit ALL profiles with 4th-order NLLD ──────────────────────────────────
    print('Fitting ALL profiles with 4th-order NLLD')

    n_profs  = int_profile.shape[0]
    all_coeffs = np.zeros((n_profs, 4), dtype=np.float64)
    fit_rng    = np.random.default_rng(fit_init_seed)
    print(f'  Fitting {n_profs} global intensity profiles ...')

    for idx in tqdm(range(n_profs)):
        prof_idx = int_profile[idx]
        mus_idx  = mus_array[idx]

        if np.all(np.abs(prof_idx) < 1e-10):
            all_coeffs[idx] = np.nan
            continue

        params = Parameters()
        for ip in range(4):
            params.add(f'c{ip+1}', value=fit_rng.uniform(0, 1))
        try:
            result = minimize(residual_fn, params, args=(mus_idx, prof_idx))
            all_coeffs[idx] = [result.params[f'c{ic+1}'].value for ic in range(4)]
        except Exception:
            all_coeffs[idx] = np.nan

    # ── Corner data: [c1, c2, c3, c4] ────────────────────────────────────────
    valid_mask  = ~np.any(np.isnan(all_coeffs), axis=1)
    corner_data = all_coeffs[valid_mask]
    corner_profiles = int_profile[valid_mask]
    corner_weights   = sample_weights[valid_mask] if sample_weights is not None else None
    print(f'  Corner plot for {corner_data.shape[0]} valid profiles')

    labels_4d = [r'$c_1$', r'$c_2$', r'$c_3$', r'$c_4$']
    ranges_4d = [
        (np.percentile(corner_data[:, ic], 1),
         np.percentile(corner_data[:, ic], 99))
        for ic in range(4)
    ]
    ndim_4d = 4

    # ── Figure 2a: density corner ─────────────────────────────────────────────
    print('    FIGURE 2a - All coefficients corner plot')
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
    meta_cmaps     = [plt.cm.inferno, plt.cm.cividis, plt.cm.get_cmap('winter')]

    # ── Figures 2b–d : coloured by physical parameters ────────────────────────
    for imeta in range(3):
        col_vals  = meta_col_vals[imeta]
        col_label = meta_col_names[imeta]
        col_key   = meta_col_keys[imeta]
        col_cmap  = meta_cmaps[imeta]
        col_norm  = mcolors.Normalize(vmin=col_vals.min(), vmax=col_vals.max())

        print(f'    FIGURE 2{chr(98 + imeta)} - Corner plot coloured by {col_key}')

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
                            alpha=0.55, color=col_cmap(col_norm(uv)), density=False,
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
                            alpha=0.55, color=col_cmap(col_norm(ctr)), density=False,
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

    # ── Figure 2e: coloured by wavelength ─────────────────────────────────────
    print('    FIGURE 2e - Corner plot coloured by wavelength')
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

    # ── Mode identification in coefficient space ──────────────────────────────
    print('\n=== MODE IDENTIFICATION IN COEFFICIENT SPACE ===')

    mode_labels_corner, _, _ = hierarchical_clustering(
        data=corner_data,
        label=f'LDC_clustering_{model}',
        save_path=save_data_path,
        feature_labels=[r'$c_1$', r'$c_2$', r'$c_3$', r'$c_4$'],
        cutoff=CLUSTER_CUTOFF,
        clustering_metric='mahalanobis',
        method='ward',
    )

    cluster_labels = mode_labels_corner - 1  # 0-indexed
    unique_cl      = np.unique(cluster_labels)

    for m in unique_cl:
        mask_m = cluster_labels == m
        print(
            f'  Mode {m}: n={mask_m.sum():5d}  '
            f'c=[{corner_data[mask_m, 0].mean():.3f}, '
            f'{corner_data[mask_m, 1].mean():.3f}, '
            f'{corner_data[mask_m, 2].mean():.3f}, '
            f'{corner_data[mask_m, 3].mean():.3f}]'
        )

    # Global mode: point in the subsampled dataset closest to the global centroid
    global_centroid = corner_data.mean(axis=0)
    dists_to_global = np.linalg.norm(corner_data - global_centroid, axis=1)
    typical_idx     = int(np.argmin(dists_to_global))

    # Clulster mode profiles: furthest from centroid per cluster
    cluster_mode_indices = []
    for cl in unique_cl:
        mask     = cluster_labels == cl
        members  = corner_data[mask]
        centroid = members.mean(axis=0)
        dists    = np.linalg.norm(members - centroid, axis=1)
        cluster_mode_indices.append(int(np.where(mask)[0][np.argmin(dists)]))

    # ── Figure 3a: NLLD corner scatter coloured by cluster ─────────────────
    print('    FIGURE 3 - NLLD corner scatter')
    n_cl = len(unique_cl)
    cluster_cmap   = plt.cm.get_cmap('tab10', n_cl)
    cluster_colors = [cluster_cmap(c) for c in range(n_cl)]

    special_indices = [typical_idx] + cluster_mode_indices
    special_labels  = ['Typical'] + [f'Mode {c}' for c in range(len(cluster_mode_indices))]
    special_colors  = ['blue'] + ['red'] * len(cluster_mode_indices)

    fig3a, axes3a = plt.subplots(4, 4,
                                    figsize=(3 * 4, 3 * 4))
    fig3a.suptitle(f'NLLD cluster structure — {model}', fontsize=12, y=1.01)

    for row in range(4):
        for col in range(4):
            ax = axes3a[row, col]
            if row == col:
                for ci, cl in enumerate(unique_cl):
                    cmask = cluster_labels == cl
                    ax.hist(corner_data[cmask, row], bins=30, alpha=0.5,
                            color=cluster_colors[ci], label=f'C{cl}',
                            density=False, linewidth=0.8, edgecolor='none')
                ax.set_xlabel(f'PC{row+1}', fontsize=8)
                ax.tick_params(labelsize=7)
                for spine in ['top', 'left', 'right']:
                    ax.spines[spine].set_visible(False)
                ax.set_yticks([])
            elif row > col:
                for ci, cl in enumerate(unique_cl):
                    cmask = cluster_labels == cl
                    ax.scatter(corner_data[cmask, col], corner_data[cmask, row],
                                color=cluster_colors[ci], s=6, alpha=0.5,
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

    last_ax  = axes3a[3, 2]
    handles, lbls = last_ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, lbls):
        if l not in seen:
            seen[l] = h
    fig3a.legend(seen.values(), seen.keys(), loc='upper right',
                    fontsize=8, markerscale=1.5, framealpha=0.8)
    fig3a.tight_layout()
    fig3a.savefig(save_data_path + f'NLLD_Corner_Scatter_{model}.png',
                    dpi=150, bbox_inches='tight')
    plt.close(fig3a)

    # ── Figure 3b: Physical parameter corner scatter coloured by cluster ──────
    print('    FIGURE 3b - Physical parameters corner scatter coloured by NLLD cluster')
    
    # 1. Extract physical parameter values matching the valid profiles
    T_eff_vals = T_vals_arr[corner_meta[:, 0]]
    logg_vals  = g_vals_arr[corner_meta[:, 1]]
    MH_vals    = m_vals_arr[corner_meta[:, 2]]

    # Stack into a 2D array: (N_profiles, 4)
    phys_data = np.column_stack([T_eff_vals, logg_vals, MH_vals, corner_wav_um])

    # Same physical parameters for the FULL, un-subsampled population (i.e. before the
    # KDE-weighted draw), so the impact of the weighting on the original sample can be
    # visualised alongside the post-weighting, clustered data.
    full_wav_um    = wavs_ref[meta_array_full[:, 3]] / 1e4
    phys_data_full = np.column_stack([
        T_vals_arr[meta_array_full[:, 0]],
        g_vals_arr[meta_array_full[:, 1]],
        m_vals_arr[meta_array_full[:, 2]],
        full_wav_um,
    ])

    phys_labels = [r'$T_{\rm eff}$ (K)', r'$\log\,g$', r'[M/H]', r'Wavelength ($\mu$m)']
    # Ranges span the FULL population so any parts of parameter space the weighting
    # thinned out or removed remain visible rather than being cropped out of view.
    phys_ranges = [
        (phys_data_full[:, d].min(), phys_data_full[:, d].max())
        for d in range(4)
    ]
    ndim_phys = 4

    # Teff / logg / [M/H] only take N_star discrete grid values, so equal-width bins can
    # straddle them and alias into a checkerboard of empty/doubly-populated bins. Bin edges
    # centred on each grid value avoid this; the wavelength axis is continuous and keeps
    # regular bins.
    def _grid_bin_edges(vals):
        step = vals[1] - vals[0]
        return np.concatenate([vals - step / 2, [vals[-1] + step / 2]])

    phys_bin_edges = [
        _grid_bin_edges(T_vals_arr),
        _grid_bin_edges(g_vals_arr),
        _grid_bin_edges(m_vals_arr),
        np.linspace(phys_ranges[3][0], phys_ranges[3][1], 21),
    ]

    # 2. Setup the base corner plot
    fig_phys = corner.corner(
        phys_data, labels=phys_labels, range=phys_ranges, bins=20, smooth1d=1.0,
        plot_datapoints=False, plot_density=False, fill_contours=False, no_fill_contours=True,
        hist_kwargs={'color': 'gray', 'linewidth': 1.2, 'alpha': 0.5},
        label_kwargs={'fontsize': 13}, show_titles=False,
        contour_kwargs={'colors': 'none'},
    )
    axes_phys = np.array(fig_phys.axes).reshape(ndim_phys, ndim_phys)

    # -------------------------------------------------------------------------
    # 3. PRE-PROCESS: Determine plot order (Z-Ordering)
    # We sort clusters from largest to smallest. By plotting the largest first,
    # they stay in the background, allowing the smaller clusters to be visible on top.
    # -------------------------------------------------------------------------
    cluster_counts = {cl: np.sum(cluster_labels == cl) for cl in unique_cl}
    # Sort clusters descending by size
    clusters_sorted_by_size = sorted(unique_cl, key=lambda c: cluster_counts[c], reverse=True)

    # 4. Populate subplots
    for row in range(ndim_phys):
        for col in range(ndim_phys):
            ax = axes_phys[row, col]
            
            if row == col:
                # Diagonal: pre-weighting population (reference) + 1D histograms per cluster
                ax.clear()

                # Reference: full, un-subsampled candidate pool (i.e. before KDE weighting).
                ax.hist(phys_data_full[:, row], bins=phys_bin_edges[row],
                        color='black', density=True, histtype='step',
                        linewidth=1.5, linestyle='--')

                # For histograms, order doesn't matter as much, but we'll stick to the original unique_cl order
                for ci, cl in enumerate(unique_cl):
                    cmask = cluster_labels == cl
                    if cmask.sum() == 0:
                        continue
                    
                    # density=True normalizes the area under the curve to 1.
                    # This ensures the 435-point cluster is visible alongside the 11k-point cluster.
                    ax.hist(phys_data[cmask, row], bins=phys_bin_edges[row],
                            alpha=0.4, color=cluster_colors[ci], density=True,
                            histtype='stepfilled', edgecolor='none')
                
                ax.set_xlim(phys_ranges[row])
                ax.set_yticks([])
                for spine in ['top', 'left', 'right']:
                    ax.spines[spine].set_visible(False)
                if row == ndim_phys - 1:
                    ax.set_xlabel(phys_labels[row], fontsize=13)
                    
            elif row > col:
                # Off-diagonal: pre-weighting population as a grayscale 2D-histogram backdrop,
                # with the post-weighting sample scattered on top, coloured by cluster.
                h2d, xedges, yedges = np.histogram2d(
                    phys_data_full[:, col], phys_data_full[:, row],
                    bins=[phys_bin_edges[col], phys_bin_edges[row]],
                )
                ax.pcolormesh(xedges, yedges, h2d.T, cmap='Greys', alpha=0.6,
                              shading='auto', rasterized=True)

                # We iterate through the SORTED list so largest goes first (bottom layer)
                for cl in clusters_sorted_by_size:
                    ci = list(unique_cl).index(cl) # Get original index for correct color
                    cmask = cluster_labels == cl
                    
                    # Notice the much smaller size (s=2.0) and lower alpha (0.15) 
                    # to handle the density of 11,000+ points overlapping.
                    ax.scatter(phys_data[cmask, col], phys_data[cmask, row],
                               color=cluster_colors[ci], s=2.0, alpha=0.15,
                               linewidths=0, rasterized=True)
                
                if col == 0:
                    ax.set_ylabel(phys_labels[row], fontsize=13)
                if row == ndim_phys - 1:
                    ax.set_xlabel(phys_labels[col], fontsize=13)
                
                ax.tick_params(labelsize=8)
                ax.grid(True, alpha=0.15)

    # 5. Add custom legend mapping colors to clusters
    handles_phys = [
        plt.Line2D([0], [0], color='black', linestyle='--', linewidth=1.5,
                   label='Before weighting (uniform grid)'),
    ] + [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=cluster_colors[ci],
                   markersize=8, label=f'Cluster {cl}')
        for ci, cl in enumerate(unique_cl)
    ]
    fig_phys.legend(handles=handles_phys, loc='upper right', fontsize=12,
                    framealpha=0.85, title='NLLD Cluster', title_fontsize=13,
                    bbox_to_anchor=(0.95, 0.95))

    fig_phys.suptitle(f'Physical parameters — before vs. after host-star KDE weighting — {model}',
                      fontsize=14, y=1.02)
    fig_phys.savefig(save_data_path + f'Corner_PhysParams_byCluster_{model}.png',
                     dpi=150, bbox_inches='tight')
    plt.close(fig_phys)

    # ── Generate profiles ────────────────────────────────────────
    # Global mode: profile closest to its own cluster centroid across ALL clusters
    typical_profile = corner_profiles[typical_idx]
    typical_coeff   = corner_data[typical_idx]
    typical_mu      = mus_array[typical_idx]

    # Per-cluster mode: profile closest to each cluster's centroid
    cluster_mode_indices = []
    for cl in unique_cl:
        mask     = cluster_labels == cl
        members  = corner_data[mask]
        centroid = members.mean(axis=0)
        dists    = np.linalg.norm(members - centroid, axis=1)
        cluster_mode_indices.append(int(np.where(mask)[0][np.argmin(dists)]))

    cluster_mode_profiles = [corner_profiles[cidx] for cidx in cluster_mode_indices]
    cluster_mode_coeffs   = [corner_data[cidx]     for cidx in cluster_mode_indices]
    cluster_mode_mus      = [mus_array[cidx]        for cidx in cluster_mode_indices]

    # ── Figure 4: NLLD curves for overall mode and per-cluster mode profiles ──
    print('    FIGURE 4 - NLLD curves for overall mode and per-cluster mode profiles')

    # Bundle specials: (label, mu array, raw profile, coeff vector, colour, linewidth)
    special_styles = [
        ('Overall mode', typical_mu, typical_profile, typical_coeff, 'orange', 2.5),
    ] + [
        (f'Cluster {cl} mode', mu, cp, cc, cluster_colors[ci], 1.8)
        for ci, (cl, mu, cp, cc) in enumerate(
            zip(unique_cl, cluster_mode_mus, cluster_mode_profiles, cluster_mode_coeffs))
    ]

    # ── Panel layout: left = NLLD curves, right = residuals vs overall mode ──
    fig4, ax4 = plt.subplots(
        n_cl + 1, 2, figsize=(10, 4 * (n_cl + 1)),
        gridspec_kw={'hspace': 0.05, 'wspace': 0.3},
        sharex=True,
    )

    # Pre-compute the overall mode NLLD curve for residuals in cluster rows
    typical_curve = fourNLLD(typical_mu, typical_coeff)

    for plot_idx, (sp_label, mu_plot, prof, coeffs, col, lw) in enumerate(special_styles):
        curve = fourNLLD(mu_plot, coeffs)

        # Left panel: raw intensity profile (solid) + NLLD fit (dashed)
        ax4[plot_idx, 0].plot(mu_plot, curve, '--', color='black', linewidth=lw, zorder=2)
        ax4[plot_idx, 0].plot(mu_plot, prof,  '-',  color=col,     linewidth=lw, zorder=1)
        ax4[plot_idx, 0].set_ylabel('Normalised intensity $I(\\mu)/I(1)$', fontsize=12)
        ax4[plot_idx, 0].set_title(sp_label, fontsize=11, pad=3)
        ax4[plot_idx, 0].grid(True)

        # Right panel: residuals
        resid = 100 * (curve - prof) / prof
        ax4[plot_idx, 1].plot(mu_plot, resid, '--', color=col, linewidth=lw, label=sp_label)
        ax4[plot_idx, 1].axhline(0, color='black', linestyle='-', linewidth=1.2, alpha=0.4)
        ax4[plot_idx, 1].grid(True)

    ax4[-1, 0].set_xlabel('$\\mu = \\cos(\\theta)$', fontsize=12)
    ax4[-1, 1].set_xlabel('$\\mu = \\cos(\\theta)$', fontsize=12)

    fig4.savefig(save_data_path + f'4thOrderNLLD_Modes_{model}.pdf',
                 dpi=150, bbox_inches='tight')
    plt.close(fig4)

    # Print a summary table of all coefficients for easy copy-paste
    print(f'\n  {"Profile":<14}  {"c1":>8}  {"c2":>8}  {"c3":>8}  {"c4":>8}')
    print(f'  {"-"*54}')
    for sp_label, _mu, _prof, coeffs, _col, _lw in special_styles:
        print(f'  {sp_label:<20}  '
              f'{coeffs[0]:>8.4f}  {coeffs[1]:>8.4f}  '
              f'{coeffs[2]:>8.4f}  {coeffs[3]:>8.4f}')
        
    results_file = save_data_path + 'results.npz'
    save_kwargs = dict(
        corner_data           = corner_data,
        corner_meta           = corner_meta,
        cluster_labels        = cluster_labels,
        unique_cl             = unique_cl,
        T_vals_arr            = T_vals_arr,
        g_vals_arr            = g_vals_arr,
        m_vals_arr            = m_vals_arr,
        wavs_ref              = wavs_ref,
        typical_idx           = np.array(typical_idx),
        cluster_mode_indices  = np.array(cluster_mode_indices),
        typical_mu            = typical_mu,
        typical_profile       = typical_profile,
        typical_coeff         = typical_coeff,
        cluster_mode_mus      = cluster_mode_mus,
        cluster_mode_profiles = cluster_mode_profiles,
        cluster_mode_coeffs   = cluster_mode_coeffs,
    )
    if corner_weights is not None:
        save_kwargs['corner_weights'] = corner_weights
    np.savez(results_file, **save_kwargs)
    print(f'  Saved results → {results_file}')