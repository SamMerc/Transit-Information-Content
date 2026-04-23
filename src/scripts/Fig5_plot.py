#############################
########## Purpose ##########
#############################

# Figures 2, 3, and 4 require a 4-th order non-linear limb-darkening law for the injection / simulation of the LC.
# Given that we are working with a made up fiducial system, we need to identify the limb-darkening values to use for this.
# In order to do this, we explore all available intensity profiles for a given grid of stellar properties, fit these profiles
# with a 4th order NLLD and perform a clustering algorithm to identify clusters, the overall mode, and the cluster modes. This
# file plots the intensity profiles selected for the analyses in figures 2, 3, and 4. To see how these intensity profiles are
# generated see Fig2_helper.py
#
# This version works exclusively with global (disc-integrated) stellar intensity profiles — no transit
# chord / impact-parameter / planet-size dependence. For that see the Paper2 related files.


######################################
########## Import libraries ##########
######################################

import numpy as np
import matplotlib
import os
import paths
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle
from lmfit import minimize, Parameters
from tqdm import tqdm
from scipy.cluster.hierarchy import (linkage, fcluster,
                                     dendrogram as scipy_dendrogram,
                                     optimal_leaf_ordering)
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist


######################################
########## Hyper-parameters ##########
######################################

input_save_path = str(paths.data / "Fig5_Storage") + "/"

models = ['mps1']  # ['phoenix','kurucz', 'stagger', 'mps1', 'mps2']

# Number of points in the grid of stellar parameters (Teff, logg, metallicity).
N_star = 10

# Number of mu values to interpolate to — set to EJ16 value.
n_mu_fine = 100

# ── Profile subsampling ───────────────────────────────────────────────────────
# If True, randomly draw n_subsample_profiles from the valid profiles before
# running clustering, fitting, etc.  Set to False to use all valid profiles.
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

# ─────────────────────────────────────────────────────────────────────────────
# Iterate over all stellar models
# ─────────────────────────────────────────────────────────────────────────────
for model in models:

    save_data_path = input_save_path + f'{model}/'

    with open(save_data_path + 'data.pkl', 'rb') as f:
        gen_dict = pickle.load(f)

    print('MASKING')

    # ── Count valid profiles per wavelength bin ───────────────────────────
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

    print(f"=== Profile filtering summary ===")
    print(f"Total profiles considered : {n_considered}")
    print(f"Total valid               : {n_valid} ({100 * n_valid / n_considered:.1f} %)")

    # ── Pass 2 — pre-allocate and fill profile matrix ─────────────────────
    # Metadata columns: [i_Teff, j_logg, k_met, i_wav]
    int_profile = np.empty((n_valid, n_mu_fine), dtype=np.float32)
    mus_array       = np.empty((n_valid, n_mu_fine), dtype=np.float32)
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

                int_profile[ptr:ptr + n_ok] = norm_prof[mask].astype(np.float32)
                mus_array[ptr:ptr + n_ok]       = np.broadcast_to(mus, (n_ok, n_mu_fine)).astype(np.float32)

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
        int_profile = int_profile[idx]
        mus_array       = mus_array[idx]
        n_valid         = n_draw

    # ── Fit ALL profiles with 4th-order NLLD ──────────────────────────────────
    print('Fitting ALL profiles with 4th-order NLLD')

    n_profs  = int_profile.shape[0]
    all_coeffs = np.zeros((n_profs, 4), dtype=np.float64)
    print(f'  Fitting {n_profs} global intensity profiles ...')

    for idx in tqdm(range(n_profs)):
        prof_idx = int_profile[idx]
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

    # ── Corner data: [c1, c2, c3, c4] ────────────────────────────────────────
    valid_mask  = ~np.any(np.isnan(all_coeffs), axis=1)
    corner_data = all_coeffs[valid_mask]
    corner_profiles = int_profile[valid_mask]
    print(f'  Corner plot for {corner_data.shape[0]} valid profiles')

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

    # Clulster mode profiles: furthest from centroid per cluster
    cluster_mode_indices = []
    for cl in unique_cl:
        mask     = cluster_labels == cl
        members  = corner_data[mask]
        centroid = members.mean(axis=0)
        dists    = np.linalg.norm(members - centroid, axis=1)
        cluster_mode_indices.append(int(np.where(mask)[0][np.argmin(dists)]))

    n_cl = len(unique_cl)
    cluster_cmap   = plt.cm.get_cmap('tab10', n_cl)
    cluster_colors = [cluster_cmap(c) for c in range(n_cl)]

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

    # ── Figure 5: NLLD curves for overall mode and per-cluster mode profiles ──
    print('    FIGURE 5 - NLLD curves for overall mode and per-cluster mode profiles')

    # Bundle specials: (label, mu array, raw profile, coeff vector, colour, linewidth)
    special_styles = [
        ('Overall mode', typical_mu, typical_profile, typical_coeff, 'orange', 2.5),
    ] + [
        (f'Cluster {cl} mode', mu, cp, cc, cluster_colors[ci], 1.8)
        for ci, (cl, mu, cp, cc) in enumerate(
            zip(unique_cl, cluster_mode_mus, cluster_mode_profiles, cluster_mode_coeffs))
    ]

    # ── Panel layout: left = NLLD curves, right = residuals vs overall mode ──
    fig5, ax5 = plt.subplots(
        2, n_cl + 1, figsize=(4 * (n_cl + 1), 10),
        gridspec_kw={'wspace': 0.1, 'hspace': 0.1},
        sharex=True, sharey='row',
    )

    # Pre-compute the overall mode NLLD curve for residuals in cluster rows
    typical_curve = fourNLLD(typical_mu, typical_coeff)

    for plot_idx, (sp_label, mu_plot, prof, coeffs, col, lw) in enumerate(special_styles):
        curve = fourNLLD(mu_plot, coeffs)

        # Left panel: raw intensity profile (solid) + NLLD fit (dashed)
        ax5[0, plot_idx].plot(mu_plot, curve, '--', color='black', linewidth=lw, zorder=2)
        ax5[0, plot_idx].plot(mu_plot, prof,  '-',  color=col,     linewidth=lw, zorder=1)
        ax5[0, plot_idx].set_title(sp_label, fontsize=11, pad=3)
        ax5[0, plot_idx].grid(True)

        # Right panel: residuals
        resid = 100 * (curve - prof) / prof
        ax5[1, plot_idx].plot(mu_plot, resid, '--', color=col, linewidth=lw, label=sp_label)
        ax5[1, plot_idx].axhline(0, color='black', linestyle='-', linewidth=1.2, alpha=0.4)
        ax5[1, plot_idx].grid(True)

        ax5[1, plot_idx].set_xlabel('$\\mu = \\cos(\\theta)$', fontsize=12)

    ax5[0, -1].set_ylabel('Normalised intensity $I(\\mu)/I(1)$', fontsize=12)
    ax5[1, -1].set_ylabel('Normalised intensity $I(\\mu)/I(1)$', fontsize=12)

    # fig5.savefig(output_save_path + f'4thOrderNLLD_Modes_{model}.pdf',
    #              dpi=150, bbox_inches='tight')
    plt.savefig(paths.figures / "Fig5.pdf", bbox_inches="tight")

    # Print a summary table of all coefficients for easy copy-paste
    print(f'\n  {"Profile":<14}  {"c1":>8}  {"c2":>8}  {"c3":>8}  {"c4":>8}')
    print(f'  {"-"*54}')
    for sp_label, _mu, _prof, coeffs, _col, _lw in special_styles:
        print(f'  {sp_label:<20}  '
              f'{coeffs[0]:>8.4f}  {coeffs[1]:>8.4f}  '
              f'{coeffs[2]:>8.4f}  {coeffs[3]:>8.4f}')