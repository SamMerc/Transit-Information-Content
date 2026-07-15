#############################
########## Purpose ##########
#############################

# Appendix 5 shows a double corner plot of the 4th-order NLLD coefficients [c1, c2, c3, c4]:
#   - Bottom-left triangle : scatter coloured by log g (viridis).
#   - Top-right triangle   : scatter coloured by metallicity (RdBu_r).
#   - Diagonal             : variable labels (c1, c2, c3, c4), no histograms.
#   - Left extra column    : 1D histograms decomposed by log g value.
#   - Right extra column   : 1D histograms decomposed by metallicity value.
# Data produced by Fig5_run.py and downloaded from Zenodo as results.npz.


######################################
########## Import libraries ##########
######################################

import numpy as np
import matplotlib
import paths
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D


######################################
########## Hyper-parameters ##########
######################################

input_save_path = str(paths.data / "Fig5_Storage") + "/"
models = ['mps1']  # ['phoenix','kurucz', 'stagger', 'mps1', 'mps2']
clusters_2_show = [0, 2, 3, 4, 7] # clusters to highlight with diamonds in the corner plot

################################
########## Code block ##########
################################

for model in models:

    # ── Load pre-computed results ─────────────────────────────────────────────
    res = np.load(input_save_path + f'{model}/results.npz', allow_pickle=False)

    corner_data          = res['corner_data']           # (N, 4)  NLLD coefficients
    corner_meta          = res['corner_meta']           # (N, 4)  [i_Teff, j_logg, k_met, i_wav]
    T_vals_arr           = res['T_vals_arr']
    g_vals_arr           = res['g_vals_arr']
    m_vals_arr           = res['m_vals_arr']
    wavs_ref             = res['wavs_ref']
    cluster_labels       = res['cluster_labels']        # (N,)   0-indexed cluster IDs
    unique_cl            = res['unique_cl']
    typical_idx          = int(res['typical_idx'])
    cluster_mode_indices = res['cluster_mode_indices']  # (n_cl,) indices of cluster modes

    # ── Physical parameter arrays ─────────────────────────────────────────────
    logg_vals = g_vals_arr[corner_meta[:, 1]]
    met_vals  = m_vals_arr[corner_meta[:, 2]]

    # ── Cluster colours (tab10, same as Appendix 3) ──────────────────────────
    n_cl           = len(unique_cl)
    cluster_cmap   = plt.cm.get_cmap('tab10', n_cl)
    cluster_colors = [cluster_cmap(c) for c in range(n_cl)]

    # ── Colormaps ─────────────────────────────────────────────────────────────
    # log g: discrete viridis with exactly one colour per sampled grid point
    logg_unique = np.unique(logg_vals)
    n_logg      = len(logg_unique)
    hw_g        = np.diff(logg_unique) / 2
    logg_bounds = np.concatenate([[logg_unique[0] - hw_g[0]],
                                   logg_unique[:-1] + hw_g,
                                   [logg_unique[-1] + hw_g[-1]]])
    logg_cmap   = plt.cm.get_cmap('viridis', n_logg)
    logg_norm   = mcolors.BoundaryNorm(logg_bounds, ncolors=logg_cmap.N)
    sm_g_disc   = cm.ScalarMappable(cmap=logg_cmap, norm=logg_norm)

    # Metallicity: discrete RdBu_r with exactly one colour per sampled grid point
    met_unique = np.unique(met_vals)
    n_met      = len(met_unique)
    hw_m       = np.diff(met_unique) / 2
    met_bounds = np.concatenate([[met_unique[0] - hw_m[0]],
                                  met_unique[:-1] + hw_m,
                                  [met_unique[-1] + hw_m[-1]]])
    met_cmap   = plt.cm.get_cmap('winter', n_met)
    met_norm   = mcolors.BoundaryNorm(met_bounds, ncolors=met_cmap.N)
    sm_m_disc  = cm.ScalarMappable(cmap=met_cmap, norm=met_norm)

    # ── Plot configuration ────────────────────────────────────────────────────
    ndim   = 4
    labels = [r'$c_1$', r'$c_2$', r'$c_3$', r'$c_4$']
    ranges = [(np.percentile(corner_data[:, i], 0.1),
               np.percentile(corner_data[:, i], 99.9)) for i in range(ndim)]

    # ── Scatter subsamples sorted by colour value for correct Z-ordering ──────
    N    = len(corner_data)
    n_sc = min(N, 40_000)
    rng_g = np.random.default_rng(42)
    idx_g = rng_g.choice(N, size=n_sc, replace=False)
    idx_g = idx_g[np.argsort(logg_vals[idx_g])]
    rng_m = np.random.default_rng(43)
    idx_m = rng_m.choice(N, size=n_sc, replace=False)
    idx_m = idx_m[np.argsort(met_vals[idx_m])]

    # ── Figure layout ─────────────────────────────────────────────────────────
    # Outer GridSpec: 4 rows × 5 cols
    #   col 0 : log g colorbar   (narrow)
    #   col 1 : log g 1D histograms
    #   col 2 : 4×4 corner block  (subdivided below, wspace=0.05)
    #   col 3 : metallicity 1D histograms
    #   col 4 : metallicity colorbar  (narrow)
    fig = plt.figure(figsize=(16, 9))
    outer_gs = GridSpec(ndim, 5, figure=fig,
                        wspace=0.05, hspace=0.27,
                        width_ratios=[0.10, 1.4, 4.0, 1.4, 0.10])

    # Inner GridSpec: 4×4 within the corner block, with tight wspace
    inner_gs = GridSpecFromSubplotSpec(ndim, ndim,
                                       subplot_spec=outer_gs[:, 2],
                                       wspace=0.05, hspace=0.05)

    # Colorbar axes span all rows
    cax_g = fig.add_subplot(outer_gs[:, 0])
    cax_m = fig.add_subplot(outer_gs[:, 4])

    # Histogram and corner axes
    ax_hg = np.empty(ndim, dtype=object)
    ax_hm = np.empty(ndim, dtype=object)
    ax_c  = np.empty((ndim, ndim), dtype=object)
    for d in range(ndim):
        ax_hg[d] = fig.add_subplot(outer_gs[d, 1])
        ax_hm[d] = fig.add_subplot(outer_gs[d, 3])
    for ir in range(ndim):
        for ic in range(ndim):
            ax_c[ir, ic] = fig.add_subplot(inner_gs[ir, ic])

    # ── Left column: 1D histograms decomposed by log g ───────────────────────
    for d in range(ndim):
        a = ax_hg[d]
        for uv in logg_unique:
            m = logg_vals == uv
            a.hist(corner_data[m, d], bins=50, range=ranges[d],
                   alpha=0.55, color=sm_g_disc.to_rgba(uv), density=False,
                   histtype='stepfilled', edgecolor='none')
        a.set_xlim(ranges[d])
        a.set_yticks([])
        for sp in ['top', 'right', 'left']:
            a.spines[sp].set_visible(False)
        a.tick_params(labelsize=7)
        a.set_xlabel(labels[d], fontsize=10)

    # ── Right column: 1D histograms decomposed by metallicity ────────────────
    for d in range(ndim):
        a = ax_hm[d]
        for uv in met_unique:
            m = met_vals == uv
            a.hist(corner_data[m, d], bins=50, range=ranges[d],
                   alpha=0.55, color=sm_m_disc.to_rgba(uv), density=False,
                   histtype='stepfilled', edgecolor='none')
        a.set_xlim(ranges[d])
        a.set_yticks([])
        for sp in ['top', 'left', 'right']:
            a.spines[sp].set_visible(False)
        a.tick_params(labelsize=7)
        a.set_xlabel(labels[d], fontsize=10)

    # ── Inner 4×4 double corner ───────────────────────────────────────────────
    for ir in range(ndim):
        for ic in range(ndim):
            a = ax_c[ir, ic]

            if ir == ic:
                # Diagonal — variable name only, no frame
                a.axis('off')
                a.text(0.5, 0.5, labels[ir],
                       ha='center', va='center',
                       fontsize=17, fontweight='bold',
                       transform=a.transAxes)

            elif ir > ic:
                # Below diagonal — scatter coloured by log g
                a.scatter(corner_data[idx_g, ic], corner_data[idx_g, ir],
                          c=logg_vals[idx_g], cmap=logg_cmap, norm=logg_norm,
                          s=1.5, alpha=0.35, linewidths=0, rasterized=True)
                # Cluster mode diamonds for clusters 1, 3, 4
                for ci, cl in enumerate(unique_cl):
                    if cl in clusters_2_show:
                        cidx = cluster_mode_indices[ci]
                        a.scatter(corner_data[cidx, ic], corner_data[cidx, ir],
                                  color=cluster_colors[ci], s=45, marker='D', zorder=9,
                                  edgecolors='black', linewidths=0.7)
                # Overall mode: orange star on top
                a.scatter(corner_data[typical_idx, ic], corner_data[typical_idx, ir],
                          color='orange', s=160, marker='*', zorder=10,
                          edgecolors='black', linewidths=0.7)
                a.set_xlim(ranges[ic])
                a.set_ylim(ranges[ir])
                a.grid(True, alpha=0.15)
                a.tick_params(labelsize=7)
                # Ticks on inner sides (right + top, toward the diagonal)
                a.yaxis.tick_right()
                a.xaxis.tick_top()
                # Labels only on cells immediately adjacent to the diagonal
                if ic != ir - 1:
                    a.tick_params(labelright=False, labeltop=False)

            else:   # ir < ic
                # Above diagonal — scatter coloured by metallicity, mirrored axes
                a.scatter(corner_data[idx_m, ic], corner_data[idx_m, ir],
                          c=met_vals[idx_m], cmap=met_cmap, norm=met_norm,
                          s=1.5, alpha=0.35, linewidths=0, rasterized=True)
                # Cluster mode diamonds for clusters 1, 3, 4
                for ci, cl in enumerate(unique_cl):
                    if cl in clusters_2_show:
                        cidx = cluster_mode_indices[ci]
                        a.scatter(corner_data[cidx, ic], corner_data[cidx, ir],
                                  color=cluster_colors[ci], s=45, marker='D', zorder=9,
                                  edgecolors='black', linewidths=0.7)
                # Overall mode: orange star on top
                a.scatter(corner_data[typical_idx, ic], corner_data[typical_idx, ir],
                          color='orange', s=160, marker='*', zorder=10,
                          edgecolors='black', linewidths=0.7)
                a.set_xlim(ranges[ic])
                a.set_ylim(ranges[ir])
                a.grid(True, alpha=0.15)
                a.tick_params(labelsize=7)
                # Ticks on inner sides (left + bottom, toward the diagonal)
                # Labels only on cells immediately adjacent to the diagonal
                if ic != ir + 1:
                    a.tick_params(labelleft=False, labelbottom=False)

    # ── Colorbars ─────────────────────────────────────────────────────────────
    sm_g_disc.set_array([])
    cb_g = fig.colorbar(sm_g_disc, cax=cax_g)
    cb_g.set_label(r'$\log g$', fontsize=11)
    cb_g.set_ticks(logg_unique)
    cb_g.set_ticklabels([f'{v:.1f}' for v in logg_unique])
    cb_g.ax.tick_params(labelsize=9)
    cax_g.yaxis.set_ticks_position('left')
    cax_g.yaxis.set_label_position('left')

    sm_m_disc.set_array([])
    cb_m = fig.colorbar(sm_m_disc, cax=cax_m)
    cb_m.set_label(r'$[\rm M/H]$', fontsize=11)
    cb_m.set_ticks(met_unique)
    cb_m.set_ticklabels([f'{v:+.1f}' for v in met_unique])
    cb_m.ax.tick_params(labelsize=9)

    # ── Cluster legend centred below the 4×4 corner block ────────────────────
    legend_handles = (
        [Line2D([0], [0], marker='D', color='w',
                markerfacecolor=cluster_colors[ci],
                markeredgecolor='black', markeredgewidth=0.7,
                markersize=8, label=f'Cluster {cl}')
         for ci, cl in enumerate(unique_cl) if cl in clusters_2_show]
        + [Line2D([0], [0], marker='*', color='w', markerfacecolor='orange',
                  markeredgecolor='black', markeredgewidth=0.7,
                  markersize=13, label='Global mode')]
    )
    pos_left  = ax_c[0, 0].get_position()
    pos_right = ax_c[0, ndim - 1].get_position()
    pos_bot   = ax_c[ndim - 1, 0].get_position()
    center_x  = (pos_left.x0 + pos_right.x1) / 2
    bottom_y  = pos_bot.y0
    fig.legend(handles=legend_handles, ncols=3,
               loc='upper center',
               bbox_to_anchor=(center_x, bottom_y - 0.01),
               bbox_transform=fig.transFigure,
               fontsize=9, framealpha=0.85)

    plt.savefig(paths.figures / "Appendix5.pdf", bbox_inches='tight', dpi=150)
    plt.close(fig)
