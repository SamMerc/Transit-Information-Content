#############################
########## Purpose ##########
#############################

# Appendix 2 shows a double corner plot of the 4th-order NLLD coefficients [c1, c2, c3, c4]:
#   - Bottom-left triangle : scatter coloured by Teff (inferno).
#   - Top-right triangle   : scatter coloured by wavelength (turbo).
#   - Diagonal             : variable labels (c1, c2, c3, c4), no histograms.
#   - Left extra column    : 1D histograms decomposed by Teff value.
#   - Right extra column   : 1D histograms decomposed by wavelength bin.
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
    teff_vals = T_vals_arr[corner_meta[:, 0]]
    wav_vals  = wavs_ref[corner_meta[:, 3]] / 1e4   # µm

    # ── Cluster colours (tab10, same as Appendix 3) ──────────────────────────
    n_cl           = len(unique_cl)
    cluster_cmap   = plt.cm.get_cmap('tab10', n_cl)
    cluster_colors = [cluster_cmap(c) for c in range(n_cl)]

    # ── Colormaps ─────────────────────────────────────────────────────────────
    # Teff: discrete inferno with exactly one colour per sampled grid point
    teff_unique = np.unique(teff_vals)
    n_teff      = len(teff_unique)
    hw_t        = np.diff(teff_unique) / 2
    teff_bounds = np.concatenate([[teff_unique[0] - hw_t[0]],
                                   teff_unique[:-1] + hw_t,
                                   [teff_unique[-1] + hw_t[-1]]])
    teff_cmap   = plt.cm.get_cmap('inferno', n_teff)
    teff_norm   = mcolors.BoundaryNorm(teff_bounds, ncolors=teff_cmap.N)
    sm_t_disc   = cm.ScalarMappable(cmap=teff_cmap, norm=teff_norm)

    # Wavelength: continuous turbo
    wav_cmap  = plt.cm.turbo
    wav_norm  = mcolors.Normalize(vmin=wav_vals.min(),  vmax=wav_vals.max())

    # ── Plot configuration ────────────────────────────────────────────────────
    ndim   = 4
    labels = [r'$c_1$', r'$c_2$', r'$c_3$', r'$c_4$']
    ranges = [(np.percentile(corner_data[:, i], 0.1),
               np.percentile(corner_data[:, i], 99.9)) for i in range(ndim)]

    # ── Scatter subsamples sorted by colour value for correct Z-ordering ──────
    N    = len(corner_data)
    n_sc = min(N, 40_000)
    rng_t = np.random.default_rng(42)
    idx_t = rng_t.choice(N, size=n_sc, replace=False)
    idx_t = idx_t[np.argsort(teff_vals[idx_t])]
    rng_w = np.random.default_rng(43)
    idx_w = rng_w.choice(N, size=n_sc, replace=False)
    idx_w = idx_w[np.argsort(wav_vals[idx_w])]

    # ── Figure layout ─────────────────────────────────────────────────────────
    # Outer GridSpec: 4 rows × 5 cols
    #   col 0 : Teff colorbar   (narrow)
    #   col 1 : Teff 1D histograms
    #   col 2 : 4×4 corner block  (subdivided below, wspace=0.05)
    #   col 3 : wavelength 1D histograms
    #   col 4 : wavelength colorbar  (narrow)
    # wspace=0.30 gives the wide gap between histograms and the corner block.
    fig = plt.figure(figsize=(16, 9))
    outer_gs = GridSpec(ndim, 5, figure=fig,
                        wspace=0.05, hspace=0.27,
                        width_ratios=[0.10, 1.4, 4.0, 1.4, 0.10])

    # Inner GridSpec: 4×4 within the corner block, with tight wspace
    inner_gs = GridSpecFromSubplotSpec(ndim, ndim,
                                       subplot_spec=outer_gs[:, 2],
                                       wspace=0.05, hspace=0.05)

    # Colorbar axes span all rows
    cax_t = fig.add_subplot(outer_gs[:, 0])
    cax_w = fig.add_subplot(outer_gs[:, 4])

    # Histogram and corner axes
    ax_ht = np.empty(ndim, dtype=object)
    ax_hw = np.empty(ndim, dtype=object)
    ax_c  = np.empty((ndim, ndim), dtype=object)
    for d in range(ndim):
        ax_ht[d] = fig.add_subplot(outer_gs[d, 1])
        ax_hw[d] = fig.add_subplot(outer_gs[d, 3])
    for ir in range(ndim):
        for ic in range(ndim):
            ax_c[ir, ic] = fig.add_subplot(inner_gs[ir, ic])

    # ── Left column: 1D histograms decomposed by Teff ─────────────────────────
    for d in range(ndim):
        a = ax_ht[d]
        for uv in teff_unique:
            m = teff_vals == uv
            a.hist(corner_data[m, d], bins=50, range=ranges[d],
                   alpha=0.55, color=sm_t_disc.to_rgba(uv), density=False,
                   histtype='stepfilled', edgecolor='none')
        a.set_xlim(ranges[d])
        a.set_yticks([])
        for sp in ['top', 'right', 'left']:
            a.spines[sp].set_visible(False)
        a.tick_params(labelsize=7)
        a.set_xlabel(labels[d], fontsize=10)

    # ── Right column: 1D histograms decomposed by wavelength ──────────────────
    n_wb    = 10
    w_edges = np.linspace(wav_vals.min(), wav_vals.max(), n_wb + 1)
    w_ctrs  = 0.5 * (w_edges[:-1] + w_edges[1:])
    for d in range(ndim):
        a = ax_hw[d]
        for ib, ctr in enumerate(w_ctrs):
            m = (wav_vals >= w_edges[ib]) & (wav_vals < w_edges[ib + 1])
            if ib == n_wb - 1:
                m |= (wav_vals == w_edges[ib + 1])
            if not m.any():
                continue
            a.hist(corner_data[m, d], bins=50, range=ranges[d],
                   alpha=0.55, color=wav_cmap(wav_norm(ctr)), density=False,
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
                # Below diagonal — scatter coloured by Teff
                a.scatter(corner_data[idx_t, ic], corner_data[idx_t, ir],
                          c=teff_vals[idx_t], cmap=teff_cmap, norm=teff_norm,
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
                # Above diagonal — scatter coloured by wavelength, mirrored axes
                a.scatter(corner_data[idx_w, ic], corner_data[idx_w, ir],
                          c=wav_vals[idx_w], cmap=wav_cmap, norm=wav_norm,
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
    sm_t_disc.set_array([])
    cb_t = fig.colorbar(sm_t_disc, cax=cax_t)
    cb_t.set_label(r'$T_{\rm eff}$ (K)', fontsize=11)
    cb_t.set_ticks(teff_unique)
    cb_t.set_ticklabels([f'{v:.0f}' for v in teff_unique])
    cb_t.ax.tick_params(labelsize=9)
    cax_t.yaxis.set_ticks_position('left')
    cax_t.yaxis.set_label_position('left')

    sm_w = cm.ScalarMappable(cmap=wav_cmap, norm=wav_norm)
    sm_w.set_array([])
    cb_w = fig.colorbar(sm_w, cax=cax_w)
    cb_w.set_label(r'Wavelength ($\mu$m)', fontsize=11)
    cb_w.ax.tick_params(labelsize=9)

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
    # Derive the horizontal centre and bottom edge of the 4×4 block in figure
    # coordinates from the corner axes positions (available before draw).
    pos_left  = ax_c[0, 0].get_position()
    pos_right = ax_c[0, ndim - 1].get_position()
    pos_bot   = ax_c[ndim - 1, 0].get_position()
    center_x  = (pos_left.x0 + pos_right.x1) / 2
    bottom_y  = pos_bot.y0
    fig.legend(handles=legend_handles, ncols=int((len(clusters_2_show) + 1)//2),
               loc='upper center',
               bbox_to_anchor=(center_x, bottom_y - 0.01),
               bbox_transform=fig.transFigure,
               fontsize=9, framealpha=0.85)

    plt.savefig(paths.figures / "Appendix2.pdf", bbox_inches='tight', dpi=150)
    plt.close(fig)
