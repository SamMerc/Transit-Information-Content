#############################
########## Purpose ##########
#############################

# Appendix 3 shows a lower-triangle corner plot of the 4th-order NLLD
# coefficients [c1, c2, c3, c4] coloured by hierarchical cluster membership.
# The overall mode (closest profile to the global centroid) is marked with an
# orange star and each per-cluster mode with a diamond in the cluster colour.
# Data produced by Fig3_run.py and downloaded from Zenodo as results.npz.


######################################
########## Import libraries ##########
######################################

import numpy as np
import matplotlib
import paths
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


######################################
########## Hyper-parameters ##########
######################################

input_save_path = str(paths.data / "Fig3_Storage") + "/"
models = ['mps1']
clusters_2_show = [0, 2, 3, 4, 7] # clusters to highlight with diamonds in the corner plot

################################
########## Code block ##########
################################

for model in models:

    # ── Load pre-computed results ─────────────────────────────────────────────
    res = np.load(input_save_path + f'{model}/results.npz', allow_pickle=False)

    corner_data          = res['corner_data']           # (N, 4) NLLD coefficients
    cluster_labels       = res['cluster_labels']        # (N,)   0-indexed cluster IDs
    unique_cl            = res['unique_cl']
    typical_idx          = int(res['typical_idx'])
    cluster_mode_indices = res['cluster_mode_indices']  # (n_cl,) indices of cluster modes

    n_cl           = len(unique_cl)
    cluster_cmap   = matplotlib.colormaps['tab10'].resampled(n_cl)
    cluster_colors = [cluster_cmap(c) for c in range(n_cl)]

    # ── Plot configuration ────────────────────────────────────────────────────
    ndim   = 4
    labels = [r'$c_1$', r'$c_2$', r'$c_3$', r'$c_4$']
    ranges = [(np.percentile(corner_data[:, i], 0.1),
               np.percentile(corner_data[:, i], 99.9)) for i in range(ndim)]

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(ndim, ndim, figsize=(10, 9),
                             gridspec_kw={'wspace': 0.05, 'hspace': 0.05})

    for row in range(ndim):
        for col in range(ndim):
            ax = axes[row, col]

            if row == col:
                # Diagonal: 1D histograms per cluster
                for ci, cl in enumerate(unique_cl):
                    mask = cluster_labels == cl
                    ax.hist(corner_data[mask, row], bins=40, range=ranges[row],
                            alpha=0.55, color=cluster_colors[ci], density=False,
                            histtype='stepfilled', edgecolor='none')
                ax.set_xlim(ranges[row])
                ax.set_yticks([])
                for sp in ['top', 'left', 'right']:
                    ax.spines[sp].set_visible(False)

                # Dashed vertical lines for cluster modes (clusters 1, 3, 4)
                for ci, cl in enumerate(unique_cl):
                    if cl in clusters_2_show:
                        cidx = cluster_mode_indices[ci]
                        ax.axvline(corner_data[cidx, row],
                                   color=cluster_colors[ci],
                                   linestyle='--', linewidth=1.3, alpha=0.9)

            elif row > col:
                # Below diagonal: scatter coloured by cluster, large→small z-order
                cluster_counts = {cl: int(np.sum(cluster_labels == cl)) for cl in unique_cl}
                for cl in sorted(unique_cl, key=lambda c: cluster_counts[c], reverse=True):
                    ci   = list(unique_cl).index(cl)
                    mask = cluster_labels == cl
                    ax.scatter(corner_data[mask, col], corner_data[mask, row],
                               color=cluster_colors[ci], s=2, alpha=0.30,
                               linewidths=0, rasterized=True, zorder=1)

                # Per-cluster modes: diamonds in cluster colour
                for ci, cidx in enumerate(cluster_mode_indices):
                    ax.scatter(corner_data[cidx, col], corner_data[cidx, row],
                               color=cluster_colors[ci], s=45, marker='D', zorder=9,
                               edgecolors='black', linewidths=0.7)

                # Overall mode: orange star on top
                ax.scatter(corner_data[typical_idx, col], corner_data[typical_idx, row],
                           color='orange', s=160, marker='*', zorder=10,
                           edgecolors='black', linewidths=0.7)

                ax.set_xlim(ranges[col])
                ax.set_ylim(ranges[row])
                ax.grid(True, alpha=0.15)

            else:
                # Above diagonal: hidden (used for legend)
                ax.set_visible(False)

            # Axis labels and tick visibility
            ax.tick_params(labelsize=7)
            if row != ndim - 1 or row == col:
                ax.tick_params(labelbottom=False)
            if col != 0 or row == col:
                ax.tick_params(labelleft=False)
            if row == ndim - 1 and row != col:
                ax.set_xlabel(labels[col], fontsize=11)
            if col == 0 and row != col:
                ax.set_ylabel(labels[row], fontsize=11)

    # ── Legend in the upper-right empty area ─────────────────────────────────
    legend_handles = (
        [Line2D([0], [0], marker='o', color='w',
                markerfacecolor=cluster_colors[ci], markersize=9,
                label=f'Cluster {cl}')
         for ci, cl in enumerate(unique_cl)]
        + [Line2D([0], [0], marker='*', color='w', markerfacecolor='orange',
                  markeredgecolor='black', markeredgewidth=0.7,
                  markersize=13, label='Global mode'),
           Line2D([0], [0], marker='D', color='w', markerfacecolor='gray',
                  markeredgecolor='black', markeredgewidth=0.7,
                  markersize=9, label='Cluster modes')]
    )

    ax_leg = axes[0, ndim - 1]
    ax_leg.set_visible(True)
    ax_leg.axis('off')
    ax_leg.legend(handles=legend_handles, loc='center', fontsize=9.5,
                  framealpha=0.85, title_fontsize=10)

    plt.savefig(paths.figures / "Appendix3.pdf", bbox_inches='tight', dpi=150)
    plt.close(fig)
