#############################
########## Purpose ##########
#############################

# Appendix 2 shows corner plots of the 4th-order NLLD coefficients [c1, c2, c3, c4]
# coloured by each physical parameter: effective temperature, surface gravity,
# metallicity, and wavelength. Data are produced by Fig5_run.py and downloaded
# from Zenodo as results.npz.


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
import corner
from matplotlib.backends.backend_pdf import PdfPages


######################################
########## Hyper-parameters ##########
######################################

input_save_path = str(paths.data / "Fig5_Storage") + "/"
models = ['mps1']  # ['phoenix','kurucz', 'stagger', 'mps1', 'mps2']


################################
########## Code block ##########
################################

for model in models:

    # ── Load pre-computed results ─────────────────────────────────────────────
    res = np.load(input_save_path + f'{model}/results.npz', allow_pickle=False)

    corner_data = res['corner_data']          # (N, 4)  NLLD coefficients
    corner_meta = res['corner_meta']          # (N, 4)  [i_Teff, j_logg, k_met, i_wav]
    T_vals_arr  = res['T_vals_arr']
    g_vals_arr  = res['g_vals_arr']
    m_vals_arr  = res['m_vals_arr']
    wavs_ref    = res['wavs_ref']

    # ── Recover physical parameter values from metadata indices ───────────────
    corner_wav_um  = wavs_ref[corner_meta[:, 3]] / 1e4  # microns

    labels_4d = [r'$c_1$', r'$c_2$', r'$c_3$', r'$c_4$']
    ranges_4d = [
        (np.percentile(corner_data[:, ic], 1),
         np.percentile(corner_data[:, ic], 99))
        for ic in range(4)
    ]
    ndim_4d = 4

    # ── Shared scatter subsample (sorted by colour value for proper layering) ─
    rng = np.random.default_rng(42)

    with PdfPages(paths.figures / "Appendix2.pdf") as pdf:

        # ── Figure 2a : density corner (black) ───────────────────────────────
        print('    FIGURE 2a - All coefficients corner plot')
        fig_a = corner.corner(
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
            ax = fig_a.axes[i * ndim_4d + i]
            for spine in ['top', 'left', 'right']:
                ax.spines[spine].set_visible(False)
        fig_a.suptitle(f'4th-order NLLD coefficients — {model}', fontsize=13, y=1.02)
        pdf.savefig(fig_a, bbox_inches='tight')
        plt.close(fig_a)

        # ── Figures 2b–d : coloured by Teff, logg, [M/H] ────────────────────
        meta_col_names = [r'$T_{\rm eff}$ (K)', r'$\log\,g$', r'[M/H]']
        meta_col_keys  = ['Teff', 'logg', 'MH']
        meta_col_vals  = [T_vals_arr[corner_meta[:, 0]],
                          g_vals_arr[corner_meta[:, 1]],
                          m_vals_arr[corner_meta[:, 2]]]
        meta_cmaps     = [plt.cm.inferno, plt.cm.cividis, plt.cm.coolwarm]

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

            idx_sub = (np.sort(rng.choice(len(corner_data), size=40_000, replace=False))
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

            fig_cm.suptitle(
                f'4th-order NLLD coefficients coloured by {col_label} — {model}',
                fontsize=13, y=1.02)
            pdf.savefig(fig_cm, bbox_inches='tight')
            plt.close(fig_cm)

        # ── Figure 2e : coloured by wavelength ───────────────────────────────
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

        idx_sub_wav = (np.sort(rng.choice(len(corner_data), size=40_000, replace=False))
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
        wav_ticks = np.arange(
            np.ceil(corner_wav_um.min() / wav_tick_step) * wav_tick_step,
            corner_wav_um.max() + wav_tick_step / 2,
            wav_tick_step)
        cbar_wav.set_ticks(wav_ticks)
        cbar_wav.set_ticklabels([f'{t:.1f}' for t in wav_ticks])

        fig_wav.suptitle(
            f'4th-order NLLD coefficients coloured by wavelength — {model}',
            fontsize=13, y=1.02)
        pdf.savefig(fig_wav, bbox_inches='tight')
        plt.close(fig_wav)
