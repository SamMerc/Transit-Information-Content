#############################
########## Purpose ##########
#############################

# Produces a GIF that cycles through four lower-triangle corner plots of the
# 4th-order NLLD coefficients [c1, c2, c3, c4], each coloured by a different
# physical parameter: Teff (inferno), logg (cividis), metallicity (coolwarm),
# wavelength (turbo). One frame per parameter, 30 seconds per frame.
#
# Output: src/static/corner_animation.gif
# This file is NOT a Snakemake rule — run it manually once and commit the GIF.
# Data produced by Fig3_run.py and downloaded from Zenodo as results.npz.


######################################
########## Import libraries ##########
######################################

import io
import os
import numpy as np
import matplotlib
import paths
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from PIL import Image


######################################
########## Hyper-parameters ##########
######################################

input_save_path = str(paths.data / "Fig3_Storage") + "/"
output_path     = paths.data.parent / "static" / "corner_animation.gif"
model           = 'mps1'
n_sc            = 40_000      # scatter subsample per frame
frame_ms        = 5_000      # milliseconds per frame (10 s)
dpi             = 1000


################################
########## Code block ##########
################################

# ── Load data ─────────────────────────────────────────────────────────────────
res = np.load(input_save_path + f'{model}/results.npz', allow_pickle=False)

corner_data = res['corner_data']   # (N, 4)
corner_meta = res['corner_meta']   # (N, 4)  [i_Teff, j_logg, k_met, i_wav]
T_vals_arr  = res['T_vals_arr']
g_vals_arr  = res['g_vals_arr']
m_vals_arr  = res['m_vals_arr']
wavs_ref    = res['wavs_ref']

teff_vals = T_vals_arr[corner_meta[:, 0]]
logg_vals = g_vals_arr[corner_meta[:, 1]]
met_vals  = m_vals_arr[corner_meta[:, 2]]
wav_vals  = wavs_ref[corner_meta[:, 3]] / 1e4   # µm

N      = len(corner_data)
ndim   = 4
labels = [r'$c_1$', r'$c_2$', r'$c_3$', r'$c_4$']
ranges = [(np.percentile(corner_data[:, i], 0.1),
           np.percentile(corner_data[:, i], 99.9)) for i in range(ndim)]

# ── Frame colour definitions (discrete=True → 10 sampled grid points) ────────
frames_config = [
    (r'$T_{\rm eff}$ (K)',   teff_vals, plt.cm.inferno,  True),
    (r'$\log g$',            logg_vals, plt.cm.cividis,  True),
    (r'[M/H]',               met_vals,  plt.cm.get_cmap('winter'), True),
    (r'Wavelength ($\mu$m)', wav_vals,  plt.cm.turbo,    False),
]

# ── Figure layout (reused across frames) ─────────────────────────────────────
fig, axes = plt.subplots(ndim, ndim, figsize=(12, 11),
                         gridspec_kw={'wspace': 0.05, 'hspace': 0.05})
fig.subplots_adjust(right=0.86)
cbar_ax = fig.add_axes([0.88, 0.12, 0.025, 0.70])


def render_frame(frame_idx):
    col_label, col_vals, cmap, discrete = frames_config[frame_idx]

    if discrete:
        unique_vals = np.unique(col_vals)
        n_uv = len(unique_vals)
        hw   = np.diff(unique_vals) / 2          # half-widths between adjacent values
        bounds = np.concatenate([[unique_vals[0] - hw[0]],
                                  unique_vals[:-1] + hw,
                                  [unique_vals[-1] + hw[-1]]])
        # Resample the cmap to exactly n_uv colours so BoundaryNorm output
        # stays in [0, n_uv-1] and the colormap lookup is consistent.
        cmap_disc = plt.cm.get_cmap(cmap.name, n_uv)
        col_norm  = mcolors.BoundaryNorm(bounds, ncolors=cmap_disc.N)
        sm_disc   = cm.ScalarMappable(cmap=cmap_disc, norm=col_norm)
        plot_cmap = cmap_disc
    else:
        unique_vals = None
        col_norm    = mcolors.Normalize(vmin=col_vals.min(), vmax=col_vals.max())
        plot_cmap   = cmap

    # Subsample sorted by colour value for correct Z-ordering
    rng    = np.random.default_rng(42)
    n_draw = min(N, n_sc)
    idx    = rng.choice(N, size=n_draw, replace=False)
    idx    = idx[np.argsort(col_vals[idx])]

    for row in range(ndim):
        for col in range(ndim):
            ax = axes[row, col]
            ax.cla()

            if row == col:
                if discrete:
                    # One histogram per sampled grid value, coloured to match scatter
                    for uv in unique_vals:
                        m = col_vals == uv
                        ax.hist(corner_data[m, row], bins=50, range=ranges[row],
                                alpha=0.55, color=sm_disc.to_rgba(uv), density=False,
                                histtype='stepfilled', edgecolor='none')
                else:
                    # Continuous: decompose into 10 wavelength bins
                    n_wb    = 10
                    w_edges = np.linspace(col_vals.min(), col_vals.max(), n_wb + 1)
                    w_ctrs  = 0.5 * (w_edges[:-1] + w_edges[1:])
                    for ib, ctr in enumerate(w_ctrs):
                        m = (col_vals >= w_edges[ib]) & (col_vals < w_edges[ib + 1])
                        if ib == n_wb - 1:
                            m |= (col_vals == w_edges[ib + 1])
                        if not m.any():
                            continue
                        ax.hist(corner_data[m, row], bins=50, range=ranges[row],
                                alpha=0.55, color=plot_cmap(col_norm(ctr)), density=True,
                                histtype='stepfilled', edgecolor='none')
                ax.set_xlim(ranges[row])
                ax.set_yticks([])
                for sp in ['top', 'left', 'right']:
                    ax.spines[sp].set_visible(False)
                ax.set_visible(True)

            elif row > col:
                ax.scatter(corner_data[idx, col], corner_data[idx, row],
                           c=col_vals[idx], cmap=plot_cmap, norm=col_norm,
                           s=2, alpha=0.40, linewidths=0, rasterized=True)
                ax.set_xlim(ranges[col])
                ax.set_ylim(ranges[row])
                ax.grid(True, alpha=0.15)
                ax.set_visible(True)

            else:
                ax.set_visible(False)

            ax.tick_params(labelsize=7)
            if row != ndim - 1 or row == col:
                ax.tick_params(labelbottom=False)
            if col != 0 or row == col:
                ax.tick_params(labelleft=False)
            if row == ndim - 1 and row != col:
                ax.set_xlabel(labels[col], fontsize=11)
            if col == 0 and row != col:
                ax.set_ylabel(labels[row], fontsize=11)

    # Colorbar
    cbar_ax.cla()
    sm = cm.ScalarMappable(cmap=plot_cmap, norm=col_norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label(col_label, fontsize=12)
    cb.ax.tick_params(labelsize=9)
    if discrete:
        cb.set_ticks(unique_vals)
        is_teff = 'eff' in col_label
        cb.set_ticklabels([f'{v:.0f}' if is_teff else f'{v:.2f}' for v in unique_vals])

    fig.suptitle(f'Limb darkening coefficients  —  coloured by {col_label}',
                 fontsize=13, y=0.995)


# ── Render all frames to PIL images ──────────────────────────────────────────
frames_pil = []
for i in range(len(frames_config)):
    print(f'  Rendering frame {i + 1}/{len(frames_config)} ...')
    render_frame(i)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi)   # no tight crop → fixed canvas size
    buf.seek(0)
    frames_pil.append(Image.open(buf).copy())
    buf.close()

plt.close(fig)

# ── Assemble GIF ──────────────────────────────────────────────────────────────
frames_gif = [f.quantize(colors=256) for f in frames_pil]
frames_gif[0].save(
    str(output_path),
    save_all=True,
    append_images=frames_gif[1:],
    duration=frame_ms,
    loop=0,
    optimize=False,
    disposal=2,   # restore to background between frames, prevents ghosting
)
print(f'Saved GIF → {output_path}')
