#############################
########## Purpose ##########
#############################

# Appendix 4 shows the inter-model spread in 4th-order NLLD coefficients
# between MPS-ATLAS set1 and set2, motivating the 10-20% Gaussian prior widths
# used in Section 5.  Both datasets are produced by Appendix4_run.py, which
# fits ALL valid profiles from each model without subsampling (~580 k each).
# Profiles are compared as matched pairs identified by sharing the same physical
# parameter values (Teff, logg, [M/H], wavelength), rounded to avoid floating-
# point mismatch.  Matching on values rather than grid indices is robust to any
# difference in the grid arrays stored in the two results files.
#
# Top panels: histograms of (c_set2 - c_set1) / |c_set1| for each coefficient.
# Bottom panels: median signed relative difference vs each physical parameter,
#   with 16th-84th percentile shading per coefficient.


######################################
########## Import libraries ##########
######################################

import numpy as np
import matplotlib
import paths
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator


######################################
########## Hyper-parameters ##########
######################################

mps1_path   = str(paths.data / "Appendix4_Storage") + "/mps1/results.npz"
mps2_path   = str(paths.data / "Appendix4_Storage") + "/mps2/results.npz"
output_path = str(paths.figures) + "/Appendix4.pdf"

# Minimum |c_mps1| to include in the relative-difference calculation.
eps = 0.05


################################
########## Code block ##########
################################

# ── Load both datasets ────────────────────────────────────────────────────────

res1 = np.load(mps1_path, allow_pickle=False)
res2 = np.load(mps2_path, allow_pickle=False)

corner_data1 = res1['corner_data']   # (N1, 4)
corner_meta1 = res1['corner_meta']   # (N1, 4)  [i_T, j_g, k_m, i_wav]
corner_data2 = res2['corner_data']   # (N2, 4)
corner_meta2 = res2['corner_meta']   # (N2, 4)

T_vals_arr1 = res1['T_vals_arr'];  g_vals_arr1 = res1['g_vals_arr']
m_vals_arr1 = res1['m_vals_arr'];  wavs_ref1   = res1['wavs_ref']    # Angstrom

T_vals_arr2 = res2['T_vals_arr'];  g_vals_arr2 = res2['g_vals_arr']
m_vals_arr2 = res2['m_vals_arr'];  wavs_ref2   = res2['wavs_ref']

# ── Match profiles by physical parameter values ───────────────────────────────

def make_phys_keys(meta, T_arr, g_arr, m_arr, wav_arr):
    T   = np.round(T_arr[meta[:, 0]], 2)
    g   = np.round(g_arr[meta[:, 1]], 4)
    m   = np.round(m_arr[meta[:, 2]], 4)
    wav = np.round(wav_arr[meta[:, 3]], 2)
    return list(zip(T, g, m, wav))

keys1 = make_phys_keys(corner_meta1, T_vals_arr1, g_vals_arr1, m_vals_arr1, wavs_ref1)
keys2 = make_phys_keys(corner_meta2, T_vals_arr2, g_vals_arr2, m_vals_arr2, wavs_ref2)

lookup1 = {k: n for n, k in enumerate(keys1)}
lookup2 = {k: n for n, k in enumerate(keys2)}

common_keys = set(lookup1.keys()) & set(lookup2.keys())
N_matched   = len(common_keys)

print(f'mps1 valid profiles : {len(lookup1):,}')
print(f'mps2 valid profiles : {len(lookup2):,}')
print(f'Matched pairs       : {N_matched:,}')

idx1 = np.array([lookup1[k] for k in common_keys])
idx2 = np.array([lookup2[k] for k in common_keys])

data1 = corner_data1[idx1]   # (N_matched, 4)
data2 = corner_data2[idx2]   # (N_matched, 4)

# Physical coordinates for the matched set
common_keys_arr = np.array(list(common_keys))   # (N_matched, 4): T, g, m, wav_A
phys_Teff = common_keys_arr[:, 0]               # K
phys_logg = common_keys_arr[:, 1]
phys_MH   = common_keys_arr[:, 2]
phys_wav  = common_keys_arr[:, 3] / 1e4         # µm

# ── Signed fractional differences (N_matched × 4), NaN where |c1| <= eps ─────

rel_diff_mat = np.full((N_matched, 4), np.nan)
for ic in range(4):
    c1    = data1[:, ic]
    c2    = data2[:, ic]
    valid = np.abs(c1) > eps
    rel_diff_mat[valid, ic] = (c2[valid] - c1[valid]) / np.abs(c1[valid]) * 100.0

coeff_labels = [r'$c_1$', r'$c_2$', r'$c_3$', r'$c_4$']
coeff_colors = plt.cm.tab10(np.linspace(0, 0.4, 4))

for ic in range(4):
    rdiff = rel_diff_mat[:, ic]
    rdiff = rdiff[~np.isnan(rdiff)]
    med   = np.median(rdiff)
    p25, p75 = np.percentile(rdiff, [25, 75])
    print(f'{coeff_labels[ic]}: median = {med:+.1f}%  '
          f'25–75th = [{p25:+.1f}%, {p75:+.1f}%]  N = {len(rdiff):,}')

# ═════════════════════════════════════════════════════════════════════════════
# Single combined figure: two GridSpec blocks with a gap between them
# ═════════════════════════════════════════════════════════════════════════════

hist_color  = '#4C72B0'
vline_color = '#C44E52'
band_color  = '#4C72B0'

fig = plt.figure(figsize=(12, 10))

gs_top = GridSpec(2, 2, figure=fig, top=0.97, bottom=0.54,
                  hspace=0.24, wspace=0.18)
gs_bot = GridSpec(2, 2, figure=fig, top=0.48, bottom=0.04,
                  hspace=0.26, wspace=0.15)

hist_axes  = [fig.add_subplot(gs_top[r, c]) for r in range(2) for c in range(2)]
param_axes = [fig.add_subplot(gs_bot[r, c]) for r in range(2) for c in range(2)]

# ── Top panels: histograms ────────────────────────────────────────────────────

for ic, ax in enumerate(hist_axes):
    row, col = divmod(ic, 2)

    rdiff = rel_diff_mat[:, ic]
    rdiff = rdiff[~np.isnan(rdiff)]

    p5,  p95 = np.percentile(rdiff, [5,  95])
    p25, p75 = np.percentile(rdiff, [25, 75])
    med       = np.median(rdiff)
    disp_range = (max(p5 - 5, -100), min(p95 + 5, 100))

    ax.hist(rdiff, bins=80, range=disp_range,
            color=hist_color, alpha=0.80, density=True, edgecolor='none')
    ax.axvline(med, color=vline_color, linestyle='--', linewidth=1.8, zorder=3)
    ax.axvspan(p25, p75, color=band_color, alpha=0.15)
    ax.axvline(0, color='black', linewidth=0.8, alpha=0.5)

    ax.text(0.97 if ic in [0, 2] else 0.27, 0.95, f'Med: {med:+.1f}%', transform=ax.transAxes,
            ha='right', va='top', fontsize=12, color=vline_color)
    ax.text([0.97, 0.4, 0.97, 0.45][ic], 0.8, f'1Q/3Q: {p25:+.1f}/{p75:+.1f}%', transform=ax.transAxes,
            ha='right', va='top', fontsize=12, color=vline_color)
    
    ax.set_title(coeff_labels[ic], fontsize=12, pad=6)
    if row == 1:
        ax.set_xlabel(r'$(c_{\rm set2} - c_{\rm set1})\,/\,|c_{\rm set1}|\ (\%)$',
                      fontsize=12)
    if col == 0:
        ax.set_ylabel('Probability density', fontsize=12)
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_minor_locator(MultipleLocator(5))

# ── Bottom panels: median relative difference vs physical parameter ────────────

param_specs = [
    (phys_Teff, r'$T_{\rm eff}$ (K)'),
    (phys_logg, r'$\log g$'),
    (phys_MH,   r'$[{\rm M/H}]$'),
    (phys_wav,  r'$\lambda\ (\mu{\rm m})$'),
]

for ip, (ax, (param_vals, xlabel)) in enumerate(zip(param_axes, param_specs)):
    row, col = divmod(ip, 2)
    unique_vals = np.sort(np.unique(param_vals))

    for ic in range(4):
        medians = np.full(len(unique_vals), np.nan)
        p25s    = np.full(len(unique_vals), np.nan)
        p75s    = np.full(len(unique_vals), np.nan)

        for iv, val in enumerate(unique_vals):
            mask = (param_vals == val) & ~np.isnan(rel_diff_mat[:, ic])
            if mask.sum() < 3:
                continue
            rdiff_bin          = rel_diff_mat[mask, ic]
            medians[iv]        = np.median(rdiff_bin)
            p25s[iv], p75s[iv] = np.percentile(rdiff_bin, [25, 75])

        finite = ~np.isnan(medians)
        ax.plot(unique_vals[finite], medians[finite],
                color=coeff_colors[ic], linewidth=1.8,
                marker='o', markersize=4, label=coeff_labels[ic])
        ax.fill_between(unique_vals[finite], p25s[finite], p75s[finite],
                        color=coeff_colors[ic], alpha=0.15)

    ax.axhline(0, color='black', linewidth=0.8, alpha=0.5)
    ax.set_xlabel(xlabel, fontsize=12)
    if col == 0:
        ax.set_ylabel(r'Median $\Delta c\,/\,|c_{\rm set1}|\ (\%)$', fontsize=12)
    ax.grid(True, alpha=0.25)
    # ax.yaxis.set_minor_locator(MultipleLocator(5))

param_axes[0].legend(fontsize=12, loc='best', framealpha=0.6, ncols=4)

fig.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved → {output_path}')
