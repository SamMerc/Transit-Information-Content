#############################
########## Purpose ##########
#############################

# Appendix 4 shows the inter-model spread in 4th-order NLLD coefficients
# between MPS-ATLAS set1 and set2, motivating the 10-20% Gaussian prior widths
# used in Section 5.  Both datasets are produced by Appendix4_run.py, which
# fits ALL valid profiles from each model without subsampling (~580 k each).
# Profiles are compared as matched pairs identified by sharing the same grid
# index (i_Teff, j_logg, k_met, i_wav), ensuring the comparison is at identical
# stellar parameters.  With both full datasets the intersection contains nearly
# all valid profiles, giving a complete picture of the inter-model spread.
#
# For each of the four NLLD coefficients the plot shows the distribution of the
# signed fractional difference (mps2 - mps1) / |mps1| across all matched pairs,
# revealing both the magnitude and any systematic offset between the two models.


######################################
########## Import libraries ##########
######################################

import numpy as np
import matplotlib
import paths
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


######################################
########## Hyper-parameters ##########
######################################

mps1_path   = str(paths.data / "Appendix4_Storage") + "/mps1/results.npz"
mps2_path   = str(paths.data / "Appendix4_Storage") + "/mps2/results.npz"
output_path = str(paths.figures) + "/Appendix4.pdf"

# Minimum |c_mps1| to include in the relative-difference calculation.
# Avoids division by near-zero when a coefficient is close to zero.
eps = 0.05


################################
########## Code block ##########
################################

# ── Load both datasets ────────────────────────────────────────────────────────

res1 = np.load(mps1_path, allow_pickle=False)
res2 = np.load(mps2_path, allow_pickle=False)

corner_data1 = res1['corner_data']   # (N1, 4)  mps1 LDC vectors
corner_meta1 = res1['corner_meta']   # (N1, 4)  [i_T, j_g, k_m, i_wav]
corner_data2 = res2['corner_data']   # (N2, 4)  mps2 LDC vectors
corner_meta2 = res2['corner_meta']   # (N2, 4)

T_vals_arr = res1['T_vals_arr']
wavs_ref   = res1['wavs_ref']        # Angstrom

# ── Match profiles by grid index ─────────────────────────────────────────────
# Two profiles are paired when they correspond to the same stellar grid point
# and wavelength, i.e. identical (i_Teff, j_logg, k_met, i_wav) tuple.

lookup1 = {tuple(corner_meta1[n]): n for n in range(len(corner_meta1))}
lookup2 = {tuple(corner_meta2[n]): n for n in range(len(corner_meta2))}

common_keys = set(lookup1.keys()) & set(lookup2.keys())
N_matched   = len(common_keys)

print(f'mps1 valid profiles : {len(lookup1):,}')
print(f'mps2 valid profiles : {len(lookup2):,}')
print(f'Matched pairs       : {N_matched:,}')

idx1 = np.array([lookup1[k] for k in common_keys])
idx2 = np.array([lookup2[k] for k in common_keys])

data1 = corner_data1[idx1]   # (N_matched, 4)
data2 = corner_data2[idx2]   # (N_matched, 4)

# Also retrieve physical coordinates for the matched set
keys_arr    = np.array(list(common_keys))         # (N_matched, 4)
Teff_matched = T_vals_arr[keys_arr[:, 0]]
wav_matched  = wavs_ref[keys_arr[:, 3]] / 1e4     # µm

# ── Build signed fractional differences ──────────────────────────────────────
# delta_c = (c_mps2 - c_mps1) / |c_mps1|  for matched pairs where |c_mps1| > eps

coeff_labels  = [r'$c_1$', r'$c_2$', r'$c_3$', r'$c_4$']
rel_diffs     = []
valid_masks   = []

for ic in range(4):
    c1     = data1[:, ic]
    c2     = data2[:, ic]
    vmask  = np.abs(c1) > eps
    rdiff  = (c2[vmask] - c1[vmask]) / np.abs(c1[vmask]) * 100.0   # percent
    rel_diffs.append(rdiff)
    valid_masks.append(vmask)

    med = np.median(rdiff)
    p10, p90 = np.percentile(rdiff, [10, 90])
    print(f'{coeff_labels[ic]}: median = {med:+.1f}%   '
          f'10-90th pct = [{p10:+.1f}%, {p90:+.1f}%]   '
          f'N = {vmask.sum():,}')

# ── Figure ────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(9, 7),
                          gridspec_kw={'hspace': 0.40, 'wspace': 0.35})
axes = axes.flatten()

hist_color   = '#4C72B0'
vline_color  = '#C44E52'
band_color   = '#4C72B0'

for ic, ax in enumerate(axes):
    rdiff = rel_diffs[ic]
    c1    = data1[valid_masks[ic], ic]

    # Clip display range to 5th–95th percentile for readability
    p5,  p95 = np.percentile(rdiff, [5,  95])
    p10, p90 = np.percentile(rdiff, [10, 90])
    med       = np.median(rdiff)
    disp_range = (max(p5 - 5, -100), min(p95 + 5, 100))

    ax.hist(rdiff, bins=80, range=disp_range,
            color=hist_color, alpha=0.80,
            density=True, edgecolor='none')

    # Median
    ax.axvline(med, color=vline_color, linestyle='--', linewidth=1.8,
               label=f'Median: {med:+.1f}%', zorder=3)

    # 10th–90th percentile shading
    ax.axvspan(p10, p90, color=band_color, alpha=0.12,
               label=f'10–90th pct: [{p10:+.1f}%, {p90:+.1f}%]')

    ax.axvline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)

    ax.set_xlabel(r'$(c_{\rm set2} - c_{\rm set1})\,/\,|c_{\rm set1}|\ (\%)$',
                  fontsize=11)
    ax.set_ylabel('Probability density', fontsize=10)
    ax.set_title(coeff_labels[ic], fontsize=13, pad=4)
    ax.legend(fontsize=8.5, loc='upper left', framealpha=0.85)
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_minor_locator(MultipleLocator(5))

subtitle = (f'{N_matched:,} matched pairs on identical '
            r'$(T_{\rm eff},\,\log g,\,[{\rm M/H}],\,\lambda)$ grid points')
fig.suptitle(
    r'Inter-model LDC differences: MPS-ATLAS set2 vs set1' + '\n(' + subtitle + ')',
    fontsize=11, y=1.01,
)

fig.savefig(output_path, dpi=150, bbox_inches='tight')
print(f'Saved → {output_path}')
