#############################
########## Purpose ##########
#############################

# Figure 6 shows the simulated transmission spectra recovered for three fiducial stellar
# types (M, G, and F dwarfs) when a flat (wavelength-independent) transit depth is fit with
# three limb-darkening laws: a quadratic law, a 3rd-order polynomial law, and the (correct)
# 4th-order non-linear limb-darkening (NLLD) law. The underlying data are produced by
# Fig6_run.py, which runs one JOINT MCMC per (star, limb-darkening law) combination -- the
# wavelength-independent orbital parameters are shared across every wavelength channel's
# light curve, while the radius ratio and limb-darkening coefficients are free per channel
# -- and saves a single summary.npz per combination containing the per-channel posterior
# median/16th/84th percentile/best-fit radius ratio (plus the shared-parameter posterior
# summary). Since Fig6_run.py is dispatched one (star, law) combination at a time (there are
# only 3 stars x 3 laws = 9 in total), this script simply assembles whichever combinations
# have already been run -- missing ones are left as a placeholder panel.


######################################
########## Import libraries ##########
######################################

import os
import numpy as np
import matplotlib
import paths
matplotlib.use('Agg')
import matplotlib.pyplot as plt


######################################
########## Hyper-parameters ##########
######################################

input_save_path = str(paths.data / "Fig6_Storage") + "/"

# Must match the stellar_types dict in Fig6_run.py (only Teff/logg/MH are needed here, for
# the panel titles).
stellar_types = {
    'M_dwarf': {'Teff': 3500.0, 'logg': 4.8, 'MH': 0.0},
    'G_dwarf': {'Teff': 5800.0, 'logg': 4.5, 'MH': 0.0},
    'F_dwarf': {'Teff': 7200.0, 'logg': 4.3, 'MH': 0.0},
}
star_order  = ['M_dwarf', 'G_dwarf', 'F_dwarf']
star_labels = {'M_dwarf': 'M dwarf', 'G_dwarf': 'G dwarf', 'F_dwarf': 'F dwarf'}

# Must match the true, injected radius ratio in Fig6_run.py
r_true = 0.1

LDLs        = ['PLD_2', 'PLD_3', '4NLLD']
LDL_labels  = {'PLD_2': 'Quadratic law', 'PLD_3': '3rd-order law', '4NLLD': '4th-order NLLD (truth)'}
LDL_colors  = {'PLD_2': 'C0', 'PLD_3': 'C1', '4NLLD': 'k'}
LDL_zorder  = {'PLD_2': 2, 'PLD_3': 3, '4NLLD': 4}

fs = 14  # font size for plots


################################
########## Code block ##########
################################

fig, axes = plt.subplots(
    len(star_order), 1, figsize=(9, 4 * len(star_order)),
    sharex=True, squeeze=False,
)
axes = axes[:, 0]

true_depth_ppm = 1e6 * r_true**2

for istar, star_name in enumerate(star_order):

    ax = axes[istar]
    any_data = False

    ax.axhline(true_depth_ppm, color='gray', linestyle='--', linewidth=1.5, zorder=1,
               label='Injected (flat) depth')

    for LDL in LDLs:
        summary_file = os.path.join(input_save_path, star_name, LDL, 'summary.npz')
        if not os.path.exists(summary_file):
            print(f'  [{star_name}] {LDL}: not run yet, skipping')
            continue

        s = np.load(summary_file, allow_pickle=False)
        wav_centers = s['wav_centers']
        r_median    = s['r_median']
        r_lo        = s['r_lo']
        r_hi        = s['r_hi']

        depth_ppm    = 1e6 * r_median**2
        depth_lo_ppm = 1e6 * r_lo**2
        depth_hi_ppm = 1e6 * r_hi**2
        yerr = np.vstack([depth_ppm - depth_lo_ppm, depth_hi_ppm - depth_ppm])

        ax.errorbar(
            wav_centers, depth_ppm, yerr=yerr,
            fmt='.', markersize=4, elinewidth=0.8, capsize=0, alpha=0.85,
            color=LDL_colors[LDL], zorder=LDL_zorder[LDL],
            label=LDL_labels[LDL],
        )
        any_data = True

    if not any_data:
        ax.text(0.5, 0.5, 'No runs found yet', ha='center', va='center',
                transform=ax.transAxes, fontsize=fs, color='gray')

    Teff = stellar_types[star_name]['Teff']
    logg = stellar_types[star_name]['logg']
    MH   = stellar_types[star_name]['MH']
    ax.set_title(
        f'{star_labels[star_name]} '
        f'($T_{{\\rm eff}}$={Teff:.0f} K, $\\log g$={logg:.2f}, [M/H]={MH:+.2f})',
        fontsize=fs,
    )
    ax.set_ylabel('Transit depth (ppm)', fontsize=fs)
    ax.tick_params(axis='both', labelsize=fs - 2)
    ax.grid(True, alpha=0.3)

for ax in axes:
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=fs - 3, loc='best', framealpha=0.9)
        break
axes[-1].set_xlabel('Wavelength (micron)', fontsize=fs)

fig.tight_layout()
plt.savefig(paths.figures / "Fig6.pdf", bbox_inches="tight")

# Print a short summary of the induced bias per star / limb-darkening law (based on the
# posterior median depth) plus the recovered shared (wavelength-independent) parameters
print(f'\n  {"Star":<10}  {"LDL":<22}  {"RMS bias (ppm)":>15}  {"Max |bias| (ppm)":>18}')
print(f'  {"-"*70}')
for star_name in star_order:
    for LDL in LDLs:
        summary_file = os.path.join(input_save_path, star_name, LDL, 'summary.npz')
        if not os.path.exists(summary_file):
            continue
        s = np.load(summary_file, allow_pickle=False)
        bias = 1e6 * s['r_median']**2 - true_depth_ppm
        print(f'  {star_labels[star_name]:<10}  {LDL_labels[LDL]:<22}  '
              f'{np.sqrt(np.mean(bias**2)):>15.2f}  {np.max(np.abs(bias)):>18.2f}')

print(f'\n  {"Star":<10}  {"LDL":<10}  ' + '  '.join(f'{n:>12}' for n in ['i', 'a', 'period', 'sqrtecosw', 'sqrtesinw']))
print(f'  {"-"*90}')
for star_name in star_order:
    for LDL in LDLs:
        summary_file = os.path.join(input_save_path, star_name, LDL, 'summary.npz')
        if not os.path.exists(summary_file):
            continue
        s = np.load(summary_file, allow_pickle=False)
        vals = s['shared_median']
        print(f'  {star_labels[star_name]:<10}  {LDL:<10}  ' + '  '.join(f'{v:>12.5f}' for v in vals))
