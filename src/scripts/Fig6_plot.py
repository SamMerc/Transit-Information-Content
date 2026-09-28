#############################
########## Purpose ##########
#############################

# Figure 6 shows the simulated transmission spectra recovered for five fiducial stellar
# types (informed from our Fig3 clustering) when a flat (wavelength-independent) transit
# depth is fit with three limb-darkening laws: a quadratic law, a 3rd-order polynomial law,
# and the (correct) 4th-order non-linear limb-darkening (NLLD) law. The underlying data are
# produced by:
#   - Fig6_prerun.py: builds the per-channel truth and the noisy chromatic dataset, then
#     fits a white light curve (WLC) per star to constrain the orbital parameters. Saves
#     <star>/wav_grid.npz and <star>/prerun.npz.
#   - Fig6_run.py: for each (star, limb-darkening law, wavelength channel), fits that
#     channel's radius ratio and limb-darkening coefficients via MCMC, with orbital
#     parameters fixed at the WLC best fit. Saves <star>/<LDL>/channel_XXXX/summary.npz.
#
# Since Fig6_run.py is dispatched one (star, law, channel) combination at a time (e.g. as an
# HPC job array), this script only assembles whichever channels have already been run --
# channels without a summary.npz yet are simply left as gaps in the spectrum.
#
# Since each star's run directory contains thousands of small summary.npz files, the assembled
# per-star result is cached to a single <star_name>_cache.pkl file the first time it's built
# (mirroring the pattern used in Fig1_plot.py / Fig5_plot.py).


######################################
########## Import libraries ##########
######################################

import os
import pickle
import numpy as np
import matplotlib
import paths
matplotlib.use('Agg')
import matplotlib.pyplot as plt


######################################
########## Hyper-parameters ##########
######################################

input_save_path = str(paths.data / "Fig6_Storage") + "/"

# Must match the stellar_types dict in Fig6_prerun.py / Fig6_run.py (only Teff/logg/MH are
# needed here, for the panel titles).
stellar_types = {
    'C5': {'Teff': 3500.0, 'logg': 4.33, 'MH': 0.06}, # M-star
    'C1': {'Teff': 4111.0, 'logg': 4.11, 'MH': 0.06}, # K-star
    'C2': {'Teff': 4111.0, 'logg': 4.78, 'MH': 0.78}, # K-star
    'C7': {'Teff': 5944.0, 'logg': 4.33, 'MH': 0.06}, # G-star
    'C6': {'Teff': 6556.0, 'logg': 3.89, 'MH': 0.06}, # F-star
}
star_order = ['C1', 'C2', 'C5', 'C6', 'C7']

# Must match the true, injected radius ratio in Fig6_prerun.py
r_true = 0.1

LDLs        = ['PLD_2', 'PLD_3', '4NLLD']
LDL_labels  = {'PLD_2': 'Quadratic law', 'PLD_3': '3rd-order law', '4NLLD': '4th-order NLLD (truth)'}
LDL_colors  = {'PLD_2': 'C0', 'PLD_3': 'C1', '4NLLD': 'k'}
LDL_zorder  = {'PLD_2': 2, 'PLD_3': 3, '4NLLD': 4}

fs = 14  # font size for plots


############################
###### Function block ######
############################

def load_star_spectrum(star_name, LDLs, base_path):
    """
    Assemble the per-channel MCMC posterior summaries for one star into per-limb-darkening-
    law arrays of the posterior median / 16th / 84th percentile radius ratio, indexed by
    wavelength channel. Channels whose summary.npz has not been produced yet (i.e. that
    combination hasn't been run) are left as NaN.

    Results are cached to a single <star_name>_cache.pkl file so that repeated calls (and
    repeated pipeline runs) don't need to re-scan the many per-channel summary.npz files.
    """
    cache_file = os.path.join(base_path, star_name, f'{star_name}_cache.pkl')
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    wav_grid_file = os.path.join(base_path, star_name, 'wav_grid.npz')
    if not os.path.exists(wav_grid_file):
        return None, None

    grid = np.load(wav_grid_file)
    wav_centers = grid['wav_centers']
    n_bins = len(wav_centers)

    spectra = {}
    for LDL in LDLs:
        r_median = np.full(n_bins, np.nan)
        r_lo     = np.full(n_bins, np.nan)
        r_hi     = np.full(n_bins, np.nan)

        ldl_dir = os.path.join(base_path, star_name, LDL)
        n_done = 0
        for ib in range(n_bins):
            summary_file = os.path.join(ldl_dir, f'channel_{ib:04d}', 'summary.npz')
            if not os.path.exists(summary_file):
                continue
            s = np.load(summary_file)
            r_median[ib] = s['r_median']
            r_lo[ib]     = s['r_lo']
            r_hi[ib]     = s['r_hi']
            n_done += 1

        print(f'  [{star_name}] {LDL}: {n_done}/{n_bins} channels completed')
        spectra[LDL] = dict(r_median=r_median, r_lo=r_lo, r_hi=r_hi)

    with open(cache_file, 'wb') as f:
        pickle.dump((wav_centers, spectra), f)

    return wav_centers, spectra


################################
########## Code block ##########
################################

# Layout: the first four stars fill a 2x2 grid, and the fifth (last) star sits centered
# on its own row below, spanning the same width as one of the upper cells.
fig = plt.figure(figsize=(16, 12))
# Two independent GridSpecs so the row0-row1 gap can be tightened without touching the
# (larger) gap above the centered row-2 plot.
gs_top = fig.add_gridspec(nrows=2, ncols=4, hspace=0.15, wspace=0.3, top=0.95, bottom=0.42)
gs_bottom = fig.add_gridspec(nrows=1, ncols=4, wspace=0.3, top=0.36, bottom=0.12)
grid_slots = [gs_top[0, 0:2], gs_top[0, 2:4], gs_top[1, 0:2], gs_top[1, 2:4], gs_bottom[0, 1:3]]

axes = []
for slot in grid_slots:
    axes.append(fig.add_subplot(slot, sharex=axes[0] if axes else None))
bottom_axes = {axes[2], axes[3], axes[4]}
right_yaxis_axes = {axes[1], axes[3]}  # second column

true_depth_ppm = 1e6 * r_true**2

for istar, (star_name, ax) in enumerate(zip(star_order, axes)):

    print(f'Loading {star_name}...')
    wav_centers, spectra = load_star_spectrum(star_name, LDLs, input_save_path)

    if ax in bottom_axes:
        ax.set_xlabel('Wavelength (micron)', fontsize=fs)
    else:
        ax.tick_params(axis='x', labelbottom=False)

    if ax in right_yaxis_axes:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('right')

    if wav_centers is None:
        ax.text(0.5, 0.5, 'No runs found yet', ha='center', va='center',
                transform=ax.transAxes, fontsize=fs, color='gray')
        ax.set_title(star_name, fontsize=fs)
        continue

    for LDL in LDLs:
        r_median = spectra[LDL]['r_median']
        r_lo     = spectra[LDL]['r_lo']
        r_hi     = spectra[LDL]['r_hi']

        good = np.isfinite(r_median) & np.isfinite(r_lo) & np.isfinite(r_hi)

        depth_ppm    = 1e6 * r_median[good]**2 - true_depth_ppm
        depth_lo_ppm = 1e6 * r_lo[good]**2 - true_depth_ppm
        depth_hi_ppm = 1e6 * r_hi[good]**2 - true_depth_ppm
        yerr = np.vstack([depth_ppm - depth_lo_ppm, depth_hi_ppm - depth_ppm])

        ax.errorbar(
            wav_centers[good], depth_ppm, yerr=yerr,
            fmt='.', markersize=4, elinewidth=0.8, capsize=0, alpha=0.85,
            color=LDL_colors[LDL], zorder=LDL_zorder[LDL],
            label=LDL_labels[LDL],
        )

    Teff = stellar_types[star_name]['Teff']
    logg = stellar_types[star_name]['logg']
    MH   = stellar_types[star_name]['MH']
    ax.set_title(
        f'{star_name} '
        f'($T_{{\\rm eff}}$={Teff:.0f} K, $\\log g$={logg:.2f}, [M/H]={MH:+.2f})',
        fontsize=fs,
    )
    ax.set_ylabel('Transit depth $-$ truth (ppm)', fontsize=fs)
    ax.tick_params(axis='both', labelsize=fs - 2)
    ax.grid(True, alpha=0.3)

for ax in axes:
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=fs - 3, loc='best', framealpha=0.9)
        break

plt.savefig(paths.figures / "Fig6.pdf", bbox_inches="tight")

# Print a short summary of the induced bias per star / limb-darkening law (based on the
# posterior median depth, for whichever channels have been run so far)
print(f'\n  {"Star":<10}  {"LDL":<22}  {"N channels":>10}  {"RMS bias (ppm)":>15}  {"Max |bias| (ppm)":>18}')
print(f'  {"-"*82}')
for star_name in star_order:
    wav_centers, spectra = load_star_spectrum(star_name, LDLs, input_save_path)
    if wav_centers is None:
        continue
    for LDL in LDLs:
        r_median = spectra[LDL]['r_median']
        good = np.isfinite(r_median)
        if not np.any(good):
            continue
        bias = 1e6 * r_median[good]**2 - true_depth_ppm
        print(f'  {star_name:<10}  {LDL_labels[LDL]:<22}  {np.sum(good):>10d}  '
              f'{np.sqrt(np.mean(bias**2)):>15.2f}  {np.max(np.abs(bias)):>18.2f}')
