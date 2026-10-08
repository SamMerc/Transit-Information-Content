#############################
########## Purpose ##########
#############################

# Appendix 7 is the companion to Figure 6: for the same five fiducial stellar types and the
# same three limb-darkening laws (quadratic, 3rd-order polynomial, 4th-order NLLD), it shows
# the amplification factor, A, as a function of wavelength instead of the transit depth bias.
# A = (2 * r_std * r_bestfit) / scatter_in_bin (see Fig1_plot.py / Fig5_plot.py), with r_std
# the standard deviation of the cleaned, post-burn-in r chain and r_bestfit the best-fit r,
# both read directly from each channel's summary.npz.
#
# The underlying data are produced by:
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
# (mirroring the pattern used in Fig1_plot.py / Fig5_plot.py / Fig6_plot.py).


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
import astropy.units as u
from astropy.constants import G


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

# Must match the fiducial photometric scatter in Fig6_prerun.py (ppm) -- used, together with
# num_IT_pts below, as the noise-limited reference against which the amplification factor is
# computed (see compute_amplification_factor()).
model_scatter = 16.68100537200059

# Re-derive the number of in-transit points of the (wavelength-independent) orbital geometry
# used by Fig6_prerun.py / Fig6_run.py, needed for the amplification factor's noise-limited
# reference scatter. Identical for every star/channel since the injected orbit is achromatic.
G_solar_units = G.to(u.Rsun**3 / (u.Msun * u.day**2)).value
R_star = (1.0 * u.R_sun).value
period  = 1.0                                                            # days
a_meters = ((G.value * (1.0 * u.M_sun).to(u.kg).value * (period * 24 * 3600)**2) / (4 * np.pi**2))**(1 / 3)
a_orbit = a_meters / (1.0 * u.R_sun).to(u.m).value                       # stellar radii
inc, omega, ecc, t0 = np.deg2rad(90), 0.0, 0.0, 0.0

b = (a_orbit * np.cos(inc)) / R_star * (1 - ecc**2) / (1 + ecc * np.sin(omega))
arg = np.clip((1 / a_orbit) * np.sqrt((1 + r_true)**2 - b**2) / np.sin(inc), -1.0, 1.0)
T_dur = (period / np.pi) * np.sqrt(1 - ecc**2) / (1 + ecc * np.sin(omega)) * np.arcsin(arg)

low_t, high_t, exposure_time = -1.5 * T_dur, 1.5 * T_dur, 5    # days, days, seconds
num_t = int(np.floor(((high_t - low_t) * 24 * 3600) / exposure_time))
times = np.linspace(low_t, high_t, num_t)
num_IT_pts = np.sum((times > t0 - T_dur / 2) & (times < t0 + T_dur / 2))

LDLs        = ['PLD_2', 'PLD_3', '4NLLD']
LDL_labels  = {'PLD_2': 'Quadratic law', 'PLD_3': '3rd-order law', '4NLLD': '4th-order NLLD (truth)'}
LDL_colors  = {'PLD_2': 'C0', 'PLD_3': 'C1', '4NLLD': 'k'}
LDL_zorder  = {'PLD_2': 2, 'PLD_3': 3, '4NLLD': 4}

# Theoretical noise-limited floor (2:1 out-of-transit:in-transit baseline, see ms.tex Appendix)
# and the "acceptable" amplification factor threshold used throughout the paper.
A_theory = np.sqrt(3 / 2)
A_acceptable = 6.0

fs = 14  # font size for plots


############################
###### Function block ######
############################

def load_star_spectrum(star_name, LDLs, base_path):
    """
    Assemble the per-channel MCMC posterior summaries for one star into per-limb-darkening-
    law arrays of the posterior median / 16th / 84th percentile / best-fit / standard
    deviation of the radius ratio, indexed by wavelength channel. Channels whose summary.npz
    has not been produced yet (i.e. that combination hasn't been run) are left as NaN.

    Results are cached to a single <star_name>_cache.pkl file so that repeated calls (and
    repeated pipeline runs) don't need to re-scan the many per-channel summary.npz files.
    This cache is shared with Fig6_plot.py, so the schema (all five fields) must match.
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
        r_median  = np.full(n_bins, np.nan)
        r_lo      = np.full(n_bins, np.nan)
        r_hi      = np.full(n_bins, np.nan)
        r_bestfit = np.full(n_bins, np.nan)
        r_std     = np.full(n_bins, np.nan)

        ldl_dir = os.path.join(base_path, star_name, LDL)
        n_done = 0
        for ib in range(n_bins):
            summary_file = os.path.join(ldl_dir, f'channel_{ib:04d}', 'summary.npz')
            if not os.path.exists(summary_file):
                continue
            s = np.load(summary_file)
            r_median[ib]  = s['r_median']
            r_lo[ib]      = s['r_lo']
            r_hi[ib]      = s['r_hi']
            r_bestfit[ib] = s['r_bestfit']
            r_std[ib]     = s['r_std']
            n_done += 1

        print(f'  [{star_name}] {LDL}: {n_done}/{n_bins} channels completed')
        spectra[LDL] = dict(r_median=r_median, r_lo=r_lo, r_hi=r_hi, r_bestfit=r_bestfit, r_std=r_std)

    with open(cache_file, 'wb') as f:
        pickle.dump((wav_centers, spectra), f)

    return wav_centers, spectra


def compute_amplification_factor(r_bestfit, r_std):
    """
    Amplification factor A = (2 * r_std * r_bestfit) / scatter_in_bin for each channel (see
    Fig1_plot.py / Fig5_plot.py), with r_std the standard deviation of the cleaned,
    post-burn-in r chain and r_bestfit the best-fit r, both read directly from summary.npz.
    """
    scatter_in_bin = (model_scatter * 1e-6) / np.sqrt(num_IT_pts)
    return (2 * r_std * r_bestfit) / scatter_in_bin


################################
########## Code block ##########
################################

# Layout: the first four stars fill a 2x2 grid, and the fifth (last) star sits centered
# on its own row below, spanning the same width as one of the upper cells -- same overall
# figure footprint as Fig6_plot.py.
fig = plt.figure(figsize=(16, 12))
# Two independent GridSpecs so the row0-row1 gap can be tightened without touching the
# (larger) gap above the centered row-2 plot.
gs_top = fig.add_gridspec(nrows=2, ncols=4, hspace=0.15, wspace=0.3, top=0.95, bottom=0.42)
gs_bottom = fig.add_gridspec(nrows=1, ncols=4, wspace=0.3, top=0.35, bottom=0.11)
grid_slots = [gs_top[0, 0:2], gs_top[0, 2:4], gs_top[1, 0:2], gs_top[1, 2:4], gs_bottom[0, 1:3]]

axes = []
for slot in grid_slots:
    axes.append(fig.add_subplot(slot, sharex=axes[0] if axes else None))
bottom_axes = {axes[2], axes[3], axes[4]}
right_yaxis_axes = {axes[1], axes[3]}  # second column

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
        r_bestfit = spectra[LDL]['r_bestfit']
        r_std     = spectra[LDL]['r_std']

        good = np.isfinite(r_bestfit) & np.isfinite(r_std)

        amp_factor = compute_amplification_factor(r_bestfit[good], r_std[good])

        ax.plot(
            wav_centers[good], amp_factor,
            marker='.', markersize=4, linewidth=0.8, alpha=0.85,
            color=LDL_colors[LDL], zorder=LDL_zorder[LDL],
            label=LDL_labels[LDL],
        )

    Teff = stellar_types[star_name]['Teff']
    logg = stellar_types[star_name]['logg']
    MH   = stellar_types[star_name]['MH']
    ax.set_title(
        f'Cluster {star_name[1]} '
        f'($T_{{\\rm eff}}$={Teff:.0f} K, $\\log g$={logg:.2f}, [M/H]={MH:.2f})',
        fontsize=fs,
    )

    ax.axhline(A_theory, linestyle='dashed', color='gray', linewidth=1, zorder=1)
    ax.axhspan(A_theory, A_acceptable, facecolor='green', alpha=0.15, edgecolor='none', zorder=0)
    ax.set_yscale('log')
    ax.set_ylabel('Amplification Factor ($A$)', fontsize=fs)
    ax.tick_params(axis='both', labelsize=fs - 2)
    ax.grid(True, alpha=0.3)

for ax in axes:
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=fs - 3, loc='best', framealpha=0.9)
        break

plt.savefig(paths.figures / "Appendix7.pdf", bbox_inches="tight")

# Print a short summary of the amplification factor per star / limb-darkening law (for
# whichever channels have been run so far)
print(f'\n  {"Star":<10}  {"LDL":<22}  {"N channels":>10}  {"A decrease factor":>18}')
print(f'  {"-"*66}')
for star_name in star_order:
    wav_centers, spectra = load_star_spectrum(star_name, LDLs, input_save_path)
    if wav_centers is None:
        continue
    for LDL in LDLs:
        r_bestfit = spectra[LDL]['r_bestfit']
        r_std     = spectra[LDL]['r_std']
        good = np.isfinite(r_bestfit) & np.isfinite(r_std)
        if not np.any(good):
            continue
        amp_factor = compute_amplification_factor(r_bestfit[good], r_std[good])
        print(f'  {star_name:<10}  {LDL_labels[LDL]:<22}  {np.sum(good):>10d}  '
              f'{amp_factor[0]/amp_factor[-1]:>18.2f}')
