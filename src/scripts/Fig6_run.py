#############################
########## Purpose ##########
#############################

# Figure 6 is a simulated transmission spectrum, showing how the choice of limb-darkening
# law biases the retrieved transit depth as a function of wavelength. For three fiducial
# stellar types (an M, a G, and an F dwarf) we:
#
#   1. Build the disc-integrated stellar intensity profile at every wavelength channel of a
#      JWST NIRSpec/PRISM-like grid (0.6-5.3 micron, nominal R~100) using the mps-atlas
#      set 1 (mps1) intensity spectra via exotic_ld, and fit each channel's profile with a
#      4th-order non-linear limb-darkening (NLLD) law -- i.e. the same profile-extraction
#      and NLLD-fitting procedure used in Figure 3, just repeated per wavelength channel
#      instead of once for the disc-integrated (band-averaged) profile. This step is a
#      direct, deterministic lmfit least-squares fit of the *known* stellar intensity
#      profile -- it is not a "retrieval" from noisy data, so it is left untouched here.
#   2. Inject a chromatic transit light curve at a FIXED planet-to-star radius ratio (i.e.
#      the true transmission spectrum is flat) using the wavelength-dependent 4NLLD
#      coefficients from step 1.
#   3. Retrieve the transit depth at every wavelength channel SIMULTANEOUSLY, in a single
#      joint MCMC per (star, limb-darkening law) combination: the wavelength-INdependent
#      orbital parameters (i, a, period, sqrt(e)cos(w), sqrt(e)sin(w)) are shared across
#      every channel's light curve, while the radius ratio and limb-darkening coefficients
#      are free PER channel. This is more statistically correct than fitting each channel
#      fully independently (which was done in earlier versions of this file): it shares the
#      orbital constraint across the whole dataset instead of re-deriving a slightly
#      different, noise-driven orbital solution in every channel, which would otherwise leak
#      extra channel-to-channel scatter into the recovered spectrum on top of the genuine
#      limb-darkening-law bias we're trying to isolate.
#
# The 4th-order NLLD retrieval should recover the injected flat spectrum, while the lower-
# order polynomial laws are expected to imprint spurious spectral features.
#
# Sampler: per the project's request, this remains a jaxoplanet + emcee_jax (affine-
# invariant ensemble, NOT NUTS/HMC) fit, exactly as in Figure 5 -- just with one big joint
# parameter vector instead of many small independent ones. This has real costs worth being
# explicit about:
#   - Dimensionality: ndim = 5 (shared) + n_channels * (1 + n_LD), e.g. for the full 219-
#     channel PRISM grid, ndim ~ 662 (PLD_2), ~881 (PLD_3), ~1100 (4NLLD).
#   - Ensemble size: the default "Stretch" move needs nwalkers well above ndim (the usual
#     rule of thumb is nwalkers >= 2*ndim) to mix at all, so nwalkers scales into the
#     thousands. That directly drives up the per-step cost, since every step evaluates the
#     transit model for every walker.
#   - Storage: the classic (nwalkers, nsteps, ndim) chain array that earlier figures save in
#     full is no longer viable here -- with nwalkers and ndim both in the thousands, even a
#     modest nsteps would require hundreds of GB to TB per run. This file instead saves a
#     fixed-size, randomly-subsampled flock of posterior draws (N_SAVE_SAMPLES rows), which
#     is enough to reconstruct percentiles/marginal histograms without the combinatorial
#     blow-up. See SAVE STRATEGY below.
#
# This script processes ONE (star, limb-darkening law) combination per invocation -- only
# 3 stars x 3 laws = 9 combinations total, each a single (larger) joint MCMC, replacing the
# ~2,000 small independent MCMCs of the previous per-channel design.


######################################
########## Import libraries ##########
######################################

from jax import random
import jax
print(f"JAX devices: {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")
import jax.numpy as jnp
import emcee_jax
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from jaxoplanet.orbits.keplerian import System, Central
from jaxoplanet.light_curves import limb_dark_light_curve
from squishyplanet.limb_darkening_laws import nonlinear_4param_ld_law

import astropy.units as u
from astropy.constants import G

from scipy.interpolate import interp1d
from lmfit import minimize, Parameters
import exotic_ld as el

import corner
import time
import arviz as az
import numpy as np
import os, itertools, sys
import paths

# For 64-bit precision since JAX defaults to 32-bit
jax.config.update("jax_enable_x64", True)

# Set random seed used for the MCMC walker initialisation (kept fixed across every
# combination -- only the noise realisation below varies star to star).
jaxnoise_key = jax.random.PRNGKey(0)


#############################################
########## Define hyper-parameters ##########
#############################################

#%% Stellar intensity data
LD_data_path = '/Volumes/Ajax/Work/PhD/Research/Transit-Information-Content/LD_simulation'
ld_model     = 'mps1'  # mps-atlas set 1

#%% Three fiducial stellar types spanning the mps1 grid (Teff 3500-9000 K, logg 3.0-5.0,
#%% [M/H] -5.0-1.5). Metallicity and logg are held fixed at typical dwarf-star values so
#%% that Teff is the main axis of variation between the three cases.
stellar_types = {
    'M_dwarf': {'Teff': 3500.0, 'logg': 4.8, 'MH': 0.0},
    'G_dwarf': {'Teff': 5800.0, 'logg': 4.5, 'MH': 0.0},
    'F_dwarf': {'Teff': 7200.0, 'logg': 4.3, 'MH': 0.0},
}
star_names = list(stellar_types.keys())

#%% JWST NIRSpec/PRISM-like wavelength grid
wav_min_um = 0.6    # micron
wav_max_um = 5.3    # micron
R_prism    = 100    # nominal (constant) resolving power, lambda / delta_lambda

#%% Number of mu values to interpolate the intensity profile to (as in Fig3)
n_mu_fine = 100

#%% Seed for the random initial guesses used when fitting each wavelength bin's 4th-order
#%% NLLD coefficients (see extract_wavelength_LDCs) -- fixes coeffs below. This is the
#%% deterministic profile characterisation step, not the MCMC retrieval.
fit_init_seed = 42

#%% Base seed for the per-star noise draw. Depends only on the star -- NOT on the limb-
#%% darkening law -- so that all three retrieval laws are jointly fit to the exact same
#%% multi-channel noisy dataset, making the comparison between laws fair.
noise_seed_base = 1000

#%% Limb-darkening laws used for the retrieval
LDLs = ['PLD_2', 'PLD_3', '4NLLD']

#%% Single limb-darkening prior strength used throughout (as in the previous version of
#%% this file -- Figure 5's grid of prior strengths is not replicated here).
prior_strength = 'gauss_10'

#%%%% Define G in units needed now to avoid JAX tracing issues
G_solar_units = G.to(u.Rsun**3 / (u.Msun * u.day**2)).value
R_star = (1.0 * u.R_sun).value

#%%%% Mock system - fiducial (same geometry as Figures 4 and 5; only the limb-darkening
#%%%% coefficients change with wavelength and stellar type here)
init_state_dic = {}
init_state_dic['period'] = 1.0                                #days
a_meters = ((G.value * (1.0 * u.M_sun).to(u.kg).value * (init_state_dic['period'] * 24 * 3600)**2) / (4 * jnp.pi**2))**(1 / 3)
init_state_dic['a'] = a_meters / (1.0 * u.R_sun).to(u.m).value #stellar radii
init_state_dic['r'] = 0.1                                      #stellar radii -- FIXED across wavelength (flat true spectrum)
init_state_dic['i'] = jnp.deg2rad(90)                           #radians
init_state_dic['omega'] = 0.0                                   #radians
init_state_dic['e'] = 0.0                                       #unitless
init_state_dic['t0'] = 0.0                                      #days

#%%%% Calculate transit duration
b = (
    (init_state_dic['a'] * jnp.cos(init_state_dic['i'])) / R_star
    * (1 - init_state_dic['e']**2) / (1 + init_state_dic['e'] * jnp.sin(init_state_dic['omega']))
)
arg = (
    (1 / init_state_dic['a'])
    * jnp.sqrt((1 + init_state_dic['r'])**2 - b**2)
    / jnp.sin(init_state_dic['i'])
)
arg = np.clip(arg, -1.0, 1.0)
T_dur = (
    (init_state_dic['period'] / jnp.pi)
    * jnp.sqrt(1 - init_state_dic['e']**2) / (1 + init_state_dic['e'] * jnp.sin(init_state_dic['omega']))
    * jnp.arcsin(arg)
)

#%%%% Model time - ensure pre- and post-transit are same duration as transit
low_t = -1.5 * T_dur                                                     #days
high_t = 1.5 * T_dur                                                     #days
exposure_time = 5                                                        #seconds
num_t = jnp.floor((((high_t - low_t) * 24 * 3600) / exposure_time))      #number of points
times = jnp.linspace(low_t, high_t, int(num_t))                          #days

#%% Fiducial photometric scatter (same value used throughout Figures 4 and 5) applied
#%% identically to every wavelength channel, so that the recovered spectrum isolates the
#%% limb-darkening-law bias rather than a wavelength-dependent noise trend.
model_scatter = 16.68100537200059  # ppm

#%% Shared (wavelength-independent) orbital parameters -- guess/bounds are only used to
#%% initialise the MCMC walkers; the actual priors used in the posterior are in
#%% shared_priors below. t0 is fixed, exactly as in Fig5_run.py.
shared_mod_prop = {
    'i'         : {'guess':jnp.deg2rad(88.5), 'bounds':[jnp.deg2rad(88.), jnp.deg2rad(92.)]},
    'a'         : {'guess':init_state_dic['a']-1, 'bounds':[init_state_dic['a']-2, init_state_dic['a']+2]},
    'period'    : {'guess':1., 'bounds':[0.9995, 1.0005]},
    'sqrtecosw' : {'guess': 0., 'bounds': [-0.2, 0.2]},
    'sqrtesinw' : {'guess': 0., 'bounds': [-0.2, 0.2]},
}
shared_param_names = list(shared_mod_prop.keys())  # ['i', 'a', 'period', 'sqrtecosw', 'sqrtesinw']
n_shared = len(shared_param_names)

shared_priors = {
    'i'             : {'type':'uf', 'bounds':[jnp.deg2rad(70), jnp.deg2rad(110)]},
    'a'             : {'type':'uf', 'bounds':[0, 50]},
    'period'        : {'type':'gauss', 'val':1.0000, 's_val':0.0005},
    'sqrtecosw'     : {'type':'uf', 'bounds':[-1, 1]},
    'sqrtesinw'     : {'type':'uf', 'bounds':[-1, 1]},
}

#%% Per-channel radius-ratio init bounds (used only for walker initialisation, exactly like
#%% Fig5_run.py's r entry); the actual prior on every channel's r is broad/uniform (see
#%% below), not this tight range.
r_init_bounds = [0.07, 0.15]
r_prior_bounds = [0., 1.]

#%% MCMC settings. nwalkers is set automatically from ndim below (the default "Stretch"
#%% move needs nwalkers well above ndim to mix -- see the module docstring). nsteps/nburn
#%% are deliberately far below Figure 5's 100,000/70,000: even with the subsampled-posterior
#%% save strategy below, the wall-clock cost per step scales with nwalkers x n_channels x
#%% len(times), so start small, check the printed cost estimate before scaling up.
nwalkers_floor = 200          # minimum, before the 2*ndim+2 rule kicks in
nsteps         = 20000
nburn          = 10000

#%% SAVE STRATEGY: rather than the full (nwalkers, nsteps, ndim) chain (infeasible at this
#%% scale -- see module docstring), save a fixed-size random subsample of the flattened,
#%% post-burn-in ensemble (every walker's post-burn steps pooled together, then subsampled
#%% down to N_SAVE_SAMPLES rows). This is enough to reconstruct percentiles and marginal
#%% distributions for every parameter (shared and per-channel) without the storage blow-up.
N_SAVE_SAMPLES = 5000

#%% Whether to additionally produce a closer-look diagnostic suite for this combination:
#%% a corner + trace plot of the (small, 5-dim) SHARED parameters only, and an
#%% autocorrelation-time plot for the shared inclination and one representative channel's r.
#%% A full corner/trace plot of every parameter is not meaningful at this dimensionality, so
#%% this is not a "full Fig5-style suite" over all ~1000 dimensions -- just the shared block.
make_full_diagnostics = False

#%% Output directory
orig_save_data_path = str(paths.data / "Fig6_Storage") + "/"

#############################################
########## Define parallelization ##########
#############################################

# # Distribute tasks - for HPC usage
# task_arrays = [star_names, LDLs]
# param_combos = list(itertools.product(*task_arrays))
# param_combos = [list(c) for c in param_combos]
#
# my_task_id = int(sys.argv[1])
# star_name, LDL = param_combos[my_task_id - 1]

star_name = 'G_dwarf'
LDL       = '4NLLD'


############################
###### Function block ######
############################

def check_dir(dir_name):
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    return dir_name


def build_R_grid(wav_min, wav_max, R):
    """
    Build a constant-resolving-power (R = lambda / delta_lambda) wavelength grid.

    Returns
    -------
    edges   : np.ndarray, bin edges (same units as wav_min / wav_max)
    centers : np.ndarray, bin centres
    """
    n_edges = int(np.ceil(np.log(wav_max / wav_min) / np.log(1.0 + 1.0 / R))) + 1
    edges = wav_min * (1.0 + 1.0 / R) ** np.arange(n_edges)
    edges = edges[edges < wav_max]
    edges = np.append(edges, wav_max)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return edges, centers


def fourNLLD(x, coeffs):
    """4th-order non-linear limb-darkening law."""
    return (1
            - coeffs[0] * (1 - x ** 0.5)
            - coeffs[1] * (1 - x)
            - coeffs[2] * (1 - x ** 1.5)
            - coeffs[3] * (1 - x ** 2))


def residual_fn(params, x, base_prof):
    """Residual function for lmfit minimisation of NLLD coefficients."""
    return fourNLLD(x, [params[f'c{ic+1}'].value for ic in range(4)]) - base_prof


def extract_wavelength_LDCs(Teff, logg, MH, wav_edges_um, ld_data_path, n_mu_fine=100, fit_seed=42):
    """
    Build a single star's disc-integrated intensity profile with exotic_ld (mps1 model,
    same call as Fig3_run.py) and fit a 4th-order NLLD to the profile averaged within each
    wavelength bin of `wav_edges_um`. This is a deterministic characterisation of the known
    stellar model (identical in spirit to Fig3_run.py's profile fit) -- it is not part of
    the MCMC light-curve retrieval below.

    Returns
    -------
    coeffs      : (n_bins, 4) best-fit [c1, c2, c3, c4] per wavelength bin (NaN if empty)
    bin_profile : (n_bins, n_mu_fine) binned, centre-normalised intensity profile
    mus_fine    : (n_mu_fine,) common mu grid (centre -> limb)
    """
    print(f'  Building intensity profiles for Teff={Teff:.0f} K, logg={logg:.2f}, [M/H]={MH:.2f}')
    sld = el.StellarLimbDarkening(
        M_H=MH, Teff=Teff, logg=logg,
        ld_model=ld_model, ld_data_path=ld_data_path,
        interpolate_type='nearest',
    )
    stellar_wavelengths = np.array(sld.stellar_wavelengths)  # Angstrom, (n_wav,)
    stellar_mus         = np.array(sld.mus)                  # (n_mu,)
    stellar_intensities = np.array(sld.stellar_intensities)  # (n_wav, n_mu)
    del sld

    # Interpolate onto a fine, uniform mu grid (centre -> limb), exactly as in Fig3_run.py
    mus_fine = np.linspace(stellar_mus[-1], stellar_mus[0], n_mu_fine)
    interp_func = interp1d(
        stellar_mus[::-1], stellar_intensities[:, ::-1],
        kind='cubic', axis=1, bounds_error=False,
    )
    intensities_fine = interp_func(mus_fine)   # (n_wav, n_mu_fine)
    mus_fine          = mus_fine[::-1]
    intensities_fine  = intensities_fine[:, ::-1]  # mu=1 (centre) -> mu~0 (limb)

    wav_edges_ang = wav_edges_um * 1e4
    n_bins = len(wav_edges_ang) - 1

    coeffs      = np.full((n_bins, 4), np.nan)
    bin_profile = np.full((n_bins, n_mu_fine), np.nan)

    fit_rng = np.random.default_rng(fit_seed)

    for ib in range(n_bins):
        in_bin = (
            (stellar_wavelengths >= wav_edges_ang[ib]) &
            (stellar_wavelengths < wav_edges_ang[ib + 1])
        )
        if not np.any(in_bin):
            continue

        profile = intensities_fine[in_bin].mean(axis=0)  # (n_mu_fine,)
        center  = profile[0] if abs(profile[0]) > 1e-10 else 1.0
        profile = profile / center

        params = Parameters()
        for ip in range(4):
            params.add(f'c{ip+1}', value=fit_rng.uniform(0, 1))
        try:
            result = minimize(residual_fn, params, args=(mus_fine, profile))
            coeffs[ib] = [result.params[f'c{ic+1}'].value for ic in range(4)]
        except Exception:
            coeffs[ib] = np.nan

        bin_profile[ib] = profile

    return coeffs, bin_profile, mus_fine


def uniform_logpdf(x, lo, hi):
    return jnp.where((x >= lo) & (x <= hi), -jnp.log(hi - lo), -jnp.inf)


def gauss_logpdf(x, val, s):
    return -0.5 * (jnp.log(2 * jnp.pi * s**2) + ((x - val) / s)**2)


def single_channel_lc(r_i, u_i, i_, a_, period_, ecc, w, convert_NLLD, times):
    """Transit light curve for one wavelength channel, given the shared orbital
    parameters and this channel's own radius ratio / limb-darkening coefficients. When
    `convert_NLLD` is True, `u_i` is the channel's 4-parameter NLLD coefficients
    [c1, c2, c3, c4], converted to an order-12 polynomial (a numerically near-exact
    representation of the true NLLD profile, same trick used for injection and the 4NLLD
    retrieval throughout this project); otherwise `u_i` is used directly as the native-
    basis polynomial limb-darkening coefficients (quadratic or 3rd-order law)."""
    stellar_rho = (3 * jnp.pi * a_**3) / (period_**2 * G_solar_units)
    star = Central(density=stellar_rho)
    planet = System(star).add_body(
        time_transit=0.0,
        period=period_,
        inclination=i_,
        eccentricity=ecc,
        omega_peri=w,
        radius=r_i * R_star,
    )
    if convert_NLLD:
        ld_u = nonlinear_4param_ld_law(u1=u_i[0], u2=u_i[1], u3=u_i[2], u4=u_i[3], order=12)
    else:
        ld_u = u_i
    return (1.0 + limb_dark_light_curve(planet, ld_u)(times)).reshape(-1)


def make_joint_log_prob(n_bins, n_u, convert_NLLD, r_lo, r_hi, LD_prior_val, LD_prior_scale):
    """
    Build the joint log-probability function for one (star, LDL) combination.

    Parameter vector layout (theta, length n_shared + n_bins*(1+n_u)):
        theta[0:5]                                        = [i, a, period, sqrtecosw, sqrtesinw]
        theta[5 + c*(1+n_u)]                               = r_c            (channel c)
        theta[5 + c*(1+n_u) + 1 : 5 + c*(1+n_u) + 1 + n_u] = LD_c[0..n_u-1] (channel c)

    LD_prior_val / LD_prior_scale: (n_bins, n_u) gaussian-prior centre/width per channel's
    limb-darkening coefficients (derived from that channel's own true NLLD profile).
    """
    def joint_log_probability(theta, times, data, err):

        i_, a_, period_, sqrtecosw_, sqrtesinw_ = theta[:n_shared]
        per_channel = theta[n_shared:].reshape(n_bins, 1 + n_u)
        r_c = per_channel[:, 0]
        u_c = per_channel[:, 1:]

        ecc = sqrtecosw_**2 + sqrtesinw_**2
        w = jnp.arctan2(sqrtesinw_, sqrtecosw_)

        #%% Priors
        lp = jnp.where(ecc <= 1.0, 0.0, -jnp.inf)
        lp += uniform_logpdf(i_, shared_priors['i']['bounds'][0], shared_priors['i']['bounds'][1])
        lp += uniform_logpdf(a_, shared_priors['a']['bounds'][0], shared_priors['a']['bounds'][1])
        lp += gauss_logpdf(period_, shared_priors['period']['val'], shared_priors['period']['s_val'])
        lp += uniform_logpdf(sqrtecosw_, -1., 1.)
        lp += uniform_logpdf(sqrtesinw_, -1., 1.)
        lp += jnp.sum(uniform_logpdf(r_c, r_lo, r_hi))
        lp += jnp.sum(gauss_logpdf(u_c, LD_prior_val, LD_prior_scale))
        lp = jnp.where(jnp.isfinite(lp), lp, -jnp.inf)

        #%% Likelihood -- vmap the single-channel model over all wavelength channels at once
        model = jax.vmap(single_channel_lc, in_axes=(0, 0, None, None, None, None, None, None, None))(
            r_c, u_c, i_, a_, period_, ecc, w, convert_NLLD, times
        )
        step_chi2 = jnp.sum((model - data)**2 / err**2)
        lk = -0.5 * (step_chi2 + jnp.sum(jnp.log(2 * jnp.pi * err**2)))
        lk = jnp.where(jnp.isnan(lk), -jnp.inf, lk)

        return lp + lk, {'step_chi2': step_chi2}

    return joint_log_probability


def next_pow_two(n):
    """Find the next power of two greater than or equal to n."""
    i = 1
    while i < n:
        i = i << 1
    return i


def autocorr_func_1d(x, norm=True):
    """Estimate the normalized autocorrelation function of a 1-D series via FFT."""
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))
    f = np.fft.fft(x - np.mean(x), n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
    acf /= 4 * n
    if norm:
        acf /= acf[0]
    return acf


def auto_window(taus, c):
    """Automated windowing procedure following Sokal (1989)."""
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1


def autocorr_new(y, c=5.0):
    """Improved autocorrelation-time estimator. Computes per-walker ACF then averages."""
    f = np.zeros(y.shape[1])
    for yy in y:
        f += autocorr_func_1d(yy)
    f /= len(y)
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return taus[window]


#############################################
################ Running code ###############
#############################################

star_props = stellar_types[star_name]
star_index = star_names.index(star_name)

print(f'RUNNING: star={star_name}, LDL={LDL}')

# JWST NIRSpec/PRISM-like wavelength grid (constant R~100, 0.6-5.3 micron)
wav_edges, wav_centers = build_R_grid(wav_min_um, wav_max_um, R_prism)
n_bins = len(wav_centers)
print(f'Wavelength grid: {n_bins} channels between {wav_min_um} and {wav_max_um} micron at R={R_prism}')

save_path = check_dir(orig_save_data_path + f'{star_name}/')
fixed_args_save_loc = check_dir(save_path + f'{LDL}/')

# ── Step 1: wavelength-dependent 4th-order NLLD coefficients (Fig3 procedure) ────────────
coeffs, _, _ = extract_wavelength_LDCs(
    star_props['Teff'], star_props['logg'], star_props['MH'],
    wav_edges, LD_data_path, n_mu_fine=n_mu_fine, fit_seed=fit_init_seed,
)
valid = ~np.any(np.isnan(coeffs), axis=1)
if not np.all(valid):
    raise RuntimeError(f'{np.sum(~valid)} channel(s) have no valid intensity data for {star_name} -- '
                        f'the joint fit requires every channel to be usable.')

wav_grid_file = save_path + 'wav_grid.npz'
if not os.path.exists(wav_grid_file):
    np.savez(wav_grid_file, wav_edges=wav_edges, wav_centers=wav_centers, coeffs=coeffs)

############################################
########## Setting limb-darkening ##########
############################################

strength = int(prior_strength.split('_')[1])

if 'PLD' in LDL:
    n_u = int(LDL[-1])
    convert_NLLD = False
    best_fit_LDCs_all = np.array([
        np.asarray(nonlinear_4param_ld_law(*coeffs[c], order=n_u)) for c in range(n_bins)
    ])  # (n_bins, n_u)
elif LDL == '4NLLD':
    n_u = 4
    convert_NLLD = True
    best_fit_LDCs_all = coeffs  # (n_bins, 4) -- true coefficients used directly
else:
    raise KeyError('Wrong limb darkening law.')

LD_prior_val   = jnp.array(best_fit_LDCs_all)
LD_prior_scale = jnp.array(np.abs(best_fit_LDCs_all) * (strength / 100))

#######################################
##### Finalizing hyper-parameters #####
#######################################

ndim = n_shared + n_bins * (1 + n_u)
nwalkers = max(nwalkers_floor, 2 * ndim + 2)
if nwalkers % 2 == 1:
    nwalkers += 1  # emcee's ensemble moves split walkers into two halves

chain_bytes_full = nwalkers * nsteps * ndim * 8
chain_bytes_saved = N_SAVE_SAMPLES * ndim * 8
print(f'ndim = {ndim} (5 shared + {n_bins} channels x {1+n_u})')
print(f'nwalkers = {nwalkers}, nsteps = {nsteps}, nburn = {nburn}')
print(f'Full (nwalkers, nsteps, ndim) chain would be {chain_bytes_full/1e9:.1f} GB -- NOT saved in full.')
print(f'Saving a {N_SAVE_SAMPLES}-sample subsample of the posterior instead ({chain_bytes_saved/1e6:.1f} MB).')

#Uniform distribution of walker starting positions -- one vectorised draw per parameter
#dimension (a Python loop over ~1000 dimensions, as Fig5_run.py does for its ~10, would be
#needlessly slow here).
minval = np.empty(ndim)
maxval = np.empty(ndim)
for k, name in enumerate(shared_param_names):
    minval[k], maxval[k] = shared_mod_prop[name]['bounds']
for c in range(n_bins):
    base = n_shared + c * (1 + n_u)
    minval[base], maxval[base] = r_init_bounds
    minval[base+1:base+1+n_u] = best_fit_LDCs_all[c] - 0.5
    maxval[base+1:base+1+n_u] = best_fit_LDCs_all[c] + 0.5

emceejax_key1, emceejax_key2, pos_key = jax.random.split(jaxnoise_key, 3)
pos = jax.random.uniform(pos_key, minval=jnp.array(minval), maxval=jnp.array(maxval), shape=(nwalkers, ndim))
pos = np.asarray(pos)

#############################
####### Generate data #######
#############################
print('GENERATING DATA')

#Pure (noiseless) multi-channel data using the near-exact order-12 polynomial
#approximation of each channel's true 4NLLD profile (same trick as Fig4/Fig5_run.py).
true_u_poly = jnp.array(np.array([
    np.asarray(nonlinear_4param_ld_law(*coeffs[c], order=12)) for c in range(n_bins)
]))
true_r = jnp.full(n_bins, init_state_dic['r'])
true_lc = jax.vmap(single_channel_lc, in_axes=(0, 0, None, None, None, None, None, None, None))(
    true_r, true_u_poly, init_state_dic['i'], init_state_dic['a'], init_state_dic['period'],
    init_state_dic['e'], init_state_dic['omega'], False, times,
)  # (n_bins, n_times)

#Build noisy data -- noise realisation depends only on the star, not on LDL
noise_seed = noise_seed_base + star_index
std = model_scatter * 1e-6
noisy_data = true_lc + std * random.normal(jax.random.PRNGKey(noise_seed), shape=true_lc.shape)
noisy_std = std * jnp.ones(true_lc.shape, dtype=float)

print(f"initial chi2: {jnp.sum((true_lc - noisy_data)**2 / noisy_std**2)}")

#########################
##### Emcee fitting #####
#########################

log_prob_fn = make_joint_log_prob(n_bins, n_u, convert_NLLD, r_prior_bounds[0], r_prior_bounds[1],
                                   LD_prior_val, LD_prior_scale)

print("Running MCMC")
st0 = time.time()
sampler = emcee_jax.EnsembleSampler(log_prob_fn, log_prob_args=(times, noisy_data, noisy_std))
state = sampler.init(emceejax_key1, pos)
trace = sampler.sample_parallel(emceejax_key2, state, num_steps=nsteps, progress=True)
raw_chain = np.asarray(trace.samples.coordinates).reshape(nwalkers, nsteps, ndim)
logprob = np.asarray(trace.samples.log_probability.T)             # (nwalkers, nsteps)
chi2_chain = np.asarray(trace.samples.deterministics['step_chi2'].T)  # (nwalkers, nsteps)
elapsed = time.time() - st0
print(f'MCMC took {elapsed:.2f} seconds / {elapsed/60.:.2f} minutes / {elapsed/3600.:.2f} hours.')

#%% Post-burn-in, flattened-and-subsampled posterior (see SAVE STRATEGY above)
post_chain = raw_chain[:, nburn:, :].reshape(-1, ndim)
post_logprob = logprob[:, nburn:].reshape(-1)
post_chi2 = chi2_chain[:, nburn:].reshape(-1)

save_rng = np.random.default_rng(0)
n_save = min(N_SAVE_SAMPLES, post_chain.shape[0])
save_idx = save_rng.choice(post_chain.shape[0], size=n_save, replace=False)

np.save(fixed_args_save_loc + 'posterior_samples.npy', post_chain[save_idx])
np.save(fixed_args_save_loc + 'posterior_logprob.npy', post_logprob[save_idx])
np.save(fixed_args_save_loc + 'posterior_chi2.npy', post_chi2[save_idx])

##################
#### Summaries ###
##################
print('SUMMARISING')

max_walker, max_step = np.unravel_index(np.argmax(logprob), logprob.shape)
bestfit_theta = raw_chain[max_walker, max_step, :]

shared_post = post_chain[:, :n_shared]
shared_median = np.median(shared_post, axis=0)
shared_lo, shared_hi = np.percentile(shared_post, [16, 84], axis=0)
shared_bestfit = bestfit_theta[:n_shared]

per_channel_post = post_chain[:, n_shared:].reshape(-1, n_bins, 1 + n_u)
r_post = per_channel_post[:, :, 0]                    # (n_post, n_bins)
r_median = np.median(r_post, axis=0)
r_lo, r_hi = np.percentile(r_post, [16, 84], axis=0)
r_bestfit = bestfit_theta[n_shared:].reshape(n_bins, 1 + n_u)[:, 0]

try:
    i_tau = float(autocorr_new(raw_chain[:, nburn:, shared_param_names.index('i')]))
except Exception:
    i_tau = np.nan

np.savez(
    fixed_args_save_loc + 'summary.npz',
    wav_centers=wav_centers,
    true_LDCs=coeffs,
    r_true=init_state_dic['r'],
    r_median=r_median, r_lo=r_lo, r_hi=r_hi, r_bestfit=r_bestfit,
    shared_param_names=np.array(shared_param_names),
    shared_median=shared_median, shared_lo=shared_lo, shared_hi=shared_hi, shared_bestfit=shared_bestfit,
    i_autocorr_time=i_tau,
    n_steps_post_burn=nsteps - nburn,
    nwalkers=nwalkers, ndim=ndim,
)
print(f'  shared params (median [16,84], bestfit):')
for k, name in enumerate(shared_param_names):
    print(f'    {name:<10} {shared_median[k]:.5f} [{shared_lo[k]:.5f}, {shared_hi[k]:.5f}]  bestfit={shared_bestfit[k]:.5f}')
print(f'  tau(i) = {i_tau:.1f}')
print(f'  r range across channels: median [{r_median.min():.5f}, {r_median.max():.5f}]')

##################
#### Plotting ####
##################
print('PLOTTING')

#Lightweight diagnostic 1: recovered spectrum preview
fig_spec, ax_spec = plt.subplots(figsize=(9, 4))
true_depth_ppm = 1e6 * init_state_dic['r']**2
ax_spec.axhline(true_depth_ppm, color='gray', linestyle='--', label='Injected (flat) depth')
depth_ppm = 1e6 * r_median**2
depth_err_ppm = np.vstack([1e6*(r_median**2-r_lo**2), 1e6*(r_hi**2-r_median**2)])
ax_spec.errorbar(wav_centers, depth_ppm, yerr=depth_err_ppm, fmt='.', markersize=4,
                  elinewidth=0.8, color='C0', label=f'{LDL} (joint fit)')
ax_spec.set_xlabel('Wavelength (micron)')
ax_spec.set_ylabel('Transit depth (ppm)')
ax_spec.set_title(f'{star_name}, {LDL} -- joint fit preview')
ax_spec.legend(fontsize=9)
ax_spec.grid(True, alpha=0.3)
fig_spec.tight_layout()
fig_spec.savefig(fixed_args_save_loc + 'spectrum_preview.pdf')
plt.close(fig_spec)

#Lightweight diagnostic 2: one representative channel's data vs best-fit/median model
rep_c = n_bins // 2
def _extract_channel_params(theta_flat, c):
    i_, a_, period_, sqrtecosw_, sqrtesinw_ = theta_flat[:n_shared]
    ecc = sqrtecosw_**2 + sqrtesinw_**2
    w = jnp.arctan2(sqrtesinw_, sqrtecosw_)
    per_channel = theta_flat[n_shared:].reshape(n_bins, 1 + n_u)
    r_c = per_channel[c, 0]
    u_c = per_channel[c, 1:]
    return i_, a_, period_, ecc, w, r_c, u_c

i_b, a_b, period_b, ecc_b, w_b, r_b, u_b = _extract_channel_params(jnp.array(bestfit_theta), rep_c)
bestfit_lc_rep = single_channel_lc(r_b, u_b, i_b, a_b, period_b, ecc_b, w_b, convert_NLLD, times)

median_theta = np.concatenate([shared_median, per_channel_post.mean(axis=0).reshape(-1)])
i_m, a_m, period_m, ecc_m, w_m, r_m, u_m = _extract_channel_params(jnp.array(median_theta), rep_c)
median_lc_rep = single_channel_lc(r_m, u_m, i_m, a_m, period_m, ecc_m, w_m, convert_NLLD, times)

fig_c, (axc1, axc2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios':[3,1]}, figsize=(10, 5))
axc1.errorbar(times, noisy_data[rep_c], yerr=noisy_std[rep_c], fmt='.', color='0.7', label='Data')
axc1.plot(times, median_lc_rep, color='orange', label='Median joint model')
axc1.plot(times, bestfit_lc_rep, color='blue', label='Best-fit joint model')
axc2.errorbar(times, 1e6*(bestfit_lc_rep - noisy_data[rep_c]), yerr=noisy_std[rep_c], fmt='.', color='blue')
axc2.axhline(0, color='k', linewidth=0.8)
axc1.set_title(f'{star_name}, {LDL} -- representative channel {rep_c} ({wav_centers[rep_c]:.2f} micron)')
axc1.set_ylabel('Relative flux')
axc2.set_ylabel('Residuals (ppm)')
axc2.set_xlabel('Time (days)')
axc1.legend(fontsize=9)
fig_c.tight_layout()
fig_c.savefig(fixed_args_save_loc + f'bestfit_channel{rep_c:04d}.pdf')
plt.close(fig_c)

# ── Optional closer-look diagnostics for the shared parameters only ───────────────────────
if make_full_diagnostics:

    shared_inf_data = az.from_dict(
        posterior={name: raw_chain[:, nburn:, k] for k, name in enumerate(shared_param_names)},
        log_likelihood={"log_like": logprob},
    )
    _ = az.plot_trace(shared_inf_data, var_names=shared_param_names,
                       backend_kwargs={"constrained_layout": True})
    plt.savefig(fixed_args_save_loc + 'trace_shared.pdf')
    plt.close()

    fig_corner = corner.corner(
        shared_inf_data, labels=shared_param_names, show_titles=True,
        title_kwargs={"fontsize": 10}, label_kwargs={"fontsize": 10}, title_fmt=".5f",
    )
    plt.savefig(fixed_args_save_loc + 'corner_shared.pdf')
    plt.close()

    n_eval = 15
    postburn_chain = raw_chain[:, nburn:, :]
    nsteps_postburn = postburn_chain.shape[1]
    N_eval = np.unique(np.exp(np.linspace(np.log(100), np.log(nsteps_postburn), n_eval)).astype(int))

    fig_ac, ax_ac = plt.subplots(figsize=(6, 4))
    i_idx = shared_param_names.index('i')
    tau_vals = np.full(len(N_eval), np.nan)
    for idx, n in enumerate(N_eval):
        try:
            tau_vals[idx] = autocorr_new(postburn_chain[:, :n, i_idx])
        except Exception:
            pass
    ax_ac.loglog(N_eval, tau_vals, "o-", color="C1", markersize=4, label='i (shared)')
    ax_ac.loglog(N_eval, N_eval / 50.0, "--k", label=r"$\tau = N/50$")
    ax_ac.set_xlabel("Number of samples, $N$")
    ax_ac.set_ylabel(r"$\tau$ estimate")
    ax_ac.legend(fontsize=8)
    fig_ac.tight_layout()
    fig_ac.savefig(fixed_args_save_loc + 'autocorrelation_shared.pdf')
    plt.close(fig_ac)
