#############################
########## Purpose ##########
#############################

# Figure 6 is a simulated transmission spectrum, showing how the choice of limb-darkening
# law biases the retrieved transit depth as a function of wavelength. For five
# stars, informed from our clustering, we:
#
#   1. Build the intensity profile at every wavelength using the mps1 intensity spectra via exotic_ld,
#      and fit each channel's profile with a 4th-order non-linear limb-darkening (NLLD) law.
#   2. Inject a chromatic transit light curve with an achromatic planet-to-star radius ratio (i.e.
#      the true transmission spectrum is flat) using the wavelength-dependent 4NLLD
#      coefficients from step 1.
#   3. Produce and fit a WLC to constrain the wavelength-independent orbital parameters.
#   4. Retrieve the transit depth and LDCs at every wavelength channel, with an MCMC.
#
# This file (Fig6_prerun.py) performs steps 1-3 and saves the results, for all five stars.


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
import numpy as np
import os
from tqdm.auto import tqdm
import paths

# For 64-bit precision since JAX defaults to 32-bit
jax.config.update("jax_enable_x64", True)

# Set random seed used for the MCMC walker initialisation (kept fixed across every star --
# only the noise realisation below varies star to star).
jaxnoise_key = jax.random.PRNGKey(0)


#############################################
########## Define hyper-parameters ##########
#############################################

#%% Stellar intensity data
LD_data_path = '/Volumes/Ajax/Work/PhD/Research/Transit-Information-Content/LD_simulation'
ld_model     = 'mps1'  # mps-atlas set 1

#%% Five fiducial stellar types informed from our clustering
stellar_types = {
    'C5': {'Teff': 3500.0, 'logg': 4.33, 'MH': 0.06}, # M-star
    'C1': {'Teff': 4111.0, 'logg': 4.11, 'MH': 0.06}, # K-star
    'C2': {'Teff': 4111.0, 'logg': 4.78, 'MH': 0.78}, # K-star
    'C7': {'Teff': 5944.0, 'logg': 4.33, 'MH': 0.06}, # G-star
    'C6': {'Teff': 6556.0, 'logg': 3.89, 'MH': 0.06}, # F-star
}
star_names = list(stellar_types.keys())

#%% JWST NIRSpec/PRISM wavelength grid
wav_min_um = 0.6    # micron
wav_max_um = 5.3    # micron
R_prism    = 100    # nominal (constant) resolving power, lambda / delta_lambda

#%% Number of mu values to interpolate the intensity profile to (as in Fig3)
n_mu_fine = 100

#%% Seed for the random initial guesses used when fitting each wavelength bin's 4th-order
#%% NLLD coefficients (see extract_wavelength_LDCs) -- fixes coeffs below.
fit_init_seed = 42

#%% Base seed for the per-star noise draw. Fig6_run.py regenerates the same noisy
#%% per-channel dataset from this seed,
#%% so the per-channel retrievals in Fig6_run.py operate on the same data that
#%% constrained the orbital parameters here.
noise_seed_base = 1000

#%%%% Define G in units needed now to avoid JAX tracing issues
G_solar_units = G.to(u.Rsun**3 / (u.Msun * u.day**2)).value
R_star = (1.0 * u.R_sun).value

#%%%% Mock system - fiducial
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

#%% Fiducial photometric scatter
model_scatter = 16.68100537200059  # ppm

#%% Achromatic orbital parameters
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

#%% WLC radius-ratio / limb-darkening init bounds
r_init_bounds   = [0.07, 0.15]
r_prior_bounds  = [0., 1.]
LD_prior_bounds = [-100., 100.]

#%% WLC MCMC settings
wlc_nwalkers = 50
wlc_nsteps   = 100000
wlc_nburn    = 70000

#%% Chain-cleaning settings -- same iterative 2D sigma-clipping procedure used in
#%% Fig5_plot.py / Fig4_run.py's load_result, applied here right after the WLC MCMC so the
#%% saved bestfit/median/percentile summaries are already based on the cleaned chain rather
#%% than the raw one (walker excursions to bad regions, temporary stuck walkers, etc. are
#%% removed before any downstream use).
SIGMA_THRESHOLDS = [5, 4, 3]   # IQR multiples, one per round
SIGMA_ROUNDS     = 3

#%% Output directory
orig_save_data_path = str(paths.data / "Fig6_Storage") + "/"


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
    the MCMC light-curve retrieval.

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
    """Transit light curve for one channel (a single wavelength bin, or the co-added white
    light curve), given the orbital parameters and this channel's own radius ratio / limb-
    darkening coefficients. When `convert_NLLD` is True, `u_i` is 4-parameter NLLD
    coefficients [c1, c2, c3, c4], converted to an order-12 polynomial (a numerically near-
    exact representation of the true NLLD profile); otherwise `u_i` is used directly as the
    native-basis polynomial limb-darkening coefficients."""
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


def wlc_log_probability(theta, times, data, err, LD_prior_bounds):
    """
    Log-probability for the white-light-curve fit. theta = [i, a, period, sqrtecosw,
    sqrtesinw, r, LD_u1, LD_u2, LD_u3, LD_u4] (10 parameters, 4NLLD law).

    Every limb-darkening coefficient gets the same uniform prior
    (LD_prior_bounds), matching Fig5_run.py's 'uniform' prior_strength option and
    Fig6_run.py's per-channel retrieval.
    """
    i_, a_, period_, sqrtecosw_, sqrtesinw_, r_, u1, u2, u3, u4 = theta
    ecc = sqrtecosw_**2 + sqrtesinw_**2
    w = jnp.arctan2(sqrtesinw_, sqrtecosw_)
    u = jnp.array([u1, u2, u3, u4])

    lp = jnp.where(ecc <= 1.0, 0.0, -jnp.inf)
    lp += uniform_logpdf(i_, shared_priors['i']['bounds'][0], shared_priors['i']['bounds'][1])
    lp += uniform_logpdf(a_, shared_priors['a']['bounds'][0], shared_priors['a']['bounds'][1])
    lp += gauss_logpdf(period_, shared_priors['period']['val'], shared_priors['period']['s_val'])
    lp += uniform_logpdf(sqrtecosw_, -1., 1.)
    lp += uniform_logpdf(sqrtesinw_, -1., 1.)
    lp += uniform_logpdf(r_, r_prior_bounds[0], r_prior_bounds[1])
    lp += jnp.sum(uniform_logpdf(u, LD_prior_bounds[0], LD_prior_bounds[1]))
    lp = jnp.where(jnp.isfinite(lp), lp, -jnp.inf)

    model = single_channel_lc(r_, u, i_, a_, period_, ecc, w, True, times)
    step_chi2 = jnp.sum((model - data)**2 / err**2)
    lk = -0.5 * (step_chi2 + jnp.sum(jnp.log(2 * jnp.pi * err**2)))
    lk = jnp.where(jnp.isnan(lk), -jnp.inf, lk)

    return lp + lk, {'step_chi2': step_chi2}


def sigma_clip_chain(raw_chain, logprob, chi2_chain, nburn, thresholds=(5, 4, 3), rounds=3, verbose=False):
    """
    Iterative 2D sigma-clipping of an MCMC chain -- the same chain-cleaning procedure used
    in Fig5_plot.py / Fig4_run.py's load_result, applied here directly to the in-memory WLC
    chain right after sampling. Removes (walker, step) pairs whose chi2 or any parameter
    value is an outlier (beyond `threshold` IQRs of the median) relative to the currently-
    surviving population, over several rounds with progressively tighter thresholds -- this
    strips out things like temporarily stuck walkers or excursions to bad regions before any
    posterior summary is computed from the chain.

    Parameters
    ----------
    raw_chain, logprob, chi2_chain : full (pre-burn-in) MCMC output, shapes
        (n_walkers, n_steps, n_params), (n_walkers, n_steps), (n_walkers, n_steps)
    nburn : number of burn-in steps to discard before clipping
    thresholds : IQR multiple used in each round (len == rounds)
    rounds : number of clipping rounds

    Returns
    -------
    good_steps_mask : (n_walkers, n_steps_post) bool -- True for surviving (walker, step) pairs
    bestfit_theta   : (n_params,) -- parameters at the highest log-probability among survivors
    post_chain      : (n_surviving, n_params) -- surviving, post-burn-in samples (flattened)
    """
    n_walkers, n_steps, n_params = raw_chain.shape
    n_steps_post = n_steps - nburn

    burnt_chain   = np.asarray(raw_chain[:, nburn:, :])
    burnt_chi2    = np.asarray(chi2_chain[:, nburn:])
    burnt_logprob = np.asarray(logprob[:, nburn:])

    good_steps_mask = np.ones((n_walkers, n_steps_post), dtype=bool)

    for round_idx in range(rounds):
        threshold = thresholds[round_idx]
        n_alive = np.sum(good_steps_mask)
        if verbose:
            print(f'    round {round_idx+1}/{rounds} (threshold={threshold}sigma, {n_alive} pairs alive)')

        # --- Filter on chi2 ---
        alive_chi2 = burnt_chi2[good_steps_mask]
        q = np.percentile(alive_chi2, [25, 50, 75])
        mu, iqr = q[1], q[2] - q[0]
        chi2_bad_2d = np.zeros((n_walkers, n_steps_post), dtype=bool)
        chi2_bad_2d[good_steps_mask] = (
            (alive_chi2 < mu - threshold * iqr) | (alive_chi2 > mu + threshold * iqr)
        )

        # --- Filter on parameters ---
        param_bad_2d = np.zeros((n_walkers, n_steps_post), dtype=bool)
        for param_idx in range(n_params):
            alive_param = burnt_chain[:, :, param_idx][good_steps_mask]
            q = np.percentile(alive_param, [25, 50, 75])
            mu, iqr = q[1], q[2] - q[0]
            outliers = (alive_param < mu - threshold * iqr) | (alive_param > mu + threshold * iqr)
            param_bad_2d[good_steps_mask] |= outliers

        round_bad = chi2_bad_2d | param_bad_2d
        good_steps_mask &= ~round_bad
        if verbose:
            print(f'      removed {np.sum(round_bad)}/{n_alive} '
                  f'({100 * np.sum(round_bad) / n_alive:.1f}%)')

    masked_logprob = np.where(good_steps_mask, burnt_logprob, -np.inf)
    best_walker, best_step = np.unravel_index(np.argmax(masked_logprob), masked_logprob.shape)
    bestfit_theta = burnt_chain[best_walker, best_step, :]

    post_chain = burnt_chain[good_steps_mask]   # (n_surviving, n_params)

    return good_steps_mask, bestfit_theta, post_chain


def run_mcmc(sampler, key1, key2, pos, num_steps, progress_desc='Sampling'):
    """
    Manually run an emcee_jax MCMC for `num_steps`, writing each step's ensemble directly
    into pre-allocated (n_walkers, n_steps, ndim)-shaped numpy arrays.

    This replaces sampler.sample_parallel() / sampler.sample(), for two reasons:

    1. Performance: on a single-device machine (no GPU/TPU), sample_parallel() falls back to
       sample(), whose progress=True code path accumulates every step's full ensemble state
       in a Python list, then calls jnp.stack() on the entire ~100,000-element list in one
       go at the end. That triggers a pathologically slow XLA compilation ("[Compiling
       module jit_stack ...] Very slow compile?") that can hang for a very long time even
       though the actual sampling itself already finished. Writing into a pre-allocated
       array incrementally avoids ever calling jnp.stack on a huge operand list.

    2. Correctness: sampler.sample()'s own post-processing reshapes the stacked coordinates
       array (native shape (n_steps, n_walkers, ndim)) via `.reshape(n_walkers, n_steps,
       ndim)` rather than `.transpose(1, 0, 2)` -- a reshape does not swap axes, so this
       silently scrambles the walker/step correspondence in the saved chain (verified
       empirically: it does not match either the untouched library trace or a correctly
       transposed one). Writing directly into a (n_walkers, n_steps, ndim) array at index
       [:, i, :] for step i sidesteps this entirely.

    Returns
    -------
    raw_chain  : (n_walkers, n_steps, ndim)
    logprob    : (n_walkers, n_steps)
    chi2_chain : (n_walkers, n_steps)
    """
    state = sampler.init(key1, pos)
    compiled_step = jax.jit(lambda s, k: sampler.step(k, s))
    keys = jax.random.split(key2, num_steps)

    n_walkers, ndim = np.asarray(pos).shape
    raw_chain  = np.empty((n_walkers, num_steps, ndim))
    logprob    = np.empty((n_walkers, num_steps))
    chi2_chain = np.empty((n_walkers, num_steps))

    iterator = tqdm(range(num_steps), desc=progress_desc)
    for i in iterator:
        state, stats = compiled_step(state, keys[i])
        raw_chain[:, i, :] = np.asarray(state.ensemble.coordinates)
        logprob[:, i]      = np.asarray(state.ensemble.log_probability)
        chi2_chain[:, i]   = np.asarray(state.ensemble.deterministics['step_chi2'])
        if 'accept_prob' in stats:
            iterator.set_postfix(accept=f"{float(np.mean(np.asarray(stats['accept_prob']))):.3f}")

    return raw_chain, logprob, chi2_chain


#############################################
################ Running code ###############
#############################################

# JWST NIRSpec/PRISM-like wavelength grid (constant R~100, 0.6-5.3 micron) -- shared by
# every star.
wav_edges, wav_centers = build_R_grid(wav_min_um, wav_max_um, R_prism)
n_bins = len(wav_centers)
print(f'Wavelength grid: {n_bins} channels between {wav_min_um} and {wav_max_um} micron at R={R_prism}')

for star_index, star_name in enumerate(star_names):
    star_props = stellar_types[star_name]
    print(f'\n=== STAR: {star_name} ===')

    save_path = check_dir(orig_save_data_path + f'{star_name}/')

    # ── Step 1: wavelength-dependent 4th-order NLLD coefficients (Fig3 procedure) ────────
    coeffs, _, _ = extract_wavelength_LDCs(
        star_props['Teff'], star_props['logg'], star_props['MH'],
        wav_edges, LD_data_path, n_mu_fine=n_mu_fine, fit_seed=fit_init_seed,
    )
    valid = ~np.any(np.isnan(coeffs), axis=1)
    if not np.all(valid):
        raise RuntimeError(f'{np.sum(~valid)} channel(s) have no valid intensity data for {star_name}.')

    wav_grid_file = save_path + 'wav_grid.npz'
    np.savez(wav_grid_file, wav_edges=wav_edges, wav_centers=wav_centers, coeffs=coeffs)

    # ── Step 2: inject the achromatic-depth chromatic light curve ────────────────────────
    print('  GENERATING CHROMATIC DATA')
    true_u_poly = jnp.array(np.array([
        np.asarray(nonlinear_4param_ld_law(*coeffs[c], order=12)) for c in range(n_bins)
    ]))
    true_r = jnp.full(n_bins, init_state_dic['r'])
    true_lc = jax.vmap(single_channel_lc, in_axes=(0, 0, None, None, None, None, None, None, None))(
        true_r, true_u_poly, init_state_dic['i'], init_state_dic['a'], init_state_dic['period'],
        init_state_dic['e'], init_state_dic['omega'], False, times,
    )  # (n_bins, n_times)

    noise_seed = noise_seed_base + star_index
    std = model_scatter * 1e-6
    noisy_data = true_lc + std * random.normal(jax.random.PRNGKey(noise_seed), shape=true_lc.shape)
    noisy_std = std * jnp.ones(true_lc.shape, dtype=float)

    # ── Step 3: build the white light curve and fit it ────────────────────────────────────
    print('  BUILDING + FITTING WHITE LIGHT CURVE')
    # Inverse-variance-weighted mean across channels (reduces to a simple mean here, since
    # every channel shares the same photometric scatter) -- exactly how a real white light
    # curve is built by co-adding the spectroscopic channels.
    weights = 1.0 / noisy_std**2
    wlc_data = jnp.sum(noisy_data * weights, axis=0) / jnp.sum(weights, axis=0)
    wlc_std  = jnp.sqrt(1.0 / jnp.sum(weights, axis=0))

    # ── Diagnostic: WLC alongside a flux colormap of the full chromatic dataset ───────────
    # (Plotted before the MCMC is run -- a sanity check on the injected data itself.)
    fig_wlc, (ax_wlc, ax_map) = plt.subplots(1, 2, figsize=(14, 5))

    ax_wlc.errorbar(times, wlc_data, yerr=wlc_std, fmt='.', markersize=3, color='0.3',
                     elinewidth=0.6, zorder=1)
    ax_wlc.set_xlabel('Time (days)')
    ax_wlc.set_ylabel('Relative flux')
    ax_wlc.set_title(f'{star_name} -- white light curve')
    ax_wlc.grid(True, alpha=0.3)

    mesh = ax_map.pcolormesh(times, wav_centers, noisy_data, shading='auto', cmap='viridis')
    ax_map.set_xlabel('Time (days)')
    ax_map.set_ylabel('Wavelength (micron)')
    ax_map.set_title(f'{star_name} -- chromatic light curve')
    cbar = fig_wlc.colorbar(mesh, ax=ax_map)
    cbar.set_label('Relative flux')

    fig_wlc.tight_layout()
    fig_wlc.savefig(save_path + 'WLC_and_chromatic_map.pdf')
    plt.close(fig_wlc)

    # Walker initialisation only -- centred on the per-channel-averaged true profile, not a
    # prior (the actual prior on the WLC's LDCs is the broad uniform LD_prior_bounds above).
    wlc_LD_init_center = jnp.array(coeffs.mean(axis=0))

    ndim_wlc = n_shared + 1 + 4
    minval = np.empty(ndim_wlc)
    maxval = np.empty(ndim_wlc)
    for k, name in enumerate(shared_param_names):
        minval[k], maxval[k] = shared_mod_prop[name]['bounds']
    minval[n_shared], maxval[n_shared] = r_init_bounds
    minval[n_shared+1:] = np.asarray(wlc_LD_init_center) - 0.5
    maxval[n_shared+1:] = np.asarray(wlc_LD_init_center) + 0.5

    emceejax_key1, emceejax_key2, pos_key = jax.random.split(jax.random.fold_in(jaxnoise_key, star_index), 3)
    pos = np.asarray(jax.random.uniform(pos_key, minval=jnp.array(minval), maxval=jnp.array(maxval),
                                         shape=(wlc_nwalkers, ndim_wlc)))

    st0 = time.time()
    sampler = emcee_jax.EnsembleSampler(
        wlc_log_probability, log_prob_args=(times, wlc_data, wlc_std, LD_prior_bounds)
    )
    raw_chain, logprob, chi2_chain = run_mcmc(
        sampler, emceejax_key1, emceejax_key2, pos, wlc_nsteps, progress_desc=f'{star_name} WLC'
    )
    elapsed = time.time() - st0
    print(f'  WLC MCMC took {elapsed:.2f} s / {elapsed/60.:.2f} min')

    np.save(save_path + 'WLC_chains.npy', raw_chain)
    np.save(save_path + 'WLC_logprob.npy', logprob)
    np.save(save_path + 'WLC_chi2_chain.npy', chi2_chain)

    # ── Pre-clipping summary (raw, post-burn-in only) -- kept for reproducibility so the
    # effect of the cleaning step can always be inspected later, but NOT what gets used
    # downstream (see post-clipping block below, which is what Fig6_run.py actually reads).
    raw_post_chain = raw_chain[:, wlc_nburn:, :].reshape(-1, ndim_wlc)
    raw_max_walker, raw_max_step = np.unravel_index(np.argmax(logprob), logprob.shape)
    raw_bestfit_theta = raw_chain[raw_max_walker, raw_max_step, :]

    raw_shared_median = np.median(raw_post_chain[:, :n_shared], axis=0)
    raw_shared_lo, raw_shared_hi = np.percentile(raw_post_chain[:, :n_shared], [16, 84], axis=0)
    raw_shared_bestfit = raw_bestfit_theta[:n_shared]

    raw_wlc_r_median = float(np.median(raw_post_chain[:, n_shared]))
    raw_wlc_r_lo, raw_wlc_r_hi = np.percentile(raw_post_chain[:, n_shared], [16, 84])
    raw_wlc_LD_median = np.median(raw_post_chain[:, n_shared+1:], axis=0)

    # ── Post-clipping summary (iterative sigma clipping) -- this is what gets saved as the
    # "primary" fields below and is the only version Fig6_run.py loads/uses.
    print('  CLEANING CHAIN (iterative sigma clipping)')
    good_steps_mask, bestfit_theta, post_chain = sigma_clip_chain(
        raw_chain, logprob, chi2_chain, wlc_nburn,
        thresholds=SIGMA_THRESHOLDS, rounds=SIGMA_ROUNDS, verbose=True,
    )
    n_post_total = good_steps_mask.size
    n_post_kept  = int(np.sum(good_steps_mask))
    print(f'    kept {n_post_kept}/{n_post_total} post-burn-in (walker, step) pairs '
          f'({100 * n_post_kept / n_post_total:.1f}%)')

    shared_median = np.median(post_chain[:, :n_shared], axis=0)
    shared_lo, shared_hi = np.percentile(post_chain[:, :n_shared], [16, 84], axis=0)
    shared_bestfit = bestfit_theta[:n_shared]

    wlc_r_median = float(np.median(post_chain[:, n_shared]))
    wlc_r_lo, wlc_r_hi = np.percentile(post_chain[:, n_shared], [16, 84])
    wlc_LD_median = np.median(post_chain[:, n_shared+1:], axis=0)

    print(f'  WLC shared params, post-clipping vs pre-clipping (median [16,84], bestfit):')
    for k, name in enumerate(shared_param_names):
        print(f'    {name:<10} post: {shared_median[k]:.6f} [{shared_lo[k]:.6f}, {shared_hi[k]:.6f}]  bestfit={shared_bestfit[k]:.6f}')
        print(f'    {"":<10} pre:  {raw_shared_median[k]:.6f} [{raw_shared_lo[k]:.6f}, {raw_shared_hi[k]:.6f}]  bestfit={raw_shared_bestfit[k]:.6f}')
    print(f'    r          post: {wlc_r_median:.6f} [{wlc_r_lo:.6f}, {wlc_r_hi:.6f}]')
    print(f'    {"":<10} pre:  {raw_wlc_r_median:.6f} [{raw_wlc_r_lo:.6f}, {raw_wlc_r_hi:.6f}]')

    np.savez(
        save_path + 'prerun.npz',
        wav_edges=wav_edges, wav_centers=wav_centers, coeffs=coeffs,
        r_true=init_state_dic['r'], model_scatter=model_scatter, noise_seed=noise_seed,
        shared_param_names=np.array(shared_param_names),
        # ── Post-clipping (the only version Fig6_run.py reads) ──────────────────────────
        shared_median=shared_median, shared_lo=shared_lo, shared_hi=shared_hi, shared_bestfit=shared_bestfit,
        wlc_r_median=wlc_r_median, wlc_r_lo=wlc_r_lo, wlc_r_hi=wlc_r_hi,
        wlc_LD_median=wlc_LD_median,
        n_post_kept=n_post_kept, n_post_total=n_post_total,
        # ── Pre-clipping (reproducibility / inspection only -- not read by Fig6_run.py) ──
        raw_shared_median=raw_shared_median, raw_shared_lo=raw_shared_lo,
        raw_shared_hi=raw_shared_hi, raw_shared_bestfit=raw_shared_bestfit,
        raw_wlc_r_median=raw_wlc_r_median, raw_wlc_r_lo=raw_wlc_r_lo, raw_wlc_r_hi=raw_wlc_r_hi,
        raw_wlc_LD_median=raw_wlc_LD_median,
    )
    print(f'  Saved -> {save_path}prerun.npz')

    # ── Lightweight diagnostics ────────────────────────────────────────────────────────
    median_params = dict(zip(shared_param_names, shared_median))
    ecc_m = median_params['sqrtecosw']**2 + median_params['sqrtesinw']**2
    w_m = jnp.arctan2(median_params['sqrtesinw'], median_params['sqrtecosw'])
    wlc_median_lc = single_channel_lc(wlc_r_median, jnp.array(wlc_LD_median), median_params['i'],
                                       median_params['a'], median_params['period'], ecc_m, w_m, True, times)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios':[3,1]}, figsize=(10, 5))
    ax1.errorbar(times, wlc_data, yerr=wlc_std, fmt='.', color='0.7', label='WLC data')
    ax1.plot(times, wlc_median_lc, color='orange', label='Median WLC model')
    ax2.errorbar(times, 1e6*(wlc_median_lc - wlc_data), yerr=wlc_std, fmt='.', color='orange')
    ax2.axhline(0, color='k', linewidth=0.8)
    ax1.set_title(f'{star_name} -- white light curve fit')
    ax1.set_ylabel('Relative flux')
    ax2.set_ylabel('Residuals (ppm)')
    ax2.set_xlabel('Time (days)')
    ax1.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path + 'WLC_bestfit.pdf')
    plt.close(fig)

    # Corner plot of the cleaned (sigma-clipped), post-burn-in posterior
    wlc_labels = shared_param_names + ['r', 'LD_u1', 'LD_u2', 'LD_u3', 'LD_u4']
    fig_corner = corner.corner(
        post_chain, labels=wlc_labels, show_titles=True,
        title_kwargs={"fontsize": 9}, label_kwargs={"fontsize": 9}, title_fmt=".5f",
    )
    fig_corner.savefig(save_path + 'WLC_corner.pdf')
    plt.close(fig_corner)
