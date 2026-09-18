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
# Fig6_prerun.py performs steps 1-3 and saves the results. Fig6_run performs the MCMC retrievals for one
# (star, limb-darkening law, wavelength) combination.
#
# The orbital parameters (i, a, period, sqrt(e)cos(w), sqrt(e)sin(w)) are FIT alongside the
# radius ratio and LDCs, with Gaussian priors set by the mean and std of the (sigma-clipped)
# white-light-curve posterior for this stellar type, saved by Fig6_prerun.py.

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

import corner
import time
import arviz as az
import numpy as np
import os, itertools, sys
from tqdm.auto import tqdm
import paths

# For 64-bit precision since JAX defaults to 32-bit
jax.config.update("jax_enable_x64", True)

# Set random seed used for the MCMC walker initialisation
jaxnoise_key = jax.random.PRNGKey(0)


def build_R_grid(wav_min, wav_max, R):
    """
    Build a constant-resolving-power (R = lambda / delta_lambda) wavelength grid -- must be
    identical to Fig6_prerun.py's build_R_grid. Defined up here (rather than in the Function
    block below) purely so n_bins can be computed for the HPC task grid in the
    parallelisation section before any star's data has been loaded.
    """
    n_edges = int(np.ceil(np.log(wav_max / wav_min) / np.log(1.0 + 1.0 / R))) + 1
    edges = wav_min * (1.0 + 1.0 / R) ** np.arange(n_edges)
    edges = edges[edges < wav_max]
    edges = np.append(edges, wav_max)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return edges, centers


#############################################
########## Define hyper-parameters ##########
#############################################

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

#%% WLC radius-ratio / limb-darkening init bounds
r_init_bounds  = [0.07, 0.15]
r_prior_bounds = [0., 1.]
LD_prior_bounds = [-100., 100.]

#%% JWST NIRSpec/PRISM wavelength grid
wav_min_um = 0.6    # micron
wav_max_um = 5.3    # micron
R_prism    = 100    # nominal (constant) resolving power, lambda / delta_lambda

#%% MCMC specific settings 
nwalkers = 50
nsteps   = 100000
nburn    = 70000

#%% Chain-cleaning settings
SIGMA_THRESHOLDS = [5, 4, 3]   # IQR multiples, one per round
SIGMA_ROUNDS     = 3

#%% Whether to produce the full diagnostic suite
make_full_diagnostics = False

#%% Output directory
orig_save_data_path = str(paths.data / "Fig6_Storage") + "/"

#############################################
########## Define parallelization ##########
#############################################

#%% All five stellar types and three limb-darkening laws
star_names = ['C5', 'C1', 'C2', 'C7', 'C6']
LDLs       = ['PLD_2', 'PLD_3', '4NLLD']

#%% Number of wavelength channels
n_bins = len(build_R_grid(wav_min_um, wav_max_um, R_prism)[1])

# # Distribute tasks - for HPC usage
# wav_bin_indices = list(range(n_bins))
# task_arrays = [star_names, wav_bin_indices, LDLs]
# param_combos = list(itertools.product(*task_arrays))
# param_combos = [list(c) for c in param_combos]
#
# my_task_id = int(sys.argv[1])
# star_name, i_bin, LDL = param_combos[my_task_id - 1]

star_name = 'C7'
i_bin     = 0
LDL       = '4NLLD'


############################
###### Function block ######
############################

def check_dir(dir_name):
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    return dir_name


def uniform_logpdf(x, lo, hi):
    return jnp.where((x >= lo) & (x <= hi), -jnp.log(hi - lo), -jnp.inf)


def gauss_logpdf(x, val, s):
    return -0.5 * (jnp.log(2 * jnp.pi * s**2) + ((x - val) / s)**2)


def single_channel_lc(r_i, u_i, i_, a_, period_, ecc, w, convert_NLLD, times):
    """Transit light curve for one channel (a single wavelength bin, or the white
    light curve), given the orbital parameters and this channel's Rp/R* / LDCs. 
    When `convert_NLLD` is True, `u_i` is 4-parameter NLLD
    coefficients [c1, c2, c3, c4], converted to an order-12 polynomial;
    otherwise `u_i` is used directly as the native-basis polynomial
    limb-darkening coefficients."""
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


def make_channel_log_prob(orbital_prior_mean, orbital_prior_std, n_u, convert_NLLD, LD_prior_bounds):
    """
    Build the log-probability function for one (star, LDL, wavelength channel) combination.
    theta = [r, LD_u1, ..., LD_u_n, i, a, period, sqrtecosw, sqrtesinw].

    - r: broad uniform prior; every LD coefficient: broad uniform prior (LD_prior_bounds),
      matching Fig5_run.py's 'uniform' prior_strength option.
    - Orbital parameters: independent Gaussian priors with the mean/std of the sigma-clipped
      WLC posterior for this stellar type (orbital_prior_mean / orbital_prior_std, in the
      order i, a, period, sqrtecosw, sqrtesinw). e = sqrtecosw^2 + sqrtesinw^2 <= 1 enforced.
    """
    orbital_prior_mean = jnp.asarray(orbital_prior_mean)
    orbital_prior_std  = jnp.asarray(orbital_prior_std)

    def channel_log_probability(theta, times, data, err):
        r_ = theta[0]
        u_ = theta[1:1 + n_u]
        orb = theta[1 + n_u:]
        i_, a_, period_, sqrtecosw_, sqrtesinw_ = orb
        ecc = sqrtecosw_**2 + sqrtesinw_**2
        w = jnp.arctan2(sqrtesinw_, sqrtecosw_)

        lp = uniform_logpdf(r_, r_prior_bounds[0], r_prior_bounds[1])
        lp += jnp.sum(uniform_logpdf(u_, LD_prior_bounds[0], LD_prior_bounds[1]))
        lp += jnp.sum(gauss_logpdf(orb, orbital_prior_mean, orbital_prior_std))
        lp += jnp.where(ecc <= 1.0, 0.0, -jnp.inf)
        lp = jnp.where(jnp.isfinite(lp), lp, -jnp.inf)

        model = single_channel_lc(r_, u_, i_, a_, period_, ecc, w, convert_NLLD, times)
        step_chi2 = jnp.sum((model - data)**2 / err**2)
        lk = -0.5 * (step_chi2 + jnp.sum(jnp.log(2 * jnp.pi * err**2)))
        lk = jnp.where(jnp.isnan(lk), -jnp.inf, lk)

        return lp + lk, {'step_chi2': step_chi2}

    return channel_log_probability


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


def sigma_clip_chain(raw_chain, logprob, chi2_chain, nburn, thresholds=(5, 4, 3), rounds=3, verbose=False):
    """
    Iterative 2D sigma-clipping of an MCMC chain.

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

        alive_chi2 = burnt_chi2[good_steps_mask]
        q = np.percentile(alive_chi2, [25, 50, 75])
        mu, iqr = q[1], q[2] - q[0]
        chi2_bad_2d = np.zeros((n_walkers, n_steps_post), dtype=bool)
        chi2_bad_2d[good_steps_mask] = (
            (alive_chi2 < mu - threshold * iqr) | (alive_chi2 > mu + threshold * iqr)
        )

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

    This replaces sampler.sample_parallel() / sampler.sample().

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

print(f'RUNNING: star={star_name}, channel={i_bin}, LDL={LDL}')

save_path = check_dir(orig_save_data_path + f'{star_name}/')
fixed_args_save_loc = check_dir(save_path + f'{LDL}/channel_{i_bin:04d}/')

# ── Load Fig6_prerun.py's output: per-channel truth + WLC-fixed orbital parameters ────────
prerun_file = save_path + 'prerun.npz'
if not os.path.exists(prerun_file):
    raise RuntimeError(f'{prerun_file} not found -- run Fig6_prerun.py for {star_name} first.')

prerun = np.load(prerun_file, allow_pickle=False)
wav_edges          = prerun['wav_edges']
wav_centers        = prerun['wav_centers']
coeffs             = prerun['coeffs']            # (n_bins, 4) -- true per-channel NLLD coefficients
r_true             = float(prerun['r_true'])
model_scatter      = float(prerun['model_scatter'])
noise_seed         = int(prerun['noise_seed'])
shared_param_names = [str(s) for s in prerun['shared_param_names']]
shared_mean        = prerun['shared_mean']       # WLC posterior mean [i, a, period, sqrtecosw, sqrtesinw]
shared_std         = prerun['shared_std']        # WLC posterior std  (same order)

n_bins = len(wav_centers)
print('  Gaussian priors on orbital parameters from the WLC posterior (mean, std):')
for k, m, sd in zip(shared_param_names, shared_mean, shared_std):
    print(f'    {k:<10} {m:.6f} +/- {sd:.6f}')

true_c = coeffs[i_bin]

############################################
########## Setting limb-darkening ##########
############################################

if 'PLD' in LDL:
    n_u = int(LDL[-1])
    convert_NLLD = False
    best_fit_LDCs = np.asarray(nonlinear_4param_ld_law(*true_c, order=n_u))
elif LDL == '4NLLD':
    n_u = 4
    convert_NLLD = True
    best_fit_LDCs = true_c
else:
    raise KeyError('Wrong limb darkening law.')

n_orb = len(shared_param_names)
ndim = 1 + n_u + n_orb    # theta = [r, LD_1..LD_n, i, a, period, sqrtecosw, sqrtesinw]
i_LD  = slice(1, 1 + n_u)
i_orb = slice(1 + n_u, ndim)

#############################
####### Generate data #######
#############################
print('GENERATING DATA')

# Regenerate the exact same noisy per-channel dataset built by Fig6_prerun.py (same true
# per-channel coefficients, same injected orbital truth, same noise seed), then take this
# channel's slice.
true_u_poly = jnp.array(np.array([
    np.asarray(nonlinear_4param_ld_law(*coeffs[c], order=12)) for c in range(n_bins)
]))
true_r_all = jnp.full(n_bins, r_true)
true_lc_all = jax.vmap(single_channel_lc, in_axes=(0, 0, None, None, None, None, None, None, None))(
    true_r_all, true_u_poly, init_state_dic['i'], init_state_dic['a'], init_state_dic['period'],
    init_state_dic['e'], init_state_dic['omega'], False, times,
)  # (n_bins, n_times)

std = model_scatter * 1e-6
noisy_data_all = true_lc_all + std * random.normal(jax.random.PRNGKey(noise_seed), shape=true_lc_all.shape)
noisy_std_all = std * jnp.ones(true_lc_all.shape, dtype=float)

noisy_lc  = noisy_data_all[i_bin]
noisy_std = noisy_std_all[i_bin]

print(f"initial chi2: {jnp.sum((true_lc_all[i_bin] - noisy_lc)**2 / noisy_std**2)}")

#Plotting (lightweight sanity-check diagnostic, kept for every combination)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=[10, 6], sharex=True, gridspec_kw={'height_ratios': [3, 1]})
ax1.errorbar(times, noisy_lc, yerr=noisy_std, fmt='.', zorder=1)
ax1.plot(times, true_lc_all[i_bin], color='red', zorder=2)
ax2.errorbar(times, 1e6*(noisy_lc - true_lc_all[i_bin]), yerr=noisy_std, fmt='r.', zorder=1)
ax1.set_title(f'{star_name}, channel {i_bin} ({wav_centers[i_bin]:.3f} micron), {model_scatter:.0f} ppm scatter')
ax2.set_xlabel('Time (days)')
ax1.set_ylabel('Flux')
ax2.set_ylabel('Difference (ppm)')
fig.tight_layout()
plt.savefig(fixed_args_save_loc + 'init_guess.pdf')
plt.close(fig)


#########################
##### Emcee fitting #####
#########################

log_prob_fn = make_channel_log_prob(shared_mean, shared_std, n_u, convert_NLLD, LD_prior_bounds)

# Walker initialisation: r and LDCs uniform around their guesses; orbital parameters drawn
# from their own Gaussian priors.
minval = np.concatenate([[r_init_bounds[0]], best_fit_LDCs - 0.5])
maxval = np.concatenate([[r_init_bounds[1]], best_fit_LDCs + 0.5])
emceejax_key1, emceejax_key2, pos_key, orb_key = jax.random.split(jaxnoise_key, 4)
pos_ld = np.asarray(jax.random.uniform(pos_key, minval=jnp.array(minval), maxval=jnp.array(maxval),
                                        shape=(nwalkers, 1 + n_u)))
pos_orb = shared_mean + shared_std * np.asarray(jax.random.normal(orb_key, shape=(nwalkers, n_orb)))
pos = np.concatenate([pos_ld, pos_orb], axis=1)

print("Running MCMC")
st0 = time.time()
sampler = emcee_jax.EnsembleSampler(log_prob_fn, log_prob_args=(times, noisy_lc, noisy_std))
raw_chain, logprob, chi2_chain = run_mcmc(
    sampler, emceejax_key1, emceejax_key2, pos, nsteps,
    progress_desc=f'{star_name} ch{i_bin:04d} {LDL}',
)
elapsed = time.time() - st0
print(f'MCMC took {elapsed:.2f} seconds / {elapsed/60.:.2f} minutes / {elapsed/3600.:.2f} hours.')

#%% Storing the full posterior for this channel
np.save(fixed_args_save_loc + 'chains.npy', raw_chain)
np.save(fixed_args_save_loc + 'logprob.npy', logprob)
np.save(fixed_args_save_loc + 'chi2_chain.npy', chi2_chain)

##################
#### Plotting ####
##################
print('PLOTTING')

# ── Pre-clipping summary (raw, post-burn-in only) -- kept for reproducibility, NOT what
# Fig6_plot.py reads (see post-clipping block below).
raw_post_chain = raw_chain[:, nburn:, :].reshape(-1, ndim)
raw_max_walker, raw_max_step = np.unravel_index(np.argmax(logprob), logprob.shape)
raw_bestfit_theta = raw_chain[raw_max_walker, raw_max_step, :]

raw_r_median = float(np.median(raw_post_chain[:, 0]))
raw_r_lo, raw_r_hi = np.percentile(raw_post_chain[:, 0], [16, 84])
raw_r_bestfit = float(raw_bestfit_theta[0])
raw_LD_median = np.median(raw_post_chain[:, i_LD], axis=0)
raw_LD_lo, raw_LD_hi = np.percentile(raw_post_chain[:, i_LD], [16, 84], axis=0)
raw_orb_median = np.median(raw_post_chain[:, i_orb], axis=0)
raw_orb_lo, raw_orb_hi = np.percentile(raw_post_chain[:, i_orb], [16, 84], axis=0)

# ── Post-clipping summary (iterative sigma clipping) -- the "primary" fields saved below,
# which are what Fig6_plot.py reads.
print('CLEANING CHAIN (iterative sigma clipping)')
good_steps_mask, bestfit_theta, post_chain = sigma_clip_chain(
    raw_chain, logprob, chi2_chain, nburn,
    thresholds=SIGMA_THRESHOLDS, rounds=SIGMA_ROUNDS, verbose=True,
)
n_post_total = good_steps_mask.size
n_post_kept  = int(np.sum(good_steps_mask))
print(f'  kept {n_post_kept}/{n_post_total} post-burn-in (walker, step) pairs '
      f'({100 * n_post_kept / n_post_total:.1f}%)')

median_theta = np.median(post_chain, axis=0)
def lc_from_theta(theta):
    i_, a_, period_, sqrtecosw_, sqrtesinw_ = theta[i_orb]
    return single_channel_lc(theta[0], jnp.array(theta[i_LD]), i_, a_, period_,
                             sqrtecosw_**2 + sqrtesinw_**2, jnp.arctan2(sqrtesinw_, sqrtecosw_),
                             convert_NLLD, times)

median_lc  = lc_from_theta(median_theta)
bestfit_lc = lc_from_theta(bestfit_theta)
median_RMS = jnp.sqrt(jnp.average((noisy_lc - median_lc)**2))
bestfit_RMS = jnp.sqrt(jnp.average((noisy_lc - bestfit_lc)**2))

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios':[3,1]}, figsize=(12, 6))
ax1.errorbar(times, noisy_lc, yerr=noisy_std, fmt='.', label='Data', color='0.7')
ax1.plot(times, median_lc, color='orange', label='Median model')
ax1.plot(times, bestfit_lc, color='blue', label='Best-fit model')
ax2.errorbar(times, 1e6*(median_lc - noisy_lc), yerr=noisy_std, fmt='.', color='orange', label=f'RMS = {1e6*median_RMS:.0f} ppm')
ax2.errorbar(times, 1e6*(bestfit_lc - noisy_lc), yerr=noisy_std, fmt='.', color='blue', label=f'RMS = {1e6*bestfit_RMS:.0f} ppm')
ax2.set_xlabel("Time [days]")
ax1.set_ylabel("Relative flux")
ax2.set_ylabel("Residuals (ppm)")
ax1.legend()
ax2.legend()
plt.tight_layout()
plt.savefig(fixed_args_save_loc + 'bestfit.pdf')
plt.close()

r_median = float(np.median(post_chain[:, 0]))
r_lo, r_hi = np.percentile(post_chain[:, 0], [16, 84])
r_bestfit = float(bestfit_theta[0])

LD_median = np.median(post_chain[:, i_LD], axis=0)
LD_lo, LD_hi = np.percentile(post_chain[:, i_LD], [16, 84], axis=0)
orb_median = np.median(post_chain[:, i_orb], axis=0)
orb_lo, orb_hi = np.percentile(post_chain[:, i_orb], [16, 84], axis=0)

# Autocorrelation needs the per-walker time series, so it's computed on the raw post-burn-in chain
try:
    r_tau = float(autocorr_new(raw_chain[:, nburn:, 0]))
except Exception:
    r_tau = np.nan

np.savez(
    fixed_args_save_loc + 'summary.npz',
    wav_um=wav_centers[i_bin],
    true_LDCs=true_c,
    r_true=r_true,
    r_median=r_median, r_lo=r_lo, r_hi=r_hi, r_bestfit=r_bestfit,
    LD_median=LD_median, LD_lo=LD_lo, LD_hi=LD_hi,
    shared_param_names=np.array(shared_param_names),
    orb_median=orb_median, orb_lo=orb_lo, orb_hi=orb_hi,
    orb_prior_mean=shared_mean, orb_prior_std=shared_std,
    r_autocorr_time=r_tau,
    n_steps_post_burn=nsteps - nburn,
    n_post_kept=n_post_kept, n_post_total=n_post_total,
    # ── Pre-clipping ──
    raw_r_median=raw_r_median, raw_r_lo=raw_r_lo, raw_r_hi=raw_r_hi, raw_r_bestfit=raw_r_bestfit,
    raw_LD_median=raw_LD_median, raw_LD_lo=raw_LD_lo, raw_LD_hi=raw_LD_hi,
    raw_orb_median=raw_orb_median, raw_orb_lo=raw_orb_lo, raw_orb_hi=raw_orb_hi,
)
print(f'  r (post-clip) = {r_median:.5f} (+{r_hi-r_median:.5f} / -{r_median-r_lo:.5f}), bestfit = {r_bestfit:.5f}, tau_r = {r_tau:.1f}')
print(f'  r (pre-clip)  = {raw_r_median:.5f} (+{raw_r_hi-raw_r_median:.5f} / -{raw_r_median-raw_r_lo:.5f}), bestfit = {raw_r_bestfit:.5f}')

# ── Optional full diagnostic suite (trace, corner, autocorrelation plots) ─────────────────
if make_full_diagnostics:

    labels = ['r'] + [f'LD_u{k+1}' for k in range(n_u)] + list(shared_param_names)
    inf_data = az.from_dict(
        posterior={name: raw_chain[:, nburn:, k] for k, name in enumerate(labels)},
        log_likelihood={"log_like": logprob},
    )

    _ = az.plot_trace(inf_data, var_names=labels, backend_kwargs={"constrained_layout": True})
    plt.savefig(fixed_args_save_loc + 'trace.pdf')
    plt.close()

    truth_list = [r_true] + list(best_fit_LDCs) + list(shared_mean)   # orbital 'truth' = WLC prior mean
    # Corner plot uses the cleaned (sigma-clipped) samples; trace/autocorrelation plots need
    # the per-walker time series so they stay on the raw post-burn-in chain.
    fig1 = corner.corner(
        post_chain, labels=labels, show_titles=True,
        title_kwargs={"fontsize": 10}, label_kwargs={"fontsize": 10}, title_fmt=".4f",
        truths=truth_list,
    )
    plt.savefig(fixed_args_save_loc + 'corner.pdf')
    plt.close()

    n_eval = 15
    postburn_chain = raw_chain[:, nburn:, :]
    nsteps_postburn = postburn_chain.shape[1]
    N_eval = np.unique(np.exp(np.linspace(np.log(100), np.log(nsteps_postburn), n_eval)).astype(int))

    ncols = 3
    nrows = int(np.ceil(ndim / ncols))
    fig_ac, axes_ac = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
    for iparam in range(ndim):
        ax = axes_ac[iparam // ncols, iparam % ncols]
        y_param = postburn_chain[:, :, iparam]
        new_vals = np.full(len(N_eval), np.nan)
        for idx, n in enumerate(N_eval):
            try:
                new_vals[idx] = autocorr_new(y_param[:, :n])
            except Exception:
                pass
        ax.loglog(N_eval, new_vals, "o-", color="C1", markersize=4)
        ylim = ax.get_ylim()
        ax.loglog(N_eval, N_eval / 50.0, "--k", label=r"$\tau = N/50$")
        ax.set_ylim(ylim)
        ax.set_xlabel("Number of samples, $N$")
        ax.set_ylabel(r"$\tau$ estimate")
        ax.set_title(labels[iparam])
        ax.legend(fontsize=8)
    for iparam in range(ndim, nrows * ncols):
        axes_ac[iparam // ncols, iparam % ncols].set_visible(False)
    fig_ac.tight_layout()
    plt.savefig(fixed_args_save_loc + 'autocorrelation.pdf')
    plt.close()
