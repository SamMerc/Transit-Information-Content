#############################
########## Purpose ##########
#############################

# Figures 2, 3, and 4 require a 4-th order non-linear limb-darkening law for the injection / simulation of the LC.
# Given that we are working with a made up fiducial system, we need to identify the limb-darkening values to use for this.
# In order to do this, we explore all available intensity profiles for a given grid of stellar models, and perform a PCA 
# analysis to identify both the median/mode and an outlier intensity profile which can be used in our analyses. 
# We perform such decomposition on each individual grid of stellar models, and in doing so this allows us to highlight the choice of 
# 1. stellar model and 2. limb-darkening prescription on the transit depth amplification factor and bias 



######################################
########## Import libraries ##########
######################################


import numpy as np
import matplotlib
import os
matplotlib.use('TkAgg')  # or 'Qt5Agg' or 'MacOSX'
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import jit, vmap
import exotic_ld as el
import pickle
import gc
cmap = plt.cm.coolwarm
import seaborn as sns
from scipy.interpolate import interp1d
from scipy.interpolate import griddata

######################################
########## Hyper-parameters ##########
######################################

LD_data_path = '/Volumes/Ajax/Work/PhD/Research/Transit-Information-Content/LD simulation'
orig_save_data_path = '/Users/samsonmercier/Desktop/Work/PhD/Research/Transit-Information-Content/Fig2_sidehelper_Storage/'

models = ['mps1'] #['phoenix','kurucz', 'stagger', 'mps1', 'mps2']

Teffs = {
    'phoenix' : [2300, 15000], 
    'kurucz' : [3500, 6500],
    'stagger' : [4000, 7000],
    'mps2' : [3500, 9000],
    'mps1' : [3500, 9000],
}

loggs = {
    'phoenix' : [0.0, 6.0], 
    'kurucz' : [4.0, 5.0],
    'stagger' : [1.5, 5.0],
    'mps1' : [3.0, 5.0],
    'mps2' : [3.0, 5.0],
}

metallicitys = {
    'phoenix' : [-1.5, 1.0], 
    'kurucz' : [-5.0, 1.0],
    'stagger' : [-3.0, 0.0],
    'mps1' : [-5.0, 1.5],
    'mps2' : [-5.0, 1.5],
}


mu_resolution = {
    'phoenix' : 78,
    'kurucz' : 17,
    'stagger' : 10,
    'mps1' : 24,
    'mps2' : 24,
}

lambda_resolution = {
    'phoenix' : 54500,
    'kurucz' : 1221,
    'stagger' : 105767,
    'mps1' : 1221,
    'mps2' : 1221,
}

N_star = 10

N_chords = 100

N_bs_ps = 5
bs = jnp.linspace(0, 1, N_bs_ps)
ps = jnp.logspace(-3, -1, N_bs_ps)

n_mu_fine = 100

mode = 'build' # 'build' or 'load'

rolling_std_window_size = 10
rolling_mean_window_size = 10

############################
###### Function block ######
############################
def rolling_std(wavelengths, intensities, window_size=10):
    """
    Computes a rolling standard deviation over a spectrum using a sliding window.

    For each wavelength, the standard deviation is computed over the `window_size`
    values centred at that point. At the edges, the window is padded by reflecting
    the signal so the output always has the same length as the input.

    Parameters
    ----------
    wavelengths : array-like
        1-D array of wavelength values.
    intensities : array-like
        1-D array of intensity values (same length as wavelengths).
    window_size : int
        Number of points in the sliding window (default 10).

    Returns
    -------
    wavelengths : np.ndarray
        Original wavelength array (unchanged).
    std_values : np.ndarray
        Rolling standard deviation, same shape as the input arrays.
    """
    wavelengths = np.asarray(wavelengths, dtype=float)
    intensities = np.asarray(intensities, dtype=float)

    if len(wavelengths) != len(intensities):
        raise ValueError("wavelengths and intensities must have the same length.")
    if window_size < 2:
        raise ValueError("window_size must be at least 2.")

    half = window_size // 2

    # Reflect-pad the intensity array so edge windows stay the same size
    padded = np.pad(intensities, pad_width=half, mode="reflect")

    std_values = np.array(
        [padded[i : i + window_size].std() for i in range(len(intensities))]
    )

    return std_values

def rolling_mean(wavelengths, intensities, window_size=10):
    """
    Computes a rolling mean over a spectrum using a sliding window.

    For each wavelength, the mean is computed over the `window_size`
    values centred at that point. At the edges, the window is padded by reflecting
    the signal so the output always has the same length as the input.

    Parameters
    ----------
    wavelengths : array-like
        1-D array of wavelength values.
    intensities : array-like
        1-D array of intensity values (same length as wavelengths).
    window_size : int
        Number of points in the sliding window (default 10).

    Returns
    -------
    wavelengths : np.ndarray
        Original wavelength array (unchanged).
    std_values : np.ndarray
        Rolling standard deviation, same shape as the input arrays.
    """
    wavelengths = np.asarray(wavelengths, dtype=float)
    intensities = np.asarray(intensities, dtype=float)

    if len(wavelengths) != len(intensities):
        raise ValueError("wavelengths and intensities must have the same length.")
    if window_size < 2:
        raise ValueError("window_size must be at least 2.")

    half = window_size // 2

    # Reflect-pad the intensity array so edge windows stay the same size
    padded = np.pad(intensities, pad_width=half, mode="reflect")

    mean_values = np.array(
        [padded[i : i + window_size].mean() for i in range(len(intensities))]
    )

    return mean_values

@jit
def calculate_annulus_overlap(r_planet, p, r_inner, r_outer):
    """
    JAX-optimized annulus overlap calculation.
    """
    overlap_outer = calculate_circle_overlap(r_planet, p, r_outer)
    overlap_inner = calculate_circle_overlap(r_planet, p, r_inner)
    return overlap_outer - overlap_inner


@jit
def calculate_circle_overlap(d, r1, r2):
    """
    JAX-optimized circle overlap calculation.
    Handles arrays efficiently.
    """
    # No overlap cases
    no_overlap = d >= r1 + r2
    
    # Complete overlap cases
    complete_overlap = d <= jnp.abs(r2 - r1)
    
    # Partial overlap - use lens formula
    d2 = d * d
    r1_2 = r1 * r1
    r2_2 = r2 * r2
    
    # Safe division - avoid division by zero
    denom = 2 * d * r1
    safe_denom = jnp.where(denom == 0, 1.0, denom)
    alpha_arg = (d2 + r1_2 - r2_2) / safe_denom
    alpha_arg = jnp.clip(alpha_arg, -1.0, 1.0)
    alpha = jnp.arccos(alpha_arg)
    
    denom2 = 2 * d * r2
    safe_denom2 = jnp.where(denom2 == 0, 1.0, denom2)
    beta_arg = (d2 + r2_2 - r1_2) / safe_denom2
    beta_arg = jnp.clip(beta_arg, -1.0, 1.0)
    beta = jnp.arccos(beta_arg)
    
    partial_area = r1_2 * alpha + r2_2 * beta - 0.5 * (r1_2 * jnp.sin(2 * alpha) + r2_2 * jnp.sin(2 * beta))
    
    # Combine all cases
    area = jnp.where(no_overlap, 0.0,
                     jnp.where(complete_overlap, jnp.pi * jnp.minimum(r1, r2)**2,
                              partial_area))
    
    return area
    
@jit
def chord_intensity(b, p, intensity_spectra, stellar_radii):
    '''
    JAX-optimized version for a single (b, p) pair.
    :param b: Transit chord impact parameter.
    :param p: Planet-to-star radius ratio.
    :param intensity_spectra: 2D array of intensity spectra (shape : n_wavelengths * n_stellar_mus).
    :param stellar_radii: Array of r values for the edges of the annuli discretizing the stellar disk. (shape : n_stellar_mus)
    :param N_chords: Number of points discretizing the transit chord.
    '''
    # Calculate the possible positions of the planet along the (half) transit chord based on the impact parameter
    # We only need half of the transit chord to trace out the intensity profile needed.
    x_min = 0.0
    x_max = jnp.sqrt((1+p)**2 - b**2)
    x_vals = jnp.linspace(x_min, x_max, N_chords)
    r_ps = jnp.sqrt(b**2 + x_vals**2)  # Shape: (N_chords,)
    
    # Add inner edge (r=0) and outer edge (r=1 or last stellar radius)
    r_inner_edges = jnp.concatenate([jnp.array([0.0]), stellar_radii[:-1]])    # Shape: (n_stellar_mus,)
    r_outer_edges = stellar_radii

    # Vectorize over chord positions and annuli
    r_ps_grid = r_ps[:, None]           # Shape: (N_chords, 1)
    r_inner_grid = r_inner_edges[None, :]  # Shape: (1, n_stellar_mus)
    r_outer_grid = r_outer_edges[None, :]  # Shape: (1, n_stellar_mus)

    # Calculate all overlaps at once
    overlap_areas = calculate_annulus_overlap(
        r_ps_grid, p, r_inner_grid, r_outer_grid
    )  # Shape: (N_chords, n_stellar_mus)
        
    # Calculate the occulted intensity spectrum by doing a weighted sum over the occulted annuli 
    # and the weights are the % of planet-occulted area covered by each annulus
    total_planet_area = jnp.pi * p**2
    weights = overlap_areas / total_planet_area  # Shape: (N_chords, n_stellar_mus)

    # Weighted sum: intensity_spectra @ weights.T -> (n_wavelengths, n_stellar_mus) @ (n_stellar_mus, N_chords)
    # We want (n_wavelengths, N_chords)
    occulted_intensity_spectra = intensity_spectra @ weights.T
    # Shape: (n_wavelengths, N_chords)
        
    return occulted_intensity_spectra

# Vectorize over b and p
chord_intensity_vectorized = jit(vmap(
    vmap(chord_intensity, in_axes=(None, 0, None, None)),
    in_axes=(0, None, None, None)
))

################################
########## Code block ##########
################################
if not os.path.exists(orig_save_data_path):os.makedirs(orig_save_data_path)

# Instantiate dictionary to store information 
gen_dict = {}

gen_dict['local_rps']={model : np.zeros((N_star, N_star, N_star, N_bs_ps, N_bs_ps, N_chords), dtype=float) for model in models}

gen_dict['local_intensity_profiles']={model: np.empty((N_star, N_star, N_star), dtype=object) for model in models}

gen_dict['rolling_standard_deviation']={model : np.zeros((N_star, N_star, N_star, n_mu_fine), dtype=object) for model in models}

gen_dict['rolling_mean']={model : np.zeros((N_star, N_star, N_star, n_mu_fine), dtype=object) for model in models}

gen_dict['global_stellar_intensity'] = {model: np.empty((N_star, N_star, N_star), dtype=object)for model in models}

gen_dict['stellar_mus'] = {model: np.zeros((N_star, N_star, N_star, n_mu_fine), dtype=float)for model in models}

gen_dict['stellar_wavelengths'] = {model: np.empty((N_star, N_star, N_star), dtype=object) for model in models}

has_increase_counter = {model: 0 for model in models}
#Iterate over all the stellar models available 
for model in models:
    
    # Create save path for each model
    save_data_path = orig_save_data_path + f'{model}/'
    if not os.path.exists(save_data_path):os.makedirs(save_data_path)

    #Build intensity profiles grid
    if mode == 'build':
        #Iterate over the three stellar parameters and retrieve intensity profiles
        #Temperature
        for i, T in enumerate(jnp.linspace(Teffs[model][0], Teffs[model][1], N_star)):
            #Surface gravity
            for j, g in enumerate(jnp.linspace(loggs[model][0], loggs[model][1], N_star)):
                #Metallicity
                for k, m in enumerate(jnp.linspace(metallicitys[model][0], metallicitys[model][1], N_star)):
                    
                    #Calculate stellar spectrum - across wavelength and viewing angle
                    print('GENERATING Teff =', T, 'logg =', g, 'metallicty =', m, 'for model', model)
                    sld = el.StellarLimbDarkening(M_H=m, Teff=T, logg=g,
                                ld_model=model,
                                ld_data_path=LD_data_path,
                                interpolate_type="nearest")
                    
                    #Store the wavelength and mu arrays
                    stellar_wavelengths = jnp.copy(sld.stellar_wavelengths)
                    stellar_mus = jnp.copy(sld.mus)

                    #Store the global stellar intensity spectrum
                    global_stellar_intensities = jnp.copy(sld.stellar_intensities)
                    del sld  # Free memory from large StellarLimbDarkening object

                    #Filter out the bad portions of the intensity spectra
                    # thresh = 
                    cond = ((stellar_wavelengths > 6000) & (stellar_wavelengths < 53000))   # Shape: (n_wavelengths,)
                    global_stellar_intensities = global_stellar_intensities[cond, :]
                    stellar_wavelengths = stellar_wavelengths[cond]

                    #Store the global stellar intensity spectrum for each model and parameter set
                    gen_dict['stellar_wavelengths'][model][i, j, k] = np.array(stellar_wavelengths)

                    ##############################################################################
                    ########## Extract intensity profile for each transit chord ##################
                    ##############################################################################

                    # ─────────────────────────────────────────────────────────────────────────────
                    # Interpolate stellar intensities onto a fine mu grid to avoid staircase
                    # ─────────────────────────────────────────────────────────────────────────────

                    # Build fine grid from just above 0 to 1
                    stellar_mus_fine = jnp.linspace(stellar_mus[-1], stellar_mus[0], n_mu_fine) # (n_mu_fine,)

                    # Interpolate each wavelength's intensity profile onto the fine mu grid
                    interp_func = interp1d(
                        stellar_mus[::-1],
                        global_stellar_intensities[:, ::-1],
                        kind='cubic',        # cubic gives smooth curves matching exotic_ld's approach
                        axis=1,              # interpolate along mu axis
                        bounds_error=False,
                    )
                    global_stellar_intensities_fine = interp_func(stellar_mus_fine)   # (n_wav, n_mu_fine)

                    # Put the order back
                    stellar_mus = stellar_mus_fine[::-1]
                    global_stellar_intensities = global_stellar_intensities_fine[:, ::-1]
                    gen_dict['stellar_mus'][model][i, j, k] = stellar_mus
                    gen_dict['global_stellar_intensity'][model][i, j, k] = global_stellar_intensities_fine

                    #Perform the rolling window mean across the spectrum
                    for mu_idx in range(len(stellar_mus)):
                        gen_dict['rolling_mean'][model][i, j, k, mu_idx] = rolling_mean(stellar_wavelengths, global_stellar_intensities[:, mu_idx], window_size=rolling_mean_window_size)

                    #Remove the rolling window mean from the spectrum to isolate high-frequency variations
                    #Perform the rolling window standard deviation across on the resulting high-frequency variations to identify regions of the spectrum with high variability which may cause issues for the interpolation and PCA decomposition steps.
                    for mu_idx in range(len(stellar_mus)):
                        gen_dict['rolling_standard_deviation'][model][i, j, k, mu_idx] = rolling_std(stellar_wavelengths, (global_stellar_intensities[:, mu_idx] - gen_dict['rolling_mean'][model][i, j, k, mu_idx])/global_stellar_intensities[:, mu_idx], window_size=rolling_std_window_size)
                
                    # Define the annuli edges - the models define intensity spectra at a specific 
                    # mu values so this spreads out these predictions over a band
                    annuli_mus = jnp.append(
                        stellar_mus[:-1] + jnp.diff(stellar_mus)/2,
                        stellar_mus[-1]  + (jnp.diff(stellar_mus)[-1]/2)
                    )
                    
                    # Compute for all (b, p) combinations at once
                    local_stellar_intensities = chord_intensity_vectorized(
                        bs, ps, global_stellar_intensities, jnp.sqrt(1 - annuli_mus**2)
                    ) #shape : (n_bs, n_ps, n_wavelengths, N_chords)

                    #Define the grid of mu values for each ps and bs considered
                    x_max = jnp.sqrt((1 + ps[None, :])**2 - bs[:, None]**2)  # Shape: (n_bs, n_ps)
                    t = jnp.linspace(0.0, 1.0, N_chords)
                    x_vals = x_max[:, :, None] * t[None, None, :]  # Shape: (n_bs, n_ps, N_chords)
                    r_ps = jnp.sqrt(bs[:, None, None]**2 + x_vals**2)  # Shape: (n_bs, n_ps, N_chords)
                    
                    # Normalize the profiles 
                    normalized_profiles = (local_stellar_intensities / local_stellar_intensities[:,:,:,0:1])

                    #Normalize and store this local intensity profile
                    gen_dict['local_intensity_profiles'][model][i, j, k] = np.array(normalized_profiles)  # (n_bs, n_ps, n_lambda, N_chords)
                    gen_dict['local_rps'][model][i, j, k, :, :, :] = r_ps
                    
                    # Find all profiles with increasing segments
                    has_increase = jnp.any(jnp.diff(local_stellar_intensities, axis=-1) > 0.0, axis=-1)
                    # Shape: (n_bs, n_ps, n_wavelengths) - True where profile increases

                    if jnp.any(has_increase):
                        has_increase_counter[model] += 1

                        print(f"Found {jnp.sum(has_increase)}/{N_bs_ps * N_bs_ps * normalized_profiles.shape[2]} ({100 * (jnp.sum(has_increase))/(N_bs_ps * N_bs_ps * normalized_profiles.shape[2]):.0f} %) profiles with increasing segments")
                        
                        # Find the profile with the largest positive jump
                        diff_profiles = jnp.diff(local_stellar_intensities, axis=-1)
                        max_diff_idx = jnp.unravel_index(jnp.argmax(diff_profiles), diff_profiles.shape)
                        print(f"Max increase at (ib={max_diff_idx[0]}, ip={max_diff_idx[1]}, iw={max_diff_idx[2]}, ir={max_diff_idx[3]})")
                        print(f"Max increase at (b={bs[max_diff_idx[0]]}, p={ps[max_diff_idx[1]]}, w={stellar_wavelengths[max_diff_idx[2]]}, r={r_ps[max_diff_idx[0], max_diff_idx[1], max_diff_idx[3]]})")
                        
                        # Plot all offending profiles
                        offending_wavelengths = set()
                        
                        for ib in range(N_bs_ps):
                            for ip in range(N_bs_ps):
                                for iw in range(len(stellar_wavelengths)):
                                    if has_increase[ib, ip, iw]:
                                        offending_wavelengths.add(iw)
                        
                        offending_wavelengths = sorted(offending_wavelengths)
                        to_color = {iw: cmap(p / max(len(offending_wavelengths) - 1, 1)) 
                                    for p, iw in enumerate(offending_wavelengths)}
                        
                        plt.figure(figsize=(10, 6))
                        for ib in range(N_bs_ps):
                            for ip in range(N_bs_ps):
                                for iw in range(len(stellar_wavelengths)):
                                    if has_increase[ib, ip, iw]:
                                        plt.plot(r_ps[ib, ip, :], local_stellar_intensities[ib, ip, iw, :], 
                                                color=to_color[iw], alpha=0.5, linewidth=0.5)
                        plt.xlabel('r_p (stellar radii)')
                        plt.ylabel('Normalized intensity')
                        plt.title(f'Profiles with increasing segments ({jnp.sum(has_increase)} total)')
                        plt.grid(True, alpha=0.3)
                        plt.savefig(save_data_path + 'increasing_profiles.pdf', dpi=300)
                        plt.close()

                        # Plot the worst offender
                        plt.figure(figsize=(10, 6))
                        plt.plot(r_ps[max_diff_idx[0], max_diff_idx[1], :], 
                                local_stellar_intensities[max_diff_idx[0], max_diff_idx[1], max_diff_idx[2], :], 
                                'r-', linewidth=2)
                        plt.xlabel('r_p (stellar radii)')
                        plt.ylabel('Normalized intensity')
                        plt.title(f'Worst offender: b={bs[max_diff_idx[0]]:.3f}, p={ps[max_diff_idx[1]]:.4f}, λ_idx={max_diff_idx[2]}')
                        plt.grid(True, alpha=0.3)
                        plt.savefig(save_data_path + 'worst_increasing_profile.pdf', dpi=300)
                        plt.close()

                    #Plot the stellar intensity spectrum for the worst offender
                    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, figsize=(10, 12))
                    to_color = {p: cmap(p / (len(stellar_mus) - 1)) for p, imu in enumerate(stellar_mus)}
                    for mu_idx in np.arange(0, len(stellar_mus), 2):
                        
                        ax1.plot(stellar_wavelengths, global_stellar_intensities[:, mu_idx],
                                color=to_color[mu_idx], label="$\mu={:.2f}$".format(stellar_mus[mu_idx]))
                        
                        ax2.plot(stellar_wavelengths, gen_dict['rolling_mean'][model][i, j, k, mu_idx],
                                color=to_color[mu_idx], label="$\mu={:.2f}$".format(stellar_mus[mu_idx]), linestyle='--')

                        ax3.plot(stellar_wavelengths, (global_stellar_intensities[:, mu_idx] - gen_dict['rolling_mean'][model][i, j, k, mu_idx])/global_stellar_intensities[:, mu_idx] ,
                                color=to_color[mu_idx], label="$\mu={:.2f}$".format(stellar_mus[mu_idx]))
                        
                        ax4.plot(stellar_wavelengths, gen_dict['rolling_standard_deviation'][model][i, j, k, mu_idx],
                                color=to_color[mu_idx], label="$\mu={:.2f}$".format(stellar_mus[mu_idx])) 

                    ax4.set_xlabel("$\lambda / \AA$", fontsize=13)
                    ax1.set_ylabel("Intensity / $n_{\gamma} s^{-1} cm^{-2} \AA{-1} sr^{-1}$", fontsize=13)
                    ax2.set_ylabel("Rolling mean", fontsize=13)
                    ax3.set_ylabel("Relative difference from rolling mean", fontsize=13)
                    ax4.set_ylabel("Relative rolling standard deviation", fontsize=13)
                    # plt.xlim(0, 5e4)
                    # ax4.set_ylim(-0.05, 0.15)
                    plt.legend(loc="upper right", fontsize=1, bbox_to_anchor=(1.2, 1.8))
                    plt.savefig(save_data_path + 'sample_stellar_spectrum.pdf', dpi=300)
                    plt.close()
                    
                    #Garbage collection
                    del local_stellar_intensities, global_stellar_intensities, normalized_profiles, annuli_mus, r_ps, x_max, x_vals, t
                    gc.collect()

        print('Total profiles with increasing segments for model', model, ':', has_increase_counter[model])
        
        #Store the stellar spectrum
        with open(save_data_path + 'data.pkl', 'wb') as f:pickle.dump(gen_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    
        #Garbage collection
        del stellar_mus
        gc.collect()

    #Load intensity profiles grid
    elif mode == 'load':
        with open(save_data_path + 'data.pkl', 'rb') as f:gen_dict = pickle.load(f)
    
    else:
        raise KeyboardInterrupt('Mode not recognized.')
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Wavelength ranges for the three columns — fill these in
    # ─────────────────────────────────────────────────────────────────────────────

    wav_ranges = [
        (6000, 20000),   # column 0: e.g. (3000, 5000) Å
        (20000, 40000),   # column 1: e.g. (5000, 7000) Å
        (40000, 53000),   # column 2: e.g. (7000, 9000) Å
    ]
    wav_range_labels = [
        r"$\lambda \in [6000, 20000]\ \AA$",
        r"$\lambda \in [20000, 40000]\ \AA$",
        r"$\lambda \in [40000, 53000]\ \AA$",
    ]

    # ─────────────────────────────────────────────────────────────────────────────
    # Pre-compute mean relative differences per wavelength range
    # shape → (n_Teff, n_logg, n_metallicity, n_mu, n_wav_ranges)
    # ─────────────────────────────────────────────────────────────────────────────

    n_T, n_g, n_m = N_star, N_star, N_star
    n_mu          = n_mu_fine
    n_wav_ranges  = len(wav_ranges)

    mean_rel_diff = np.full((n_T, n_g, n_m, n_mu, n_wav_ranges), np.nan, dtype=float)

    for i in range(n_T):
        for j in range(n_g):
            for k in range(n_m):
                glob  = gen_dict['global_stellar_intensity'][model][i, j, k].astype(float)  # (n_wav, n_mu)
                wavs  = gen_dict['stellar_wavelengths'][model][i, j, k].astype(float)       # (n_wav,)
                rmean = np.stack(
                    [gen_dict['rolling_mean'][model][i, j, k, mu_idx] for mu_idx in range(n_mu)],
                    axis=1
                ).astype(float)  # (n_wav, n_mu)

                safe_glob = np.where(glob == 0, np.nan, glob)
                rel_diff  = (glob - rmean) / safe_glob  # (n_wav, n_mu)

                for w, (wav_lo, wav_hi) in enumerate(wav_ranges):
                    # Build wavelength mask — treat None as unbounded
                    mask = np.ones(len(wavs), dtype=bool)
                    if wav_lo is not None:
                        mask &= wavs >= wav_lo
                    if wav_hi is not None:
                        mask &= wavs <= wav_hi

                    if mask.sum() == 0:
                        continue  # no wavelengths in this range for this star

                    mean_rel_diff[i, j, k, :, w] = np.abs(
                        np.nanmax(rel_diff[mask, :], axis=0)
                    )  # average over wavelengths in range → (n_mu,)

    # ─────────────────────────────────────────────────────────────────────────────
    # Axis tick labels
    # ─────────────────────────────────────────────────────────────────────────────

    teff_vals  = np.linspace(Teffs[model][0],        Teffs[model][1],        n_T)
    met_vals   = np.linspace(metallicitys[model][0],  metallicitys[model][1], n_m)
    teff_labels = [f"{v:.0f}" for v in teff_vals]
    met_labels  = [f"{v:.1f}" for v in met_vals]

    # Three mu indices: first, middle, last
    mu_indices = [0, n_mu // 2, n_mu - 1]
    stellar_mus_ref = gen_dict['stellar_mus'][model][0, 0, 0]  # use (0,0,0) as reference
    mu_labels = [f"μ = {stellar_mus_ref[idx]:.2f}" for idx in mu_indices]

    # ─────────────────────────────────────────────────────────────────────────────
    # Build the 3 (rows=mu) × 3 (cols=wavelength range) figure
    # Always: x = Teff, y = metallicity, averaged over logg (axis 1)
    # ─────────────────────────────────────────────────────────────────────────────

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle(
        f"{model.upper()} — max $|\\Delta I / I|$ averaged over $\\log g$\n"
        r"$x = T_{\rm eff}$,  $y = $ [M/H]",
        fontsize=14, y=1.01
    )

    # Pre-compute per-column colour limits (shared across all mu rows for that wavelength range)
    # col_vlims = [
    #     (np.nanmin(mean_rel_diff[:, :, :, :, col]),
    #     np.nanmax(mean_rel_diff[:, :, :, :, col]))
    #     for col in range(n_wav_ranges)
    # ]

    for row, (mu_idx, mu_label) in enumerate(zip(mu_indices, mu_labels)):

        for col, wav_label in enumerate(wav_range_labels):

            ax = axes[row, col]
            vmin, vmax = 0.01, 0.5 #col_vlims[col]  # shared across rows for this column

            # Slice this mu and wavelength range → (n_T, n_g, n_m)
            data_3d = mean_rel_diff[:, :, :, mu_idx, col]

            # Average over logg (axis 1) → (n_T, n_m)
            # heatmap wants (n_y, n_x) = (n_met, n_Teff) → transpose
            data_2d      = np.nanmean(data_3d, axis=1)   # (n_T, n_m)
            heatmap_data = data_2d.T                      # (n_m, n_T)

            sns.heatmap(
                heatmap_data,
                ax=ax,
                cmap="coolwarm",
                vmin=vmin, vmax=vmax,
                xticklabels=teff_labels,
                yticklabels=met_labels,
                annot=False,
                cbar=(col == 2),
                cbar_kws={"label": r"$max_\lambda|\Delta I / I|$"},
            )

            ax.set_xlabel("$T_{\\rm eff}$ / K", fontsize=11)
            ax.set_ylabel("[M/H]",              fontsize=11)

            # Thin out tick labels
            for tick_ax in (ax.xaxis, ax.yaxis):
                for idx_t, label in enumerate(tick_ax.get_ticklabels()):
                    if idx_t % (N_star // 5) != 0:
                        label.set_visible(False)

            # Column title (wavelength range) on top row only
            if row == 0:
                ax.set_title(wav_label, fontsize=12, fontweight="bold")

            # Row annotation (mu value) on left column only
            if col == 0:
                ax.set_ylabel(f"{mu_label}\n[M/H]", fontsize=11)

    plt.tight_layout()
    plt.savefig(save_data_path + f"{model}_heatmap_rel_diff.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved heatmap for {model}")

# ─────────────────────────────────────────────────────────────────────────────
# Corner plot — heatmap version via 2D interpolation
# ─────────────────────────────────────────────────────────────────────────────

teff_grid = np.linspace(Teffs[model][0],        Teffs[model][1],        n_T)
logg_grid = np.linspace(loggs[model][0],         loggs[model][1],        n_g)
met_grid  = np.linspace(metallicitys[model][0],  metallicitys[model][1], n_m)

T_flat = np.repeat(np.repeat(teff_grid, n_g * n_m).reshape(n_T, n_g, n_m), 1).ravel()
g_flat = np.tile(np.repeat(logg_grid, n_m), n_T)
m_flat = np.tile(met_grid, n_T * n_g)

color_vals = np.nanmax(mean_rel_diff, axis=(3, 4)).ravel()  # (n_T * n_g * n_m,)

params       = [T_flat,   g_flat,  m_flat ]
param_labels = ["$T_{\\rm eff}$ / K", "$\\log g$", "[M/H]"]
n_params     = len(params)

c_vmin = np.nanpercentile(color_vals, 2)
c_vmax = np.nanpercentile(color_vals, 98)
corner_cmap = plt.cm.coolwarm
norm = matplotlib.colors.Normalize(vmin=c_vmin, vmax=c_vmax)

# Resolution of the interpolated grid in each panel
n_interp = 10

fig_corner, axes_corner = plt.subplots(
    n_params, n_params,
    figsize=(12, 12),
)
fig_corner.suptitle(
    f"{model.upper()} — corner plot\n"
    r"colour $= \max_{\lambda,\,\mu}|\Delta I / I|$",
    fontsize=14, y=1.01,
)

for row in range(n_params):
    for col in range(n_params):
        ax = axes_corner[row, col]

        if col > row:
            ax.set_visible(False)
            continue

        if col == row:
            # Diagonal — 1D histogram
            ax.hist(params[row], bins=n_T, color="steelblue",
                    alpha=0.7, edgecolor="none")
            ax.set_ylabel("Count", fontsize=9)

        else:
            # Lower triangle — interpolated heatmap
            x_data = params[col]   # x = column parameter
            y_data = params[row]   # y = row parameter

            # Build a regular grid spanning the data range
            xi = np.linspace(x_data.min(), x_data.max(), n_interp)
            yi = np.linspace(y_data.min(), y_data.max(), n_interp)
            xi_grid, yi_grid = np.meshgrid(xi, yi)

            # Interpolate scattered (x, y, value) onto the regular grid
            # 'linear' is safe; use 'cubic' for smoother result if no NaNs appear
            zi = griddata(
                points=np.column_stack([x_data, y_data]),
                values=color_vals,
                xi=(xi_grid, yi_grid),
                method='linear',
                fill_value=np.nan,
            )

            im = ax.imshow(
                zi,
                origin='lower',
                extent=[x_data.min(), x_data.max(),
                        y_data.min(), y_data.max()],
                aspect='auto',
                cmap=corner_cmap,
                norm=norm,
                interpolation='bilinear',
            )

        # Axis labels on edges only
        if row == n_params - 1:
            ax.set_xlabel(param_labels[col], fontsize=11)
        else:
            ax.tick_params(labelbottom=False)

        if col == 0 and row != col:
            ax.set_ylabel(param_labels[row], fontsize=11)
        elif col != 0:
            ax.tick_params(labelleft=False)

        ax.tick_params(labelsize=8)

# Single shared colorbar
fig_corner.subplots_adjust(right=0.88, hspace=0.08, wspace=0.08)
cbar_ax = fig_corner.add_axes([0.91, 0.15, 0.02, 0.65])
sm = plt.cm.ScalarMappable(cmap=corner_cmap, norm=norm)
sm.set_array([])
fig_corner.colorbar(sm, cax=cbar_ax,
                    label=r"$\max_{\lambda,\,\mu}|\Delta I / I|$")

fig_corner.savefig(
    save_data_path + f"{model}_heat_max_rel_diff.pdf",
    dpi=300, bbox_inches="tight",
)
plt.close(fig_corner)
print(f"Saved corner plot for {model}")

