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
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from lmfit import minimize, Parameters
import gc
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.interpolate import interp1d

######################################
########## Hyper-parameters ##########
######################################

LD_data_path = '/Volumes/Ajax/Work/PhD/Research/Transit-Information-Content/LD_simulation'
orig_save_data_path = '/Volumes/Ajax/Work/PhD/Research/Transit-Information-Content/Fig2_helper_Storage/'

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

#Number of mu values to interpolate to - set to EJ16 value
n_mu_fine = 100000   # much finer than the native 24 points

n_components = 4
cmap=plt.cm.coolwarm
colors = cmap(np.linspace(0, 1, n_components))

n_clusters = 5

wav_region = [6000, 53000] #0.6 - 5.3 micron

intr_prof_mode = 'build' # 'build' or 'load'
PCA_mode = 'build'

excluded_bp_pairs = [
    # (N_bs_ps - 1, 0),   # b=1 (grazing), p=0.001 (smallest planet)
    # (N_bs_ps - 1, 1),   # b=1 (grazing), p=0.001 (smallest planet)
    # (N_bs_ps - 1, 2),   # b=1 (grazing), p=0.001 (smallest planet)
    # (N_bs_ps - 1, 3),   # b=1 (grazing), p=0.001 (smallest planet)
]

############################
###### Function block ######
############################
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

gen_dict['stellar_wavelengths'] = {model: np.empty((N_star, N_star, N_star), dtype=object) for model in models}

gen_dict['local_rps']={model : np.zeros((N_star, N_star, N_star, N_bs_ps, N_bs_ps, N_chords), dtype=float) for model in models}

gen_dict['local_intensity_profiles']={model: np.empty((N_star, N_star, N_star), dtype=object) for model in models}

gen_dict['intensity_profiles_mask']={model: np.empty((N_star, N_star, N_star), dtype=object) for model in models}

#Iterate over all the stellar models available 
for model in models:
    
    # Create save path for each model
    save_data_path = orig_save_data_path + f'{model}/'
    if not os.path.exists(save_data_path):os.makedirs(save_data_path)

    #Build intensity profiles grid
    if intr_prof_mode == 'build':
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
                    global_stellar_intensities = jnp.copy(sld.stellar_intensities) # Shape: (n_wav, n_mu)
                    del sld  # Free memory from large StellarLimbDarkening object
                
                    #Filter out the portions of wavelength space we don't want
                    cond = ((stellar_wavelengths > wav_region[0]) & (stellar_wavelengths < wav_region[1]))   # Shape: (n_wav,)
                    print(f'    Removing {100 * (len(stellar_wavelengths) - np.sum(cond))/(len(stellar_wavelengths)):.2f} % of the wavelength range')
                    global_stellar_intensities = global_stellar_intensities[cond, :] # Shape: (n_wav, n_mu)
                    stellar_wavelengths = stellar_wavelengths[cond] # Shape: (n_wav,)
                    gen_dict['stellar_wavelengths'][model][i, j, k] = stellar_wavelengths

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
                    stellar_mus_fine = stellar_mus_fine[::-1]
                    global_stellar_intensities_fine = global_stellar_intensities_fine[:, ::-1]

                
                    # Define the annuli edges - the models define intensity spectra at a specific 
                    # mu values so this spreads out these predictions over a band
                    annuli_mus = jnp.append(
                        stellar_mus_fine[:-1] + jnp.diff(stellar_mus_fine)/2,
                        stellar_mus_fine[-1]  + (jnp.diff(stellar_mus_fine)[-1]/2)
                    )
                    
                    # Compute for all (b, p) combinations at once
                    local_stellar_intensities = chord_intensity_vectorized(
                        bs, ps, global_stellar_intensities_fine, jnp.sqrt(1 - annuli_mus**2)
                    ) #shape : (n_bs, n_ps, n_wavelengths, N_chords)

                    #Define the grid of mu values for each ps and bs considered
                    x_max = jnp.sqrt((1 + ps[None, :])**2 - bs[:, None]**2)  # Shape: (n_bs, n_ps)
                    t = jnp.linspace(0.0, 1.0, N_chords)
                    x_vals = x_max[:, :, None] * t[None, None, :]  # Shape: (n_bs, n_ps, N_chords)
                    r_ps = jnp.sqrt(bs[:, None, None]**2 + x_vals**2)  # Shape: (n_bs, n_ps, N_chords)
                    
                    # Normalize the profiles 
                    normalized_profiles = (local_stellar_intensities / local_stellar_intensities[:,:,:,0:1])

                    # Filter out profiles that have increases 
                    mask = ~jnp.any(jnp.diff(local_stellar_intensities, axis=-1) > 0.0, axis=-1)
                    gen_dict['intensity_profiles_mask'][model][i, j, k]  = np.array(mask)                 # (n_bs, n_ps, n_wav)

                    #Printing mask results
                    n_total   = N_bs_ps * N_bs_ps * normalized_profiles.shape[2]
                    n_removed = n_total - int(jnp.sum(mask))
                    if n_removed!=0:print(f'    Removing {100 * n_removed / n_total:.2f} % of individual profiles')

                    #Normalize and store this local intensity profile
                    gen_dict['local_intensity_profiles'][model][i, j, k] = normalized_profiles # (n_bs, n_ps, n_wav_valid, N_chords)
                    gen_dict['local_rps'][model][i, j, k, :, :, :] = jnp.linspace(0.0, 1.0, N_chords)

                    #Garbage collection
                    del local_stellar_intensities, global_stellar_intensities, normalized_profiles, annuli_mus, \
                    x_max, x_vals, t, mask, cond, stellar_wavelengths, stellar_mus, interp_func, stellar_mus_fine, global_stellar_intensities_fine
                    gc.collect()


        #Store the stellar spectrum
        with open(save_data_path + 'data.pkl', 'wb') as f:pickle.dump(gen_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    #Load intensity profiles grid
    elif intr_prof_mode == 'load':
        with open(save_data_path + 'data.pkl', 'rb') as f:gen_dict = pickle.load(f)
    
    else:
        raise KeyboardInterrupt('Mode not recognized.')

    ##########################################
    ########## PCA analysis ##################
    ##########################################
    outlier_profiles = []
    outlier_xs = []

    if PCA_mode == 'build':
            
        print('MASKING')

        # ─────────────────────────────────────────────────────────────────────────────
        # Pass 1 — count n_valid separately for each b, p, and wavelength region
        # ─────────────────────────────────────────────────────────────────────────────
        n_wav_bins     = 10

        # Compute once — split wavelength array directly into 10 equal chunks by index
        wavs_ref   = np.array(gen_dict['stellar_wavelengths'][model][0, 0, 0])  # (n_wav,)
        wav_bins   = np.array_split(wavs_ref, n_wav_bins)                        # list of 10 sub-arrays
        wav_bin_labels = [f'{b[0]/1e4:.2f}-{b[-1]/1e4:.2f} μm' for b in wav_bins]

        # Pre-compute the index slices once — avoids rebuilding boolean masks in the loop
        wav_bin_slices = np.array_split(np.arange(len(wavs_ref)), n_wav_bins)   # list of 10 index arrays

        n_valid_per_b   = np.zeros(N_bs_ps,   dtype=int)
        n_valid_per_p   = np.zeros(N_bs_ps,   dtype=int)
        n_valid_per_wav = np.zeros(n_wav_bins, dtype=int)
        n_considered    = 0
        n_valid         = 0

        for i in range(N_star):
            for j in range(N_star):
                for k in range(N_star):
                    mask_entry = gen_dict['intensity_profiles_mask'][model][i, j, k].copy()  # (n_bs, n_ps, n_wav)
                    wavs       = gen_dict['stellar_wavelengths'][model][i, j, k]              # (n_wav,)

                    for ib_excl, ip_excl in excluded_bp_pairs:
                        mask_entry[ib_excl, ip_excl, :] = False

                    n_considered += mask_entry.size
                    n_valid      += int(np.sum(mask_entry))

                    for ib in range(N_bs_ps):
                        n_valid_per_b[ib] += int(np.sum(mask_entry[ib, :, :]))
                    for ip in range(N_bs_ps):
                        n_valid_per_p[ip] += int(np.sum(mask_entry[:, ip, :]))
                    for iw_bin in range(n_wav_bins):
                        idx = wav_bin_slices[iw_bin]
                        n_valid_per_wav[iw_bin] += int(np.sum(mask_entry[:, :, idx]))

        print(f"=== Profile filtering summary ===")
        print(f"Total profiles considered : {n_considered}")
        print(f"Total valid               : {n_valid} ({100 * n_valid / n_considered:.1f} %)")
        print(f"=================================")
        print(f"Per impact parameter b:")
        for ib in range(N_bs_ps):
            n_considered_b = n_considered // N_bs_ps
            print(f"  b[{ib}]={float(bs[ib]):.3f} : {n_valid_per_b[ib]}/{n_considered_b} valid "
                f"({100 * n_valid_per_b[ib] / n_considered_b:.1f} %)")
        print(f"=================================")
        print(f"Per planet-to-star radius ratio p:")
        for ip in range(N_bs_ps):
            n_considered_p = n_considered // N_bs_ps
            print(f"  p[{ip}]={float(ps[ip]):.5f} : {n_valid_per_p[ip]}/{n_considered_p} valid "
                f"({100 * n_valid_per_p[ip] / n_considered_p:.1f} %)")
        print(f"=================================")
        print(f"Per wavelength region:")
        n_total_per_wav = np.array([len(idx) * N_bs_ps * N_bs_ps * N_star * N_star * N_star for idx in wav_bin_slices])
        for iw_bin in range(n_wav_bins):
            print(f"  {wav_bin_labels[iw_bin]} : {n_valid_per_wav[iw_bin]}/{n_total_per_wav[iw_bin]} valid "
                f"({100 * n_valid_per_wav[iw_bin] / n_total_per_wav[iw_bin]:.1f} %)")

        # ─────────────────────────────────────────────────────────────────────────────
        # Pass 2 — pre-allocate one array per b, fill with write pointers
        # Also store group_id per valid profile so we can later find the common
        # subset valid across all b values
        # group_id = unique integer per (i,j,k,ip,iw) combination
        # ─────────────────────────────────────────────────────────────────────────────
        pca_int_profile = [np.empty((n_valid_per_b[ib], N_chords),  dtype=np.float32) for ib in range(N_bs_ps)]
        xs_per_b        = [np.empty((n_valid_per_b[ib], N_chords),  dtype=np.float32) for ib in range(N_bs_ps)]
        group_ids       = [np.empty( n_valid_per_b[ib],             dtype=np.int64  ) for ib in range(N_bs_ps)]
        # Add metadata arrays — 5 columns: [ip, i(Teff), j(logg), k(met), iw]
        meta_per_b = [np.empty((n_valid_per_b[ib], 5), dtype=np.int32) for ib in range(N_bs_ps)]

        ptrs       = np.zeros(N_bs_ps, dtype=int)
        group_base = 0   # running offset so each (i,j,k) star gets a unique block of IDs

        for i in range(N_star):
            for j in range(N_star):
                for k in range(N_star):
                    entry      = np.array(gen_dict['local_intensity_profiles'][model][i, j, k])  # (n_bs, n_ps, n_wav, N_chords)
                    mask_entry = gen_dict['intensity_profiles_mask'][model][i, j, k]              # (n_bs, n_ps, n_wav)
                    rps_entry  = gen_dict['local_rps'][model][i, j, k]                            # (n_bs, n_ps, N_chords)

                    for ib_excl, ip_excl in excluded_bp_pairs:
                        mask_entry[ib_excl, ip_excl, :] = False

                    n_ps_  = entry.shape[1]
                    n_wav_ = entry.shape[2]

                    # group_id for each (ip, iw) pair within this star
                    # shape (n_ps * n_wav,) — unique across all stars via group_base
                    local_group_ids = (group_base
                                    + np.arange(n_ps_ * n_wav_, dtype=np.int64))  # (n_ps*n_wav,)

                    for ib in range(N_bs_ps):
                        # profiles at this b: (n_ps, n_wav, N_chords) → (n_ps*n_wav, N_chords)
                        prof_ib   = entry[ib].reshape(-1, N_chords)        # (n_ps*n_wav, N_chords)
                        mask_ib   = mask_entry[ib].ravel()                  # (n_ps*n_wav,)
                        rps_ib    = rps_entry[ib].reshape(n_ps_, 1, N_chords)
                        rps_ib    = np.repeat(rps_ib, n_wav_, axis=1).reshape(-1, N_chords)

                        # Metadata: for each (ip, iw) row, record its indices
                        ip_idx = np.repeat(np.arange(n_ps_), n_wav_)   # (n_ps*n_wav,)
                        iw_idx = np.tile  (np.arange(n_wav_), n_ps_)   # (n_ps*n_wav,)
                        i_idx  = np.full(n_ps_ * n_wav_, i,  dtype=np.int32)
                        j_idx  = np.full(n_ps_ * n_wav_, j,  dtype=np.int32)
                        k_idx  = np.full(n_ps_ * n_wav_, k,  dtype=np.int32)
                        meta   = np.stack([ip_idx, i_idx, j_idx, k_idx, iw_idx], axis=1)  # (n_ps*n_wav, 5)

                        n_ijk = int(np.sum(mask_ib))
                        if n_ijk == 0:
                            continue

                        ptr = ptrs[ib]
                        pca_int_profile[ib][ptr : ptr + n_ijk] = prof_ib        [mask_ib]
                        xs_per_b       [ib][ptr : ptr + n_ijk] = rps_ib         [mask_ib]
                        group_ids      [ib][ptr : ptr + n_ijk] = local_group_ids[mask_ib]
                        meta_per_b     [ib][ptr : ptr + n_ijk] = meta   [mask_ib]
                        ptrs[ib] += n_ijk

                    group_base += n_ps_ * n_wav_
                    del entry, mask_entry, rps_entry, prof_ib, mask_ib, rps_ib
                    gc.collect()

        for ib in range(N_bs_ps):
            assert ptrs[ib] == n_valid_per_b[ib], f"Pointer mismatch at b[{ib}]"
            #Reshape the xs
            xs_per_b[ib] = xs_per_b[ib][:, ::-1]

        # ─────────────────────────────────────────────────────────────────────────────
        # PCA — one per b value
        # ─────────────────────────────────────────────────────────────────────────────
        print('PCA ANALYSIS')
        pcas         = []
        profiles_pca = []   # PCA scores per b: list of (n_valid_ib, n_components)

        for ib in range(N_bs_ps):
            if ib == N_bs_ps - 1:  # grazing case

                meta_ib = meta_per_b[ib]   # (n_valid_ib, 5): [ip, i, j, k, iw]
                med     = np.median(pca_int_profile[ib], axis=0)
                resid   = pca_int_profile[ib] - med    # (n_valid_ib, N_chords)

                # Assign each profile to a wavelength bin
                wav_bin_of_profile = np.empty(len(meta_ib), dtype=np.int32)
                for iw_bin, idx_slice in enumerate(wav_bin_slices):
                    in_bin = np.isin(meta_ib[:, 4], idx_slice)
                    wav_bin_of_profile[in_bin] = iw_bin

                # Recover physical values for each profile
                T_vals   = np.linspace(Teffs[model][0],       Teffs[model][1],       N_star)
                g_vals   = np.linspace(loggs[model][0],        loggs[model][1],        N_star)
                m_vals   = np.linspace(metallicitys[model][0], metallicitys[model][1], N_star)
                p_vals   = np.array(ps)

                color_sources = {
                    'Planet size $p$'  : p_vals  [meta_ib[:, 0]],
                    '$T_{eff}$ (K)'    : T_vals  [meta_ib[:, 1]],
                    '$\\log g$'        : g_vals  [meta_ib[:, 2]],
                    'Metallicity [M/H]': m_vals  [meta_ib[:, 3]],
                    'Wavelength bin'   : wav_bin_of_profile.astype(float),
                }

                step      = max(1, len(resid) // 20000)   # subsample for speed
                fig_g, axes_g = plt.subplots(1, 5, figsize=(30, 5), sharex=True, sharey=True)
                fig_g.suptitle(f'Grazing profiles (b={float(bs[ib]):.2f}) coloured by parameter',
                               fontsize=12)

                for ax, (clabel, cvals) in zip(axes_g, color_sources.items()):
                    cmap_g = cm.get_cmap('coolwarm')
                    norm_g = mcolors.Normalize(vmin=np.min(cvals), vmax=np.max(cvals))

                    for idx in range(0, len(resid), step):
                        ax.plot(xs_per_b[ib][idx], resid[idx],
                                alpha=0.15, linewidth=0.4,
                                color=cmap_g(norm_g(cvals[idx])))

                    # Colorbar
                    sm = cm.ScalarMappable(cmap=cmap_g, norm=norm_g)
                    sm.set_array([])
                    plt.colorbar(sm, ax=ax, label=clabel, fraction=0.046, pad=0.04)

                    ax.set_xlabel('$r / R_\\star$')
                    ax.set_ylabel('Residual intensity' if ax is axes_g[0] else '')
                    ax.set_title(clabel)
                    ax.grid(True, alpha=0.3)

                fig_g.tight_layout()
                plt.show()
                # fig_g.savefig(save_data_path + f'Grazing_profiles_coloured_{model}.png',
                #               dpi=150, bbox_inches='tight')
                # plt.close(fig_g)
                
        for ib in range(N_bs_ps):
            print(f'  Fitting PCA for b[{ib}]={float(bs[ib]):.3f} '
                f'on {n_valid_per_b[ib]} profiles ...')
            pca_b = PCA(n_components=n_components)
            scores_b = pca_b.fit_transform(pca_int_profile[ib])
            pcas.append(pca_b)
            profiles_pca.append(scores_b)
            print(f'    Variance captured: {np.sum(pca_b.explained_variance_ratio_)*100:.1f}%')

        # ─────────────────────────────────────────────────────────────────────────────
        # Find common group IDs valid across ALL b values
        # These are the only profiles we can meaningfully compare across b
        # ─────────────────────────────────────────────────────────────────────────────
        print('FINDING COMMON PROFILES')
        common_ids = set(group_ids[0])
        for ib in range(1, N_bs_ps):
            common_ids &= set(group_ids[ib])
        common_ids = np.array(sorted(common_ids), dtype=np.int64)
        n_common   = len(common_ids)
        print(f'  Profiles valid for all b values: {n_common}')

        # For each b, build index arrays mapping common_ids → row in pca_int_profile[ib]
        # and a combined score matrix (n_common, N_bs_ps * n_components) for clustering
        id_to_row = []
        for ib in range(N_bs_ps):
            sorter      = np.argsort(group_ids[ib])
            rows        = sorter[np.searchsorted(group_ids[ib], common_ids, sorter=sorter)]
            id_to_row.append(rows)

        combined_scores = np.concatenate(
            [profiles_pca[ib][id_to_row[ib]] for ib in range(N_bs_ps)],
            axis=1
        )  # (n_common, N_bs_ps * n_components)

        # ─────────────────────────────────────────────────────────────────────────────
        # Clustering on combined PCA space
        # ─────────────────────────────────────────────────────────────────────────────
        print('CLUSTERING')
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(combined_scores[:, :3])

        distances_from_centers = kmeans.transform(combined_scores[:, :3])
        typical_common_idx     = int(np.argmin(np.min(distances_from_centers, axis=1)))

        outlier_common_indices = []
        for cluster_id in range(n_clusters):
            cmask     = cluster_labels == cluster_id
            cdists    = distances_from_centers[cmask, cluster_id]
            outlier_common_indices.append(int(np.where(cmask)[0][np.argmax(cdists)]))

        # ─────────────────────────────────────────────────────────────────────────────
        # Figure 1: one row per b — scree, cumulative variance, eigen profiles
        # ─────────────────────────────────────────────────────────────────────────────
        print('PLOTTING')
        print('    FIGURE 1')
        b_colors  = plt.cm.plasma(np.linspace(0.1, 0.9, N_bs_ps))
        ncols_f1  = 2 + n_components   # scree | cumvar | eigen_1 ... eigen_n
        fig1, axes1 = plt.subplots(N_bs_ps, ncols_f1,
                                    figsize=(4 * ncols_f1, 4 * N_bs_ps))

        for ib in range(N_bs_ps):
            pca_b  = pcas[ib]
            eigen  = pca_b.components_   # (n_components, N_chords)
            evr    = pca_b.explained_variance_ratio_

            # Scree
            ax = axes1[ib, 0]
            ax.plot(range(1, n_components+1), evr, 'o-', color=b_colors[ib], linewidth=2)
            ax.set_title(f'b={float(bs[ib]):.3f}  Scree')
            ax.set_xlabel('PC')
            ax.set_ylabel('Expl. var. ratio')
            ax.grid(True, alpha=0.3)

            # Cumulative
            ax = axes1[ib, 1]
            ax.plot(range(1, n_components+1), np.cumsum(evr), 'o-', color=b_colors[ib], linewidth=2)
            ax.axhline(0.95, color='g', linestyle='--', label='95%')
            ax.set_title(f'b={float(bs[ib]):.3f}  Cumul. var.')
            ax.set_xlabel('PC')
            ax.set_ylabel('Cumul. expl. var.')
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

            # Eigen profiles
            for i_plot in range(n_components):
                ax = axes1[ib, 2 + i_plot]
                ax.plot(xs_per_b[ib][0], eigen[i_plot],
                        color=colors[i_plot], linewidth=1.5)
                ax.axhline(0, color='k', linestyle='--', alpha=0.3)
                ax.set_title(f'b={float(bs[ib]):.2f}  PC{i_plot+1} ({evr[i_plot]*100:.1f}%)')
                ax.set_xlabel('$r/R_\\star$')
                ax.set_ylabel('Component value')
                ax.grid(True, alpha=0.3)

        fig1.tight_layout()
        fig1.savefig(save_data_path + 'PCA_Analysis.png', dpi=150, bbox_inches='tight')
        plt.close(fig1)

        # ─────────────────────────────────────────────────────────────────────────────
        # Figure 2: combined PCA scatter + typical/outlier profiles
        # Top row: PCA scatter per b coloured by cluster
        # Bottom rows: typical and outlier profiles per b
        # ─────────────────────────────────────────────────────────────────────────────
        print('    FIGURE 2')
        n_specials  = 1 + n_clusters   # typical + one outlier per cluster
        fig2, axes2 = plt.subplots(1 + N_bs_ps, n_specials,
                                    figsize=(5 * n_specials, 4 * (1 + N_bs_ps)),
                                    sharey='row')

        # Top row: PCA scatter (PC1 vs PC2) per b, coloured by cluster label
        for ib in range(N_bs_ps):
            ax = axes2[0, ib] if n_specials > 1 else axes2[0]
            sc = ax.scatter(profiles_pca[ib][id_to_row[ib], 0],
                            profiles_pca[ib][id_to_row[ib], 1],
                            c=cluster_labels, cmap='viridis', s=15, alpha=0.5)
            ax.set_xlabel(f'PC1 b={float(bs[ib]):.2f}')
            ax.set_ylabel('PC2')
            ax.set_title(f'PCA space b={float(bs[ib]):.3f}')
            plt.colorbar(sc, ax=ax, label='Cluster')
            ax.grid(True, alpha=0.3)

        # Hide unused top-row panels
        for col in range(N_bs_ps, n_specials):
            axes2[0, col].set_visible(False)

        # Bottom N_bs_ps rows: one special profile per column, one b per row
        special_common_indices = [typical_common_idx] + outlier_common_indices
        special_labels         = ['Typical'] + [f'Outlier {c}' for c in range(n_clusters)]
        special_colors         = ['blue'] + ['red'] * n_clusters

        for ib in range(N_bs_ps):
            rows_ib = id_to_row[ib]
            for col, (cidx, slabel, scol) in enumerate(
                    zip(special_common_indices, special_labels, special_colors)):
                ax = axes2[1 + ib, col]
                # Background: all profiles at this b (subsampled)
                n_v = n_valid_per_b[ib]
                for nval in range(0, n_v, max(1, n_v // 200)):
                    ax.plot(xs_per_b[ib][nval], pca_int_profile[ib][nval],
                            alpha=0.15, color='gray', linewidth=0.3)
                # Highlighted profile
                row_in_b = rows_ib[cidx]
                ax.plot(xs_per_b[ib][row_in_b], pca_int_profile[ib][row_in_b],
                        color=scol, linewidth=2, label=slabel, zorder=10)
                ax.set_xlabel('$r/R_\\star$')
                ax.set_ylabel('Norm. Intensity')
                ax.set_title(f'{slabel}  b={float(bs[ib]):.3f}')
                ax.legend(fontsize=7)
                ax.grid(True, alpha=0.3)

        fig2.tight_layout()
        fig2.savefig(save_data_path + 'Mode_and_Outliers.png', dpi=150, bbox_inches='tight')
        plt.close(fig2)

        # ─────────────────────────────────────────────────────────────────────────────
        # Figure 3: reconstruction quality for the typical profile — one col per b
        # ─────────────────────────────────────────────────────────────────────────────
        print('    FIGURE 3')
        from matplotlib.gridspec import GridSpec

        n_comp_list = [1, max(2, n_components // 2), n_components]
        fig3 = plt.figure(figsize=(5 * N_bs_ps, 6 * len(n_comp_list)))
        gs   = GridSpec(len(n_comp_list) * 2, N_bs_ps, figure=fig3,
                        hspace=0.05, wspace=0.3)

        for col_ib, ib in enumerate(range(N_bs_ps)):
            row_in_b = id_to_row[ib][typical_common_idx]
            original = pca_int_profile[ib][row_in_b]
            x_orig   = xs_per_b[ib][row_in_b]

            for row_idx, n_comp_plot in enumerate(n_comp_list):
                pca_temp      = PCA(n_components=n_comp_plot)
                pca_temp.fit(pca_int_profile[ib])
                reconstructed = pca_temp.inverse_transform(
                                    pca_temp.transform(original.reshape(1, -1)))[0]
                residual = original - reconstructed
                rmse     = np.sqrt(np.mean(residual**2))

                ax_top = fig3.add_subplot(gs[row_idx * 2,     col_ib])
                ax_bot = fig3.add_subplot(gs[row_idx * 2 + 1, col_ib], sharex=ax_top)

                ax_top.plot(x_orig, original,      color=b_colors[ib], linewidth=2,  label='Original')
                ax_top.plot(x_orig, reconstructed, color=b_colors[ib], linewidth=1.2,
                            linestyle='--', label='Recon.')
                ax_top.set_title(f'b={float(bs[ib]):.2f}  {n_comp_plot}PC  RMSE={rmse:.4f}',
                                fontsize=8)
                ax_top.set_ylabel('Norm. Intensity', fontsize=7)
                ax_top.legend(fontsize=6)
                ax_top.grid(True, alpha=0.3)
                ax_top.tick_params(labelbottom=False)

                safe = np.where(np.abs(original) < 1e-10, np.nan, original)
                ax_bot.plot(x_orig, 100 * residual / safe, color=b_colors[ib], linewidth=1.0)
                ax_bot.axhline(0, color='k', linestyle='--', alpha=0.5)
                ax_bot.set_xlabel('$r/R_\\star$', fontsize=7)
                ax_bot.set_ylabel('Rel. diff. (%)', fontsize=7)
                ax_bot.grid(True, alpha=0.3)

        fig3.savefig(save_data_path + 'Reconstruction_Quality.png', dpi=150, bbox_inches='tight')
        plt.close(fig3)

        # ─────────────────────────────────────────────────────────────────────────────
        # Save
        # ─────────────────────────────────────────────────────────────────────────────
        print(f"\n=== PCA Analysis Summary ===")
        for ib in range(N_bs_ps):
            print(f"  b[{ib}]={float(bs[ib]):.3f}: "
                f"{np.sum(pcas[ib].explained_variance_ratio_)*100:.1f}% variance in {n_components} PCs")
        print(f"Common profiles (valid for all b): {n_common}")

        # Save one profile vector per b for each special profile
        for ib in range(N_bs_ps):
            row_typical = id_to_row[ib][typical_common_idx]
            np.save(save_data_path + f'mode_intensity_profile_{model}_b{ib}.npy',
                    pca_int_profile[ib][row_typical])
            np.save(save_data_path + f'mode_rs_{model}_b{ib}.npy',
                    xs_per_b[ib][row_typical])
            for i_save, cidx in enumerate(outlier_common_indices):
                row_out = id_to_row[ib][cidx]
                np.save(save_data_path + f'outlier{i_save+1}_intensity_profile_{model}_b{ib}.npy',
                        pca_int_profile[ib][row_out])
                np.save(save_data_path + f'outlier{i_save+1}_rs_{model}_b{ib}.npy',
                        xs_per_b[ib][row_out])

        # Also build the flat lists needed for Figure 4
        typical_profile = [pca_int_profile[ib][id_to_row[ib][typical_common_idx]] for ib in range(N_bs_ps)]
        typical_xs      = [xs_per_b[ib]       [id_to_row[ib][typical_common_idx]] for ib in range(N_bs_ps)]
        outlier_profiles = [[pca_int_profile[ib][id_to_row[ib][cidx]] for ib in range(N_bs_ps)]
                            for cidx in outlier_common_indices]
        outlier_xs       = [[xs_per_b[ib]       [id_to_row[ib][cidx]] for ib in range(N_bs_ps)]
                            for cidx in outlier_common_indices]
        print(f"Saved profiles to {save_data_path}")

    elif PCA_mode == 'load':
        b_colors  = plt.cm.plasma(np.linspace(0.1, 0.9, N_bs_ps))

        # Load one profile per b per special case
        typical_profile = [np.load(save_data_path + f'mode_intensity_profile_{model}_b{ib}.npy')
                           for ib in range(N_bs_ps)]
        typical_xs      = [np.load(save_data_path + f'mode_rs_{model}_b{ib}.npy')
                           for ib in range(N_bs_ps)]

        outlier_profiles = []
        outlier_xs       = []
        for i_save in range(n_clusters):
            outlier_profiles.append(
                [np.load(save_data_path + f'outlier{i_save+1}_intensity_profile_{model}_b{ib}.npy')
                 for ib in range(N_bs_ps)]
            )
            outlier_xs.append(
                [np.load(save_data_path + f'outlier{i_save+1}_rs_{model}_b{ib}.npy')
                 for ib in range(N_bs_ps)]
            )
        print(f"\nLoaded profiles from {save_data_path}")

    else:
        raise KeyboardInterrupt('Wrong PCA mode')

    # ─────────────────────────────────────────────────────────────────────────────
    # Figure 4: 4th-order NLLD fit
    # One figure per special profile (typical + n_clusters outliers)
    # Within each figure: one row per b value
    # ─────────────────────────────────────────────────────────────────────────────
    print('    FIGURE 4')

    def fourNLLD(x, coeffs):
        return (1 - coeffs[0] * (1 - x**0.5)
                  - coeffs[1] * (1 - x)
                  - coeffs[2] * (1 - x**1.5)
                  - coeffs[3] * (1 - x**2))

    def residual_fn(params, x, base_prof):
        return fourNLLD(x, [params[f'c{ic+1}'].value for ic in range(4)]) - base_prof

    # Bundle all special profiles: list of (label, profiles_per_b, xs_per_b)
    specials = (
        [('mode', typical_profile, typical_xs)]
        + [(f'outlier{i+1}', outlier_profiles[i], outlier_xs[i])
           for i in range(n_clusters)]
    )

    for label, prof_per_b, rps_per_b in specials:

        fig4, axes4 = plt.subplots(
            N_bs_ps, 2,
            figsize=(12, 4 * N_bs_ps),
            sharex=False,
            gridspec_kw={'width_ratios': [3, 1]}
        )
        fig4.suptitle(f'4th Order NLLD Fit — {label}', fontsize=13)

        for ib in range(N_bs_ps):
            mus_ib  = np.array(rps_per_b[ib])   # (N_chords,)  mu from star centre
            prof_ib = np.array(prof_per_b[ib])  # (N_chords,)  normalised intensity

            # ── Fit ───────────────────────────────────────────────────────────
            params = Parameters()
            for ip in range(4):
                params.add(f'c{ip+1}', value=np.random.uniform(0, 1))
            result  = minimize(residual_fn, params, args=(mus_ib, prof_ib))
            coeffs  = [result.params[f'c{ic+1}'].value for ic in range(4)]
            fit_ib  = fourNLLD(mus_ib, coeffs)

            # ── Plot ──────────────────────────────────────────────────────────
            ax1 = axes4[ib, 0]
            ax2 = axes4[ib, 1]

            ax1.plot(mus_ib, prof_ib, 'o',
                     color=b_colors[ib], markersize=3, alpha=0.6,
                     label=f'b={float(bs[ib]):.3f}')
            ax1.plot(mus_ib, fit_ib, 'r--', linewidth=2, label='4th order NLLD')
            ax1.set_ylabel('Norm. Intensity')
            ax1.set_title(f'b = {float(bs[ib]):.3f}', fontsize=9)
            ax1.legend(fontsize=7)
            ax1.grid(True, alpha=0.3)
            if ib < N_bs_ps - 1:
                ax1.tick_params(labelbottom=False)
            else:
                ax1.set_xlabel('μ = cos(θ)')

            safe = np.where(np.abs(prof_ib) < 1e-10, np.nan, prof_ib)
            ax2.plot(mus_ib, 100 * (prof_ib - fit_ib) / safe,
                     '--', color=b_colors[ib], linewidth=1.5)
            ax2.axhline(0, color='k', linestyle='--', alpha=0.5)
            ax2.set_ylabel('Rel. diff. (%)')
            ax2.grid(True, alpha=0.3)
            if ib < N_bs_ps - 1:
                ax2.tick_params(labelbottom=False)
            else:
                ax2.set_xlabel('μ = cos(θ)')

            # Print fit coefficients
            print(f"  {label}  b[{ib}]={float(bs[ib]):.3f}  "
                  f"c=[{', '.join(f'{c:.4f}' for c in coeffs)}]  "
                  f"redchi={result.redchi:.4e}")

        fig4.tight_layout()
        fig4.savefig(
            save_data_path + f'4thOrderNLLD_Fit_Profile_{label}_{model}.png',
            dpi=150, bbox_inches='tight'
        )
        plt.close(fig4)