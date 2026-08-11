#############################
########## Purpose ##########
#############################

# Generates the full (unsubsampled) LDC datasets for both MPS-ATLAS set1 and
# set2 for the inter-model comparison in Appendix 5.  For each model the script
# loads the raw intensity profiles from a data.pkl (produced by running
# Fig3_run.py with intr_prof_mode='build' for both mps1 and mps2), applies the
# same wavelength-filtering and monotonicity masking as Fig3_run.py, fits ALL
# valid profiles with the 4th-order NLLD, and saves corner_data + corner_meta
# to Appendix5_Storage/{model}/results.npz.  No subsampling is applied so that
# the plot script can intersect on (i_Teff, j_logg, k_met, i_wav) tuples and
# obtain ~580 k matched pairs from the full parameter space.


######################################
########## Import libraries ##########
######################################

import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import exotic_ld as el
import pickle
from lmfit import minimize, Parameters
import gc
from scipy.interpolate import interp1d
from tqdm import tqdm
import paths


######################################
########## Hyper-parameters ##########
######################################

LD_data_path = '/Volumes/Ajax/Work/PhD/Research/Transit-Information-Content/LD_simulation'

# Paths to pre-built data.pkl files for each model.  Set a value to None to
# build from scratch via ExoTiC-LD (slow — only needed if pkl is missing).
pkl_load_paths = {
    'mps1': str(paths.data / "Fig3_Storage") + "/mps1/data.pkl",
    'mps2': str(paths.data / "Fig3_Storage") + "/mps2/data.pkl",
}

models = ['mps1', 'mps2']

N_star    = 10        # grid points per stellar parameter axis — must match Fig3_run.py
n_mu_fine = 100       # mu points after cubic-spline interpolation
wav_region = [6000, 53000]  # Angstrom  (0.6 – 5.3 µm, JWST NIRSpec PRISM)

Teff_ranges = {'mps1': [3500, 9000], 'mps2': [3500, 9000]}
logg_ranges = {'mps1': [3.0,  5.0],  'mps2': [3.0,  5.0]}
met_ranges  = {'mps1': [-5.0, 1.5],  'mps2': [-5.0, 1.5]}

# Seed for the random initial guesses used when fitting each profile's 4th-order NLLD
# coefficients (see Fig3_run.py's `fit_init_seed` for the same convention) - fixed so the
# fit initial guesses, and therefore results.npz, are reproducible across runs.
fit_init_seed = 42


############################
###### Function block ######
############################

def fourNLLD(x, coeffs):
    return (1
            - coeffs[0] * (1 - x ** 0.5)
            - coeffs[1] * (1 - x)
            - coeffs[2] * (1 - x ** 1.5)
            - coeffs[3] * (1 - x ** 2))


def residual_fn(params, x, base_prof):
    return fourNLLD(x, [params[f'c{ic+1}'].value for ic in range(4)]) - base_prof


################################
########## Code block ##########
################################

for model in models:

    save_data_path = str(paths.data / "Appendix5_Storage") + f"/{model}/"
    os.makedirs(save_data_path, exist_ok=True)

    T_vals_arr = np.linspace(Teff_ranges[model][0], Teff_ranges[model][1], N_star)
    g_vals_arr = np.linspace(logg_ranges[model][0], logg_ranges[model][1], N_star)
    m_vals_arr = np.linspace(met_ranges[model][0],  met_ranges[model][1],  N_star)

    # ── 1. Build or load intensity profiles ──────────────────────────────────

    gen_dict = {
        'stellar_wavelengths':       np.empty((N_star, N_star, N_star), dtype=object),
        'global_intensity_profiles': np.empty((N_star, N_star, N_star), dtype=object),
        'global_mus':                np.empty((N_star, N_star, N_star), dtype=object),
    }

    pkl_path = pkl_load_paths.get(model)

    # Also check a local copy that may have been saved on a previous run
    local_pkl = save_data_path + 'data.pkl'
    resolved_pkl = pkl_path if (pkl_path and os.path.exists(pkl_path)) else \
                   (local_pkl if os.path.exists(local_pkl) else None)

    if resolved_pkl:
        print(f'[{model}] Loading intensity profiles from {resolved_pkl}')
        with open(resolved_pkl, 'rb') as f:
            raw = pickle.load(f)
        # data.pkl written by Fig3_run.py wraps arrays in a model-keyed dict
        for key in ('stellar_wavelengths', 'global_intensity_profiles', 'global_mus'):
            src = raw[key]
            gen_dict[key] = src[model] if isinstance(src, dict) else src
    else:
        print(f'[{model}] Building intensity profiles from scratch ...')
        for i, T in enumerate(T_vals_arr):
            for j, g in enumerate(g_vals_arr):
                for k, m in enumerate(m_vals_arr):
                    print(f'  T={T:.0f} K  logg={g:.2f}  [M/H]={m:.2f}')
                    sld = el.StellarLimbDarkening(
                        M_H=m, Teff=T, logg=g,
                        ld_model=model,
                        ld_data_path=LD_data_path,
                        interpolate_type='nearest',
                    )
                    stellar_wavelengths        = np.array(sld.stellar_wavelengths)
                    stellar_mus                = np.array(sld.mus)
                    global_stellar_intensities = np.array(sld.stellar_intensities)
                    del sld

                    cond = ((stellar_wavelengths > wav_region[0]) &
                            (stellar_wavelengths < wav_region[1]))
                    global_stellar_intensities = global_stellar_intensities[cond, :]
                    stellar_wavelengths        = stellar_wavelengths[cond]
                    gen_dict['stellar_wavelengths'][i, j, k] = stellar_wavelengths

                    stellar_mus_fine = np.linspace(stellar_mus[-1], stellar_mus[0], n_mu_fine)
                    interp_func = interp1d(
                        stellar_mus[::-1],
                        global_stellar_intensities[:, ::-1],
                        kind='cubic', axis=1, bounds_error=False,
                    )
                    fine             = interp_func(stellar_mus_fine)
                    stellar_mus_fine = stellar_mus_fine[::-1]
                    fine             = fine[:, ::-1]

                    gen_dict['global_intensity_profiles'][i, j, k] = fine
                    gen_dict['global_mus'][i, j, k]                = stellar_mus_fine

                    del global_stellar_intensities, stellar_wavelengths, stellar_mus
                    del interp_func, stellar_mus_fine, fine, cond
                    gc.collect()

        with open(save_data_path + 'data.pkl', 'wb') as f:
            pickle.dump(gen_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'[{model}] Saved data.pkl → {save_data_path}')

    # ── 2. Monotonicity filtering ─────────────────────────────────────────────

    wavs_ref = np.array(gen_dict['stellar_wavelengths'][0, 0, 0])
    n_wav    = len(wavs_ref)

    n_valid = 0
    for i in range(N_star):
        for j in range(N_star):
            for k in range(N_star):
                prof        = gen_dict['global_intensity_profiles'][i, j, k]
                center_vals = prof[:, 0:1]
                safe_center = np.where(np.abs(center_vals) < 1e-10, 1.0, center_vals)
                norm_prof   = prof / safe_center
                n_valid    += int(np.sum(~np.any(np.diff(norm_prof, axis=1) > 0.0, axis=1)))

    print(f'[{model}] Valid profiles: {n_valid:,} / {N_star**3 * n_wav:,}')

    int_profile = np.empty((n_valid, n_mu_fine), dtype=np.float32)
    mus_array   = np.empty((n_valid, n_mu_fine), dtype=np.float32)
    meta_array  = np.empty((n_valid, 4),         dtype=np.int32)
    ptr = 0

    for i in range(N_star):
        for j in range(N_star):
            for k in range(N_star):
                prof = gen_dict['global_intensity_profiles'][i, j, k]
                mus  = gen_dict['global_mus'][i, j, k]

                center_vals = prof[:, 0:1]
                safe_center = np.where(np.abs(center_vals) < 1e-10, 1.0, center_vals)
                norm_prof   = prof / safe_center
                mask        = ~np.any(np.diff(norm_prof, axis=1) > 0.0, axis=1)
                n_ok        = int(np.sum(mask))
                if n_ok == 0:
                    continue

                iw_indices                = np.where(mask)[0]
                int_profile[ptr:ptr+n_ok] = norm_prof[mask].astype(np.float32)
                mus_array[ptr:ptr+n_ok]   = np.broadcast_to(mus, (n_ok, n_mu_fine)).astype(np.float32)
                meta_array[ptr:ptr+n_ok]  = np.column_stack([
                    np.full(n_ok, i, dtype=np.int32),
                    np.full(n_ok, j, dtype=np.int32),
                    np.full(n_ok, k, dtype=np.int32),
                    iw_indices.astype(np.int32),
                ])
                ptr += n_ok

    assert ptr == n_valid

    # ── 3. Fit ALL valid profiles with 4th-order NLLD ─────────────────────────

    print(f'[{model}] Fitting {n_valid:,} profiles with 4th-order NLLD ...')
    all_coeffs = np.zeros((n_valid, 4), dtype=np.float64)
    fit_rng    = np.random.default_rng(fit_init_seed)

    for idx in tqdm(range(n_valid), desc=f'[{model}] NLLD fit'):
        prof_idx = int_profile[idx]
        mus_idx  = mus_array[idx]
        if np.all(np.abs(prof_idx) < 1e-10):
            all_coeffs[idx] = np.nan
            continue
        params = Parameters()
        for ip in range(4):
            params.add(f'c{ip+1}', value=fit_rng.uniform(0, 1))
        try:
            result          = minimize(residual_fn, params, args=(mus_idx, prof_idx))
            all_coeffs[idx] = [result.params[f'c{ic+1}'].value for ic in range(4)]
        except Exception:
            all_coeffs[idx] = np.nan

    valid_mask  = ~np.any(np.isnan(all_coeffs), axis=1)
    corner_data = all_coeffs[valid_mask]
    corner_meta = meta_array[valid_mask]
    print(f'[{model}] Valid LDC fits: {corner_data.shape[0]:,}')

    # ── 4. Save ───────────────────────────────────────────────────────────────

    np.savez(
        save_data_path + 'results.npz',
        corner_data = corner_data,
        corner_meta = corner_meta,
        T_vals_arr  = T_vals_arr,
        g_vals_arr  = g_vals_arr,
        m_vals_arr  = m_vals_arr,
        wavs_ref    = wavs_ref,
    )
    print(f'[{model}] Saved results.npz → {save_data_path}')
