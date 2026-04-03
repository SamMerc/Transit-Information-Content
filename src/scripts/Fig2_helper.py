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
from lmfit import minimize, Parameters
import gc
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.interpolate import interp1d
import corner
from tqdm import tqdm
from scipy.cluster.hierarchy import (linkage, fcluster,
                                        dendrogram as scipy_dendrogram,
                                        optimal_leaf_ordering)
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

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

N_star = 5

N_chords = 100 

N_bs_ps = 5
bs = jnp.linspace(0, 1, N_bs_ps)
ps = jnp.logspace(-3, -1, N_bs_ps)

#Number of mu values to interpolate to - set to EJ16 value
n_mu_fine = 1000  # much finer than the native 24 points # run 1x, 25x, 50x, 75x, 100x
#Percetnage of accepted profiles : 
# 1x : 94.9%
# 25x : 66%
# 50x : 63.8%
# 75x : 68.9%
# 100x : 59.3%

n_components = 4
cmap=plt.cm.coolwarm
colors = cmap(np.linspace(0, 1, n_components))

n_clusters = 5

wav_region = [6000, 53000] #0.6 - 5.3 micron

intr_prof_mode = 'load' # 'build' or 'load'
PCA_mode = 'build'
All_Corner = 'build'

excluded_bp_pairs = [
    # (N_bs_ps - 1, 0),   # b=1 (grazing), p=0.001 (smallest planet)
    # (N_bs_ps - 1, 1),   # b=1 (grazing), p=0.001 (smallest planet)
    # (N_bs_ps - 1, 2),   # b=1 (grazing), p=0.001 (smallest planet)
    # (N_bs_ps - 1, 3),   # b=1 (grazing), p=0.001 (smallest planet)
]

# ── Profile subsampling ──────────────────────────────────────────────────────
# If True, randomly draw n_subsample_profiles from each b's valid profiles
# before running PCA, clustering, fitting, etc.
# Set to False to use all valid profiles.
subsample_profiles = True
n_subsample_profiles = 10000
subsample_seed = 42  # for reproducibility

plot_dendogram = False

############################
###### Function block ######
############################
def hierarchical_clustering(
    data,
    label,
    save_path,
    feature_labels  = None,
    cutoff          = None,
    method          = 'complete',
    max_display     = 60,
    n_subsample     = 30_000,
    external_labels = None,
    clustering_metric = 'euclidean',
):
    """
    Hierarchical clustering with scipy pairwise distance computation.

    Parameters
    ----------
    data            : np.ndarray (N, D)
    label           : str
    save_path       : str
    feature_labels  : list of str, optional
    cutoff          : float or None
    method          : str            linkage method, default 'complete', Can use 'single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward'
    max_display     : int            max dendrogram leaves shown
    n_subsample     : int            scatter plot cap per cluster
    external_labels : np.ndarray or None   pre-computed labels (skips clustering)
    clustering_metric: str           distance metric for pdist, default 'euclidean', can use any metric supported by scipy's cdist ('braycurtis', 'canberra', 
                                    'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon', 'mahalanobis',
                                    'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalsneath', 'sqeuclidean', 'yule')

    Returns
    -------
    labels      : np.ndarray (N,)   1-indexed cluster labels
    cutoff_used : float
    Z           : np.ndarray (N-1, 3) or None
    """

    N, D = data.shape
    if feature_labels is None:
        feature_labels = [f'Feature {d}' for d in range(D)]

    # ─────────────────────────────────────────────────────────────────────────
    # External-labels path — skip clustering entirely, go straight to plotting
    # ─────────────────────────────────────────────────────────────────────────
    if external_labels is not None:
        valid_ext  = external_labels >= 0
        data       = data[valid_ext]
        labels     = (external_labels[valid_ext] + 1).astype(int)
        unique_cl  = np.unique(labels)
        n_cls      = len(unique_cl)
        cutoff_used = np.nan
        Z           = None
        print(f'  [{label}] External labels: {n_cls} modes, '
              f'{valid_ext.sum():,} valid profiles')
        
        # ── 6. Colour palette (shared by both paths) ──────────────────────────────
        cls_colors = plt.cm.tab10(np.linspace(0, 1, min(n_cls, 10)))
        if n_cls > 10:
            cls_colors = plt.cm.hsv(np.linspace(0, 0.9, n_cls))

    else:
        # ── 1. Standardise ────────────────────────────────────────────────────
        scaler      = StandardScaler()
        data_scaled = scaler.fit_transform(data).astype(np.float32)

        # ── 2. Scipy pdist with chunked tqdm progress ─────────────────────────
        # scipy's pdist is a single C call so we can't track it mid-flight.
        # Instead we split the row indices into chunks and call pdist on each
        # block-row, accumulating the condensed vector ourselves.  This gives
        # a meaningful progress bar while keeping all the speed of scipy.
        
        n_pairs  = N * (N - 1) // 2
        dist_vec = np.empty(n_pairs, dtype=np.float32)

        # Choose a chunk size that gives ~50 bar updates regardless of N
        CHUNK = max(1, N // 50)

        cdist_kwargs = {}
        if clustering_metric == 'mahalanobis':
            cov = np.cov(data_scaled, rowvar=False)          # (D, D)
            VI  = np.linalg.inv(cov).astype(np.float64)      # (D, D)
            cdist_kwargs['VI'] = VI

        ptr = 0
        print(f'  [{label}] Computing {N:,}×{N:,} distance matrix '
              f'({n_pairs:,} pairs) with scipy cdist in row-chunks of {CHUNK} ...')

        with tqdm(
            total        = N,
            desc         = f'  [{label}] pdist',
            unit         = ' rows',
            dynamic_ncols= True,
            bar_format   = ('{l_bar}{bar}| {n_fmt}/{total_fmt} rows '
                            '[{elapsed}<{remaining}, {rate_fmt}]'),
        ) as pbar:
            for chunk_start in range(0, N, CHUNK):
                chunk_end   = min(chunk_start + CHUNK, N)
                query_rows  = data_scaled[chunk_start:chunk_end]   # (C, D)

                # Only compute distances to columns strictly right of the diagonal
                # to fill the condensed vector in the correct scipy order.
                for local_i, global_i in enumerate(range(chunk_start, chunk_end)):
                    if global_i == N - 1:
                        break   # last row has no right-of-diagonal entries
                    right_cols  = data_scaled[global_i + 1:]       # (N-global_i-1, D)
                    row_dists   = cdist(
                        query_rows[local_i : local_i + 1],
                        right_cols,
                        metric = clustering_metric,
                        **cdist_kwargs,
                    )[0]
                    n_vals      = len(row_dists)
                    if clustering_metric == 'cityblock':
                        dist_vec[ptr : ptr + n_vals] = row_dists / D
                    else:
                        dist_vec[ptr : ptr + n_vals] = row_dists
                    ptr        += n_vals

                pbar.update(chunk_end - chunk_start)
                pbar.set_postfix(
                    pairs_filled = f'{ptr:,}/{n_pairs:,}',
                    pct          = f'{100 * ptr / n_pairs:.1f}%',
                    refresh      = False,
                )

        assert ptr == n_pairs, f'Distance vector incomplete: {ptr} / {n_pairs}'

        # ── 3. Linkage tree ───────────────────────────────────────────────────
        print(f'  [{label}] Building linkage tree ...')
        with tqdm(total=1, desc=f'  [{label}] linkage',
                  bar_format='{l_bar}{bar}| {elapsed}') as pbar:
            Z = linkage(dist_vec.astype(np.float64), method=method)
            pbar.update(1)

        # ── 4. Auto-select cutoff ─────────────────────────────────────────────
        if cutoff is None:
            merge_dists = Z[:, 2]
            gaps        = np.diff(merge_dists)
            elbow       = np.argmax(gaps)
            cutoff      = float((merge_dists[elbow] + merge_dists[elbow + 1]) / 2)
            print(f'  [{label}] Auto cutoff : {cutoff:.4f}  '
                  f'(gap {merge_dists[elbow]:.4f} → {merge_dists[elbow+1]:.4f})')
        else:
            print(f'  [{label}] Using cutoff: {cutoff:.4f}')
        cutoff_used = cutoff

        # ── 5. Cut tree → labels ──────────────────────────────────────────────
        labels     = fcluster(Z, t=cutoff, criterion='distance')
        unique_cl  = np.unique(labels)
        n_cls      = len(unique_cl)
        print(f'  [{label}] {n_cls} clusters found')
        for cl in unique_cl:
            print(f'    Cluster {cl}: {np.sum(labels == cl):6d} profiles')

        # ── 6. Colour palette (shared by both paths) ──────────────────────────────
        cls_colors = plt.cm.tab10(np.linspace(0, 1, min(n_cls, 10)))
        if n_cls > 10:
            cls_colors = plt.cm.hsv(np.linspace(0, 0.9, n_cls))

        # ── 7. Dendrogram ─────────────────────────────────────────────────────
        if plot_dendogram:
            print(f'  [{label}] Plotting dendrogram ...')
            fig_d, axes_d = plt.subplots(1, 2, figsize=(16, 6),
                                        gridspec_kw={'width_ratios': [3, 1]})
            fig_d.suptitle(f'Hierarchical clustering — {label}', fontsize=12)

            Z_ol = optimal_leaf_ordering(Z, dist_vec.astype(np.float64))
            scipy_dendrogram(
                Z_ol,
                ax                    = axes_d[0],
                truncate_mode         = 'lastp',
                p                     = max_display,
                color_threshold       = cutoff,
                above_threshold_color = 'gray',
                no_labels             = True,
            )
            axes_d[0].axhline(cutoff, color='red', linestyle='--',
                            linewidth=1.5, label=f'cutoff = {cutoff:.3f}')
            axes_d[0].set_xlabel('Profiles (truncated to last merges)')
            axes_d[0].set_ylabel('Merge distance')
            axes_d[0].set_title('Dendrogram')
            axes_d[0].legend(fontsize=8)
            axes_d[0].grid(True, alpha=0.2)

            counts = [np.sum(labels == cl) for cl in unique_cl]
            axes_d[1].barh(unique_cl, counts, color=cls_colors[:n_cls],
                        edgecolor='k', linewidth=0.5)
            axes_d[1].set_xlabel('Profiles per cluster')
            axes_d[1].set_ylabel('Cluster label')
            axes_d[1].set_title('Cluster sizes')
            axes_d[1].set_yticks(unique_cl)
            axes_d[1].grid(True, alpha=0.2, axis='x')

            fig_d.tight_layout()
            path_d = os.path.join(save_path, f'HC_Dendrogram_{label}.pdf')
            fig_d.savefig(path_d, dpi=150, bbox_inches='tight')
            plt.close(fig_d)
            print(f'  [{label}] Saved dendrogram → {path_d}')

    # ── 8. Corner scatter (always produced) ───────────────────────────────────
    print(f'  [{label}] Plotting corner scatter ...')
    fig_c, axes_c = plt.subplots(D, D, figsize=(3 * D, 3 * D))
    if D == 1:
        axes_c = np.array([[axes_c]])
    elif D > 1:
        axes_c = np.atleast_2d(axes_c)
    fig_c.suptitle(f'Corner plot — {label}', fontsize=12, y=1.01)

    for row in range(D):
        for col in range(D):
            ax = axes_c[row, col]
            if row == col:
                for ci, cl in enumerate(unique_cl):
                    mask = labels == cl
                    ax.hist(data[mask, row], bins=40, alpha=0.55,
                            color=cls_colors[ci % len(cls_colors)],
                            density=True, histtype='stepfilled', edgecolor='none')
                ax.set_xlabel(feature_labels[row], fontsize=8)
                for spine in ['top', 'left', 'right']:
                    ax.spines[spine].set_visible(False)
                ax.set_yticks([])
                ax.tick_params(labelsize=7)
            elif row > col:
                step = max(1, N // n_subsample)
                for ci, cl in enumerate(unique_cl):
                    mask = labels == cl
                    idx  = np.where(mask)[0][::step]
                    ax.scatter(data[idx, col], data[idx, row],
                               color=cls_colors[ci % len(cls_colors)],
                               s=4, alpha=0.3, linewidths=0, rasterized=True)
                if col == 0:
                    ax.set_ylabel(feature_labels[row], fontsize=8)
                if row == D - 1:
                    ax.set_xlabel(feature_labels[col], fontsize=8)
                ax.tick_params(labelsize=6)
                ax.grid(True, alpha=0.15)
            else:
                ax.set_visible(False)

    handles_c = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=cls_colors[ci % len(cls_colors)],
                   markersize=7, label=f'Cluster {cl}')
        for ci, cl in enumerate(unique_cl)
    ]
    fig_c.legend(handles=handles_c, loc='upper right', fontsize=8,
                 framealpha=0.85, title='HC cluster', title_fontsize=8)
    fig_c.tight_layout()
    path_c = os.path.join(save_path, f'HC_Corner_{label}.pdf')
    fig_c.savefig(path_c, dpi=150, bbox_inches='tight')
    plt.close(fig_c)
    print(f'  [{label}] Saved corner plot → {path_c}')

    return labels, cutoff_used, Z

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
        # Optional subsampling — draw a fixed number of profiles per b
        # This reduces memory and compute for PCA, clustering, and fitting.
        # ─────────────────────────────────────────────────────────────────────────────
        if subsample_profiles:
            rng = np.random.default_rng(subsample_seed)
            print(f'\nSUBSAMPLING: drawing {n_subsample_profiles} profiles per b '
                  f'(seed={subsample_seed})')

            for ib in range(N_bs_ps):
                n_avail = n_valid_per_b[ib]
                n_draw  = min(n_subsample_profiles, n_avail)

                if n_draw < n_avail:
                    idx = np.sort(rng.choice(n_avail, size=n_draw, replace=False))
                    print(f'  b[{ib}]={float(bs[ib]):.3f}: {n_avail} → {n_draw} profiles')
                else:
                    idx = np.arange(n_avail)
                    print(f'  b[{ib}]={float(bs[ib]):.3f}: {n_avail} profiles '
                          f'(fewer than {n_subsample_profiles}, keeping all)')

                pca_int_profile[ib] = pca_int_profile[ib][idx]
                xs_per_b[ib]        = xs_per_b[ib][idx]
                group_ids[ib]       = group_ids[ib][idx]
                meta_per_b[ib]      = meta_per_b[ib][idx]
                n_valid_per_b[ib]   = n_draw

        # ─────────────────────────────────────────────────────────────────────────────
        # Plot showing dependence of intensity profiles on other parameters
        # ─────────────────────────────────────────────────────────────────────────────
        print('RESIDUAL PLOTTING')
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
                fig_g.savefig(save_data_path + f'Grazing_profiles_coloured_{model}.pdf',
                              dpi=150, bbox_inches='tight')
                plt.close(fig_g)

        # ─────────────────────────────────────────────────────────────────────────────
        # PCA — one per b value
        # ─────────────────────────────────────────────────────────────────────────────
        print('PCA ANALYSIS')
        pcas         = []
        profiles_pca = []   # PCA scores per b: list of (n_valid_ib, n_components)
                
        for ib in range(N_bs_ps):
            print(f'  Fitting PCA for b[{ib}]={float(bs[ib]):.3f} '
                f'on {n_valid_per_b[ib]} profiles ...')
            pca_b = PCA(n_components=n_components)
            scores_b = pca_b.fit_transform(pca_int_profile[ib])
            pcas.append(pca_b)
            profiles_pca.append(scores_b)
            print(f'    Variance captured: {np.sum(pca_b.explained_variance_ratio_)*100:.1f}%')

        # ─────────────────────────────────────────────────────────────────────────────
        # Clustering on combined PCA space — hierarchical (supervisor's method)
        # ─────────────────────────────────────────────────────────────────────────────
        print('CLUSTERING')

        # Cluster independently per b value
        cluster_labels_per_b = []
        typical_idx_per_b = []
        outlier_indices_per_b = []

        for ib in range(N_bs_ps):
            print(f'  Clustering b[{ib}]={float(bs[ib]):.3f} on {n_valid_per_b[ib]} profiles ...')

            hc_labels_ib, hc_cutoff_ib, Z_ib = hierarchical_clustering(
                data           = profiles_pca[ib],
                label          = f'PCA_b{ib}_{model}',
                save_path      = save_data_path,
                feature_labels = [f'PC{k+1}' for k in range(n_components)],
                clustering_metric = 'mahalanobis',
                method= 'single',
                cutoff = [0.8, 0.8, 0.9, 0.865, 1.6][ib]
            )

            # Convert to 0-indexed
            cl_labels_ib = hc_labels_ib - 1
            unique_cl_ib = np.unique(cl_labels_ib)
            cluster_labels_per_b.append(cl_labels_ib)

            # Find typical profile: closest to its cluster's centroid (overall closest)
            typical_idx_ib = None
            min_dist = np.inf
            for cl in unique_cl_ib:
                mask = cl_labels_ib == cl
                members = profiles_pca[ib][mask]
                centroid = members.mean(axis=0)
                dists = np.linalg.norm(members - centroid, axis=1)
                closest = int(np.where(mask)[0][np.argmin(dists)])
                if dists.min() < min_dist:
                    min_dist = dists.min()
                    typical_idx_ib = closest
            typical_idx_per_b.append(typical_idx_ib)

            # Find outlier profiles: furthest from centroid in each cluster
            outlier_idx_ib = []
            for cl in unique_cl_ib:
                mask = cl_labels_ib == cl
                members = profiles_pca[ib][mask]
                centroid = members.mean(axis=0)
                dists = np.linalg.norm(members - centroid, axis=1)
                outlier_idx_ib.append(int(np.where(mask)[0][np.argmax(dists)]))
            outlier_indices_per_b.append(outlier_idx_ib)

        # For downstream code, define n_clusters from the first b value
        # (each b may have a different number of clusters)
        n_clusters_per_b = [len(np.unique(cl)) for cl in cluster_labels_per_b]
        
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
        # Figure 2a: Corner-style PCA scatter across all PC pairs, coloured by cluster
        # One figure per b value, grid of (n_components x n_components) panels
        # ─────────────────────────────────────────────────────────────────────────────
        print('    FIGURE 2a - PCA corner scatter')

        for ib in range(N_bs_ps):
            scores_ib = profiles_pca[ib]  # (n_valid_per_b[ib], n_components)
            cl_labels_ib = cluster_labels_per_b[ib]
            unique_cl_ib = np.unique(cl_labels_ib)
            n_cl_ib = len(unique_cl_ib)

            cluster_cmap_ib = plt.cm.get_cmap('tab10', n_cl_ib)
            cluster_colors_ib = [cluster_cmap_ib(c) for c in range(n_cl_ib)]

            # Special profiles for this b
            special_indices_ib = [typical_idx_per_b[ib]] + outlier_indices_per_b[ib]
            special_labels_ib  = ['Typical'] + [f'Outlier {c}' for c in range(len(outlier_indices_per_b[ib]))]
            special_colors_ib  = ['blue'] + ['red'] * len(outlier_indices_per_b[ib])

            fig2a, axes2a = plt.subplots(
                n_components, n_components,
                figsize=(3 * n_components, 3 * n_components)
            )
            fig2a.suptitle(
                f'PCA cluster structure — b = {float(bs[ib]):.3f}',
                fontsize=12, y=1.01
            )

            for row in range(n_components):
                for col in range(n_components):
                    ax = axes2a[row, col]

                    if row == col:
                        for ci, cl in enumerate(unique_cl_ib):
                            cmask = cl_labels_ib == cl
                            ax.hist(
                                scores_ib[cmask, row],
                                bins=30, alpha=0.5,
                                color=cluster_colors_ib[ci],
                                label=f'C{cl}',
                                density=True,
                                linewidth=0.8,
                                edgecolor='none',
                            )
                        ax.set_xlabel(f'PC{row+1}', fontsize=8)
                        ax.tick_params(labelsize=7)
                        for spine in ['top', 'left', 'right']:
                            ax.spines[spine].set_visible(False)
                        ax.set_yticks([])

                    elif row > col:
                        for ci, cl in enumerate(unique_cl_ib):
                            cmask = cl_labels_ib == cl
                            ax.scatter(
                                scores_ib[cmask, col],
                                scores_ib[cmask, row],
                                color=cluster_colors_ib[ci],
                                s=6, alpha=0.35,
                                linewidths=0,
                                label=f'C{cl}',
                                rasterized=True,
                            )
                        for cidx, slabel, scol in zip(
                                special_indices_ib, special_labels_ib, special_colors_ib):
                            ax.scatter(
                                scores_ib[cidx, col],
                                scores_ib[cidx, row],
                                color=scol,
                                s=60, zorder=10,
                                marker='*' if slabel == 'Typical' else 'D',
                                edgecolors='k', linewidths=0.5,
                                label=slabel,
                            )
                        if col == 0:
                            ax.set_ylabel(f'PC{row+1}', fontsize=8)
                        if row == n_components - 1:
                            ax.set_xlabel(f'PC{col+1}', fontsize=8)
                        ax.tick_params(labelsize=6)
                        ax.grid(True, alpha=0.2)

                    else:
                        ax.set_visible(False)

            last_scatter_ax = axes2a[n_components - 1, n_components - 2]
            handles, lbls = last_scatter_ax.get_legend_handles_labels()
            seen = {}
            for h, l in zip(handles, lbls):
                if l not in seen:
                    seen[l] = h
            fig2a.legend(
                seen.values(), seen.keys(),
                loc='upper right',
                fontsize=8,
                markerscale=1.5,
                framealpha=0.8,
            )

            fig2a.tight_layout()
            fig2a.savefig(
                save_data_path + f'PCA_Corner_Scatter_{model}_b{ib}.png',
                dpi=150, bbox_inches='tight'
            )
            plt.close(fig2a)
            print(f'      Saved PCA corner scatter for b[{ib}]={float(bs[ib]):.3f}')

        # ─────────────────────────────────────────────────────────────────────────────
        # Figure 2b: typical and outlier profiles
        # Rows = b values, columns = special profiles (typical + outliers)
        # ─────────────────────────────────────────────────────────────────────────────
        print('    FIGURE 2b - Mode and outlier profiles')

        # Use maximum number of outliers across all b for column count
        max_n_outliers = max(len(ol) for ol in outlier_indices_per_b)
        n_specials = 1 + max_n_outliers

        fig2b, axes2b = plt.subplots(N_bs_ps, n_specials,
                                     figsize=(5 * n_specials, 4 * N_bs_ps),
                                     sharey='row')
        if N_bs_ps == 1:
            axes2b = axes2b[np.newaxis, :]
        if n_specials == 1:
            axes2b = axes2b[:, np.newaxis]

        for ib in range(N_bs_ps):
            special_indices_ib = [typical_idx_per_b[ib]] + outlier_indices_per_b[ib]
            special_labels_ib  = ['Typical'] + [f'Outlier {c}' for c in range(len(outlier_indices_per_b[ib]))]
            special_colors_ib  = ['blue'] + ['red'] * len(outlier_indices_per_b[ib])

            for col, (cidx, slabel, scol) in enumerate(
                    zip(special_indices_ib, special_labels_ib, special_colors_ib)):
                ax = axes2b[ib, col]
                n_v = n_valid_per_b[ib]
                for nval in range(0, n_v, max(1, n_v // 200)):
                    ax.plot(xs_per_b[ib][nval], pca_int_profile[ib][nval],
                            alpha=0.15, color='gray', linewidth=0.3)
                ax.plot(xs_per_b[ib][cidx], pca_int_profile[ib][cidx],
                        color=scol, linewidth=2, label=slabel, zorder=10)
                ax.set_xlabel('$r/R_\\star$')
                ax.set_ylabel('Norm. Intensity')
                ax.set_title(f'{slabel}  b={float(bs[ib]):.3f}')
                ax.legend(fontsize=7)
                ax.grid(True, alpha=0.3)

            # Hide unused columns if this b has fewer outliers
            for col in range(len(special_indices_ib), n_specials):
                axes2b[ib, col].set_visible(False)

        fig2b.tight_layout()
        fig2b.savefig(save_data_path + 'Mode_and_Outliers.png', dpi=150, bbox_inches='tight')
        plt.close(fig2b)

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
            row_in_b = typical_idx_per_b[ib]
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
                f"{np.sum(pcas[ib].explained_variance_ratio_)*100:.1f}% variance in {n_components} PCs, "
                f"{n_valid_per_b[ib]} valid profiles")

        # Save one profile vector per b for typical and outlier profiles
        for ib in range(N_bs_ps):
            row_typical = typical_idx_per_b[ib]
            np.save(save_data_path + f'mode_intensity_profile_{model}_b{ib}.npy',
                    pca_int_profile[ib][row_typical])
            np.save(save_data_path + f'mode_rs_{model}_b{ib}.npy',
                    xs_per_b[ib][row_typical])
            for i_save, cidx in enumerate(outlier_indices_per_b[ib]):
                np.save(save_data_path + f'outlier{i_save+1}_intensity_profile_{model}_b{ib}.npy',
                        pca_int_profile[ib][cidx])
                np.save(save_data_path + f'outlier{i_save+1}_rs_{model}_b{ib}.npy',
                        xs_per_b[ib][cidx])

        # Build the flat lists needed for Figure 4
        typical_profile = [pca_int_profile[ib][typical_idx_per_b[ib]] for ib in range(N_bs_ps)]
        typical_xs      = [xs_per_b[ib][typical_idx_per_b[ib]] for ib in range(N_bs_ps)]
        outlier_profiles = []
        outlier_xs       = []
        # Use max number of outliers; pad with None for b values with fewer clusters
        max_n_outliers = max(len(ol) for ol in outlier_indices_per_b)
        for i_out in range(max_n_outliers):
            prof_list = []
            xs_list   = []
            for ib in range(N_bs_ps):
                if i_out < len(outlier_indices_per_b[ib]):
                    cidx = outlier_indices_per_b[ib][i_out]
                    prof_list.append(pca_int_profile[ib][cidx])
                    xs_list.append(xs_per_b[ib][cidx])
                else:
                    prof_list.append(None)
                    xs_list.append(None)
            outlier_profiles.append(prof_list)
            outlier_xs.append(xs_list)
        print(f"Saved profiles to {save_data_path}")

        # Save metadata arrays for each b (needed for coloured corner plots)
        for ib in range(N_bs_ps):
            np.save(save_data_path + f'meta_{model}_b{ib}.npy', meta_per_b[ib])

    elif PCA_mode == 'load':
        b_colors  = plt.cm.plasma(np.linspace(0.1, 0.9, N_bs_ps))

        typical_profile = [np.load(save_data_path + f'mode_intensity_profile_{model}_b{ib}.npy')
                           for ib in range(N_bs_ps)]
        typical_xs      = [np.load(save_data_path + f'mode_rs_{model}_b{ib}.npy')
                           for ib in range(N_bs_ps)]
        
        # Load metadata arrays
        meta_per_b = [np.load(save_data_path + f'meta_{model}_b{ib}.npy')
                      for ib in range(N_bs_ps)]

        # Discover how many outlier files exist per b
        outlier_profiles = []
        outlier_xs       = []
        i_save = 0
        while True:
            path_check = save_data_path + f'outlier{i_save+1}_intensity_profile_{model}_b0.npy'
            if not os.path.exists(path_check):
                break
            prof_list = []
            xs_list   = []
            for ib in range(N_bs_ps):
                p_path = save_data_path + f'outlier{i_save+1}_intensity_profile_{model}_b{ib}.npy'
                x_path = save_data_path + f'outlier{i_save+1}_rs_{model}_b{ib}.npy'
                if os.path.exists(p_path):
                    prof_list.append(np.load(p_path))
                    xs_list.append(np.load(x_path))
                else:
                    prof_list.append(None)
                    xs_list.append(None)
            outlier_profiles.append(prof_list)
            outlier_xs.append(xs_list)
            i_save += 1
        n_clusters = len(outlier_profiles) if outlier_profiles else 1
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
            for i in range(len(outlier_profiles))]
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
            
            # Skip if this b has no profile for this outlier
            if rps_per_b[ib] is None or prof_per_b[ib] is None:
                axes4[ib, 0].set_visible(False)
                axes4[ib, 1].set_visible(False)
                continue

            mus_ib  = np.array(rps_per_b[ib])
            prof_ib = np.array(prof_per_b[ib])
            
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

    # ─────────────────────────────────────────────────────────────────────────────
    # Figure 5: Fit ALL profiles with 4th-order NLLD and make corner plots
    # One corner plot per b value
    # ─────────────────────────────────────────────────────────────────────────────
    print('Fitting ALL profiles with 4th-order NLLD','\n')

    # For each b value, fit every valid profile and collect coefficients
    all_coeffs_per_b = []   # list of (n_profs, 4) arrays

    if All_Corner == 'build':
        for ib in tqdm(range(N_bs_ps)):
            n_profs = pca_int_profile[ib].shape[0]
            coeffs_ib = np.zeros((n_profs, 4), dtype=np.float64)
            print(f'  Fitting {n_profs} profiles for b[{ib}]={float(bs[ib]):.3f} ...')

            for idx in tqdm(range(n_profs)):
                mus_idx  = xs_per_b[ib][idx]
                prof_idx = pca_int_profile[ib][idx]

                # Skip profiles that are degenerate / all-zero
                if np.all(np.abs(prof_idx) < 1e-10):
                    coeffs_ib[idx] = np.nan
                    continue

                params = Parameters()
                for ip in range(4):
                    params.add(f'c{ip+1}', value=np.random.uniform(0, 1))

                try:
                    result = minimize(residual_fn, params, args=(mus_idx, prof_idx))
                    coeffs_ib[idx] = [result.params[f'c{ic+1}'].value for ic in range(4)]
                except Exception:
                    coeffs_ib[idx] = np.nan

            all_coeffs_per_b.append(coeffs_ib)
            np.save(save_data_path + f'all_coeffs_{model}_b{ib}.npy', coeffs_ib)

    # Load intensity profiles grid
    elif All_Corner == 'load':
        for ib in tqdm(range(N_bs_ps)):
            coeffs_ib = np.load(save_data_path + f'all_coeffs_{model}_b{ib}.npy')
            all_coeffs_per_b.append(coeffs_ib)

    else:
        raise KeyboardInterrupt('Wrong corner plot data mode')

    print('  Done fitting. Now making corner plots...')

    # ── Corner plots ──────────────────────────────────────────────────────────────
    # Stack all b-value arrays and append the impact parameter as a 5th column
    # so that corner_data has shape (N_total, 5): [c1, c2, c3, c4, b]
    print('    FIGURE 5a - All coefficients with density corner plot')

    corner_pieces = []
    for ib in range(N_bs_ps):
        n_profs = all_coeffs_per_b[ib].shape[0]
        b_col   = np.full((n_profs, 1), float(bs[ib]))
        corner_pieces.append(np.hstack([all_coeffs_per_b[ib], b_col]))

    corner_data = np.vstack(corner_pieces)                 # shape (N_total, 5)

    # FIX 2: drop rows where ANY coefficient is NaN (failed fits / degenerate profiles)
    valid_mask  = ~np.any(np.isnan(corner_data), axis=1)
    corner_data = corner_data[valid_mask]

    print(f'  Corner plot for {corner_data.shape[0]} valid profiles')

    # columns are c1, c2, c3, c4, b
    labels = [r'$c_1$', r'$c_2$', r'$c_3$', r'$c_4$', r'$b$']

    # FIX 3: use corner_data (numpy array) for percentile ranges
    ranges = [
        (np.percentile(corner_data[:, ic], 1),
        np.percentile(corner_data[:, ic], 99))
        for ic in range(5)
    ]
    
    # Create the corner plot
    fig_corner = corner.corner(
        corner_data,
        labels=labels,
        range=ranges,
        bins=50,
        smooth=1.0,
        smooth1d=1.0,
        plot_datapoints=True,
        plot_density=False,
        fill_contours=False,
        levels=(0.5, 0.68, 0.95, 0.99),
        hist_kwargs={'color': 'black', 'linewidth': 1.5},
        label_kwargs={'fontsize': 13},
        title_kwargs={'fontsize': 11},
        show_titles=False,
        data_kwargs={'alpha': 0.2, 'ms': 1.5, 'color': 'black'},
        contourf_kwargs={'alpha': 0.5},
        contour_kwargs={'colors': ['brown', 'red', 'orange', 'yellow']},
    )

    # Remove spines on diagonal histograms
    ndim = corner_data.shape[1]
    for i in range(ndim):
        ax = fig_corner.axes[i * ndim + i]
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # FIX 5: merge f-strings into one (was two separate string literals with no separator)
    fig_corner.suptitle(
        f'4th-order NLLD coefficients + Impact parameter',
        fontsize=13, y=1.02
    )

    fig_corner.savefig(
        save_data_path + f'Corner_NLLD_{model}.png',
        dpi=150, bbox_inches='tight'
    )
    plt.close(fig_corner)
    print(f'    Saved corner plot')



    # ─────────────────────────────────────────────────────────────────────────────
    # Figure 5b: 5D corner plot coloured by impact parameter
    # ─────────────────────────────────────────────────────────────────────────────
    print('    FIGURE 5b - Corner plot coloured by impact parameter')

    # The impact parameter lives in the 5th column (index 4) of corner_data
    b_values = corner_data[:, 4]

    # Build a colour array: one RGBA per profile, mapped from b
    cmap_b  = plt.cm.viridis
    norm_b  = mcolors.Normalize(vmin=b_values.min(), vmax=b_values.max())
    point_colors = cmap_b(norm_b(b_values))  # (N_valid, 4) RGBA

    # Percentile ranges (same as before)
    ranges_5d = [
        (np.percentile(corner_data[:, ic], 1),
         np.percentile(corner_data[:, ic], 99))
        for ic in range(5)
    ]

    labels_5d = [r'$c_1$', r'$c_2$', r'$c_3$', r'$c_4$', r'$b$']

    # ── First pass: create the corner figure with histograms only ─────────────
    #    We suppress the default scatter so we can draw our own coloured version.
    fig_corner_b = corner.corner(
        corner_data,
        labels=labels_5d,
        range=ranges_5d,
        bins=50,
        smooth1d=1.0,
        plot_datapoints=False,       # we will add our own coloured scatter
        plot_density=False,
        fill_contours=False,
        no_fill_contours=True,
        levels=(0.5, 0.68, 0.95, 0.99),
        hist_kwargs={'color': 'gray', 'linewidth': 1.2, 'alpha': 0.5},
        label_kwargs={'fontsize': 13},
        title_kwargs={'fontsize': 11},
        show_titles=False,
        contour_kwargs={'colors': 'none'},   # hide contour lines
    )

    ndim_5d = corner_data.shape[1]
    axes_cb = np.array(fig_corner_b.axes).reshape(ndim_5d, ndim_5d)

    # ── Second pass: coloured scatter in lower-triangle panels ────────────────
    # Subsample for readability if the dataset is very large
    max_scatter = 40_000
    if len(corner_data) > max_scatter:
        rng_cb   = np.random.default_rng(42)
        idx_sub  = np.sort(rng_cb.choice(len(corner_data), size=max_scatter, replace=False))
    else:
        idx_sub = np.arange(len(corner_data))

    # Sort by b so that the highest-b points are drawn on top (most visible)
    order = np.argsort(b_values[idx_sub])
    idx_sub = idx_sub[order]

    for row in range(ndim_5d):
        for col in range(row):
            ax = axes_cb[row, col]
            ax.scatter(
                corner_data[idx_sub, col],
                corner_data[idx_sub, row],
                c=b_values[idx_sub],
                cmap=cmap_b,
                norm=norm_b,
                s=1.5,
                alpha=0.35,
                linewidths=0,
                rasterized=True,
            )

    # ── Coloured histograms on the diagonal ───────────────────────────────────
    # Overwrite the grey histograms with stacked per-b histograms
    unique_bs = np.unique(b_values)
    for d in range(ndim_5d):
        ax = axes_cb[d, d]
        ax.clear()
        for bval in unique_bs:
            mask_bval = b_values == bval
            ax.hist(
                corner_data[mask_bval, d],
                bins=50,
                range=ranges_5d[d],
                alpha=0.55,
                color=cmap_b(norm_b(bval)),
                density=True,
                histtype='stepfilled',
                edgecolor='none',
                label=f'b={bval:.2f}',
            )
        ax.set_xlim(ranges_5d[d])
        ax.set_yticks([])
        for spine in ['top', 'left', 'right']:
            ax.spines[spine].set_visible(False)
        if d == ndim_5d - 1:
            ax.set_xlabel(labels_5d[d], fontsize=13)

    # ── Restore axis labels that corner.corner normally sets ──────────────────
    for row in range(ndim_5d):
        for col in range(row):
            ax = axes_cb[row, col]
            if col == 0:
                ax.set_ylabel(labels_5d[row], fontsize=13)
            if row == ndim_5d - 1:
                ax.set_xlabel(labels_5d[col], fontsize=13)
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.15)

    # ── Colorbar ──────────────────────────────────────────────────────────────
    # Place the colorbar in the upper-right empty region of the corner grid
    # Create a new axes spanning roughly the top-right quarter
    cbar_ax = fig_corner_b.add_axes([0.72, 0.72, 0.025, 0.20])  # [left, bottom, width, height]
    sm = cm.ScalarMappable(cmap=cmap_b, norm=norm_b)
    sm.set_array([])
    cbar = fig_corner_b.colorbar(sm, cax=cbar_ax)
    cbar.set_label(r'Impact parameter $b$', fontsize=13)
    cbar.ax.tick_params(labelsize=10)

    # If b takes only discrete values, set the ticks explicitly
    if len(unique_bs) <= 15:
        cbar.set_ticks(unique_bs)
        cbar.set_ticklabels([f'{bv:.2f}' for bv in unique_bs])

    fig_corner_b.suptitle(
        f'4th-order NLLD coefficients coloured by impact parameter — {model}',
        fontsize=13, y=1.02,
    )

    fig_corner_b.savefig(
        save_data_path + f'Corner_NLLD_byB_{model}.png',
        dpi=150, bbox_inches='tight',
    )
    plt.close(fig_corner_b)
    print(f'    Saved b-coloured corner plot')

    # ─────────────────────────────────────────────────────────────────────────────
    # Build corner_meta — physical parameter values for every row in corner_data
    # Columns: [p, Teff, logg, metallicity]
    # ─────────────────────────────────────────────────────────────────────────────
    T_vals_arr = np.linspace(Teffs[model][0],        Teffs[model][1],        N_star)
    g_vals_arr = np.linspace(loggs[model][0],         loggs[model][1],         N_star)
    m_vals_arr = np.linspace(metallicitys[model][0],  metallicitys[model][1],  N_star)
    p_vals_arr = np.array(ps)

    meta_pieces = []
    for ib in range(N_bs_ps):
        meta_ib = meta_per_b[ib]  # (n_profs, 5): columns [ip, i, j, k, iw]
        phys_ib = np.column_stack([
            p_vals_arr[meta_ib[:, 0]],    # p
            T_vals_arr[meta_ib[:, 1]],    # Teff
            g_vals_arr[meta_ib[:, 2]],    # logg
            m_vals_arr[meta_ib[:, 3]],    # metallicity
        ])                                 # (n_profs, 4)
        meta_pieces.append(phys_ib)

    corner_meta = np.vstack(meta_pieces)   # (N_total, 4)
    corner_meta = corner_meta[valid_mask]  # same valid_mask used for corner_data

    meta_col_names  = [r'$p$', r'$T_{\rm eff}$ (K)', r'$\log\,g$', r'[M/H]']
    meta_col_keys   = ['p', 'Teff', 'logg', 'MH']
    meta_cmaps      = [plt.cm.plasma, plt.cm.inferno, plt.cm.cividis, plt.cm.coolwarm]

    # ── Wavelength value for every row in corner_data ─────────────────────────
    # meta_per_b[ib][:, 4] is the wavelength index iw into the filtered
    # wavelength grid.  All (i,j,k) combinations share the same grid for a
    # given model, so we use wavs_ref as the look-up table.
    wavs_ref = np.array(gen_dict['stellar_wavelengths'][model][0, 0, 0])  # (n_wav,)

    wav_pieces = []
    for ib in range(N_bs_ps):
        meta_ib = meta_per_b[ib]          # (n_profs, 5): [ip, i, j, k, iw]
        wav_pieces.append(wavs_ref[meta_ib[:, 4]])   # (n_profs,)

    corner_wav = np.concatenate(wav_pieces)            # (N_total,)
    corner_wav = corner_wav[valid_mask]                # same mask as corner_data

    # Convert from Angstroms to microns for a cleaner colorbar
    corner_wav_um = corner_wav / 1e4

    # ─────────────────────────────────────────────────────────────────────────────
    # Figures 5c–5f: 5D corner plots coloured by p, Teff, logg, metallicity
    # ─────────────────────────────────────────────────────────────────────────────
    for imeta in range(4):                      # loop over p, Teff, logg, [M/H]
        col_vals   = corner_meta[:, imeta]
        col_label  = meta_col_names[imeta]
        col_key    = meta_col_keys[imeta]
        col_cmap   = meta_cmaps[imeta]
        col_norm   = mcolors.Normalize(vmin=col_vals.min(), vmax=col_vals.max())

        print(f'    FIGURE 5{chr(99 + imeta)} - Corner plot coloured by {col_key}')

        # ── Base corner figure (histograms only) ─────────────────────────────
        fig_cm = corner.corner(
            corner_data,
            labels=labels_5d,
            range=ranges_5d,
            bins=50,
            smooth1d=1.0,
            plot_datapoints=False,
            plot_density=False,
            fill_contours=False,
            no_fill_contours=True,
            levels=(0.5, 0.68, 0.95, 0.99),
            hist_kwargs={'color': 'gray', 'linewidth': 1.2, 'alpha': 0.5},
            label_kwargs={'fontsize': 13},
            title_kwargs={'fontsize': 11},
            show_titles=False,
            contour_kwargs={'colors': 'none'},
        )

        axes_cm = np.array(fig_cm.axes).reshape(ndim_5d, ndim_5d)

        # ── Coloured scatter in lower-triangle panels ────────────────────────
        max_scatter = 40_000
        if len(corner_data) > max_scatter:
            rng_cm  = np.random.default_rng(42)
            idx_sub = np.sort(rng_cm.choice(len(corner_data),
                                            size=max_scatter, replace=False))
        else:
            idx_sub = np.arange(len(corner_data))

        # Sort so highest colour values render on top
        order = np.argsort(col_vals[idx_sub])
        idx_sub = idx_sub[order]

        for row in range(ndim_5d):
            for col in range(row):
                ax = axes_cm[row, col]
                ax.scatter(
                    corner_data[idx_sub, col],
                    corner_data[idx_sub, row],
                    c=col_vals[idx_sub],
                    cmap=col_cmap,
                    norm=col_norm,
                    s=1.5,
                    alpha=0.35,
                    linewidths=0,
                    rasterized=True,
                )

        # ── Coloured diagonal histograms ─────────────────────────────────────
        # Bin the colour variable into n_hist_bins groups for stacked histograms
        unique_vals = np.unique(col_vals)
        if len(unique_vals) <= 20:
            hist_bins_vals = unique_vals          # discrete values (e.g. 5 p values)
        else:
            n_hist_bins = 8
            hist_bins_vals = np.linspace(col_vals.min(), col_vals.max(),
                                         n_hist_bins + 1)
            # Use bin centres as representative values
            hist_bins_vals = 0.5 * (hist_bins_vals[:-1] + hist_bins_vals[1:])

        for d in range(ndim_5d):
            ax = axes_cm[d, d]
            ax.clear()

            if len(unique_vals) <= 20:
                # One histogram per unique value
                for uv in unique_vals:
                    mask_uv = col_vals == uv
                    ax.hist(
                        corner_data[mask_uv, d],
                        bins=50,
                        range=ranges_5d[d],
                        alpha=0.55,
                        color=col_cmap(col_norm(uv)),
                        density=True,
                        histtype='stepfilled',
                        edgecolor='none',
                    )
            else:
                # Bin into groups
                edges = np.linspace(col_vals.min(), col_vals.max(),
                                    len(hist_bins_vals) + 1)
                for ibin, centre in enumerate(hist_bins_vals):
                    mask_bin = ((col_vals >= edges[ibin]) &
                                (col_vals < edges[ibin + 1]))
                    if ibin == len(hist_bins_vals) - 1:
                        mask_bin |= (col_vals == edges[ibin + 1])
                    if mask_bin.sum() == 0:
                        continue
                    ax.hist(
                        corner_data[mask_bin, d],
                        bins=50,
                        range=ranges_5d[d],
                        alpha=0.55,
                        color=col_cmap(col_norm(centre)),
                        density=True,
                        histtype='stepfilled',
                        edgecolor='none',
                    )

            ax.set_xlim(ranges_5d[d])
            ax.set_yticks([])
            for spine in ['top', 'left', 'right']:
                ax.spines[spine].set_visible(False)
            if d == ndim_5d - 1:
                ax.set_xlabel(labels_5d[d], fontsize=13)

        # ── Restore axis labels ──────────────────────────────────────────────
        for row in range(ndim_5d):
            for col in range(row):
                ax = axes_cm[row, col]
                if col == 0:
                    ax.set_ylabel(labels_5d[row], fontsize=13)
                if row == ndim_5d - 1:
                    ax.set_xlabel(labels_5d[col], fontsize=13)
                ax.tick_params(labelsize=8)
                ax.grid(True, alpha=0.15)

        # ── Colorbar ─────────────────────────────────────────────────────────
        cbar_ax = fig_cm.add_axes([0.72, 0.72, 0.025, 0.20])
        sm_cm   = cm.ScalarMappable(cmap=col_cmap, norm=col_norm)
        sm_cm.set_array([])
        cbar_cm = fig_cm.colorbar(sm_cm, cax=cbar_ax)
        cbar_cm.set_label(col_label, fontsize=13)
        cbar_cm.ax.tick_params(labelsize=10)

        # For discrete parameters, set ticks explicitly
        if len(unique_vals) <= 15:
            cbar_cm.set_ticks(unique_vals)
            if col_key == 'Teff':
                cbar_cm.set_ticklabels([f'{v:.0f}' for v in unique_vals])
            elif col_key == 'p':
                cbar_cm.set_ticklabels([f'{v:.4f}' for v in unique_vals])
            else:
                cbar_cm.set_ticklabels([f'{v:.2f}' for v in unique_vals])

        fig_cm.suptitle(
            f'4th-order NLLD coefficients coloured by {col_label} — {model}',
            fontsize=13, y=1.02,
        )

        fig_cm.savefig(
            save_data_path + f'Corner_NLLD_by{col_key}_{model}.png',
            dpi=150, bbox_inches='tight',
        )
        plt.close(fig_cm)
        print(f'    Saved {col_key}-coloured corner plot')

    # ─────────────────────────────────────────────────────────────────────────────
    # Figure 5g: 5D corner plot coloured by wavelength
    # ─────────────────────────────────────────────────────────────────────────────
    print('    FIGURE 5g - Corner plot coloured by wavelength')

    wav_cmap  = plt.cm.turbo
    wav_norm  = mcolors.Normalize(vmin=corner_wav_um.min(),
                                  vmax=corner_wav_um.max())

    # ── Base corner figure (histograms only) ─────────────────────────────────
    fig_wav = corner.corner(
        corner_data,
        labels=labels_5d,
        range=ranges_5d,
        bins=50,
        smooth1d=1.0,
        plot_datapoints=False,
        plot_density=False,
        fill_contours=False,
        no_fill_contours=True,
        levels=(0.5, 0.68, 0.95, 0.99),
        hist_kwargs={'color': 'gray', 'linewidth': 1.2, 'alpha': 0.5},
        label_kwargs={'fontsize': 13},
        title_kwargs={'fontsize': 11},
        show_titles=False,
        contour_kwargs={'colors': 'none'},
    )

    axes_wav = np.array(fig_wav.axes).reshape(ndim_5d, ndim_5d)

    # ── Coloured scatter in lower-triangle panels ────────────────────────────
    max_scatter_wav = 40_000
    if len(corner_data) > max_scatter_wav:
        rng_wav  = np.random.default_rng(42)
        idx_sub_wav = np.sort(rng_wav.choice(len(corner_data),
                                             size=max_scatter_wav, replace=False))
    else:
        idx_sub_wav = np.arange(len(corner_data))

    # Sort so longest wavelengths render on top
    order_wav   = np.argsort(corner_wav_um[idx_sub_wav])
    idx_sub_wav = idx_sub_wav[order_wav]

    for row in range(ndim_5d):
        for col in range(row):
            ax = axes_wav[row, col]
            ax.scatter(
                corner_data[idx_sub_wav, col],
                corner_data[idx_sub_wav, row],
                c=corner_wav_um[idx_sub_wav],
                cmap=wav_cmap,
                norm=wav_norm,
                s=1.5,
                alpha=0.35,
                linewidths=0,
                rasterized=True,
            )

    # ── Coloured diagonal histograms ─────────────────────────────────────────
    # Wavelength is (quasi-)continuous, so bin into groups for the histograms
    n_wav_hist_bins = 10
    wav_edges  = np.linspace(corner_wav_um.min(), corner_wav_um.max(),
                             n_wav_hist_bins + 1)
    wav_centres = 0.5 * (wav_edges[:-1] + wav_edges[1:])

    for d in range(ndim_5d):
        ax = axes_wav[d, d]
        ax.clear()

        for ibin, centre in enumerate(wav_centres):
            mask_bin = ((corner_wav_um >= wav_edges[ibin]) &
                        (corner_wav_um < wav_edges[ibin + 1]))
            if ibin == n_wav_hist_bins - 1:          # include right edge
                mask_bin |= (corner_wav_um == wav_edges[ibin + 1])
            if mask_bin.sum() == 0:
                continue
            ax.hist(
                corner_data[mask_bin, d],
                bins=50,
                range=ranges_5d[d],
                alpha=0.55,
                color=wav_cmap(wav_norm(centre)),
                density=True,
                histtype='stepfilled',
                edgecolor='none',
            )

        ax.set_xlim(ranges_5d[d])
        ax.set_yticks([])
        for spine in ['top', 'left', 'right']:
            ax.spines[spine].set_visible(False)
        if d == ndim_5d - 1:
            ax.set_xlabel(labels_5d[d], fontsize=13)

    # ── Restore axis labels ──────────────────────────────────────────────────
    for row in range(ndim_5d):
        for col in range(row):
            ax = axes_wav[row, col]
            if col == 0:
                ax.set_ylabel(labels_5d[row], fontsize=13)
            if row == ndim_5d - 1:
                ax.set_xlabel(labels_5d[col], fontsize=13)
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.15)

    # ── Colorbar ─────────────────────────────────────────────────────────────
    cbar_ax_wav = fig_wav.add_axes([0.72, 0.72, 0.025, 0.20])
    sm_wav      = cm.ScalarMappable(cmap=wav_cmap, norm=wav_norm)
    sm_wav.set_array([])
    cbar_wav    = fig_wav.colorbar(sm_wav, cax=cbar_ax_wav)
    cbar_wav.set_label(r'Wavelength ($\mu$m)', fontsize=13)
    cbar_wav.ax.tick_params(labelsize=10)

    # Nice tick spacing: every 0.5 µm
    wav_tick_step = 0.5
    wav_ticks = np.arange(
        np.ceil(corner_wav_um.min() / wav_tick_step) * wav_tick_step,
        corner_wav_um.max() + wav_tick_step / 2,
        wav_tick_step,
    )
    cbar_wav.set_ticks(wav_ticks)
    cbar_wav.set_ticklabels([f'{t:.1f}' for t in wav_ticks])

    fig_wav.suptitle(
        f'4th-order NLLD coefficients coloured by wavelength — {model}',
        fontsize=13, y=1.02,
    )

    fig_wav.savefig(
        save_data_path + f'Corner_NLLD_byWav_{model}.png',
        dpi=150, bbox_inches='tight',
    )
    plt.close(fig_wav)
    print(f'    Saved wavelength-coloured corner plot')

    # ─────────────────────────────────────────────────────────────────────────────
    # Figure 5h: 5D coefficient corner plot coloured by PCA-space cluster labels
    # This is the reverse of the existing plot where coefficient-space modes
    # are projected onto PCA space — here PCA clusters are projected onto
    # coefficient space.
    # ─────────────────────────────────────────────────────────────────────────────
    print('    FIGURE 5h - Coefficient corner plot coloured by PCA clusters')

    # ── Back-propagate PCA cluster labels onto the stacked coefficient array ──
    # cluster_labels_per_b[ib] is 0-indexed, shape (n_profs_ib,), one per b.
    # We concatenate them in the same order that corner_data was built
    # (i.e. b0, b1, ..., b_{N_bs_ps-1}) and apply the same valid_mask.
    pca_cl_stacked = np.concatenate(cluster_labels_per_b)   # (N_total,)
    pca_cl_corner  = pca_cl_stacked[valid_mask]             # (N_valid,)

    unique_pca_cl   = np.unique(pca_cl_corner)
    n_pca_clusters  = len(unique_pca_cl)

    # Because different b values may have produced different cluster counts,
    # we offset each b's labels so they are globally unique.
    # This way cluster 0 from b=0.0 is not conflated with cluster 0 from b=0.5.
    pca_cl_global = np.empty_like(pca_cl_stacked)
    offset = 0
    for ib in range(N_bs_ps):
        n_cl_ib = len(np.unique(cluster_labels_per_b[ib]))
        start   = sum(len(c) for c in cluster_labels_per_b[:ib])
        end     = start + len(cluster_labels_per_b[ib])
        pca_cl_global[start:end] = cluster_labels_per_b[ib] + offset
        offset += n_cl_ib

    pca_cl_global_corner = pca_cl_global[valid_mask]
    unique_global_cl     = np.unique(pca_cl_global_corner)
    n_global_clusters    = len(unique_global_cl)

    print(f'    {n_global_clusters} global PCA clusters '
          f'(across {N_bs_ps} b values) mapped onto coefficient space')
    for cl in unique_global_cl:
        n_in = np.sum(pca_cl_global_corner == cl)
        print(f'      Global PCA cluster {cl}: {n_in} profiles')

    # ── Colour palette ───────────────────────────────────────────────────────
    if n_global_clusters <= 10:
        pca_cl_cmap = plt.cm.tab10
    elif n_global_clusters <= 20:
        pca_cl_cmap = plt.cm.tab20
    else:
        pca_cl_cmap = plt.cm.hsv

    pca_cl_colors = {
        cl: pca_cl_cmap(i / max(n_global_clusters - 1, 1))
        for i, cl in enumerate(unique_global_cl)
    }

    # Build an RGBA array for every valid profile
    point_colors_pca = np.array([pca_cl_colors[cl] for cl in pca_cl_global_corner])

    # ── Base corner figure ───────────────────────────────────────────────────
    fig_pca_on_coeff = corner.corner(
        corner_data,
        labels=labels_5d,
        range=ranges_5d,
        bins=50,
        smooth1d=1.0,
        plot_datapoints=False,
        plot_density=False,
        fill_contours=False,
        no_fill_contours=True,
        levels=(0.5, 0.68, 0.95, 0.99),
        hist_kwargs={'color': 'gray', 'linewidth': 1.2, 'alpha': 0.5},
        label_kwargs={'fontsize': 13},
        title_kwargs={'fontsize': 11},
        show_titles=False,
        contour_kwargs={'colors': 'none'},
    )

    axes_pc = np.array(fig_pca_on_coeff.axes).reshape(ndim_5d, ndim_5d)

    # ── Coloured scatter in lower-triangle panels ────────────────────────────
    max_scatter_pc = 40_000
    if len(corner_data) > max_scatter_pc:
        rng_pc      = np.random.default_rng(42)
        idx_sub_pc  = np.sort(rng_pc.choice(len(corner_data),
                                            size=max_scatter_pc, replace=False))
    else:
        idx_sub_pc = np.arange(len(corner_data))

    for row in range(ndim_5d):
        for col in range(row):
            ax = axes_pc[row, col]
            for cl in unique_global_cl:
                mask_cl = pca_cl_global_corner[idx_sub_pc] == cl
                if mask_cl.sum() == 0:
                    continue
                ax.scatter(
                    corner_data[idx_sub_pc[mask_cl], col],
                    corner_data[idx_sub_pc[mask_cl], row],
                    color=pca_cl_colors[cl],
                    s=1.5,
                    alpha=0.35,
                    linewidths=0,
                    rasterized=True,
                )

    # ── Coloured diagonal histograms ─────────────────────────────────────────
    for d in range(ndim_5d):
        ax = axes_pc[d, d]
        ax.clear()

        for cl in unique_global_cl:
            mask_cl = pca_cl_global_corner == cl
            if mask_cl.sum() == 0:
                continue
            ax.hist(
                corner_data[mask_cl, d],
                bins=50,
                range=ranges_5d[d],
                alpha=0.55,
                color=pca_cl_colors[cl],
                density=True,
                histtype='stepfilled',
                edgecolor='none',
            )

        ax.set_xlim(ranges_5d[d])
        ax.set_yticks([])
        for spine in ['top', 'left', 'right']:
            ax.spines[spine].set_visible(False)
        if d == ndim_5d - 1:
            ax.set_xlabel(labels_5d[d], fontsize=13)

    # ── Restore axis labels ──────────────────────────────────────────────────
    for row in range(ndim_5d):
        for col in range(row):
            ax = axes_pc[row, col]
            if col == 0:
                ax.set_ylabel(labels_5d[row], fontsize=13)
            if row == ndim_5d - 1:
                ax.set_xlabel(labels_5d[col], fontsize=13)
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.15)

    # ── Legend in upper-right empty space ─────────────────────────────────────
    # Build one entry per b value, grouping its PCA clusters together
    legend_handles = []
    cl_offset = 0
    for ib in range(N_bs_ps):
        n_cl_ib    = len(np.unique(cluster_labels_per_b[ib]))
        b_val_str  = f'{float(bs[ib]):.2f}'
        for ic in range(n_cl_ib):
            global_cl = cl_offset + ic
            legend_handles.append(
                plt.Line2D(
                    [0], [0],
                    marker='o', color='w',
                    markerfacecolor=pca_cl_colors[global_cl],
                    markersize=7,
                    label=f'b={b_val_str} PCA-cl {ic}',
                )
            )
        cl_offset += n_cl_ib

    # Place legend; use two columns if many clusters
    ncol_leg = 2 if n_global_clusters > 8 else 1
    fig_pca_on_coeff.legend(
        handles=legend_handles,
        loc='upper right',
        fontsize=7,
        ncol=ncol_leg,
        framealpha=0.85,
        title='PCA clusters',
        title_fontsize=8,
        bbox_to_anchor=(0.98, 0.98),
    )

    fig_pca_on_coeff.suptitle(
        f'NLLD coefficients coloured by PCA-space clusters — {model}',
        fontsize=13, y=1.02,
    )

    fig_pca_on_coeff.savefig(
        save_data_path + f'Corner_NLLD_byPCAcluster_{model}.png',
        dpi=150, bbox_inches='tight',
    )
    plt.close(fig_pca_on_coeff)
    print(f'    Saved PCA-cluster-coloured coefficient corner plot')

    # ─────────────────────────────────────────────────────────────────────────────
    # Figure 5i: Corner plot of NLLD coefficients restricted to b=0, p=0.1
    # Also saves a dedicated .npy file with coefficients + stellar/wavelength info
    # ─────────────────────────────────────────────────────────────────────────────
    print('    FIGURE 5i - Coefficient corner plot for b=0, p=1e-1')

    # ── Identify the target b and p values ───────────────────────────────────
    target_b = 0.0
    target_p = 1e-1

    # corner_data columns: [c1, c2, c3, c4, b]
    # corner_meta columns: [p, Teff, logg, metallicity]
    # corner_wav_um      : wavelength in microns
    mask_bp = (
        np.isclose(corner_data[:, 4], target_b, atol=1e-8) &
        np.isclose(corner_meta[:, 0], target_p, rtol=1e-4)
    )

    n_selected = mask_bp.sum()
    print(f'    Selected {n_selected} / {len(corner_data)} profiles '
          f'with b={target_b} and p={target_p}')

    if n_selected == 0:
        print('    WARNING: no profiles match b=0, p=1e-1. Skipping Figure 5i.')
    else:
        # ── Extract the subset ───────────────────────────────────────────────
        coeffs_bp   = corner_data[mask_bp, :4]        # (n_selected, 4) — c1..c4
        meta_bp     = corner_meta[mask_bp]             # (n_selected, 4) — p, Teff, logg, met
        wav_bp      = corner_wav_um[mask_bp]           # (n_selected,)

        # ── Save dedicated .npy ──────────────────────────────────────────────
        # Columns: c1, c2, c3, c4, Teff, logg, metallicity, wavelength_um
        save_array = np.column_stack([
            coeffs_bp,                   # c1, c2, c3, c4
            meta_bp[:, 1],               # Teff
            meta_bp[:, 2],               # logg
            meta_bp[:, 3],               # metallicity
            wav_bp,                      # wavelength (µm)
        ])  # (n_selected, 8)

        col_description = (
            'Columns: c1, c2, c3, c4, Teff, logg, metallicity, wavelength_um'
        )

        npy_path = save_data_path + f'coeffs_b0_p0.1_{model}.npy'
        np.save(npy_path, save_array)
        print(f'    Saved {save_array.shape} array to {npy_path}')
        print(f'    {col_description}')

        # Print summary statistics
        print(f'    --- Subset summary ---')
        print(f'    c1   : [{coeffs_bp[:,0].min():.4f}, {coeffs_bp[:,0].max():.4f}]  '
              f'mean={coeffs_bp[:,0].mean():.4f}')
        print(f'    c2   : [{coeffs_bp[:,1].min():.4f}, {coeffs_bp[:,1].max():.4f}]  '
              f'mean={coeffs_bp[:,1].mean():.4f}')
        print(f'    c3   : [{coeffs_bp[:,2].min():.4f}, {coeffs_bp[:,2].max():.4f}]  '
              f'mean={coeffs_bp[:,2].mean():.4f}')
        print(f'    c4   : [{coeffs_bp[:,3].min():.4f}, {coeffs_bp[:,3].max():.4f}]  '
              f'mean={coeffs_bp[:,3].mean():.4f}')
        print(f'    Teff : [{meta_bp[:,1].min():.0f}, {meta_bp[:,1].max():.0f}] K')
        print(f'    logg : [{meta_bp[:,2].min():.2f}, {meta_bp[:,2].max():.2f}]')
        print(f'    [M/H]: [{meta_bp[:,3].min():.2f}, {meta_bp[:,3].max():.2f}]')
        print(f'    λ    : [{wav_bp.min():.2f}, {wav_bp.max():.2f}] µm')

        # ── 4D corner plot of c1–c4 ─────────────────────────────────────────
        labels_4d  = [r'$c_1$', r'$c_2$', r'$c_3$', r'$c_4$']
        ranges_4d  = [
            (np.percentile(coeffs_bp[:, ic], 1),
             np.percentile(coeffs_bp[:, ic], 99))
            for ic in range(4)
        ]

        # ── Base figure: uncoloured corner ───────────────────────────────────
        fig_5i = corner.corner(
            coeffs_bp,
            labels=labels_4d,
            range=ranges_4d,
            bins=50,
            smooth=1.0,
            smooth1d=1.0,
            plot_datapoints=False,
            plot_density=False,
            fill_contours=False,
            no_fill_contours=True,
            levels=(0.5, 0.68, 0.95, 0.99),
            hist_kwargs={'color': 'gray', 'linewidth': 1.2, 'alpha': 0.5},
            label_kwargs={'fontsize': 13},
            title_kwargs={'fontsize': 11},
            show_titles=False,
            contour_kwargs={'colors': 'none'},
        )

        ndim_4d  = 4
        axes_5i  = np.array(fig_5i.axes).reshape(ndim_4d, ndim_4d)

        # ── Colour by wavelength (the main remaining variable) ───────────────
        wav_cmap_5i = plt.cm.turbo
        wav_norm_5i = mcolors.Normalize(vmin=wav_bp.min(), vmax=wav_bp.max())

        # Scatter — sort so long wavelengths render on top
        max_scatter_5i = 40_000
        if n_selected > max_scatter_5i:
            rng_5i      = np.random.default_rng(42)
            idx_sub_5i  = np.sort(rng_5i.choice(n_selected,
                                                 size=max_scatter_5i, replace=False))
        else:
            idx_sub_5i = np.arange(n_selected)

        order_5i    = np.argsort(wav_bp[idx_sub_5i])
        idx_sub_5i  = idx_sub_5i[order_5i]

        for row in range(ndim_4d):
            for col in range(row):
                ax = axes_5i[row, col]
                ax.scatter(
                    coeffs_bp[idx_sub_5i, col],
                    coeffs_bp[idx_sub_5i, row],
                    c=wav_bp[idx_sub_5i],
                    cmap=wav_cmap_5i,
                    norm=wav_norm_5i,
                    s=2.5,
                    alpha=0.4,
                    linewidths=0,
                    rasterized=True,
                )

        # ── Coloured diagonal histograms by wavelength bin ───────────────────
        n_wav_hist_5i = 10
        wav_edges_5i  = np.linspace(wav_bp.min(), wav_bp.max(),
                                    n_wav_hist_5i + 1)
        wav_centres_5i = 0.5 * (wav_edges_5i[:-1] + wav_edges_5i[1:])

        for d in range(ndim_4d):
            ax = axes_5i[d, d]
            ax.clear()
            for ibin, centre in enumerate(wav_centres_5i):
                mask_bin = ((wav_bp >= wav_edges_5i[ibin]) &
                            (wav_bp < wav_edges_5i[ibin + 1]))
                if ibin == n_wav_hist_5i - 1:
                    mask_bin |= (wav_bp == wav_edges_5i[ibin + 1])
                if mask_bin.sum() == 0:
                    continue
                ax.hist(
                    coeffs_bp[mask_bin, d],
                    bins=50,
                    range=ranges_4d[d],
                    alpha=0.55,
                    color=wav_cmap_5i(wav_norm_5i(centre)),
                    density=True,
                    histtype='stepfilled',
                    edgecolor='none',
                )
            ax.set_xlim(ranges_4d[d])
            ax.set_yticks([])
            for spine in ['top', 'left', 'right']:
                ax.spines[spine].set_visible(False)
            if d == ndim_4d - 1:
                ax.set_xlabel(labels_4d[d], fontsize=13)

        # ── Restore axis labels ──────────────────────────────────────────────
        for row in range(ndim_4d):
            for col in range(row):
                ax = axes_5i[row, col]
                if col == 0:
                    ax.set_ylabel(labels_4d[row], fontsize=13)
                if row == ndim_4d - 1:
                    ax.set_xlabel(labels_4d[col], fontsize=13)
                ax.tick_params(labelsize=8)
                ax.grid(True, alpha=0.15)

        # ── Wavelength colorbar ──────────────────────────────────────────────
        cbar_ax_5i = fig_5i.add_axes([0.72, 0.72, 0.025, 0.20])
        sm_5i      = cm.ScalarMappable(cmap=wav_cmap_5i, norm=wav_norm_5i)
        sm_5i.set_array([])
        cbar_5i    = fig_5i.colorbar(sm_5i, cax=cbar_ax_5i)
        cbar_5i.set_label(r'Wavelength ($\mu$m)', fontsize=13)
        cbar_5i.ax.tick_params(labelsize=10)

        wav_tick_step_5i = 0.5
        wav_ticks_5i = np.arange(
            np.ceil(wav_bp.min() / wav_tick_step_5i) * wav_tick_step_5i,
            wav_bp.max() + wav_tick_step_5i / 2,
            wav_tick_step_5i,
        )
        cbar_5i.set_ticks(wav_ticks_5i)
        cbar_5i.set_ticklabels([f'{t:.1f}' for t in wav_ticks_5i])

        # ── Annotation box with the fixed parameters ─────────────────────────
        info_text = (
            f'Fixed parameters:\n'
            f'  $b = {target_b:.1f}$\n'
            f'  $p = {target_p}$\n'
            f'  $N = {n_selected}$ profiles'
        )
        fig_5i.text(
            0.73, 0.58, info_text,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.7),
        )

        fig_5i.suptitle(
            f'NLLD coefficients at $b=0$, $p=10^{{-1}}$ — {model}',
            fontsize=13, y=1.02,
        )

        fig_5i.savefig(
            save_data_path + f'Corner_NLLD_b0_p0.1_{model}.png',
            dpi=150, bbox_inches='tight',
        )
        plt.close(fig_5i)
        print(f'    Saved b=0, p=0.1 corner plot')

    # ─────────────────────────────────────────────────────────────────────────────
    # Mode identification in coefficient space — hierarchical clustering
    # ─────────────────────────────────────────────────────────────────────────────
    print('\n=== MODE IDENTIFICATION IN COEFFICIENT SPACE ===')

    # ── Coefficient space ─────────────────────────────────────────────────────
    # Returns 1-indexed labels; we keep them 1-indexed throughout this block
    # to match the printed summaries and saved filenames.
    mode_labels_corner, _, _ = hierarchical_clustering(
        data           = corner_data,
        label          = f'Coefficients_{model}',
        save_path      = save_data_path,
        feature_labels = [r'$c_1$', r'$c_2$', r'$c_3$', r'$c_4$', r'$b$'],
        cutoff         = 75,
        clustering_metric = 'mahalanobis',
        method = 'ward'
    )
    # mode_labels_corner : (N_valid,)  integers 1 … N_MODES_FOUND
    N_MODES     = len(np.unique(mode_labels_corner))
    MODE_COLORS = plt.cm.tab10(np.linspace(0, 1, min(N_MODES, 10)))

    for m in np.unique(mode_labels_corner):
        mask_m = mode_labels_corner == m
        print(
            f'  Mode {m}: n={mask_m.sum():5d}  '
            f'c=[{corner_data[mask_m,0].mean():.3f}, '
            f'{corner_data[mask_m,1].mean():.3f}, '
            f'{corner_data[mask_m,2].mean():.3f}, '
            f'{corner_data[mask_m,3].mean():.3f}]'
            f'{corner_data[mask_m,4].mean():.3f}]'
        )

    # ─────────────────────────────────────────────────────────────────────────────
    # Propagate mode labels back onto every PCA score in the common-profile set.
    # This block is unchanged in logic — it only needs labels from any clustering
    # source, so hierarchical labels drop straight in.
    # ─────────────────────────────────────────────────────────────────────────────
    block_sizes   = [len(c) for c in all_coeffs_per_b]
    block_offsets = np.concatenate([[0], np.cumsum(block_sizes)])
    full_stacked  = np.vstack(all_coeffs_per_b)
    valid_rows    = np.where(~np.any(np.isnan(full_stacked), axis=1))[0]

    mode_per_b_profile = []

    for ib in range(N_bs_ps):
        n_profs        = block_sizes[ib]
        global_rows_ib = np.arange(block_offsets[ib], block_offsets[ib] + n_profs)
        pos_in_valid   = np.searchsorted(valid_rows, global_rows_ib)
        pos_in_valid   = np.clip(pos_in_valid, 0, len(valid_rows) - 1)
        matched        = valid_rows[pos_in_valid] == global_rows_ib

        labels_ib          = np.full(n_profs, -1, dtype=int)
        labels_ib[matched] = mode_labels_corner[pos_in_valid[matched]]
        mode_per_b_profile.append(labels_ib)

    # ── PCA space coloured by coefficient-space mode (one figure per b) ──────
    for ib in range(N_bs_ps):
        hierarchical_clustering(
            data           = profiles_pca[ib],
            label          = f'PCA_byMode_{model}_b{ib}',
            save_path      = save_data_path,
            feature_labels = [f'PC{k+1}' for k in range(n_components)],
            cutoff         = 75,
            # Pass the back-propagated coefficient-space labels as an override
            # so the corner plot colours come from c-space, not a fresh PCA cluster.
            external_labels = mode_per_b_profile[ib],
            clustering_metric = 'mahalanobis',
            method = 'ward'
        )

    # ── Save ─────────────────────────────────────────────────────────────────
    np.save(save_data_path + f'mode_labels_corner_{model}.npy', mode_labels_corner)
    for ib in range(N_bs_ps):
        np.save(save_data_path + f'mode_labels_pca_{model}_b{ib}.npy',
                mode_per_b_profile[ib])
    print(f'  Saved mode label arrays to {save_data_path}')

    # ═════════════════════════════════════════════════════════════════════════════
    # DEDICATED ANALYSIS BLOCK: b = 0, p = 0.1
    # — Collect profiles, PCA, hierarchical clustering, 4th-order NLLD fits,
    #   corner plot of coefficients, and save everything.
    # ═════════════════════════════════════════════════════════════════════════════
    print('\n' + '=' * 80)
    print('  DEDICATED ANALYSIS: b = 0, p = 1e-1')
    print('=' * 80)

    # ── 0. Identify the target indices ───────────────────────────────────────
    target_b_val = 0.0
    target_p_val = 1e-1
    target_ib    = int(np.argmin(np.abs(np.array(bs) - target_b_val)))
    target_ip    = int(np.argmin(np.abs(np.array(ps) - target_p_val)))

    print(f'  Target b = {target_b_val}  →  ib = {target_ib}  '
          f'(actual b = {float(bs[target_ib]):.4f})')
    print(f'  Target p = {target_p_val}  →  ip = {target_ip}  '
          f'(actual p = {float(ps[target_ip]):.6f})')

    bp_save_path = os.path.join(save_data_path, 'b0_p0.1_analysis/')
    if not os.path.exists(bp_save_path):
        os.makedirs(bp_save_path)

    # ── 1. Collect every valid intensity profile for this (b, p) pair ────────
    print('  Collecting profiles …')

    bp_profiles_list = []
    bp_xs_list       = []
    bp_meta_list     = []          # columns: [i_Teff, j_logg, k_met, iw]

    T_vals_bp = np.linspace(Teffs[model][0],        Teffs[model][1],        N_star)
    g_vals_bp = np.linspace(loggs[model][0],         loggs[model][1],         N_star)
    m_vals_bp = np.linspace(metallicitys[model][0],  metallicitys[model][1],  N_star)
    wavs_ref_bp = np.array(gen_dict['stellar_wavelengths'][model][0, 0, 0])

    for i in range(N_star):
        for j in range(N_star):
            for k in range(N_star):
                entry      = np.array(
                    gen_dict['local_intensity_profiles'][model][i, j, k]
                )                                                          # (n_bs, n_ps, n_wav, N_chords)
                mask_entry = gen_dict['intensity_profiles_mask'][model][i, j, k]  # (n_bs, n_ps, n_wav)

                # Exclude specific (b, p) pairs if requested
                for ib_excl, ip_excl in excluded_bp_pairs:
                    mask_entry[ib_excl, ip_excl, :] = False

                profs_ijk = entry[target_ib, target_ip]                    # (n_wav, N_chords)
                mask_ijk  = mask_entry[target_ib, target_ip]               # (n_wav,)

                n_valid_ijk = int(np.sum(mask_ijk))
                if n_valid_ijk == 0:
                    continue

                valid_profs = profs_ijk[mask_ijk]                          # (n_valid_ijk, N_chords)
                bp_profiles_list.append(valid_profs)

                # x-values (mu-like, center→limb after reversal)
                rps_row = np.array(
                    gen_dict['local_rps'][model][i, j, k, target_ib, target_ip]
                )[::-1]                                                    # (N_chords,)
                bp_xs_list.append(
                    np.tile(rps_row, (n_valid_ijk, 1))
                )

                # Metadata: Teff index, logg index, met index, wavelength index
                wav_idx = np.where(mask_ijk)[0].astype(np.int32)
                meta_block = np.column_stack([
                    np.full(n_valid_ijk, i, dtype=np.int32),
                    np.full(n_valid_ijk, j, dtype=np.int32),
                    np.full(n_valid_ijk, k, dtype=np.int32),
                    wav_idx,
                ])
                bp_meta_list.append(meta_block)

    bp_profiles = np.vstack(bp_profiles_list).astype(np.float64)   # (N_bp, N_chords)
    bp_xs       = np.vstack(bp_xs_list).astype(np.float64)         # (N_bp, N_chords)
    bp_meta     = np.vstack(bp_meta_list)                          # (N_bp, 4)
    N_bp        = bp_profiles.shape[0]

    # Physical values for every profile
    bp_Teff = T_vals_bp[bp_meta[:, 0]]
    bp_logg = g_vals_bp[bp_meta[:, 1]]
    bp_met  = m_vals_bp[bp_meta[:, 2]]
    bp_wav  = wavs_ref_bp[bp_meta[:, 3]] / 1e4                    # → µm

    print(f'  Collected {N_bp} valid profiles')
    print(f'    Teff  : [{bp_Teff.min():.0f}, {bp_Teff.max():.0f}] K')
    print(f'    logg  : [{bp_logg.min():.2f}, {bp_logg.max():.2f}]')
    print(f'    [M/H] : [{bp_met.min():.2f}, {bp_met.max():.2f}]')
    print(f'    λ     : [{bp_wav.min():.2f}, {bp_wav.max():.2f}] µm')

    # ── 1b. Optional sub-sampling ────────────────────────────────────────────
    n_subsample_bp = min(n_subsample_profiles, N_bp) if subsample_profiles else N_bp

    if n_subsample_bp < N_bp:
        rng_bp  = np.random.default_rng(subsample_seed)
        idx_bp  = np.sort(rng_bp.choice(N_bp, size=n_subsample_bp, replace=False))
        print(f'  Sub-sampled {N_bp} → {n_subsample_bp} profiles')
    else:
        idx_bp = np.arange(N_bp)

    bp_profiles = bp_profiles[idx_bp]
    bp_xs       = bp_xs[idx_bp]
    bp_meta     = bp_meta[idx_bp]
    bp_Teff     = bp_Teff[idx_bp]
    bp_logg     = bp_logg[idx_bp]
    bp_met      = bp_met[idx_bp]
    bp_wav      = bp_wav[idx_bp]
    N_bp        = len(idx_bp)

    # ── 2. PCA ───────────────────────────────────────────────────────────────
    print(f'  Running PCA ({n_components} components) on {N_bp} profiles …')
    pca_bp        = PCA(n_components=n_components)
    bp_scores     = pca_bp.fit_transform(bp_profiles)              # (N_bp, n_components)
    bp_eigen      = pca_bp.components_                             # (n_components, N_chords)
    bp_evr        = pca_bp.explained_variance_ratio_

    print(f'    Variance explained per PC : '
          f'{", ".join(f"{v*100:.2f}%" for v in bp_evr)}')
    print(f'    Total variance captured   : {bp_evr.sum()*100:.2f}%')

    # ── 2a. PCA summary figure ───────────────────────────────────────────────
    ncols_pca_bp = 2 + n_components
    fig_pca_bp, ax_pca_bp = plt.subplots(1, ncols_pca_bp,
                                          figsize=(4 * ncols_pca_bp, 4))

    # Scree
    ax_pca_bp[0].plot(range(1, n_components + 1), bp_evr, 'o-',
                      color='teal', linewidth=2)
    ax_pca_bp[0].set_title('Scree')
    ax_pca_bp[0].set_xlabel('PC')
    ax_pca_bp[0].set_ylabel('Expl. var. ratio')
    ax_pca_bp[0].grid(True, alpha=0.3)

    # Cumulative
    ax_pca_bp[1].plot(range(1, n_components + 1), np.cumsum(bp_evr), 'o-',
                      color='teal', linewidth=2)
    ax_pca_bp[1].axhline(0.95, color='g', linestyle='--', label='95 %')
    ax_pca_bp[1].set_title('Cumul. variance')
    ax_pca_bp[1].set_xlabel('PC')
    ax_pca_bp[1].set_ylabel('Cumul. expl. var.')
    ax_pca_bp[1].legend(fontsize=7)
    ax_pca_bp[1].grid(True, alpha=0.3)

    # Eigen-profiles
    x_ref_bp = bp_xs[0]
    for ic in range(n_components):
        ax = ax_pca_bp[2 + ic]
        ax.plot(x_ref_bp, bp_eigen[ic], color=colors[ic], linewidth=1.5)
        ax.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax.set_title(f'PC{ic+1}  ({bp_evr[ic]*100:.1f} %)')
        ax.set_xlabel(r'$r / R_\star$')
        ax.set_ylabel('Component value')
        ax.grid(True, alpha=0.3)

    fig_pca_bp.suptitle(
        f'PCA — b = {target_b_val}, p = {target_p_val}  ({model})',
        fontsize=12, y=1.03)
    fig_pca_bp.tight_layout()
    fig_pca_bp.savefig(bp_save_path + f'PCA_summary_b0_p0.1_{model}.png',
                       dpi=150, bbox_inches='tight')
    plt.close(fig_pca_bp)
    print('    Saved PCA summary figure')

    # ── 3. Hierarchical clustering on PCA scores ────────────────────────────
    print('  Clustering on PCA scores …')

    bp_cl_labels, bp_cl_cutoff, bp_cl_Z = hierarchical_clustering(
        data              = bp_scores,
        label             = f'b0_p0.1_PCA_{model}',
        save_path         = bp_save_path,
        feature_labels    = [f'PC{k+1}' for k in range(n_components)],
        clustering_metric = 'mahalanobis',
        method            = 'single',
        cutoff            = None,          # auto-detect
    )

    bp_cl_0idx    = bp_cl_labels - 1       # → 0-indexed
    bp_unique_cl  = np.unique(bp_cl_0idx)
    n_cl_bp       = len(bp_unique_cl)

    # Identify typical (closest to centroid) and outlier (furthest) per cluster
    bp_typical_idx  = None
    bp_min_dist     = np.inf
    bp_outlier_idxs = []

    for cl in bp_unique_cl:
        mask_cl  = bp_cl_0idx == cl
        members  = bp_scores[mask_cl]
        centroid = members.mean(axis=0)
        dists    = np.linalg.norm(members - centroid, axis=1)

        closest_local = np.argmin(dists)
        closest_global = int(np.where(mask_cl)[0][closest_local])
        if dists[closest_local] < bp_min_dist:
            bp_min_dist    = dists[closest_local]
            bp_typical_idx = closest_global

        farthest_local  = np.argmax(dists)
        farthest_global = int(np.where(mask_cl)[0][farthest_local])
        bp_outlier_idxs.append(farthest_global)

    print(f'    Typical profile index : {bp_typical_idx}')
    print(f'    Outlier indices       : {bp_outlier_idxs}')

    # ── 3a. PCA corner scatter coloured by cluster ───────────────────────────
    bp_cluster_cmap   = plt.cm.get_cmap('tab10', n_cl_bp)
    bp_cluster_colors = [bp_cluster_cmap(c) for c in range(n_cl_bp)]

    fig_pca_corner_bp, axes_pcc = plt.subplots(
        n_components, n_components,
        figsize=(3 * n_components, 3 * n_components))
    fig_pca_corner_bp.suptitle(
        f'PCA cluster structure — b = 0, p = 0.1  ({model})',
        fontsize=12, y=1.01)

    for row in range(n_components):
        for col in range(n_components):
            ax = axes_pcc[row, col]

            if row == col:
                for ci, cl in enumerate(bp_unique_cl):
                    cmask = bp_cl_0idx == cl
                    ax.hist(bp_scores[cmask, row], bins=30, alpha=0.5,
                            color=bp_cluster_colors[ci], density=True,
                            edgecolor='none')
                ax.set_xlabel(f'PC{row+1}', fontsize=8)
                for sp in ['top', 'left', 'right']:
                    ax.spines[sp].set_visible(False)
                ax.set_yticks([])

            elif row > col:
                for ci, cl in enumerate(bp_unique_cl):
                    cmask = bp_cl_0idx == cl
                    ax.scatter(bp_scores[cmask, col], bp_scores[cmask, row],
                               color=bp_cluster_colors[ci], s=6, alpha=0.35,
                               linewidths=0, rasterized=True)

                # Typical
                ax.scatter(bp_scores[bp_typical_idx, col],
                           bp_scores[bp_typical_idx, row],
                           color='blue', s=80, marker='*',
                           edgecolors='k', linewidths=0.5, zorder=10,
                           label='Typical')
                # Outliers
                for oi, oidx in enumerate(bp_outlier_idxs):
                    ax.scatter(bp_scores[oidx, col], bp_scores[oidx, row],
                               color='red', s=60, marker='D',
                               edgecolors='k', linewidths=0.5, zorder=10,
                               label=f'Outlier {oi}' if col == 0 and row == 1 else '')

                if col == 0:
                    ax.set_ylabel(f'PC{row+1}', fontsize=8)
                if row == n_components - 1:
                    ax.set_xlabel(f'PC{col+1}', fontsize=8)
                ax.tick_params(labelsize=6)
                ax.grid(True, alpha=0.2)
            else:
                ax.set_visible(False)

    # Legend
    h_leg = ([plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=bp_cluster_colors[ci], markersize=7,
                          label=f'Cluster {cl}')
              for ci, cl in enumerate(bp_unique_cl)]
             + [plt.Line2D([0], [0], marker='*', color='w',
                           markerfacecolor='blue', markersize=9,
                           label='Typical'),
                plt.Line2D([0], [0], marker='D', color='w',
                           markerfacecolor='red', markersize=7,
                           label='Outlier')])
    fig_pca_corner_bp.legend(handles=h_leg, loc='upper right', fontsize=8,
                             framealpha=0.85)
    fig_pca_corner_bp.tight_layout()
    fig_pca_corner_bp.savefig(
        bp_save_path + f'PCA_Corner_b0_p0.1_{model}.png',
        dpi=150, bbox_inches='tight')
    plt.close(fig_pca_corner_bp)
    print('    Saved PCA corner scatter')

    # ── 4. Fit every profile with 4th-order NLLD ────────────────────────────
    print(f'  Fitting {N_bp} profiles with 4th-order NLLD …')

    bp_coeffs = np.zeros((N_bp, 4), dtype=np.float64)

    for idx_fit in tqdm(range(N_bp), desc='  NLLD fits'):
        mus_fit  = bp_xs[idx_fit]
        prof_fit = bp_profiles[idx_fit]

        if np.all(np.abs(prof_fit) < 1e-10):
            bp_coeffs[idx_fit] = np.nan
            continue

        params_fit = Parameters()
        for ipc in range(4):
            params_fit.add(f'c{ipc+1}', value=np.random.uniform(0, 1))

        try:
            res_fit = minimize(residual_fn, params_fit,
                               args=(mus_fit, prof_fit))
            bp_coeffs[idx_fit] = [
                res_fit.params[f'c{ic+1}'].value for ic in range(4)
            ]
        except Exception:
            bp_coeffs[idx_fit] = np.nan

    valid_fit = ~np.any(np.isnan(bp_coeffs), axis=1)
    n_valid_fit = int(valid_fit.sum())
    print(f'    {n_valid_fit} / {N_bp} fits converged')

    # ── 4a. Save coefficients (+metadata) ────────────────────────────────────
    bp_save_array = np.column_stack([
        bp_coeffs,          # c1, c2, c3, c4
        bp_Teff,            # Teff (K)
        bp_logg,            # log g
        bp_met,             # [M/H]
        bp_wav,             # wavelength (µm)
        bp_cl_0idx,         # PCA cluster label (0-indexed)
    ])   # (N_bp, 9)

    npy_bp_path = bp_save_path + f'coeffs_b0_p0.1_full_{model}.npy'
    np.save(npy_bp_path, bp_save_array)
    print(f'    Saved {bp_save_array.shape} array → {npy_bp_path}')
    print(f'    Columns: c1, c2, c3, c4, Teff, logg, [M/H], λ(µm), cluster')

    # ── 5. Figure: typical + outlier profiles with NLLD fits ─────────────────
    print('  Plotting typical & outlier profiles with NLLD fits …')

    special_bp = (
        [('Typical', bp_typical_idx, 'blue')]
        + [(f'Outlier {oi}', oidx, 'red')
           for oi, oidx in enumerate(bp_outlier_idxs)]
    )

    fig_prof_bp, axes_prof_bp = plt.subplots(
        len(special_bp), 2,
        figsize=(14, 4.5 * len(special_bp)),
        gridspec_kw={'width_ratios': [3, 1]})
    if len(special_bp) == 1:
        axes_prof_bp = axes_prof_bp[np.newaxis, :]

    fig_prof_bp.suptitle(
        f'4th-order NLLD fits — b = 0, p = 0.1  ({model})', fontsize=13)

    for isp, (sp_label, sp_idx, sp_color) in enumerate(special_bp):
        mus_sp  = bp_xs[sp_idx]
        prof_sp = bp_profiles[sp_idx]
        c_sp    = bp_coeffs[sp_idx]
        fit_sp  = fourNLLD(mus_sp, c_sp)

        ax_l = axes_prof_bp[isp, 0]
        ax_r = axes_prof_bp[isp, 1]

        # Background: all profiles in light grey
        step_bg = max(1, N_bp // 300)
        for ibg in range(0, N_bp, step_bg):
            ax_l.plot(bp_xs[ibg], bp_profiles[ibg],
                      alpha=0.1, color='gray', linewidth=0.3)

        # The special profile + fit
        ax_l.plot(mus_sp, prof_sp, 'o', color=sp_color, markersize=3,
                  alpha=0.7, label=f'{sp_label} profile')
        ax_l.plot(mus_sp, fit_sp, '--', color='black', linewidth=2,
                  label='4th-order NLLD')
        ax_l.set_ylabel('Norm. Intensity')
        ax_l.set_title(
            f'{sp_label}   '
            f'c=[{c_sp[0]:.4f}, {c_sp[1]:.4f}, {c_sp[2]:.4f}, {c_sp[3]:.4f}]',
            fontsize=9)
        ax_l.legend(fontsize=7)
        ax_l.grid(True, alpha=0.3)
        if isp < len(special_bp) - 1:
            ax_l.tick_params(labelbottom=False)
        else:
            ax_l.set_xlabel(r'$\mu = \cos(\theta)$')

        # Residual panel
        safe_sp = np.where(np.abs(prof_sp) < 1e-10, np.nan, prof_sp)
        ax_r.plot(mus_sp, 100 * (prof_sp - fit_sp) / safe_sp,
                  '--', color=sp_color, linewidth=1.5)
        ax_r.axhline(0, color='k', linestyle='--', alpha=0.5)
        ax_r.set_ylabel('Rel. diff. (%)')
        ax_r.grid(True, alpha=0.3)
        if isp < len(special_bp) - 1:
            ax_r.tick_params(labelbottom=False)
        else:
            ax_r.set_xlabel(r'$\mu = \cos(\theta)$')

        # Print coefficients
        redchi_sp = np.nan
        try:
            params_sp = Parameters()
            for ipc in range(4):
                params_sp.add(f'c{ipc+1}', value=c_sp[ipc])
            res_sp = minimize(residual_fn, params_sp, args=(mus_sp, prof_sp))
            redchi_sp = res_sp.redchi
        except Exception:
            pass

        print(f'    {sp_label:12s}  '
              f'c=[{c_sp[0]:+.4f}, {c_sp[1]:+.4f}, '
              f'{c_sp[2]:+.4f}, {c_sp[3]:+.4f}]  '
              f'red-χ²={redchi_sp:.4e}  '
              f'Teff={bp_Teff[sp_idx]:.0f} K  '
              f'logg={bp_logg[sp_idx]:.2f}  '
              f'[M/H]={bp_met[sp_idx]:.2f}  '
              f'λ={bp_wav[sp_idx]:.2f} µm')

    fig_prof_bp.tight_layout()
    fig_prof_bp.savefig(
        bp_save_path + f'Profiles_NLLD_b0_p0.1_{model}.png',
        dpi=150, bbox_inches='tight')
    plt.close(fig_prof_bp)
    print('    Saved typical & outlier profile figure')

    # ── 6. Figure: one profile per cluster (centroid-closest) with NLLD fit ──
    print('  Plotting per-cluster representative profiles …')

    fig_cl_bp, axes_cl_bp = plt.subplots(
        n_cl_bp, 2,
        figsize=(14, 4.5 * n_cl_bp),
        gridspec_kw={'width_ratios': [3, 1]})
    if n_cl_bp == 1:
        axes_cl_bp = axes_cl_bp[np.newaxis, :]

    fig_cl_bp.suptitle(
        f'Cluster representatives with NLLD fits — b = 0, p = 0.1  ({model})',
        fontsize=13)

    for ci, cl in enumerate(bp_unique_cl):
        mask_cl   = bp_cl_0idx == cl
        members   = bp_scores[mask_cl]
        centroid  = members.mean(axis=0)
        dists_cl  = np.linalg.norm(members - centroid, axis=1)
        rep_local = np.argmin(dists_cl)
        rep_idx   = int(np.where(mask_cl)[0][rep_local])

        mus_rep  = bp_xs[rep_idx]
        prof_rep = bp_profiles[rep_idx]
        c_rep    = bp_coeffs[rep_idx]
        fit_rep  = fourNLLD(mus_rep, c_rep)

        ax_l = axes_cl_bp[ci, 0]
        ax_r = axes_cl_bp[ci, 1]

        # All cluster members in background
        cl_indices = np.where(mask_cl)[0]
        step_cl    = max(1, len(cl_indices) // 200)
        for ii in cl_indices[::step_cl]:
            ax_l.plot(bp_xs[ii], bp_profiles[ii],
                      alpha=0.15, color=bp_cluster_colors[ci], linewidth=0.3)

        ax_l.plot(mus_rep, prof_rep, 'o',
                  color=bp_cluster_colors[ci], markersize=3, alpha=0.8,
                  label=f'Cluster {cl} representative')
        ax_l.plot(mus_rep, fit_rep, '--', color='black', linewidth=2,
                  label='4th-order NLLD')
        ax_l.set_ylabel('Norm. Intensity')
        ax_l.set_title(
            f'Cluster {cl}  (n={mask_cl.sum()})   '
            f'c=[{c_rep[0]:.4f}, {c_rep[1]:.4f}, '
            f'{c_rep[2]:.4f}, {c_rep[3]:.4f}]',
            fontsize=9)
        ax_l.legend(fontsize=7)
        ax_l.grid(True, alpha=0.3)
        if ci < n_cl_bp - 1:
            ax_l.tick_params(labelbottom=False)
        else:
            ax_l.set_xlabel(r'$\mu = \cos(\theta)$')

        safe_rep = np.where(np.abs(prof_rep) < 1e-10, np.nan, prof_rep)
        ax_r.plot(mus_rep, 100 * (prof_rep - fit_rep) / safe_rep,
                  '--', color=bp_cluster_colors[ci], linewidth=1.5)
        ax_r.axhline(0, color='k', linestyle='--', alpha=0.5)
        ax_r.set_ylabel('Rel. diff. (%)')
        ax_r.grid(True, alpha=0.3)
        if ci < n_cl_bp - 1:
            ax_r.tick_params(labelbottom=False)
        else:
            ax_r.set_xlabel(r'$\mu = \cos(\theta)$')

        print(f'    Cluster {cl:2d}  n={mask_cl.sum():5d}  '
              f'c=[{c_rep[0]:+.4f}, {c_rep[1]:+.4f}, '
              f'{c_rep[2]:+.4f}, {c_rep[3]:+.4f}]  '
              f'Teff={bp_Teff[rep_idx]:.0f}  '
              f'logg={bp_logg[rep_idx]:.2f}  '
              f'[M/H]={bp_met[rep_idx]:.2f}  '
              f'λ={bp_wav[rep_idx]:.2f} µm')

    fig_cl_bp.tight_layout()
    fig_cl_bp.savefig(
        bp_save_path + f'ClusterReps_NLLD_b0_p0.1_{model}.png',
        dpi=150, bbox_inches='tight')
    plt.close(fig_cl_bp)
    print('    Saved per-cluster representative figure')

    # ── 7. Corner plot of NLLD coefficients coloured by cluster ──────────────
    print('  Corner plot of coefficients coloured by cluster …')

    bp_coeffs_valid   = bp_coeffs[valid_fit]
    bp_cl_valid       = bp_cl_0idx[valid_fit]
    bp_wav_valid      = bp_wav[valid_fit]

    labels_bp_corner  = [r'$c_1$', r'$c_2$', r'$c_3$', r'$c_4$']
    ranges_bp_corner  = [
        (np.percentile(bp_coeffs_valid[:, ic], 1),
         np.percentile(bp_coeffs_valid[:, ic], 99))
        for ic in range(4)
    ]

    # ── 7a. Coloured by PCA cluster ─────────────────────────────────────────
    fig_cc_bp = corner.corner(
        bp_coeffs_valid,
        labels=labels_bp_corner,
        range=ranges_bp_corner,
        bins=50,
        smooth1d=1.0,
        plot_datapoints=False,
        plot_density=False,
        fill_contours=False,
        no_fill_contours=True,
        levels=(0.5, 0.68, 0.95, 0.99),
        hist_kwargs={'color': 'gray', 'linewidth': 1.2, 'alpha': 0.5},
        label_kwargs={'fontsize': 13},
        show_titles=False,
        contour_kwargs={'colors': 'none'},
    )

    ndim_bp    = 4
    axes_cc_bp = np.array(fig_cc_bp.axes).reshape(ndim_bp, ndim_bp)

    # Scatter in lower triangle
    max_scatter_bp = 40_000
    if n_valid_fit > max_scatter_bp:
        rng_cc     = np.random.default_rng(42)
        idx_cc_sub = np.sort(rng_cc.choice(n_valid_fit, size=max_scatter_bp,
                                           replace=False))
    else:
        idx_cc_sub = np.arange(n_valid_fit)

    for row in range(ndim_bp):
        for col in range(row):
            ax = axes_cc_bp[row, col]
            for ci, cl in enumerate(bp_unique_cl):
                mask_cl_v = bp_cl_valid[idx_cc_sub] == cl
                if mask_cl_v.sum() == 0:
                    continue
                ax.scatter(
                    bp_coeffs_valid[idx_cc_sub[mask_cl_v], col],
                    bp_coeffs_valid[idx_cc_sub[mask_cl_v], row],
                    color=bp_cluster_colors[ci],
                    s=2, alpha=0.35, linewidths=0, rasterized=True)

    # Coloured diagonal histograms
    for d in range(ndim_bp):
        ax = axes_cc_bp[d, d]
        ax.clear()
        for ci, cl in enumerate(bp_unique_cl):
            mask_cl_v = bp_cl_valid == cl
            if mask_cl_v.sum() == 0:
                continue
            ax.hist(bp_coeffs_valid[mask_cl_v, d], bins=50,
                    range=ranges_bp_corner[d], alpha=0.55,
                    color=bp_cluster_colors[ci], density=True,
                    histtype='stepfilled', edgecolor='none')
        ax.set_xlim(ranges_bp_corner[d])
        ax.set_yticks([])
        for sp in ['top', 'left', 'right']:
            ax.spines[sp].set_visible(False)
        if d == ndim_bp - 1:
            ax.set_xlabel(labels_bp_corner[d], fontsize=13)

    # Axis labels
    for row in range(ndim_bp):
        for col in range(row):
            ax = axes_cc_bp[row, col]
            if col == 0:
                ax.set_ylabel(labels_bp_corner[row], fontsize=13)
            if row == ndim_bp - 1:
                ax.set_xlabel(labels_bp_corner[col], fontsize=13)
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.15)

    # Legend
    h_cc = [plt.Line2D([0], [0], marker='o', color='w',
                        markerfacecolor=bp_cluster_colors[ci], markersize=7,
                        label=f'Cluster {cl}')
            for ci, cl in enumerate(bp_unique_cl)]
    fig_cc_bp.legend(handles=h_cc, loc='upper right', fontsize=8,
                     framealpha=0.85, title='PCA cluster')

    info_box = (f'$b = {target_b_val:.1f}$,  $p = {target_p_val}$\n'
                f'$N = {n_valid_fit}$ converged fits\n'
                f'{n_cl_bp} clusters')
    fig_cc_bp.text(0.72, 0.55, info_box, fontsize=10,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.4',
                             facecolor='wheat', alpha=0.7))

    fig_cc_bp.suptitle(
        f'NLLD coefficients by PCA cluster — b=0, p=0.1  ({model})',
        fontsize=13, y=1.02)

    fig_cc_bp.savefig(
        bp_save_path + f'Corner_coeffs_byCluster_b0_p0.1_{model}.png',
        dpi=150, bbox_inches='tight')
    plt.close(fig_cc_bp)
    print('    Saved cluster-coloured coefficient corner plot')

    # ── 7b. Coloured by wavelength ───────────────────────────────────────────
    print('  Corner plot of coefficients coloured by wavelength …')

    wav_cmap_bp  = plt.cm.turbo
    wav_norm_bp  = mcolors.Normalize(vmin=bp_wav_valid.min(),
                                     vmax=bp_wav_valid.max())

    fig_cw_bp = corner.corner(
        bp_coeffs_valid,
        labels=labels_bp_corner,
        range=ranges_bp_corner,
        bins=50,
        smooth1d=1.0,
        plot_datapoints=False,
        plot_density=False,
        fill_contours=False,
        no_fill_contours=True,
        levels=(0.5, 0.68, 0.95, 0.99),
        hist_kwargs={'color': 'gray', 'linewidth': 1.2, 'alpha': 0.5},
        label_kwargs={'fontsize': 13},
        show_titles=False,
        contour_kwargs={'colors': 'none'},
    )

    axes_cw_bp = np.array(fig_cw_bp.axes).reshape(ndim_bp, ndim_bp)

    order_wav_bp = np.argsort(bp_wav_valid[idx_cc_sub])
    idx_wav_sub  = idx_cc_sub[order_wav_bp]

    for row in range(ndim_bp):
        for col in range(row):
            ax = axes_cw_bp[row, col]
            ax.scatter(
                bp_coeffs_valid[idx_wav_sub, col],
                bp_coeffs_valid[idx_wav_sub, row],
                c=bp_wav_valid[idx_wav_sub],
                cmap=wav_cmap_bp, norm=wav_norm_bp,
                s=2, alpha=0.35, linewidths=0, rasterized=True)

    # Coloured diagonal histograms by wavelength bin
    n_wav_hist_bp  = 10
    wav_edges_bp   = np.linspace(bp_wav_valid.min(), bp_wav_valid.max(),
                                 n_wav_hist_bp + 1)
    wav_centres_bp = 0.5 * (wav_edges_bp[:-1] + wav_edges_bp[1:])

    for d in range(ndim_bp):
        ax = axes_cw_bp[d, d]
        ax.clear()
        for ibin, centre in enumerate(wav_centres_bp):
            mask_bin = ((bp_wav_valid >= wav_edges_bp[ibin]) &
                        (bp_wav_valid < wav_edges_bp[ibin + 1]))
            if ibin == n_wav_hist_bp - 1:
                mask_bin |= (bp_wav_valid == wav_edges_bp[ibin + 1])
            if mask_bin.sum() == 0:
                continue
            ax.hist(bp_coeffs_valid[mask_bin, d], bins=50,
                    range=ranges_bp_corner[d], alpha=0.55,
                    color=wav_cmap_bp(wav_norm_bp(centre)),
                    density=True, histtype='stepfilled', edgecolor='none')
        ax.set_xlim(ranges_bp_corner[d])
        ax.set_yticks([])
        for sp in ['top', 'left', 'right']:
            ax.spines[sp].set_visible(False)
        if d == ndim_bp - 1:
            ax.set_xlabel(labels_bp_corner[d], fontsize=13)

    for row in range(ndim_bp):
        for col in range(row):
            ax = axes_cw_bp[row, col]
            if col == 0:
                ax.set_ylabel(labels_bp_corner[row], fontsize=13)
            if row == ndim_bp - 1:
                ax.set_xlabel(labels_bp_corner[col], fontsize=13)
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.15)

    # Wavelength colorbar
    cbar_ax_cw = fig_cw_bp.add_axes([0.72, 0.72, 0.025, 0.20])
    sm_cw      = cm.ScalarMappable(cmap=wav_cmap_bp, norm=wav_norm_bp)
    sm_cw.set_array([])
    cbar_cw    = fig_cw_bp.colorbar(sm_cw, cax=cbar_ax_cw)
    cbar_cw.set_label(r'Wavelength ($\mu$m)', fontsize=13)
    cbar_cw.ax.tick_params(labelsize=10)
    wav_tick_bp = 0.5
    wav_ticks_cw = np.arange(
        np.ceil(bp_wav_valid.min() / wav_tick_bp) * wav_tick_bp,
        bp_wav_valid.max() + wav_tick_bp / 2,
        wav_tick_bp)
    cbar_cw.set_ticks(wav_ticks_cw)
    cbar_cw.set_ticklabels([f'{t:.1f}' for t in wav_ticks_cw])

    fig_cw_bp.suptitle(
        f'NLLD coefficients by wavelength — b=0, p=0.1  ({model})',
        fontsize=13, y=1.02)
    fig_cw_bp.savefig(
        bp_save_path + f'Corner_coeffs_byWav_b0_p0.1_{model}.png',
        dpi=150, bbox_inches='tight')
    plt.close(fig_cw_bp)
    print('    Saved wavelength-coloured coefficient corner plot')

    # ── 7c. Coloured by Teff, logg, metallicity ─────────────────────────────
    bp_Teff_valid = bp_Teff[valid_fit]
    bp_logg_valid = bp_logg[valid_fit]
    bp_met_valid  = bp_met[valid_fit]

    bp_meta_sources = [
        (r'$T_{\rm eff}$ (K)', 'Teff',  bp_Teff_valid, plt.cm.inferno),
        (r'$\log\,g$',         'logg',  bp_logg_valid, plt.cm.cividis),
        (r'[M/H]',             'MH',    bp_met_valid,  plt.cm.coolwarm),
    ]

    for meta_label, meta_key, meta_vals, meta_cmap in bp_meta_sources:
        print(f'  Corner plot of coefficients coloured by {meta_key} …')

        meta_norm = mcolors.Normalize(vmin=meta_vals.min(),
                                      vmax=meta_vals.max())

        fig_cm_bp = corner.corner(
            bp_coeffs_valid,
            labels=labels_bp_corner,
            range=ranges_bp_corner,
            bins=50, smooth1d=1.0,
            plot_datapoints=False, plot_density=False,
            fill_contours=False, no_fill_contours=True,
            levels=(0.5, 0.68, 0.95, 0.99),
            hist_kwargs={'color': 'gray', 'linewidth': 1.2, 'alpha': 0.5},
            label_kwargs={'fontsize': 13}, show_titles=False,
            contour_kwargs={'colors': 'none'},
        )

        axes_cm_bp = np.array(fig_cm_bp.axes).reshape(ndim_bp, ndim_bp)

        order_meta = np.argsort(meta_vals[idx_cc_sub])
        idx_meta_sub = idx_cc_sub[order_meta]

        for row in range(ndim_bp):
            for col in range(row):
                ax = axes_cm_bp[row, col]
                ax.scatter(
                    bp_coeffs_valid[idx_meta_sub, col],
                    bp_coeffs_valid[idx_meta_sub, row],
                    c=meta_vals[idx_meta_sub],
                    cmap=meta_cmap, norm=meta_norm,
                    s=2, alpha=0.35, linewidths=0, rasterized=True)

        # Coloured diagonal
        unique_meta = np.unique(meta_vals)
        for d in range(ndim_bp):
            ax = axes_cm_bp[d, d]
            ax.clear()
            if len(unique_meta) <= 20:
                for uv in unique_meta:
                    mask_uv = meta_vals == uv
                    if mask_uv.sum() == 0:
                        continue
                    ax.hist(bp_coeffs_valid[mask_uv, d], bins=50,
                            range=ranges_bp_corner[d], alpha=0.55,
                            color=meta_cmap(meta_norm(uv)),
                            density=True, histtype='stepfilled',
                            edgecolor='none')
            else:
                n_bins_meta = 8
                meta_edges  = np.linspace(meta_vals.min(), meta_vals.max(),
                                          n_bins_meta + 1)
                meta_ctrs   = 0.5 * (meta_edges[:-1] + meta_edges[1:])
                for ib_m, ctr in enumerate(meta_ctrs):
                    mask_bm = ((meta_vals >= meta_edges[ib_m]) &
                               (meta_vals < meta_edges[ib_m + 1]))
                    if ib_m == len(meta_ctrs) - 1:
                        mask_bm |= (meta_vals == meta_edges[ib_m + 1])
                    if mask_bm.sum() == 0:
                        continue
                    ax.hist(bp_coeffs_valid[mask_bm, d], bins=50,
                            range=ranges_bp_corner[d], alpha=0.55,
                            color=meta_cmap(meta_norm(ctr)),
                            density=True, histtype='stepfilled',
                            edgecolor='none')
            ax.set_xlim(ranges_bp_corner[d])
            ax.set_yticks([])
            for sp in ['top', 'left', 'right']:
                ax.spines[sp].set_visible(False)
            if d == ndim_bp - 1:
                ax.set_xlabel(labels_bp_corner[d], fontsize=13)

        for row in range(ndim_bp):
            for col in range(row):
                ax = axes_cm_bp[row, col]
                if col == 0:
                    ax.set_ylabel(labels_bp_corner[row], fontsize=13)
                if row == ndim_bp - 1:
                    ax.set_xlabel(labels_bp_corner[col], fontsize=13)
                ax.tick_params(labelsize=8)
                ax.grid(True, alpha=0.15)

        cbar_ax_m = fig_cm_bp.add_axes([0.72, 0.72, 0.025, 0.20])
        sm_m      = cm.ScalarMappable(cmap=meta_cmap, norm=meta_norm)
        sm_m.set_array([])
        cbar_m    = fig_cm_bp.colorbar(sm_m, cax=cbar_ax_m)
        cbar_m.set_label(meta_label, fontsize=13)
        cbar_m.ax.tick_params(labelsize=10)
        if len(unique_meta) <= 15:
            cbar_m.set_ticks(unique_meta)
            if meta_key == 'Teff':
                cbar_m.set_ticklabels([f'{v:.0f}' for v in unique_meta])
            else:
                cbar_m.set_ticklabels([f'{v:.2f}' for v in unique_meta])

        fig_cm_bp.suptitle(
            f'NLLD coefficients by {meta_label} — b=0, p=0.1  ({model})',
            fontsize=13, y=1.02)
        fig_cm_bp.savefig(
            bp_save_path + f'Corner_coeffs_by{meta_key}_b0_p0.1_{model}.png',
            dpi=150, bbox_inches='tight')
        plt.close(fig_cm_bp)
        print(f'    Saved {meta_key}-coloured coefficient corner plot')

    # ── 8. Reconstruction quality for typical profile ────────────────────────
    print('  Reconstruction quality for typical profile …')

    n_comp_check = [1, max(2, n_components // 2), n_components]

    fig_recon_bp, axes_recon_bp = plt.subplots(
        len(n_comp_check), 2,
        figsize=(12, 4 * len(n_comp_check)),
        gridspec_kw={'width_ratios': [3, 1]})

    fig_recon_bp.suptitle(
        f'PCA reconstruction — typical profile, b=0, p=0.1  ({model})',
        fontsize=13)

    original_bp = bp_profiles[bp_typical_idx]
    x_orig_bp   = bp_xs[bp_typical_idx]

    for ri, nc in enumerate(n_comp_check):
        pca_temp_bp = PCA(n_components=nc)
        pca_temp_bp.fit(bp_profiles)
        recon_bp = pca_temp_bp.inverse_transform(
            pca_temp_bp.transform(original_bp.reshape(1, -1)))[0]
        resid_bp = original_bp - recon_bp
        rmse_bp  = np.sqrt(np.mean(resid_bp ** 2))

        ax_t = axes_recon_bp[ri, 0]
        ax_b = axes_recon_bp[ri, 1]

        ax_t.plot(x_orig_bp, original_bp, color='teal', linewidth=2,
                  label='Original')
        ax_t.plot(x_orig_bp, recon_bp, color='teal', linewidth=1.2,
                  linestyle='--', label='Reconstructed')
        ax_t.set_title(f'{nc} PC   RMSE = {rmse_bp:.6f}', fontsize=9)
        ax_t.set_ylabel('Norm. Intensity')
        ax_t.legend(fontsize=7)
        ax_t.grid(True, alpha=0.3)

        safe_bp = np.where(np.abs(original_bp) < 1e-10, np.nan, original_bp)
        ax_b.plot(x_orig_bp, 100 * resid_bp / safe_bp,
                  color='teal', linewidth=1.0)
        ax_b.axhline(0, color='k', linestyle='--', alpha=0.5)
        ax_b.set_ylabel('Rel. diff. (%)')
        ax_b.grid(True, alpha=0.3)
        if ri == len(n_comp_check) - 1:
            ax_t.set_xlabel(r'$r / R_\star$')
            ax_b.set_xlabel(r'$r / R_\star$')

    fig_recon_bp.tight_layout()
    fig_recon_bp.savefig(
        bp_save_path + f'ReconQuality_b0_p0.1_{model}.png',
        dpi=150, bbox_inches='tight')
    plt.close(fig_recon_bp)
    print('    Saved reconstruction quality figure')

    # ── 9. Final summary ─────────────────────────────────────────────────────
    print(f'\n  ═══ b=0, p=0.1 analysis complete for {model} ═══')
    print(f'  Profiles collected      : {N_bp}')
    print(f'  PCA components          : {n_components}')
    print(f'  Variance captured       : {bp_evr.sum()*100:.2f} %')
    print(f'  Clusters found          : {n_cl_bp}')
    print(f'  Converged NLLD fits     : {n_valid_fit} / {N_bp}')
    print(f'  All outputs saved to    : {bp_save_path}')

    # Print coefficient summary statistics
    print(f'\n  ── Coefficient summary (converged fits) ──')
    for ic in range(4):
        vals = bp_coeffs_valid[:, ic]
        print(f'    c{ic+1}:  mean={vals.mean():+.4f}  '
              f'median={np.median(vals):+.4f}  '
              f'std={vals.std():.4f}  '
              f'[{vals.min():+.4f}, {vals.max():+.4f}]')

    # Per-cluster coefficient summary
    print(f'\n  ── Per-cluster coefficient means ──')
    for ci, cl in enumerate(bp_unique_cl):
        mask_cl_v = bp_cl_valid == cl
        n_in_cl   = mask_cl_v.sum()
        if n_in_cl == 0:
            continue
        means_cl = bp_coeffs_valid[mask_cl_v].mean(axis=0)
        stds_cl  = bp_coeffs_valid[mask_cl_v].std(axis=0)
        print(f'    Cluster {cl} (n={n_in_cl:5d}):  '
              f'c=[{means_cl[0]:+.4f}±{stds_cl[0]:.4f}, '
              f'{means_cl[1]:+.4f}±{stds_cl[1]:.4f}, '
              f'{means_cl[2]:+.4f}±{stds_cl[2]:.4f}, '
              f'{means_cl[3]:+.4f}±{stds_cl[3]:.4f}]')

    # ── 10. Save a summary pickle with everything for easy downstream use ────
    bp_summary = {
        'model'             : model,
        'target_b'          : target_b_val,
        'target_p'          : target_p_val,
        'ib'                : target_ib,
        'ip'                : target_ip,
        'N_profiles'        : N_bp,
        'n_components'      : n_components,
        'explained_variance': bp_evr,
        'pca_object'        : pca_bp,
        'pca_scores'        : bp_scores,
        'pca_eigen'         : bp_eigen,
        'profiles'          : bp_profiles,
        'xs'                : bp_xs,
        'coeffs'            : bp_coeffs,
        'valid_fit_mask'    : valid_fit,
        'cluster_labels'    : bp_cl_0idx,
        'n_clusters'        : n_cl_bp,
        'cluster_cutoff'    : bp_cl_cutoff,
        'typical_idx'       : bp_typical_idx,
        'outlier_idxs'      : bp_outlier_idxs,
        'meta_indices'      : bp_meta,          # [i_Teff, j_logg, k_met, iw]
        'Teff'              : bp_Teff,
        'logg'              : bp_logg,
        'metallicity'       : bp_met,
        'wavelength_um'     : bp_wav,
        'stellar_params'    : {
            'T_vals' : T_vals_bp,
            'g_vals' : g_vals_bp,
            'm_vals' : m_vals_bp,
        },
        'wavs_ref_angstrom' : wavs_ref_bp,
    }

    pkl_path = bp_save_path + f'summary_b0_p0.1_{model}.pkl'
    with open(pkl_path, 'wb') as f:
        pickle.dump(bp_summary, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'\n  Saved summary pickle → {pkl_path}')

    # ── 11. Also save the typical and outlier coefficients as plain text ──────
    #        for quick copy-paste into downstream scripts
    txt_path = bp_save_path + f'special_coeffs_b0_p0.1_{model}.txt'
    with open(txt_path, 'w') as ftxt:
        ftxt.write(f'# 4th-order NLLD coefficients for b={target_b_val}, '
                   f'p={target_p_val}, model={model}\n')
        ftxt.write(f'# Format: label  c1  c2  c3  c4  Teff  logg  [M/H]  '
                   f'wavelength_um  cluster\n')
        ftxt.write('#\n')

        c_typ = bp_coeffs[bp_typical_idx]
        ftxt.write(f'typical  '
                   f'{c_typ[0]:+.6f}  {c_typ[1]:+.6f}  '
                   f'{c_typ[2]:+.6f}  {c_typ[3]:+.6f}  '
                   f'{bp_Teff[bp_typical_idx]:.0f}  '
                   f'{bp_logg[bp_typical_idx]:.2f}  '
                   f'{bp_met[bp_typical_idx]:.2f}  '
                   f'{bp_wav[bp_typical_idx]:.4f}  '
                   f'{bp_cl_0idx[bp_typical_idx]}\n')

        for oi, oidx in enumerate(bp_outlier_idxs):
            c_out = bp_coeffs[oidx]
            ftxt.write(f'outlier{oi}  '
                       f'{c_out[0]:+.6f}  {c_out[1]:+.6f}  '
                       f'{c_out[2]:+.6f}  {c_out[3]:+.6f}  '
                       f'{bp_Teff[oidx]:.0f}  '
                       f'{bp_logg[oidx]:.2f}  '
                       f'{bp_met[oidx]:.2f}  '
                       f'{bp_wav[oidx]:.4f}  '
                       f'{bp_cl_0idx[oidx]}\n')

        # Also write per-cluster centroid-closest coefficients
        ftxt.write('#\n')
        ftxt.write('# Per-cluster representatives (closest to centroid)\n')
        for ci, cl in enumerate(bp_unique_cl):
            mask_cl = bp_cl_0idx == cl
            members = bp_scores[mask_cl]
            centroid = members.mean(axis=0)
            dists_cl = np.linalg.norm(members - centroid, axis=1)
            rep_idx  = int(np.where(mask_cl)[0][np.argmin(dists_cl)])
            c_rep    = bp_coeffs[rep_idx]
            ftxt.write(f'cluster{cl}_rep  '
                       f'{c_rep[0]:+.6f}  {c_rep[1]:+.6f}  '
                       f'{c_rep[2]:+.6f}  {c_rep[3]:+.6f}  '
                       f'{bp_Teff[rep_idx]:.0f}  '
                       f'{bp_logg[rep_idx]:.2f}  '
                       f'{bp_met[rep_idx]:.2f}  '
                       f'{bp_wav[rep_idx]:.4f}  '
                       f'{cl}\n')

        # Per-cluster mean coefficients
        ftxt.write('#\n')
        ftxt.write('# Per-cluster mean coefficients (± std)\n')
        for ci, cl in enumerate(bp_unique_cl):
            mask_cl_v = bp_cl_valid == cl
            if mask_cl_v.sum() == 0:
                continue
            means = bp_coeffs_valid[mask_cl_v].mean(axis=0)
            stds  = bp_coeffs_valid[mask_cl_v].std(axis=0)
            ftxt.write(f'cluster{cl}_mean  '
                       f'{means[0]:+.6f}±{stds[0]:.6f}  '
                       f'{means[1]:+.6f}±{stds[1]:.6f}  '
                       f'{means[2]:+.6f}±{stds[2]:.6f}  '
                       f'{means[3]:+.6f}±{stds[3]:.6f}  '
                       f'n={mask_cl_v.sum()}\n')

    print(f'  Saved special coefficients → {txt_path}')

    # ── Clean up ─────────────────────────────────────────────────────────────
    del bp_profiles_list, bp_xs_list, bp_meta_list
    del bp_profiles, bp_xs, bp_meta, bp_scores, bp_coeffs
    del bp_coeffs_valid, bp_cl_valid, bp_wav_valid
    del bp_Teff_valid, bp_logg_valid, bp_met_valid
    gc.collect()

    print('\n' + '=' * 80)
    print(f'  DEDICATED b=0, p=0.1 ANALYSIS COMPLETE — {model}')
    print('=' * 80 + '\n')