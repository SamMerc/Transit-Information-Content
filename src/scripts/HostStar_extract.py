#############################
########## Purpose ##########
#############################

# A reviewer of the paper pointed out that the grid of stellar properties (Teff, logg, metallicity)
# explored in Fig5_run.py for the limb-darkening clustering analysis is not representative of the
# population of known exoplanet-hosting stars. This is step 1 of 3 in addressing this:
#   1. Extract the Teff / logg / metallicity distribution of confirmed exoplanet hosts from the
#      NASA Exoplanet Archive and histogram it (this file).
#   2. Fit the resulting histogram with a Gaussian mixture / density estimator.
#   3. Use the fitted density to weight the limb-darkening coefficients during clustering.


######################################
########## Import libraries ##########
######################################

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
from scipy.stats import gaussian_kde
import paths


######################################
########## Hyper-parameters ##########
######################################

save_data_path = str(paths.data / "HostStar_Storage") + "/"

# Either cache the response from a NASA Exoplanet Archive query or load the cached response from disk.
host_star_mode = 'load'  # 'query' or 'load'

# Pin the archive query to this date so the extracted sample matches what would have been
# returned on that day, regardless of when this script is actually run.
query_asof_date = '2026-07-08'

archive_table  = 'ps'
archive_select = 'hostname,st_teff,st_logg,st_met,rowupdate'
archive_where  = (
    'st_teff is not null and st_logg is not null and st_met is not null '
    f"and rowupdate <= '{query_asof_date}'"
)

n_bins = 30

# ---- Joint KDE (step 2) ----
kde_grid_resolution   = 100          # grid points per axis, used to evaluate / marginalize the joint KDE
kde_percentile_range  = [0.1, 99.9] # per-axis percentile range spanned by the evaluation grid
kde_scatter_subsample = 2000        # number of raw data points shown in the corner-plot scatter panels


############################
###### Function block ######
############################

def fetch_host_star_properties():
    """Query the NASA Exoplanet Archive and return one (Teff, logg, [M/H]) row per unique host star."""
    table = NasaExoplanetArchive.query_criteria(
        table=archive_table, select=archive_select, where=archive_where,
    )
    df = table.to_pandas()
    df = df.dropna(subset=['st_teff', 'st_logg', 'st_met'])
    df = df.sort_values('rowupdate', ascending=False)
    df = df.drop_duplicates(subset='hostname', keep='first')
    return df


def plot_histograms(df, save_path):
    """Plot the Teff / logg / [M/H] histograms of exoplanet-hosting stars side by side."""
    columns = ['st_teff', 'st_logg', 'st_met']
    labels  = [r'$T_{\rm eff}$ (K)', r'$\log g$', r'[M/H]']

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'Exoplanet host-star properties (N = {len(df)} stars)', fontsize=12)

    for ax, col, label in zip(axes, columns, labels):
        ax.hist(df[col], bins=n_bins, color='steelblue', edgecolor='k', linewidth=0.5)
        ax.set_xlabel(label)
        ax.set_ylabel('Number of host stars')
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path + 'HostStar_histograms.pdf', dpi=150, bbox_inches='tight')
    plt.close(fig)


def fit_joint_kde(df):
    """Fit a joint Gaussian KDE to (Teff, logg, [M/H]).

    Fitting the three parameters jointly (rather than one 1-D KDE per parameter) preserves the
    correlations between them (e.g. the Teff-logg main-sequence locus), so the density does not
    assign weight to physically implausible parameter combinations.
    """
    data = np.vstack([df['st_teff'], df['st_logg'], df['st_met']])
    return gaussian_kde(data)


def evaluate_kde_grid(kde, df):
    """Evaluate the joint KDE on a 3D grid and marginalize it for each corner-plot panel.

    Because a Gaussian KDE is a mixture of 3-D Gaussians sharing one bandwidth matrix, its exact
    marginal over any subset of axes is again a Gaussian mixture — no re-fitting is needed, only
    numerical integration (`np.trapezoid`) of the evaluated grid over the axes being marginalized out.
    """
    columns = ['st_teff', 'st_logg', 'st_met']
    axes_1d = [np.linspace(*np.percentile(df[col], kde_percentile_range), kde_grid_resolution)
               for col in columns]

    grids       = np.meshgrid(*axes_1d, indexing='ij')  # axis order: Teff, logg, met
    grid_points = np.vstack([g.ravel() for g in grids])
    dens_3d     = kde(grid_points).reshape(grids[0].shape)

    marginal_2d = {
        (0, 1): np.trapezoid(dens_3d, axes_1d[2], axis=2),  # Teff-logg, met integrated out
        (0, 2): np.trapezoid(dens_3d, axes_1d[1], axis=1),  # Teff-met,  logg integrated out
        (1, 2): np.trapezoid(dens_3d, axes_1d[0], axis=0),  # logg-met,  Teff integrated out
    }
    marginal_1d = {
        0: np.trapezoid(marginal_2d[(0, 1)], axes_1d[1], axis=1),  # Teff
        1: np.trapezoid(marginal_2d[(0, 1)], axes_1d[0], axis=0),  # logg
        2: np.trapezoid(marginal_2d[(0, 2)], axes_1d[0], axis=0),  # met
    }
    return axes_1d, marginal_1d, marginal_2d


def plot_kde_corner(df, axes_1d, marginal_1d, marginal_2d, save_path):
    """Corner plot comparing the raw host-star data to the fitted joint-KDE marginals."""
    columns = ['st_teff', 'st_logg', 'st_met']
    labels  = [r'$T_{\rm eff}$ (K)', r'$\log g$', r'[M/H]']
    ndim    = 3

    rng          = np.random.default_rng(42)
    n_scatter    = min(kde_scatter_subsample, len(df))
    scatter_idx  = rng.choice(len(df), size=n_scatter, replace=False)

    fig, axes = plt.subplots(ndim, ndim, figsize=(9, 9))
    fig.suptitle('Joint KDE of exoplanet host-star properties', fontsize=13, y=1.01)

    for row in range(ndim):
        for col in range(ndim):
            ax = axes[row, col]
            if row == col:
                ax.hist(df[columns[row]], bins=n_bins,
                        range=(axes_1d[row][0], axes_1d[row][-1]), density=True,
                        color='lightgray', edgecolor='k', linewidth=0.5)
                ax.plot(axes_1d[row], marginal_1d[row], color='crimson', linewidth=1.8)
                ax.set_xlim(axes_1d[row][0], axes_1d[row][-1])
                ax.set_yticks([])
                for spine in ['top', 'left', 'right']:
                    ax.spines[spine].set_visible(False)
                if row == ndim - 1:
                    ax.set_xlabel(labels[row])
            elif row > col:
                ax.scatter(df[columns[col]].values[scatter_idx], df[columns[row]].values[scatter_idx],
                           s=4, alpha=0.25, color='gray', linewidths=0, rasterized=True)
                dens_2d = marginal_2d[(col, row)]  # shape (len(axes_1d[col]), len(axes_1d[row]))
                ax.contour(axes_1d[col], axes_1d[row], dens_2d.T, colors='crimson', linewidths=1.0)
                ax.set_xlim(axes_1d[col][0], axes_1d[col][-1])
                ax.set_ylim(axes_1d[row][0], axes_1d[row][-1])
                if col == 0:
                    ax.set_ylabel(labels[row])
                if row == ndim - 1:
                    ax.set_xlabel(labels[col])
                ax.grid(True, alpha=0.15)
            else:
                ax.set_visible(False)

    fig.tight_layout()
    fig.savefig(save_path + 'HostStar_KDE_corner.pdf', dpi=150, bbox_inches='tight')
    plt.close(fig)


################################
########## Code block 1: NASA Exoplanet Archive extraction ##########
################################

if not os.path.exists(save_data_path):
    os.makedirs(save_data_path)

if host_star_mode == 'query':
    print('Querying NASA Exoplanet Archive ...')
    host_star_df = fetch_host_star_properties()
    print(f'  {len(host_star_df)} unique host stars with Teff, logg, and metallicity')

    host_star_df.to_pickle(save_data_path + 'host_star_properties.pkl')
    print(f'  Saved host star properties -> {save_data_path}host_star_properties.pkl')

elif host_star_mode == 'load':
    print('Loading cached host star properties (host_star_mode="load") ...')
    host_star_df = pd.read_pickle(save_data_path + 'host_star_properties.pkl')
    print(f'  {len(host_star_df)} unique host stars with Teff, logg, and metallicity '
          f'(from {save_data_path}host_star_properties.pkl)')

else:
    raise ValueError('host_star_mode not recognized.')

print('Plotting histograms ...')
plot_histograms(host_star_df, save_data_path)
print(f'  Saved histograms -> {save_data_path}HostStar_histograms.pdf')


################################
########## Code block 2: joint 3D KDE fit ##########
################################

print('Fitting joint 3D KDE to (Teff, logg, [M/H]) ...')
host_star_kde = fit_joint_kde(host_star_df)

with open(save_data_path + 'host_star_kde.pkl', 'wb') as f:
    pickle.dump(host_star_kde, f)
print(f'  Saved fitted KDE -> {save_data_path}host_star_kde.pkl')

print('Evaluating KDE on a grid and marginalizing for the corner plot ...')
kde_axes_1d, kde_marginal_1d, kde_marginal_2d = evaluate_kde_grid(host_star_kde, host_star_df)

print('Plotting KDE corner plot ...')
plot_kde_corner(host_star_df, kde_axes_1d, kde_marginal_1d, kde_marginal_2d, save_data_path)
print(f'  Saved KDE corner plot -> {save_data_path}HostStar_KDE_corner.pdf')
