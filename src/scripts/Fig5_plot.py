#############################
########## Purpose ##########
#############################

# Figure 5 shows the intensity profiles and NLLD fits for the overall mode and
# selected cluster modes identified in the coefficient space. The underlying data
# (profiles, coefficients, cluster labels) are produced by Fig5_run.py and stored
# in results.npz, which is downloaded from Zenodo by the Snakemake workflow.
#
# This version works exclusively with global (disc-integrated) stellar intensity
# profiles — no transit chord / impact-parameter / planet-size dependence.


######################################
########## Import libraries ##########
######################################

import numpy as np
import matplotlib
import paths
matplotlib.use('Agg')
import matplotlib.pyplot as plt


######################################
########## Hyper-parameters ##########
######################################

input_save_path = str(paths.data / "Fig5_Storage") + "/"
models = ['mps1']  # ['phoenix','kurucz', 'stagger', 'mps1', 'mps2']


############################
###### Function block ######
############################

def fourNLLD(x, coeffs):
    """4th-order non-linear limb-darkening law."""
    return (1
            - coeffs[0] * (1 - x ** 0.5)
            - coeffs[1] * (1 - x)
            - coeffs[2] * (1 - x ** 1.5)
            - coeffs[3] * (1 - x ** 2))


################################
########## Code block ##########
################################

for model in models:

    # ── Load pre-computed results ─────────────────────────────────────────────
    res = np.load(input_save_path + f'{model}/results.npz', allow_pickle=False)

    unique_cl             = res['unique_cl']
    typical_mu            = res['typical_mu']
    typical_profile       = res['typical_profile']
    typical_coeff         = res['typical_coeff']
    cluster_mode_mus      = res['cluster_mode_mus']
    cluster_mode_profiles = res['cluster_mode_profiles']
    cluster_mode_coeffs   = res['cluster_mode_coeffs']

    n_cl           = len(unique_cl)
    cluster_cmap   = plt.cm.get_cmap('tab10', n_cl)
    cluster_colors = [cluster_cmap(c) for c in range(n_cl)]

    # ── Figure 5: NLLD curves for overall mode and per-cluster mode profiles ──
    print('    FIGURE 5 - NLLD curves for overall mode and per-cluster mode profiles')

    # Bundle specials: (label, mu array, raw profile, coeff vector, colour, linewidth)
    special_styles = [
        ('Overall mode', typical_mu, typical_profile, typical_coeff, 'orange', 2.5),
    ] + [
        (f'Cluster {cl} mode', cluster_mode_mus[ci], cluster_mode_profiles[ci], cluster_mode_coeffs[ci], cluster_colors[ci], 1.8)
        for ci, cl in enumerate(unique_cl)
    ]

    # ── Panel layout: left = NLLD curves, right = residuals vs overall mode ──
    n_cl_plot = 4
    cl_to_plot = ['Overall mode', 'Cluster 1 mode', 'Cluster 3 mode', 'Cluster 4 mode']
    special_styles_to_plot = [s for s in special_styles if s[0] in cl_to_plot]

    fig5, ax5 = plt.subplots(
        2, n_cl_plot, figsize=(5 * n_cl_plot, 7),
        gridspec_kw={'wspace': 0.15, 'hspace': 0.05},
        sharex=True
    )

    for plot_idx, (sp_label, mu_plot, prof, coeffs, col, lw) in enumerate(special_styles_to_plot):
        curve = fourNLLD(mu_plot, coeffs)

        # Left panel: raw intensity profile (solid) + NLLD fit (dashed)
        ax5[0, plot_idx].plot(mu_plot, curve, '--', color='black', linewidth=lw, zorder=2, label='4th-order NLLD fit' if plot_idx == 0 else None)
        ax5[0, plot_idx].plot(mu_plot, prof,  '-',  color=col,     linewidth=lw, zorder=1)
        ax5[0, plot_idx].set_title(sp_label, fontsize=11, pad=3)
        ax5[0, plot_idx].grid(True)
        if plot_idx==0:ax5[0, plot_idx].legend(fontsize=12, loc='lower right')

        # Right panel: residuals
        resid = 100 * (curve - prof) / prof
        ax5[1, plot_idx].plot(mu_plot, resid, '--', color=col, linewidth=lw, label=sp_label)
        ax5[1, plot_idx].axhline(0, color='black', linestyle='-', linewidth=1.2, alpha=0.4)
        ax5[1, plot_idx].grid(True)

        ax5[1, plot_idx].set_xlabel('$\\mu = \\cos(\\theta)$', fontsize=12)

    ax5[0, 0].set_ylabel('Normalised intensity', fontsize=12)
    ax5[1, 0].set_ylabel('Residuals (%)', fontsize=12)

    plt.savefig(paths.figures / "Fig5.pdf", bbox_inches="tight")

    # Print a summary table of all coefficients for easy copy-paste
    print(f'\n  {"Profile":<14}  {"c1":>8}  {"c2":>8}  {"c3":>8}  {"c4":>8}')
    print(f'  {"-"*54}')
    for sp_label, _mu, _prof, coeffs, _col, _lw in special_styles:
        print(f'  {sp_label:<20}  '
              f'{coeffs[0]:>8.4f}  {coeffs[1]:>8.4f}  '
              f'{coeffs[2]:>8.4f}  {coeffs[3]:>8.4f}')
