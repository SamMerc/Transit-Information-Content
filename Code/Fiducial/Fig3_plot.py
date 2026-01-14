#############################
########## Purpose ##########
#############################

# Figure 3 is the sensitivity analysis, showcasing the family of correlations that exist within the transit parameters.
# To generate this figure we must explore the chi-squared space of pair of parameters. To do this, we need perform an MCMC
# to identify the parts of parameter space that are most relevant to explore i.e. in the locality of the best-fit solution.
# The same parameters for the injection and retrieval are used in Figure 2 and 3 so we can just use the results of Figure 2's MCMC
# for this first step. 
# Once these chi-squared maps are evaluated, a correlation metric is calculated from their shape, and these correlation values
# are plotted in a colourful matrix.
# This file retrieves the chi-squared spaces for pairs of parameters and uses it to plots Figure 3 and 6.

######################################
########## Import libraries ##########
######################################
from jax import random
import jax
print(f"JAX devices: {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")
import jax.numpy as jnp
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.constants import G
from squishyplanet.limb_darkening_laws import nonlinear_4param_ld_law
import numpy as np
import pandas as pd
import os
import matplotlib.cm as cm

# For 64-bit precision since JAX defaults to 32-bit
jax.config.update("jax_enable_x64", True)


#############################################
########## Define hyper-parameters ##########
#############################################
#%% Mock light curve

#%%%% Define G in units needed now to avoid JAX tracing issues
G_solar_units = G.to(u.Rsun**3 / (u.Msun * u.day**2)).value
G_cgday = G.to(u.cm**3 / (u.g * u.day**2)).value
R_star = (1.0 * u.R_sun).value
#%%%% Mock system - fiducial (+ some eccentricity and argument of periastron)
init_state_dic = {}
init_state_dic['period'] = 1.                                 #days
a_meters = ( (G.value * (1.0 * u.M_sun).to(u.kg).value * (init_state_dic['period'] * 24 * 3600)**2)/(4 * jnp.pi**2) )**(1/3)  
init_state_dic['a'] = a_meters / (1.0 * u.R_sun).to(u.m).value  #stellar radius
init_state_dic['r'] = 0.1                                     #stellar radius
init_state_dic['i'] = jnp.deg2rad(90)                         #radians
init_state_dic['omega'] = 0.5                                 #radians
init_state_dic['e'] = 0.1                                     #unitless
init_state_dic['t0'] = 0.0                                    #days

#Setting base LDCs
init_NLLD_coeffs = nonlinear_4param_ld_law(u1=0.1, u2=0.2, u3=0.4, u4=0.3)

#Updating initial state dictionary
for iLD, LD_coeff in enumerate(init_NLLD_coeffs):
    init_state_dic[f'LD_u{iLD+1}'] = LD_coeff

#%%%% Calculate transit duration
# Convert angles to radians
# Impact parameter (eccentricity-corrected)
b = (
    (init_state_dic['a'] * jnp.cos(init_state_dic['i'])) / R_star
    * (1 - init_state_dic['e']**2) / (1 + init_state_dic['e'] * jnp.sin(init_state_dic['omega']))
)
# Argument inside arcsin
arg = (
    (1/init_state_dic['a'])
    * jnp.sqrt((1 + init_state_dic['r'])**2 - b**2)
    / jnp.sin(init_state_dic['i'])
)
# Numerical safety
arg = np.clip(arg, -1.0, 1.0)
# Transit duration
T_dur = (
    (init_state_dic['period'] / jnp.pi)
    * jnp.sqrt(1 - init_state_dic['e']**2) / (1 + init_state_dic['e'] * jnp.sin(init_state_dic['omega']))
    * jnp.arcsin(arg)
)

#%%%% Model time - ensure pre- and post-transit are same duration as transit
low_t = -1.5*T_dur                                                      #days
high_t = 1.5*T_dur                                                      #days
exposure_time = 5                                                       #seconds
num_t = jnp.floor((((high_t - low_t) * 24 * 3600)/exposure_time))       #number of points
init_state_dic['times'] = jnp.linspace(low_t, high_t, int(num_t))       #days


#%% Input and outputs directories
input_dir = '/Users/samsonmercier/Desktop/Work/PhD/Research/TIC/Fig2_Storage/'
output_dir = '/Users/samsonmercier/Desktop/Work/PhD/Research/TIC/Fig3_Storage/'

#%% Model parameters
mod_prop = {
    'r'         : {'vary':True, 'guess':0.11, 'bounds':[0.07, 0.15]},
    'i'         : {'vary':True, 'guess':jnp.deg2rad(88.5), 'bounds':[jnp.deg2rad(88.), jnp.deg2rad(92.)]},
    'a'         : {'vary':True, 'guess':init_state_dic['a']-1, 'bounds':[init_state_dic['a']-2, init_state_dic['a']+2]},
    'LD_u1'     : {'vary':True, 'guess':0., 'bounds':[-0.4, 0.7]},
    'LD_u2'     : {'vary':True, 'guess':0.1, 'bounds':[-0.3, 0.7]},
    'LD_u3'     : {'vary':True, 'guess':0.3, 'bounds':[-0.1, 0.9]},
    'period'    : {'vary':True, 'guess':1., 'bounds':[0.9995, 1.0005]}, #Gaussian prior
    'sqrtecosw' : {'vary':True, 'guess':0.2, 'bounds':[0., 0.4]},
    'sqrtesinw' : {'vary':True, 'guess':0.1, 'bounds':[-0.1, 0.3]},
    't0'        : {'vary':False, 'guess':0., 'bounds':[-100,100]},
}

#%% Defining important lists
var_param_list = []
for key in mod_prop:
    if mod_prop[key]['vary']:
        var_param_list.append(str(key))

#%% Defining dictionary to store additional info. needed for the model
fixed_args={}
fixed_args['var_param_list']=var_param_list
fixed_args['labels'] = [r'R$_p$/R$_{\star}$',r'i (rad)',r'$\rho_{\star}$ (g/cm$^{3}$)',r'u$_1$', r'u$_2$',r'u$_3$',r'P (days)',r'$\sqrt{e}$cos($\omega$)',r'$\sqrt{e}$sin($\omega$)']

#%% Fitting mode
fixed_args['run_mode'] = 'reuse'

#% Define number of points to sample the parameter space with
fixed_args['sample_pts'] = 5

#% Delat chi2 thresholds to include in heatmaps
fixed_args['delta_chi2_thresh'] = 0.45

#%% Number of burn-in steps used in MCMC
fixed_args['nburn'] = 70000

#%% Model scatter and seed to use for the plot
model_scatter =  16.68100537200059 
seed = 80

##############################
##### Relevant functions #####
##############################

#Helper function to fit an ellipse given points
def fit_ellipse_conic(x, y):
    # Build design matrix
    D = jnp.vstack([x**2, x*y, y**2, x, y, jnp.ones_like(x)]).T
    # Solve normal equations: minimize ||D @ p||²
    _, _, V = jnp.linalg.svd(D)
    p = V[-1]  # solution is last row of V
    return p  # [A, B, C, D, E, F]

#Helper function to calculate ellipse properties given its fit
def ellipse_parameters_from_conic(p):
    a, B, c, D, E, g = p
    b = B/2.
    d = D/2.
    f = E/2.

    #Get center values
    x0 = ( c*d - b*f )/( b*b - a*c )
    y0 = ( a*f - b*d )/( b*b - a*c )

    #Get semi-major and semi-minor axes
    up = 2.0 * (a*f*f + c*d*d + g*b*b - 2.0*b*d*f - a*c*g)
    down1 = (b*b - a*c) * (jnp.sqrt((a - c)*(a - c) + 4.0*b*b) - (a + c))                 
    down2 = (b*b - a*c) * (-jnp.sqrt((a - c)*(a - c) + 4.0*b*b) - (a + c))
    am = jnp.sqrt( up/down1 )
    bm = jnp.sqrt( up/down2 )
    
#Get angle
    if (a == c):  # Circle case
        phi = 0.0
    elif (b == 0):
        if (a < c):
            phi = 0.0
        else:  # a > c
            phi = 0.5 * jnp.pi
    else:  # b != 0 and a != c
        if (a < c):
            phi = 0.5 * jnp.arctan((2.0 * b) / (a - c))
        else:  # a > c
            phi = 0.5 * jnp.pi + 0.5 * jnp.arctan((2.0 * b) / (a - c))

    #Get cardinal points
    # x = xmid
    y_card_low = y0 - jnp.sqrt((am*am*bm*bm) / (bm*bm*jnp.sin(phi)**2 + am*am*jnp.cos(phi)**2))
    y_card_high = y0 + jnp.sqrt((am*am*bm*bm) / (bm*bm*jnp.sin(phi)**2 + am*am*jnp.cos(phi)**2))
    # y = ymid
    x_card_low = x0 - jnp.sqrt((am*am*bm*bm) / (bm*bm*jnp.cos(phi)**2 + am*am*jnp.sin(phi)**2))
    x_card_high = x0 + jnp.sqrt((am*am*bm*bm) / (bm*bm*jnp.cos(phi)**2 + am*am*jnp.sin(phi)**2))

    return {
        "center": (x0, y0),
        "angle_rad": phi,
        "angle_deg": jnp.degrees(phi),
        "semi_major": am,
        "semi_minor": bm,
        'y_card' : (y_card_low, y_card_high),
        'x_card' : (x_card_low, x_card_high),
    }


#############################################
################ Running code ###############
#############################################

fixed_args['save_loc'] = output_dir

print(f"MODEL SCATTER = {model_scatter:.2f}")
print(f"SEED = {seed}")
seed_dir = input_dir+f'{jnp.floor(model_scatter)}ppm/Seed{seed}/'


####################################
##### Chi-squared map plotting #####
####################################

#Load chi2 dictionary
chi2_dic = jnp.load(fixed_args['save_loc']+"chi2_dict.npy", allow_pickle=True).item()

#Loading the MCMC results
print(f'Retrieving MCMC')
raw_chain = jnp.load(seed_dir+"chains.npy")
logprob = jnp.load(seed_dir+"logprob.npy")

#Finding the index of max log-probability
max_step, max_walker = jnp.unravel_index(jnp.argmax(logprob), logprob.shape)

# Figure 5
print('STEP 1: FIGURE 5')

n_params = len(fixed_args['var_param_list'])

#Initializing correlation matrix
corr_matrix = np.zeros((n_params, n_params))

#Initializing figure
fig, axes = plt.subplots(n_params, n_params, figsize=(2.5 * n_params, 2.5 * n_params))

for i, param1 in enumerate(fixed_args['var_param_list']):
    for j, param2 in enumerate(fixed_args['var_param_list']):
        ax = axes[i, j]
        ax.tick_params(labelsize=14)
        print(param1, ' vs ', param2)

        # remove top/right spines for nicer look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if j > i:
            ax.axis('off')  # upper triangle: turn off
            continue

        # Diagonal: 1D chi2 line plot
        if i == j:
            param_vals = jnp.linspace(
                raw_chain[max_walker, max_step, i] - jnp.std(raw_chain[:, fixed_args['nburn'], i]),
                raw_chain[max_walker, max_step, i] + jnp.std(raw_chain[:, fixed_args['nburn'], i]),
                fixed_args['sample_pts']
            )
            chi2_vals = chi2_dic[f'{param1}_{param1}']-np.min(chi2_dic[f'{param1}_{param1}'])

            ax.plot(param_vals, chi2_vals, color='black')
            ax.axvline(raw_chain[max_walker, max_step, i], color='red', linestyle='--', lw=1)
            ax.set_yticklabels([])
            if i!=n_params - 1:ax.set_xticklabels([])
            else:ax.set_xlabel(param1, fontsize=14)

        # Lower triangle: 2D chi2 contours (filled)
        else:
            # Use param2 on x-axis, param1 on y-axis (lower triangle convention)
            param2_vals = jnp.linspace(
                raw_chain[max_walker, max_step, j] - jnp.std(raw_chain[:, fixed_args['nburn'], j]),
                raw_chain[max_walker, max_step, j] + jnp.std(raw_chain[:, fixed_args['nburn'], j]),
                fixed_args['sample_pts']
            )
            param1_vals = jnp.linspace(
                raw_chain[max_walker, max_step, i] - jnp.std(raw_chain[:, fixed_args['nburn'], i]),
                raw_chain[max_walker, max_step, i] + jnp.std(raw_chain[:, fixed_args['nburn'], i]),
                fixed_args['sample_pts']
            )

            #Get normalized arrays
            norm_param2_vals = jnp.linspace(
                - 1,
                + 1,
                fixed_args['sample_pts']
            )
            norm_param1_vals = jnp.linspace(
                - 1,
                + 1,
                fixed_args['sample_pts']
            )

            key1 = f'{param1}_{param2}'
            key2 = f'{param2}_{param1}'

            if key1 in chi2_dic:
                chi2_grid = chi2_dic[key1]
            elif key2 in chi2_dic:
                chi2_grid = chi2_dic[key2].T
            else:
                raise KeyError(f"Neither '{key1}' nor '{key2}' found in chi2_dic keys: {list(chi2_dic.keys())}")
            
            A, B = np.meshgrid(param2_vals, param1_vals)  # (x=param2, y=param1)
            norm_A, norm_B = np.meshgrid(norm_param2_vals, norm_param1_vals)  # (x=param2, y=param1)

            chi2_grid = chi2_grid - np.min(chi2_grid)

            # Define three nicely spaced contour levels (multiples of the delta threshold)
            lv1 = 1.0 * fixed_args['delta_chi2_thresh']
            lv2 = 5.0 * fixed_args['delta_chi2_thresh']
            lv3 = 10.0 * fixed_args['delta_chi2_thresh']
            lv4 = 50.0 * fixed_args['delta_chi2_thresh']
            lv5 = 100.0 * fixed_args['delta_chi2_thresh']
            contour_levels = [lv1, lv2, lv3, lv4, lv5]

            # Filled contours: 3 regions (0..lv1, lv1..lv2, lv2..lv3) with different colors
            fill_levels = [0.0, lv1, lv2, lv3, lv4, lv5]
            fill_colors = plt.get_cmap('Greens')(jnp.linspace(0., 1, 5))  # light -> dark green
            ax.contourf(A, B, chi2_grid, levels=fill_levels, colors=fill_colors, alpha=0.5, antialiased=True)

            # Outline the three contours with black lines
            cs = ax.contour(A, B, chi2_grid, levels=contour_levels, colors=['black']*len(contour_levels), linewidths=0.8)

            # Also compute normalized contours for later geometry fitting (do not rely on plotted result)
            norm_cs = ax.contour(norm_A, norm_B, chi2_grid, levels=contour_levels, colors=['none']*len(contour_levels), linewidths=0)

            ax.axvline(raw_chain[max_walker, max_step, j], color='black', linestyle='--', lw=1)
            ax.axhline(raw_chain[max_walker, max_step, i], color='black', linestyle='--', lw=1)

            # Find largest valid contour fully within bounds using the computed contour objects
            valid_contours = []

            # Get axis bounds
            x_bounds = (param2_vals[0], param2_vals[-1])
            y_bounds = (param1_vals[0], param1_vals[-1])

            for norm_contour_level, contour_level in zip(norm_cs.allsegs, cs.allsegs):
                for norm_segment, segment in zip(norm_contour_level, contour_level):
                    if len(segment) < 15:
                        continue
                    x, y = segment[:, 0], segment[:, 1]

                    # Check if entire segment lies within bounds
                    if (jnp.all((x >= x_bounds[0]) & (x <= x_bounds[1])) and
                        jnp.all((y >= y_bounds[0]) & (y <= y_bounds[1]))):
                        valid_contours.append((segment, norm_segment))

            # Select the longest valid contour
            if valid_contours:
                selected, norm_selected = max(valid_contours, key=lambda seg_pair: len(seg_pair[0]))
            else:
                # If no valid contour found, fall back to the longest available contour in cs
                all_segments = []
                for level_segs in cs.allsegs:
                    for seg in level_segs:
                        all_segments.append((seg, None))
                if len(all_segments) == 0:
                    raise KeyError('Need bigger contours')
                selected, norm_selected = max(all_segments, key=lambda seg_pair: len(seg_pair[0]))

            #Retrieving contours
            contour_x, contour_y = selected[:, 0], selected[:, 1]
            if norm_selected is None:
                # build a dummy normalized contour by scaling to unit stds if we don't have normalized seg
                norm_contour_x = (contour_x - raw_chain[max_walker, max_step, j]) / jnp.std(raw_chain[max_walker, fixed_args['nburn']:, j])
                norm_contour_y = (contour_y - raw_chain[max_walker, max_step, i]) / jnp.std(raw_chain[max_walker, fixed_args['nburn']:, i])
            else:
                norm_contour_x, norm_contour_y = norm_selected[:, 0], norm_selected[:, 1]

            #Fit the normalized contour to retrieve slope and correlation value
            ##Fit ellipse
            bestfit_norm_ellipse = fit_ellipse_conic(norm_contour_x, norm_contour_y)
            ##Retrieve properties
            ellipse_norm_params = ellipse_parameters_from_conic(bestfit_norm_ellipse)
            ##Plot semi-major axis
            norm_orig_a = ellipse_norm_params['semi_major']
            norm_orig_b = ellipse_norm_params['semi_minor']
            norm_xmid, norm_ymid = ellipse_norm_params['center']
            norm_theta = ellipse_norm_params['angle_rad']
            ##Swap axes if necessary
            if norm_orig_a < norm_orig_b:
                norm_a = norm_orig_b
                norm_theta += jnp.pi/2
            else:norm_a = norm_orig_a
            ##Plot the slope
            norm_x0, norm_x1 = norm_xmid - 1 * (norm_a*jnp.cos(norm_theta)), norm_xmid + 1 * (norm_a*jnp.cos(norm_theta))
            norm_y0, norm_y1 = norm_ymid - 1 * (norm_a*jnp.sin(norm_theta)), norm_ymid + 1 * (norm_a*jnp.sin(norm_theta))
            ##Getting correlation value
            norm_slope = (norm_y1-norm_y0)/(norm_x1-norm_x0)   
            corr_value = jnp.abs( jnp.sin(2 * jnp.arctan(norm_slope)) )     

            if jnp.isnan(corr_value):
                # fallback: fit line to all points in largest contour
                largest_idx = 0
                largest_size = 0
                for seg_idx, seg in enumerate(cs.allsegs):
                    if len(seg[0][:,0]) > largest_size:
                        largest_size = len(seg[0][:,0])
                        largest_idx = seg_idx

                x, y = cs.allsegs[largest_idx][0][:, 0], cs.allsegs[largest_idx][0][:, 1]
                slope, intercept = jnp.polyfit(x, y, 1)
                corr_value = jnp.abs(jnp.sin(2 * jnp.arctan(slope)))

            #Populate correlation matrix
            corr_matrix[i, j] = corr_value

            if i == n_params - 1:
                ax.set_xlabel(fixed_args['labels'][j],fontsize=14)
                ax.tick_params(axis="x", labelsize=12, rotation=45)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(fixed_args['labels'][i],fontsize=14)
                ax.tick_params(axis="y", labelsize=12, rotation=45)
            else:
                ax.set_yticklabels([])

            #Force the limits of the plot
            ax.set_xlim([param2_vals[0], param2_vals[-1]])
            ax.set_ylim([param1_vals[0], param1_vals[-1]])

fig.tight_layout()
fig.savefig(fixed_args['save_loc']+'Fig5.pdf')
plt.close(fig)

# Figure 3
print('STEP 2: FIGURE 3')

# Place correlation matrix on top and bottom triangle
full_corr_matrix = (corr_matrix + corr_matrix.T)
corr_df = pd.DataFrame(full_corr_matrix, index=fixed_args['labels'], columns=fixed_args['labels'])   
#Re-order the matrix
desired_order_labels = [r'R$_p$/R$_{\star}$', 'i (rad)',r'$\rho_{\star}$ (g/cm$^{3}$)', 'P (days)',r'$\sqrt{e}$cos($\omega$)',r'$\sqrt{e}$sin($\omega$)', r'u$_1$', r'u$_2$', r'u$_3$']
corr_df_reordered = corr_df.loc[desired_order_labels, desired_order_labels]


#% Make corrplot 
print('BUILD CORRELATION HEATMAP')
plot_labels = [r'D', 'i',r'$\rho_{\star}$', r'P', r'$\sqrt{e}$cos($\omega$)',r'$\sqrt{e}$sin($\omega$)', r'u$_1$', r'u$_2$', r'u$_3$']
matrix = corr_df_reordered.values
labels = plot_labels
n = len(labels)

fig, ax = plt.subplots(figsize=(20, 16))

ax.set_xlim(-0.5, n - 0.5)
ax.set_ylim(-0.5, n - 0.5)
ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(labels, ha='center', rotation = 45, fontsize=18)
ax.set_yticklabels(labels, ha='center', rotation = 45, fontsize=18)
ax.tick_params(axis='y', pad=32)
ax.invert_yaxis()
ax.set_aspect('equal')

# Define color normalization
cmap_diverging = cm.get_cmap('Blues')
max_radius = 0.4

# Draw cells
for i in range(n):
    for j in range(n):
        corr_val = matrix[i, j]
        strength = abs(corr_val)
        color = cmap_diverging(corr_val)

        if i < j:
            # Upper triangle: circle plot
            circle = plt.Circle((j, i), radius=strength * max_radius, color=color)
            ax.add_patch(circle)

        elif i > j:
            # Lower triangle: heatmap square with annotation
            rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=color, edgecolor='white')
            ax.add_patch(rect)
            ax.text(j, i, f"{corr_val:.2f}", ha='center', va='center', color='black', fontsize=18)

# Add colorbar
sm = cm.ScalarMappable(cmap=cmap_diverging)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad = -0.00005)
cbar.set_label('Correlation', fontsize=18)
cbar.ax.tick_params(labelsize=18)

# Title
plt.savefig(fixed_args['save_loc']+'Fig3.pdf')
plt.show()
