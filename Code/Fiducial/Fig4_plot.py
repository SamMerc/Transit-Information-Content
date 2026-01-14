#############################
########## Purpose ##########
#############################

# Figure 4 showcases the amplification factor change across limb-darkening laws and priors used.
# In particular the grid is 7 polynomial limb-darkening laws by 5 limb darkening priors by 10 noise seeds, i.e. 350 injection-retrievals.
# In these tests, the injected LC is done with a 4th order non-linear LDL, while the retrieval is done with polynomial LDLs going from 2-9 and 4NLLD.
# The goal of this file is to retrieve results from all the MCMC runs and compile them into Figure 4.


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
import numpy as np
import matplotlib.gridspec as gridspec


# For 64-bit precision since JAX defaults to 32-bit
jax.config.update("jax_enable_x64", True)


#############################################
########## Define hyper-parameters ##########
#############################################

#%% Mock light curve

#%%%% Define G in units needed now to avoid JAX tracing issues
G_solar_units = G.to(u.Rsun**3 / (u.Msun * u.day**2)).value
R_star = (1.0 * u.R_sun).value
#%%%% Mock system - fiducial
init_state_dic = {}
init_state_dic['period'] = 1.                                 #days
a_meters = ( (G.value * (1.0 * u.M_sun).to(u.kg).value * (init_state_dic['period'] * 24 * 3600)**2)/(4 * jnp.pi**2) )**(1/3)  
init_state_dic['a'] = a_meters / (1.0 * u.R_sun).to(u.m).value  #stellar radius
init_state_dic['r'] = 0.1                                     #stellar radius
init_state_dic['i'] = jnp.deg2rad(90)                         #radians
init_state_dic['omega'] = 0.0                                 #radians
init_state_dic['e'] = 0.                                      #unitless
init_state_dic['t0'] = 0.0                                    #days


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

#%% Location of MCMC results and where plot will be output
raw_save_dir = '/Users/samsonmercier/Desktop/Work/PhD/Research/TIC/Fig4_Storage/'

#%% Set model scatter
model_scatter = 16.68100537200059

# Define the fit parameters
LDLs = ['PLD_2','PLD_3','PLD_4','PLD_5','PLD_6','PLD_9','4NLLD']
prior_strengths = ['uniform', 'gauss_20', 'gauss_10', 'gauss_5', 'gauss_1']
prior_strengths_labels = ['Uniform', '20\% Gaussian','10\% Gaussian', '5\% Gaussian', '1\% Gaussian']
seeds = [40, 50, 60, 70, 80, 90, 100, 110, 120, 130]

#%% Number of burn-in steps
fixed_args = {}
fixed_args['nburn'] = 70000

#############################################
################ Running code ###############
#############################################

#Initialize storage arrays
bestfit_amp_array = jnp.zeros( (len(LDLs), len(prior_strengths), len(seeds)) , dtype=float)
bestfit_percent_bias_array = jnp.zeros( (len(LDLs), len(prior_strengths), len(seeds)) , dtype=float)
bestfit_sigma_bias_array = jnp.zeros( (len(LDLs), len(prior_strengths), len(seeds)) , dtype=float)

#Populate arrays

#Loop over limb-darkening laws
for iLDL, LDL in enumerate(LDLs):

    #Loop over prior strengths 
    for ips, prior_strength in enumerate(prior_strengths):

        #Loop over noise seeds
        for iseed, seed in enumerate(seeds):

            #Load the MCMC results
            raw_chain = jnp.load(raw_save_dir+f'{jnp.floor(model_scatter)}ppm/Seed{seed}/chains.npy')
            logprob = jnp.load(raw_save_dir+f'{jnp.floor(model_scatter)}ppm/Seed{seed}/logprob.npy')
            max_step, max_walker = jnp.unravel_index(jnp.argmax(logprob), logprob.shape)

            #Retrieve the amplification factor
            r_chain = raw_chain[:, fixed_args['nburn']:, 0]
            bestfit_r = raw_chain[max_walker, max_step, 0]
            bestfit_r_error = 2*jnp.std(r_chain)*bestfit_r
            num_IT_pts = jnp.sum(((init_state_dic['times'] > init_state_dic['t0'] - T_dur/2) & (init_state_dic['times'] < init_state_dic['t0'] + T_dur/2)))
            scatter_in_bin = (model_scatter*1e-6)/jnp.sqrt(num_IT_pts)
            bestfit_amp_array[iLDL, ips, iseed] = bestfit_r_error/scatter_in_bin

            #Retrieve the bias
            bestfit_percent_bias_array[iLDL, ips, iseed] = 100*(bestfit_r**2 - init_state_dic['r']**2)/init_state_dic['r']**2
            bestfit_sigma_bias_array[iLDL, ips, iseed] = (bestfit_r**2 - init_state_dic['r']**2)/bestfit_r_error


####################
##### Plotting #####
####################

#Plotting Figure 3 right hand panel
LDCs_plot = [2, 3, 4, 5, 6, 7, 8]

# Create broken-axis figure: three columns (left: 2-6, middle: 9 & right: 4NLLD) and two rows (top: amp, bottom: bias)

fig = plt.figure(figsize=(13, 6))
gs = gridspec.GridSpec(2, 3, width_ratios=[3.5, 1, 1], height_ratios=[1, 1], wspace=0.05, hspace=0.05)

# left column axes (wide) - amplification (top) and bias (bottom)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])

# middle column axes (narrow) - amplification (top) and bias (bottom)
ax1m = fig.add_subplot(gs[0, 1])
ax2m = fig.add_subplot(gs[1, 1])

# right column axes (narrow) - amplification (top) and bias (bottom)
ax1r = fig.add_subplot(gs[0, 2])
ax2r = fig.add_subplot(gs[1, 2])

# define which LDC indices go to left and right panels
left_LDLs = ['PLD_2','PLD_3','PLD_4','PLD_5','PLD_6']   # correspond to LDCs 2..6
middle_LDLs = ['PLD_9']
right_LDLs = ['4NLLD']           # correspond to LDCs 9 and '4NLLD'

positions_left = np.array([2, 3, 4, 5, 6])
positions_middle = np.array([9])
positions_right = np.array([10])  # arbitrary separated positions for the small/right panel

# Iterating over prior strengths
for ips, (prior_strength, prior_strength_label) in enumerate(zip(prior_strengths, prior_strengths_labels)):
    
    #Define opacity for prior strength
    prior_strength_alpha = 1.0 - (0.2 * ips)

    #Iterating over the orientations - top and bottom
    for orientation_axes, boxplot_content in zip([[ax1, ax1m, ax1r],[ax2, ax2m, ax2r]], [bestfit_amp_array, bestfit_sigma_bias_array]):
    
        #Iterating over the top panels
        for panel_ax, xpositions, panel_LDLs in zip(orientation_axes, [positions_left, positions_middle, positions_right], [left_LDLs,middle_LDLs,right_LDLs]):
        
            #Iterating over limb-darkening laws
            for iLDL, LDL in enumerate(panel_LDLs):
                pos = xpositions[iLDL] - 0.16 + (0.08 * ips)
                if prior_strength == 'gauss_20':
                    boxcolor = 'blue'
                elif prior_strength == 'gauss_10':
                    boxcolor = 'royalblue'
                else:
                    boxcolor = 'black'
                panel_ax.boxplot(boxplot_content[iLDL, ips, :], positions=[pos], patch_artist=True,
                            boxprops=dict(facecolor=boxcolor, color=boxcolor, alpha=prior_strength_alpha), widths=[0.1],
                            medianprops=dict(color='gold', linewidth=1, alpha=prior_strength_alpha),
                            whiskerprops=dict(color=boxcolor, linewidth=1.5, alpha=prior_strength_alpha),
                            capprops=dict(color=boxcolor, linewidth=1.5, alpha=prior_strength_alpha),
                            flierprops=dict(marker='o', color=boxcolor, markersize=5, alpha=prior_strength_alpha),
                            showfliers=False,
                            label=prior_strength_label if iLDL==0 else "")

# Apply scales, labels and limits
ax1.set_yscale('log')
ax1m.set_yscale('log')
ax1r.set_yscale('log')
ax2.set_yscale('log')
ax2m.set_yscale('log')
ax2r.set_yscale('log')

# Force same limits on the left and right axes
ax1.set_ylim([1., 120])
ax1m.set_ylim([1., 120])
ax1r.set_ylim([1., 120])
ax2.set_ylim([0.1, 120])
ax2m.set_ylim([0.1, 120])
ax2r.set_ylim([0.1, 120])

ax1.set_xlim([1.5, 6.5])
ax2.set_xlim([1.5, 6.5])
ax1m.set_xlim([8.5, 9.5])
ax2m.set_xlim([8.5, 9.5])
ax1r.set_xlim([9.5, 10.5])
ax2r.set_xlim([9.5, 10.5])

# Remove y ticks and labels on right-side axes
ax1r.set_yticks([])
ax2r.set_yticks([])
ax1r.tick_params(axis='y', which='both', left=False, labelleft=False)
ax2r.tick_params(axis='y', which='both', left=False, labelleft=False)
ax1m.set_yticks([])
ax2m.set_yticks([])
ax1m.tick_params(axis='y', which='both', left=False, labelleft=False)
ax2m.tick_params(axis='y', which='both', left=False, labelleft=False)

# xticks and labels for left and right panels
ax2.set_xticks(positions_left)
ax2.set_xticklabels([2, 3, 4, 5, 6], color='red')
ax2.get_xticklabels()[1].set_color('green')

ax1.set_xticks(positions_left)
ax1.set_xticklabels([])

ax2m.set_xticks(positions_middle)
ax2m.set_xticklabels([9], color='red')

ax1m.set_xticks(positions_middle)
ax1m.set_xticklabels([])

ax2r.set_xticks(positions_right)
ax2r.set_xticklabels(['4NLLD'], color='green')

ax1r.set_xticks(positions_right)
ax1r.set_xticklabels([])

# y ticks (only on left/bottom axis)
ax2.set_yticks([0.1, 1, 10, 100])
ax2.set_yticklabels([0.1, 1, 10, 100])

ax1.set_yticks([10, 100])
ax1.set_yticklabels([10, 100])

# y labels and legends (keep legends on the top-left wide axis)
ax2.set_ylabel(r'Transit Depth Bias ($\sigma$)', fontsize=12, color='black')
ax1.set_ylabel(r'Amplification Factor ($A$)', fontsize=12, color='black')
ax1.tick_params(axis='y', labelcolor='black')
ax1.legend(fontsize=12, loc='upper left', ncols=2)

# cosmetic: hide spines between left and right axes and draw diagonal break marks
ax1.spines['right'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax1r.spines['left'].set_visible(False)
ax2r.spines['left'].set_visible(False)
ax1m.spines['left'].set_visible(False)
ax2m.spines['left'].set_visible(False)
ax1m.spines['right'].set_visible(False)
ax2m.spines['right'].set_visible(False)

# Draw matched diagonal break marks using figure coordinates so they
# appear the same size regardless of differing axis width ratios.
from matplotlib.lines import Line2D

def draw_matched_breaks(fig, left_ax, right_ax, size=0.012, lw=1.0, color='k', positive_slope=True):
    # Get axis bounding boxes in figure coordinates
    lb = left_ax.get_position()
    rb = right_ax.get_position()

    # central y positions for the top and bottom slashes (average to avoid tiny misalign)
    y_top = 0.5 * (lb.y1 + rb.y1)
    y_bot = 0.5 * (lb.y0 + rb.y0)

    # x positions at the interior edges of each axis (figure coords)
    xL = lb.x1
    xR = rb.x0

    # Choose orientation:
    # positive_slope=True  -> diagonals go from bottom-left to top-right ("/")
    # positive_slope=False -> diagonals go from top-left to bottom-right ("\")
    if positive_slope:
        # left top
        fig.add_artist(Line2D([xL - size, xL + size], [y_top - size, y_top + size],
                                transform=fig.transFigure, color=color, lw=lw, clip_on=False))
        # left bottom
        fig.add_artist(Line2D([xL - size, xL + size], [y_bot - size, y_bot + size],
                                transform=fig.transFigure, color=color, lw=lw, clip_on=False))
        # right top
        fig.add_artist(Line2D([xR - size, xR + size], [y_top - size, y_top + size],
                                transform=fig.transFigure, color=color, lw=lw, clip_on=False))
        # right bottom
        fig.add_artist(Line2D([xR - size, xR + size], [y_bot - size, y_bot + size],
                                transform=fig.transFigure, color=color, lw=lw, clip_on=False))
    else:
        # left top
        fig.add_artist(Line2D([xL - size, xL + size], [y_top + size, y_top - size],
                                transform=fig.transFigure, color=color, lw=lw, clip_on=False))
        # left bottom
        fig.add_artist(Line2D([xL - size, xL + size], [y_bot + size, y_bot - size],
                                transform=fig.transFigure, color=color, lw=lw, clip_on=False))
        # right top
        fig.add_artist(Line2D([xR - size, xR + size], [y_top + size, y_top - size],
                                transform=fig.transFigure, color=color, lw=lw, clip_on=False))
        # right bottom
        fig.add_artist(Line2D([xR - size, xR + size], [y_bot + size, y_bot - size],
                                transform=fig.transFigure, color=color, lw=lw, clip_on=False))

# draw matched breaks for both rows
draw_matched_breaks(fig, ax1, ax1m, size=0.006, lw=1.0, color='k')
draw_matched_breaks(fig, ax2, ax2m, size=0.006, lw=1.0, color='k')
draw_matched_breaks(fig, ax1m, ax1r, size=0.006, lw=1.0, color='k')
draw_matched_breaks(fig, ax2m, ax2r, size=0.006, lw=1.0, color='k')

# Add a translucent green band on the bottom subplots (from y=0.1 to y=2).
# Use zorder < 0 so the band sits below gridlines (which use zorder=0 above),
# but still below the boxplot artists (which have zorder >= 1 by default).
band_kwargs = dict(facecolor='green', alpha=0.2, edgecolor='none', zorder=-1)
ax2.axhspan(0.1, 2.0, **band_kwargs)
ax2m.axhspan(0.1, 2.0, **band_kwargs)
ax2r.axhspan(0.1, 2.0, **band_kwargs)

ax1.axhspan(1, 10.0, **band_kwargs)
ax1m.axhspan(1, 10.0, **band_kwargs)
ax1r.axhspan(1, 10.0, **band_kwargs)

# horizontal reference
ax1.axhline(np.sqrt(3), linestyle='dashed', color='black')
ax1m.axhline(np.sqrt(3), linestyle='dashed', color='black')
ax1r.axhline(np.sqrt(3), linestyle='dashed', color='black')
ax1.text(1.6, np.sqrt(3) - 0.65, r'Theoretical limit @ $\sqrt{3}$', fontsize=12, color='black')
ax1.text(1.6, 6, r'Acceptable $A$', fontsize=12, color='seagreen')
ax2.text(1.6, 0.9, f'No bias', fontsize=12, color='seagreen')

# Place a single xlabel centered across the full bottom row (left + right axes).
# Compute figure-coordinate x position halfway between the left edge of the left axis
# and the right edge of the right axis, and place the text slightly below the axes.
left_box = ax2.get_position()
right_box = ax2r.get_position()
x_center = 0.5 * (left_box.x0 + right_box.x1)
# choose a y position slightly below the lower edge of the bottom axes,
# but keep it inside the figure (not negative).
y_label = min(left_box.y0, right_box.y0) - 0.04
if y_label < 0.01:
    y_label = 0.01
fig.text(x_center, y_label, 'Number of LDCs', ha='center', va='top', fontsize=12)

# Grids 
ax1.axhline(10, color='0.8', zorder=0)
ax1.axhline(100, color='0.8', zorder=0)
for pos in positions_left:ax1.axvline(pos, color='0.8', zorder=0)
ax1m.axhline(10, color='0.8', zorder=0)
ax1m.axhline(100, color='0.8', zorder=0)
for pos in positions_middle:ax1m.axvline(pos, color='0.8', zorder=0)
ax1r.axhline(10, color='0.8', zorder=0)
ax1r.axhline(100, color='0.8', zorder=0)
for pos in positions_right:ax1r.axvline(pos, color='0.8', zorder=0)

ax2.axhline(0.1, color='0.8', zorder=0)
ax2.axhline(1, color='0.8', zorder=0)
ax2.axhline(10, color='0.8', zorder=0)
ax2.axhline(100, color='0.8', zorder=0)
for pos in positions_left:ax2.axvline(pos, color='0.8', zorder=0)
ax2m.axhline(0.1, color='0.8', zorder=0)
ax2m.axhline(1, color='0.8', zorder=0)
ax2m.axhline(10, color='0.8', zorder=0)
ax2m.axhline(100, color='0.8', zorder=0)
for pos in positions_middle:ax2m.axvline(pos, color='0.8', zorder=0)
ax2r.axhline(0.1, color='0.8', zorder=0)
ax2r.axhline(1, color='0.8', zorder=0)
ax2r.axhline(10, color='0.8', zorder=0)
ax2r.axhline(100, color='0.8', zorder=0)
for pos in positions_right:ax2r.axvline(pos, color='0.8', zorder=0)

plt.savefig(raw_save_dir+'Fig4.pdf')
plt.close()