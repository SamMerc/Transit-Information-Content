#############################
########## Purpose ##########
#############################

# Figure 1 showcases the amplification factor change across model scatters and limb-darkening law used (aka the dimensionality).
# The goal of this file is to run injection-retrievals for a grid of 1. model scatters 2. noise seeds and 3. limb-darkening laws
# In particular the grid is 14 model scatters by 3 polynomial limb-darkening laws by 10 noise seeds, i.e. 420 injection-retrievals.


######################################
########## Import libraries ##########
######################################
from jax import random
import jax
print(f"JAX devices: {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")
import jax.numpy as jnp
import emcee_jax
import matplotlib.pyplot as plt
from jaxoplanet.orbits.keplerian import System, Central
import astropy.units as u
from astropy.constants import G
from jaxoplanet.light_curves import limb_dark_light_curve
import corner
import time
import arviz as az
import numpy as np
import os, itertools, sys
import numpyro
from numpyro.distributions import Normal, Uniform
from numpyro.infer import MCMC, NUTS, HMC, init_to_value

# For multi-core parallelism (useful when running multiple MCMC chains in parallel)
numpyro.set_host_device_count(1)

# For CPU (use "gpu" for GPU)
numpyro.set_platform("cpu")

# For 64-bit precision since JAX defaults to 32-bit
jax.config.update("jax_enable_x64", True)

# Set random seed
jaxnoise_key = jax.random.PRNGKey(0)

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

#%% Storing outputs of nested sampling and plots
raw_save_dir = '/Users/samsonmercier/Desktop/Work/PhD/Research/TIC/Fig1_Storage/'

#%% Model parameters
mod_prop = {
    'r'         : {'vary':True, 'guess':0.11, 'bounds':[0.07, 0.15]},
    'i'         : {'vary':True, 'guess':jnp.deg2rad(88.5), 'bounds':[jnp.deg2rad(88.), jnp.deg2rad(92.)]},
    'a'         : {'vary':True, 'guess':init_state_dic['a']-1, 'bounds':[init_state_dic['a']-2, init_state_dic['a']+2]},
    'period'    : {'vary':True, 'guess':1., 'bounds':[0.9995, 1.0005]}, #Gaussian prior
    'sqrtecosw' : {'vary':True, 'guess': 0., 'bounds': [-0.2, 0.2]},
    'sqrtesinw' : {'vary':True, 'guess': 0., 'bounds': [-0.2, 0.2]},
    't0'        : {'vary':False, 'guess':0., 'bounds':[-100,100]},
}

#%% Priors
priors_dic = {
    'r'             : {'type':'uf', 'bounds':[0., 1.]},
    'i'             : {'type':'uf', 'bounds':[jnp.deg2rad(70), jnp.deg2rad(110)]},
    'a'             : {'type':'uf', 'bounds':[0, 50]},
    'period'        : {'type':'gauss', 'val':1.0000, 's_val':0.0005},
    'sqrtecosw'     : {'type':'uf', 'bounds':[-1, 1]},
    'sqrtesinw'     : {'type':'uf', 'bounds':[-1, 1]},
}

#%% Fitting mode
fixed_args={}
fixed_args['run_mode'] = 'use'
fixed_args['nthreads'] = jax.device_count()

#%% MCMC specific settings
fixed_args['nwalkers'] = 50
fixed_args['nsteps'] = 1000
fixed_args['nburn'] = 700
fixed_args['kernel'] = 'emcee' # 'emcee', 'HMC' or 'NUTS'
#%% Numpyro specific - set to False for faster
fixed_args['adapt_step_size'] = True
fixed_args['dense_matrix'] = True
fixed_args['regularize_mass_matrix'] = True

#############################################
########## Define parrallelization ##########
#############################################
# Define the fit parameters
model_scatters = [0.1, 1, 10, 16.68100537200059, 27.825594022071243, 46.41588833612777, 77.4263682681127,
                   129.1549665014884, 215.44346900318823, 359.38136638046257, 599.4842503189409, 1000.0, 3000.0, 10000.0]
seeds = [40, 50, 60, 70, 80, 90, 100, 110, 120, 130]
PLD_orders = [2, 3, 4]

# # Distribute tasks - for HPC usage
# task_arrays = [model_scatters, seeds, PLD_orders]
# param_combos = list(itertools.product(*task_arrays))
# param_combos = [list(c) for c in param_combos]

# my_task_id = int(sys.argv[1])
# model_scatter, seed, PLD_order = param_combos[my_task_id - 1]

model_scatter = 129.1549665014884
seed = 50
PLD_order = 2


############################################
########## Setting limb-darkening ##########
############################################

if PLD_order == 2:
    #Setting base LDCs
    init_PLD_coeffs = [0.1, 0.2]

    #Updating initial state dictionary
    for iLD, LD_coeff in enumerate(init_PLD_coeffs):
        init_state_dic[f'LD_u{iLD+1}'] = LD_coeff

    #Updating fit dictionaries
    mod_prop.update(
        {
            'LD_u1'     : {'vary':True, 'guess':0., 'bounds':[-0.4, 0.6]},
            'LD_u2'     : {'vary':True, 'guess':0.1, 'bounds':[-0.3, 0.7]},
        }
    )

    #Updating priors
    priors_dic.update(
        {
            'LD_u1'         : {'type':'uf', 'bounds':[-100, 100]},
            'LD_u2'         : {'type':'uf', 'bounds':[-100, 100]},
        }
    )

    #Define plotting labels
    fixed_args['labels'] = ["RpR$_*$", 'i', "aR$_*$", "P", "$\sqrt{e}\cos{\omega}$", "$\sqrt{e}\sin{\omega}$", "", "u$_1$", "u$_2$"]

elif PLD_order == 3:
    #Setting base LDCs
    init_PLD_coeffs = [0.1, 0.2, 0.4]

    #Updating initial state dictionary
    for iLD, LD_coeff in enumerate(init_PLD_coeffs):
        init_state_dic[f'LD_u{iLD+1}'] = LD_coeff

    #Updating fit dictionaries
    mod_prop.update(
        {
            'LD_u1'     : {'vary':True, 'guess':0., 'bounds':[-0.4, 0.6]},
            'LD_u2'     : {'vary':True, 'guess':0.1, 'bounds':[-0.3, 0.7]},
            'LD_u3'     : {'vary':True, 'guess':0.3, 'bounds':[-0.1, 0.9]},
        }
    )

    #Updating priors
    priors_dic.update(
        {
            'LD_u1'         : {'type':'uf', 'bounds':[-100, 100]},
            'LD_u2'         : {'type':'uf', 'bounds':[-100, 100]},
            'LD_u3'         : {'type':'uf', 'bounds':[-100, 100]},
        }
    )

    #Define plotting labels
    fixed_args['labels'] = ["RpR$_*$", 'i', "aR$_*$", "P", "$\sqrt{e}\cos{\omega}$", "$\sqrt{e}\sin{\omega}$", "", "u$_1$", "u$_2$", "u$_3$"]

elif PLD_order == 4:
    #Setting base LDCs
    init_PLD_coeffs = [0.1, 0.2, 0.4, 0.3]

    #Updating initial state dictionary
    for iLD, LD_coeff in enumerate(init_PLD_coeffs):
        init_state_dic[f'LD_u{iLD+1}'] = LD_coeff

    #Updating fit dictionaries
    mod_prop.update(
        {
            'LD_u1'     : {'vary':True, 'guess':0., 'bounds':[-0.4, 0.6]},
            'LD_u2'     : {'vary':True, 'guess':0.1, 'bounds':[-0.3, 0.7]},
            'LD_u3'     : {'vary':True, 'guess':0.3, 'bounds':[-0.1, 0.9]},
            'LD_u4'     : {'vary':True, 'guess':0.2, 'bounds':[-0.2, 0.8]},
        }
    )

    #Updating priors
    priors_dic.update(
        {
            'LD_u1'         : {'type':'uf', 'bounds':[-100, 100]},
            'LD_u2'         : {'type':'uf', 'bounds':[-100, 100]},
            'LD_u3'         : {'type':'uf', 'bounds':[-100, 100]},
            'LD_u4'         : {'type':'uf', 'bounds':[-100, 100]},
        }
    )

    #Define plotting labels
    fixed_args['labels'] = ["RpR$_*$", 'i', "aR$_*$", "P", "$\sqrt{e}\cos{\omega}$", "$\sqrt{e}\sin{\omega}$", "", "u$_1$", "u$_2$", "u$_3$", "u$_4$"]

else:
    raise KeyError('Wrong order of polynomial limb darkening law.')


#######################################
##### Finalizing hyper-parameters #####
#######################################

#%% Defining important lists
ndim=0
var_param_list = []
fix_param_list = []
fix_param_val = []
for key in mod_prop:
    if mod_prop[key]['vary']:
        var_param_list.append(str(key))
        ndim+=1
    else:
        fix_param_list.append(str(key))
        fix_param_val.append(mod_prop[key]['guess'])

#%% Defining dictionary to store additional info. needed for the model
fixed_args['priors_dic']=priors_dic
fixed_args['var_param_list']=var_param_list
fixed_args['fix_param_list']=fix_param_list
fixed_args['fix_param_val']=fix_param_val
fixed_args['ndim']=ndim

#Uniform distribution of walker starting positions
if fixed_args['kernel'] == 'emcee':
    fixed_args['pos'] = np.zeros((fixed_args['nwalkers'], fixed_args['ndim']), dtype=float)
    for i in range(fixed_args['ndim']):
        fixed_args['pos'][:, i] = jax.random.uniform(jaxnoise_key, minval=mod_prop[var_param_list[i]]['bounds'][0], maxval=mod_prop[var_param_list[i]]['bounds'][1], shape=(fixed_args['nwalkers'],))
else:
    fixed_args['pos'] = {}
    for i in range(fixed_args['ndim']):
        fixed_args['pos'][var_param_list[i]] = jnp.array(mod_prop[var_param_list[i]]['guess'])


##############################
##### Relevant functions #####
##############################
#Helper function to check the existence of directories
def check_dir(dir_name):
    if not os.path.isdir(dir_name):os.makedirs(dir_name)
    return dir_name

def create_jaxoplanet_model(x, p):
    
    #Retrieving ecc and w
    if ('sqrtecosw' in p) and ('sqrtesinw' in p):
        ecc = p['sqrtecosw']**2 + p['sqrtesinw']**2
        w = jnp.arccos(p['sqrtecosw']/jnp.sqrt(ecc))
    elif ('e' in p) and ('omega' in p):
        ecc = p['e']
        w = p['omega']
    else:
        ecc = 0.
        w = 0.

    #Retrieving inclination
    if 'cosi' in p:
        inc = jnp.arccos(p['cosi'])
    else:
        inc = p['i']

    #Define star
    stellar_rho =  (3 * jnp.pi * p['a']**3)/ ( p['period']**2 * G_solar_units )
    star = Central(density=stellar_rho)

    #Define planet
    planet = System(star).add_body(
        time_transit = p['t0'],
        period = p['period'],
        inclination = inc,
        eccentricity = ecc,
        omega_peri = w, 
        radius = (p['r'] * R_star),
    )

    #Apply limb-darkening
    max_coeff = len([param for param in p if 'LD' in param])
    ld_u_coeffs = jnp.array([p[f"LD_u{i}"] for i in range(1, max_coeff+1)])

    jaxo_lc = 1.0 + limb_dark_light_curve(planet, ld_u_coeffs)(x)
    return jaxo_lc.reshape((-1))


def fit_func_numpyro(x, y, yerr):

    p_step = []
    lp = 0.0

    #% Defining the priors
    for param in fixed_args['var_param_list']:

        #Parameter prior properties
        prior=fixed_args['priors_dic'][param]

        #Uniform prior
        if prior['type']=='uf':
            sample_val = numpyro.sample( param, Uniform( jnp.array(prior['bounds'][0]), jnp.array(prior['bounds'][1]) ) )
            lp += -jnp.log(prior['bounds'][1] - prior['bounds'][0])
        #Gaussian prior
        elif prior['type']=='gauss':
            sample_val = numpyro.sample( param, Normal( jnp.array(prior['val']), jnp.array(prior['s_val']) ) )
            lp += -0.5 * (jnp.log(2*jnp.pi*prior['s_val']**2) + ((sample_val - prior['val'])/prior['s_val'])**2)
        #Undefined prior
        else:raise KeyError(f"Undefined prior type for '{param}'")

        #Appending values
        p_step.append(sample_val)

    #% Formatting the dictionary of parameters
    p_step_all={}
    for parname in fixed_args['var_param_list']:
        p_step_all[parname] = p_step[fixed_args['var_param_list'].index(parname)]
    for parname in fixed_args['fix_param_list']:
        p_step_all[parname] = fixed_args['fix_param_val'][fixed_args['fix_param_list'].index(parname)]
    
    #% Generate predicted LC
    y_pred = create_jaxoplanet_model(x, p_step_all)

    # The likelihood function assuming Gaussian uncertainty
    numpyro.sample("obs", Normal(y_pred, yerr), obs=y)

    #% Keep track of chi2
    step_chi2 = jnp.sum( (y_pred - y)**2/yerr**2 )
    numpyro.deterministic("step_chi2", jnp.array(step_chi2))

    #% Keep track of log-probability 
    lk = -0.5 * (step_chi2 + jnp.sum(jnp.log(2*jnp.pi*yerr**2)))
    numpyro.deterministic("logprob", lk + lp)



#Prior function
def emcee_log_prior(param_dict):

    p=param_dict.copy()
    
    ln_p = 0.
    
    #Restricting sqrtecosw and sqrtesinw to physical values
    if ('sqrtecosw' in p) and ('sqrtesinw' in p):
        ecc = p['sqrtecosw']**2 + p['sqrtesinw']**2
        ln_p += jnp.where(ecc <= 1.0, 0.0, -jnp.inf)
    
    if ((('sqrtecosw' in p) and ('sqrtecosw' not in fixed_args['priors_dic'])) or (('sqrtesinw' in p) and ('sqrtesinw' not in fixed_args['priors_dic']))) and (('e' in fixed_args['priors_dic']) and ('w' in fixed_args['priors_dic'])):
        #Get the values
        ecc = p['sqrtecosw']**2 + p['sqrtesinw']**2
        w = jnp.rad2deg(jnp.arccos(p['sqrtecosw']/jnp.sqrt(ecc)))

        #Get prior info.
        prior_e = fixed_args['priors_dic']['e']
        prior_w = fixed_args['priors_dic']['omega']

        #Update log-prior
        for param_prior, param_val in zip([prior_e,prior_w], [ecc,w]):
            if param_prior['type']=='uf':
                in_bounds = (param_val >= param_prior['bounds'][0]) & (param_val <= param_prior['bounds'][1])
                ln_p += jnp.where(
                    in_bounds,
                    -jnp.log(param_prior['bounds'][1] - param_prior['bounds'][0]),
                    -jnp.inf,
                )
            
            #Gaussian prior
            elif param_prior['type']=='gauss':
                ln_p += - 0.5*(jnp.log(2.*jnp.pi*param_prior['s_val']**2.) + ( (param_val - param_prior['val'])/param_prior['s_val']  )**2.)        
            
            else:raise KeyError(f"Undefined prior type for '{param}'")

        #Remove the concerned parameter from p
        if (('sqrtecosw' in p) and ('sqrtecosw' not in fixed_args['priors_dic'])):p.pop('sqrtecosw')
        if (('sqrtesinw' in p) and ('sqrtesinw' not in fixed_args['priors_dic'])):p.pop('sqrtesinw')

    #Going through parameters
    for param in p:

        #Looking only at variable parameters
        if param not in fixed_args['fix_param_list']:
            
            #Check if parameter's priors have been defined
            if param not in fixed_args['priors_dic']:
                raise KeyError(f"Parameter '{param}' is missing priors.")
            
            prior = fixed_args['priors_dic'][param]
            #Uniform prior
            if prior['type']=='uf':
                in_bounds = (p[param] >= prior['bounds'][0]) & (p[param] <= prior['bounds'][1])
                ln_p += jnp.where(
                    in_bounds,
                    -jnp.log(prior['bounds'][1] - prior['bounds'][0]),
                    -jnp.inf,
                )
            
            #Gaussian prior
            elif prior['type']=='gauss':
                ln_p += - 0.5*(jnp.log(2.*jnp.pi*prior['s_val']**2.) + ( (p[param] - prior['val'])/prior['s_val']  )**2.)        
            
            else:raise KeyError(f"Undefined prior type for '{param}'")
    
    return ln_p

#Posterior function
def emcee_log_probability(p_step, x, y, yerr):
    
    #Format p_step into what is required for further functions
    p = {}
    for ipar, parname in enumerate(fixed_args['var_param_list']):
        p[parname] = p_step[ipar]
    for ipar, parname in enumerate(fixed_args['fix_param_list']):
        p[parname] = fixed_args['fix_param_val'][ipar]
    
    #Log-prior
    lp = emcee_log_prior(p)
    lp = jnp.where(jnp.isfinite(lp), lp, -jnp.inf)

    #Log-likelihood
    y_pred = create_jaxoplanet_model(x, p)
    step_chi2 = jnp.sum( (y_pred - y)**2/yerr**2 )
    lk = -0.5 * (step_chi2 + jnp.sum(jnp.log(2*jnp.pi*yerr**2)))
    lk = jnp.where(jnp.isnan(lk), -jnp.inf, lk)

    return lp + lk, {'step_chi2': step_chi2}

#############################################
################ Running code ###############
#############################################
#Check directory exists
check_dir(raw_save_dir)

#Check model scatter directory exists
scatter_dir = check_dir(raw_save_dir+f'{jnp.floor(model_scatter)}ppm/')
print(f"MODEL SCATTER = {model_scatter:.2f}")

#Check seed directory exists
fixed_args['save_loc'] = check_dir(scatter_dir+f'Seed{seed}/')
print(f"SEED = {seed}")

#############################
####### Generate data #######
#############################
print('GENERATING DATA')

#Pure data
true_lc = create_jaxoplanet_model(init_state_dic['times'], init_state_dic)

#Build noisy data
std = model_scatter * 1e-6
noisy_LC = true_lc + std * random.normal(jax.random.PRNGKey(seed), shape=true_lc.shape)
noisy_std = std * jnp.ones(true_lc.shape, dtype=float)

# evaluate this likelihood
print(f"initial chi2: {jnp.sum( (true_lc - noisy_LC)**2/noisy_std**2 )}, initial chi2: {-0.5* ( jnp.sum( (true_lc - noisy_LC)**2/noisy_std**2 ) + jnp.sum(jnp.log(2*jnp.pi*noisy_std**2)) ) }")

#Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=[10, 6], sharex=True, gridspec_kw={'height_ratios': [3, 1]})
ax1.errorbar(init_state_dic['times'], noisy_LC, yerr=noisy_std, fmt='.', zorder=1)
ax1.plot(init_state_dic['times'], true_lc, color='red', zorder=2)
ax2.errorbar(init_state_dic['times'], 1e6*(noisy_LC - true_lc), yerr=noisy_std, fmt='r.', zorder=1)
for ax in [ax1, ax2]:
    ax.axvline(-0.5 * T_dur, color='black', linestyle='dashed')
    ax.axvline(0.5 * T_dur, color='black', linestyle='dashed')
    ax.axvline(-1.5 * T_dur, color='black', linestyle='dotted')
    ax.axvline(1.5 * T_dur, color='black', linestyle='dotted')
ax1.set_title('Model LC with %.f ppm scatter'%model_scatter)
ax2.set_xlabel('Time (BJD)')
ax1.set_ylabel('Flux')
ax2.set_ylabel('Difference (ppm)')
fig.tight_layout()
plt.savefig(fixed_args['save_loc']+'init_guess.pdf')
plt.close()



#########################
##### Emcee fitting #####
#########################

# Running the MCMC
if fixed_args['run_mode']=='use':
    print(f"Running MCMC") 

    if fixed_args['kernel']=='emcee':
        st0 = time.time()
        emceejax_key1, emceejax_key2 = jax.random.split(jaxnoise_key, 2)

        sampler = emcee_jax.EnsembleSampler(emcee_log_probability, log_prob_args=(init_state_dic['times'],noisy_LC,noisy_std))
        state = sampler.init(emceejax_key1, fixed_args['pos'])
        if fixed_args['nthreads']>1:
            trace = sampler.sample_parallel(emceejax_key2, state, num_steps=fixed_args['nsteps'], progress=True)
        else:
            trace = sampler.sample(emceejax_key2, state, num_steps=fixed_args['nsteps'], progress=True)
        raw_chain = np.asarray(trace.samples.coordinates).reshape(fixed_args['nwalkers'],fixed_args['nsteps'],fixed_args['ndim'])  # shape (walkers, nsteps, nparams)
        logprob = np.asarray(trace.samples.log_probability.T) # shape (nwalkers, nsteps)
        chi2_chain = np.asarray(trace.samples.deterministics['step_chi2'].T) # shape (nwalkers, nsteps)

        #Convert to ArviZ data structure
        inf_data = az.from_dict(
        posterior={
            p: raw_chain[:, fixed_args['nburn']:, i]
            for i, p in enumerate(fixed_args['var_param_list'])
        },
        log_likelihood={"log_like": logprob},
        )

    else:
        st0 = time.time()
        #Define the kernel
        if fixed_args['kernel'] == 'NUTS':
            kernel = NUTS(fit_func_numpyro, 
                            dense_mass=fixed_args['dense_matrix'],
                            regularize_mass_matrix=fixed_args['regularize_mass_matrix'],
                            adapt_step_size=fixed_args['adapt_step_size'],
                            init_strategy=init_to_value(values=fixed_args['pos']))
        elif fixed_args['kernel'] == 'HMC':
            kernel = HMC(fit_func_numpyro, 
                            dense_mass=fixed_args['dense_matrix'],
                            regularize_mass_matrix=fixed_args['regularize_mass_matrix'],
                            adapt_step_size=fixed_args['adapt_step_size'],
                            init_strategy=init_to_value(values=fixed_args['pos']))
        else:print('Invalid kernel specified')
        
        # Define the MCMC sampler
        mcmc = MCMC(kernel, num_warmup=fixed_args['nburn'], num_samples=fixed_args['nsteps']-fixed_args['nburn'], num_chains=fixed_args['nwalkers'], progress_bar=True)
        
        #Run the MCMC
        mcmc.run(random.PRNGKey(0), init_state_dic['times'], noisy_LC, noisy_std)

        #Convert to ArviZ data structure
        inf_data = az.from_numpyro(mcmc)
        #Remove unneeded variables
        inf_data.posterior = inf_data.posterior.drop_vars(
            ["step_chi2","logprob"]
        )

        #Walkers chain
        #     - pyro_chains is of shape (nwalkers, nsteps, n_free)
        #     - parameters have the same order as in 'initial_distribution' and 'var_param_list'
        pyro_chains = mcmc.get_samples(group_by_chain=True)
        
        # Collect parameter arrays in consistent order
        raw_chains = []
        for p in fixed_args['var_param_list']:
            raw_chains.append(pyro_chains[p][..., None])  # add trailing dim for stacking
        # Stack into shape (nwalkers, nprodsteps, nparams)
        raw_chain = jnp.concatenate(raw_chains, axis=-1)

        #%Retrieve step-by-step chi2
        chi2_chain = pyro_chains['step_chi2']

        #%Retrieve log-probability
        logprob =  pyro_chains['logprob']

        fixed_args['nsteps'] = fixed_args['nsteps'] - fixed_args['nburn']
        fixed_args['nburn'] = 0  


    #%% Storing the chains
    output_file = fixed_args['save_loc']+"chains.npy"
    jnp.save(output_file, raw_chain)

    #%% Storing the log probability
    output_file = fixed_args['save_loc']+"logprob.npy"
    jnp.save(output_file, logprob)

    #%% Storing the chi2 values
    output_file = fixed_args['save_loc']+"chi2_chain.npy"
    jnp.save(output_file, chi2_chain) 

elif fixed_args['run_mode']=='reuse':
    print(f'Retrieving MCMC')
    raw_chain = jnp.load(fixed_args['save_loc']+"chains.npy")
    logprob = jnp.load(fixed_args['save_loc']+"logprob.npy")
    chi2_chain = jnp.load(fixed_args['save_loc']+"chi2_chain.npy", allow_pickle=True)

    # Convert to ArviZ data structure
    inf_data = az.from_dict(
    posterior={
        p: raw_chain[:, fixed_args['nburn']:, i]
        for i, p in enumerate(fixed_args['var_param_list'])
    },
    log_likelihood={"log_like": logprob},
    )

else:
    raise KeyError('Invalid run mode')

##################
#### Plotting ####
##################
if fixed_args['run_mode']=='use':
    elapsed = time.time() - st0
    print(f'MCMC took {elapsed:.2f} seconds / {elapsed/60.:.2f} minutes.')

print('PLOTTING')

# 0. Plotting the chi2 chains
print('STEP 0: CHI2 CHAINS')
plt.figure(figsize=[12, 6])
for iwalk in range(fixed_args['nwalkers']):
    plt.loglog(jnp.arange(fixed_args['nsteps']), chi2_chain[iwalk, :])
plt.xlabel('Steps')
plt.ylabel('Chi-squared')
plt.savefig(fixed_args['save_loc']+'chi2.pdf')
plt.close()

# 1. Plotting the log probability
print('STEP 1: LOG PROB')
plt.figure(figsize=[12, 6])
for w in range(fixed_args['nwalkers']):
    plt.loglog(jnp.arange(fixed_args['nburn']), -logprob[w, :fixed_args['nburn']], color='red', alpha=0.5, lw=0.7, zorder=1)
    plt.loglog(jnp.arange(fixed_args['nburn'], fixed_args['nsteps']), -logprob[w, fixed_args['nburn']:], color='blue', alpha=0.5, lw=0.7, zorder=1)
plt.ylabel('Log-probability')
plt.xlabel("Step")
plt.savefig(fixed_args['save_loc']+'logprob.pdf')
plt.close()

# 2. Plot the trace of each parameter's chains
print('STEP 2: TRACE')
_ = az.plot_trace(
    inf_data,
    var_names=fixed_args['var_param_list'],
    backend_kwargs={"constrained_layout": True},
)
plt.savefig(fixed_args['save_loc']+'trace.pdf')
plt.close()

# 3. Plot the cornerplot of the chains
print('STEP 3: CORNER')
#% Build truth dictionary
truth_dic={}
for param,parlabel in zip(fixed_args['var_param_list'],fixed_args['labels']): 
    if ('LD' in param):
        truth_dic[parlabel] = init_state_dic[param]
    elif param=='sqrtecosw':
        truth_dic[parlabel] = jnp.sqrt(init_state_dic['e'])*jnp.cos(init_state_dic['omega'])
    elif param=='sqrtesinw':
        truth_dic[parlabel] = jnp.sqrt(init_state_dic['e'])*jnp.sin(init_state_dic['omega'])
    elif param=='cosi':
        truth_dic[parlabel] = jnp.cos(init_state_dic['i'])
    else:
        truth_dic[parlabel] = init_state_dic[param]
truth_list = [truth_dic.get(label, None) for label in fixed_args['labels']]

#% Plot
fig1 = corner.corner(
    inf_data,
    labels=fixed_args['labels'],
    show_titles=True,
    title_kwargs={"fontsize": 10},
    label_kwargs={"fontsize": 10},
    title_fmt=".4f",
    truths = truth_list,
)
#Find bestfit parameters
max_walker, max_step = jnp.unravel_index(jnp.argmax(logprob), logprob.shape)
best_fit_params = raw_chain[max_walker, max_step, :]

# Flatten the chain
medians = jnp.median(raw_chain.reshape(-1, raw_chain.shape[-1]), axis=0)

# Dynamically determine how many parameters were plotted
flat_chain = raw_chain.reshape(-1, raw_chain.shape[-1])
nparams = flat_chain.shape[1]
axes = np.array(fig1.axes).reshape((nparams, nparams))

# Add median lines
for i in range(nparams):
    axes[i, i].axvline(medians[i], color='red', linestyle='--')
    axes[i, i].axvline(best_fit_params[i], color='yellow', linestyle='--')
    for j in range(i):
        axes[i, j].axvline(medians[j], color='red', linestyle='--')
        axes[i, j].axhline(medians[i], color='red', linestyle='--')
        axes[i, j].scatter(medians[j], medians[i], color='red', s=20, zorder=10)

        axes[i, j].axvline(best_fit_params[j], color='yellow', linestyle='--')
        axes[i, j].axhline(best_fit_params[i], color='yellow', linestyle='--')
        axes[i, j].scatter(best_fit_params[j], best_fit_params[i], color='yellow', s=20, zorder=10)

plt.savefig(fixed_args['save_loc']+'corner.pdf')
plt.close()

# 4. Plot the best-fit and median models against the data
print('STEP 4: BESTFIT LC')
# Compute median parameter values
median_params = {}
for i, param in enumerate(fixed_args['var_param_list']):
    median_params[param] = jnp.median(raw_chain[:, fixed_args['nburn']:, i])
for idx, param in enumerate(fixed_args['fix_param_list']):
    median_params[param] = fixed_args['fix_param_val'][idx]

# Compute median model
median_lc = create_jaxoplanet_model(init_state_dic['times'], median_params)
median_RMS = jnp.sqrt(jnp.average((noisy_LC - median_lc)**2))

# Get highest log probability parameters
best_params = {}
for i, param in enumerate(fixed_args['var_param_list']):
    best_params[param] = raw_chain[max_walker, max_step, i]
for idx, param in enumerate(fixed_args['fix_param_list']):
    best_params[param] = fixed_args['fix_param_val'][idx]

# Compute bestfit model
bestfit_lc = create_jaxoplanet_model(init_state_dic['times'], best_params)
bestfit_RMS = jnp.sqrt(jnp.average((noisy_LC - bestfit_lc)**2))

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios':[3,1]}, figsize=(12, 6))
ax1.errorbar(init_state_dic['times'], noisy_LC, yerr=noisy_std, fmt='.', label='Data', color='0.7')
ax1.plot(init_state_dic['times'], median_lc, color='orange', label='Median model')
ax1.plot(init_state_dic['times'], bestfit_lc, color='blue', label='Best-fit model')
ax2.errorbar(init_state_dic['times'], 1e6*(median_lc - noisy_LC), yerr=noisy_std, fmt='.', color='orange', label=f'RMS = {1e6*median_RMS:.0f} ppm')
ax2.errorbar(init_state_dic['times'], 1e6*(bestfit_lc - noisy_LC), yerr=noisy_std, fmt='.', color='blue', label=f'RMS = {1e6*bestfit_RMS:.0f} ppm')
ax2.set_xlabel("Time [days]")
ax1.set_ylabel("Relative flux")
ax2.set_ylabel("Residuals (ppm)")
ax1.legend()
ax2.legend()
plt.tight_layout()
plt.savefig(fixed_args['save_loc']+'bestfit.pdf')
plt.close()

# 5. Compute the amplification factor between the retrieved transit depth error and the expected error
print('STEP 5: AMP FACTOR')       
#% Get chain of radius ratio values
r_chain = raw_chain[:, fixed_args['nburn']:, fixed_args['var_param_list'].index('r')]
#% Get bestfit and median values
median_r = jnp.median(r_chain)
bestfit_r = raw_chain[max_walker, max_step, fixed_args['var_param_list'].index('r')]
#% Get error on median and bestfit values
median_r_error = 2*jnp.std(r_chain)*median_r
bestfit_r_error = 2*jnp.std(r_chain)*bestfit_r
#% Get the number of in transit points
T14 = (init_state_dic['period']/jnp.pi) * jnp.arcsin( (1/init_state_dic['a']) * ( jnp.sqrt( (1+init_state_dic['r'])**2 - (init_state_dic['a'] * jnp.cos(init_state_dic['i']) ) ) / jnp.sin(init_state_dic['i']) ) ) * ( jnp.sqrt( 1 - init_state_dic['e']**2 ) / ( 1 + init_state_dic['e']*jnp.sin(init_state_dic['omega']) ) )
num_IT_pts = jnp.sum(((init_state_dic['times'] > init_state_dic['t0'] - T14/2) & (init_state_dic['times'] < init_state_dic['t0'] + T14/2)))
scatter_in_bin = (model_scatter*1e-6)/jnp.sqrt(num_IT_pts)
#% Get the scatter in an out-of-transit bin

#% Print results
print(f'Median radius ratio: {median_r} +/- {jnp.std(r_chain)}. Median amplification factor: {median_r_error/scatter_in_bin}')
print(f'Bestfit radius ratio: {bestfit_r} +/- {jnp.std(r_chain)}. Bestfit amplification factor: {bestfit_r_error/scatter_in_bin}')