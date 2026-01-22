#############################
########## Purpose ##########
#############################

# Figure 1 showcases the amplification factor change across model scatters and limb-darkening law used (aka the dimensionality).
# In particular the grid is 14 model scatters by 3 polynomial limb-darkening laws by 10 noise seeds, i.e. 420 injection-retrievals.
# The goal of this file is to generate the directory structure needed for the storage of the results from all these retrievals.


######################################
########## Import libraries ##########
######################################
import os
import jax.numpy as jnp

##########################
########## Code ##########
##########################

#Define base directory
raw_save_dir = '/Users/samsonmercier/Desktop/Work/PhD/Research/TIC/Fig1_Storage/'


# Define the fit parameters
model_scatters = [0.1, 1, 10, 16.68100537200059, 27.825594022071243, 46.41588833612777, 77.4263682681127,
                   129.1549665014884, 215.44346900318823, 359.38136638046257, 599.4842503189409, 1000.0, 3000.0, 10000.0]
seeds = [40, 50, 60, 70, 80, 90, 100, 110, 120, 130]
PLD_orders = [2, 3, 4]


#Helper function to check the existence of directories
def check_dir(dir_name):
    if not os.path.isdir(dir_name):os.makedirs(dir_name)
    return dir_name

#Check base directory exists
check_dir(raw_save_dir)

#Check PLD order directories exist
for PLD_order in PLD_orders:
    PLD_dir = check_dir(raw_save_dir+f'PLD_{PLD_order}/')
    print(f"PLD order = {PLD_order}")

    #Check model scatter directories exist
    for model_scatter in model_scatters:
        scatter_dir = check_dir(PLD_dir+f'{jnp.floor(model_scatter)}ppm/')
        print(f"MODEL SCATTER = {model_scatter:.2f}")

        #Check seed directory exists
        for seed in seeds:
            _ = check_dir(scatter_dir+f'Seed{seed}/')
            print(f"SEED = {seed}")

