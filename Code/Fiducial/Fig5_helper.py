#############################
########## Purpose ##########
#############################

# In this file, we retrieve all planet-hosting stars from the NASA exoplanet archive.
# From these, we build a histogram of Teff for all these stars and use this histogram 
# to weight the array of Teff values tested in Fig5. The point of this is to have 
# some inclusion of the distribution of stellar temperatures included in our grid of values tested. 

######################################
########## Import libraries ##########
######################################

import numpy as np
import matplotlib.pyplot as plt
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

########################
########## Code ########
########################

# Query the NASA Exoplanet Archive for confirmed planets
print("Querying NASA Exoplanet Archive...")
planets = NasaExoplanetArchive.query_criteria(
    table="ps",  # Planetary Systems table
    select="pl_name,hostname,st_teff",
    where="st_teff is not null"
)

print(f"Retrieved {len(planets)} planets with stellar temperature data")

# Get unique host stars only (remove duplicate systems)
unique_data = {}
for row in planets:
    hostname = row['hostname']
    st_teff = row['st_teff']
    # Extract the numerical value and handle units
    if hasattr(st_teff, 'value'):
        temp_value = st_teff.value  # Strip astropy units
    else:
        temp_value = st_teff
    
    if hostname not in unique_data and not np.isnan(temp_value):
        unique_data[hostname] = temp_value

print(f"Number of unique host stars: {len(unique_data)}")

# Extract stellar effective temperatures for unique stars
stellar_temps = np.array(list(unique_data.values()))

print(f"Valid temperature measurements: {len(stellar_temps)}")
print(f"Temperature range: {stellar_temps.min():.0f} K to {stellar_temps.max():.0f} K")
print(f"Median temperature: {np.median(stellar_temps):.0f} K")

# Create histogram
fig, ax = plt.subplots(figsize=(10, 6))
counts, bins, patches = ax.hist(stellar_temps, bins=50, color='steelblue', 
                                 edgecolor='black', alpha=0.7)

ax.set_yscale('log')
ax.set_xlabel('Stellar Effective Temperature (K)', fontsize=12)
ax.set_ylabel('Number of Exoplanets', fontsize=12)
ax.set_title('Distribution of Stellar Temperatures for Exoplanet Host Stars', 
             fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Add statistics text box
stats_text = f'Total planets: {len(stellar_temps)}\n'
stats_text += f'Median: {np.median(stellar_temps):.0f} K\n'
stats_text += f'Mean: {np.mean(stellar_temps):.0f} K'
ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        fontsize=10)

plt.tight_layout()
plt.show()

# Sample 10 temperatures weighted by the histogram distribution
# This gives more samples from temperature ranges with more planets
print("\n" + "="*60)
print("Generating weighted temperature array...")
print("="*60)

# Sample from the actual data (each planet counts equally)
np.random.seed(42)  # For reproducibility
sampled_temps = np.random.choice(stellar_temps, size=10, replace=False)
sampled_temps = np.sort(sampled_temps)

print("\n10 Temperature values sampled from the distribution:")
print("-" * 60)
for i, temp in enumerate(sampled_temps, 1):
    print(f"T{i:2d} = {temp:6.1f} K")

print("\nArray for your code:")
print(f"temperatures = {list(sampled_temps.round(1))}")

# Alternative: Create a more evenly spaced sample using percentiles
# This ensures coverage across the distribution
print("\n" + "="*60)
print("Alternative: Percentile-based sampling")
print("="*60)

percentiles = np.linspace(5, 95, 10)  # Avoid extreme edges
percentile_temps = np.percentile(stellar_temps, percentiles)

print("\n10 Temperature values based on distribution percentiles:")
print("-" * 60)
for i, (p, temp) in enumerate(zip(percentiles, percentile_temps), 1):
    print(f"T{i:2d} = {temp:6.1f} K  (at {p:.0f}th percentile)")

print("\nArray for your code:")
print(f"temperatures = {list(percentile_temps.round(1))}")

# Show which approach gives better coverage
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Random sampling
ax1.hist(stellar_temps, bins=50, color='steelblue', alpha=0.5, 
         edgecolor='black', label='All planets')
ax1.scatter(sampled_temps, np.zeros_like(sampled_temps), 
           color='red', s=100, zorder=5, marker='^', 
           label='Random samples')
# ax1.set_yscale('log')
ax1.set_xlabel('Stellar Effective Temperature (K)', fontsize=11)
ax1.set_ylabel('Number of Exoplanets', fontsize=11)
ax1.set_title('Random Weighted Sampling', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Plot 2: Percentile sampling
ax2.hist(stellar_temps, bins=50, color='steelblue', alpha=0.5, 
         edgecolor='black', label='All planets')
ax2.scatter(percentile_temps, np.zeros_like(percentile_temps), 
           color='darkgreen', s=100, zorder=5, marker='^', 
           label='Percentile samples')
# ax2.set_yscale('log')
ax2.set_xlabel('Stellar Effective Temperature (K)', fontsize=11)
ax2.set_ylabel('Number of Exoplanets', fontsize=11)
ax2.set_title('Percentile-Based Sampling', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("Note: The percentile-based approach ensures better coverage")
print("across the temperature distribution while still being weighted")
print("by the number of planets at each temperature range.")
print("="*60)