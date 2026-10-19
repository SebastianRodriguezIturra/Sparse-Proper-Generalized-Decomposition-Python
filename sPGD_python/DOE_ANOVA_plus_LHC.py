import numpy as np
from scipy.stats import qmc
import pdb


# Position [-3, 3] for each x,y,z for 2 faces.
# Orientations [-1.57, 1.57] angle axis on x,y,z for 2 faces.




# 1. Definition of parameter bounds
# Using a list of tuples (min, max)

if 1:

    param_names = ['x_left', 'y_left', 'z_left', 'rx_left', 'ry_left', 'rz_left', 'x_right', 'y_right', 'z_right', 'rx_right', 'ry_right', 'rz_right']
    param_bounds = [
        (-3, 3),   # x_left
        (-3, 3),  # y_left
        (-3, 3),   # z_left
        (-1.57, 1.57),  # rx_left
        (-1.57, 1.57),  # ry_left
        (-1.57, 1.57),  # rz_left
        (-3, 3),   # x_right
        (-3, 3),  # y_right
        (-3, 3),   # z_right
        (-1.57, 1.57),  # rx_right
        (-1.57, 1.57),  # ry_right
        (-1.57, 1.57)   # rz_right
    ]
else:

    param_names = ['x_left', 'y_left']
    param_bounds = [
        (-3, 3),   # x_left
        (-3, 3)  # y_left
    ]


param_bounds = np.array(param_bounds)
n_param = len(param_bounds)




# --- SECTION 2: OAT / ANOVA INITIAL SAMPLING ---
# Step 1: Calculate the center point
centers = param_bounds.mean(axis=1)

# Step 2: Generate points for Min and Max of each parameter while others stay at center
oat_points = [centers] # Start with the center point
for i in range(n_param):
    # Add point with param[i] at Min
    p_min = centers.copy()
    p_min[i] = param_bounds[i, 0]
    oat_points.append(p_min)
    
    # Add point with param[i] at Max
    p_max = centers.copy()
    p_max[i] = param_bounds[i, 1]
    oat_points.append(p_max)

param_anova = np.array(oat_points)

# --- SECTION 3: LATIN HYPERCUBE DEFINITION ---
#n_data_by_param = 25
#n_lhs_data = n_data_by_param * n_param
n_lhs_data = 25

# Initialize the LHS sampler (standard Latin Hypercube)
#sampler = qmc.LatinHypercube(d=n_param)
#sample = sampler.random(n=n_lhs_data)

# 'random-cd' optimizes the sample to be more spread out
sampler = qmc.LatinHypercube(d=n_param, optimization="random-cd")
sample = sampler.random(n=n_lhs_data)

# Scale the samples from [0, 1] to the actual bounds
l_bounds = param_bounds[:, 0]
u_bounds = param_bounds[:, 1]
param_lhs = qmc.scale(sample, l_bounds, u_bounds)

# --- SECTION 4: MERGING AND FORMATTING ---
# Combine the ANOVA points and LHS points
# Transposing at the end to match the MATLAB 'param' output (params as rows)
param_total = np.vstack((param_anova, param_lhs)).T

print(f"Total points generated: {param_total.shape[1]}")
print(f"Shape of param matrix (Rows=Params, Cols=Samples): {param_total.shape}")



import matplotlib.pyplot as plt

# --- SECTION 5: PLOTTING (Only for 2D) ---
if n_param == 2:
    plt.figure(figsize=(8, 8))
    
    # Plot LHS points (the blue cloud)
    # We slice [:, 0] and [:, 1] because param_lhs is (N_samples, N_params)
    plt.scatter(param_lhs[:, 0], param_lhs[:, 1], 
                color='royalblue', label='LHS Points', alpha=0.7, edgecolors='k')

    # Plot ANOVA / OAT points (the red star/cross)
    plt.scatter(param_anova[:, 0], param_anova[:, 1], 
                color='crimson', label='ANOVA (Center/Min/Max)', s=100, marker='X', edgecolors='white')

    # Add labels and styling
    plt.title('Sampling Cloud: ANOVA vs. Latin Hypercube (2D)')
    plt.xlabel(param_names[0])
    plt.ylabel(param_names[1])
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Add lines for the axes crossing at the center
    plt.axhline(centers[1], color='gray', linewidth=0.5, linestyle=':')
    plt.axvline(centers[0], color='gray', linewidth=0.5, linestyle=':')
    
    plt.legend()
    plt.show()
else:
    print(f"Skipping plot: Current case is {n_param}D. Plotting is configured for 2D.")




# Save the data on a Excel file:

import pandas as pd

# ... (rest of your existing sampling code) ...

# --- SECTION 5: EXCEL EXPORT ---

# Create a DataFrame where:
# - Rows are the parameter names
# - Columns are the individual samples (Sample 1, Sample 2, etc.)
# Use param_total.shape[1] to get the number of columns (30)
df = pd.DataFrame(
    param_total, 
    index=param_names, 
    columns=[f'Sample_{i+1}' for i in range(param_total.shape[1])]
)

# Save to Excel
filename = 'parameter_sampling_results.xlsx'
df.to_excel(filename, sheet_name='Sampling_Data')

print(f"Data successfully saved to {filename}")