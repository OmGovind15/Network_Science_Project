import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm # Import tqdm

# --- 1. Load Data and Re-create Correlation Matrix ---
# We need the full correlation matrix to build graphs at different thresholds.
local_filename = "nifty500_adj_close_2023_2024.csv"
try:
    prices_df = pd.read_csv(local_filename, index_col=0, parse_dates=True)
except FileNotFoundError:
    print(f"Error: File '{local_filename}' not found.")
    exit()

log_returns_df = np.log(prices_df / prices_df.shift(1))
log_returns_df = log_returns_df.dropna()
corr_matrix = log_returns_df.corr(method='pearson')
N = len(corr_matrix.columns)
print(f"Loaded and processed data for {N} stocks.")

# --- 2. Conduct the Parameter Study ---
print("Running parameter study for different thresholds...")

# We will test a range of thresholds
thresholds = np.arange(0.2, 0.71, 0.02) 

# Lists to store our results
avg_degrees = []
avg_clusterings = []
largest_component_sizes = [] # This will track 'S'

# Use tqdm to wrap the loop for a progress bar
for theta in tqdm(thresholds, desc="Testing Thresholds"):
    # 1. Create the graph for this threshold
    G_theta = nx.Graph()
    G_theta.add_nodes_from(corr_matrix.columns)
    for i in range(N):
        for j in range(i + 1, N):
            if corr_matrix.iloc[i, j] >= theta:
                G_theta.add_edge(corr_matrix.columns[i], corr_matrix.columns[j])
                
    # 2. Calculate metrics
    if G_theta.number_of_nodes() == 0:
        avg_k = 0
    else:
        avg_k = (G_theta.number_of_edges() * 2) / G_theta.number_of_nodes()
    
    C = nx.average_clustering(G_theta)
    
    components = list(nx.connected_components(G_theta))
    if not components:
        S = 0
    else:
        largest_comp_size = len(max(components, key=len))
        S = largest_comp_size / N
        
    # 3. Store results
    avg_degrees.append(avg_k)
    avg_clusterings.append(C)
    largest_component_sizes.append(S)

print("Parameter study finished.")

# --- 3. Plot the results of the study (WITH FIX) ---
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
fig.suptitle("Network Properties as a Function of Correlation Threshold (θ)", fontsize=16)

# Plot 1: Average Degree
ax1.plot(thresholds, avg_degrees, 'o-', c='blue')
ax1.set_title("Average Degree <k>")
ax1.set_ylabel("Average Degree")
ax1.grid(True, ls='--', alpha=0.5)

# Plot 2: Average Clustering
ax2.plot(thresholds, avg_clusterings, 'o-', c='green')
ax2.set_title("Average Clustering Coefficient (C)")
ax2.set_ylabel("Clustering (C)")
ax2.grid(True, ls='--', alpha=0.5)

# Plot 3: Largest Component Size (S)
ax3.plot(thresholds, largest_component_sizes, 'o-', c='red')
ax3.set_title("Largest Component Size (S) - Percolation")
ax3.set_ylabel("Size (Fraction of Network)")
ax3.set_xlabel("Correlation Threshold (θ)")
ax3.grid(True, ls='--', alpha=0.5)

# --- START OF FIX ---
# Find the index of the threshold value *closest* to 0.5
theta_to_find = 0.5
# np.argmin finds the index of the minimum value
# np.abs(thresholds - theta_to_find) creates an array of differences
idx_0_5 = np.argmin(np.abs(thresholds - theta_to_find))

# Get the actual threshold and S value at that index
actual_theta = thresholds[idx_0_5]
S_at_0_5 = largest_component_sizes[idx_0_5]

# Add a vertical line for your chosen theta
ax1.axvline(x=actual_theta, color='grey', linestyle='--')
ax2.axvline(x=actual_theta, color='grey', linestyle='--')
ax3.axvline(x=actual_theta, color='grey', linestyle='--', label=f'θ ≈ {actual_theta:.2f} (S = {S_at_0_5:.2f})')
ax3.legend()
# --- END OF FIX ---

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.savefig("parameter_study_plot.png")
plt.show()

print("\n--- Parameter Study Complete ---")
print("Plots saved to 'parameter_study_plot.png'.")
print("These plots show:")
print("1. How the network gets sparser as the threshold increases.")
print("2. How clustering changes (it often increases as only tight cliques remain).")
print("3. The 'Percolation Transition' where the network shatters, explaining why your LCC was only 23.7% at theta=0.5.")