import networkx as nx
import pandas as pd
import numpy as np
import time
import pickle  # <-- ADD THIS IMPORT

# --- 1. Load Local Nifty 500 Price Data ---
local_filename = "nifty500_adj_close_2023_2024.csv"
try:
    prices_df = pd.read_csv(local_filename, index_col=0, parse_dates=True)
    print(f"Successfully loaded data from '{local_filename}'.")
    print(f"Shape of price data: {prices_df.shape}")
except FileNotFoundError:
    print(f"Error: File '{local_filename}' not found.")
    print("Please make sure the file is in the same directory as your script.")
    exit() 

# --- 2. Calculate Log-Returns ---
log_returns_df = np.log(prices_df / prices_df.shift(1))
log_returns_df = log_returns_df.dropna()
print(f"Log-returns calculated. Shape: {log_returns_df.shape}")

# --- 3. Create the Correlation Matrix ---
print("Calculating correlation matrix...")
start_time = time.time()
corr_matrix = log_returns_df.corr(method='pearson')
end_time = time.time()
print(f"Correlation matrix calculated in {end_time - start_time:.2f} seconds.")
print(f"Correlation matrix shape: {corr_matrix.shape}") 

# --- 4. Build the Unweighted Graph via Thresholding ---
theta = 0.5  
print(f"Building graph with threshold theta = {theta}...")
G = nx.Graph()
G.add_nodes_from(corr_matrix.columns) # Nodes are the stock symbols

for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        stock_i = corr_matrix.columns[i]
        stock_j = corr_matrix.columns[j]
        correlation = corr_matrix.iloc[i, j]
        if correlation >= theta:
            G.add_edge(stock_i, stock_j, weight=correlation)

print("\n--- Network Created Successfully ---")
print(f"Number of nodes (stocks): {G.number_of_nodes()}")
print(f"Number of edges (correlations >= {theta}): {G.number_of_edges()}")
try:
    avg_degree = G.number_of_edges() * 2 / G.number_of_nodes()
    print(f"Average Degree <k>: {avg_degree:.2f}")
    density = nx.density(G)
    print(f"Network Density: {density:.4f}")
except ZeroDivisionError:
    print("Error: Graph has no nodes.")

# --- 5. SAVE GRAPH (NEW ROBUST METHOD) ---
# Use Python's built-in pickle library
graph_filename = "nifty500_network.gpickle" # .gpickle extension is fine
with open(graph_filename, 'wb') as f:
    pickle.dump(G, f)

print(f"Main graph saved to '{graph_filename}' (Robust Pickle Format)")