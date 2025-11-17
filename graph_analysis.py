import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import collections
import time
import pickle  # <-- ADD THIS IMPORT

# --- 0. Load the Graph (NEW ROBUST METHOD) ---
# We now load the .gpickle file using pickle
graph_filename = "nifty500_network.gpickle"
try:
    with open(graph_filename, 'rb') as f:
        G = pickle.load(f)
    
    print(f"Successfully loaded graph from '{graph_filename}'")
    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

except FileNotFoundError:
    print(f"Error: File '{graph_filename}' not found.")
    print("Please make sure you have run the network construction script first.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the graph: {e}")
    exit()


N = G.number_of_nodes()
M = G.number_of_edges()

# Helper function to print top 10 nodes from a centrality dictionary
def print_top_nodes(centrality_dict, name):
    """Sorts and prints the top 10 nodes from a centrality dict."""
    print(f"\n--- Top 10 Stocks by {name} ---")
    sorted_centrality = sorted(centrality_dict.items(), key=lambda item: item[1], reverse=True)
    
    nodes_to_print = min(10, len(sorted_centrality))
    
    if nodes_to_print == 0:
        print("No nodes to display.")
        return

    for i in range(nodes_to_print):
        stock, score = sorted_centrality[i]
        # Format for alignment: 2 chars for rank, 12 for stock name, 6 for score
        print(f"{i+1:2}. {stock:<12} (Score: {score:.4f})")


# --- 3.1. Degree Distribution (Chapter 8.3, 8.4) ---
print("\n--- 3.1. Degree Distribution (Ch 8.3, 8.4) ---")

# Get the list of degrees for all nodes
degrees = [G.degree(n) for n in G.nodes()]

if not degrees:
    print("Graph has no nodes or degrees to analyze.")
else:
    # Count the frequency of each degree
    degree_counts = collections.Counter(degrees)
    deg, cnt = zip(*sorted(degree_counts.items()))

    # Calculate the probability P(k)
    cnt = np.array(cnt)
    probabilities = cnt / N

    # --- Plot 1: Standard Histogram ---
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.bar(deg, probabilities)
    plt.title("Degree Distribution P(k)")
    plt.xlabel("Degree (k)")
    plt.ylabel("Probability P(k)")

    # --- Plot 2: Log-Log Plot ---
    plt.subplot(1, 2, 2)
    plt.loglog(deg, probabilities, 'o', markersize=4) # 'o' for scatter plot
    plt.title("Degree Distribution (Log-Log Plot)")
    plt.xlabel("Degree (k) [log scale]")
    plt.ylabel("P(k) [log scale]")
    plt.grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("degree_distribution_nifty500.png")
    plt.show()

    print("Log-Log plot saved as 'degree_distribution_nifty500.png'.")
    print("Interpretation: A straight line on the log-log plot suggests a power-law (scale-free) distribution.")
    print("Your plot will likely show a sharp drop-off at the end, which is a common finite-size effect.")


# --- 3.2. Clustering & Random Graph Comparison (Chapter 7.9, 8.6, 12) ---
print("\n--- 3.2. Clustering & G(n,p) Comparison (Ch 7.9, 8.6, 12) ---")

# Calculate the average clustering coefficient for your stock network
C_stock = nx.average_clustering(G)
print(f"Stock Network Average Clustering (C): {C_stock:.4f}")

# --- Create the equivalent G(n, p) Random Graph ---
# Calculate the edge probability 'p'
max_edges = (N * (N - 1)) / 2
p = M / max_edges  # This is the 'p' for G(n,p)

# The expected clustering for a G(n,p) graph is just p
C_random_theory = p
print(f"Equivalent G(n,p) Random Graph Clustering (C ≈ p): {C_random_theory:.4f}")

if C_random_theory > 0:
    print(f"\nInterpretation: Your stock network's clustering is {C_stock/C_random_theory:.1f} times higher than a random graph.")
    print("This indicates high transitivity (cliques/groups), a key feature of real-world networks (like social networks).")
else:
    print("Random graph has no clustering, comparison is not applicable.")


# --- 3.3. Centrality Scores (Chapter 7) ---
print("\n--- 3.3. Centrality Analysis (Ch 7) ---")

# 1. Degree Centrality (Ch 7.1)
# Interpretation: Stocks correlated with the highest *number* of other stocks.
degree_cent = nx.degree_centrality(G)
print_top_nodes(degree_cent, "Degree Centrality")

# 2. Betweenness Centrality (Ch 7.7)
# Interpretation: Stocks that act as "brokers" or "bridges" between different 
# sectors or clusters. They lie on many shortest paths.
print("\nCalculating Betweenness Centrality (this may take a few seconds)...")
start_time = time.time()
betweenness_cent = nx.betweenness_centrality(G, normalized=True)
end_time = time.time()
print(f"Done in {end_time - start_time:.2f} seconds.")
print_top_nodes(betweenness_cent, "Betweenness Centrality")

# 3. Eigenvector Centrality (Ch 7.2)
# Interpretation: Stocks connected to *other* highly connected/influential stocks.
# This measure is best used on a single connected component.
print("\nCalculating Eigenvector Centrality...")

# Find the largest connected component (LCC)
components = list(nx.connected_components(G))
if components:
    largest_comp_nodes = max(components, key=len)
    G_giant = G.subgraph(largest_comp_nodes)
    print(f"Largest component has {G_giant.number_of_nodes()} nodes ({(G_giant.number_of_nodes()/N*100):.1f}% of network).")
    
    try:
        # We run this on the LCC, as it's not well-defined on disconnected graphs
        eigenvector_cent = nx.eigenvector_centrality(G_giant, max_iter=1000, tol=1.0e-8)
        print_top_nodes(eigenvector_cent, "Eigenvector Centrality (on Largest Component)")
    except nx.PowerIterationFailedConvergence:
        print("\nEigenvector Centrality did not converge. This can happen, but is less likely on the LCC.")
else:
    print("Graph is empty or fully disconnected. Skipping Eigenvector Centrality.")


# --- 3.4. Assortative Mixing (Chapter 7.13, 8.7) ---
print("\n--- 3.4. Assortative Mixing (Ch 7.13, 8.7) ---")

# Calculate the degree assortativity coefficient
r = nx.degree_assortativity_coefficient(G)
print(f"Degree Assortativity Coefficient (r): {r:.4f}")

if r > 0.05:
    print("Interpretation: The network is ASSORTATIVE (r > 0).")
    print("This suggests a 'rich-club' phenomenon, where high-degree stocks (hubs) are highly correlated with other high-degree stocks.")
elif r < -0.05:
    print("Interpretation: The network is DISASSORTATIVE (r < 0).")
    print("This suggests that high-degree stocks (hubs) tend to connect to low-degree (peripheral) stocks.")
else:
    print("Interpretation: The network is not strongly assortative or disassortative (r ≈ 0).")

print("\n--- Network Analysis Complete ---")