import networkx as nx
import pandas as pd
import numpy as np
import pickle
import requests
import io

# --- 1. Load Stock-to-Sector Mapping ---
csv_url = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
}
print("Loading sector information...")
try:
    response = requests.get(csv_url, headers=headers)
    response.raise_for_status() 
    csv_data = io.StringIO(response.text)
    nifty500_df = pd.read_csv(csv_data)
    
    # Create a simple dictionary to map Symbol -> Industry
    # We add .NS to the symbol to match our graph nodes
    # --- THIS IS THE FIX ---
    sector_map = {row['Symbol'] + ".NS": row['Industry'] for index, row in nifty500_df.iterrows()}
    # --- END OF FIX ---
    
    print("Sector mapping created.")
except Exception as e:
    print(f"Could not load sector data: {e}")
    sector_map = {} # Continue with an empty map

# --- 2. Load the Graph ---
graph_filename = "nifty500_network.gpickle"
try:
    with open(graph_filename, 'rb') as f:
        G = pickle.load(f)
    print(f"Successfully loaded graph from '{graph_filename}'")
except FileNotFoundError:
    print(f"Error: File '{graph_filename}' not found.")
    exit()

# --- 3. Assign Sector Attribute to Nodes ---
# We assign the 'industry' as an attribute to each node in the graph
for node in G.nodes():
    G.nodes[node]['industry'] = sector_map.get(node, 'Unknown') # Using 'industry'

# --- 4. Calculate Metrics per Sector ---
print("Analyzing network properties by sector...")
# Get all unique industry names
industries = set(nx.get_node_attributes(G, 'industry').values())
sector_analysis = []

# Get centralities for all nodes at once
degrees = dict(G.degree())
clustering = nx.clustering(G)
betweenness = nx.betweenness_centrality(G, normalized=True)

for industry in sorted(list(industries)):
    if industry == 'Unknown': continue # Skip unknown
    
    # Find all nodes belonging to this industry
    sector_nodes = [node for node, data in G.nodes(data=True) if data.get('industry') == industry]
    
    if not sector_nodes:
        continue
    
    # Calculate properties
    num_nodes = len(sector_nodes)
    avg_degree = np.mean([degrees[n] for n in sector_nodes])
    avg_clustering = np.mean([clustering[n] for n in sector_nodes])
    avg_betweenness = np.mean([betweenness[n] for n in sector_nodes])
    
    sector_analysis.append({
        "Industry": industry, # Renamed column for clarity
        "Num Stocks": num_nodes,
        "Avg. Degree": avg_degree,
        "Avg. Clustering": avg_clustering,
        "Avg. Betweenness": avg_betweenness
    })

# --- 5. Display Results as a DataFrame ---
results_df = pd.DataFrame(sector_analysis)

# Check if the DataFrame is empty before trying to sort it
if not results_df.empty:
    results_df = results_df.sort_values(by="Avg. Degree", ascending=False)

    # Set pandas to display all rows and formatted floats
    pd.set_option('display.max_rows', None)
    pd.set_option('display.precision', 4)

    print("\n--- Network Analysis by Sector (at θ=0.5) ---")
    print(results_df)

    # You can save this table to a file
    results_df.to_csv("sector_analysis.csv")
    print("\nResults saved to 'sector_analysis.csv'")

    # Example of how to get the LaTeX code for your report
    print("\n--- LaTeX Code for Report Table ---")
    print(results_df.to_latex(index=False, float_format="%.4f"))
else:
    print("\nError: No sector data was found. The analysis table is empty.")
    print("This might happen if none of the stocks in your graph were in the Nifty 500 list.")