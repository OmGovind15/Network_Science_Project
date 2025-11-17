import networkx as nx
import matplotlib.pyplot as plt
import pickle
import numpy as np

# --- Load the Graph ---
graph_filename = "nifty500_network.gpickle"
try:
    with open(graph_filename, 'rb') as f:
        G = pickle.load(f)
    print(f"Successfully loaded graph from '{graph_filename}'")
except FileNotFoundError:
    print(f"Error: File '{graph_filename}' not found.")
    exit()

# --- 1. Isolate the Largest Connected Component (LCC) ---
components = list(nx.connected_components(G))
if not components:
    print("Graph is empty, nothing to visualize.")
    exit()

largest_comp_nodes = max(components, key=len)
G_giant = G.subgraph(largest_comp_nodes).copy()
print(f"Visualizing the LCC with {G_giant.number_of_nodes()} nodes.")

# --- 2. Get Attributes and Identify Shells ---
degrees = dict(G_giant.degree())
min_degree = min(degrees.values())
max_degree = max(degrees.values())
node_sizes = [((degrees[node]**2) * 5) + 50 for node in G_giant.nodes()] 
node_colors = [degrees[node] for node in G_giant.nodes()]

# --- 3. Create Shells ---
sorted_degrees = sorted(degrees.items(), key=lambda item: item[1], reverse=True)
core_nodes = [node for node, degree in sorted_degrees[:10]]
periphery_nodes = [node for node in G_giant.nodes() if node not in core_nodes]
shells = [core_nodes, periphery_nodes]
labels = {node: node if node in core_nodes else '' for node in G_giant.nodes()}

# --- 4. Draw the Graph (WITH FIX) ---
fig = plt.figure(figsize=(18, 18)) 
# Create an axes object that occupies most of the figure
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

# Use the shell layout
pos = nx.shell_layout(G_giant, nlist=shells)

print("Drawing shell layout...")

# Draw the nodes
nx.draw_networkx_nodes(
    G_giant,
    pos,
    node_color=node_colors,
    node_size=node_sizes,
    cmap=plt.cm.coolwarm,
    vmin=min_degree,
    vmax=max_degree,
    alpha=0.9,
    ax=ax # Tell it to draw on our axes
)
# Draw the edges
# NEW CORRECTED LINE:
nx.draw_networkx_edges(G_giant, pos, edge_color='grey', alpha=0.3, width=0.7, ax=ax)
# Draw *only* the labels for the hubs
nx.draw_networkx_labels(G_giant, pos, labels=labels, font_size=10, font_weight='bold', ax=ax)

ax.set_title("LCC Core-Periphery Structure (θ=0.5)", fontsize=25)

# --- START OF FIX ---
# We will manually adjust the plot to leave space at the bottom.
# This prevents tight_layout or bbox_inches from cutting off the text.
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.15)

# Use plt.figtext() to place text in FIGURE coordinates (0.0 to 1.0)
plt.figtext(0.5, 0.05, # x=center, y=5% from bottom
            f"Inner Shell: Top 10 Hubs (by Degree)\nOuter Shell: {len(periphery_nodes)} Other Nodes", 
            ha='center', va='bottom', fontsize=15)
# --- END OF FIX ---

# Create a new axes for the colorbar so it doesn't get messed up
cax = fig.add_axes([0.9, 0.25, 0.03, 0.5]) # [left, bottom, width, height]
sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, 
                            norm=plt.Normalize(vmin=min_degree, vmax=max_degree))
cbar = fig.colorbar(sm, cax=cax)
cbar.set_label('Node Degree', size=15)

ax.axis('off') # Turn off the axis of the main plot
plt.savefig("lcc_visualization_shell.png", dpi=300)
plt.show()

print("Saved LCC Shell visualization to 'lcc_visualization_shell.png'")