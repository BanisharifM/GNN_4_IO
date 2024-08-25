import json
import networkx as nx
import matplotlib.pyplot as plt

# Load the graph structure from the JSON file
json_file_path = 'results/correlation/graph_structure.json'
with open(json_file_path, 'r') as file:
    graph_data = json.load(file)

# Create a directed graph
G = nx.DiGraph()

# Add nodes and edges to the graph
for primary_node, secondary_nodes in graph_data.items():
    for secondary_node, tertiary_nodes in secondary_nodes.items():
        G.add_edge(primary_node, secondary_node)
        for tertiary_node in tertiary_nodes:
            G.add_edge(secondary_node, tertiary_node)

# Define positions manually for a hierarchical layout
pos = {}
width = 1.
root = 'tag'
primary_nodes = list(graph_data[root].keys())

# Root position
pos[root] = (width / 2, 1.0)

# Primary nodes positions
for i, primary_node in enumerate(primary_nodes):
    pos[primary_node] = (width * (i + 1) / (len(primary_nodes) + 1), 0.5)
    
    # Secondary nodes positions
    secondary_nodes = graph_data[root][primary_node]
    for j, secondary_node in enumerate(secondary_nodes):
        pos[secondary_node] = (width * (i + 1) / (len(primary_nodes) + 1) + (j - len(secondary_nodes) / 2) * 0.03, 0)

# Visualize the graph
plt.figure(figsize=(30, 20))
nx.draw(G, pos, with_labels=True, node_size=2000, node_color='skyblue', font_size=8, font_weight='bold', edge_color='gray', arrows=True)
plt.title('Graph Visualization')
plt.savefig("results/correlation/graph_structure_from_json.png")
plt.show()
