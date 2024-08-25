import yaml
import networkx as nx
import matplotlib.pyplot as plt

# Load the graph structure from the YAML file
yaml_file = 'results/correlation/graph_structure.yaml'
with open(yaml_file, 'r') as file:
    graph_data = yaml.safe_load(file)

# Create a directed graph
G = nx.DiGraph()

# Add nodes and edges to the graph
for node, edges in graph_data.items():
    for edge in edges:
        G.add_edge(node, edge)

# Visualize the graph
plt.figure(figsize=(15, 10))
pos = nx.spring_layout(G, seed=42)  # Position nodes using the Fruchterman-Reingold force-directed algorithm
nx.draw(G, pos, with_labels=True, node_size=2000, node_color='skyblue', font_size=10, font_weight='bold', edge_color='gray')
plt.title('Graph Visualization')
plt.savefig("results/correlation/graph_structure.png")
plt.show()
