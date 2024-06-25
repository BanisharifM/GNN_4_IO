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
def add_edges(graph_data, parent=None):
    for node, children in graph_data.items():
        if parent:
            G.add_edge(parent, node)
        for child in children:
            G.add_edge(node, child)
        if isinstance(children, dict):
            add_edges(children, node)

# Since the input YAML is structured with 'tag' as the root, we handle it separately
root_node = 'tag'
add_edges({root_node: graph_data[root_node]})

# Using spring_layout for a reasonable layout
pos = nx.spring_layout(G)

# Draw the graph
plt.figure(figsize=(20, 20))
nx.draw(G, pos=pos, with_labels=True, node_size=2000, node_color='skyblue', font_size=10, font_weight='bold', edge_color='gray', arrows=True)
plt.title('Graph Visualization')
plt.savefig("results/correlation/graph_structure.png")
plt.show()
