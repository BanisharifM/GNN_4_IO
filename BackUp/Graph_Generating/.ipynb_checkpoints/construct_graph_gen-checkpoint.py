import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load the sorted correlation results CSV file
sorted_results_path = "results/correlation/sorted_attribute_correlations.csv"
sorted_results = pd.read_csv(sorted_results_path)

# Create a graph
G = nx.Graph()

# Add nodes for all attributes including 'tag'
attributes = sorted_results["Attribute"].tolist() + ["tag"]
G.add_nodes_from(attributes)

# Add edges based on top correlations
top_attributes = sorted_results["Attribute"].head(5).tolist()

# Connect 'tag' to top attributes
for attr in top_attributes:
    G.add_edge("tag", attr)

# For each top attribute, connect to its top 5 correlated attributes
for attr in top_attributes:
    correlations = sorted_results[
        [
            "Attribute",
            "Pearson_Correlation",
            "Spearman_Correlation",
            "Mutual_Information",
        ]
    ]
    correlations = correlations[correlations["Attribute"] != attr]
    correlations["Combined_Score"] = (
        correlations["Pearson_Correlation"]
        + correlations["Spearman_Correlation"]
        + correlations["Mutual_Information"]
    ) / 3
    top_related_attrs = (
        correlations.sort_values(by="Combined_Score", ascending=False)["Attribute"]
        .head(5)
        .tolist()
    )
    for related_attr in top_related_attrs:
        G.add_edge(attr, related_attr)

# Visualize the graph
plt.figure(figsize=(12, 12))
nx.draw(
    G,
    with_labels=True,
    node_color="lightblue",
    edge_color="gray",
    node_size=2000,
    font_size=10,
)
plt.title("Attribute Correlation Graph")
plt.savefig("results/correlation/attributes_correlation_graph.png")
# plt.show()
