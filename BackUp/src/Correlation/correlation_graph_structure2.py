import os
import pandas as pd
import yaml

# Define the top k attributes
top_k = 5

# Ensure the output directory exists
sorted_dir = "results/correlation/full_data"

# Load the sorted correlation results for 'tag'
sorted_tag_path = os.path.join(sorted_dir, "sorted_tag_correlations.xlsx")
sorted_tag = pd.read_excel(sorted_tag_path)

# Get the top 45 connections for 'tag' (if there are that many)
top_45_primary_connections = sorted_tag.nlargest(45, "Combined_Score")["attr2"].tolist()

# Function to identify secondary connections based on the top k
def get_top_k_connections(attribute, top_k):
    sorted_path = os.path.join(sorted_dir, f"sorted_{attribute}_correlations.xlsx")
    sorted_results = pd.read_excel(sorted_path)

    # Get the top k connections based on the combined score
    top_connections = sorted_results.nlargest(top_k, "Combined_Score")["attr2"].tolist()
    
    return top_connections

# Create the graph structure dictionary
graph_structure = {"tag": {}}

# Get secondary connections for each of the top 45 primary connections
for primary in top_45_primary_connections:
    secondary_connections = get_top_k_connections(primary, top_k)
    graph_structure["tag"][primary] = secondary_connections

# Save the graph structure to a YAML file
yaml_file_path = "Graphs/Graph13/graph_structure_2.yaml"
with open(yaml_file_path, "w") as yaml_file:
    yaml.dump(graph_structure, yaml_file, default_flow_style=False)

print(f"Graph structure saved to {yaml_file_path}")
