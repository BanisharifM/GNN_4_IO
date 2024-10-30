import os
import pandas as pd
import yaml

# Define the threshold and top k attributes
threshold = 0.75
top_k = 5

# Ensure the output directory exists
sorted_dir = "results/correlation/full_data"

# Load the sorted correlation results for 'tag'
sorted_tag_path = os.path.join(sorted_dir, "sorted_tag_correlations.xlsx")
sorted_tag = pd.read_excel(sorted_tag_path)

# Function to identify primary connections based on the threshold
def get_primary_connections(attribute, threshold):
    sorted_path = os.path.join(sorted_dir, f"sorted_{attribute}_correlations.xlsx")
    sorted_results = pd.read_excel(sorted_path)

    # Filter based on the threshold
    filtered_results = sorted_results[
        (sorted_results["Pearson_Correlation"] > threshold)
        | (sorted_results["Spearman_Correlation"] > threshold)
        | (sorted_results["Combined_Score"] > threshold)
    ]

    # Get the list of attributes that meet the criteria
    connections = filtered_results["attr2"].tolist()
    return connections

# Function to identify secondary connections based on the threshold and top k
def get_secondary_connections(attribute, threshold, top_k):
    sorted_path = os.path.join(sorted_dir, f"sorted_{attribute}_correlations.xlsx")
    sorted_results = pd.read_excel(sorted_path)

    # Filter based on the threshold
    filtered_results = sorted_results[
        (sorted_results["Pearson_Correlation"] > threshold)
        | (sorted_results["Spearman_Correlation"] > threshold)
        | (sorted_results["Combined_Score"] > threshold)
    ]

    # Get the top k connections based on the combined score
    top_connections = filtered_results.nlargest(top_k, "Combined_Score")["attr2"].tolist()
    
    return top_connections

# Get primary connections for 'tag'
primary_connections = get_primary_connections("tag", threshold)

# Create the graph structure dictionary
graph_structure = {"tag": {}}

# Get secondary connections for each primary connection
for primary in primary_connections:
    secondary_connections = get_secondary_connections(primary, threshold, top_k)
    graph_structure["tag"][primary] = secondary_connections

# Save the graph structure to a YAML file
yaml_file_path = "Graphs/Graph13/graph_structure.yaml"
with open(yaml_file_path, "w") as yaml_file:
    yaml.dump(graph_structure, yaml_file, default_flow_style=False)

print(f"Graph structure saved to {yaml_file_path}")
