import os
import pandas as pd
import yaml

# Define the threshold
threshold = 0.8

# Ensure the output directory exists
sorted_dir = "results/correlation/full_data"

# Load the sorted correlation results for 'tag'
sorted_tag_path = os.path.join(sorted_dir, "sorted_tag_correlations.xlsx")
sorted_tag = pd.read_excel(sorted_tag_path)


# Function to identify primary and secondary connections based on the threshold
def get_connections(attribute, threshold):
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


# Get primary connections for 'tag'
primary_connections = get_connections("tag", threshold)

# Create the graph structure dictionary
graph_structure = {"tag": {}}

# Get secondary connections for each primary connection
for primary in primary_connections:
    secondary_connections = get_connections(primary, threshold)
    graph_structure["tag"][primary] = secondary_connections

# Save the graph structure to a YAML file
yaml_file_path = "Graphs/Graph19/graph_structure.yaml"
with open(yaml_file_path, "w") as yaml_file:
    yaml.dump(graph_structure, yaml_file, default_flow_style=False)

print(f"Graph structure saved to {yaml_file_path}")
