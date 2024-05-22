import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
import pygraphviz as pgv


# Function to visualize the graph as a tree and save it as an image
def visualize_and_save_tree_graph(edges_df, sheet_name, output_dir):
    G = nx.DiGraph()

    # Add nodes and edges to the graph
    for _, row in edges_df.iterrows():
        source = row["Source"]
        source_value = (
            round(row["Source_Value"], 2) if not pd.isna(row["Source_Value"]) else ""
        )
        target = row["Target"]
        target_value = (
            round(row["Target_Value"], 2) if not pd.isna(row["Target_Value"]) else ""
        )

        source_label = f"{source}\n{source_value}" if source_value != "" else source
        target_label = f"{target}\n{target_value}" if target_value != "" else target

        G.add_node(source_label)
        G.add_node(target_label)
        G.add_edge(source_label, target_label)

    # Create a pygraphviz graph
    A = nx.nx_agraph.to_agraph(G)
    A.layout("dot")  # Use the dot layout for tree structure
    image_path = os.path.join(output_dir, f"{sheet_name}.png")
    A.draw(image_path)
    print(f"Graph saved to {image_path}")


# Directory to save the graph images
output_dir = "Graph_Images"
os.makedirs(output_dir, exist_ok=True)

# Load the Excel file
excel_file_path = "Graph3/graphs.xlsx"
excel_data = pd.ExcelFile(excel_file_path)

# Iterate through each sheet and visualize the graph
for sheet_name in excel_data.sheet_names:
    edges_df = pd.read_excel(excel_file_path, sheet_name=sheet_name)
    visualize_and_save_tree_graph(edges_df, sheet_name, output_dir)

print("Graph visualization completed.")
