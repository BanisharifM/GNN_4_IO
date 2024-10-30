import json
import numpy as np
import yaml
import pandas as pd
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_graph(yaml_path, data_path, save_path):
    logging.info("Starting the graph generation process.")
    
    # Load the graph structure from the YAML file
    logging.info("Loading graph structure from YAML file.")
    with open(yaml_path, 'r') as file:
        graph_structure = yaml.safe_load(file)
    
    # Read the dataset from CSV
    logging.info("Reading dataset from CSV file.")
    df = pd.read_csv(data_path)
    
    graphs = []

    logging.info("Processing each row in the dataset.")
    for index, row in df.iterrows():
        if index >= 10000:  # Limit to the first 10,000 rows for testing
            break
        if index % 1000 == 0 and index > 0:
            logging.info(f"Processed {index} rows out of {len(df)}")

        nodes = ['tag'] + [node for primary, secondaries in graph_structure['tag'].items() for node in [primary] + secondaries]
        edge_index = []
        features = []

        # Root node features
        tag_node = 'tag'
        features.append({"id": tag_node, "value": row[tag_node]})

        # Primary and secondary nodes
        for primary_node, secondary_nodes in graph_structure[tag_node].items():
            if primary_node in row:
                primary_index = nodes.index(primary_node)
                edge_index.append({"source": tag_node, "target": primary_node, "data": {}})
                features.append({"id": primary_node, "value": row[primary_node]})

                for secondary_node in secondary_nodes:
                    if secondary_node in row:
                        secondary_index = nodes.index(secondary_node)
                        edge_index.append({"source": primary_node, "target": secondary_node, "data": {}})
                        features.append({"id": secondary_node, "value": row[secondary_node]})

        graph = {
            "index": index,
            "nodes": features,
            "edges": edge_index
        }

        graphs.append(graph)

    logging.info("Finished processing rows.")
    logging.info(f"Number of graphs: {len(graphs)}")

    # Ensure the save directory exists
    logging.info(f"Ensuring the save directory exists: {save_path}")
    os.makedirs(save_path, exist_ok=True)

    # Save to JSON file
    logging.info("Saving data to JSON file.")
    with open(os.path.join(save_path, 'graphs_100.json'), 'w') as f:
        json.dump(graphs, f, indent=4)

    logging.info(f"Graph data saved to JSON file in {save_path}")

if __name__ == "__main__":
    yaml_path = 'results/correlation/full_data/graph_structure.yaml'
    data_path = 'CSVs/sample_train_100.csv'
    save_path = 'Graphs/Graph10/'
    generate_graph(yaml_path, data_path, save_path)
