import numpy as np
import yaml
import pandas as pd
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_graph(yaml_path, train_path, val_path, test_path, save_path):
    logging.info("Starting the graph generation process.")
    
    # Load the graph structure from the YAML file
    logging.info("Loading graph structure from YAML file.")
    with open(yaml_path, 'r') as file:
        graph_structure = yaml.safe_load(file)
    
    def process_nodes(df, split):
        nodes_data = []
        for index, row in df.iterrows():
            if index >= 1000:  # Limit to the first 10,000 rows for testing
                break
            if index % 10000 == 0 and index > 0:
                logging.info(f"Processed {index} rows out of {len(df)}")

            nodes = {}

            # Root node features
            tag_node = 'tag'
            if tag_node in row:
                nodes[tag_node] = row[tag_node]

            # Primary and secondary nodes
            for primary_node, secondary_nodes in graph_structure[tag_node].items():
                if primary_node in row:
                    nodes[primary_node] = row[primary_node]
                    
                    for secondary_node in secondary_nodes:
                        if secondary_node in row:
                            nodes[secondary_node] = row[secondary_node]

            # Convert nodes to the required format
            nodes_list = [{"index": index, "id": node, "value": value} for node, value in nodes.items()]
            nodes_data.extend(nodes_list)

        logging.info(f"Finished processing {split} nodes.")
        logging.info(f"Number of {split} nodes: {len(nodes_data)}")

        # Save nodes to CSV file
        nodes_df = pd.DataFrame(nodes_data)
        nodes_csv_path = os.path.join(save_path, f'{split}_data_nodes.csv')
        nodes_df.to_csv(nodes_csv_path, index=False)
        logging.info(f"{split.capitalize()} nodes saved to {nodes_csv_path}")

    # Load the data splits
    logging.info("Loading and processing training data.")
    train_df = pd.read_csv(train_path)
    process_nodes(train_df, 'train')

    logging.info("Loading and processing validation data.")
    val_df = pd.read_csv(val_path)
    process_nodes(val_df, 'val')

    logging.info("Loading and processing test data.")
    test_df = pd.read_csv(test_path)
    process_nodes(test_df, 'test')

    # Process and save edges
    edges_data = []
    for tag_node, primary_nodes in graph_structure.items():
        for primary_node, secondary_nodes in primary_nodes.items():
            edges_data.append({"source": tag_node, "target": primary_node})
            for secondary_node in secondary_nodes:
                edges_data.append({"source": primary_node, "target": secondary_node})

    logging.info(f"Finished processing edges.")
    logging.info(f"Number of edges: {len(edges_data)}")

    edges_df = pd.DataFrame(edges_data)
    edges_csv_path = os.path.join(save_path, 'edges.csv')
    edges_df.to_csv(edges_csv_path, index=False)
    logging.info(f"Edges saved to {edges_csv_path}")

if __name__ == "__main__":
    yaml_path = 'Graphs/Graph22/graph_structure.yaml'
    train_path = 'CSVs/train_data.csv'
    val_path = 'CSVs/val_data.csv'
    test_path = 'CSVs/test_data.csv'
    save_path = 'Graphs/Graph22/'
    generate_graph(yaml_path, train_path, val_path, test_path, save_path)
