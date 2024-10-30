import pandas as pd
import numpy as np
import yaml
import os
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_to_csv(data, filename):
    pd.DataFrame(data).to_csv(filename, index=False)

def save_list_of_arrays(data_list, folder_path, file_prefix):
    os.makedirs(folder_path, exist_ok=True)
    metadata = []
    for i, data in enumerate(data_list):
        file_path = os.path.join(folder_path, f"{file_prefix}_{i}.csv")
        save_to_csv(data, file_path)
        metadata.append(file_path)
    return metadata

def generate_graph(yaml_path, data_path, save_path):
    logging.info("Starting the graph generation process.")
    
    # Load the graph structure from the YAML file
    logging.info("Loading graph structure from YAML file.")
    with open(yaml_path, 'r') as file:
        graph_structure = yaml.safe_load(file)
    
    # Read the dataset from CSV
    logging.info("Reading dataset from CSV file.")
    df = pd.read_csv(data_path)
    
    nodes_list = []
    edge_index_list = []
    features_list = []
    labels_list = []
    
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
        features.append(row[tag_node])

        # Primary and secondary nodes
        for primary_node, secondary_nodes in graph_structure[tag_node].items():
            if primary_node in row:
                primary_index = nodes.index(primary_node)
                edge_index.append([0, primary_index])  # Root to primary
                features.append(row[primary_node])

                for secondary_node in secondary_nodes:
                    if secondary_node in row:
                        secondary_index = nodes.index(secondary_node)
                        edge_index.append([primary_index, secondary_index])  # Primary to secondary
                        features.append(row[secondary_node])

        features = np.array(features, dtype=np.float32)
        edge_index = np.array(edge_index, dtype=np.int32)
        label = int(row[tag_node])  # Ensure the target is an integer
        
        nodes_list.append(nodes)
        edge_index_list.append(edge_index)
        features_list.append(features)
        labels_list.append(label)

    logging.info("Finished processing rows.")
    logging.info(f"Number of labels: {len(labels_list)}")

    # Ensure the save directory exists
    logging.info(f"Ensuring the save directory exists: {save_path}")
    os.makedirs(save_path, exist_ok=True)

    # Save data to CSV files
    logging.info("Saving data to CSV files.")
    save_to_csv(nodes_list, os.path.join(save_path, 'nodes.csv'))
    edge_index_metadata = save_list_of_arrays(edge_index_list, os.path.join(save_path, 'edge_index'), 'edge_index')
    features_metadata = save_list_of_arrays(features_list, os.path.join(save_path, 'features'), 'features')
    save_to_csv(labels_list, os.path.join(save_path, 'labels.csv'))

    # Save metadata
    metadata = {
        'edge_index': edge_index_metadata,
        'features': features_metadata
    }
    with open(os.path.join(save_path, 'metadata.json'), 'w') as f:
        json.dump(metadata, f)

    logging.info(f"Graph data saved to CSV files in {save_path}")

if __name__ == "__main__":
    yaml_path = 'results/correlation/full_data/graph_structure.yaml'
    data_path = 'CSVs/sample_train_100.csv'
    save_path = 'Graphs/Graph9/'
    generate_graph(yaml_path, data_path, save_path)
