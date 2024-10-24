import pandas as pd
import yaml
import os
import logging
from sklearn.model_selection import train_test_split
import gdown

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the base directory path for all graph-related files
BASE_DIR = 'Graphs/Graph110'

def load_graph_structure(yaml_file_path):
    logging.info(f"Loading graph structure from {yaml_file_path}")
    with open(yaml_file_path, 'r') as file:
        return yaml.safe_load(file)

def save_graph_data(output_file, tag_file, data, total_rows, graph_structure):
    os.makedirs(BASE_DIR, exist_ok=True)
    if os.path.exists(output_file):
        os.remove(output_file)
    if os.path.exists(tag_file):
        os.remove(tag_file)

    graph_data = []
    tag_data = []  # List to hold tag values and graph indices
    total_processed_rows = 0

    for idx, row in data.iterrows():
        graph_index = total_processed_rows  # Assign a unique index to each graph (row)
        tag_data.append({'graph_index': graph_index, 'tag': row['tag']})  # Save tag with its corresponding graph index

        # For each attribute in the graph structure
        for node_id_1, connections in graph_structure.items():
            node_1_value = row.get(node_id_1, None)
            if node_1_value is None:
                continue

            # Iterate through the connections of the node
            for connection in connections:
                if isinstance(connection, dict):
                    for node_id_2, edge_weight in connection.items():
                        pass
                else:
                    node_id_2 = connection
                    edge_weight = None  # Set a default weight if not specified

                node_2_value = row.get(node_id_2, None)
                if node_2_value is None:
                    continue

                graph_row = {
                    'graph_index': graph_index,
                    'node_id_1': node_id_1,
                    'node_1_value': node_1_value,
                    'node_id_2': node_id_2,
                    'node_2_value': node_2_value,
                    'edge_weight': edge_weight
                }
                graph_data.append(graph_row)

        total_processed_rows += 1
        if total_processed_rows >= total_rows:
            break

    # Save graph data and tags to their respective files
    graph_df = pd.DataFrame(graph_data)
    graph_df.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)
    
    tags_df = pd.DataFrame(tag_data)
    tags_df.to_csv(tag_file, mode='a', header=not os.path.exists(tag_file), index=False)
    
    logging.info(f"All graphs saved to {output_file}. Tags saved to {tag_file}.")

def load_and_process_data(train_file_path, total_rows, yaml_file_path, test_size=0.25, log_interval=10000):
    # Load graph structure from YAML file
    graph_structure = load_graph_structure(yaml_file_path)
    logging.info("Graph structure loaded successfully.")

    # Load dataset
    logging.info(f"Loading dataset from {train_file_path}")
    data = pd.read_csv(train_file_path)
    logging.info("Dataset loaded successfully.")

    # Split dataset into training and testing sets
    logging.info(f"Splitting dataset into training and testing sets with test size = {test_size}")
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=1)
    logging.info(f"Dataset split into {len(train_data)} training rows and {len(test_data)} testing rows.")

    # Save training graph data and tags
    logging.info(f"Processing training data with {len(train_data)} rows.")
    train_graph_file = os.path.join(BASE_DIR, 'train_graphs.csv')
    train_tag_file = os.path.join(BASE_DIR, 'train_tags.csv')
    save_graph_data(train_graph_file, train_tag_file, train_data, total_rows, graph_structure)

    # Save testing graph data and tags
    logging.info(f"Processing testing data with {len(test_data)} rows.")
    test_graph_file = os.path.join(BASE_DIR, 'test_graphs.csv')
    test_tag_file = os.path.join(BASE_DIR, 'test_tags.csv')
    save_graph_data(test_graph_file, test_tag_file, test_data, total_rows, graph_structure)

def download_file_from_google_drive(drive_url, output_path):
    logging.info(f"Downloading file from {drive_url} to {output_path}")
    gdown.download(drive_url, output_path, quiet=False)

# Define path to your training data CSV file
train_file_path = 'CSVs/sample_train.csv'
yaml_file_path = os.path.join(BASE_DIR, "graph_structure.yaml")
total_rows = 10000  # Adjusted value

logging.info("Starting data processing.")
load_and_process_data(train_file_path, total_rows, yaml_file_path, log_interval=1000)
logging.info("Data processing completed.")
