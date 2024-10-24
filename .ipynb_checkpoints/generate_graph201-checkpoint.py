import pandas as pd
import json
import yaml
import os
import logging
from sklearn.model_selection import train_test_split
import gdown

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def load_graph_structure(yaml_file_path):
    logging.info(f"Loading graph structure from {yaml_file_path}")
    with open(yaml_file_path, 'r') as file:
        return yaml.safe_load(file)

def load_counter_mapping(mapping_file_path):
    logging.info(f"Loading counter to integer mapping from {mapping_file_path}")
    with open(mapping_file_path, 'r') as file:
        return json.load(file)

def normalize_node_name(node_name):
    """Normalize node names by stripping whitespace and ensuring case consistency."""
    return node_name.strip().lower()

def save_graph_data(output_dir, output_file, data, total_rows, graph_structure, counter_mapping):
    os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(output_file):
        os.remove(output_file)

    graph_data = []
    total_processed_rows = 0

    # Convert the counter_mapping values to lowercase for better matching
    normalized_counter_mapping = {int(k): v.lower() for k, v in counter_mapping.items()}

    logging.debug(f"Available counter mappings (normalized): {normalized_counter_mapping}")

    for idx, row in data.iterrows():
        # For each attribute in the graph structure
        for node_id_1, connections in graph_structure.items():
            node_1_value = row.get(node_id_1, None)
            if node_1_value is None:
                continue

            # Get integer mapping for node_id_1 after normalizing its case
            node_id_1_normalized = normalize_node_name(node_id_1).lower()
            node_id_1_int = next((key for key, value in normalized_counter_mapping.items() if value == node_id_1_normalized), None)

            if node_id_1_int is None:
                logging.warning(f"Skipping node {node_id_1} as no mapping was found.")
                continue

            # Iterate through the connections of the node
            for connection in connections:
                if isinstance(connection, dict):
                    for node_id_2, edge_weight in connection.items():
                        pass
                else:
                    node_id_2 = connection
                    edge_weight = None  # Set a default weight if not specified

                # Get integer mapping for node_id_2 after normalizing its case
                node_id_2_normalized = normalize_node_name(node_id_2).lower()
                node_id_2_int = next((key for key, value in normalized_counter_mapping.items() if value == node_id_2_normalized), None)

                if node_id_2_int is None:
                    logging.warning(f"Skipping node {node_id_2} as no mapping was found.")
                    continue

                node_2_value = row.get(node_id_2, None)
                if node_2_value is None:
                    continue

                # Add the edge to the graph data
                graph_row = {
                    'graph_index': total_processed_rows,
                    'node_id_1': node_id_1_int,
                    'node_1_value': node_1_value,
                    'node_id_2': node_id_2_int,
                    'node_2_value': node_2_value,
                    'edge_weight': edge_weight
                }
                graph_data.append(graph_row)

        total_processed_rows += 1
        if total_processed_rows >= total_rows:
            break

    # Convert list of dictionaries to DataFrame and save to CSV file
    graph_df = pd.DataFrame(graph_data)
    graph_df.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)
    logging.info(f"All graphs saved to {output_file}.")

def load_and_process_data(train_file_path, total_rows, yaml_file_path, counter_mapping_file, output_dir, test_size=0.25, log_interval=10000):
    # Load graph structure from YAML file
    graph_structure = load_graph_structure(yaml_file_path)
    logging.info("Graph structure loaded successfully.")

    # Load counter to integer mapping
    counter_mapping = load_counter_mapping(counter_mapping_file)
    logging.info("Counter to integer mapping loaded successfully.")

    # Load dataset
    logging.info(f"Loading dataset from {train_file_path}")
    data = pd.read_csv(train_file_path)
    logging.info(f"Dataset loaded successfully with {len(data)} total rows.")

    # Randomly sample 'total_rows' from the dataset
    logging.info(f"Randomly selecting {total_rows} rows from the dataset")
    if total_rows > len(data):
        logging.warning(f"Requested total_rows ({total_rows}) exceeds dataset size ({len(data)}). Using the full dataset.")
        total_rows = len(data)

    data_sample = data.sample(n=total_rows, random_state=1)  # Use random_state to ensure reproducibility
    logging.info(f"Randomly selected {len(data_sample)} rows.")

    # Split the sampled data into training and testing sets
    logging.info(f"Splitting selected data into training and testing sets with test size = {test_size}")
    train_data, test_data = train_test_split(data_sample, test_size=test_size, random_state=1)
    logging.info(f"Selected data split into {len(train_data)} training rows and {len(test_data)} testing rows.")

    # Save training graph data
    logging.info(f"Processing training data with {len(train_data)} rows.")
    output_file = os.path.join(output_dir, 'train_graphs.csv')
    save_graph_data(output_dir, output_file, train_data, len(train_data), graph_structure, counter_mapping)

    # Save testing graph data
    logging.info(f"Processing testing data with {len(test_data)} rows.")
    output_file = os.path.join(output_dir, 'test_graphs.csv')
    save_graph_data(output_dir, output_file, test_data, len(test_data), graph_structure, counter_mapping)

def download_file_from_google_drive(drive_url, output_path):
    logging.info(f"Downloading file from {drive_url} to {output_path}")
    gdown.download(drive_url, output_path, quiet=False)

# Define path to your training data CSV file
train_file_path = 'CSVs/sample_train_total.csv'
base_url = "Graphs/Graph201/"
yaml_file_path = base_url+"graph_structure.yaml"
counter_mapping_file = base_url+"counter_mapping.json"
output_dir = base_url+"6,647,219"
total_rows = 6647219  # Adjusted value

logging.info("Starting data processing.")
load_and_process_data(train_file_path, total_rows, yaml_file_path, counter_mapping_file, output_dir, log_interval=1000)
logging.info("Data processing completed.")
