import pandas as pd
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

def save_graph_data(output_dir, output_file, data, total_rows, graph_structure):
    os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(output_file):
        os.remove(output_file)

    graph_data = []
    total_processed_rows = 0

    for idx, row in data.iterrows():
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

                # Avoid duplicate edges by ensuring the pair (node_id_1, node_id_2) is only processed once
                if node_id_2 in graph_structure and node_id_1 in graph_structure[node_id_2]:
                    continue

                node_2_value = row.get(node_id_2, None)
                if node_2_value is None:
                    continue

                # Add the edge to the graph data
                graph_row = {
                    'graph_index': total_processed_rows,
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

    # Convert list of dictionaries to DataFrame and save to CSV file
    graph_df = pd.DataFrame(graph_data)
    graph_df.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)
    logging.info(f"All graphs saved to {output_file}.")

def load_and_process_data(train_file_path, total_rows, yaml_file_path, test_size=0.25, log_interval=10000):
    # Load graph structure from YAML file
    graph_structure = load_graph_structure(yaml_file_path)
    logging.info("Graph structure loaded successfully.")

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
    output_dir = 'Graphs/Graph200/6,000,000'
    output_file = os.path.join(output_dir, 'train_graphs.csv')
    save_graph_data(output_dir, output_file, train_data, len(train_data), graph_structure)

    # Save testing graph data
    logging.info(f"Processing testing data with {len(test_data)} rows.")
    output_dir = 'Graphs/Graph200/6,000,000'
    output_file = os.path.join(output_dir, 'test_graphs.csv')
    save_graph_data(output_dir, output_file, test_data, len(test_data), graph_structure)

def download_file_from_google_drive(drive_url, output_path):
    logging.info(f"Downloading file from {drive_url} to {output_path}")
    gdown.download(drive_url, output_path, quiet=False)

# Define path to your training data CSV file
train_file_path = 'CSVs/sample_train_total.csv'
yaml_file_path = "Graphs/Graph200/graph_structure.yaml"
total_rows = 6000000  # Adjusted value

logging.info("Starting data processing.")
load_and_process_data(train_file_path, total_rows, yaml_file_path, log_interval=1000)
logging.info("Data processing completed.")
