import pandas as pd
import yaml
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_graph_structure(yaml_file_path):
    logging.info(f"Loading graph structure from {yaml_file_path}")
    with open(yaml_file_path, 'r') as file:
        return yaml.safe_load(file)

def load_and_process_data(train_file_path, total_rows, yaml_file_path, log_interval):
    # Load graph structure from YAML file
    graph_structure = load_graph_structure(yaml_file_path)
    logging.info("Graph structure loaded successfully.")

    # Calculate the number of rows to read from the training data
    logging.info(f"Processing training data from {train_file_path} with {total_rows} rows.")
    
    # Create directory for output CSVs if it doesn't exist
    output_dir = 'Graphs/Graph38/'
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare the output file
    output_file = os.path.join(output_dir, 'train_graphs.csv')
    if os.path.exists(output_file):
        os.remove(output_file)

    total_processed_rows = 0

    # Process the training data
    data_iterator = pd.read_csv(train_file_path, chunksize=log_interval)
    
    for chunk_num, chunk in enumerate(data_iterator):
        logging.info(f"Processing chunk {chunk_num + 1} for training data.")
        
        graph_data = []
        for idx, row in chunk.iterrows():
            # Create the primary connections for the 'tag'
            for node, subnodes in graph_structure['tag'].items():
                # Connect 'tag' to primary node
                graph_row = {
                    'graph_index': total_processed_rows,
                    'source': 'tag',
                    'target': node,
                    'tag_value': row['tag'],
                    'node_value': row[node] if node in row else None
                }
                graph_data.append(graph_row)

                # Connect primary node to its subnodes
                for subnode in subnodes:
                    graph_row_sub = {
                        'graph_index': total_processed_rows,
                        'source': node,
                        'target': subnode,
                        'tag_value': row['tag'],
                        'node_value': row[subnode] if subnode in row else None
                    }
                    graph_data.append(graph_row_sub)

            total_processed_rows += 1

            if total_processed_rows >= total_rows:
                break

        # Convert list of dictionaries to DataFrame and append to the CSV file
        graph_df = pd.DataFrame(graph_data)
        graph_df.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)
        logging.info(f"Processed {total_processed_rows} rows for training data.")

        if total_processed_rows >= total_rows:
            break

    logging.info(f"All training graphs saved to {output_file}.")

# Define path to your training data CSV file
train_file_path = 'CSVs/train_data.csv'
yaml_file_path = "Graphs/Graph38/graph_structure.yaml"
total_rows = 10000  # Adjusted to keep the output small

logging.info("Starting data processing.")
load_and_process_data(train_file_path, total_rows, yaml_file_path, log_interval=1000)
logging.info("Data processing completed.")
