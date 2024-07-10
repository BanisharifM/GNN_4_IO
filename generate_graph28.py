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

def load_and_process_data(train_file_path, total_rows, yaml_file_path, log_interval=100):
    # Load graph structure from YAML file
    graph_structure = load_graph_structure(yaml_file_path)
    attributes = list(graph_structure['tag'].keys())
    logging.info("Graph structure loaded successfully.")

    # Calculate the number of rows to read from the training data
    logging.info(f"Processing training data from {train_file_path} with {total_rows} rows.")
    
    # Create directory for output CSVs if it doesn't exist
    output_dir = 'Graphs/Graph_Test/'
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
            # For each attribute, create a link to the tag
            for attr in attributes:
                graph_row = {
                    'graph_index': total_processed_rows,
                    'tag_value': row['tag'],
                    'node_id': attr,
                    'node_value': row[attr]
                }
                graph_data.append(graph_row)
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
yaml_file_path = "Graphs/Graph_Test/graph_structure.yaml"
total_rows = 1000  # You can adjust this value as needed

logging.info("Starting data processing.")
load_and_process_data(train_file_path, total_rows, yaml_file_path, log_interval=100)
logging.info("Data processing completed.")
