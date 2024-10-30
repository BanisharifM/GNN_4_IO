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

def load_and_process_data(file_paths, total_rows, yaml_file_path, log_interval=100000):
    # Load graph structure from YAML file
    graph_structure = load_graph_structure(yaml_file_path)
    attributes = list(graph_structure['tag'].keys())
    logging.info("Graph structure loaded successfully.")

    proportions = {
        'train': 0.64,  # 64% of total rows for training data
        'val': 0.16,    # 16% of total rows for validation data
        'test': 0.20    # 20% of total rows for testing data
    }
    
    # Calculate actual number of rows for each dataset
    row_counts = {key: int(total_rows * prop) for key, prop in proportions.items()}
    logging.info(f"Row counts calculated: {row_counts}")

    # Create directory for output CSVs if it doesn't exist
    output_dir = 'Graphs/Graph37/'
    os.makedirs(output_dir, exist_ok=True)

    # Initialize data storage for final CSVs
    final_graph_data = {key: [] for key in file_paths}

    # Processing each file
    for key, file_path in file_paths.items():
        logging.info(f"Processing {key} data from {file_path} with {row_counts[key]} rows.")
        data_iterator = pd.read_csv(file_path, chunksize=row_counts[key])
        total_processed_rows = 0
        
        for chunk_num, chunk in enumerate(data_iterator):
            logging.info(f"Processing chunk {chunk_num + 1} for {key} data.")
            
            for idx, row in chunk.iterrows():
                # For each attribute, create a link to the tag
                for attr in attributes:
                    graph_row = {
                        'graph_index': total_processed_rows,
                        'tag_value': row['tag'],
                        'node_id': attr,
                        'node_value': row[attr]
                    }
                    final_graph_data[key].append(graph_row)
                total_processed_rows += 1

                if total_processed_rows % log_interval == 0:
                    logging.info(f"Processed {total_processed_rows} rows for {key} data.")
            
            # Stop reading if the required number of rows have been processed
            if total_processed_rows >= row_counts[key]:
                break
        
        # Convert list of dictionaries to DataFrame and save to CSV
        graph_df = pd.DataFrame(final_graph_data[key])
        output_file = os.path.join(output_dir, f'{key}_graphs.csv')
        graph_df.to_csv(output_file, index=False)
        logging.info(f"All {key} graphs saved to {output_file}.")

# Define paths to your CSV files
file_paths = {
    'train': 'CSVs/train_data.csv',
    'val': 'CSVs/val_data.csv',
    'test': 'CSVs/test_data.csv'
}

yaml_file_path = "Graphs/Graph37/graph_structure.yaml"
total_rows = 60000  # You can adjust this value as needed

logging.info("Starting data processing.")
load_and_process_data(file_paths, total_rows, yaml_file_path, log_interval=1000)
logging.info("Data processing completed.")
