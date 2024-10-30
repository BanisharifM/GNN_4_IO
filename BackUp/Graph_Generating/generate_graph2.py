import h5py
import numpy as np
import yaml
import pandas as pd
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def generate_graph(yaml_path, data_path, save_path):
    # Load the graph structure from the YAML file
    with open(yaml_path, 'r') as file:
        graph_structure = yaml.safe_load(file)
    
    # Read the dataset from CSV
    df = pd.read_csv(data_path)
    
    # Logging the total number of rows to process
    total_rows = len(df)
    logging.info(f"Total rows to process: {total_rows}")
    
    # Ensure the save directory exists
    os.makedirs(save_path, exist_ok=True)

    # Open the HDF5 file for writing
    h5_file_path = os.path.join(save_path, 'graph_data.h5')
    with h5py.File(h5_file_path, 'w') as f:
        # Create datasets with max shape and chunking enabled
        str_dt = h5py.special_dtype(vlen=str)
        int_vlen_dt = h5py.vlen_dtype(np.dtype('int32'))
        float_vlen_dt = h5py.vlen_dtype(np.dtype('float32'))
        
        nodes_ds = f.create_dataset('nodes', shape=(0,), maxshape=(None,), dtype=str_dt, chunks=True, compression='gzip')
        edge_index_ds = f.create_dataset('edge_index', shape=(0,), maxshape=(None,), dtype=int_vlen_dt, chunks=True, compression='gzip')
        features_ds = f.create_dataset('features', shape=(0,), maxshape=(None,), dtype=float_vlen_dt, chunks=True, compression='gzip')
        labels_ds = f.create_dataset('labels', shape=(0,), maxshape=(None,), dtype=np.int32, chunks=True, compression='gzip')
        
        for idx, row in enumerate(df.iterrows()):
            # Log the progress every 1000 rows
            if idx % 1000 == 0:
                logging.info(f"Processing row {idx} of {total_rows}")
            
            row = row[1]
            nodes = ['tag'] + [node for primary, secondaries in graph_structure['tag'].items() for node in [primary] + secondaries]
            edge_index = []
            features = []

            # Root node features
            tag_node = 'tag'
            features.append([row[tag_node]])

            # Primary and secondary nodes
            for primary_node, secondary_nodes in graph_structure[tag_node].items():
                if primary_node in row:
                    primary_index = nodes.index(primary_node)
                    edge_index.append([0, primary_index])  # Root to primary
                    features.append([row[primary_node]])

                    for secondary_node in secondary_nodes:
                        if secondary_node in row:
                            secondary_index = nodes.index(secondary_node)
                            edge_index.append([primary_index, secondary_index])  # Primary to secondary
                            features.append([row[secondary_node]])

            x = np.array(features, dtype=np.float32)
            edge_index = np.array(edge_index, dtype=np.int32)
            y = np.array([int(row[tag_node])], dtype=np.int32)  # Ensure the target is an integer
            
            # Incrementally write data to the datasets
            nodes_ds.resize((nodes_ds.shape[0] + 1,))
            nodes_ds[-1] = np.array(nodes, dtype=object).astype('S')
            
            edge_index_ds.resize((edge_index_ds.shape[0] + 1,))
            edge_index_ds[-1] = np.array(edge_index, dtype=object)
            
            features_ds.resize((features_ds.shape[0] + 1,))
            features_ds[-1] = np.array(x, dtype=object)
            
            labels_ds.resize((labels_ds.shape[0] + 1,))
            labels_ds[-1] = y
        
    logging.info("Graph data saved successfully")

if __name__ == "__main__":
    yaml_path = 'results/correlation/graph_structure.yaml'
    data_path = 'CSVs/sample_train.csv'
    save_path = 'Graph5/'
    generate_graph(yaml_path, data_path, save_path)
