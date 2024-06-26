import h5py
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
    
    nodes_list = []
    edge_index_list = []
    features_list = []
    labels_list = []
    
    logging.info("Processing each row in the dataset.")
    for index, row in df.iterrows():
        if index % 10000 == 0 and index > 0:
            logging.info(f"Processed {index} rows out of {len(df)}")

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
        
        nodes_list.append(nodes)
        edge_index_list.append(edge_index)
        features_list.append(x)
        labels_list.append(y)

    logging.info("Finished processing rows.")

    # Ensure the save directory exists
    logging.info(f"Ensuring the save directory exists: {save_path}")
    os.makedirs(save_path, exist_ok=True)

    logging.info("Preparing data for HDF5 storage.")
    try:
        # Convert lists to object arrays
        edge_index_obj_array = np.empty(len(edge_index_list), dtype=object)
        edge_index_obj_array[:] = edge_index_list
        
        features_obj_array = np.empty(len(features_list), dtype=object)
        features_obj_array[:] = features_list

        logging.info("Saving the graph data to an HDF5 file.")
        with h5py.File(os.path.join(save_path, 'graph_data.h5'), 'w') as f:
            f.create_dataset('nodes', data=np.array(nodes_list, dtype=h5py.special_dtype(vlen=str)), chunks=True, compression='gzip')
            f.create_dataset('edge_index', (len(edge_index_list),), dtype=h5py.vlen_dtype(np.dtype('int32')), chunks=True, compression='gzip')
            f.create_dataset('features', (len(features_list),), dtype=h5py.vlen_dtype(np.dtype('float32')), chunks=True, compression='gzip')
            
            for i, edge_index in enumerate(edge_index_list):
                f['edge_index'][i] = edge_index
            
            for i, features in enumerate(features_list):
                f['features'][i] = features

            f.create_dataset('labels', data=np.array(labels_list, dtype=np.int32), chunks=True, compression='gzip')

        logging.info(f"Graph data saved to {os.path.join(save_path, 'graph_data.h5')}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    yaml_path = 'results/correlation/graph_structure.yaml'
    data_path = 'CSVs/sample_train.csv'
    save_path = 'Graph6/'
    generate_graph(yaml_path, data_path, save_path)
