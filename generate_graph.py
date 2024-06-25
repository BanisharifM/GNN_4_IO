import h5py
import numpy as np
import yaml
import pandas as pd
import os
import torch

def generate_graph(yaml_path, data_path, save_path):
    # Load the graph structure from the YAML file
    with open(yaml_path, 'r') as file:
        graph_structure = yaml.safe_load(file)
    
    # Read the dataset from CSV
    df = pd.read_csv(data_path)
    
    nodes_list = []
    edge_index_list = []
    features_list = []
    labels_list = []
    
    for _, row in df.iterrows():
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

    # Ensure the save directory exists
    os.makedirs(save_path, exist_ok=True)

    # Save the graph to an HDF5 file
    with h5py.File(os.path.join(save_path, 'graph_data.h5'), 'w') as f:
        f.create_dataset('nodes', data=np.array(nodes_list, dtype=h5py.special_dtype(vlen=str)), chunks=True, compression='gzip')
        f.create_dataset('edge_index', data=np.array(edge_index_list, dtype=object), chunks=True, compression='gzip')
        f.create_dataset('features', data=np.array(features_list, dtype=object), chunks=True, compression='gzip')
        f.create_dataset('labels', data=np.array(labels_list, dtype=np.int32), chunks=True, compression='gzip')

if __name__ == "__main__":
    yaml_path = 'results/correlation/graph_structure.yaml'
    data_path = 'CSVs/sample_train.csv'
    save_path = 'Graph5/'
    generate_graph(yaml_path, data_path, save_path)
