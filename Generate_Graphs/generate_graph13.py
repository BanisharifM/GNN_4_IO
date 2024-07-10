import numpy as np
import yaml
import pandas as pd
import os
import logging
import pyarrow as pa
import pyarrow.parquet as pq

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
    
    nodes_data = []
    edges_data = []

    logging.info("Processing each row in the dataset.")
    for index, row in df.iterrows():
        if index >= 1000:  # Limit to the first 10,000 rows for testing
            break
        if index % 1000 == 0 and index > 0:
            logging.info(f"Processed {index} rows out of {len(df)}")

        nodes = {}
        edges = []

        # Root node features
        tag_node = 'tag'
        if tag_node in row:
            nodes[tag_node] = row[tag_node]

        # Primary and secondary nodes
        for primary_node, secondary_nodes in graph_structure[tag_node].items():
            if primary_node in row:
                nodes[primary_node] = row[primary_node]
                edges.append({"source": tag_node, "target": primary_node, "data": {"dummy": None}})
                
                for secondary_node in secondary_nodes:
                    if secondary_node in row:
                        nodes[secondary_node] = row[secondary_node]
                        edges.append({"source": primary_node, "target": secondary_node, "data": {"dummy": None}})

        # Convert nodes to the required format
        nodes_list = [{"index": index, "id": node, "value": value} for node, value in nodes.items()]
        edges_list = [{"index": index, "source": edge["source"], "target": edge["target"], "data": edge["data"]} for edge in edges]

        nodes_data.extend(nodes_list)
        edges_data.extend(edges_list)

    logging.info("Finished processing rows.")
    logging.info(f"Number of nodes: {len(nodes_data)}")
    logging.info(f"Number of edges: {len(edges_data)}")

    # Ensure the save directory exists
    logging.info(f"Ensuring the save directory exists: {save_path}")
    os.makedirs(save_path, exist_ok=True)

    # Save nodes and edges to Parquet files
    logging.info("Saving nodes and edges to Parquet files.")
    nodes_df = pd.DataFrame(nodes_data)
    edges_df = pd.DataFrame(edges_data)

    nodes_table = pa.Table.from_pandas(nodes_df)
    edges_table = pa.Table.from_pandas(edges_df)

    pq.write_table(nodes_table, os.path.join(save_path, 'nodes_1000.parquet'))
    pq.write_table(edges_table, os.path.join(save_path, 'edges_1000.parquet'))

    logging.info(f"Graph data saved to Parquet files in {save_path}")

if __name__ == "__main__":
    yaml_path = 'Graphs/Graph22/graph_structure.yaml'
    data_path = 'CSVs/train_data.csv'
    save_path = 'Graphs/Graph22/'
    generate_graph(yaml_path, data_path, save_path)
