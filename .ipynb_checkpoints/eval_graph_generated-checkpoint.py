import pandas as pd
import json
import logging
from sklearn.preprocessing import StandardScaler
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# File paths
nodes_file_path = 'Graphs/Graph22/nodes_1000.parquet'
edges_file_path = 'Graphs/Graph22/edges_1000.parquet'
json_file_path = 'Graphs/Graph22/first_graph.json'

# Load the graph data from Parquet
logging.info(f"Loading nodes from {nodes_file_path}")
nodes_df = pd.read_parquet(nodes_file_path)
logging.info(f"Loading edges from {edges_file_path}")
edges_df = pd.read_parquet(edges_file_path)

# Normalize node features
scaler = StandardScaler()
nodes_df['value'] = scaler.fit_transform(nodes_df[['value']])

# Get the first graph
first_graph_index = nodes_df['index'].unique()[0]
first_graph_nodes = nodes_df[nodes_df['index'] == first_graph_index]
first_graph_edges = edges_df[edges_df['index'] == first_graph_index]

# Prepare the graph data for JSON
graph_data = {
    'nodes': first_graph_nodes.to_dict(orient='records'),
    'edges': first_graph_edges.to_dict(orient='records')
}

# Save the first graph to JSON file
with open(json_file_path, 'w') as json_file:
    json.dump(graph_data, json_file, indent=4)
logging.info(f"First graph saved to {json_file_path}")

# Log the first row of the training dataset
logging.info("First row of the training dataset (nodes):")
logging.info(first_graph_nodes.head(1).to_dict(orient='records')[0])

logging.info("First row of the training dataset (edges):")
logging.info(first_graph_edges.head(1).to_dict(orient='records')[0])




# File paths
csv_file_path = 'CSVs/train_data.csv'
output_csv_file_path = 'Graphs/Graph22/first_10_rows.csv'

# Load the first 10 rows of the CSV file
logging.info(f"Loading the first 10 rows from {csv_file_path}")
df = pd.read_csv(csv_file_path, nrows=10)

# Print the first 10 rows
logging.info("First 10 rows of the training dataset:")
print(df)

# Ensure the output directory exists
output_dir = os.path.dirname(output_csv_file_path)
os.makedirs(output_dir, exist_ok=True)

# Save the first 10 rows to a new CSV file
df.to_csv(output_csv_file_path, index=False)
logging.info(f"First 10 rows saved to {output_csv_file_path}")