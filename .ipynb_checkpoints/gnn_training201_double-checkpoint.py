import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import json
import signal
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the enhanced GNN model with more layers and units
class EnhancedGNN(torch.nn.Module):
    def __init__(self, num_node_features):
        super(EnhancedGNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, 128)
        self.conv3 = GCNConv(128, 64)
        self.conv4 = GCNConv(64, 1)  # Output a single value per graph
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, data):
        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_weight, data.batch
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv3(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv4(x, edge_index, edge_weight=edge_weight)
        x = global_mean_pool(x, batch)
        return x  # Output a single value for each graph

# Load data from CSV files and create PyTorch Geometric Data objects
def load_data(file_path, counter_mapping_path):
    df = pd.read_csv(file_path)
    graphs = []
    scaler = StandardScaler()

    # Load counter to integer mapping from the JSON file
    with open(counter_mapping_path, 'r') as file:
        counter_mapping = json.load(file)

    # Target node ID for prediction ("tag")
    target_node_id = 46

    for graph_index in df['graph_index'].unique():
        graph_df = df[df['graph_index'] == graph_index]
        
        # Extract node features and standardize them
        node_features = graph_df[['node_1_value', 'node_2_value']].values.flatten().reshape(-1, 1)
        node_features = scaler.fit_transform(node_features)
        x = torch.tensor(node_features, dtype=torch.double)

        # Extract edges and map node IDs to integers
        edges = graph_df[['node_id_1', 'node_id_2']].values
        edge_weight = graph_df['edge_weight'].values
        edge_index = []
        edge_weight_final = []
        
        seen_edges = set()
        for i, (n1, n2) in enumerate(edges):
            # Replace node IDs with their mapped integer values
            node_id_1 = counter_mapping.get(str(n1))
            node_id_2 = counter_mapping.get(str(n2))

            if node_id_1 is not None and node_id_2 is not None:
                edge = tuple(sorted((node_id_1, node_id_2)))
                
                if edge not in seen_edges:
                    seen_edges.add(edge)
                    edge_index.append([node_id_1, node_id_2])
                    edge_index.append([node_id_2, node_id_1])  # Add the reverse edge for undirected graph
                    edge_weight_final.append(edge_weight[i])
                    edge_weight_final.append(edge_weight[i])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(edge_weight_final, dtype=torch.double)

        # Target value (for 'tag' or node 46)
        y_value = float(graph_df[graph_df['node_id_1'] == target_node_id]['node_1_value'].iloc[0])
        y = torch.tensor([y_value], dtype=torch.double).view(1, -1)  # Ensure target is of shape [1, 1]

        data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight, y=y)
        graphs.append(data)

    return graphs

# Define paths to the data files
train_file_path = "Graphs/Graph200/double_batch128_100,000/train_graphs.csv"
test_file_path = "Graphs/Graph200/double_batch128_100,000/test_graphs.csv"
counter_mapping_file = "Graphs/Graph201/counter_mapping.json"

# Load the data
train_data = load_data(train_file_path, counter_mapping_file)
test_data = load_data(test_file_path, counter_mapping_file)

# Hyperparameters
num_epochs = 200
batch_size = 128
learning_rate = 0.001
weight_decay = 1e-4

# Create data loader
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Initialize the model, optimizer, and loss function
num_node_features = 1  # As we only have one feature per node
model = EnhancedGNN(num_node_features).double()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, min_lr=1e-6)
criterion = torch.nn.L1Loss()

# Signal handler for graceful termination
def signal_handler(sig, frame):
    print('Graceful termination initiated...')
    torch.save(model.state_dict(), os.path.join('Graphs/Graph200/double_batch128_100,000', 'best_model.pt'))
    sys.exit(0)

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)

# Define checkpoint path
checkpoint_path = os.path.join('Graphs/Graph200/double_batch128_100,000', 'checkpoint.pt')

# Train the model
train_losses, train_rmse_scores, test_rmse_scores, train_mae_scores, test_mae_scores, train_r2_scores, test_r2_scores = train(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs, checkpoint_path, False)

# Ensure final predictions are logged
evaluate(model, train_loader, num_epochs, 'train', True)
evaluate(model, test_loader, num_epochs, 'test', True)
