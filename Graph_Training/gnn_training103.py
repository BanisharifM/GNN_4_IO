import os
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.distributed import DistributedSampler
import matplotlib.pyplot as plt
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np
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

# Initialize the process group for distributed training
def setup(rank, world_size):
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

# Load data from CSV files and create PyTorch Geometric Data objects
def load_data(file_path):
    df = pd.read_csv(file_path)
    graphs = []
    scaler = StandardScaler()

    # Checking the columns to find the correct target column
    target_column = None
    for col in df.columns:
        if 'tag' in col.lower() or 'value' in col.lower():
            target_column = col
            break
    
    if not target_column:
        raise KeyError("No suitable column found for target values in the CSV file.")

    for graph_index in df['graph_index'].unique():
        graph_df = df[df['graph_index'] == graph_index]
        
        # Extract node features and standardize them
        node_features = graph_df[['node_1_value', 'node_2_value']].values.flatten().reshape(-1, 1)
        node_features = scaler.fit_transform(node_features)
        x = torch.tensor(node_features, dtype=torch.float)

        # Extract edges and remove duplicates (considering undirected graph)
        edges = graph_df[['node_id_1', 'node_id_2']].values
        edge_weight = graph_df['edge_weight'].values
        edge_index = []
        edge_weight_final = []
        
        # Use a dictionary to map string-based node IDs to unique indices
        node_mapping = {}
        current_index = 0
        
        seen_edges = set()
        for i, (n1, n2) in enumerate(edges):
            if n1 not in node_mapping:
                node_mapping[n1] = current_index
                current_index += 1
            if n2 not in node_mapping:
                node_mapping[n2] = current_index
                current_index += 1
            
            idx1 = node_mapping[n1]
            idx2 = node_mapping[n2]
            edge = tuple(sorted((idx1, idx2)))
            
            if edge not in seen_edges:
                seen_edges.add(edge)
                edge_index.append([idx1, idx2])
                edge_index.append([idx2, idx1])  # Add the reverse edge for undirected graph
                edge_weight_final.append(edge_weight[i])
                edge_weight_final.append(edge_weight[i])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(edge_weight_final, dtype=torch.float)

        # Target value
        y_value = float(graph_df[target_column].iloc[0])
        y = torch.tensor([y_value], dtype=torch.float).view(1, -1)  # Ensure target is of shape [1, 1]

        data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight, y=y)
        graphs.append(data)

    return graphs

# Distributed Dataloader
def distributed_dataloader(train_data, test_data, rank, world_size, batch_size):
    train_sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank)
    test_sampler = DistributedSampler(test_data, num_replicas=world_size, rank=rank)

    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(test_data, batch_size=batch_size, sampler=test_sampler)

    return train_loader, test_loader

# Train function modified for DDP
def train_ddp(rank, world_size, model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs, checkpoint_path, update_logs_and_charts):
    setup(rank, world_size)
    
    # Move model to the appropriate device
    model = model.to(rank)
    
    # Wrap the model in DDP
    model = DDP(model, device_ids=[rank])

    # Load checkpoint
    start_epoch, train_losses, train_rmse_scores, test_rmse_scores, train_mae_scores, test_mae_scores, train_r2_scores, test_r2_scores = load_checkpoint(checkpoint_path, model, optimizer, scheduler)
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0
        train_loader.sampler.set_epoch(epoch)  # Ensure different data shuffling each epoch

        for data in train_loader:
            optimizer.zero_grad()
            output = model(data).view(-1)
            loss = criterion(output, data.y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs

        train_loss = total_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # Perform evaluation
        train_rmse, train_mae, train_r2 = evaluate(model, train_loader, epoch, 'train', update_logs_and_charts)
        train_rmse_scores.append(train_rmse)
        train_mae_scores.append(train_mae)
        train_r2_scores.append(train_r2)

        test_rmse, test_mae, test_r2 = evaluate(model, test_loader, epoch, 'test', update_logs_and_charts)
        test_rmse_scores.append(test_rmse)
        test_mae_scores.append(test_mae)
        test_r2_scores.append(test_r2)

        logging.info(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}, Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}, Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}')

        scheduler.step(train_loss)

        # Save checkpoint
        save_checkpoint(epoch, model, optimizer, scheduler, train_losses, train_rmse_scores, test_rmse_scores, train_mae_scores, test_mae_scores, train_r2_scores, test_r2_scores, checkpoint_path)

        # Optionally update logs and charts after each epoch
        if update_logs_and_charts:
            save_plots(train_losses, train_rmse_scores, test_rmse_scores, train_mae_scores, test_mae_scores, train_r2_scores, test_r2_scores)

    cleanup()

def evaluate(model, loader, epoch, phase, update_logs_and_charts):
    model.eval()
    targets = []
    predictions = []
    with torch.no_grad():
        for data in loader:
            output = model(data).view(-1)  # Flatten the output
            pred = output
            targets.extend(data.y.view(-1).tolist())  # Flatten the target
            predictions.extend(pred.tolist())

    mse_score = mean_squared_error(targets, predictions)
    rmse_score = np.sqrt(mse_score)
    mae_score = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    return rmse_score, mae_score, r2

# Main function
if __name__ == "__main__":
    # Define paths to the data files
    train_file_path = "Graphs/Graph103/train_graphs.csv"
    test_file_path = "Graphs/Graph103/test_graphs.csv"

    # Load data
    train_data = load_data(train_file_path)
    test_data = load_data(test_file_path)

    # Set up distributed training
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])

    # Hyperparameters
    num_epochs = 100
    batch_size = 64
    learning_rate = 0.001
    weight_decay = 1e-4

    # Initialize the model
    num_node_features = 1
    model = EnhancedGNN(num_node_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, min_lr=1e-6)
    criterion = torch.nn.L1Loss()

    # Create data loaders
    train_loader, test_loader = distributed_dataloader(train_data, test_data, rank, world_size, batch_size)

    # Set checkpoint path
    checkpoint_path = os.path.join('Graphs/Graph103', 'checkpoint.pt')

    # Train the model
    train_ddp(rank, world_size, model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs, checkpoint_path, update_logs_and_charts=False)
