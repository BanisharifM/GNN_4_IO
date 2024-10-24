import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv, global_max_pool
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import signal
import sys
from sklearn.model_selection import train_test_split

base_dir = "Graphs/Graph126"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EnhancedGNNWithEmbeddings(torch.nn.Module):
    def __init__(self, num_node_features):
        super(EnhancedGNNWithEmbeddings, self).__init__()
        self.conv1 = GCNConv(num_node_features, 128)
        self.conv2 = GCNConv(128, 256)
        self.conv3 = GCNConv(256, 128)
        self.conv4 = GCNConv(128, 64)  
        # self.attention_conv = GATConv(64, 64)

        # Fully connected layers for final prediction
        self.fc1 = torch.nn.Linear(128, 64)
        self.fc2 = torch.nn.Linear(64, 1)
        # self.dropout = torch.nn.Dropout(p=0.5) #overfiting
        self.batch_norm1 = torch.nn.BatchNorm1d(128)
        self.batch_norm2 = torch.nn.BatchNorm1d(64)

    def forward(self, data):
        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_weight, data.batch
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.batch_norm1(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)

        x = self.conv4(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)

        # Attention layer
        node_embeddings = self.attention_conv(x, edge_index)
        x = F.relu(node_embeddings)

        # Pooling layer to aggregate node features across the graph
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)

        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        # Returning the output and node embeddings
        return x, node_embeddings

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
    
    # Loop through unique graph indices
    for graph_index in df['graph_index'].unique():
        graph_df = df[df['graph_index'] == graph_index]
        
        # Extract node features and check for NaNs
        node_features = graph_df[['node_1_value', 'node_2_value']].values.flatten().reshape(-1, 1)
        
        if np.isnan(node_features).any():
            logging.error(f"NaN values found in node features for graph index {graph_index}. Skipping this graph.")
            continue  # Skip this graph
        
        node_features = scaler.fit_transform(node_features)  # Standardize the features
        x = torch.tensor(node_features, dtype=torch.float)
        
        # Extract edges and check for missing or invalid node IDs (strings)
        edges = graph_df[['node_id_1', 'node_id_2']].values
        
        # Check if any of the node IDs are NaN or empty
        if graph_df[['node_id_1', 'node_id_2']].isnull().values.any():
            logging.error(f"NaN or missing values found in node IDs for graph index {graph_index}. Skipping this graph.")
            continue  # Skip this graph
        
        edge_weight = graph_df['edge_weight'].values
        edge_index = []
        edge_weight_final = []
        
        # Use a dictionary to map string-based node IDs to unique indices
        node_mapping = {}
        current_index = 0
        
        seen_edges = set()
        for i, (n1, n2) in enumerate(edges):
            # Ensure both node IDs are valid (non-empty strings)
            if not n1 or not n2:
                logging.error(f"Invalid node ID(s) in graph index {graph_index}. Skipping this edge.")
                continue  # Skip this edge if node IDs are missing
            
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
        
        if np.isnan(y_value):
            logging.error(f"NaN value found in target column for graph index {graph_index}. Skipping this graph.")
            continue  # Skip this graph
        
        y = torch.tensor([y_value], dtype=torch.float).view(1, -1)  # Ensure target is of shape [1, 1]

        # Create the data object and add the graph_index as an attribute
        data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight, y=y)
        data.graph_index = graph_index  # Add graph_index as an attribute

        graphs.append(data)

    return graphs

# Load tag data from CSV
def load_tags(tags_file):
    tags_df = pd.read_csv(tags_file)
    tags_dict = pd.Series(tags_df['tag'].values, index=tags_df['graph_index']).to_dict()  # Create a mapping from graph_index to tag
    return tags_dict

# Define paths to the data files
train_file_path = os.path.join(base_dir, "train_graphs.csv")
val_file_path = os.path.join(base_dir, "val_graphs.csv")
test_file_path = os.path.join(base_dir, "test_graphs.csv")
train_tags_file = os.path.join(base_dir, "train_tags.csv")
val_tags_file = os.path.join(base_dir, "val_tags.csv")
test_tags_file = os.path.join(base_dir, "test_tags.csv")

# Load the data
train_data = load_data(train_file_path)
val_data = load_data(val_file_path)
test_data = load_data(test_file_path)

# Load the tags
tags_train = load_tags(train_tags_file)
tags_val = load_tags(val_tags_file)
tags_test = load_tags(test_tags_file)

# Hyperparameters
num_epochs = 100
batch_size = 128
# learning_rate = 0.001
# learning_rate = 0.0001
# learning_rate = 0.00005
weight_decay = 1e-4

# Create data loaders
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Initialize the model, optimizer, and loss function
num_node_features = len(train_data[0].x[0])  # Number of node features based on dataset
model = EnhancedGNNWithEmbeddings(num_node_features)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, min_lr=1e-6)
criterion = torch.nn.L1Loss()

# Updated checkpointing function to save model, optimizer, and scheduler states
def save_checkpoint(epoch, model, optimizer, scheduler, train_losses, val_losses, train_rmse_scores, val_rmse_scores, test_rmse_scores, train_mae_scores, val_mae_scores, test_mae_scores, train_r2_scores, val_r2_scores, test_r2_scores, checkpoint_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_rmse_scores': train_rmse_scores,
        'val_rmse_scores': val_rmse_scores,
        'test_rmse_scores': test_rmse_scores,
        'train_mae_scores': train_mae_scores,
        'val_mae_scores': val_mae_scores,
        'test_mae_scores': test_mae_scores,
        'train_r2_scores': train_r2_scores,
        'val_r2_scores': val_r2_scores,
        'test_r2_scores': test_r2_scores,
    }, checkpoint_path)

# Updated function to load model, optimizer, and scheduler states safely
def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        
        # Load model state dict with partial matching (handles new layers)
        model_state_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if k in model_state_dict and v.size() == model_state_dict[k].size()}
        model_state_dict.update(pretrained_dict)
        model.load_state_dict(model_state_dict)

        # Load optimizer state if matching
        if optimizer and 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except ValueError as e:
                logging.warning(f"Optimizer state could not be loaded due to a mismatch: {e}")
                logging.warning("Optimizer state will be reset. Continuing with the new optimizer settings.")
                # You can reset the optimizer if needed, or skip the optimizer state

        # Load scheduler state if matching
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Retrieve the saved epoch and losses
        epoch = checkpoint.get('epoch', 0)
        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        train_rmse_scores = checkpoint.get('train_rmse_scores', [])
        val_rmse_scores = checkpoint.get('val_rmse_scores', [])
        test_rmse_scores = checkpoint.get('test_rmse_scores', [])
        train_mae_scores = checkpoint.get('train_mae_scores', [])
        val_mae_scores = checkpoint.get('val_mae_scores', [])
        test_mae_scores = checkpoint.get('test_mae_scores', [])
        train_r2_scores = checkpoint.get('train_r2_scores', [])
        val_r2_scores = checkpoint.get('val_r2_scores', [])
        test_r2_scores = checkpoint.get('test_r2_scores', [])

        logging.info(f'Loaded checkpoint from {checkpoint_path}, starting at epoch {epoch + 1}')
        return epoch, train_losses, val_losses, train_rmse_scores, val_rmse_scores, test_rmse_scores, train_mae_scores, val_mae_scores, test_mae_scores, train_r2_scores, val_r2_scores, test_r2_scores
    else:
        return 0, [], [], [], [], [], [], [], [], [], [], []

# Validation function similar to the evaluate function
def evaluate_validation(model, val_loader, epoch, update_logs_and_charts, tags_val):
    model.eval()
    val_targets = []
    val_predictions = []

    with torch.no_grad():
        for data in val_loader:
            output, _ = model(data)  # Unpack the tuple, output is the first element
            output = output.view(-1)  # Flatten the output to match the target shape
            
            # Get the graph indices for the current batch
            graph_indices = data.batch.unique().tolist()
            target_tags = [tags_val.get(graph_index) for graph_index in graph_indices]
            
            if any(tag is None for tag in target_tags):
                raise ValueError(f"Missing tags for graph indices {graph_indices}")

            # Convert the target tags to a tensor
            target_tag_tensor = torch.tensor(target_tags, dtype=torch.float).to(output.device)
            
            # Append the entire batch of predictions and targets to the lists
            val_predictions.extend(output.tolist())  # Extend by adding the whole batch, not a single item
            val_targets.extend(target_tag_tensor.tolist())  # Similarly extend the targets

    # Calculate the metrics over the entire validation set
    val_rmse = np.sqrt(mean_squared_error(val_targets, val_predictions))
    val_mae = mean_absolute_error(val_targets, val_predictions)
    val_r2 = r2_score(val_targets, val_predictions)

    return val_rmse, val_mae, val_r2

# Training function with validation
def train(model, train_loader, val_loader, test_loader, criterion, optimizer, scheduler, num_epochs, checkpoint_path, update_logs_and_charts):
    # Load from the checkpoint if it exists
    start_epoch, train_losses, val_losses, train_rmse_scores, val_rmse_scores, test_rmse_scores, train_mae_scores, val_mae_scores, test_mae_scores, train_r2_scores, val_r2_scores, test_r2_scores = load_checkpoint(checkpoint_path, model, optimizer, scheduler)

    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0

        # Training loop
        for data in train_loader:
            optimizer.zero_grad()

            # Forward pass
            output, _ = model(data)  # Unpack the tuple, output is the first element
            output = output.view(-1)  # Then apply .view() to the output

            # Get the graph indices and retrieve the corresponding tags for all graphs in the batch
            graph_indices = data.batch.unique().tolist()
            target_tags = [tags_train.get(graph_index) for graph_index in graph_indices]

            if any(tag is None for tag in target_tags):
                raise ValueError(f"Missing tags for graph indices {graph_indices}")

            # Convert the target tags to a tensor and ensure it has the same shape as output
            target_tag_tensor = torch.tensor(target_tags, dtype=torch.double).to(output.device)  # Should match output size [batch_size]

            # Compute loss
            loss = criterion(output, target_tag_tensor)
            loss.backward()

            optimizer.step()
            total_loss += loss.item() * data.num_graphs

        train_loss = total_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # Evaluate model on train, validation, and test sets
        train_rmse, train_mae, train_r2 = evaluate(model, train_loader, epoch, 'train', update_logs_and_charts, tags_train)
        train_rmse_scores.append(train_rmse)
        train_mae_scores.append(train_mae)
        train_r2_scores.append(train_r2)
        
        # val_rmse, val_mae, val_r2 = evaluate(model, val_loader, epoch, 'val', update_logs_and_charts, tags_val)
        val_rmse, val_mae, val_r2 = evaluate_validation(model, val_loader, epoch, update_logs_and_charts, tags_val)
        val_rmse_scores.append(val_rmse)
        val_mae_scores.append(val_mae)
        val_r2_scores.append(val_r2)
        
        test_rmse, test_mae, test_r2 = evaluate(model, test_loader, epoch, 'test', update_logs_and_charts, tags_test)
        test_rmse_scores.append(test_rmse)
        test_mae_scores.append(test_mae)
        test_r2_scores.append(test_r2)

        logging.info(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train RMSE: {train_rmse:.4f}, Val RMSE: {val_rmse:.4f}, Test RMSE: {test_rmse:.4f}, Train MAE: {train_mae:.4f}, Val MAE: {val_mae:.4f}, Test MAE: {test_mae:.4f}, Train R²: {train_r2:.4f}, Val R²: {val_r2:.4f}, Test R²: {test_r2:.4f}')

        # Adjust the learning rate based on the scheduler
        scheduler.step(train_loss)

        # Save checkpoint
        save_checkpoint(epoch, model, optimizer, scheduler, train_losses, val_losses, train_rmse_scores, val_rmse_scores, test_rmse_scores, train_mae_scores, val_mae_scores, test_mae_scores, train_r2_scores, val_r2_scores, test_r2_scores, checkpoint_path)

        # Optionally update logs and charts after each epoch
        if update_logs_and_charts:
            save_plots(train_losses, val_losses, train_rmse_scores, val_rmse_scores, test_rmse_scores, train_mae_scores, val_mae_scores, test_mae_scores, train_r2_scores, val_r2_scores, test_r2_scores)

    # Always update logs and charts at the end of training
    if not update_logs_and_charts:
        save_plots(train_losses, val_losses, train_rmse_scores, val_rmse_scores, test_rmse_scores, train_mae_scores, val_mae_scores, test_mae_scores, train_r2_scores, val_r2_scores, test_r2_scores)

    # Save the model at the end of training
    torch.save(model.state_dict(), os.path.join(base_dir, 'best_model.pt'))

    return train_losses, val_losses, train_rmse_scores, val_rmse_scores, test_rmse_scores, train_mae_scores, val_mae_scores, test_mae_scores, train_r2_scores, val_r2_scores, test_r2_scores

def save_plots(train_losses, train_rmse_scores, test_rmse_scores, train_mae_scores, test_mae_scores, train_r2_scores, test_r2_scores):
    plot_dir = base_dir
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, 'loss_chart.png'))
    plt.close()

    plt.figure()
    plt.plot(train_rmse_scores, label='Train RMSE')
    plt.plot(test_rmse_scores, label='Test RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, 'rmse_chart.png'))
    plt.close()

    plt.figure()
    plt.plot(train_mae_scores, label='Train MAE')
    plt.plot(test_mae_scores, label='Test MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, 'mae_chart.png'))
    plt.close()

    plt.figure()
    plt.plot(train_r2_scores, label='Train R²')
    plt.plot(test_r2_scores, label='Test R²')
    plt.xlabel('Epoch')
    plt.ylabel('R² Score')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, 'r2_chart.png'))
    plt.close()

# Evaluation function
def evaluate(model, loader, epoch, phase, update_logs_and_charts, tags_dict):
    model.eval()
    targets = []
    predictions = []
    total_loss = 0

    with torch.no_grad():
        for data in loader:
            # output = model(data).view(-1)  # The model outputs one value per graph, so we flatten it
            output, _ = model(data)  # Unpack the tuple, output is the first element
            output = output.view(-1)  # Then apply .view() to the output
            
            # Get graph indices for the batch and the corresponding tags
            graph_indices = data.batch.unique().tolist()
            target_tags = [tags_dict.get(graph_index) for graph_index in graph_indices]

            # Check for missing target tags
            if any(tag is None for tag in target_tags):
                raise ValueError(f"Missing tags for graph indices {graph_indices}")

            target_tag_tensor = torch.tensor(target_tags, dtype=torch.double).to(output.device)  # Convert to tensor

            # Add predictions and targets
            predictions.extend(output.tolist())  # Add all predictions in the batch
            targets.extend(target_tag_tensor.tolist())  # Add all corresponding targets

            # Compute the loss
            loss = criterion(output, target_tag_tensor)
            total_loss += loss.item() * data.num_graphs

    # Compute evaluation metrics
    mse_score = mean_squared_error(targets, predictions)
    rmse_score = np.sqrt(mse_score)
    mae_score = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)

    if update_logs_and_charts:
        log_predictions(predictions, targets, epoch, phase)

    return rmse_score, mae_score, r2

# Define signal handler for graceful termination
def signal_handler(sig, frame):
    print('Graceful termination initiated...')
    save_checkpoint(epoch, model, optimizer, scheduler, train_losses, train_rmse_scores, test_rmse_scores, train_mae_scores, test_mae_scores, train_r2_scores, test_r2_scores, checkpoint_path)
    torch.save(model.state_dict(), os.path.join(base_dir, 'best_model.pt'))
    sys.exit(0)

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)

# Define checkpoint path
checkpoint_path = os.path.join(base_dir, 'checkpoint.pt')

# Variable to control logging and chart updates
update_logs_and_charts = False  # Set to False to update only at the end

# Train the model
train_losses, val_losses, train_rmse_scores, val_rmse_scores, test_rmse_scores, train_mae_scores, val_mae_scores, test_mae_scores, train_r2_scores, val_r2_scores, test_r2_scores = train(
    model, train_loader, val_loader, test_loader, criterion, optimizer, scheduler, num_epochs, checkpoint_path, update_logs_and_charts)

# Ensure final predictions are logged
if not update_logs_and_charts:
    evaluate(model, train_loader, num_epochs, 'train', True, tags_train)
    evaluate(model, test_loader, num_epochs, 'test', True, tags_test)
