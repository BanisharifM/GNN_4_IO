import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import logging
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Parameters
num_epochs = 200
batch_size = 32
learning_rate = 0.001
weight_decay = 0.0001
hidden_dim = 128
patience = 20
dropout_rate = 0.5

logging.info("Starting the graph training process.")

# Load the graph data from CSV files
train_nodes_file = 'Graphs/Graph24/train_data_nodes.csv'
val_nodes_file = 'Graphs/Graph24/val_data_nodes.csv'
edges_file = 'Graphs/Graph24/edges.csv'
yaml_path = 'Graphs/Graph24/graph_structure.yaml'

def load_graph_structure(yaml_path):
    logging.info(f"Loading graph structure from {yaml_path}")
    with open(yaml_path, 'r') as file:
        graph_structure = yaml.safe_load(file)
    return graph_structure

def load_edges(edges_file):
    logging.info(f"Loading edges from {edges_file}")
    edges_df = pd.read_csv(edges_file)
    return edges_df

def load_graph_data(nodes_file, edges_df, graph_structure):
    logging.info(f"Loading nodes from {nodes_file}")
    nodes_df = pd.read_csv(nodes_file)

    # Normalize node features
    scaler = StandardScaler()
    nodes_df['value'] = scaler.fit_transform(nodes_df[['value']])

    data_list = []
    graph_indices = nodes_df['index'].unique()

    logging.info(f"Processing each graph in {nodes_file}.")
    for index in graph_indices:
        if index % 1000 == 0 and index > 0:
            logging.info(f"Processing graph with index {index}")
        nodes = nodes_df[nodes_df['index'] == index]
        
        # Create a mapping from node IDs to indices
        node_id_to_index = {node_id: idx for idx, node_id in enumerate(nodes['id'])}
        
        # Adjust edge indices for the current graph based on YAML structure
        adjusted_edge_index = []
        for _, edge in edges_df.iterrows():
            if edge['source'] in node_id_to_index and edge['target'] in node_id_to_index:
                adjusted_edge_index.append([node_id_to_index[edge['source']], node_id_to_index[edge['target']]])
        
        # Extract features and labels
        features = nodes[nodes['id'] != 'tag']['value'].values
        labels = nodes[nodes['id'] == 'tag']['value'].values[0]  # Assuming one 'tag' node per graph
        
        # Convert to torch tensors
        x = torch.tensor(features, dtype=torch.float).unsqueeze(1)
        edge_index = torch.tensor(adjusted_edge_index, dtype=torch.long).t().contiguous()
        y = torch.tensor([labels], dtype=torch.long)
        
        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)

    return data_list

# Load graph structure and edges from YAML and CSV files
graph_structure = load_graph_structure(yaml_path)
edges_df = load_edges(edges_file)

# Load training and validation data
train_data = load_graph_data(train_nodes_file, edges_df, graph_structure)
val_data = load_graph_data(val_nodes_file, edges_df, graph_structure)

logging.info("Creating DataLoader.")
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

logging.info("Data and labels processed successfully.")

# Determine the number of classes in the target variable
num_classes = len(pd.read_csv(train_nodes_file)[pd.read_csv(train_nodes_file)['id'] == 'tag']['value'].unique())
logging.info(f"Number of classes: {num_classes}")

# GNN model definition
class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.5):
        super(GNN, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = torch.mean(x, dim=0, keepdim=True)  # Aggregate node features
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

logging.info("Initializing model, optimizer, and loss function.")
# Model, optimizer, and loss function
model = GNN(input_dim=1, hidden_dim=hidden_dim, output_dim=num_classes, dropout_rate=dropout_rate)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
losses = []
accuracies = []
val_losses = []
val_accuracies = []
best_accuracy = 0
patience_counter = 0

logging.info("Starting training loop.")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in train_loader:
        optimizer.zero_grad()
        out = model(batch)

        # Identify root nodes using the batch attribute
        root_node_indices = torch.arange(batch.num_graphs)
        
        out = out[root_node_indices]
        target = batch.y.squeeze()

        loss = criterion(out, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = out.max(dim=1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    losses.append(avg_loss)
    accuracies.append(accuracy)

    # Validation step
    model.eval()
    val_total_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for batch in val_loader:
            out = model(batch)
            root_node_indices = torch.arange(batch.num_graphs)
            out = out[root_node_indices]
            target = batch.y.squeeze()
            val_loss = criterion(out, target)
            val_total_loss += val_loss.item()
            _, val_predicted = out.max(dim=1)
            val_total += target.size(0)
            val_correct += val_predicted.eq(target).sum().item()

    avg_val_loss = val_total_loss / len(val_loader)
    val_accuracy = val_correct / val_total
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_accuracy)
    
    logging.info(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
    
    # Save the best model based on validation accuracy
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        model_path = "Graphs/Graph24/best_model.pt"
        torch.save(model.state_dict(), model_path)
        logging.info(f"Best model saved to {model_path}")
        patience_counter = 0
    else:
        patience_counter += 1

    # Update and save the training metrics plot at each epoch
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(losses) + 1), losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(accuracies) + 1), accuracies, label='Training Accuracy')
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    metrics_path = "Graphs/Graph24/training_metrics.png"
    plt.savefig(metrics_path)
    plt.close()
    logging.info(f"Training metrics plot saved to {metrics_path}")

    # Early stopping check
    if patience_counter >= patience:
        logging.info("Early stopping triggered.")
        break

logging.info("Training completed.")
