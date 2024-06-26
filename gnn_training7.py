import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Parameters
num_epochs = 200
batch_size = 32
learning_rate = 0.001  # Adjusted learning rate
weight_decay = 0.0001  # Added weight decay
hidden_dim = 128  # Increased hidden dimension
patience = 20  # Increased patience

logging.info("Starting the graph training process.")

# Load the graph data from Parquet
nodes_file_path = 'Graphs/Graph20/nodes_10000.parquet'
edges_file_path = 'Graphs/Graph20/edges_10000.parquet'
logging.info(f"Loading nodes from {nodes_file_path}")
nodes_df = pd.read_parquet(nodes_file_path)
logging.info(f"Loading edges from {edges_file_path}")
edges_df = pd.read_parquet(edges_file_path)

# Process each graph in the Parquet files
data_list = []
graph_indices = nodes_df['index'].unique()

logging.info("Processing each graph in the Parquet files.")
for index in graph_indices:
    if index % 1000 == 0 and index > 0:
        logging.info(f"Processing graph with index {index}")
    nodes = nodes_df[nodes_df['index'] == index]
    edges = edges_df[edges_df['index'] == index]
    
    # Create a mapping from node IDs to indices
    node_id_to_index = {node_id: idx for idx, node_id in enumerate(nodes['id'])}
    
    # Extract features and labels
    features = nodes['value'].tolist()
    labels = nodes[nodes['id'] == 'tag']['value'].values[0]  # Assuming one 'tag' node per graph
    
    # Create edge index
    edge_index = [[node_id_to_index[edge['source']], node_id_to_index[edge['target']]] for _, edge in edges.iterrows()]
    
    # Convert to torch tensors
    x = torch.tensor(features, dtype=torch.float).unsqueeze(1)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    y = torch.tensor([labels], dtype=torch.long)  # Ensure labels are batched correctly
    
    data = Data(x=x, edge_index=edge_index, y=y)
    data_list.append(data)

logging.info("Creating DataLoader.")
dataloader = DataLoader(data_list, batch_size=batch_size, shuffle=True)

logging.info("Data and labels processed successfully.")

# Determine the number of classes in the target variable
num_classes = len(nodes_df[nodes_df['id'] == 'tag']['value'].unique())
logging.info(f"Number of classes: {num_classes}")

# GNN model definition
class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim)
        self.conv3 = GATConv(hidden_dim, output_dim)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)

logging.info("Initializing model, optimizer, and loss function.")
# Model, optimizer, and loss function
model = GNN(input_dim=1, hidden_dim=hidden_dim, output_dim=num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)  # Learning rate scheduler
criterion = torch.nn.CrossEntropyLoss()

# Training loop
losses = []
accuracies = []
best_accuracy = 0
patience_counter = 0

logging.info("Starting training loop.")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in dataloader:
        optimizer.zero_grad()
        out = model(batch)

        # Identify root nodes using the batch attribute
        root_node_indices = torch.arange(batch.num_graphs)
        
        out = out[root_node_indices]
        target = batch.y.squeeze()  # Ensure the target has the correct shape

        loss = criterion(out, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = out.max(dim=1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    losses.append(avg_loss)
    accuracies.append(accuracy)
    logging.info(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')
    
    # Save the best model based on validation accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        model_path = "Graphs/Graph20/best_model.pt"
        torch.save(model.state_dict(), model_path)
        logging.info(f"Best model saved to {model_path}")
        patience_counter = 0
    else:
        patience_counter += 1

    # Update and save the training metrics plot at each epoch
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(losses) + 1), losses, label='Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(accuracies) + 1), accuracies, label='Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    metrics_path = "Graphs/Graph20/training_metrics.png"
    plt.savefig(metrics_path)
    plt.close()
    logging.info(f"Training metrics plot saved to {metrics_path}")

    # # Early stopping check
    # if patience_counter >= patience:
    #     logging.info("Early stopping triggered.")
    #     break

logging.info("Training completed.")
