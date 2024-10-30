import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the GNN model
class GNN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        # Pooling layer to get a graph-level output
        x = global_mean_pool(x, batch)
        return F.log_softmax(x, dim=1)

# Load data from CSV files and create PyTorch Geometric Data objects
def load_data(file_path):
    df = pd.read_csv(file_path)
    graphs = []
    for graph_index in df['graph_index'].unique():
        graph_df = df[df['graph_index'] == graph_index]
        x = torch.tensor(graph_df[['node_value']].values, dtype=torch.float)
        edge_index = torch.tensor([[i, 0] for i in range(1, len(graph_df))], dtype=torch.long).t().contiguous()
        y = torch.tensor([graph_df['tag_value'].iloc[0]], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, y=y)
        graphs.append(data)
    return graphs

# Define paths to the data files
train_file_path = "Graphs/Graph29/train_graphs.csv"
val_file_path = "Graphs/Graph29/val_graphs.csv"

# Load the data
train_data = load_data(train_file_path)
val_data = load_data(val_file_path)

# Hyperparameters
num_epochs = 200
batch_size = 32
learning_rate = 0.01

# Create data loaders
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

# Initialize the model, optimizer, and loss function
num_node_features = 1
num_classes = len(pd.read_csv(train_file_path)['tag_value'].unique())
model = GNN(num_node_features, num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

# Training and validation function
def train_and_validate(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        for data in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs
            pred = output.argmax(dim=1)
            correct += (pred == data.y).sum().item()

        train_loss = total_loss / len(train_loader.dataset)
        train_accuracy = correct / len(train_loader.dataset)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        model.eval()
        total_loss = 0
        correct = 0
        with torch.no_grad():
            for data in val_loader:
                output = model(data)
                loss = criterion(output, data.y)
                total_loss += loss.item() * data.num_graphs
                pred = output.argmax(dim=1)
                correct += (pred == data.y).sum().item()

        val_loss = total_loss / len(val_loader.dataset)
        val_accuracy = correct / len(val_loader.dataset)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        logging.info(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join('Graphs/Graph29', 'best_model.pt'))

    return train_losses, val_losses, train_accuracies, val_accuracies

# Train and validate the model
train_losses, val_losses, train_accuracies, val_accuracies = train_and_validate(model, train_loader, val_loader, criterion, optimizer, num_epochs)

# Plot and save the loss and accuracy charts
plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join('Graphs/Graph29', 'loss_chart.png'))

plt.figure()
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(os.path.join('Graphs/Graph29', 'accuracy_chart.png'))
