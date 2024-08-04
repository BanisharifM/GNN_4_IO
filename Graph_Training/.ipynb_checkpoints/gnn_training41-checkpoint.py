import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import logging
from sklearn.metrics import mean_squared_error
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the enhanced GNN model
class EnhancedGNN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(EnhancedGNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 32)
        self.conv2 = GCNConv(32, 64)
        self.conv3 = GCNConv(64, num_classes)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
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
train_file_path = "Graphs/Graph41/train_graphs.csv"
test_file_path = "Graphs/Graph41/test_graphs.csv"

# Load the data
train_data = load_data(train_file_path)
test_data = load_data(test_file_path)

# Hyperparameters
num_epochs = 200
batch_size = 32
learning_rate = 0.01
weight_decay = 5e-4
step_size = 20
gamma = 0.5

# Create data loader
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Initialize the model, optimizer, and loss function
num_node_features = 1
num_classes = len(pd.read_csv(train_file_path)['tag_value'].unique())
model = EnhancedGNN(num_node_features, num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
criterion = torch.nn.CrossEntropyLoss()

# Training function
def train(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs):
    train_losses = []
    test_rmse_scores = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for data in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs

        train_loss = total_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        test_rmse = evaluate(model, test_loader)
        test_rmse_scores.append(test_rmse)

        logging.info(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Test RMSE: {test_rmse:.4f}')

        scheduler.step()

        # Save the model at the end of each epoch
        torch.save(model.state_dict(), os.path.join('Graphs/Graph41', 'best_model.pt'))

    return train_losses, test_rmse_scores

# Evaluation function
def evaluate(model, test_loader):
    model.eval()
    targets = []
    predictions = []
    with torch.no_grad():
        for data in test_loader:
            output = model(data)
            pred = output.argmax(dim=1)
            targets.extend(data.y.tolist())
            predictions.extend(pred.tolist())

    rmse_score = mean_squared_error(targets, predictions, squared=False)
    return rmse_score

# Train the model
train_losses, test_rmse_scores = train(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs)

# Plot and save the loss and accuracy charts
plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join('Graphs/Graph41', 'loss_chart.png'))

plt.figure()
plt.plot(test_rmse_scores, label='Test RMSE')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.legend()
plt.savefig(os.path.join('Graphs/Graph41', 'rmse_chart.png'))
