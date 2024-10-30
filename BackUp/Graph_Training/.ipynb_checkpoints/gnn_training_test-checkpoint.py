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
import pickle

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
train_file_path = "Graphs/Graph_Test/train_graphs.csv"

# Load the data
train_data = load_data(train_file_path)

# Hyperparameters
num_epochs = 200
batch_size = 32
learning_rate = 0.01
weight_decay = 5e-4
step_size = 20
gamma = 0.5

# Create data loader
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Initialize the model, optimizer, and loss function
num_node_features = 1
num_classes = len(pd.read_csv(train_file_path)['tag_value'].unique())
model = EnhancedGNN(num_node_features, num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
criterion = torch.nn.CrossEntropyLoss()

# Function to save the training state
def save_checkpoint(epoch, model, optimizer, scheduler, train_losses, train_accuracies, filepath):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_losses': train_losses,
        'train_accuracies': train_accuracies
    }
    torch.save(state, filepath)

# Function to load the training state
def load_checkpoint(filepath):
    if os.path.exists(filepath):
        state = torch.load(filepath)
        return state
    return None

# Training function
def train(model, train_loader, criterion, optimizer, scheduler, num_epochs, checkpoint_path):
    state = load_checkpoint(checkpoint_path)
    start_epoch = 0
    train_losses = []
    train_accuracies = []

    if state:
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        scheduler.load_state_dict(state['scheduler_state_dict'])
        start_epoch = state['epoch'] + 1
        train_losses = state['train_losses']
        train_accuracies = state['train_accuracies']
        logging.info(f'Resuming from epoch {start_epoch}')

    for epoch in range(start_epoch, num_epochs):
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

        logging.info(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}')

        scheduler.step()

        # Save the model and checkpoint at the end of each epoch
        torch.save(model.state_dict(), os.path.join('Graphs/Graph_Test', 'best_model.pt'))
        save_checkpoint(epoch, model, optimizer, scheduler, train_losses, train_accuracies, checkpoint_path)

        # Plot and save the loss and accuracy charts after each epoch
        plt.figure()
        plt.plot(train_losses, label='Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join('Graphs/Graph_Test', 'loss_chart.png'))
        plt.close()

        plt.figure()
        plt.plot(train_accuracies, label='Train Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(os.path.join('Graphs/Graph_Test', 'accuracy_chart.png'))
        plt.close()

    return train_losses, train_accuracies

# Path to save the checkpoint
checkpoint_path = os.path.join('Graphs/Graph_Test', 'checkpoint.pth')

# Train the model
train_losses, train_accuracies = train(model, train_loader, criterion, optimizer, scheduler, num_epochs, checkpoint_path)
