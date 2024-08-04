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
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the enhanced GNN model with more layers and units
class EnhancedGNN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(EnhancedGNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, 128)
        self.conv3 = GCNConv(128, 64)
        self.conv4 = GCNConv(64, num_classes)
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
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv4(x, edge_index)
        # Pooling layer to get a graph-level output
        x = global_mean_pool(x, batch)
        return F.log_softmax(x, dim=1)

# Load data from CSV files and create PyTorch Geometric Data objects
def load_data(file_path):
    df = pd.read_csv(file_path)
    graphs = []
    scaler = StandardScaler()
    for graph_index in df['graph_index'].unique():
        graph_df = df[df['graph_index'] == graph_index]
        x = torch.tensor(scaler.fit_transform(graph_df[['node_value']].values), dtype=torch.float)
        edge_index = torch.tensor([[i, 0] for i in range(1, len(graph_df))], dtype=torch.long).t().contiguous()
        y_value = int(graph_df['tag_value'].iloc[0])
        y = torch.tensor([y_value], dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, y=y)
        graphs.append(data)
    return graphs

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

# Plotting functions
def plot_and_save(train_losses, train_rmse_scores, test_rmse_scores, train_accuracies, test_accuracies):
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join('Graphs/Graph48', 'loss_chart.png'))
    plt.close()

    plt.figure()
    plt.plot(train_rmse_scores, label='Train RMSE')
    plt.plot(test_rmse_scores, label='Test RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend()
    plt.savefig(os.path.join('Graphs/Graph48', 'rmse_chart.png'))
    plt.close()

    plt.figure()
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join('Graphs/Graph48', 'accuracy_chart.png'))
    plt.close()

# Training function
def train(rank, world_size, model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs, checkpoint_path):
    setup(rank, world_size)
    model = DDP(model, device_ids=[rank])

    start_epoch, train_losses, train_rmse_scores, test_rmse_scores, train_accuracies, test_accuracies = load_checkpoint(checkpoint_path, model, optimizer, scheduler)

    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0
        correct_train = 0
        for data in train_loader:
            data = data.to(rank)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs
            pred = output.argmax(dim=1)
            correct_train += (pred == data.y).sum().item()

        train_loss = total_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        train_rmse, train_acc = evaluate(rank, model, train_loader)
        train_rmse_scores.append(train_rmse)
        train_accuracies.append(train_acc)
        
        test_rmse, test_acc = evaluate(rank, model, test_loader)
        test_rmse_scores.append(test_rmse)
        test_accuracies.append(test_acc)

        logging.info(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

        scheduler.step(train_loss)

        # Save checkpoint
        if rank == 0:
            save_checkpoint(epoch, model, optimizer, scheduler, train_losses, train_rmse_scores, test_rmse_scores, train_accuracies, test_accuracies, checkpoint_path)
            torch.save(model.state_dict(), os.path.join('Graphs/Graph48', 'best_model.pt'))

            # Update and save the plots
            plot_and_save(train_losses, train_rmse_scores, test_rmse_scores, train_accuracies, test_accuracies)

    cleanup()
    return train_losses, train_rmse_scores, test_rmse_scores, train_accuracies, test_accuracies

# Evaluation function
def evaluate(rank, model, loader):
    model.eval()
    targets = []
    predictions = []
    with torch.no_grad():
        for data in loader:
            data = data.to(rank)
            output = model(data)
            pred = output.argmax(dim=1)
            targets.extend(data.y.tolist())
            predictions.extend(pred.tolist())

    mse_score = mean_squared_error(targets, predictions)
    rmse_score = np.sqrt(mse_score)
    accuracy = accuracy_score(targets, predictions)
    return rmse_score, accuracy

def main():
    train_file_path = "Graphs/Graph48/train_graphs.csv"
    test_file_path = "Graphs/Graph48/test_graphs.csv"
    train_data = load_data(train_file_path)
    test_data = load_data(test_file_path)

    num_epochs = 200
    batch_size = 64
    learning_rate = 0.001
    weight_decay = 1e-4

    world_size = 2  # Number of nodes

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    num_node_features = 1
    num_classes = len(pd.read_csv(train_file_path)['tag_value'].unique())
    model = EnhancedGNN(num_node_features, num_classes).to(rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, min_lr=1e-6)
    criterion = torch.nn.CrossEntropyLoss()

    checkpoint_path = os.path.join('Graphs/Graph48', 'checkpoint.pt')

    mp.spawn(train,
             args=(world_size, model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs, checkpoint_path),
             nprocs=world_size,
             join=True)

if __name__ == "__main__":
    main()
