import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import logging
import matplotlib.pyplot as plt
import signal
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Custom Dataset class for IO data
class IODataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)

# Define the neural network model
class IONeuralNetwork(nn.Module):
    def __init__(self, input_dim):
        super(IONeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        x = self.dropout(x)
        x = self.fc5(x)
        return x

# Function to load and preprocess the data
def load_data(file_path):
    df = pd.read_csv(file_path)
    
    # Split the data into features (X) and target (y)
    X = df.iloc[:, :-1].values  # All columns except the last one
    y = df.iloc[:, -1].values   # The last column ('tag')
    
    # Normalize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y

# Checkpointing function
def save_checkpoint(epoch, model, optimizer, scheduler, train_losses, train_rmse_scores, test_rmse_scores, train_mae_scores, test_mae_scores, train_r2_scores, test_r2_scores, checkpoint_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_losses': train_losses,
        'train_rmse_scores': train_rmse_scores,
        'test_rmse_scores': test_rmse_scores,
        'train_mae_scores': train_mae_scores,
        'test_mae_scores': test_mae_scores,
        'train_r2_scores': train_r2_scores,
        'test_r2_scores': test_r2_scores,
    }, checkpoint_path)

def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)  # Use the default (weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        train_losses = checkpoint['train_losses']
        train_rmse_scores = checkpoint['train_rmse_scores']
        test_rmse_scores = checkpoint['test_rmse_scores']
        train_mae_scores = checkpoint['train_mae_scores']
        test_mae_scores = checkpoint['test_mae_scores']
        train_r2_scores = checkpoint['train_r2_scores']
        test_r2_scores = checkpoint['test_r2_scores']
        logging.info(f'Loaded checkpoint from {checkpoint_path}, starting at epoch {epoch + 1}')
        return epoch, train_losses, train_rmse_scores, test_rmse_scores, train_mae_scores, test_mae_scores, train_r2_scores, test_r2_scores
    else:
        return 0, [], [], [], [], [], [], []

def log_predictions(predictions, targets, epoch, phase):
    log_dir = 'NN/NN_103/logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, f'{phase}_predictions.csv')

    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
        new_df = pd.DataFrame({
            'Epoch': [epoch] * len(predictions),
            'Target': targets,
            'Prediction': predictions
        })
        df = pd.concat([df, new_df], ignore_index=True)
    else:
        df = pd.DataFrame({
            'Epoch': [epoch] * len(predictions),
            'Target': targets,
            'Prediction': predictions
        })
    
    df.to_csv(log_file, index=False)

# Training function
def train(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs, checkpoint_path, update_logs_and_charts):
    start_epoch, train_losses, train_rmse_scores, test_rmse_scores, train_mae_scores, test_mae_scores, train_r2_scores, test_r2_scores = load_checkpoint(checkpoint_path, model, optimizer, scheduler)

    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch).view(-1)  # Flatten the output
            loss = criterion(outputs, y_batch)  # Flatten the target
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X_batch.size(0)

        train_loss = total_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
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
            # Save the plots
            save_plots(train_losses, train_rmse_scores, test_rmse_scores, train_mae_scores, test_mae_scores, train_r2_scores, test_r2_scores)

    # Always update logs and charts at the end of training
    if not update_logs_and_charts:
        save_plots(train_losses, train_rmse_scores, test_rmse_scores, train_mae_scores, test_mae_scores, train_r2_scores, test_r2_scores)

    # Save the model at the end of training
    torch.save(model.state_dict(), os.path.join('NN/NN_103', 'best_model.pt'))

    return train_losses, train_rmse_scores, test_rmse_scores, train_mae_scores, test_mae_scores, train_r2_scores, test_r2_scores

def save_plots(train_losses, train_rmse_scores, test_rmse_scores, train_mae_scores, test_mae_scores, train_r2_scores, test_r2_scores):
    plot_dir = 'NN/NN_103'
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
def evaluate(model, loader, epoch, phase, update_logs_and_charts):
    model.eval()
    targets = []
    predictions = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            outputs = model(X_batch).view(-1)  # Flatten the output
            predictions.extend(outputs.tolist())
            targets.extend(y_batch.tolist())

    mse_score = mean_squared_error(targets, predictions)
    rmse_score = np.sqrt(mse_score)
    mae_score = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)

    # Log predictions and targets to a file if updating logs and charts
    if update_logs_and_charts:
        log_predictions(predictions, targets, epoch, phase)

    return rmse_score, mae_score, r2

# Signal handler for graceful termination
def signal_handler(sig, frame):
    print('Graceful termination initiated...')
    save_checkpoint(epoch, model, optimizer, scheduler, train_losses, train_rmse_scores, test_rmse_scores, train_mae_scores, test_mae_scores, train_r2_scores, test_r2_scores, checkpoint_path)
    torch.save(model.state_dict(), os.path.join('NN_logs', 'best_model.pt'))
    sys.exit(0)

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)

# Load the data
train_file_path = "NN/NN_103/train_data.csv"
test_file_path = "NN/NN_103/test_data.csv"
X_train, y_train = load_data(train_file_path)
X_test, y_test = load_data(test_file_path)

# Create datasets and loaders
train_dataset = IODataset(X_train, y_train)
test_dataset = IODataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Hyperparameters
input_dim = X_train.shape[1]  # Number of features
num_epochs = 101
learning_rate = 0.001
weight_decay = 1e-4

# Initialize the model, optimizer, and loss function
model = IONeuralNetwork(input_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, min_lr=1e-6)
criterion = nn.MSELoss()  # You can change to L1Loss if needed

# Define checkpoint path
checkpoint_path = os.path.join('NN/NN_103', 'checkpoint.pt')

# Variable to control logging and chart updates
update_logs_and_charts = False  # Set to False to update only at the end

# Train the model
train_losses, train_rmse_scores, test_rmse_scores, train_mae_scores, test_mae_scores, train_r2_scores, test_r2_scores = train(
    model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs, checkpoint_path, update_logs_and_charts
)

# Ensure final predictions are logged
if not update_logs_and_charts:
    evaluate(model, train_loader, num_epochs, 'train', True)
    evaluate(model, test_loader, num_epochs, 'test', True)
