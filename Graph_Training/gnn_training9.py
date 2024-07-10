import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
from torch.optim import Adam
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

# Paths
train_path = 'CSVs/train_data.csv'
val_path = 'CSVs/val_data.csv'
test_path = 'CSVs/test_data.csv'
save_path = 'Graphs/Graph26/'

# Create the directory if it doesn't exist
os.makedirs(save_path, exist_ok=True)

# Load data
train_data = pd.read_csv(train_path)
val_data = pd.read_csv(val_path)
test_data = pd.read_csv(test_path)

# Preprocess data
def preprocess_data(df):
    features = df.iloc[:, :-1].values  # All columns except the last one
    labels = df.iloc[:, -1].values  # The last column
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    return features, labels

train_features, train_labels = preprocess_data(train_data)
val_features, val_labels = preprocess_data(val_data)
test_features, test_labels = preprocess_data(test_data)

# Create PyTorch Geometric Data objects
def create_data_object(features, labels):
    edge_index = torch.tensor([[i, j] for i in range(features.shape[0]) for j in range(features.shape[0])], dtype=torch.long).t().contiguous()
    x = torch.tensor(features, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y)

train_dataset = create_data_object(train_features, train_labels)
val_dataset = create_data_object(val_features, val_labels)
test_dataset = create_data_object(test_features, test_labels)

train_loader = DataLoader([train_dataset], batch_size=1, shuffle=True)
val_loader = DataLoader([val_dataset], batch_size=1)
test_loader = DataLoader([test_dataset], batch_size=1)

# Define GNN model
class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Initialize model, optimizer, and loss function
model = GNN(in_channels=train_features.shape[1], hidden_channels=64, out_channels=len(set(train_labels)))
optimizer = Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

# Training function
def train(epoch):
    model.train()
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# Validation function
def validate(loader):
    model.eval()
    correct = 0
    for data in loader:
        out = model(data)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
    return correct / len(loader.dataset)

# Training loop with logging and saving the best model
best_val_accuracy = 0
train_losses = []
val_accuracies = []

for epoch in range(1, 201):
    train_loss = train(epoch)
    val_accuracy = validate(val_loader)
    
    train_losses.append(train_loss)
    val_accuracies.append(val_accuracy)
    
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pt'))
    
    print(f'Epoch: {epoch:03d}, Loss: {train_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

# Plot and save training loss and validation accuracy
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(os.path.join(save_path, 'training_visualization.png'))
plt.show()

# Test the model
model.load_state_dict(torch.load(os.path.join(save_path, 'best_model.pt')))
test_accuracy = validate(test_loader)
print(f'Test Accuracy: {test_accuracy:.4f}')
