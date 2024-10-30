import os
import h5py
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# Parameters
num_epochs = 200
batch_size = 32
learning_rate = 0.01

# Load the graph data from HDF5
h5_file_path = 'Graph7/graph_data.h5'
with h5py.File(h5_file_path, 'r') as f:
    for name in f:
        print(name)
    edge_index = np.array(f['edge_index'])
    features = np.array(f['features'])
    nodes = np.array(f['nodes'])
#     labels = nodes  # Assuming 'nodes' are the labels

# print("Datasets loaded successfully.")

# # Inspect the type and content of 'nodes'
# print(f"nodes dtype: {nodes.dtype}")
# print(f"nodes content: {nodes}")

# # Ensure labels are of a supported type
# labels = np.array(nodes, dtype=np.int64)  # Convert to int64 if possible

# Decode the byte strings
nodes_decoded = np.vectorize(lambda x: x.decode('utf-8'))(nodes)

# Inspect the content of decoded nodes
# print(f"nodes decoded content: {nodes_decoded}")

# Create a mapping from unique labels to integers
unique_labels = np.unique(nodes_decoded)
label_to_int = {label: idx for idx, label in enumerate(unique_labels)}

# Convert labels to integers using the mapping
labels = np.vectorize(lambda x: label_to_int[x])(nodes_decoded)


# Prepare PyTorch Geometric Data objects
data_list = []
for i in range(len(labels)):
    x = torch.tensor(features[i], dtype=torch.float)
    edge_idx = torch.tensor(edge_index[i], dtype=torch.long).t().contiguous()
    y = torch.tensor(labels[i], dtype=torch.long)
    data_list.append(Data(x=x, edge_index=edge_idx, y=y))

dataloader = DataLoader(data_list, batch_size=batch_size, shuffle=True)

print("Labels processed successfully.")

# Determine the number of classes in the target variable
num_classes = len(np.unique(labels))

# GNN model definition
class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Model, optimizer, and loss function
model = GNN(input_dim=features.shape[-1], hidden_dim=64, output_dim=num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
losses = []
accuracies = []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in dataloader:
        optimizer.zero_grad()
        out = model(batch)

        # Identify root nodes using the batch attribute
        root_node_indices = []
        for i in range(batch.num_graphs):
            root_node_indices.append(torch.where(batch.batch == i)[0][0])
        root_node_indices = torch.tensor(root_node_indices, dtype=torch.long)
        
        out = out[root_node_indices]
        target = batch.y

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
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')

# Save the model
model_path = "results/correlation/full_data/training/model.pt"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# Plotting the results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), losses, label='Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), accuracies, label='Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig("results/correlation/full_data/training/training_metrics.png")
print("Training metrics plot saved to results/correlation/full_data/training/training_metrics.png")