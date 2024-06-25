import os
import yaml
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the YAML graph structure
yaml_file_path = "results/correlation/graph_structure.yaml"
with open(yaml_file_path, 'r') as file:
    graph_structure = yaml.safe_load(file)

# Load the CSV data
print("Loading data")
data_path = 'CSVs/sample_train.csv'
data = pd.read_csv(data_path)
print("data is loaded...!")


# Parameters
num_epochs = 200
batch_size = 32
learning_rate = 0.01
threshold = 0.7

# Determine the number of classes in the target variable
num_classes = data['tag'].nunique()

# GNN model definition
class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        print("__init__ def")
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        print("forward def")
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Data preprocessing
def generate_graph(row, graph_structure):
    # print("generate_graph def",row)
    nodes = ['tag'] + [node for primary, secondaries in graph_structure['tag'].items() for node in [primary] + secondaries]
    edge_index = []
    features = []

    # Root node features
    tag_node = 'tag'
    features.append([row[tag_node]])

    # Primary and secondary nodes
    for primary_node, secondary_nodes in graph_structure[tag_node].items():
        if primary_node in row:
            primary_index = nodes.index(primary_node)
            edge_index.append([0, primary_index])  # Root to primary
            features.append([row[primary_node]])

            for secondary_node in secondary_nodes:
                if secondary_node in row:
                    secondary_index = nodes.index(secondary_node)
                    edge_index.append([primary_index, secondary_index])  # Primary to secondary
                    features.append([row[secondary_node]])

    x = torch.tensor(features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    y = torch.tensor([int(row[tag_node])], dtype=torch.long)  # Ensure the target is an integer
    return Data(x=x, edge_index=edge_index, y=y)

# Create dataset and dataloader
dataset = [generate_graph(row, graph_structure) for _, row in data.iterrows()]
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model, optimizer, and loss function
model = GNN(input_dim=1, hidden_dim=64, output_dim=num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
losses = []
accuracies = []

print("befor epoch")

for epoch in range(num_epochs):
    print(epoch)
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
model_path = "results/model.pt"
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
plt.savefig("results/training_metrics.png")
print("Training metrics plot saved to results/training_metrics.png")
