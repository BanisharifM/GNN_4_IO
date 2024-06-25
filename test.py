import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader  # Use loader.DataLoader instead of data.DataLoader
from torch_geometric.nn import GCNConv
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV data
data_path = 'CSVs/sample_train_100.csv'  # Adjust this path to your test data
data = pd.read_csv(data_path)

# Determine the number of classes in the target variable
num_classes = data['tag'].nunique()

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

# Data preprocessing
def generate_graph(row):
    nodes = row.index.tolist()
    edge_index = []
    features = []

    # Root node features
    tag_node = 'tag'
    features.append([row[tag_node]])

    # Other nodes
    for node in nodes:
        if node != 'tag':
            node_index = nodes.index(node)
            edge_index.append([0, node_index])  # Root to node
            features.append([row[node]])

    x = torch.tensor(features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    y = torch.tensor([int(row[tag_node])], dtype=torch.long)  # Ensure the target is an integer
    return Data(x=x, edge_index=edge_index, y=y)

# Create dataset and dataloader
dataset = [generate_graph(row) for _, row in data.iterrows()]
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)  # Batch size of 1 for testing

# Load the model
model = GNN(input_dim=1, hidden_dim=64, output_dim=num_classes)
model_path = "results/model.pt"
model.load_state_dict(torch.load(model_path))
model.eval()

# Testing loop
all_targets = []
all_predictions = []

for batch in dataloader:
    with torch.no_grad():
        out = model(batch)

        # Identify root nodes using the batch attribute
        root_node_indices = []
        for i in range(batch.num_graphs):
            root_node_indices.append(torch.where(batch.batch == i)[0][0])
        root_node_indices = torch.tensor(root_node_indices, dtype=torch.long)

        out = out[root_node_indices]
        target = batch.y

        _, predicted = out.max(dim=1)
        all_targets.extend(target.tolist())
        all_predictions.extend(predicted.tolist())

# Calculate accuracy and classification report
accuracy = accuracy_score(all_targets, all_predictions)
class_report = classification_report(all_targets, all_predictions, zero_division=0)

print(f'Accuracy: {accuracy:.4f}')
print('Classification Report:')
print(class_report)

# Save the results
results_path = "results/test_results.txt"
with open(results_path, 'w') as f:
    f.write(f'Accuracy: {accuracy:.4f}\n')
    f.write('Classification Report:\n')
    f.write(class_report)

print(f"Test results saved to {results_path}")

# Confusion matrix
conf_matrix = confusion_matrix(all_targets, all_predictions)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(num_classes), yticklabels=range(num_classes))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
conf_matrix_path = "results/confusion_matrix.png"
plt.savefig(conf_matrix_path)
plt.close()
print(f"Confusion matrix saved to {conf_matrix_path}")

# Accuracy visualization
plt.figure(figsize=(10, 5))
plt.plot(all_targets, label='Actual')
plt.plot(all_predictions, label='Predicted', alpha=0.7)
plt.xlabel('Sample Index')
plt.ylabel('Class')
plt.title('Actual vs Predicted Classes')
plt.legend()
accuracy_plot_path = "results/accuracy_plot.png"
plt.savefig(accuracy_plot_path)
plt.close()
print(f"Accuracy plot saved to {accuracy_plot_path}")
