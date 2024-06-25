import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# Create the edge index for PyTorch Geometric
edge_index = torch.tensor(list(G.edges)).t().contiguous()

# Create node features (use the actual data from your dataset)
data = pd.read_csv("CSVs/sample_train_100.csv")
x = torch.tensor(data.drop(columns=["tag"]).values, dtype=torch.float)
y = torch.tensor(data["tag"].values, dtype=torch.float)

# Create the PyTorch Geometric data object
data = Data(x=x, edge_index=edge_index, y=y)


# Define the GNN model
class GNN(torch.nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(data.num_node_features, 16)
        self.conv2 = GCNConv(16, 1)

    def forward(self):
        x = self.conv1(data.x, data.edge_index)
        x = F.relu(x)
        x = self.conv2(x, data.edge_index)
        return x


# Instantiate the model, define the loss and optimizer
model = GNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.MSELoss()

# Training loop
model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model()
    loss = loss_fn(out.squeeze(), data.y)
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Evaluate the model
model.eval()
pred = model().detach().numpy()
print("Predictions:", pred)
