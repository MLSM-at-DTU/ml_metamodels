import torch
from torch import nn
from torch_geometric.nn import GCNConv, GATv2Conv


class GATConvLayer(nn.Module):
    """GNN layer for learning node embeddings."""

    def __init__(self, hidden_dim: int, num_gnn_layers: int) -> None:
        super().__init__()
        self.convs = nn.ModuleList([GATv2Conv(hidden_dim, hidden_dim) for _ in range(num_gnn_layers)])
        self.residual = nn.Linear(hidden_dim, hidden_dim)  # Residual connection

    def forward(self, x, edge_index):
        for conv in self.convs:
            x_res = x
            x = torch.relu(conv(x, edge_index) + self.residual(x_res))  # Update node embeddings
        return x


class GCNConvLayer(nn.Module):
    """GNN layer for learning node embeddings."""

    def __init__(self, hidden_dim: int, num_gnn_layers: int) -> None:
        super().__init__()
        self.convs = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for _ in range(num_gnn_layers)])
        self.residual = nn.Linear(hidden_dim, hidden_dim)  # Residual connection

    def forward(self, x, edge_index, edge_weight=None):
        for conv in self.convs:
            x_res = x

            x = torch.relu(
                conv(x, edge_index, edge_weight=edge_weight) + self.residual(x_res)
            )  # Update node embeddings
        return x
