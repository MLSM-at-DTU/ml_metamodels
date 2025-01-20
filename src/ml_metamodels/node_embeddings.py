import torch
from torch import nn


class LinearEncoder(nn.Module):
    """Encodes node features and edge features."""

    def __init__(self, node_feature_dim: int, edge_feature_dim: int, hidden_dim: int) -> None:
        super().__init__()

        self.node_encoder = nn.Linear(node_feature_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_feature_dim, hidden_dim)

    def forward(self, x, edge_attr):
        # Encoded nodes
        encoded_nodes = torch.relu(self.node_encoder(x))

        # Encoded edges
        encoded_edges = torch.relu(self.edge_encoder(edge_attr))  # Directly use edge_attr

        return encoded_nodes, encoded_edges
