import torch
from torch import nn


class GNNConvDecoder(nn.Module):
    """Decodes node embeddings and edge features for predictions."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.edge_predictor = nn.Linear(hidden_dim * 3, 1)

    def forward(self, x_embeddings, edge_embeddings, edge_index):
        row, col = edge_index

        # Embedded node features for edge endpoints
        embedded_node_features = torch.cat(
            [x_embeddings[row], x_embeddings[col]], dim=1
        )  # Shape: [num_edges, 2 * hidden_dim]

        # Combine raw features, embedded features, and edge attributes
        edge_features = torch.cat(
            [embedded_node_features, edge_embeddings], dim=1
        )  # Shape: [num_edges, combined_feature_dim]

        # Predict edge outputs
        edge_output = self.edge_predictor(edge_features)
        return edge_output.squeeze(-1)
