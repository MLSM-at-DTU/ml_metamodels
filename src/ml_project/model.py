import torch
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data


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


class GCNConvLayer(nn.Module):
    """GNN layer for learning node embeddings."""

    def __init__(self, hidden_dim: int, num_gnn_layers: int) -> None:
        super().__init__()
        self.convs = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for _ in range(num_gnn_layers)])
        self.residual = nn.Linear(hidden_dim, hidden_dim)  # Residual connection

    def forward(self, x, edge_index):
        for conv in self.convs:
            x_res = x
            x = torch.relu(conv(x, edge_index) + self.residual(x_res))  # Update node embeddings
        return x


class GCNConvDecoder(nn.Module):
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


class GCN(nn.Module):
    """GCN model with Encoder-GNN-Decoder structure."""

    def __init__(self, node_feature_dim: int, edge_feature_dim: int, hidden_dim: int, num_gnn_layers: int) -> None:
        super().__init__()
        self.encoder = LinearEncoder(node_feature_dim, edge_feature_dim, hidden_dim)
        self.gnn = GCNConvLayer(hidden_dim, num_gnn_layers)
        self.decoder = GCNConvDecoder(hidden_dim)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Step 1: Encode node and edge features
        x_encoded, edge_encoded = self.encoder(x, edge_attr)

        # Step 2: Process node embeddings using the GNN
        x_embeddings = self.gnn(x_encoded, edge_index)

        # Step 3: Decode embeddings and edge features for predictions
        edge_output = self.decoder(x_embeddings, edge_encoded, edge_index)

        return edge_output


if __name__ == "__main__":
    model = GCN(input_dim=24, hidden_dim=64)
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    dummy_data = Data(x=torch.randn(24, 24), edge_index=torch.randint(0, 24, (2, 76)), edge_attr=torch.randn(76 * 2, 3))

    output = model(dummy_data)
    print(f"Output shape: {output.shape}")
