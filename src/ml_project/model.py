import torch
from torch import nn
from torch_geometric.data import Data
from ml_project.node_embeddings import LinearEncoder
from ml_project.gnn_layers import GCNConvLayer, GATConvLayer
from ml_project.gnn_decoders import GNNConvDecoder

class GCN(nn.Module):
    """GCN model with Encoder-GNN-Decoder structure."""

    def __init__(self, node_feature_dim: int, edge_feature_dim: int, hidden_dim: int, num_gnn_layers: int) -> None:
        super().__init__()
        self.encoder = LinearEncoder(node_feature_dim, edge_feature_dim, hidden_dim)
        self.gnn = GCNConvLayer(hidden_dim, num_gnn_layers)
        self.decoder = GNNConvDecoder(hidden_dim)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Step 1: Encode node and edge features
        x_encoded, edge_encoded = self.encoder(x, edge_attr)

        # Step 2: Process node embeddings using the GNN
        x_embeddings = self.gnn(x_encoded, edge_index)

        # Step 3: Decode embeddings and edge features for predictions
        edge_output = self.decoder(x_embeddings, edge_encoded, edge_index)

        return edge_output

class GAT(nn.Module):
    """GAT model with Encoder-GNN-Decoder structure."""

    def __init__(self, node_feature_dim: int, edge_feature_dim: int, hidden_dim: int, num_gnn_layers: int) -> None:
        super().__init__()
        self.encoder = LinearEncoder(node_feature_dim, edge_feature_dim, hidden_dim)
        self.gnn = GATConvLayer(hidden_dim, num_gnn_layers)
        self.decoder = GNNConvDecoder(hidden_dim)

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
