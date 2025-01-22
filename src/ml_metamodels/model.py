from typing import Optional
import torch
from torch import nn
from torch_geometric.data import Data
from ml_metamodels.node_embeddings import LinearEncoder
from ml_metamodels.gnn_layers import GCNConvLayer, GATConvLayer
from ml_metamodels.gnn_decoders import GNNConvDecoder
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self,
                 node_feature_dim: int, 
                 edge_feature_dim: int,
                 hidden_dim: int, 
                 num_gnn_layers: int,
                 normalize: bool = True, 
                 bias: bool = True,
                 add_self_loops: Optional[bool] = None, 
                 ) -> None:
        
        super().__init__()
        self.encoder = LinearEncoder(node_feature_dim, edge_feature_dim, hidden_dim)
        self.gnn = GCNConvLayer(hidden_dim = hidden_dim, num_gnn_layers = num_gnn_layers, normalize = normalize, bias = bias, add_self_loops = add_self_loops)
        self.decoder = GNNConvDecoder(hidden_dim)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        x, edge_index, edge_attr, edge_weight = data.x, data.edge_index, data.edge_attr, data.edge_weight

        # Step 1: Encode node and edge features
        x_encoded, edge_encoded = self.encoder(x, edge_attr)

        # Step 2: Process node embeddings using the GNN
        x_embeddings = self.gnn(x = x_encoded, edge_index = edge_index, edge_weight = edge_weight)

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


class DiffusionTestModel(nn.Module):
    """Simple model for testing purposes."""

    def __init__(self, num_nodes: int, num_edges: int) -> None:
        super().__init__()

        # 1) Graph Convolution Layer
        self.gconv = GCNConv(
            in_channels=num_nodes, out_channels=num_nodes, add_self_loops=True, normalize=True, bias=True
        )

        # 2) Linear layer to map H1 (N x N) -> (N x E)
        self.Wq = nn.Linear(num_nodes, num_edges, bias=True)

        # 3) Final layer
        self.WF = nn.Linear(num_nodes, 1, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # Unpack inputs
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight

        # Step 1: Graph Convolution => H1
        H1 = self.gconv(x, edge_index, edge_weight)  # shape: [N, N]
        H1 = torch.tanh(H1)

        # Step 2: H1 (N x N) -> H2 (N x E)
        H2 = self.Wq(H1)  # [N, E]
        H2 = torch.tanh(H2)  # f2

        # Step 3: Transpose => [E, N]
        H2_t = H2.transpose(0, 1)  # [E, N]

        # Step 4: Apply final linear layer => shape [E, 1]
        F_hat = self.WF(H2_t)  # => [E, 1]
        F_hat = F_hat.squeeze(-1)  # => [E]
        return F_hat


if __name__ == "__main__":
    model = GCN(input_dim=24, hidden_dim=64)
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    dummy_data = Data(x=torch.randn(24, 24), edge_index=torch.randint(0, 24, (2, 76)), edge_attr=torch.randn(76 * 2, 3))

    output = model(dummy_data)
    print(f"Output shape: {output.shape}")
