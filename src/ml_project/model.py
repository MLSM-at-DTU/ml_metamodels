import torch
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

class LinearEncoder(nn.Module):
    """Encodes node features and edge features."""
    def __init__(self, node_feature_dim: int, edge_feature_dim: int, hidden_dim: int, num_edges: int) -> None:
        super().__init__()

        self.num_edges = num_edges

        self.node_encoder = nn.Linear(node_feature_dim, hidden_dim) 
        self.edge_encoder = nn.Linear(edge_feature_dim + num_edges, hidden_dim)

    def forward(self, x, edge_attr, batch):

        num_batches = batch.max().item() + 1

        # Create one identity matrix and repeat it for all graphs
        identity_matrix = torch.eye(self.num_edges, device=edge_attr.device)  # Shape: [num_edges_per_graph, num_edges_per_graph]
        stacked_identity_matrix = identity_matrix.repeat(num_batches, 1)  # Shape: [num_batches * num_edges_per_graph, num_edges_per_graph]

        # Encoded nodes
        encoded_nodes = torch.relu(self.node_encoder(x))
        
        # Encoded edges
        combined_edge_features = torch.cat([edge_attr, stacked_identity_matrix], dim=1)
        encoded_edges = torch.relu(self.edge_encoder(combined_edge_features))  # Shape: [num_edges_total, hidden_dim]

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
        embedded_node_features = torch.cat([x_embeddings[row], x_embeddings[col]], dim=1)  # Shape: [num_edges, 2 * hidden_dim]

        # Combine raw features, embedded features, and edge attributes
        edge_features = torch.cat([embedded_node_features, edge_embeddings], dim=1)  # Shape: [num_edges, combined_feature_dim]

        # Predict edge outputs
        edge_output = self.edge_predictor(edge_features)
        return edge_output.squeeze(-1)

class GCN(nn.Module):
    """GCN model with Encoder-GNN-Decoder structure."""
    def __init__(self, node_feature_dim: int, edge_feature_dim: int, hidden_dim: int, num_gnn_layers: int, num_nodes: int = 24, num_edges: int = 76) -> None:
        super().__init__()
        self.encoder = LinearEncoder(node_feature_dim, edge_feature_dim, hidden_dim, num_edges)
        self.gnn = GCNConvLayer(hidden_dim, num_gnn_layers)
        self.decoder = GCNConvDecoder(hidden_dim)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Step 1: Encode node and edge features
        x_encoded, edge_encoded = self.encoder(x, edge_attr, batch)

        # Step 2: Process node embeddings using the GNN
        x_embeddings = self.gnn(x_encoded, edge_index)

        # Step 3: Decode embeddings and edge features for predictions
        edge_output = self.decoder(x_embeddings, edge_encoded, edge_index)

        return edge_output

class MLP(nn.Module):
    """MLP model for edge regression."""

    def __init__(self, node_feature_dim: int, edge_feature_dim: int, hidden_dim: int, num_nodes: int = 24, num_edges: int = 76) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.num_edges = num_edges

        # Node and edge embedding MLP
        self.node_edge_embedding_mlp = nn.Sequential(
            nn.Linear(node_feature_dim * num_nodes + edge_feature_dim * num_edges, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Edge prediction layer
        self.edge_predictor = nn.Linear(hidden_dim, 1)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x, edge_attr, batch = data.x, data.edge_attr, data.batch

        # Step 1: Flatten node features per graph
        num_graphs = batch.max().item() + 1
        flattened_node_features = x.view(num_graphs, -1)  # Shape: [num_graphs, 24 * node_feature_dim]

        # Step 2: Flatten edge attributes per graph
        flattened_edge_features = edge_attr.view(num_graphs, -1)  # Shape: [num_graphs, 76 * edge_feature_dim]

        # Step 3: Concatenate node and edge features
        combined_features = torch.cat([flattened_node_features, flattened_edge_features], dim=1)  # Shape: [num_graphs, 24 * node_feature_dim + 76 * edge_feature_dim]

        # Step 4: Compute graph-level embeddings
        graph_embeddings = self.node_edge_embedding_mlp(combined_features)  # Shape: [num_graphs, hidden_dim]

        # Step 5: Predict edge flows
        edge_output = graph_embeddings.repeat_interleave(76, dim=0)  # Shape: [num_graphs * 76, hidden_dim]
        edge_output = self.edge_predictor(edge_output)  # Shape: [num_graphs * 76, 1]

        return edge_output.view(-1)  # Shape: [num_graphs, 76]

if __name__ == "__main__":
    model = GCN(input_dim=24, hidden_dim=64)
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    dummy_data = Data(
        x=torch.randn(24, 24), 
        edge_index=torch.randint(0, 24, (2, 76)), 
        edge_attr=torch.randn(76,3)
    )

    output = model(dummy_data)
    print(f"Output shape: {output.shape}")
