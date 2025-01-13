from src.ml_project.model import LinearEncoder, GCNConvLayer, GCNConvDecoder, GCN
import torch
import pytest
import torch_geometric


@pytest.fixture
def common_params():
    return {
        "num_edges": 76,
        "num_edge_features": 3,
        "num_nodes": 24,
        "node_feature_dim": 24,
        "hidden_dim": 64,
        "batch_size": 32,
        "num_gnn_layers": 2,
    }


class TestGCN:
    def test_linear_encoder(self, common_params):
        # Extract parameters
        num_edges = common_params["num_edges"]
        num_edge_features = common_params["num_edge_features"]
        num_nodes = common_params["num_nodes"]
        node_feature_dim = common_params["node_feature_dim"]
        hidden_dim = common_params["hidden_dim"]
        batch_size = common_params["batch_size"]

        # Instantiate LinearEncoder
        linear_encoder = LinearEncoder(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=num_edge_features,
            hidden_dim=hidden_dim,
            num_edges=num_edges,
        )

        # Mock input data
        x = torch.rand(num_nodes * batch_size, node_feature_dim)  # Node features
        edge_attr = torch.rand(num_edges * batch_size, num_edge_features)  # Edge features
        batch = torch.arange(batch_size).repeat_interleave(num_nodes)  # Batch tensor for nodes

        # Forward pass
        encoded_nodes, encoded_edges = linear_encoder(x, edge_attr, batch)

        # Assertions
        assert encoded_nodes.shape == (num_nodes * batch_size, hidden_dim)  # Shape: [num_nodes_total, hidden_dim]
        assert encoded_edges.shape == (num_edges * batch_size, hidden_dim)  # Shape: [num_edges_total, hidden_dim]

    def test_gcn_conv(self, common_params):
        # Extract parameters
        hidden_dim = common_params["hidden_dim"]
        num_gnn_layers = common_params["num_gnn_layers"]
        num_nodes = common_params["num_nodes"]
        batch_size = common_params["batch_size"]
        num_edges = common_params["num_edges"]

        # Instantiate GCNConvLayer
        gcn_conv_layer = GCNConvLayer(
            hidden_dim=hidden_dim,
            num_gnn_layers=num_gnn_layers,
        )

        # Mock input data
        x = torch.rand(num_nodes * batch_size, hidden_dim)  # Node embeddings
        edge_index = torch.randint(0, num_nodes * batch_size, (2, num_edges * batch_size))  # Edge index

        # Forward pass
        x_embeddings = gcn_conv_layer(x, edge_index)

        # Assertions
        assert x_embeddings.shape == (num_nodes * batch_size, hidden_dim)  # Shape: [num_nodes_total, hidden_dim]

    def test_gcn_conv_decoder(self, common_params):
        # Extract parameters
        hidden_dim = common_params["hidden_dim"]
        num_nodes = common_params["num_nodes"]
        batch_size = common_params["batch_size"]
        num_edges = common_params["num_edges"]

        # Instantiate GCNConvDecoder
        gcn_decoder = GCNConvDecoder(hidden_dim=hidden_dim)

        # Mock input data
        x_embeddings = torch.rand(num_nodes * batch_size, hidden_dim)  # Node embeddings
        edge_embeddings = torch.rand(num_edges * batch_size, hidden_dim)  # Edge embeddings
        edge_index = torch.randint(0, num_nodes * batch_size, (2, num_edges * batch_size))  # Edge index

        # Forward pass
        edge_output = gcn_decoder(x_embeddings, edge_embeddings, edge_index)

        # Assertions
        assert edge_output.shape == (num_edges * batch_size,)  # Shape: [num_edges_total]

    def test_gcn_model(self, common_params):
        # Extract parameters
        node_feature_dim = common_params["node_feature_dim"]
        num_edge_features = common_params["num_edge_features"]
        hidden_dim = common_params["hidden_dim"]
        num_gnn_layers = common_params["num_gnn_layers"]
        num_nodes = common_params["num_nodes"]
        num_edges = common_params["num_edges"]
        batch_size = common_params["batch_size"]

        # Instantiate GCN model
        gcn_model = GCN(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=num_edge_features,
            hidden_dim=hidden_dim,
            num_gnn_layers=num_gnn_layers,
            num_nodes=num_nodes,
            num_edges=num_edges,
        )

        # Mock input data
        data = torch_geometric.data.Data(
            x=torch.rand(num_nodes * batch_size, node_feature_dim),  # Node features
            edge_index=torch.randint(0, num_nodes * batch_size, (2, num_edges * batch_size)),  # Edge index
            edge_attr=torch.rand(num_edges * batch_size, num_edge_features),  # Edge attributes
            batch=torch.arange(batch_size).repeat_interleave(num_nodes),  # Batch tensor
        )

        # Forward pass
        edge_output = gcn_model(data)

        # Assertions
        assert edge_output.shape == (num_edges * batch_size,)  # Shape: [num_edges_total]
