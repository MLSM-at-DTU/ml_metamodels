from torch_geometric.utils import to_scipy_sparse_matrix, from_scipy_sparse_matrix
import scipy.sparse as sp
import numpy as np

def preprocess_diffusion(edge_index, edge_weight, num_nodes):
    """
    Preprocess edge_index and edge_weight to create a diffusion-aware graph.

    Args:
        edge_index (Tensor): Edge index of shape [2, num_edges].
        edge_weight (Tensor): Edge weights of shape [num_edges].
        num_nodes (int): Number of nodes in the graph.
        symmetric (bool): Whether to use symmetric normalization.

    Returns:
        edge_index (Tensor): Updated edge index.
        edge_weight (Tensor): Updated edge weights after normalization.
    """
    # Create a weighted adjacency matrix
    adj = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes=num_nodes)

    # Symmetric normalization: D^(-1/2) * A * D^(-1/2)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj_normalized = adj @ d_mat_inv_sqrt @ d_mat_inv_sqrt.T

    # Convert back to PyG format
    edge_index, edge_weight = from_scipy_sparse_matrix(adj_normalized)

    return edge_index, edge_weight