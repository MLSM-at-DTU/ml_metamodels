import torch
import torch_sparse

def calculate_diffusion(edge_index, edge_weight):
    """
    Compute P^k, the k-step transition probability matrix.
    
    Args:
        edge_index (Tensor): Edge index of the graph.
        edge_weight (Tensor): Edge weights for the graph.
        k (int): Number of diffusion steps.

    Returns:
        Tensor: P^k as a sparse matrix or dense tensor.
    """
    # Create adjacency matrix (sparse)
    num_nodes = edge_index.max().item() + 1
    adj = torch_sparse.SparseTensor.from_edge_index(edge_index, edge_weight, sparse_sizes=(num_nodes, num_nodes))

    print("Adjacency Matrix (Sparse):")
    print(adj.to_dense())  # For debugging, convert sparse to dense

    # Row-normalize the adjacency matrix to get the transition matrix P
    row_sum = torch_sparse.sum(adj, dim=1)
    row_inv = 1.0 / (row_sum + 1e-6)
    row_inv = row_inv.view(-1, 1)
    transition_matrix = adj.mul(row_inv)
    print("\nTransition Matrix (Sparse):")
    print(transition_matrix.to_dense())  # Convert to dense for debugging
    
    return transition_matrix