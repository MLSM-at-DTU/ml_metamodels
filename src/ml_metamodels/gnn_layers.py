from typing import Optional
import torch
from torch import nn
from torch_geometric.nn import GCNConv, GATv2Conv


class GATConvLayer(nn.Module):
    """GNN layer for learning node embeddings."""

    def __init__(self, hidden_dim: int, num_gnn_layers: int) -> None:
        super().__init__()
        self.convs = nn.ModuleList([GATv2Conv(hidden_dim, hidden_dim) for _ in range(num_gnn_layers)])
        self.residual = nn.Linear(hidden_dim, hidden_dim)  # Residual connection

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x_res = x
            x = torch.relu(conv(x, edge_index) + self.residual(x_res))  # Update node embeddings
        return x


class GCNConvLayer(nn.Module):
    """GNN layer for learning node embeddings."""

    def __init__(
        self,
        hidden_dim: int,
        num_gnn_layers: int,
        normalize: bool = True,
        bias: bool = True,
        add_self_loops: Optional[bool] = None,
    ) -> torch.Tensor:
        super().__init__()
        self.convs = nn.ModuleList(
            [
                GCNConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    normalize=normalize,
                    bias=bias,
                    add_self_loops=add_self_loops,
                )
                for _ in range(num_gnn_layers)
            ]
        )

        self.residual = nn.Linear(hidden_dim, hidden_dim)  # Residual connection

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for conv in self.convs:
            x_res = x
            x = torch.relu(conv(x=x, edge_index=edge_index, edge_weight=edge_weight) + self.residual(x_res))
        return x
