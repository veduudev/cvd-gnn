"""Relational Graph Convolutional Network (R-GCN) for heterogeneous graphs.

Uses relation-specific weight matrices with basis decomposition
to manage parameter count. Includes residual connections for stability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv


class RGCNEncoder(nn.Module):
    """R-GCN encoder for heterogeneous biomedical knowledge graph.

    Architecture:
    - Per-type input projection to align features to hidden_dim
    - N R-GCN layers with basis decomposition
    - Residual connections between layers
    - ReLU activation + dropout
    """

    def __init__(
        self,
        node_types: list,
        node_feature_dims: dict,
        hidden_dim: int = 128,
        num_relations: int = 8,
        num_layers: int = 2,
        num_bases: int = 30,
        dropout: float = 0.2,
        embed_dim: int = 64,
    ):
        """
        Args:
            node_types: list of node type strings
            node_feature_dims: dict mapping node_type -> handcrafted feature dimension
            hidden_dim: hidden layer dimension
            num_relations: number of relation types (including reverse)
            num_layers: number of R-GCN layers
            num_bases: number of basis matrices for decomposition
            dropout: dropout rate
            embed_dim: dimension of learnable node embeddings
        """
        super().__init__()
        self.node_types = node_types
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Per-type input projections (kept for checkpoint compatibility)
        self.input_projections = nn.ModuleDict()
        for ntype in node_types:
            feat_dim = node_feature_dims.get(ntype, 0)
            input_dim = feat_dim + embed_dim
            self.input_projections[ntype] = nn.Linear(input_dim, hidden_dim)

        # R-GCN layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = RGCNConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                num_relations=num_relations,
                num_bases=min(num_bases, num_relations),
            )
            self.convs.append(conv)

        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x_dict: dict, edge_index: torch.Tensor,
                edge_type: torch.Tensor, node_offsets: dict) -> dict:
        """Forward pass through R-GCN.

        Args:
            x_dict: dict of {node_type: feature_tensor [N_type, hidden_dim]}
            edge_index: global edge_index [2, E] with offsets applied
            edge_type: edge type indices [E]
            node_offsets: dict of {node_type: offset_int}

        Returns:
            dict of {node_type: embedding_tensor [N_type, hidden_dim]}
        """
        # Concatenate all node features into single tensor (ordered by node_types)
        x_list = []
        for ntype in self.node_types:
            if ntype in x_dict:
                x_list.append(x_dict[ntype])
        x = torch.cat(x_list, dim=0)

        # R-GCN message passing
        for i, conv in enumerate(self.convs):
            x_new = conv(x, edge_index, edge_type)
            x_new = F.relu(x_new)
            x_new = self.dropout_layer(x_new)
            # Residual connection
            x = x_new + x
            # Don't apply activation after residual (already applied above)

        # Split back into per-type embeddings
        result = {}
        for ntype in self.node_types:
            if ntype in node_offsets:
                offset = node_offsets[ntype]
                n_nodes = x_dict[ntype].shape[0]
                result[ntype] = x[offset : offset + n_nodes]

        return result
