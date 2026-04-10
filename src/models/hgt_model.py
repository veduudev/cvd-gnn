"""Heterogeneous Graph Transformer (HGT) for biomedical knowledge graphs.

Uses type-aware multi-head attention with relation-aware message passing.
Separate K/Q/V projections per (source_type, edge_type, target_type) triplet.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv


class HGTEncoder(nn.Module):
    """HGT encoder for heterogeneous biomedical knowledge graph.

    Architecture:
    - Per-type input projection to hidden_dim
    - N HGT layers with type-aware attention
    - Layer normalization + dropout
    """

    def __init__(
        self,
        node_types: list,
        edge_types: list,
        node_feature_dims: dict,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.2,
        embed_dim: int = 64,
    ):
        """
        Args:
            node_types: list of node type strings
            edge_types: list of (src_type, rel_type, dst_type) tuples
            node_feature_dims: dict mapping node_type -> handcrafted feature dimension
            hidden_dim: hidden layer dimension (must be divisible by num_heads)
            num_layers: number of HGT layers
            num_heads: number of attention heads
            dropout: dropout rate
            embed_dim: dimension of learnable node embeddings
        """
        super().__init__()
        self.node_types = node_types
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Per-type input projections
        self.input_projections = nn.ModuleDict()
        for ntype in node_types:
            feat_dim = node_feature_dims.get(ntype, 0)
            input_dim = feat_dim + embed_dim
            self.input_projections[ntype] = nn.Linear(input_dim, hidden_dim)

        # HGT layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                metadata=(node_types, edge_types),
                heads=num_heads,
            )
            self.convs.append(conv)

        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x_dict: dict, edge_index_dict: dict) -> dict:
        """Forward pass through HGT.

        Args:
            x_dict: dict of {node_type: feature_tensor [N_type, hidden_dim]}
            edge_index_dict: dict of {(src, rel, dst): edge_index [2, E]}

        Returns:
            dict of {node_type: embedding_tensor [N_type, hidden_dim]}
        """
        for conv in self.convs:
            x_dict_new = conv(x_dict, edge_index_dict)
            # Residual connection + dropout
            x_dict = {
                ntype: self.dropout_layer(F.relu(x_dict_new[ntype])) + x_dict[ntype]
                for ntype in x_dict
                if ntype in x_dict_new
            }

        return x_dict
