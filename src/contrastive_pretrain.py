"""Type-aware graph contrastive pretraining for heterogeneous GNNs.

Key design decisions (from review):
1. Type-aware contrastive loss: negatives only from same node type
2. Relation-aware augmentations: different dropout rates per relation type
3. Projection heads: MLP per type, discarded after pretraining
4. Feature masking: random zero-out of feature dimensions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from copy import deepcopy

logger = logging.getLogger(__name__)


class ProjectionHead(nn.Module):
    """2-layer MLP projection head for contrastive learning.

    Maps encoder output to a lower-dimensional space for contrastive loss.
    Discarded after pretraining.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def augment_graph(data, edge_dropout_rates: dict, feature_mask_rate: float = 0.2):
    """Create an augmented view of the heterogeneous graph.

    Applies relation-aware edge dropout and feature masking.

    Args:
        data: PyG HeteroData (will be deep-copied, original unchanged)
        edge_dropout_rates: dict mapping relation_type -> dropout probability
        feature_mask_rate: fraction of feature dimensions to zero out

    Returns:
        augmented HeteroData copy
    """
    aug_data = deepcopy(data)

    # Relation-aware edge dropout
    for edge_type in aug_data.edge_types:
        src_type, rel_type, dst_type = edge_type
        rate = edge_dropout_rates.get(rel_type, 0.05)  # default 5%

        edge_index = aug_data[edge_type].edge_index
        num_edges = edge_index.shape[1]

        if num_edges > 0:
            mask = torch.rand(num_edges) > rate
            aug_data[edge_type].edge_index = edge_index[:, mask]
            if hasattr(aug_data[edge_type], "edge_attr") and aug_data[edge_type].edge_attr is not None:
                aug_data[edge_type].edge_attr = aug_data[edge_type].edge_attr[mask]

    # Feature masking: randomly zero out feature dimensions
    for ntype in aug_data.node_types:
        if hasattr(aug_data[ntype], "x") and aug_data[ntype].x is not None:
            x = aug_data[ntype].x.clone()
            feat_dim = x.shape[1]
            mask = torch.rand(feat_dim) > feature_mask_rate
            x = x * mask.float().unsqueeze(0)
            aug_data[ntype].x = x

    return aug_data


def type_aware_contrastive_loss(
    z1_dict: dict, z2_dict: dict, tau: float = 0.5
) -> torch.Tensor:
    """Compute type-aware InfoNCE contrastive loss.

    For each node type, contrast same-node embeddings across views (positive)
    against other nodes of the SAME TYPE (negatives). This prevents cross-type
    noise that destabilizes training.

    Args:
        z1_dict: {node_type: projected embeddings from view 1 [N_t, proj_dim]}
        z2_dict: {node_type: projected embeddings from view 2 [N_t, proj_dim]}
        tau: temperature parameter

    Returns:
        scalar loss (average across types)
    """
    total_loss = 0.0
    num_types = 0

    for ntype in z1_dict:
        if ntype not in z2_dict:
            continue

        z1 = F.normalize(z1_dict[ntype], p=2, dim=-1)
        z2 = F.normalize(z2_dict[ntype], p=2, dim=-1)
        n = z1.shape[0]

        if n < 2:
            continue

        # Similarity matrix within same type
        sim_matrix = torch.mm(z1, z2.t()) / tau  # [N, N]

        # Positive pairs are on the diagonal
        labels = torch.arange(n, device=z1.device)

        # InfoNCE: cross-entropy where positive is diagonal entry
        loss_12 = F.cross_entropy(sim_matrix, labels)
        loss_21 = F.cross_entropy(sim_matrix.t(), labels)
        loss = (loss_12 + loss_21) / 2

        total_loss += loss
        num_types += 1

    if num_types == 0:
        return torch.tensor(0.0, requires_grad=True)

    return total_loss / num_types


def pretrain_contrastive(
    encoder,
    data,
    config: dict,
    device: torch.device,
    model_type: str = "hgt",
):
    """Run type-aware contrastive pretraining.

    Args:
        encoder: GNN encoder (R-GCN or HGT)
        data: PyG HeteroData with full training graph
        config: config dict
        device: torch device
        model_type: 'rgcn' or 'hgt'

    Returns:
        pretrained encoder (projection heads discarded)
    """
    from src.utils import build_global_edge_index

    hidden_dim = config["model"]["hidden_dim"]
    projection_dim = config["contrastive"]["projection_dim"]
    epochs = config["contrastive"]["epochs"]
    lr = config["contrastive"]["lr"]
    tau = config["contrastive"]["tau"]
    edge_dropout_rates = config["graph"]["edge_dropout_rates"]
    feature_mask_rate = config["graph"]["feature_mask_rate"]

    encoder = encoder.to(device)

    # Create per-type projection heads
    proj_heads = nn.ModuleDict(
        {
            ntype: ProjectionHead(hidden_dim, hidden_dim, projection_dim)
            for ntype in encoder.node_types
        }
    ).to(device)

    # Optimizer for encoder + projection heads
    params = list(encoder.parameters()) + list(proj_heads.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)

    encoder.train()
    proj_heads.train()

    node_types = encoder.node_types

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Generate two augmented views
        view1 = augment_graph(data, edge_dropout_rates, feature_mask_rate)
        view2 = augment_graph(data, edge_dropout_rates, feature_mask_rate)

        # Encode both views
        if model_type == "hgt":
            # HGT uses dict-based interface
            x_dict1 = {ntype: view1[ntype].x.to(device) for ntype in node_types if hasattr(view1[ntype], "x")}
            ei_dict1 = {
                et: view1[et].edge_index.to(device)
                for et in view1.edge_types
                if view1[et].edge_index.shape[1] > 0
            }
            z1_dict = encoder(x_dict1, ei_dict1)

            x_dict2 = {ntype: view2[ntype].x.to(device) for ntype in node_types if hasattr(view2[ntype], "x")}
            ei_dict2 = {
                et: view2[et].edge_index.to(device)
                for et in view2.edge_types
                if view2[et].edge_index.shape[1] > 0
            }
            z2_dict = encoder(x_dict2, ei_dict2)

        else:  # rgcn
            # R-GCN uses global edge_index
            x_dict1 = {ntype: view1[ntype].x.to(device) for ntype in node_types if hasattr(view1[ntype], "x")}
            ei1, et1, _ = build_global_edge_index(view1, node_types, device)
            z1_dict = encoder(x_dict1, ei1, et1, _get_node_offsets(view1, node_types))

            x_dict2 = {ntype: view2[ntype].x.to(device) for ntype in node_types if hasattr(view2[ntype], "x")}
            ei2, et2, _ = build_global_edge_index(view2, node_types, device)
            z2_dict = encoder(x_dict2, ei2, et2, _get_node_offsets(view2, node_types))

        # Project embeddings
        proj1 = {ntype: proj_heads[ntype](z1_dict[ntype]) for ntype in z1_dict if ntype in proj_heads}
        proj2 = {ntype: proj_heads[ntype](z2_dict[ntype]) for ntype in z2_dict if ntype in proj_heads}

        # Type-aware contrastive loss
        loss = type_aware_contrastive_loss(proj1, proj2, tau=tau)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            logger.info(f"Contrastive epoch {epoch + 1}/{epochs}, loss: {loss.item():.4f}")

    logger.info("Contrastive pretraining complete.")
    # Discard projection heads, return only encoder
    return encoder


def _get_node_offsets(data, node_types):
    """Compute node offsets for global edge_index construction."""
    offsets = {}
    offset = 0
    for ntype in node_types:
        offsets[ntype] = offset
        offset += data[ntype].num_nodes
    return offsets
