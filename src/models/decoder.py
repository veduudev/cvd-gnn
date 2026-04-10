"""DistMult bilinear decoder for heterogeneous link prediction.

score(gene, relation, disease) = sum_k (z_gene_k * r_relation_k * z_disease_k)
probability = sigmoid(score)

Supports L2 embedding normalization and regularization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistMultDecoder(nn.Module):
    """DistMult-style bilinear decoder for relation-aware link prediction.

    Each relation type has its own learnable embedding vector.
    Score is computed as element-wise product of (source, relation, target) embeddings.
    """

    def __init__(self, hidden_dim: int, num_relations: int):
        """
        Args:
            hidden_dim: dimension of node embeddings
            num_relations: number of relation types to decode
        """
        super().__init__()
        self.relation_embeddings = nn.Embedding(num_relations, hidden_dim)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)
        # Learnable temperature to control score spread
        self.score_scale = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        z_src: torch.Tensor,
        z_dst: torch.Tensor,
        rel_idx: int,
    ) -> torch.Tensor:
        """Compute link prediction scores.

        Args:
            z_src: source node embeddings [N, hidden_dim]
            z_dst: target node embeddings [N, hidden_dim]
            rel_idx: relation type index

        Returns:
            scores: [N] raw scores (apply sigmoid for probabilities)
        """
        r = self.relation_embeddings.weight[rel_idx]  # [hidden_dim]
        # DistMult: element-wise product then sum, scaled by learnable temperature
        scores = (z_src * r.unsqueeze(0) * z_dst).sum(dim=-1) * self.score_scale
        return scores

    def get_reg_loss(self, lambda_reg: float = 1e-5) -> torch.Tensor:
        """L2 regularization on relation embeddings."""
        return lambda_reg * self.relation_embeddings.weight.norm(p=2) ** 2
