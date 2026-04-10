"""Model modules for heterogeneous GNN gene-disease prediction."""

from src.models.rgcn_model import RGCNEncoder
from src.models.hgt_model import HGTEncoder
from src.models.decoder import DistMultDecoder

__all__ = ["RGCNEncoder", "HGTEncoder", "DistMultDecoder"]
