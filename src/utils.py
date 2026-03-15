"""Utility functions: config loading, seeding, logging, graph helpers."""

import yaml
import torch
import numpy as np
import random
import logging
from pathlib import Path


def load_config(config_path: str = "configs/default.yaml", overrides: dict = None) -> dict:
    """Load YAML config and apply optional overrides."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if overrides:
        _deep_update(config, overrides)
    return config


def _deep_update(base: dict, updates: dict):
    """Recursively update nested dict."""
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return best available device.

    Note: MPS (Apple Silicon) is excluded because pyg_lib segment_matmul
    is not implemented for MPS, causing HGT to fail. Use CUDA or CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def setup_logger(name: str, level=logging.INFO) -> logging.Logger:
    """Configure and return a logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def build_global_edge_index(data, node_types, device):
    """Build global edge_index and edge_type tensors for R-GCN.

    Concatenates per-type node embeddings and offsets edge indices
    to create a single homogeneous edge_index + edge_type tensor.

    Returns: (edge_index [2, E], edge_type [E], rel_to_idx dict)
    """
    node_offsets = {}
    offset = 0
    for ntype in node_types:
        node_offsets[ntype] = offset
        offset += data[ntype].num_nodes

    all_edges = []
    all_types = []
    rel_to_idx = {}

    for edge_type in data.edge_types:
        src_type, rel_type, dst_type = edge_type
        if rel_type not in rel_to_idx:
            rel_to_idx[rel_type] = len(rel_to_idx)

        ei = data[edge_type].edge_index.to(device)
        offset_ei = torch.stack([
            ei[0] + node_offsets[src_type],
            ei[1] + node_offsets[dst_type]
        ])
        all_edges.append(offset_ei)
        all_types.append(torch.full((ei.shape[1],), rel_to_idx[rel_type],
                                     dtype=torch.long, device=device))

    if all_edges:
        return torch.cat(all_edges, dim=1), torch.cat(all_types), rel_to_idx
    return (torch.zeros(2, 0, dtype=torch.long, device=device),
            torch.zeros(0, dtype=torch.long, device=device),
            rel_to_idx)
