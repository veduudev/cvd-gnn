"""Load heterogeneous graph into PyTorch Geometric HeteroData.

Reads nodes.csv, edges.csv, node_features.csv and creates a HeteroData object
with train/val/test edge splits for gene-disease link prediction.
"""

import torch
import pandas as pd
import numpy as np
from torch_geometric.data import HeteroData
from pathlib import Path
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


def load_hetero_data(config: dict) -> HeteroData:
    """Load processed CSVs into a PyG HeteroData object.

    Returns HeteroData with:
    - Node features (handcrafted + learnable embedding placeholder)
    - All edge types with edge_index tensors
    - Edge weights as edge_attr
    """
    processed_dir = Path("data/processed")

    nodes_df = pd.read_csv(processed_dir / "nodes.csv")
    edges_df = pd.read_csv(processed_dir / "edges.csv")
    features_df = pd.read_csv(processed_dir / "node_features.csv")

    data = HeteroData()

    # --- Set up node features ---
    feat_cols = [c for c in features_df.columns if c.startswith("f")]
    handcrafted_dim = len(feat_cols)

    for ntype in ["gene", "disease", "go_term"]:
        type_nodes = nodes_df[nodes_df["node_type"] == ntype]
        n_nodes = len(type_nodes)

        if n_nodes == 0:
            continue

        # Get handcrafted features (z-score normalized)
        type_feats = features_df[features_df["node_type"] == ntype].sort_values("node_id")
        feat_tensor = torch.tensor(
            type_feats[feat_cols].values, dtype=torch.float32
        )

        data[ntype].x = feat_tensor
        data[ntype].num_nodes = n_nodes

        # Store node names for later lookup
        name_list = type_nodes.sort_values("node_id")["name"].tolist()
        data[ntype].node_names = name_list

    # --- Set up edges ---
    # Map (source_type, relation_type, target_type) to edge lists
    edge_dict = defaultdict(lambda: {"src": [], "dst": [], "weight": []})

    for _, row in edges_df.iterrows():
        src_type = row["source_type"]
        rel_type = row["relation_type"]
        tgt_type = row["target_type"]
        key = (src_type, rel_type, tgt_type)
        edge_dict[key]["src"].append(int(row["source_id"]))
        edge_dict[key]["dst"].append(int(row["target_id"]))
        edge_dict[key]["weight"].append(float(row["weight"]))

    for (src_type, rel_type, tgt_type), vals in edge_dict.items():
        edge_index = torch.tensor([vals["src"], vals["dst"]], dtype=torch.long)
        edge_weight = torch.tensor(vals["weight"], dtype=torch.float32)
        data[src_type, rel_type, tgt_type].edge_index = edge_index
        data[src_type, rel_type, tgt_type].edge_attr = edge_weight

    logger.info(f"Loaded HeteroData: {data}")
    return data


def split_gene_disease_edges(
    data: HeteroData, config: dict, seed: int = 42
) -> dict:
    """Split gene-disease edges into train/val/test sets.

    Stratified by disease so every disease retains training edges.
    Returns dict with 'train', 'val', 'test' edge_index and edge_attr tensors,
    and modifies `data` in-place to only contain training gene-disease edges.
    """
    train_ratio = config["training"]["train_ratio"]
    val_ratio = config["training"]["val_ratio"]

    # Get gene-disease forward edges
    edge_key = ("gene", "gene_associated_with_disease", "disease")
    rev_key = ("disease", "disease_has_gene", "gene")

    edge_index = data[edge_key].edge_index  # [2, num_edges]
    edge_attr = data[edge_key].edge_attr

    num_edges = edge_index.shape[1]
    logger.info(f"Total gene-disease edges: {num_edges}")

    # Manual stratified split: guarantees every disease keeps ≥1 training edge
    disease_ids = edge_index[1].numpy()
    rng = np.random.RandomState(seed)

    train_idx_list = []
    val_idx_list = []
    test_idx_list = []

    # Group edge indices by disease
    disease_to_indices = defaultdict(list)
    for i in range(num_edges):
        disease_to_indices[disease_ids[i]].append(i)

    for disease, indices in disease_to_indices.items():
        rng.shuffle(indices)
        n = len(indices)

        if n == 1:
            # Only 1 edge: must go to train (can't evaluate what we can't train on)
            train_idx_list.extend(indices)
        elif n == 2:
            # 2 edges: 1 train, 1 val (no test for this disease)
            train_idx_list.append(indices[0])
            val_idx_list.append(indices[1])
        else:
            # Normal split
            n_train = max(1, int(n * train_ratio))
            n_val = max(1, int(n * val_ratio))
            n_test = n - n_train - n_val
            if n_test <= 0:
                n_test = 0
                n_val = n - n_train

            train_idx_list.extend(indices[:n_train])
            val_idx_list.extend(indices[n_train:n_train + n_val])
            test_idx_list.extend(indices[n_train + n_val:])

    train_idx = np.array(train_idx_list)
    val_idx = np.array(val_idx_list)
    test_idx = np.array(test_idx_list)

    logger.info(f"Stratified split: {len(disease_to_indices)} diseases, "
                f"{sum(1 for v in disease_to_indices.values() if len(v) < 3)} with <3 edges")
    logger.info(f"Split sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    splits = {
        "train": {
            "edge_index": edge_index[:, train_idx],
            "edge_attr": edge_attr[train_idx],
        },
        "val": {
            "edge_index": edge_index[:, val_idx],
            "edge_attr": edge_attr[val_idx],
        },
        "test": {
            "edge_index": edge_index[:, test_idx],
            "edge_attr": edge_attr[test_idx],
        },
    }

    logger.info(
        f"Edge splits - train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}"
    )

    # Update data to only contain training edges for message passing
    data[edge_key].edge_index = splits["train"]["edge_index"]
    data[edge_key].edge_attr = splits["train"]["edge_attr"]

    # Also update reverse edges
    data[rev_key].edge_index = torch.stack(
        [splits["train"]["edge_index"][1], splits["train"]["edge_index"][0]]
    )
    data[rev_key].edge_attr = splits["train"]["edge_attr"]

    return splits


def sample_negative_edges(
    data: HeteroData,
    positive_edge_index: torch.Tensor,
    num_genes: int,
    num_diseases: int,
    ratio: float = 1.0,
    hard_ratio: float = 0.5,
    all_known_edges: set = None,
) -> torch.Tensor:
    """Sample negative gene-disease edges.

    Args:
        data: HeteroData with PPI edges for hard negative mining
        positive_edge_index: [2, num_pos] positive edges
        num_genes: total number of gene nodes
        num_diseases: total number of disease nodes
        ratio: negative-to-positive ratio
        hard_ratio: fraction of negatives that are hard (PPI-neighbor based)
        all_known_edges: set of (gene, disease) tuples from ALL splits to avoid
                         false negatives. If None, only rejects from positive_edge_index.

    Returns: [2, num_neg] negative edge_index
    """
    num_pos = positive_edge_index.shape[1]
    num_neg = int(num_pos * ratio)
    num_hard = int(num_neg * hard_ratio)
    num_random = num_neg - num_hard

    # Build set of ALL known positive edges for rejection
    if all_known_edges is not None:
        pos_set = all_known_edges
    else:
        pos_set = set(
            zip(positive_edge_index[0].tolist(), positive_edge_index[1].tolist())
        )

    neg_src = []
    neg_dst = []

    # Hard negatives: for a positive (gene_a, disease_x),
    # find PPI neighbor gene_b and create (gene_b, disease_x) as hard negative
    if num_hard > 0 and ("gene", "gene_interacts_with_gene", "gene") in data.edge_types:
        ppi_ei = data["gene", "gene_interacts_with_gene", "gene"].edge_index
        # Build adjacency list
        ppi_adj = defaultdict(list)
        for i in range(ppi_ei.shape[1]):
            ppi_adj[ppi_ei[0, i].item()].append(ppi_ei[1, i].item())

        hard_count = 0
        perm = torch.randperm(num_pos)
        for idx in perm:
            if hard_count >= num_hard:
                break
            gene_a = positive_edge_index[0, idx].item()
            disease_x = positive_edge_index[1, idx].item()
            neighbors = ppi_adj.get(gene_a, [])
            if neighbors:
                gene_b = neighbors[np.random.randint(len(neighbors))]
                if (gene_b, disease_x) not in pos_set:
                    neg_src.append(gene_b)
                    neg_dst.append(disease_x)
                    hard_count += 1

        # Fill remaining hard slots with random
        num_random += (num_hard - hard_count)

    # Random negatives
    random_count = 0
    max_attempts = num_random * 10
    attempts = 0
    while random_count < num_random and attempts < max_attempts:
        g = np.random.randint(num_genes)
        d = np.random.randint(num_diseases)
        if (g, d) not in pos_set:
            neg_src.append(g)
            neg_dst.append(d)
            random_count += 1
        attempts += 1

    neg_edge_index = torch.tensor([neg_src, neg_dst], dtype=torch.long)
    return neg_edge_index


@torch.no_grad()
def mine_dynamic_hard_negatives(
    decoder,
    z_gene: torch.Tensor,
    z_disease: torch.Tensor,
    positive_edge_index: torch.Tensor,
    all_known_edges: set,
    num_negatives: int,
    pool_size: int = 10000,
) -> torch.Tensor:
    """Mine hard negatives by scoring random candidates and keeping the highest.

    Args:
        decoder: trained DistMult decoder
        z_gene: gene embeddings [N_gene, hidden_dim]
        z_disease: disease embeddings [N_disease, hidden_dim]
        positive_edge_index: [2, num_pos] positive edges
        all_known_edges: set of all known (gene, disease) tuples
        num_negatives: number of hard negatives to return
        pool_size: number of random candidates to score

    Returns: [2, num_negatives] hard negative edge_index
    """
    num_genes = z_gene.shape[0]
    num_diseases = z_disease.shape[0]

    # Sample a large pool of random negative candidates
    candidates_g = []
    candidates_d = []
    attempts = 0
    while len(candidates_g) < pool_size and attempts < pool_size * 5:
        g = np.random.randint(num_genes)
        d = np.random.randint(num_diseases)
        if (g, d) not in all_known_edges:
            candidates_g.append(g)
            candidates_d.append(d)
        attempts += 1

    if not candidates_g:
        return torch.zeros(2, 0, dtype=torch.long)

    cand_g = torch.tensor(candidates_g, device=z_gene.device)
    cand_d = torch.tensor(candidates_d, device=z_gene.device)

    # Score all candidates
    scores = decoder(z_gene[cand_g], z_disease[cand_d], rel_idx=0)

    # Take the top-scoring negatives (hardest for the model)
    k = min(num_negatives, len(candidates_g))
    top_idx = scores.topk(k).indices

    hard_g = cand_g[top_idx].cpu()
    hard_d = cand_d[top_idx].cpu()
    return torch.stack([hard_g, hard_d])
