"""Unified training script for gene-disease link prediction.

Usage:
    python src/train.py --model rgcn --pretrain          # R-GCN with contrastive pretraining
    python src/train.py --model hgt --pretrain            # HGT with contrastive pretraining
    python src/train.py --model rgcn                      # R-GCN without pretraining
    python src/train.py --model hgt                       # HGT without pretraining
    python src/train.py --model rgcn --pretrain --debug   # Fast debug run
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import logging
from pathlib import Path

from src.utils import load_config, set_seed, get_device, setup_logger, build_global_edge_index
from src.dataset import load_hetero_data, split_gene_disease_edges, sample_negative_edges
from src.models.rgcn_model import RGCNEncoder
from src.models.hgt_model import HGTEncoder
from src.models.decoder import DistMultDecoder
from src.contrastive_pretrain import pretrain_contrastive
from src.evaluate import compute_metrics

logger = logging.getLogger(__name__)


def prepare_input_features(data, node_types, embed_dim, hidden_dim, device):
    """Prepare input features: concat handcrafted features with learnable embeddings.

    For each node type:
    1. Get handcrafted features from data[ntype].x
    2. Create learnable embedding of size embed_dim
    3. Concatenate and project to hidden_dim via linear layer

    Returns:
        learnable_embeds: nn.ParameterDict of learnable embeddings
        input_projections: nn.ModuleDict of projection layers
    """
    learnable_embeds = nn.ParameterDict()
    input_projections = nn.ModuleDict()

    for ntype in node_types:
        if not hasattr(data[ntype], "x"):
            continue
        n_nodes = data[ntype].num_nodes
        feat_dim = data[ntype].x.shape[1]
        input_dim = feat_dim + embed_dim

        learnable_embeds[ntype] = nn.Parameter(
            torch.randn(n_nodes, embed_dim) * 0.01
        )
        input_projections[ntype] = nn.Linear(input_dim, hidden_dim)

    return learnable_embeds, input_projections


def get_projected_features(data, node_types, learnable_embeds, input_projections, device):
    """Compute projected features for all node types."""
    x_dict = {}
    for ntype in node_types:
        if not hasattr(data[ntype], "x"):
            continue
        feat = data[ntype].x.to(device)
        embed = learnable_embeds[ntype].to(device)
        combined = torch.cat([feat, embed], dim=-1)
        x_dict[ntype] = input_projections[ntype](combined)
    return x_dict


def train_supervised(
    config: dict,
    model_type: str = "rgcn",
    use_pretrain: bool = False,
):
    """Full supervised training pipeline.

    1. Load data and split edges
    2. Optionally pretrain with contrastive learning
    3. Train encoder + DistMult decoder with BCE loss
    4. Evaluate on val/test sets
    5. Save results
    """
    device = get_device()
    set_seed(42)

    hidden_dim = config["model"]["hidden_dim"]
    embed_dim = config["graph"]["embed_dim"]
    num_layers = config["model"]["num_layers"]
    dropout = config["model"]["dropout"]

    # Load data
    logger.info("Loading heterogeneous graph data...")
    data = load_hetero_data(config)

    node_types = list(set(nt for nt in ["gene", "disease", "go_term"] if nt in data.node_types))
    edge_types = list(data.edge_types)

    logger.info(f"Node types: {node_types}")
    logger.info(f"Edge types: {edge_types}")

    # Split gene-disease edges
    splits = split_gene_disease_edges(data, config)

    # Feature dimensions per type
    node_feature_dims = {}
    for ntype in node_types:
        if hasattr(data[ntype], "x"):
            node_feature_dims[ntype] = data[ntype].x.shape[1]
        else:
            node_feature_dims[ntype] = 0

    # Create encoder
    if model_type == "rgcn":
        num_relations = len(edge_types)
        encoder = RGCNEncoder(
            node_types=node_types,
            node_feature_dims=node_feature_dims,
            hidden_dim=hidden_dim,
            num_relations=num_relations,
            num_layers=num_layers,
            num_bases=config["model"]["rgcn_num_bases"],
            dropout=dropout,
            embed_dim=embed_dim,
        )
    else:
        encoder = HGTEncoder(
            node_types=node_types,
            edge_types=edge_types,
            node_feature_dims=node_feature_dims,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=config["model"]["hgt_num_heads"],
            dropout=dropout,
            embed_dim=embed_dim,
        )

    # Create learnable embeddings and input projections
    learnable_embeds, input_projections = prepare_input_features(
        data, node_types, embed_dim, hidden_dim, device
    )

    # Contrastive pretraining
    if use_pretrain:
        logger.info(f"Starting contrastive pretraining ({model_type})...")

        # Save original features before modifying for pretraining
        original_features = {ntype: data[ntype].x.clone() for ntype in node_types if hasattr(data[ntype], "x")}

        # Temporarily set projected features on data for pretraining
        try:
            with torch.no_grad():
                for ntype in node_types:
                    if ntype in learnable_embeds:
                        feat = data[ntype].x
                        embed = learnable_embeds[ntype].detach()
                        combined = torch.cat([feat, embed], dim=-1)
                        data[ntype].x = input_projections[ntype](combined).detach()

            encoder = pretrain_contrastive(
                encoder, data, config, device, model_type=model_type
            )
        finally:
            # Restore original features (no reload needed, preserves edge splits)
            for ntype, feat in original_features.items():
                data[ntype].x = feat

    # Decoder
    # For gene-disease prediction, we use relation index 0
    decoder = DistMultDecoder(hidden_dim, num_relations=1)

    # Move everything to device
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    learnable_embeds = nn.ParameterDict(
        {k: v.to(device) for k, v in learnable_embeds.items()}
    )
    input_projections = nn.ModuleDict(
        {k: v.to(device) for k, v in input_projections.items()}
    )

    # Optimizer
    all_params = (
        list(encoder.parameters())
        + list(decoder.parameters())
        + list(learnable_embeds.parameters())
        + list(input_projections.parameters())
    )
    optimizer = torch.optim.Adam(
        all_params,
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=config["training"]["lr_scheduler_factor"],
        patience=config["training"]["lr_scheduler_patience"],
    )

    # Training loop
    epochs = config["training"]["epochs"]
    patience = config["training"]["early_stopping_patience"]
    best_val_ap = 0.0
    patience_counter = 0
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    tag = f"{model_type}_{'pretrain' if use_pretrain else 'nopretrain'}"

    num_genes = data["gene"].num_nodes
    num_diseases = data["disease"].num_nodes

    logger.info(f"Starting supervised training: {tag}")
    logger.info(f"  Genes: {num_genes}, Diseases: {num_diseases}")
    logger.info(f"  Train edges: {splits['train']['edge_index'].shape[1]}")

    for epoch in range(epochs):
        encoder.train()
        decoder.train()
        optimizer.zero_grad()

        # Get projected features
        x_dict = get_projected_features(
            data, node_types, learnable_embeds, input_projections, device
        )

        # Forward through encoder
        if model_type == "hgt":
            edge_index_dict = {
                et: data[et].edge_index.to(device)
                for et in data.edge_types
                if data[et].edge_index.shape[1] > 0
            }
            z_dict = encoder(x_dict, edge_index_dict)
        else:
            node_offsets = {}
            offset = 0
            for ntype in node_types:
                node_offsets[ntype] = offset
                offset += data[ntype].num_nodes
            ei, et, _ = build_global_edge_index(data, node_types, device)
            z_dict = encoder(x_dict, ei, et, node_offsets)

        # Get gene and disease embeddings
        z_gene = z_dict["gene"]
        z_disease = z_dict["disease"]

        # Positive edges (training set)
        pos_ei = splits["train"]["edge_index"].to(device)
        pos_src = z_gene[pos_ei[0]]
        pos_dst = z_disease[pos_ei[1]]
        pos_scores = decoder(pos_src, pos_dst, rel_idx=0)

        # Negative edges (re-sampled each epoch)
        neg_ei = sample_negative_edges(
            data,
            splits["train"]["edge_index"],
            num_genes,
            num_diseases,
            ratio=config["training"]["negative_ratio"],
            hard_ratio=config["training"]["hard_negative_ratio"],
        ).to(device)
        neg_src = z_gene[neg_ei[0]]
        neg_dst = z_disease[neg_ei[1]]
        neg_scores = decoder(neg_src, neg_dst, rel_idx=0)

        # BCE loss
        pos_labels = torch.ones(pos_scores.shape[0], device=device)
        neg_labels = torch.zeros(neg_scores.shape[0], device=device)
        scores = torch.cat([pos_scores, neg_scores])
        labels = torch.cat([pos_labels, neg_labels])
        loss = F.binary_cross_entropy_with_logits(scores, labels)

        # Embedding regularization
        reg_loss = decoder.get_reg_loss(config["decoder"]["embedding_reg_lambda"])
        loss = loss + reg_loss

        loss.backward()
        optimizer.step()

        # Validation
        if (epoch + 1) % 5 == 0:
            val_metrics = evaluate_split(
                encoder, decoder, data, splits["val"],
                learnable_embeds, input_projections,
                node_types, model_type, device, config,
            )
            val_ap = val_metrics["average_precision"]
            scheduler.step(val_ap)

            logger.info(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Loss: {loss.item():.4f} | "
                f"Val AP: {val_ap:.4f} | "
                f"Val AUC: {val_metrics['roc_auc']:.4f}"
            )

            if val_ap > best_val_ap:
                best_val_ap = val_ap
                patience_counter = 0
                # Save best checkpoint
                torch.save(
                    {
                        "encoder": encoder.state_dict(),
                        "decoder": decoder.state_dict(),
                        "learnable_embeds": {k: v.data for k, v in learnable_embeds.items()},
                        "input_projections": input_projections.state_dict(),
                        "epoch": epoch,
                        "val_ap": val_ap,
                    },
                    results_dir / f"best_{tag}.pt",
                )
            else:
                patience_counter += 5
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

    # Load best checkpoint and evaluate on test
    ckpt = torch.load(results_dir / f"best_{tag}.pt", map_location=device, weights_only=False)
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    for k, v in ckpt["learnable_embeds"].items():
        learnable_embeds[k].data = v

    test_metrics = evaluate_split(
        encoder, decoder, data, splits["test"],
        learnable_embeds, input_projections,
        node_types, model_type, device, config,
    )

    logger.info(f"=== Test Results ({tag}) ===")
    for k, v in test_metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    # Save metrics
    results = {
        "model": model_type,
        "pretrained": use_pretrain,
        "best_val_ap": best_val_ap,
        "test_metrics": test_metrics,
    }
    with open(results_dir / f"metrics_{tag}.json", "w") as f:
        json.dump(results, f, indent=2)

    # Generate novel predictions
    generate_predictions(
        encoder, decoder, data, splits,
        learnable_embeds, input_projections,
        node_types, model_type, device, config, tag,
    )

    return results


@torch.no_grad()
def evaluate_split(
    encoder, decoder, data, split,
    learnable_embeds, input_projections,
    node_types, model_type, device, config,
):
    """Evaluate on a val/test split."""
    encoder.eval()
    decoder.eval()

    # Get projected features
    x_dict = get_projected_features(
        data, node_types, learnable_embeds, input_projections, device
    )

    # Encode
    if model_type == "hgt":
        edge_index_dict = {
            et: data[et].edge_index.to(device)
            for et in data.edge_types
            if data[et].edge_index.shape[1] > 0
        }
        z_dict = encoder(x_dict, edge_index_dict)
    else:
        node_offsets = {}
        offset = 0
        for ntype in node_types:
            node_offsets[ntype] = offset
            offset += data[ntype].num_nodes
        ei, et, _ = build_global_edge_index(data, node_types, device)
        z_dict = encoder(x_dict, ei, et, node_offsets)

    z_gene = z_dict["gene"]
    z_disease = z_dict["disease"]

    # Score positive edges
    pos_ei = split["edge_index"].to(device)
    pos_scores = decoder(z_gene[pos_ei[0]], z_disease[pos_ei[1]], rel_idx=0)

    # Sample negatives for evaluation (same size as positives)
    num_genes = data["gene"].num_nodes
    num_diseases = data["disease"].num_nodes
    neg_ei = sample_negative_edges(
        data, split["edge_index"], num_genes, num_diseases,
        ratio=1.0, hard_ratio=0.0,  # random negatives for fair eval
    ).to(device)
    neg_scores = decoder(z_gene[neg_ei[0]], z_disease[neg_ei[1]], rel_idx=0)

    scores = torch.cat([pos_scores, neg_scores]).sigmoid().cpu().numpy()
    labels = np.concatenate([
        np.ones(pos_scores.shape[0]),
        np.zeros(neg_scores.shape[0]),
    ])

    metrics = compute_metrics(scores, labels, config)
    return metrics


@torch.no_grad()
def generate_predictions(
    encoder, decoder, data, splits,
    learnable_embeds, input_projections,
    node_types, model_type, device, config, tag,
):
    """Generate ranked novel gene-disease predictions.

    Scores all gene-disease pairs NOT in the training data and ranks them.
    """
    encoder.eval()
    decoder.eval()

    x_dict = get_projected_features(
        data, node_types, learnable_embeds, input_projections, device
    )

    if model_type == "hgt":
        edge_index_dict = {
            et: data[et].edge_index.to(device)
            for et in data.edge_types
            if data[et].edge_index.shape[1] > 0
        }
        z_dict = encoder(x_dict, edge_index_dict)
    else:
        node_offsets = {}
        offset = 0
        for ntype in node_types:
            node_offsets[ntype] = offset
            offset += data[ntype].num_nodes
        ei, et, _ = build_global_edge_index(data, node_types, device)
        z_dict = encoder(x_dict, ei, et, node_offsets)

    z_gene = z_dict["gene"]
    z_disease = z_dict["disease"]
    num_genes = z_gene.shape[0]
    num_diseases = z_disease.shape[0]

    # Build set of known edges
    known_edges = set()
    for split_name in ["train", "val", "test"]:
        ei = splits[split_name]["edge_index"]
        for i in range(ei.shape[1]):
            known_edges.add((ei[0, i].item(), ei[1, i].item()))

    # Score candidate pairs (sample to avoid O(N*M) for large graphs)
    max_candidates = min(50000, num_genes * num_diseases)
    candidates = []
    attempts = 0
    while len(candidates) < max_candidates and attempts < max_candidates * 5:
        g = np.random.randint(num_genes)
        d = np.random.randint(num_diseases)
        if (g, d) not in known_edges:
            candidates.append((g, d))
            known_edges.add((g, d))  # avoid duplicates
        attempts += 1

    if not candidates:
        logger.warning("No novel candidates found.")
        return

    cand_genes = torch.tensor([c[0] for c in candidates], device=device)
    cand_diseases = torch.tensor([c[1] for c in candidates], device=device)

    scores = decoder(z_gene[cand_genes], z_disease[cand_diseases], rel_idx=0)
    scores = scores.sigmoid().cpu().numpy()

    # Get node names
    gene_names = data["gene"].node_names
    disease_names = data["disease"].node_names

    # Sort by score descending
    ranked_idx = np.argsort(scores)[::-1]

    results_dir = Path("results")
    with open(results_dir / f"predictions_{tag}.tsv", "w") as f:
        f.write("rank\tgene\tdisease\tscore\n")
        for rank, idx in enumerate(ranked_idx[:500]):  # top 500
            g, d = candidates[idx]
            f.write(f"{rank + 1}\t{gene_names[g]}\t{disease_names[d]}\t{scores[idx]:.6f}\n")

    logger.info(f"Saved top 500 predictions to results/predictions_{tag}.tsv")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train gene-disease link prediction")
    parser.add_argument("--model", choices=["rgcn", "hgt"], required=True)
    parser.add_argument("--pretrain", action="store_true", help="Use contrastive pretraining")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--debug", action="store_true", help="Use debug graph (fast)")
    args = parser.parse_args()

    setup_logger("src")
    setup_logger(__name__)
    for mod in ["src.dataset", "src.models", "src.contrastive_pretrain", "src.evaluate"]:
        setup_logger(mod)

    config = load_config(args.config)

    if args.debug:
        # Reduce epochs for fast testing
        config["contrastive"]["epochs"] = 5
        config["training"]["epochs"] = 20
        config["training"]["early_stopping_patience"] = 10

    results = train_supervised(
        config,
        model_type=args.model,
        use_pretrain=args.pretrain,
    )
    print(f"\nFinal test metrics: {results['test_metrics']}")
