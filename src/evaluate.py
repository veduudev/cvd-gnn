"""Evaluation metrics for gene-disease link prediction.

Computes: ROC-AUC, Average Precision (sampled negatives),
          Hits@K, MRR (full filtered ranking protocol).
Also provides comparison across all trained configurations.
"""

import csv
import json
import logging
from pathlib import Path

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

logger = logging.getLogger(__name__)


def compute_classification_metrics(scores: np.ndarray, labels: np.ndarray) -> dict:
    """Compute classification metrics (ROC-AUC, AP) from sampled pos/neg scores."""
    metrics = {}
    try:
        metrics["roc_auc"] = float(roc_auc_score(labels, scores))
    except ValueError:
        metrics["roc_auc"] = 0.0
    try:
        metrics["average_precision"] = float(average_precision_score(labels, scores))
    except ValueError:
        metrics["average_precision"] = 0.0
    return metrics


@torch.no_grad()
def compute_filtered_ranking_metrics(
    decoder,
    z_gene: torch.Tensor,
    z_disease: torch.Tensor,
    test_edge_index: torch.Tensor,
    all_known_edges: set,
    config: dict,
) -> dict:
    """Compute Hits@K and MRR using full filtered ranking protocol.

    For each test edge (gene_i, disease_j):
    1. Score gene_i against ALL diseases
    2. Filter out other known positives for gene_i (filtered setting)
    3. Compute rank of disease_j among remaining candidates
    """
    hits_k_values = config.get("evaluation", {}).get("hits_k", [10, 50])
    num_test = test_edge_index.shape[1]
    num_diseases = z_disease.shape[0]
    device = z_gene.device

    ranks = []

    for i in range(num_test):
        gene_idx = test_edge_index[0, i].item()
        true_disease = test_edge_index[1, i].item()

        # Score this gene against ALL diseases
        gene_emb = z_gene[gene_idx].unsqueeze(0).expand(num_diseases, -1)
        all_scores = decoder(gene_emb, z_disease, rel_idx=0)  # [num_diseases]

        # Filter: mask out other known positives for this gene (except the test edge itself)
        for d in range(num_diseases):
            if d != true_disease and (gene_idx, d) in all_known_edges:
                all_scores[d] = float('-inf')

        # Rank: count how many candidates score higher than the true disease
        true_score = all_scores[true_disease]
        rank = (all_scores > true_score).sum().item() + 1  # 1-based
        ranks.append(rank)

    ranks = np.array(ranks)

    metrics = {}
    for k in hits_k_values:
        metrics[f"hits@{k}"] = float((ranks <= k).mean())
    metrics["mrr"] = float((1.0 / ranks).mean())

    logger.info(f"Filtered ranking: median rank={np.median(ranks):.0f}, "
                f"mean rank={ranks.mean():.1f}/{num_diseases}")

    return metrics


def compute_metrics(scores: np.ndarray, labels: np.ndarray, config: dict) -> dict:
    """Compute all metrics. Backwards-compatible wrapper.

    For classification metrics (AUC, AP): uses sampled pos/neg scores.
    For ranking metrics (Hits@K, MRR): falls back to sampled ranking
    if full ranking is not available (use compute_filtered_ranking_metrics instead).
    """
    metrics = compute_classification_metrics(scores, labels)

    # Fallback sampled ranking (used when full ranking is not available)
    pos_mask = labels == 1
    neg_mask = labels == 0

    if pos_mask.sum() > 0 and neg_mask.sum() > 0:
        pos_scores = scores[pos_mask]
        neg_scores = scores[neg_mask]

        ranks = []
        for ps in pos_scores:
            rank = (neg_scores > ps).sum() + 1
            ranks.append(rank)
        ranks = np.array(ranks)

        hits_k_values = config.get("evaluation", {}).get("hits_k", [10, 50])
        for k in hits_k_values:
            metrics[f"hits@{k}"] = float((ranks <= k).mean())
        metrics["mrr"] = float((1.0 / ranks).mean())
    else:
        for k in config.get("evaluation", {}).get("hits_k", [10, 50]):
            metrics[f"hits@{k}"] = 0.0
        metrics["mrr"] = 0.0

    return metrics


def compare_all_models(results_dir: str = "results"):
    """Load all metrics files and produce a comparison table.

    Scans results/ for metrics_*.json files and prints a formatted table.
    """
    results_dir = Path(results_dir)
    metric_files = sorted(results_dir.glob("metrics_*.json"))

    if not metric_files:
        logger.warning("No metrics files found in results/")
        return

    all_results = []
    for f in metric_files:
        with open(f) as fp:
            data = json.load(fp)
        all_results.append(data)

    # Print comparison table
    print("\n" + "=" * 80)
    print("MODEL COMPARISON: Gene-Disease Link Prediction")
    print("=" * 80)

    # Header
    metric_names = list(all_results[0]["test_metrics"].keys())
    header = f"{'Model':<10} {'Pretrain':<10}"
    for m in metric_names:
        header += f" {m:>15}"
    print(header)
    print("-" * len(header))

    # Rows
    for r in all_results:
        row = f"{r['model']:<10} {'Yes' if r['pretrained'] else 'No':<10}"
        for m in metric_names:
            val = r["test_metrics"].get(m, 0.0)
            row += f" {val:>15.4f}"
        print(row)

    print("=" * 80)

    # Save comparison CSV
    with open(results_dir / "comparison.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "pretrained"] + metric_names)
        for r in all_results:
            writer.writerow(
                [r["model"], r["pretrained"]]
                + [r["test_metrics"].get(m, 0.0) for m in metric_names]
            )
    print(f"\nComparison saved to {results_dir / 'comparison.csv'}")


if __name__ == "__main__":
    import argparse
    from src.utils import setup_logger

    parser = argparse.ArgumentParser(description="Evaluate and compare models")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    setup_logger(__name__)
    compare_all_models(args.results_dir)
