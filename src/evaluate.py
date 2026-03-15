"""Evaluation metrics for gene-disease link prediction.

Computes: ROC-AUC, Average Precision, Hits@K, Mean Reciprocal Rank.
Also provides comparison across all trained configurations.
"""

import numpy as np
import json
import logging
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score

logger = logging.getLogger(__name__)


def compute_metrics(scores: np.ndarray, labels: np.ndarray, config: dict) -> dict:
    """Compute link prediction evaluation metrics.

    Args:
        scores: predicted probabilities [N]
        labels: binary ground truth [N] (1 = positive, 0 = negative)
        config: config dict with evaluation.hits_k

    Returns:
        dict with roc_auc, average_precision, hits@k, mrr
    """
    metrics = {}

    # ROC-AUC
    try:
        metrics["roc_auc"] = float(roc_auc_score(labels, scores))
    except ValueError:
        metrics["roc_auc"] = 0.0

    # Average Precision
    try:
        metrics["average_precision"] = float(average_precision_score(labels, scores))
    except ValueError:
        metrics["average_precision"] = 0.0

    # Hits@K and MRR
    # Sort by score descending, check where positives appear
    pos_mask = labels == 1
    neg_mask = labels == 0

    if pos_mask.sum() > 0 and neg_mask.sum() > 0:
        # For each positive, count how many negatives score higher
        pos_scores = scores[pos_mask]
        neg_scores = scores[neg_mask]

        # Compute rank of each positive among all scores
        ranks = []
        for ps in pos_scores:
            rank = (neg_scores > ps).sum() + 1  # 1-based rank
            ranks.append(rank)
        ranks = np.array(ranks)

        # Hits@K
        hits_k_values = config.get("evaluation", {}).get("hits_k", [10, 50])
        for k in hits_k_values:
            hits = (ranks <= k).mean()
            metrics[f"hits@{k}"] = float(hits)

        # MRR
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
    import csv

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
