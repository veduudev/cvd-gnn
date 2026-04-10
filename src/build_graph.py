"""Build heterogeneous knowledge graph from processed data.

Creates nodes.csv, edges.csv, node_features.csv from:
- CVD Atlas gene-disease associations
- STRING PPI interactions
- GO functional annotations

Node types: gene, disease, go_term (no SNPs in CVD Atlas gene-disease file)
Edge types (forward + reverse): 6 total
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import argparse

from src.utils import load_config, setup_logger

logger = logging.getLogger(__name__)


def build_graph(config: dict, debug: bool = False):
    """Build heterogeneous graph from processed data files.

    In debug mode, filters to top N diseases by association count
    to enable fast iteration.
    """
    processed_dir = Path("data/processed")

    # Load processed data
    cvd_df = pd.read_csv(processed_dir / "cvd_associations.csv")
    string_df = pd.read_csv(processed_dir / "string_ppi.csv")
    go_df = pd.read_csv(processed_dir / "go_annotations.csv")
    gene_map = pd.read_csv(processed_dir / "gene_id_mapping.csv")

    symbol_to_ensembl = dict(zip(gene_map["gene_symbol"], gene_map["ensembl_id"]))

    # Debug mode: filter to top N diseases by association count
    if debug:
        top_n = config["data"]["debug_top_diseases"]
        disease_counts = cvd_df["disease_name"].value_counts()
        top_diseases = disease_counts.head(top_n).index.tolist()
        cvd_df = cvd_df[cvd_df["disease_name"].isin(top_diseases)]
        logger.info(f"Debug mode: filtered to top {top_n} diseases ({len(cvd_df)} associations)")

    # --- Build node registry ---
    nodes = []
    node_id_counter = {"gene": 0, "disease": 0, "go_term": 0}
    node_registry = {}  # (node_type, name) -> int_id

    def get_or_create_node(node_type, name, metadata=""):
        key = (node_type, name)
        if key not in node_registry:
            nid = node_id_counter[node_type]
            node_id_counter[node_type] += 1
            node_registry[key] = nid
            # Map gene symbols to Ensembl for internal ID
            internal_id = symbol_to_ensembl.get(name, name) if node_type == "gene" else name
            nodes.append(
                {
                    "node_id": nid,
                    "node_type": node_type,
                    "name": name,
                    "internal_id": internal_id,
                    "metadata": metadata,
                }
            )
        return node_registry[key]

    # --- Build edges ---
    edges = []

    # 1. Gene-disease associations from CVD Atlas
    # Weight = Score from CVD Atlas (0-1 range, higher = stronger evidence)
    logger.info("Building gene-disease edges from CVD Atlas...")
    for _, row in cvd_df.iterrows():
        gene_symbol = str(row["gene_symbol"]).strip()
        disease = str(row["disease_name"]).strip()

        gene_nid = get_or_create_node("gene", gene_symbol, metadata=str(row.get("gene_type", "")))
        disease_nid = get_or_create_node("disease", disease)

        weight = float(row.get("score", 0.4))

        # Forward: gene -> disease
        edges.append(
            {
                "source_type": "gene",
                "source_id": gene_nid,
                "relation_type": "gene_associated_with_disease",
                "target_type": "disease",
                "target_id": disease_nid,
                "weight": weight,
            }
        )
        # Reverse: disease -> gene
        edges.append(
            {
                "source_type": "disease",
                "source_id": disease_nid,
                "relation_type": "disease_has_gene",
                "target_type": "gene",
                "target_id": gene_nid,
                "weight": weight,
            }
        )

    # 2. Gene-gene PPI from STRING (only for genes already in graph)
    logger.info("Building PPI edges from STRING...")
    known_genes = {name for (ntype, name) in node_registry if ntype == "gene"}
    ppi_count = 0
    for _, row in string_df.iterrows():
        g1 = str(row["gene1_symbol"])
        g2 = str(row["gene2_symbol"])
        if g1 in known_genes and g2 in known_genes and g1 != g2:
            g1_nid = get_or_create_node("gene", g1)
            g2_nid = get_or_create_node("gene", g2)
            w = float(row["weight"])
            # Forward
            edges.append(
                {
                    "source_type": "gene",
                    "source_id": g1_nid,
                    "relation_type": "gene_interacts_with_gene",
                    "target_type": "gene",
                    "target_id": g2_nid,
                    "weight": w,
                }
            )
            # Reverse
            edges.append(
                {
                    "source_type": "gene",
                    "source_id": g2_nid,
                    "relation_type": "gene_has_interactor",
                    "target_type": "gene",
                    "target_id": g1_nid,
                    "weight": w,
                }
            )
            ppi_count += 1
    logger.info(f"Added {ppi_count} PPI edges (x2 with reverse)")

    # 3. Gene-GO from GO annotations (only for genes already in graph)
    logger.info("Building GO annotation edges...")
    known_genes = {name for (ntype, name) in node_registry if ntype == "gene"}
    go_count = 0
    for _, row in go_df.iterrows():
        gene = str(row["gene_symbol"])
        if gene in known_genes:
            gene_nid = get_or_create_node("gene", gene)
            go_nid = get_or_create_node("go_term", str(row["go_id"]))
            w = float(row["weight"])
            # Forward
            edges.append(
                {
                    "source_type": "gene",
                    "source_id": gene_nid,
                    "relation_type": "gene_has_function",
                    "target_type": "go_term",
                    "target_id": go_nid,
                    "weight": w,
                }
            )
            # Reverse
            edges.append(
                {
                    "source_type": "go_term",
                    "source_id": go_nid,
                    "relation_type": "go_term_of_gene",
                    "target_type": "gene",
                    "target_id": gene_nid,
                    "weight": w,
                }
            )
            go_count += 1
    logger.info(f"Added {go_count} GO annotation edges (x2 with reverse)")

    # --- Build DataFrames ---
    nodes_df = pd.DataFrame(nodes)
    edges_df = pd.DataFrame(edges)

    # --- Compute node features ---
    logger.info("Computing node features...")
    features_df = compute_node_features(nodes_df, edges_df)

    # --- Save ---
    nodes_df.to_csv(processed_dir / "nodes.csv", index=False)
    edges_df.to_csv(processed_dir / "edges.csv", index=False)
    features_df.to_csv(processed_dir / "node_features.csv", index=False)

    # Print summary
    logger.info("=== Graph Summary ===")
    for ntype in ["gene", "disease", "go_term"]:
        count = len(nodes_df[nodes_df["node_type"] == ntype])
        logger.info(f"  {ntype}: {count} nodes")
    for rtype in edges_df["relation_type"].unique():
        count = len(edges_df[edges_df["relation_type"] == rtype])
        logger.info(f"  {rtype}: {count} edges")
    logger.info(f"  Total: {len(nodes_df)} nodes, {len(edges_df)} edges")
    logger.info("Graph construction complete.")


def compute_node_features(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> pd.DataFrame:
    """Compute and z-score normalize node features.

    Gene features (4): PPI degree, # diseases, # GO terms, # disease-gene PPI neighbors
    Disease features (2): # associated genes, (placeholder for uniformity)
    GO_term features (0): learnable embeddings only (stored as empty)
    """
    rows = []

    # Precompute edge lookups for speed
    gene_ppi = edges_df[
        (edges_df["source_type"] == "gene")
        & (edges_df["relation_type"] == "gene_interacts_with_gene")
    ]
    gene_disease = edges_df[
        (edges_df["source_type"] == "gene")
        & (edges_df["relation_type"] == "gene_associated_with_disease")
    ]
    gene_go = edges_df[
        (edges_df["source_type"] == "gene") & (edges_df["relation_type"] == "gene_has_function")
    ]
    disease_gene = edges_df[
        (edges_df["source_type"] == "disease") & (edges_df["relation_type"] == "disease_has_gene")
    ]

    # PPI degree per gene
    ppi_degree = gene_ppi.groupby("source_id").size().to_dict()

    # Disease count per gene
    disease_count = gene_disease.groupby("source_id").size().to_dict()

    # GO count per gene
    go_count = gene_go.groupby("source_id").size().to_dict()

    # Disease genes set (genes associated with any disease)
    disease_gene_set = set(gene_disease["source_id"].unique())

    # PPI neighbors per gene
    ppi_neighbors = gene_ppi.groupby("source_id")["target_id"].apply(set).to_dict()

    # Gene features
    gene_nodes = nodes_df[nodes_df["node_type"] == "gene"]
    for _, node in gene_nodes.iterrows():
        nid = node["node_id"]
        feat_ppi_degree = ppi_degree.get(nid, 0)
        feat_n_diseases = disease_count.get(nid, 0)
        feat_n_go = go_count.get(nid, 0)
        # Disease-gene PPI neighbors: how many PPI neighbors are disease genes
        neighbors = ppi_neighbors.get(nid, set())
        feat_disease_ppi = len(neighbors & disease_gene_set)
        rows.append(
            {
                "node_type": "gene",
                "node_id": nid,
                "f0": feat_ppi_degree,
                "f1": feat_n_diseases,
                "f2": feat_n_go,
                "f3": feat_disease_ppi,
            }
        )

    # Disease features
    gene_count_per_disease = disease_gene.groupby("source_id").size().to_dict()
    disease_nodes = nodes_df[nodes_df["node_type"] == "disease"]
    for _, node in disease_nodes.iterrows():
        nid = node["node_id"]
        feat_n_genes = gene_count_per_disease.get(nid, 0)
        rows.append(
            {
                "node_type": "disease",
                "node_id": nid,
                "f0": feat_n_genes,
                "f1": 0.0,  # placeholder for uniform feature dim
                "f2": 0.0,
                "f3": 0.0,
            }
        )

    # GO_term features: learnable only, store zeros
    go_nodes = nodes_df[nodes_df["node_type"] == "go_term"]
    for _, node in go_nodes.iterrows():
        rows.append(
            {
                "node_type": "go_term",
                "node_id": node["node_id"],
                "f0": 0.0,
                "f1": 0.0,
                "f2": 0.0,
                "f3": 0.0,
            }
        )

    feat_df = pd.DataFrame(rows)

    # Z-score normalize per node type (skip go_term which is all zeros)
    feat_cols = ["f0", "f1", "f2", "f3"]
    for ntype in ["gene", "disease"]:
        mask = feat_df["node_type"] == ntype
        for col in feat_cols:
            vals = feat_df.loc[mask, col]
            std = vals.std()
            if std > 0:
                feat_df.loc[mask, col] = (vals - vals.mean()) / std
            else:
                feat_df.loc[mask, col] = 0.0

    return feat_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build heterogeneous knowledge graph")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    setup_logger("src.build_graph")
    setup_logger(__name__)
    config = load_config(args.config)
    build_graph(config, debug=args.debug)
