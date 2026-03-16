# CVD Gene-Disease Prediction with Heterogeneous GNNs

Predict gene-disease associations for cardiovascular diseases using heterogeneous graph neural networks trained on a biomedical knowledge graph derived from the [CVD Atlas](https://ngdc.cncb.ac.cn/cvd/) database.

## Overview

This project builds a heterogeneous biomedical knowledge graph integrating:
- **CVD Atlas**: 215K+ gene-disease associations across 229 cardiovascular diseases
- **STRING DB**: Protein-protein interaction network (filtered to high-confidence ≥700)
- **Gene Ontology**: Functional annotations (molecular function, biological process, cellular component)

Two heterogeneous GNN architectures are compared:
1. **R-GCN** (Relational Graph Convolutional Network) — relation-specific weight matrices with basis decomposition
2. **HGT** (Heterogeneous Graph Transformer) — type-aware multi-head attention

Key techniques:
- **Type-aware contrastive pretraining**: self-supervised node embedding learning with per-type InfoNCE loss
- **DistMult bilinear decoder**: relation-aware scoring for link prediction
- **Mixed negative sampling**: 50% random + 50% hard negatives from PPI neighbors

## Graph Schema

| Node Type | ~Count | Features |
|-----------|--------|----------|
| Gene | ~20K | PPI degree, # diseases, # GO terms, # disease-PPI neighbors |
| Disease | ~229 | # associated genes |
| GO_term | ~18K | Learnable embedding |

| Edge Type (forward / reverse) | Source |
|-------------------------------|--------|
| gene_associated_with_disease / disease_has_gene | CVD Atlas |
| gene_interacts_with_gene / gene_has_interactor | STRING |
| gene_has_function / go_term_of_gene | Gene Ontology |

## Setup

```bash
# Install PyTorch (CPU)
pip install torch==2.6.0
pip install torch-geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.6.0+cpu.html

# Or with CUDA (recommended for full-data training)
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric
pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.6.0+cu121.html

# Other dependencies
pip install pandas numpy scikit-learn pyyaml requests tqdm matplotlib seaborn
```

## Data Preparation

1. Download gene-disease associations from CVD Atlas:
   - Go to https://ngdc.cncb.ac.cn/cvd/
   - Download `Disease-gene_association.txt`
   - Place in `data/raw/`

2. Run data processing (auto-downloads STRING and GO):
```bash
python -m src.data_processing
```

3. Build heterogeneous graph:
```bash
python -m src.build_graph
# For fast debug mode (top 20 diseases only):
python -m src.build_graph --debug
```

## Training

Run all 4 configurations (R-GCN/HGT × pretrained/not):

```bash
# R-GCN with contrastive pretraining
python -m src.train --model rgcn --pretrain

# HGT with contrastive pretraining
python -m src.train --model hgt --pretrain

# R-GCN without pretraining (ablation)
python -m src.train --model rgcn

# HGT without pretraining (ablation)
python -m src.train --model hgt

# Debug mode (fast, reduced graph + fewer epochs):
python -m src.train --model rgcn --pretrain --debug
```

## Evaluation

Compare all trained models:
```bash
python -m src.evaluate
```

Outputs:
- `results/comparison.csv` — metric comparison table
- `results/predictions_*.tsv` — ranked novel gene-disease predictions
- `results/metrics_*.json` — per-model metric files

## Results

Full-data training on all 229 cardiovascular diseases:

| Configuration | ROC-AUC | AP | Hits@10 | Hits@50 | MRR |
|--------------|---------|------|---------|---------|-------|
| **R-GCN + pretrain** | **0.893** | **0.873** | 0.003 | **0.024** | 0.003 |
| R-GCN | 0.892 | 0.871 | **0.004** | 0.021 | **0.003** |
| HGT + pretrain | 0.889 | 0.861 | 0.003 | 0.024 | 0.003 |
| HGT | 0.886 | 0.860 | 0.004 | 0.016 | 0.003 |

R-GCN with contrastive pretraining achieves the best ROC-AUC (0.893) and AP (0.873). Pretraining consistently improves Hits@50 for both architectures, suggesting it helps with ranking quality.

## Metrics

- **ROC-AUC**: Area under ROC curve
- **Average Precision (AP)**: Area under precision-recall curve
- **Hits@10, Hits@50**: Fraction of positives ranked in top K
- **MRR**: Mean Reciprocal Rank

## Project Structure

```
project/
├── data/
│   ├── raw/                      # Disease-gene_association.txt + downloaded files
│   └── processed/                # nodes.csv, edges.csv, node_features.csv
├── src/
│   ├── data_processing.py        # Parse CVD Atlas, download STRING/GO
│   ├── build_graph.py            # Build heterogeneous graph
│   ├── dataset.py                # PyG HeteroData loader + edge splitting
│   ├── contrastive_pretrain.py   # Type-aware contrastive pretraining
│   ├── models/
│   │   ├── rgcn_model.py         # R-GCN encoder
│   │   ├── hgt_model.py          # HGT encoder
│   │   └── decoder.py            # DistMult bilinear decoder
│   ├── train.py                  # Unified training script
│   ├── evaluate.py               # Metrics + model comparison
│   └── utils.py                  # Config, seeding, logging
├── configs/
│   └── default.yaml              # All hyperparameters
├── results/                      # Checkpoints, metrics, predictions
├── requirements.txt
└── README.md
```

## Configuration

All hyperparameters in `configs/default.yaml`. Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| hidden_dim | 128 | GNN hidden dimension |
| num_layers | 2 | GNN depth |
| contrastive epochs | 100 | Pretraining epochs |
| supervised epochs | 200 | Training epochs |
| early_stopping_patience | 20 | Validation AP patience |
| tau | 0.5 | Contrastive temperature |
| negative_ratio | 1.0 | Neg:pos sampling ratio |
| hard_negative_ratio | 0.5 | Fraction of hard negatives |
