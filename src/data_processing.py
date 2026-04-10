"""Data processing pipeline: parse CVD Atlas, download STRING PPI, download GO annotations.

The CVD Atlas Disease-gene_association.txt contains:
- Gene (HGNC symbol), Gene ID (CVD internal), Disease/Trait, Disease/Trait ID,
  Score (0-1 confidence), Gene type, Association ID

No SNP data in this file, so we build a 3-node-type graph: Gene, Disease, GO_term.
"""

import pandas as pd
import requests
import gzip
import logging
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger(__name__)


def parse_cvd_atlas(filepath: str) -> pd.DataFrame:
    """Parse CVD Atlas Disease-gene_association.txt.

    Returns DataFrame with columns:
        gene_symbol, disease_name, score, gene_type, association_id
    """
    logger.info(f"Parsing CVD Atlas file: {filepath}")
    df = pd.read_csv(filepath, sep="\t")
    logger.info(f"Loaded {len(df)} rows with columns: {list(df.columns)}")

    # Rename columns to our schema
    col_map = {
        "Gene": "gene_symbol",
        "Gene ID": "gene_id_cvd",
        "Disease/Trait": "disease_name",
        "Disease/Trait ID": "disease_id_cvd",
        "Score": "score",
        "Gene type": "gene_type",
        "Association ID": "association_id",
    }
    df = df.rename(columns=col_map)

    # Drop rows with missing gene or disease
    df = df.dropna(subset=["gene_symbol", "disease_name"])
    df["gene_symbol"] = df["gene_symbol"].astype(str).str.strip()
    df["disease_name"] = df["disease_name"].astype(str).str.strip()

    # Score to numeric
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0.4)

    # Remove duplicates (keep highest score)
    df = df.sort_values("score", ascending=False).drop_duplicates(
        subset=["gene_symbol", "disease_name"], keep="first"
    )

    logger.info(
        f"After cleaning: {len(df)} associations, "
        f"{df['gene_symbol'].nunique()} genes, "
        f"{df['disease_name'].nunique()} diseases"
    )
    return df


def download_string_ppi(output_dir: str, score_threshold: int = 700) -> pd.DataFrame:
    """Download and filter STRING PPI network for human.

    Maps STRING protein IDs to gene symbols via protein.info file.
    Returns DataFrame with: gene1_symbol, gene2_symbol, combined_score, weight
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # STRING URLs for human (taxid 9606)
    links_url = "https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz"
    info_url = "https://stringdb-downloads.org/download/protein.info.v12.0/9606.protein.info.v12.0.txt.gz"

    links_path = output_dir / "9606.protein.links.txt.gz"
    info_path = output_dir / "9606.protein.info.txt.gz"

    # Download files if not cached
    if not links_path.exists():
        logger.info("Downloading STRING PPI links (~600MB compressed)...")
        _download_file(links_url, links_path)

    if not info_path.exists():
        logger.info("Downloading STRING protein info...")
        _download_file(info_url, info_path)

    # Parse protein info for ENSP -> gene symbol mapping
    logger.info("Parsing STRING protein info...")
    info_df = pd.read_csv(info_path, sep="\t", compression="gzip")
    # STRING IDs: "9606.ENSP00000000233", preferred_name is gene symbol
    ensp_to_symbol = dict(
        zip(
            info_df["#string_protein_id"].str.replace("9606.", "", regex=False),
            info_df["preferred_name"],
        )
    )

    # Parse and filter PPI links
    logger.info("Parsing STRING PPI links (this may take a few minutes)...")
    links_df = pd.read_csv(links_path, sep=" ", compression="gzip")
    logger.info(f"Loaded {len(links_df)} total interactions")

    links_df = links_df[links_df["combined_score"] >= score_threshold]
    logger.info(f"After filtering (score >= {score_threshold}): {len(links_df)} interactions")

    # Map to gene symbols
    links_df["protein1_ensp"] = links_df["protein1"].str.replace("9606.", "", regex=False)
    links_df["protein2_ensp"] = links_df["protein2"].str.replace("9606.", "", regex=False)
    links_df["gene1_symbol"] = links_df["protein1_ensp"].map(ensp_to_symbol)
    links_df["gene2_symbol"] = links_df["protein2_ensp"].map(ensp_to_symbol)

    # Drop unmapped pairs
    links_df = links_df.dropna(subset=["gene1_symbol", "gene2_symbol"])

    result = links_df[["gene1_symbol", "gene2_symbol", "combined_score"]].copy()
    result["weight"] = result["combined_score"] / 1000.0

    logger.info(
        f"STRING PPI: {len(result)} edges, "
        f"{pd.concat([result['gene1_symbol'], result['gene2_symbol']]).nunique()} unique genes"
    )
    return result


def download_go_annotations(output_dir: str) -> pd.DataFrame:
    """Download and parse GO annotations for human genes.

    Assigns weights based on evidence code:
    - Experimental (EXP, IDA, IPI, IMP, IGI, IEP): 1.0
    - Electronic (IEA) and others: 0.6

    Returns DataFrame with: gene_symbol, go_id, evidence_code, go_aspect, weight
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gaf_url = "http://geneontology.org/gene-associations/goa_human.gaf.gz"
    gaf_path = output_dir / "goa_human.gaf.gz"

    if not gaf_path.exists():
        logger.info("Downloading GO annotations...")
        _download_file(gaf_url, gaf_path)

    logger.info("Parsing GO annotations (GAF format)...")
    # GAF columns: DB, DB_Object_ID, DB_Object_Symbol, Qualifier, GO_ID,
    #   DB:Reference, Evidence_Code, With/From, Aspect, ...
    rows = []
    with gzip.open(gaf_path, "rt") as f:
        for line in f:
            if line.startswith("!"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 15:
                continue
            rows.append(
                {
                    "gene_symbol": parts[2],
                    "go_id": parts[4],
                    "evidence_code": parts[6],
                    "go_aspect": parts[8],  # F=MF, P=BP, C=CC
                }
            )

    df = pd.DataFrame(rows)

    # Assign weights by evidence code
    experimental_codes = {"EXP", "IDA", "IPI", "IMP", "IGI", "IEP"}
    df["weight"] = df["evidence_code"].apply(
        lambda x: 1.0 if x in experimental_codes else 0.6
    )

    # Deduplicate (keep highest weight per gene-GO pair)
    df = (
        df.sort_values("weight", ascending=False)
        .drop_duplicates(subset=["gene_symbol", "go_id"], keep="first")
    )

    logger.info(
        f"GO annotations: {len(df)} associations, "
        f"{df['gene_symbol'].nunique()} genes, "
        f"{df['go_id'].nunique()} GO terms"
    )
    return df


def build_gene_id_mapping(gene_symbols: set) -> dict:
    """Map HGNC gene symbols to Ensembl gene IDs via Ensembl BioMart.

    Returns dict: gene_symbol -> ensembl_id
    Falls back to using the symbol as ID if BioMart is unavailable.
    """
    logger.info(f"Building gene ID mapping for {len(gene_symbols)} symbols...")

    symbol_to_ensembl = {}
    symbols_list = sorted(gene_symbols)
    biomart_url = "http://www.ensembl.org/biomart/martservice"
    batch_size = 500

    for i in tqdm(range(0, len(symbols_list), batch_size), desc="Mapping gene IDs"):
        batch = symbols_list[i : i + batch_size]
        symbols_str = ",".join(batch)

        xml_query = f"""<?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE Query>
        <Query virtualSchemaName="default" formatter="TSV" header="1"
               uniqueRows="1" count="" datasetConfigVersion="0.6">
            <Dataset name="hsapiens_gene_ensembl" interface="default">
                <Filter name="hgnc_symbol" value="{symbols_str}"/>
                <Attribute name="hgnc_symbol"/>
                <Attribute name="ensembl_gene_id"/>
            </Dataset>
        </Query>"""

        try:
            resp = requests.get(biomart_url, params={"query": xml_query}, timeout=60)
            resp.raise_for_status()
            for line in resp.text.strip().split("\n")[1:]:  # skip header
                parts = line.split("\t")
                if len(parts) == 2 and parts[0] and parts[1]:
                    symbol_to_ensembl[parts[0]] = parts[1]
        except Exception as e:
            logger.warning(f"BioMart batch {i // batch_size} failed: {e}. Using symbols as fallback.")
            for s in batch:
                symbol_to_ensembl[s] = s

    # Fill unmapped symbols with themselves
    for s in gene_symbols:
        if s not in symbol_to_ensembl:
            symbol_to_ensembl[s] = s

    mapped = len([v for v in symbol_to_ensembl.values() if v.startswith("ENSG")])
    logger.info(f"Mapped {mapped}/{len(gene_symbols)} genes to Ensembl IDs")
    return symbol_to_ensembl


def _download_file(url: str, output_path: Path):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    with open(output_path, "wb") as f:
        with tqdm(total=total, unit="B", unit_scale=True, desc=output_path.name) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))


def process_all(config: dict):
    """Run the full data processing pipeline.

    1. Parse CVD Atlas
    2. Download STRING PPI
    3. Download GO annotations
    4. Build gene ID mapping (HGNC -> Ensembl)
    5. Save intermediate CSVs to data/processed/
    """
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    # 1. Parse CVD Atlas
    cvd_df = parse_cvd_atlas(config["data"]["cvd_atlas_file"])

    # 2. Download STRING PPI
    string_df = download_string_ppi(
        str(raw_dir), score_threshold=config["data"]["string_score_threshold"]
    )

    # 3. Download GO annotations
    go_df = download_go_annotations(str(raw_dir))

    # 4. Collect all gene symbols and build Ensembl mapping
    all_genes = set()
    all_genes.update(cvd_df["gene_symbol"].unique())
    all_genes.update(string_df["gene1_symbol"].unique())
    all_genes.update(string_df["gene2_symbol"].unique())
    all_genes.update(go_df["gene_symbol"].unique())

    gene_id_map = build_gene_id_mapping(all_genes)

    # 5. Save intermediate results
    cvd_df.to_csv(processed_dir / "cvd_associations.csv", index=False)
    string_df.to_csv(processed_dir / "string_ppi.csv", index=False)
    go_df.to_csv(processed_dir / "go_annotations.csv", index=False)

    mapping_df = pd.DataFrame(
        [{"gene_symbol": k, "ensembl_id": v} for k, v in gene_id_map.items()]
    )
    mapping_df.to_csv(processed_dir / "gene_id_mapping.csv", index=False)

    logger.info("Data processing complete. Files saved to data/processed/")


if __name__ == "__main__":
    import argparse
    from src.utils import load_config, setup_logger

    parser = argparse.ArgumentParser(description="Process CVD Atlas and auxiliary data")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    setup_logger("src.data_processing")
    setup_logger(__name__)
    config = load_config(args.config)
    if args.debug:
        config["_debug"] = True
    process_all(config)
