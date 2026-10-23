#!/usr/bin/env python3
"""Build expression and first-intron splicing tables using classify TSVs.

Read-info tables provide expression counts and transcript lengths. All
first-intron spliced/retained counts come from the corresponding classify TSV.
The script writes pooled and per-sample tables for
``2.splicing_expression.ipynb``.

Usage:
    python src/first_intron/build_first_intron_expression.py
    python src/first_intron/build_first_intron_expression.py --min-informative-reads 5
    python src/first_intron/build_first_intron_expression.py --output results/expression/custom.tsv
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

_src = next(p for p in Path(__file__).resolve().parents if (p / "project_config.py").is_file())
sys.path.insert(0, str(_src))
import bootstrap
bootstrap.setup(__file__)

from classify_splicing_tables import summarize_first_intron
from project_config import (
    RESULT_DIRS,
    SAMPLES,
    classify_tsv,
    existing_read_info_tsv,
)

USE_COLUMNS = (
    "mRNA",
    "mRNA_length",
)


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build TPM and first-intron splicing tables from read info."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULT_DIRS["expression"] / "Warm_1st_intron_df.tsv",
        help="Pooled output TSV",
    )
    parser.add_argument(
        "--by-sample-output",
        type=Path,
        default=RESULT_DIRS["expression"] / "Warm_1st_intron_by_rep.tsv",
        help="Per-sample output TSV",
    )
    parser.add_argument(
        "--min-informative-reads",
        type=int,
        default=5,
        help="Minimum reads spanning intron 1 required for a splicing ratio",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=250_000,
        help="Rows read per pandas chunk",
    )
    return parser.parse_args()


# ============================================================
# DATA PROCESSING
# ============================================================

def summarize_expression(
    path: Path,
    sample: str,
    chunksize: int,
) -> pd.DataFrame:
    """Aggregate expression counts and transcript lengths for one sample."""
    read_counts: defaultdict[str, int] = defaultdict(int)
    lengths: dict[str, int] = {}

    for chunk in pd.read_csv(
        path,
        sep="\t",
        usecols=USE_COLUMNS,
        chunksize=chunksize,
        low_memory=False,
    ):
        chunk["mRNA"] = chunk["mRNA"].astype(str)
        counts = chunk["mRNA"].value_counts()
        for gene_id, count in counts.items():
            read_counts[gene_id] += int(count)

        length_rows = chunk[["mRNA", "mRNA_length"]].drop_duplicates("mRNA")
        for row in length_rows.itertuples(index=False):
            lengths[row.mRNA] = int(row.mRNA_length)

    rows = []
    for gene_id in sorted(read_counts):
        rows.append(
            {
                "sample": sample,
                "gene_id": gene_id,
                "reads": read_counts[gene_id],
                "length": lengths.get(gene_id, np.nan),
            }
        )

    result = pd.DataFrame(rows)
    result["RPK"] = result["reads"] / result["length"] * 1_000
    rpk_sum = result["RPK"].sum()
    result["TPM"] = result["RPK"] / rpk_sum * 1_000_000
    return result


def pooled_table(by_sample: pd.DataFrame, min_reads: int) -> pd.DataFrame:
    """Pool sample counts and calculate combined TPM and splicing ratio."""
    pooled = (
        by_sample.groupby("gene_id", as_index=False, observed=True)
        .agg(
            reads=("reads", "sum"),
            length=("length", "first"),
            total_reads=("total_reads", "sum"),
            spliced_reads=("spliced_reads", "sum"),
            unspliced_reads=("unspliced_reads", "sum"),
        )
    )
    pooled["RPK"] = pooled["reads"] / pooled["length"] * 1_000
    pooled["TPM"] = pooled["RPK"] / pooled["RPK"].sum() * 1_000_000
    pooled["spliced_ratio"] = (
        pooled["spliced_reads"] / pooled["total_reads"].replace(0, np.nan)
    )
    pooled.loc[pooled["total_reads"] < min_reads, "spliced_ratio"] = np.nan
    return pooled


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    """Build and write pooled and per-sample expression tables."""
    args = parse_args()
    if args.min_informative_reads < 1:
        raise ValueError("--min-informative-reads must be at least 1")

    sample_tables = []
    for sample in SAMPLES:
        read_info_path = existing_read_info_tsv(sample)
        classify_path = classify_tsv(sample)
        if not read_info_path.is_file():
            print(f"[ERROR] Missing read-info table for {sample}: {read_info_path}", file=sys.stderr)
            return 1
        if not classify_path.is_file():
            print(f"[ERROR] Missing classify table for {sample}: {classify_path}", file=sys.stderr)
            return 1
        print(f"[INFO] Reading expression for {sample}: {read_info_path}", file=sys.stderr)
        expression = summarize_expression(read_info_path, sample, args.chunksize)
        print(f"[INFO] Reading splicing for {sample}: {classify_path}", file=sys.stderr)
        first_intron = summarize_first_intron(
            classify_path,
            chunksize=args.chunksize,
        )
        sample_table = expression.merge(first_intron, on="gene_id", how="left")
        for column in ("total_reads", "spliced_reads", "unspliced_reads"):
            sample_table[column] = sample_table[column].fillna(0).astype(int)
        sample_table["spliced_ratio"] = (
            sample_table["spliced_reads"]
            / sample_table["total_reads"].replace(0, np.nan)
        )
        sample_tables.append(sample_table)

    by_sample = pd.concat(sample_tables, ignore_index=True)
    by_sample.loc[
        by_sample["total_reads"] < args.min_informative_reads,
        "spliced_ratio",
    ] = np.nan
    pooled = pooled_table(by_sample, args.min_informative_reads)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.by_sample_output.parent.mkdir(parents=True, exist_ok=True)
    pooled.to_csv(args.output, sep="\t", index=False)
    by_sample.to_csv(args.by_sample_output, sep="\t", index=False)
    print(f"[INFO] Wrote pooled table: {args.output}", file=sys.stderr)
    print(f"[INFO] Wrote per-sample table: {args.by_sample_output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
