#!/usr/bin/env python3
"""Build intron-level splicing summaries from a classify TSV.

Input is the headered TSV produced by ``classify_bam_splicing_readthrough.py``.
Only direct intron evidence is counted: IDs in ``spliced_introns`` are spliced
and IDs in ``retained_introns`` are unspliced. Alternative splice junctions are
already included in ``spliced_introns`` by the classifier.

The output schema matches the historical splicing-ratio tables so existing
notebooks can consume the new classify-based results without redefining
splicing.

Usage:
    python src/splicing_ratio/classify_splicing_tables.py \
      --classify results/read_through/601.elongating.classify.tsv \
      --output results/splicing/601.elongating.splicing_ratio.tsv
    python src/splicing_ratio/classify_splicing_tables.py --classify input.tsv --output out.tsv \
      --min-reads 5
    python src/splicing_ratio/classify_splicing_tables.py --classify input.tsv --output all_p5.tsv \
      --include-invalid-p5
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

CLASSIFY_COLUMNS = (
    "gene_id",
    "n_introns",
    "spliced_introns",
    "retained_introns",
    "p3_feat",
    "splicing_class",
)
NEW_CLASSIFY_COLUMNS = frozenset(
    {
        "alternative_spliced_introns",
        "alternative_junctions",
    }
)
EXCLUDED_CLASSES = frozenset({"invalid_p5_start"})
INCLUDE_INVALID_P5_ENV = "POINTSEQ_INCLUDE_INVALID_P5"
TRUE_ENV = frozenset({"1", "true", "yes", "y"})
OUTPUT_COLUMNS = (
    "gene_id",
    "intron_num",
    "3end_feature_name",
    "intron_id",
    "total_reads",
    "spliced_reads",
    "unspliced_reads",
    "spliced_ratio",
    "unspliced_ratio",
)


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classify", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--min-reads",
        type=int,
        default=0,
        help="Minimum informative reads per output row (default: 0)",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=250_000,
        help="Rows read per pandas chunk (default: 250000)",
    )
    parser.add_argument(
        "--include-invalid-p5",
        action="store_true",
        default=None,
        help=(
            "Include reads whose P5 is outside promoter/exon_1. Only introns with "
            "direct spliced or retained evidence are counted. Default follows "
            "POINTSEQ_INCLUDE_INVALID_P5 in P5 mode."
        ),
    )
    return parser.parse_args()


# ============================================================
# DATA PROCESSING
# ============================================================

def parse_intron_ids(value: object) -> set[int]:
    """Return positive intron indices from one comma-separated field."""
    if pd.isna(value):
        return set()
    result: set[int] = set()
    for token in str(value).split(","):
        token = token.strip()
        if token.isdigit() and int(token) > 0:
            result.add(int(token))
    return result


def normalize_p3_feature(value: object) -> str:
    """Convert classifier labels such as exon_3 to legacy exon3 labels."""
    if pd.isna(value):
        return "unknown"
    return str(value).strip().replace("_", "")


def include_invalid_p5_default() -> bool:
    """Return True when P5-in-intron reads should enter splicing ratios.

    Original analyses omit this flag (False). P5 mode sets
    ``POINTSEQ_INCLUDE_INVALID_P5=1`` so every caller that does not
    explicitly pass ``include_invalid_p5`` keeps those reads.
    """
    return os.environ.get(INCLUDE_INVALID_P5_ENV, "").strip().lower() in TRUE_ENV


def build_splicing_ratio_table(
    classify_path: Path,
    *,
    chunksize: int = 250_000,
    min_reads: int = 0,
    include_invalid_p5: bool | None = None,
) -> pd.DataFrame:
    """Aggregate direct intron evidence, optionally without restricting P5."""
    if include_invalid_p5 is None:
        include_invalid_p5 = include_invalid_p5_default()
    header = set(pd.read_csv(classify_path, sep="\t", nrows=0).columns)
    missing_new_columns = NEW_CLASSIFY_COLUMNS.difference(header)
    if missing_new_columns:
        raise ValueError(
            f"{classify_path} is an old classification table; missing "
            f"{sorted(missing_new_columns)}. Rebuild it with "
            "classify_bam_splicing_readthrough.py before splicing analysis."
        )
    counts: defaultdict[tuple[str, int, str, int], list[int]] = defaultdict(
        lambda: [0, 0]
    )

    for chunk in pd.read_csv(
        classify_path,
        sep="\t",
        usecols=CLASSIFY_COLUMNS,
        dtype=str,
        chunksize=chunksize,
        low_memory=False,
    ):
        if not include_invalid_p5:
            chunk = chunk.loc[
                ~chunk["splicing_class"].isin(EXCLUDED_CLASSES)
            ]
        for row in chunk.itertuples(index=False):
            spliced = parse_intron_ids(row.spliced_introns)
            retained = parse_intron_ids(row.retained_introns)
            if not spliced and not retained:
                continue
            try:
                intron_num = int(row.n_introns)
            except (TypeError, ValueError):
                continue
            p3_feature = normalize_p3_feature(row.p3_feat)
            for intron_idx in spliced:
                counts[(row.gene_id, intron_num, p3_feature, intron_idx)][0] += 1
            for intron_idx in retained - spliced:
                counts[(row.gene_id, intron_num, p3_feature, intron_idx)][1] += 1

    rows = []
    for (gene_id, intron_num, p3_feature, intron_idx), (
        spliced_reads,
        unspliced_reads,
    ) in sorted(counts.items()):
        total_reads = spliced_reads + unspliced_reads
        if total_reads < min_reads:
            continue
        rows.append(
            {
                "gene_id": gene_id,
                "intron_num": intron_num,
                "3end_feature_name": p3_feature,
                "intron_id": f"intron{intron_idx}",
                "total_reads": total_reads,
                "spliced_reads": spliced_reads,
                "unspliced_reads": unspliced_reads,
                "spliced_ratio": spliced_reads / total_reads,
                "unspliced_ratio": unspliced_reads / total_reads,
            }
        )
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def summarize_first_intron(
    classify_path: Path,
    *,
    chunksize: int = 250_000,
) -> pd.DataFrame:
    """Return per-gene spliced and retained counts for intron 1."""
    table = build_splicing_ratio_table(classify_path, chunksize=chunksize)
    first = table.loc[table["intron_id"].eq("intron1")].copy()
    if first.empty:
        return pd.DataFrame(
            columns=("gene_id", "total_reads", "spliced_reads", "unspliced_reads")
        )
    return (
        first.groupby("gene_id", as_index=False, observed=True)
        .agg(
            total_reads=("total_reads", "sum"),
            spliced_reads=("spliced_reads", "sum"),
            unspliced_reads=("unspliced_reads", "sum"),
        )
    )


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    """Build and write one classify-derived splicing table."""
    args = parse_args()
    if args.min_reads < 0:
        raise ValueError("--min-reads must be >= 0")
    if not args.classify.is_file():
        print(f"[ERROR] Classification TSV not found: {args.classify}", file=sys.stderr)
        return 1

    result = build_splicing_ratio_table(
        args.classify,
        chunksize=args.chunksize,
        min_reads=args.min_reads,
        include_invalid_p5=args.include_invalid_p5,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, sep="\t", index=False)
    print(f"[INFO] Wrote {len(result):,} rows to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
