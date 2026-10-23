#!/usr/bin/env python3
"""Build the first-intron length versus spliced-ratio table.

Input is the per-replicate table written by
``src/first_intron/build_first_intron_expression.py`` (columns sample, gene_id, TPM,
total_reads, spliced_reads, unspliced_reads, spliced_ratio) and the transcript
BED ``data/refs/exon_intron_pos.chr.bed``.

A replicate contributes to a gene when TPM > 0 and spliced_ratio is defined
(at least 5 reads covering intron 1). A gene is kept when at least two
replicates of the chosen group contribute. spliced_ratio is the unweighted
mean of those replicate ratios. unspliced_reads, spliced_reads and total_reads
are the sums of the same replicates, so spliced_reads / total_reads is not
equal to spliced_ratio when replicate depths differ.

Intron 1 length is end - start from the BED row named ``<gene_id>_intron1``.

Usage:
    python src/first_intron/build_intron1_length_table.py
    python src/first_intron/build_intron1_length_table.py --group Cold --plot
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

def pipeline_root() -> Path:
    """Return the POINT-seq project that holds results and configs."""
    env = os.environ.get("POINTSEQ_PROJECT_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "results").is_dir() and (candidate / "configs" / "samples.tsv").is_file():
            return candidate
    for candidate in here.parents:
        if (candidate / "src" / "project_config.py").is_file():
            return candidate
    return here.parents[1]


PIPELINE_ROOT = pipeline_root()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Mean first-intron spliced ratio versus intron 1 length."
    )
    parser.add_argument(
        "--by-sample",
        type=Path,
        default=PIPELINE_ROOT / "results" / "expression" / "Warm_1st_intron_by_rep.tsv",
        help="Per-replicate first-intron table (default: results/expression/Warm_1st_intron_by_rep.tsv)",
    )
    parser.add_argument(
        "--bed",
        type=Path,
        default=PIPELINE_ROOT / "data" / "refs" / "exon_intron_pos.chr.bed",
        help="Exon/intron BED (default: data/refs/exon_intron_pos.chr.bed)",
    )
    parser.add_argument(
        "--group",
        default="Warm",
        help="Sample prefix to average, Warm or Cold (default: Warm)",
    )
    parser.add_argument(
        "--min-reps",
        type=int,
        default=2,
        help="Minimum contributing replicates (default: 2)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output TSV (default: results/expression/<group>.mean.elongating.intron1_bp_vs_spliced_ratio.tsv)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Also write the log-x scatter plot next to the TSV",
    )
    return parser.parse_args()


def log(message: str) -> None:
    """Print a progress message to stderr."""
    print(f"[INFO] {message}", file=sys.stderr)


def intron1_lengths(bed_path: Path) -> pd.DataFrame:
    """Return intron 1 length for each transcript in the BED."""
    bed = pd.read_csv(
        bed_path,
        sep="\t",
        header=None,
        names=["chrom", "start", "end", "name", "score", "strand"],
    )
    name = bed["name"].astype(str)
    intron1 = bed.loc[name.str.endswith("_intron1"), ["name", "start", "end"]].copy()
    intron1["gene_id"] = intron1["name"].str.slice(stop=-len("_intron1"))
    intron1["intron1_bp"] = (intron1["end"] - intron1["start"]).astype(int)
    return (
        intron1[["gene_id", "intron1_bp"]]
        .drop_duplicates("gene_id")
        .loc[lambda frame: frame["intron1_bp"] > 0]
    )


def group_mean_table(by_sample: pd.DataFrame, group: str, min_reps: int) -> pd.DataFrame:
    """Average ratios and sum read counts across contributing replicates."""
    required = {
        "sample",
        "gene_id",
        "TPM",
        "spliced_ratio",
        "spliced_reads",
        "unspliced_reads",
        "total_reads",
    }
    missing = required - set(by_sample.columns)
    if missing:
        raise SystemExit(f"[ERROR] by-sample table missing columns: {', '.join(sorted(missing))}")
    sub = by_sample.loc[by_sample["sample"].astype(str).str.startswith(group)].copy()
    sub = sub.loc[(sub["TPM"] > 0) & sub["spliced_ratio"].notna()]
    if sub.empty:
        raise SystemExit(f"[ERROR] no {group} replicates passed TPM > 0 and a defined spliced ratio")
    grouped = sub.groupby("gene_id", observed=True)
    n_reps = grouped.size()
    keep = n_reps[n_reps >= min_reps].index
    out = grouped.agg(
        spliced_ratio=("spliced_ratio", "mean"),
        unspliced_reads=("unspliced_reads", "sum"),
        spliced_reads=("spliced_reads", "sum"),
        total_reads=("total_reads", "sum"),
    ).loc[keep].reset_index()
    for column in ("unspliced_reads", "spliced_reads", "total_reads"):
        out[column] = out[column].astype(int)
    return out


def plot_scatter(table: pd.DataFrame, stem: Path, title: str) -> None:
    """Save the intron-length scatter used for the published PNG."""
    import matplotlib

    matplotlib.use("Agg")
    from plotnine import (
        aes,
        element_blank,
        element_text,
        geom_point,
        geom_smooth,
        ggplot,
        ggtitle,
        labs,
        scale_x_log10,
        theme,
        theme_bw,
    )

    plot_df = table.loc[table["intron1_bp"] > 0, ["intron1_bp", "spliced_ratio"]].dropna()
    cor_r, cor_p = stats.pearsonr(plot_df["intron1_bp"], plot_df["spliced_ratio"])
    p_txt = "p < 1e-300" if (not np.isfinite(cor_p) or cor_p <= 0) else f"p = {cor_p:.3e}"
    plot = (
        ggplot(plot_df, aes(x="intron1_bp", y="spliced_ratio"))
        + geom_point(alpha=0.25, size=1, color="grey")
        + geom_smooth(method="lm", se=False, color="black", size=0.5)
        + theme_bw()
        + theme(
            axis_text=element_text(size=10, color="black"),
            axis_title=element_text(size=12, color="black"),
            panel_grid=element_blank(),
            plot_title=element_text(size=11),
        )
        + labs(x="Intron1 Length (bp)", y="Spliced Ratio")
        + ggtitle(f"{title}: Pearson r = {cor_r:.3f} ({p_txt})")
        + scale_x_log10()
    )
    stem.parent.mkdir(parents=True, exist_ok=True)
    plot.save(stem.with_suffix(".png"), width=4, height=3.5, dpi=600)
    plot.save(stem.with_suffix(".pdf"), width=4, height=3.5)
    log(f"Wrote {stem.with_suffix('.png')}")


def main() -> None:
    """Join intron 1 length with the group-mean splicing table."""
    args = parse_args()
    if args.min_reps < 1:
        raise SystemExit("[ERROR] --min-reps must be at least 1")
    output = args.output
    if output is None:
        output = (
            PIPELINE_ROOT
            / "results"
            / "expression"
            / f"{args.group}.mean.elongating.intron1_bp_vs_spliced_ratio.tsv"
        )
    by_sample = pd.read_csv(args.by_sample, sep="\t")
    lengths = intron1_lengths(args.bed)
    table = group_mean_table(by_sample, args.group, args.min_reps)
    table = table.merge(lengths, on="gene_id", how="inner")
    table = table.sort_values("gene_id", kind="mergesort")
    table = table[
        [
            "gene_id",
            "intron1_bp",
            "spliced_ratio",
            "unspliced_reads",
            "spliced_reads",
            "total_reads",
        ]
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, sep="\t", index=False)
    log(f"Wrote {output} ({len(table):,} genes)")
    if args.plot:
        plot_scatter(table, output.with_suffix(""), f"{args.group}.mean")


if __name__ == "__main__":
    main()
