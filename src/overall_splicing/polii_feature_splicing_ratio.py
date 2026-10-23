#!/usr/bin/env python3
"""Splicing ratio of each intron by the feature that holds the Pol II 3' end.

Reads come from the classify tables. A read contributes to intron k when k is
in spliced_introns or retained_introns. The Pol II position is p3_feat, rewritten
as exonN or intronN. Reads with splicing_class invalid_p5_start are kept
(all-P5 denominator).

Counts are summed within each gene, gene intron number, Pol II feature, and
target intron. A bin is kept when it has more than 10 reads. spliced_ratio is
spliced_reads / total_reads in that bin. Warm.mean and Cold.mean are the unweighted
mean of replicate ratios for the same bin, kept when at least two replicates
contribute.

The boxplot uses Pol II features exon2–exon6 and intron2–intron6. When the 3'
end is inside intron k, the box for target intron k is omitted. The number above
each box is the number of gene bins.

Usage:
    python src/overall_splicing/polii_feature_splicing_ratio.py
    python src/overall_splicing/polii_feature_splicing_ratio.py --samples Warm1,Warm2,Warm3
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
_src = next(p for p in Path(__file__).resolve().parents if (p / "project_config.py").is_file())
sys.path.insert(0, str(_src))
import bootstrap
_ROOT = bootstrap.setup(__file__)
os.chdir(_src)

import pandas as pd  # noqa: E402
from plotnine import (  # noqa: E402
    aes,
    element_text,
    facet_grid,
    geom_boxplot,
    geom_text,
    ggplot,
    labs,
    scale_y_continuous,
    theme,
    theme_bw,
)

from classify_splicing_tables import build_splicing_ratio_table  # noqa: E402
from project_config import (  # noqa: E402
    RESULT_DIRS,
    SAMPLES,
    classify_tsv,
    ensure_result_dirs,
    out_name,
)

MIN_READS = 10
MIN_REPS = 2
FEATURE_INDICES = range(2, 7)
GROUP_COLORS = {"Warm": "#29A2C6", "Cold": "#FFCB18"}
MEAN_KEYS = ["gene_id", "intron_num", "3end_feature_name", "intron_id"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Boxplots of intron splicing ratio by Pol II 3' feature."
    )
    parser.add_argument(
        "--samples",
        default=",".join(SAMPLES),
        help="Comma-separated sample ids (default: every sample in the sheet)",
    )
    parser.add_argument(
        "--min-reads",
        type=int,
        default=MIN_READS,
        help="Minimum reads in a gene bin (default: 10)",
    )
    parser.add_argument(
        "--min-reps",
        type=int,
        default=MIN_REPS,
        help="Minimum replicates in a group mean (default: 2)",
    )
    return parser.parse_args()


def sample_group(sample: str) -> str:
    """Return Warm or Cold from a sample or group-mean label."""
    name = str(sample)
    if name.startswith("Cold"):
        return "Cold"
    return "Warm"


def aggregate_splicing_table(data: pd.DataFrame, min_reads: int) -> pd.DataFrame:
    """Sum reads in each gene bin and drop bins at or below min_reads."""
    grouped = (
        data.groupby(MEAN_KEYS, as_index=False, observed=True)
        .agg(total_reads=("total_reads", "sum"), spliced_reads=("spliced_reads", "sum"))
    )
    grouped["spliced_ratio"] = grouped["spliced_reads"] / grouped["total_reads"]
    return grouped.loc[grouped["total_reads"] > min_reads].copy()


def build_group_mean(
    tables: dict[str, pd.DataFrame],
    reps: list[str],
    label: str,
    min_reps: int,
) -> pd.DataFrame:
    """Average spliced_ratio across replicates for the same gene bin."""
    parts = []
    for sample in reps:
        frame = tables[sample][MEAN_KEYS + ["spliced_ratio"]].copy()
        frame["sample"] = sample
        parts.append(frame)
    stacked = pd.concat(parts, ignore_index=True).dropna(subset=["spliced_ratio"])
    grouped = stacked.groupby(MEAN_KEYS, observed=True)
    n_reps = grouped["sample"].nunique()
    out = grouped.agg(spliced_ratio=("spliced_ratio", "mean")).reset_index()
    out["n_reps"] = out.set_index(MEAN_KEYS).index.map(n_reps).astype(int)
    out = out.loc[out["n_reps"] >= min_reps].copy()
    out["sample"] = label
    return out


def plot_frame(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep exon/intron 2–6 and drop the intron that currently holds Pol II."""
    allowed = [
        f"{feature}{index}"
        for index in FEATURE_INDICES
        for feature in ("exon", "intron")
    ]
    plot_df = data.loc[data["3end_feature_name"].isin(allowed)].copy()
    plot_df["feature_type"] = plot_df["3end_feature_name"].str.extract(
        r"(exon|intron)", expand=False
    )
    plot_df["feature_index"] = (
        plot_df["3end_feature_name"]
        .str.extract(r"(?:exon|intron)(\d+)", expand=False)
        .astype(int)
    )
    target_idx = plot_df["intron_id"].str.extract(r"(\d+)$", expand=False).astype(int)
    plot_df = plot_df.loc[
        ~((plot_df["feature_type"] == "intron") & (target_idx == plot_df["feature_index"]))
    ].copy()
    counts = (
        plot_df.groupby(["feature_type", "feature_index", "intron_id"], observed=True)
        .size()
        .reset_index(name="n")
    )
    return plot_df, counts


def save_plot(data: pd.DataFrame, sample: str, splicing_dir: Path) -> None:
    """Write the faceted boxplot for one sample or one group mean."""
    plot_df, counts = plot_frame(data)
    plot_id = out_name(sample) if sample in SAMPLES else sample
    plot = (
        ggplot(plot_df, aes(x="intron_id", y="spliced_ratio"))
        + geom_boxplot(fill=GROUP_COLORS[sample_group(sample)], width=0.5)
        + geom_text(
            data=counts,
            mapping=aes(x="intron_id", y=1.04, label="n"),
            size=7,
            va="bottom",
            inherit_aes=False,
        )
        + facet_grid("feature_type ~ feature_index", scales="free_x", space="free_x")
        + scale_y_continuous(limits=(0, 1.13))
        + theme_bw()
        + theme(
            axis_text_x=element_text(angle=45, hjust=1),
            panel_grid=element_blank(),
        )
        + labs(
            x="Target intron",
            y="Splicing ratio",
            title=f"{sample}, all P5 features",
        )
    )
    prefix = splicing_dir / f"{plot_id}.all_p5.total_exon_intron.splicing_ratio"
    plot.save(Path(f"{prefix}.png"), width=7, height=5, dpi=600, verbose=False)
    plot.save(Path(f"{prefix}.pdf"), width=7, height=5, verbose=False)
    print(f"[INFO] wrote {prefix.name}.png/.pdf", file=sys.stderr)


def main() -> None:
    """Build all-P5 splicing tables and the Pol II-feature boxplots."""
    args = parse_args()
    samples = [item.strip() for item in args.samples.split(",") if item.strip()]
    unknown = [sample for sample in samples if sample not in SAMPLES]
    if unknown:
        raise SystemExit(f"Unknown samples: {unknown}. Available: {list(SAMPLES)}")

    ensure_result_dirs()
    splicing_dir = RESULT_DIRS["splicing"]
    splicing_dir.mkdir(parents=True, exist_ok=True)

    by_sample: dict[str, pd.DataFrame] = {}
    for sample in samples:
        raw = build_splicing_ratio_table(classify_tsv(sample), include_invalid_p5=True)
        raw_path = splicing_dir / f"{out_name(sample)}.elongating.all_p5.splicing_ratio.tsv"
        raw.to_csv(raw_path, sep="\t", index=False)
        filtered = aggregate_splicing_table(raw, args.min_reads)
        filtered_path = (
            splicing_dir / f"{out_name(sample)}.elongating.all_p5.splicing_ratio.filtered.tsv"
        )
        filtered.to_csv(filtered_path, sep="\t", index=False)
        by_sample[sample] = filtered
        print(
            f"[INFO] {sample}: bins={len(filtered):,} "
            f"reads={int(filtered['total_reads'].sum()):,}",
            file=sys.stderr,
        )
        save_plot(filtered, sample, splicing_dir)

    groups = {
        "Warm": [sample for sample in samples if sample.startswith("Warm")],
        "Cold": [sample for sample in samples if sample.startswith("Cold")],
    }
    for group, reps in groups.items():
        if len(reps) < args.min_reps:
            print(f"[INFO] skip {group}.mean: {len(reps)} replicate(s)", file=sys.stderr)
            continue
        label = f"{group}.mean"
        mean_df = build_group_mean(by_sample, reps, label, args.min_reps)
        mean_path = splicing_dir / f"{label}.elongating.all_p5.splicing_ratio.filtered.tsv"
        mean_df.to_csv(mean_path, sep="\t", index=False)
        print(f"[INFO] {label}: bins={len(mean_df):,}", file=sys.stderr)
        save_plot(mean_df, label, splicing_dir)


if __name__ == "__main__":
    main()
