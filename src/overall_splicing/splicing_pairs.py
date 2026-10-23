#!/usr/bin/env python3
"""Splicing pairs: which intron of an adjacent pair is spliced first.

A pair counts when two neighboring introns are both called and they disagree:
one is in spliced_introns and the other is in retained_introns. Upstream spliced
and downstream retained is "upstream first". The reverse is "downstream first".
Reads must start in promoter or exon_1. Genes are split by intron number:
2–9 and 10+.

Warm.mean and Cold.mean are the mean of the three replicate percentages. The error
bar is the standard deviation of those percentages.

Usage:
    python src/overall_splicing/splicing_pairs.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
_src = next(p for p in Path(__file__).resolve().parents if (p / "project_config.py").is_file())
sys.path.insert(0, str(_src))
import bootstrap
_ROOT = bootstrap.setup(__file__)
os.chdir(_src)

import numpy as np
import pandas as pd
from plotnine import (
    aes,
    element_blank,
    geom_col,
    geom_errorbar,
    geom_text,
    ggplot,
    labs,
    position_stack,
    scale_fill_manual,
    scale_y_continuous,
    theme,
    theme_bw,
)

from project_config import RESULT_DIRS, classify_tsv, ensure_result_dirs

SAMPLES = ("Warm1", "Warm2", "Warm3", "Cold1", "Cold2", "Cold3")
VALID_P5_FEATURES = {"promoter", "exon_1"}
INTRON_GROUPS = [str(n) for n in range(2, 10)] + ["10+"]
ORDER_LEVELS = (
    "Upstream intron spliced first",
    "Downstream intron spliced first",
)
ORDER_COLORS = {
    "Upstream intron spliced first": "#F8766D",
    "Downstream intron spliced first": "#00BFC4",
}


def intron_set(value) -> set[int]:
    """Parse a comma-separated intron-index field."""
    if pd.isna(value) or str(value).strip() == "":
        return set()
    return {int(item) for item in str(value).split(",")}


def intron_status_map(row) -> dict[int, str]:
    """Return spliced or retained for introns called on one read.

    A retained call overwrites a spliced call when both lists contain the index.
    """
    states = {idx: "spliced" for idx in intron_set(row.spliced_introns)}
    for idx in intron_set(row.retained_introns):
        states[idx] = "retained"
    return states


def intron_group(n_introns: int) -> str | None:
    """Return 2–9 as written, and 10 or more as 10+."""
    if n_introns < 2:
        return None
    return str(n_introns) if n_introns < 10 else "10+"


def count_splicing_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Count discordant adjacent pairs by which intron is spliced."""
    counts = {label: 0 for label in ORDER_LEVELS}
    for row in df.itertuples(index=False):
        states = intron_status_map(row)
        n_introns = int(row.n_introns)
        for upstream_idx in range(1, n_introns):
            upstream = states.get(upstream_idx)
            downstream = states.get(upstream_idx + 1)
            if upstream == "spliced" and downstream == "retained":
                counts["Upstream intron spliced first"] += 1
            elif upstream == "retained" and downstream == "spliced":
                counts["Downstream intron spliced first"] += 1
    result = pd.DataFrame({"splice_order": list(ORDER_LEVELS), "n_pairs": [counts[label] for label in ORDER_LEVELS]})
    total = result["n_pairs"].sum()
    result["percentage"] = 100 * result["n_pairs"] / total if total else np.nan
    result["splice_order"] = pd.Categorical(result["splice_order"], categories=list(ORDER_LEVELS), ordered=True)
    return result


def intron_group_distribution(df: pd.DataFrame, sample: str) -> pd.DataFrame:
    """Read and gene counts in each intron-number bin."""
    work = df.loc[df["n_introns"].ge(2), ["gene_id", "n_introns"]].copy()
    work["intron_group"] = work["n_introns"].map(lambda n: intron_group(int(n)))
    read_counts = work["intron_group"].value_counts()
    gene_counts = work.groupby("intron_group", observed=True)["gene_id"].nunique()
    total_reads = len(work)
    total_genes = work["gene_id"].nunique()
    rows = []
    for group in INTRON_GROUPS:
        n_reads = int(read_counts.get(group, 0))
        n_genes = int(gene_counts.get(group, 0))
        rows.append(
            {
                "sample": sample,
                "intron_group": group,
                "n_reads": n_reads,
                "n_genes": n_genes,
                "read_percentage": 100 * n_reads / total_reads if total_reads else np.nan,
                "gene_percentage": 100 * n_genes / total_genes if total_genes else np.nan,
            }
        )
    result = pd.DataFrame(rows)
    result["intron_group"] = pd.Categorical(result["intron_group"], categories=INTRON_GROUPS, ordered=True)
    return result


def count_splicing_pairs_by_group(df: pd.DataFrame) -> pd.DataFrame:
    """Count discordant adjacent pairs inside each intron-number bin."""
    counts = {group: {label: 0 for label in ORDER_LEVELS} for group in INTRON_GROUPS}
    for row in df.itertuples(index=False):
        group = intron_group(int(row.n_introns))
        if group is None:
            continue
        states = intron_status_map(row)
        for upstream_idx in range(1, int(row.n_introns)):
            upstream = states.get(upstream_idx)
            downstream = states.get(upstream_idx + 1)
            if upstream == "spliced" and downstream == "retained":
                counts[group]["Upstream intron spliced first"] += 1
            elif upstream == "retained" and downstream == "spliced":
                counts[group]["Downstream intron spliced first"] += 1
    rows = []
    for group in INTRON_GROUPS:
        total_pairs = sum(counts[group].values())
        for splice_order in ORDER_LEVELS:
            n_pairs = counts[group][splice_order]
            rows.append(
                {
                    "intron_group": group,
                    "splice_order": splice_order,
                    "n_pairs": n_pairs,
                    "percentage": 100 * n_pairs / total_pairs if total_pairs else np.nan,
                }
            )
    result = pd.DataFrame(rows)
    result["intron_group"] = pd.Categorical(result["intron_group"], categories=INTRON_GROUPS, ordered=True)
    result["splice_order"] = pd.Categorical(result["splice_order"], categories=list(ORDER_LEVELS), ordered=True)
    return result


def summarize_group_mean(frames: list[pd.DataFrame], label: str) -> pd.DataFrame:
    """Mean and SD of splicing-pair percentages across replicates."""
    stacked = pd.concat(frames, ignore_index=True)
    out = (
        stacked.groupby(["intron_group", "splice_order"], observed=False)
        .agg(
            percentage=("percentage", "mean"),
            sd=("percentage", "std"),
            n_pairs=("n_pairs", "sum"),
            n_reps=("sample", "nunique"),
        )
        .reset_index()
    )
    out["sd"] = out["sd"].fillna(0)
    out["ymin"] = out["percentage"] - out["sd"]
    out["ymax"] = out["percentage"] + out["sd"]
    out["intron_group"] = pd.Categorical(out["intron_group"].astype(str), categories=INTRON_GROUPS, ordered=True)
    out["splice_order"] = pd.Categorical(out["splice_order"], categories=list(ORDER_LEVELS), ordered=True)
    out["sample"] = label
    return out.sort_values(["intron_group", "splice_order"])


def replicates_wide(frames: list[pd.DataFrame], samples: tuple[str, ...]) -> pd.DataFrame:
    """One row per intron group and splice order, with replicates side by side."""
    stacked = pd.concat(frames, ignore_index=True)
    wide = stacked.pivot(index=["intron_group", "splice_order"], columns="sample", values="percentage")
    wide = wide.reindex(columns=list(samples))
    wide["mean"] = wide[list(samples)].mean(axis=1)
    wide["sd"] = wide[list(samples)].std(axis=1, ddof=1)
    wide = wide.reset_index()
    wide["intron_group"] = pd.Categorical(wide["intron_group"].astype(str), categories=INTRON_GROUPS, ordered=True)
    wide["splice_order"] = pd.Categorical(wide["splice_order"], categories=list(ORDER_LEVELS), ordered=True)
    return wide.sort_values(["intron_group", "splice_order"]).reset_index(drop=True)


def plot_theme():
    """Shared axis theme."""
    from plotnine import element_text

    return theme(
        panel_grid=element_blank(),
        axis_text=element_text(color="black", size=12),
        axis_title=element_text(color="black", size=12),
    )


def plot_overall(pair_df: pd.DataFrame, sample: str):
    """Stacked bar of upstream-first versus downstream-first pairs."""
    return (
        ggplot(pair_df, aes(x="sample", y="percentage", fill="splice_order"))
        + geom_col(width=0.65)
        + geom_text(
            aes(label="percentage"),
            position=position_stack(vjust=0.5),
            format_string="{:.1f}%",
            size=10,
        )
        + scale_fill_manual(values=ORDER_COLORS)
        + scale_y_continuous(limits=(0, 100), expand=(0, 0))
        + theme_bw()
        + plot_theme()
        + theme(axis_title_x=element_blank(), legend_title=element_blank(), legend_position="bottom")
        + labs(y="Splicing pairs (%)", title=f"{sample}: splicing pairs")
    )


def plot_distribution(dist_df: pd.DataFrame, sample: str):
    """Read counts across intron-number bins."""
    return (
        ggplot(dist_df, aes(x="intron_group", y="read_percentage", fill="intron_group"))
        + geom_col(width=0.65)
        + geom_text(aes(label="n_reads"), va="bottom", size=9, format_string="{:,}")
        + scale_y_continuous(expand=(0, 0.08))
        + theme_bw()
        + plot_theme()
        + theme(axis_title_x=element_blank(), legend_position="none")
        + labs(y="Reads (%)", title=f"{sample}: splicing pairs, intron-number distribution")
    )


def plot_by_group(pair_df: pd.DataFrame, sample: str):
    """Splicing pairs split by intron number."""
    plot_df = pair_df.loc[pair_df.groupby("intron_group", observed=True)["n_pairs"].transform("sum").gt(0)].copy()
    return (
        ggplot(plot_df, aes(x="intron_group", y="percentage", fill="splice_order"))
        + geom_col(width=0.65)
        + geom_text(
            aes(label="percentage"),
            position=position_stack(vjust=0.5),
            format_string="{:.1f}%",
            size=8,
        )
        + scale_fill_manual(values=ORDER_COLORS)
        + scale_y_continuous(limits=(0, 100), expand=(0, 0))
        + theme_bw()
        + plot_theme()
        + theme(axis_title_x=element_blank(), legend_title=element_blank(), legend_position="bottom")
        + labs(y="Splicing pairs (%)", title=f"{sample}: splicing pairs by intron number")
    )


def plot_group_mean(pair_df: pd.DataFrame, sample: str):
    """Mean stacked bars. The two orders share one SD, drawn on the boundary."""
    plot_df = pair_df.loc[pair_df["n_reps"].ge(2)].copy()
    boundary = plot_df.loc[plot_df["splice_order"].eq("Downstream intron spliced first")].copy()
    return (
        ggplot(plot_df, aes(x="intron_group", y="percentage", fill="splice_order"))
        + geom_col(width=0.65)
        + geom_errorbar(
            aes(x="intron_group", ymin="ymin", ymax="ymax"),
            data=boundary,
            width=0.18,
            size=0.6,
            color="black",
            inherit_aes=False,
        )
        + scale_fill_manual(values=ORDER_COLORS)
        + scale_y_continuous(limits=(0, 100), expand=(0, 0))
        + theme_bw()
        + plot_theme()
        + theme(axis_title_x=element_blank(), legend_title=element_blank(), legend_position="bottom")
        + labs(y="Splicing pairs (%)", title=f"{sample}: splicing pairs by intron number")
    )


def save_plot(plot, stem: Path, width: float, height: float) -> None:
    """Write png and pdf for one plot."""
    plot.save(Path(str(stem) + ".png"), width=width, height=height, dpi=600, verbose=False)
    plot.save(Path(str(stem) + ".pdf"), width=width, height=height, verbose=False)
    print(f"[INFO] wrote {stem.name}.png/.pdf", file=sys.stderr)


def main() -> None:
    """Count splicing pairs per replicate and average Warm and Cold."""
    ensure_result_dirs()
    out_dir = RESULT_DIRS["splicing_order"]
    out_dir.mkdir(parents=True, exist_ok=True)
    by_sample = {}
    for sample in SAMPLES:
        classify_path = classify_tsv(sample)
        raw = pd.read_csv(classify_path, sep="\t", low_memory=False)
        reads = raw.loc[raw["p5_feat"].isin(VALID_P5_FEATURES)].copy()
        pairs = count_splicing_pairs(reads)
        pairs["sample"] = sample
        distribution = intron_group_distribution(reads, sample)
        by_group = count_splicing_pairs_by_group(reads)
        by_group["sample"] = sample
        pairs.to_csv(out_dir / f"{sample}.splicing_pairs.tsv", sep="\t", index=False)
        distribution.to_csv(out_dir / f"{sample}.splicing_pairs.intron_group_distribution.tsv", sep="\t", index=False)
        by_group.to_csv(out_dir / f"{sample}.splicing_pairs.by_intron_group.tsv", sep="\t", index=False)
        save_plot(plot_overall(pairs, sample), out_dir / f"{sample}.splicing_pairs", 2.5, 4.5)
        save_plot(
            plot_distribution(distribution, sample),
            out_dir / f"{sample}.splicing_pairs.intron_group_distribution",
            4.5,
            4.5,
        )
        save_plot(
            plot_by_group(by_group, sample),
            out_dir / f"{sample}.splicing_pairs.by_intron_group",
            10,
            5.5,
        )
        by_sample[sample] = by_group
        print(f"[INFO] {sample}: reads={len(reads):,}", file=sys.stderr)
        print(pairs[["splice_order", "n_pairs", "percentage"]].to_string(index=False))

    for label, samples in (("Warm.mean", ("Warm1", "Warm2", "Warm3")), ("Cold.mean", ("Cold1", "Cold2", "Cold3"))):
        frames = [by_sample[sample] for sample in samples]
        replicates_wide(frames, samples).to_csv(
            out_dir / f"{label}.splicing_pairs.by_intron_group.tsv",
            sep="\t",
            index=False,
        )
        mean_df = summarize_group_mean(frames, label)
        save_plot(
            plot_group_mean(mean_df, label),
            out_dir / f"{label}.splicing_pairs.by_intron_group",
            10,
            5.5,
        )


if __name__ == "__main__":
    main()
