#!/usr/bin/env python3
"""Splicing of intron positions 1–9 and 10+ versus Pol II distance past the 3' splice site.

A read contributes to an intron when that intron is in spliced_introns or
retained_introns. A retained call overwrites a spliced call. Distance is how
far the read 3' end has passed that intron's 3' splice site: p3_coord minus the
intron end on the plus strand, and the intron start minus p3_coord on the minus
strand. Distances outside 0–2000 bp are dropped. Bins are 100 bp. A bin is kept
when it contains at least 20 intron calls. Reads must start in promoter or exon_1.

Each replicate is plotted on its own. Warm.mean and Cold.mean average the spliced
percentage in bins present in at least two replicates. The Cold mean uses a
y-axis from 30 to 102.

Usage:
    python src/overall_splicing/multiple_intron_splicing_dynamics.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
_src = next(p for p in Path(__file__).resolve().parents if (p / "project_config.py").is_file())
sys.path.insert(0, str(_src))
import bootstrap
_ROOT = bootstrap.setup(__file__)
os.chdir(_src)

import pandas as pd
from plotnine import (
    aes,
    element_blank,
    element_rect,
    element_text,
    geom_line,
    geom_point,
    ggplot,
    guide_legend,
    guides,
    labs,
    scale_color_manual,
    scale_size_identity,
    scale_x_continuous,
    scale_y_continuous,
    theme,
    theme_bw,
)

from project_config import REF_DIR, RESULT_DIRS, SAMPLES, classify_tsv, ensure_result_dirs

MAX_DISTANCE = 2000
BIN_WIDTH = 100
MIN_BIN_INTRONS = 20
MIN_REPS = 2
VALID_P5_FEATURES = {"promoter", "exon_1"}
POSITION_ORDER = [str(i) for i in range(1, 10)] + ["10+"]
POSITION_COLORS = {
    "1": "#241D4F",
    "2": "#2F3A8F",
    "3": "#2F6DB3",
    "4": "#29A2C6",
    "5": "#3CB89A",
    "6": "#73B66B",
    "7": "#B7C75B",
    "8": "#E0A106",
    "9": "#FF6D31",
    "10+": "#C0392B",
}


def intron_set(value) -> set[int]:
    """Parse a comma-separated intron-index field."""
    if pd.isna(value) or str(value).strip() == "":
        return set()
    return {int(item) for item in str(value).split(",")}


def intron_status(spliced_value, retained_value) -> dict[int, str]:
    """Return spliced or retained. A retained call overwrites a spliced call."""
    states = {idx: "spliced" for idx in intron_set(spliced_value)}
    for idx in intron_set(retained_value):
        states[idx] = "retained"
    return states


def load_intron_coordinates() -> dict[tuple[str, int], tuple[int, int, str]]:
    """Map (gene_id, intron index) to start, end, and strand."""
    bed = pd.read_csv(
        REF_DIR / "exon_intron_pos.chr.bed",
        sep="\t",
        header=None,
        names=["chrom", "start", "end", "name", "score", "strand"],
    )
    parsed = bed["name"].str.extract(r"^(?P<gene_id>.+)_intron(?P<intron_idx>\d+)$")
    intron_df = bed.loc[parsed["gene_id"].notna()].copy()
    intron_df[["gene_id", "intron_idx"]] = parsed.dropna().values
    intron_df["intron_idx"] = intron_df["intron_idx"].astype(int)
    return {
        (row.gene_id, int(row.intron_idx)): (int(row.start), int(row.end), row.strand)
        for row in intron_df.itertuples(index=False)
    }


def distance_table(
    df: pd.DataFrame,
    lookup: dict[tuple[str, int], tuple[int, int, str]],
    max_distance: int = MAX_DISTANCE,
    bin_width: int = BIN_WIDTH,
    min_bin_introns: int = MIN_BIN_INTRONS,
) -> pd.DataFrame:
    """Spliced percentage versus distance for each intron position."""
    rows = []
    for row in df.itertuples(index=False):
        states = intron_status(row.spliced_introns, row.retained_introns)
        for intron_idx, status in states.items():
            coords = lookup.get((row.gene_id, intron_idx))
            if coords is None:
                continue
            start, end, strand = coords
            distance = row.p3_coord - end if strand == "+" else start - row.p3_coord
            if not 0 <= distance <= max_distance:
                continue
            rows.append(
                {
                    "intron_position": str(intron_idx) if intron_idx < 10 else "10+",
                    "distance": distance,
                    "is_spliced": status == "spliced",
                }
            )
    long_df = pd.DataFrame(rows)
    if long_df.empty:
        return long_df
    long_df["distance_mid"] = (long_df["distance"] // bin_width) * bin_width + bin_width / 2
    result = (
        long_df.groupby(["intron_position", "distance_mid"], observed=True)
        .agg(
            percentage_spliced=("is_spliced", lambda values: 100 * values.mean()),
            n_introns=("is_spliced", "size"),
        )
        .reset_index()
    )
    result = result.loc[result["n_introns"] >= min_bin_introns].copy()
    result["intron_position"] = pd.Categorical(
        result["intron_position"], categories=POSITION_ORDER, ordered=True
    )
    return result.sort_values(["intron_position", "distance_mid"])


def mean_table(frames: dict[str, pd.DataFrame], reps: list[str], label: str) -> pd.DataFrame:
    """Average spliced percentages for bins present in at least two replicates."""
    parts = []
    for sample in reps:
        frame = frames.get(sample)
        if frame is None or frame.empty:
            continue
        part = frame[["intron_position", "distance_mid", "percentage_spliced", "n_introns"]].copy()
        part["sample"] = sample
        parts.append(part)
    columns = [
        "intron_position",
        "distance_mid",
        "percentage_spliced",
        "n_introns",
        "n_reps",
        "sample",
    ]
    if not parts:
        return pd.DataFrame(columns=columns)
    stacked = pd.concat(parts, ignore_index=True)
    grouped = stacked.groupby(["intron_position", "distance_mid"], observed=True)
    n_reps = grouped["sample"].nunique()
    out = grouped.agg(
        percentage_spliced=("percentage_spliced", "mean"),
        n_introns=("n_introns", "sum"),
    ).reset_index()
    out["n_reps"] = out.set_index(["intron_position", "distance_mid"]).index.map(n_reps).astype(int)
    out = out.loc[out["n_reps"] >= MIN_REPS].copy()
    out["sample"] = label
    out["intron_position"] = pd.Categorical(
        out["intron_position"].astype(str), categories=POSITION_ORDER, ordered=True
    )
    return out.sort_values(["intron_position", "distance_mid"])


def plot_dynamics(frame: pd.DataFrame, sample: str, y_limits: tuple[float, float] | None = None):
    """One curve per intron position."""
    plot_df = frame.copy()
    plot_df["intron_position"] = pd.Categorical(
        plot_df["intron_position"].astype(str), categories=POSITION_ORDER, ordered=True
    )
    plot_df["line_size"] = [1.6 if str(pos) == "1" else 0.95 for pos in plot_df["intron_position"]]
    plot_df["point_size"] = [2.6 if str(pos) == "1" else 2.0 for pos in plot_df["intron_position"]]
    if y_limits is None:
        y_min = max(0, int(plot_df["percentage_spliced"].min() // 10 * 10) - 10)
        y_max = 102
        y_breaks = list(range(y_min, 101, 10))
    else:
        y_min, y_max = y_limits
        y_breaks = list(range(int(y_min), int(y_max) + 1, 10))
    axis_theme = theme(
        panel_grid=element_blank(),
        axis_text=element_text(color="black", size=12),
        axis_title=element_text(color="black", size=12),
    )
    return (
        ggplot(
            plot_df,
            aes(
                x="distance_mid",
                y="percentage_spliced",
                color="intron_position",
                group="intron_position",
            ),
        )
        + geom_line(aes(size="line_size"))
        + geom_point(aes(size="point_size"), alpha=0.95)
        + scale_size_identity(guide=None)
        + scale_color_manual(values=POSITION_COLORS)
        + scale_x_continuous(
            limits=(0, MAX_DISTANCE),
            breaks=range(0, MAX_DISTANCE + 1, 500),
            expand=(0.02, 0),
        )
        + scale_y_continuous(limits=(y_min, y_max), breaks=y_breaks, expand=(0, 0))
        + theme_bw()
        + axis_theme
        + theme(
            legend_title=element_text(size=10),
            legend_text=element_text(size=9),
            legend_key_height=8,
            plot_title=element_text(size=12, face="bold", ha="left"),
            panel_border=element_rect(color="black", size=0.6),
        )
        + guides(color=guide_legend(override_aes={"size": 2.4, "alpha": 1}))
        + labs(
            x="Distance from 3' SS (nt)",
            y="Spliced introns (%)",
            color="Intron position",
            title=sample,
        )
    )


def save_plot(plot, stem: Path) -> None:
    """Write png and pdf."""
    plot.save(Path(str(stem) + ".png"), width=7, height=5, dpi=600, verbose=False)
    plot.save(Path(str(stem) + ".pdf"), width=7, height=5, verbose=False)
    print(f"[INFO] wrote {stem.name}.png/.pdf", file=sys.stderr)


def main() -> None:
    """Build per-replicate curves and the Warm and Cold means."""
    ensure_result_dirs()
    out_dir = RESULT_DIRS["splicing_order"]
    out_dir.mkdir(parents=True, exist_ok=True)
    lookup = load_intron_coordinates()
    by_sample: dict[str, pd.DataFrame] = {}
    samples = [sample for sample in SAMPLES if str(sample).startswith(("Warm", "Cold"))]
    for sample in samples:
        raw = pd.read_csv(classify_tsv(sample), sep="\t", low_memory=False)
        reads = raw.loc[raw["p5_feat"].isin(VALID_P5_FEATURES)].copy()
        table = distance_table(reads, lookup)
        table.to_csv(out_dir / f"{sample}.multiple_intron_splicing_dynamics.tsv", sep="\t", index=False)
        save_plot(plot_dynamics(table, sample), out_dir / f"{sample}.multiple_intron_splicing_dynamics")
        by_sample[sample] = table
        print(f"[INFO] {sample}: bins={len(table):,} reads={len(reads):,}", file=sys.stderr)

    groups = {
        "Warm": [sample for sample in samples if str(sample).startswith("Warm")],
        "Cold": [sample for sample in samples if str(sample).startswith("Cold")],
    }
    for group, reps in groups.items():
        if len(reps) < MIN_REPS:
            print(f"[INFO] skip {group}.mean: {len(reps)} replicate(s)", file=sys.stderr)
            continue
        label = f"{group}.mean"
        mean_df = mean_table(by_sample, reps, label)
        mean_df.to_csv(out_dir / f"{label}.multiple_intron_splicing_dynamics.tsv", sep="\t", index=False)
        y_limits = (30, 102) if label == "Cold.mean" else None
        save_plot(plot_dynamics(mean_df, label, y_limits=y_limits), out_dir / f"{label}.multiple_intron_splicing_dynamics")
        print(f"[INFO] {label}: bins={len(mean_df):,}", file=sys.stderr)


if __name__ == "__main__":
    main()
