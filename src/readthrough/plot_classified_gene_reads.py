#!/usr/bin/env python3
"""Plot classified reads for one gene from one BAM file.

The BAM must be indexed. The classification input is the headered TSV produced
by ``classify_bam_splicing_readthrough.py`` and must contain ``read_id``,
``gene_id``, and ``splicing_class``. Default coloring uses the three splicing
classes. Pass ``--color-by readthrough`` to color only readthrough versus
non-readthrough reads; that mode also requires ``is_readthrough``. All other
reads in the gene interval are drawn in gray. The inset pie chart summarizes
the selected color classes. In readthrough mode the pie reports readthrough
versus non-readthrough fractions, and two stacked bars report intron-1 and
intron-2 splice ratios in each group.

Outputs are a PNG, PDF, and a TSV summary containing plotted read counts and
percentages.

Usage:
    python src/readthrough/plot_classified_gene_reads.py --gene-id AT1G01040.2 \
      --bam data/bams/601_elongating.bam \
      --classify results/read_through/601.elongating.classify.tsv
    python src/readthrough/plot_classified_gene_reads.py --gene-id AT1G01040.2 \
      --bam data/bams/603_elongating.bam \
      --classify results/read_through/603.elongating.classify.tsv \
      --color-by readthrough --label rep2 --out-dir results/gene_reads
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator

_src = next(p for p in Path(__file__).resolve().parents if (p / "project_config.py").is_file())
sys.path.insert(0, str(_src))
import bootstrap
bootstrap.setup(__file__)

from plot_gene_reads import (
    _draw_pas_3end_vline,
    _fetch_reads_for_locus,
    _filter_bed_interval_rows,
    _lookup_pas_interval,
    _sort_reads_for_strand,
    parse_cigar,
)


# ============================================================
# CONSTANTS
# ============================================================

COLOR_BY_CHOICES = ("splicing_class", "readthrough")
CLASS_ORDER = ("fully_spliced", "fully_unspliced", "partial_spliced")
CLASS_LABELS = {
    "fully_spliced": "Fully spliced",
    "fully_unspliced": "Fully unspliced",
    "partial_spliced": "Partially/alternatively spliced",
}
CLASS_COLORS = {
    "fully_spliced": "#1F77B4",
    "fully_unspliced": "#FF7F0E",
    "partial_spliced": "#2CA02C",
}
READTHROUGH_ORDER = ("readthrough", "non_readthrough")
READTHROUGH_LABELS = {
    "readthrough": "Readthrough",
    "non_readthrough": "Non-readthrough",
}
READTHROUGH_COLORS = {
    "readthrough": "#D62728",
    "non_readthrough": "#1F77B4",
}
LAST_INTRON_ORDER = ("last_spliced", "last_retained")
LAST_INTRON_LABELS = {
    "last_spliced": "Last intron spliced",
    "last_retained": "Last intron unspliced",
}
LAST_INTRON_COLORS = {
    "last_spliced": "#1F77B4",
    "last_retained": "#FF7F0E",
}
INTRON_BAR_INDICES = (1, 2)
INTRON_SPLICED_COLOR = "#1F77B4"
INTRON_RETAINED_COLOR = "#FF7F0E"
READTHROUGH_BAR_ORDER = ("non_readthrough", "readthrough")
READTHROUGH_BAR_TICKS = {
    "non_readthrough": "Non-RT",
    "readthrough": "RT",
}
OTHER_READ_COLOR = "#BDBDBD"
NONCANONICAL_JUNCTION_COLOR = "#9467BD"
REQUIRED_CLASSIFY_COLUMNS = {
    "read_id",
    "gene_id",
    "splicing_class",
    "spliced_introns",
    "retained_introns",
    "unannotated_junctions",
}
OPTIONAL_CLASSIFY_COLUMNS = {
    "alternative_spliced_introns",
    "alternative_junctions",
    "is_readthrough",
    "n_introns",
    "covered_introns",
}


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gene-id", required=True, help="Transcript ID, e.g. AT1G01040.2.")
    parser.add_argument("--bam", type=Path, required=True, help="One indexed BAM file.")
    parser.add_argument(
        "--classify",
        type=Path,
        required=True,
        help="Classification TSV for the same BAM.",
    )
    parser.add_argument(
        "--bed",
        type=Path,
        default=Path("data/refs/exon_intron_pos.chr.bed"),
        help="Exon/intron BED.",
    )
    parser.add_argument(
        "--pas-bed",
        type=Path,
        default=Path("data/refs/Ath.PAS.completed.bed"),
        help="PAS BED used for the dashed 3-prime-end marker.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/gene_reads"),
        help="Output directory.",
    )
    parser.add_argument("--label", default=None, help="Sample label shown above the read track.")
    parser.add_argument(
        "--color-by",
        choices=COLOR_BY_CHOICES,
        default="splicing_class",
        help="Color reads by splicing class or by readthrough past this gene's PAS 3-prime end.",
    )
    parser.add_argument(
        "--min-readthrough-distance",
        type=int,
        default=50,
        help="Minimum bp past this gene's PAS 3-prime end to call readthrough.",
    )
    parser.add_argument("--boundary-bp", type=int, default=200)
    parser.add_argument("--width", type=float, default=10.0)
    parser.add_argument("--height", type=float, default=6.0)
    parser.add_argument("--height-ratio", type=float, default=0.06)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--read-exon-height", type=float, default=0.08)
    parser.add_argument("--read-intron-height", type=float, default=0.012)
    parser.add_argument("--read-row-gap", type=float, default=0.10)
    return parser.parse_args()


# ============================================================
# DATA LOADING
# ============================================================

def _is_true(value) -> bool:
    """Return True for boolean/string encodings of True."""
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "t", "yes"}


def color_scheme(color_by: str) -> tuple[tuple[str, ...], dict[str, str], dict[str, str], str, str]:
    """Return class order, colors, labels, pie title, and output stem suffix."""
    if color_by == "readthrough":
        return (
            READTHROUGH_ORDER,
            READTHROUGH_COLORS,
            READTHROUGH_LABELS,
            "Readthrough",
            "readthrough_gene_reads",
        )
    if color_by == "splicing_class":
        return (
            CLASS_ORDER,
            CLASS_COLORS,
            CLASS_LABELS,
            "Splicing classes",
            "classified_gene_reads",
        )
    raise ValueError(f"Unsupported color_by={color_by!r}")


def pas_3end_coord(
    strand: str,
    pas_hit: tuple[int, int, str] | None,
    gene_start: int,
    gene_end: int,
) -> int:
    """Return the PAS 3-prime coordinate used for the dashed line."""
    if pas_hit is not None:
        pas_start, pas_end, _pas_strand = pas_hit
        return pas_end if strand == "+" else pas_start
    return gene_end if strand == "+" else gene_start


def assign_readthrough_classes(
    reads: list[dict],
    strand: str,
    pas_3end: int,
    min_readthrough_distance: int,
) -> dict[str, str]:
    """Label each BAM read by whether its 3-prime end passes this gene's PAS."""
    labels: dict[str, str] = {}
    for read in reads:
        p3 = read["end"] if strand == "+" else read["start"]
        extension = (p3 - pas_3end) if strand == "+" else (pas_3end - p3)
        labels[read["read_id"]] = (
            "readthrough" if extension > min_readthrough_distance else "non_readthrough"
        )
    return labels


def _intron_index_set(value) -> set[int]:
    """Parse a comma-separated intron-index field into integers."""
    if pd.isna(value):
        return set()
    text = str(value).strip()
    if text in {"", "nan"}:
        return set()
    return {int(part) for part in text.split(",") if part.strip()}


def intron_status(intron_idx, spliced_introns, retained_introns, covered_introns=None) -> str:
    """Return spliced/retained/undecided for one intron index."""
    if intron_idx in _intron_index_set(spliced_introns):
        return "spliced"
    if intron_idx in _intron_index_set(retained_introns):
        return "retained"
    return "undecided"


def last_intron_status(n_introns, spliced_introns, retained_introns, covered_introns) -> str:
    """Return last-intron splicing status for one classified read."""
    try:
        n_intron = int(str(n_introns).strip())
    except (TypeError, ValueError):
        return "no_intron"
    if n_intron < 1:
        return "no_intron"
    status = intron_status(
        n_intron, spliced_introns, retained_introns, covered_introns
    )
    if status == "spliced":
        return "last_spliced"
    if status == "retained":
        return "last_retained"
    if n_intron in _intron_index_set(covered_introns):
        return "last_covered"
    return "last_not_covered"


def load_pas_hit(gene_id: str, pas_bed_path: Path) -> tuple[int, int, str] | None:
    """Return (start, end, strand) for this gene's PAS interval, if present."""
    if not pas_bed_path.is_file():
        return None
    pas = pd.read_csv(
        pas_bed_path,
        sep="\t",
        header=None,
        names=["chrom", "start", "end", "gene_id", "score", "strand"],
    )
    pas = _filter_bed_interval_rows(pas)
    pas["start"] = pas["start"].astype(int)
    pas["end"] = pas["end"].astype(int)
    return _lookup_pas_interval(gene_id, pas)


def read_gene_model(gene_id: str, bed_path: Path) -> pd.DataFrame:
    """Return BED segments for one transcript."""
    bed = pd.read_csv(bed_path, sep="\t", header=None)
    bed.columns = ["chrom", "start", "end", "feature", "score", "strand"]
    bed = _filter_bed_interval_rows(bed)
    bed["start"] = bed["start"].astype(int)
    bed["end"] = bed["end"].astype(int)
    bed["gene_id"] = bed["feature"].str.split("_").str[0]
    selected = bed[bed["gene_id"] == gene_id].copy()
    if selected.empty:
        raise ValueError(f"No annotation rows for gene_id={gene_id!r} in {bed_path}")
    return selected.sort_values(["start", "end"])


def read_gene_classes(gene_id: str, classify_path: Path) -> pd.DataFrame:
    """Return unique read-class assignments for one transcript."""
    classify = pd.read_csv(
        classify_path,
        sep="\t",
        usecols=lambda column: (
            column in REQUIRED_CLASSIFY_COLUMNS
            or column in OPTIONAL_CLASSIFY_COLUMNS
        ),
        dtype=str,
        low_memory=False,
    )
    missing = REQUIRED_CLASSIFY_COLUMNS.difference(classify.columns)
    if missing:
        raise ValueError(
            f"{classify_path} is missing columns: {', '.join(sorted(missing))}"
        )
    for column in OPTIONAL_CLASSIFY_COLUMNS:
        if column not in classify.columns:
            classify[column] = ""

    selected = classify[classify["gene_id"].eq(gene_id)].copy()
    # Compatibility with classification tables created before retained-only
    # evidence was made symmetric with spliced-only evidence.
    retained_only = (
        selected["splicing_class"].eq("partial_spliced")
        & selected["spliced_introns"].fillna("").str.strip().eq("")
        & selected["retained_introns"].fillna("").str.strip().ne("")
        & selected["unannotated_junctions"].fillna("").str.strip().eq("")
    )
    selected.loc[retained_only, "splicing_class"] = "fully_unspliced"
    if "is_readthrough" not in selected.columns:
        selected["is_readthrough"] = False
    selected["is_readthrough"] = selected["is_readthrough"].map(_is_true)
    conflicts = selected.groupby("read_id")["splicing_class"].nunique()
    conflict_ids = conflicts[conflicts > 1].index
    if len(conflict_ids):
        print(
            f"[WARN] Excluding {len(conflict_ids)} read IDs with conflicting classes.",
            file=sys.stderr,
        )
        selected = selected[~selected["read_id"].isin(conflict_ids)]
    return selected.drop_duplicates("read_id")


# ============================================================
# PLOTTING HELPERS
# ============================================================

def draw_classified_reads(
    ax: plt.Axes,
    reads: list[dict],
    class_by_read: dict[str, str],
    noncanonical_by_read: dict[str, set[tuple[int, int]]],
    min_pos: int,
    max_pos: int,
    boundary_bp: int,
    read_exon_height: float,
    read_intron_height: float,
    read_row_gap: float,
    class_colors: dict[str, str] | None = None,
    highlight_noncanonical: bool = True,
) -> None:
    """Draw one colored row per classified read."""
    palette = class_colors or CLASS_COLORS
    y_step = read_exon_height + read_row_gap
    y_position = 0.0
    patch_options = {
        "linewidth": 0.01,
        "edgecolor": "none",
        "antialiased": False,
    }

    for read in reads:
        read_class = class_by_read.get(read["read_id"], "other")
        color = palette.get(read_class, OTHER_READ_COLOR)
        cigar = parse_cigar(
            read["read_cigar"],
            start_pos=read["start"],
            end_pos=read["end"],
        )
        for start, end in cigar["blocks"]:
            ax.add_patch(
                Rectangle(
                    (start, y_position - read_exon_height / 2),
                    end - start,
                    read_exon_height,
                    facecolor=color,
                    **patch_options,
                )
            )
        noncanonical = (
            noncanonical_by_read.get(read["read_id"], set())
            if highlight_noncanonical
            else set()
        )
        for start, end in cigar["junctions"]:
            junction_color = (
                NONCANONICAL_JUNCTION_COLOR
                if (start, end) in noncanonical
                else color
            )
            junction_height = (
                max(read_intron_height * 3, read_exon_height * 0.35)
                if (start, end) in noncanonical
                else read_intron_height
            )
            ax.add_patch(
                Rectangle(
                    (start, y_position - junction_height / 2),
                    end - start,
                    junction_height,
                    facecolor=junction_color,
                    **patch_options,
                )
            )
        y_position -= y_step

    ax.set_xlim(min_pos - boundary_bp, max_pos + boundary_bp)
    ax.set_ylim(y_position, max(read_exon_height, 0.1))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_gene_model(
    ax: plt.Axes,
    gene_model: pd.DataFrame,
    min_pos: int,
    max_pos: int,
    boundary_bp: int,
) -> None:
    """Draw exon/intron structure and transcription direction."""
    gene_start = int(gene_model["start"].min())
    gene_end = int(gene_model["end"].max())
    strand = str(gene_model["strand"].iloc[0])
    exon_height = 0.1

    ax.plot([gene_start, gene_end], [0, 0], color="black", linewidth=0.5)
    for row in gene_model.itertuples():
        if "exon" in row.feature:
            ax.add_patch(
                Rectangle(
                    (row.start, -exon_height / 2),
                    row.end - row.start,
                    exon_height,
                    facecolor="#4C78A8",
                    edgecolor="none",
                )
            )

    arrow_length = min((max_pos - min_pos) * 0.03, boundary_bp * 0.8)
    anchor = gene_end if strand == "+" else gene_start
    target = anchor + arrow_length if strand == "+" else anchor - arrow_length
    origin = anchor - arrow_length * 0.1 if strand == "+" else anchor + arrow_length * 0.1
    ax.annotate("", xy=(target, 0), xytext=(origin, 0), arrowprops={"arrowstyle": "->"})

    ax.set_xlim(min_pos - boundary_bp, max_pos + boundary_bp)
    ax.set_ylim(-0.07, 0.07)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:.2f}"))
    ax.set_xlabel("Position (kb)")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    if strand == "-":
        ax.invert_xaxis()


def add_class_pie(
    ax: plt.Axes,
    summary: pd.DataFrame,
    class_colors: dict[str, str] | None = None,
    class_labels: dict[str, str] | None = None,
    pie_title: str = "Splicing classes",
    show_noncanonical: bool = True,
    show_other: bool = True,
    bounds: tuple[float, float, float, float] = (0.77, 0.68, 0.22, 0.30),
    legend_anchor: tuple[float, float] = (0.5, -0.30),
    other_label: str = "Other reads",
    title_fontsize: float = 9,
    pct_fontsize: float = 8,
    legend_fontsize: float = 7,
) -> None:
    """Add a class-percentage pie chart as an inset."""
    palette = class_colors or CLASS_COLORS
    labels = class_labels or CLASS_LABELS
    pie_ax = ax.inset_axes(list(bounds))
    nonzero = summary[summary["count"] > 0]
    if nonzero.empty:
        pie_ax.set_axis_off()
        pie_ax.set_title(f"{pie_title}\n(none)", fontsize=title_fontsize, pad=2)
        return
    pie_ax.pie(
        nonzero["count"],
        colors=[palette[value] for value in nonzero["read_class"]],
        autopct=lambda pct: f"{pct:.1f}%",
        startangle=90,
        counterclock=False,
        textprops={"fontsize": pct_fontsize},
        wedgeprops={"linewidth": 0.6, "edgecolor": "white"},
    )
    pie_ax.set_title(pie_title, fontsize=title_fontsize, pad=2)
    pie_ax.set_aspect("equal")

    handles = [
        Patch(
            facecolor=palette[row.read_class],
            label=f"{labels[row.read_class]}: {row.count}",
        )
        for row in summary.itertuples()
    ]
    other_count = int(summary.attrs.get("other_count", 0))
    if show_other:
        handles.append(
            Patch(
                facecolor=OTHER_READ_COLOR,
                label=f"{other_label}: {other_count} (not in pie)",
            )
        )
    noncanonical_count = int(summary.attrs.get("noncanonical_count", 0))
    if show_noncanonical and noncanonical_count:
        handles.append(
            Patch(
                facecolor=NONCANONICAL_JUNCTION_COLOR,
                label=f"Alternative/unannotated junction: {noncanonical_count} reads",
            )
        )
    pie_ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=legend_anchor,
        frameon=False,
        fontsize=legend_fontsize,
    )


def intron_splice_summaries(
    reads: list[dict],
    class_by_read: dict[str, str],
    classes: pd.DataFrame,
    intron_indices: tuple[int, ...] = INTRON_BAR_INDICES,
) -> dict[int, pd.DataFrame]:
    """Count decided spliced/retained events for intron 1 and 2 by RT class."""
    counts = {
        intron_idx: {
            rt_class: {"spliced": 0, "retained": 0}
            for rt_class in READTHROUGH_BAR_ORDER
        }
        for intron_idx in intron_indices
    }
    if classes.empty:
        return {
            intron_idx: _intron_summary_frame(counts[intron_idx])
            for intron_idx in intron_indices
        }
    lookup = classes.set_index("read_id", drop=False)
    for read in reads:
        rt_class = class_by_read.get(read["read_id"])
        if rt_class not in READTHROUGH_BAR_ORDER:
            continue
        if read["read_id"] not in lookup.index:
            continue
        row = lookup.loc[read["read_id"]]
        try:
            n_intron = int(str(row["n_introns"]).strip())
        except (TypeError, ValueError):
            n_intron = 0
        for intron_idx in intron_indices:
            if intron_idx > n_intron:
                continue
            status = intron_status(
                intron_idx,
                row["spliced_introns"],
                row["retained_introns"],
                row["covered_introns"],
            )
            if status in {"spliced", "retained"}:
                counts[intron_idx][rt_class][status] += 1
    return {
        intron_idx: _intron_summary_frame(counts[intron_idx])
        for intron_idx in intron_indices
    }


def _intron_summary_frame(class_counts: dict[str, dict[str, int]]) -> pd.DataFrame:
    """Build one intron summary table from spliced/retained counts."""
    rows = []
    for rt_class in READTHROUGH_BAR_ORDER:
        n_spliced = int(class_counts[rt_class]["spliced"])
        n_retained = int(class_counts[rt_class]["retained"])
        n_decided = n_spliced + n_retained
        rows.append(
            {
                "read_class": rt_class,
                "n_spliced": n_spliced,
                "n_retained": n_retained,
                "count": n_decided,
                "percentage": (100.0 * n_spliced / n_decided) if n_decided else 0.0,
            }
        )
    return pd.DataFrame(rows)


def add_intron_splice_stacked_bars(
    ax: plt.Axes,
    intron_summaries: dict[int, pd.DataFrame],
    bounds: tuple[float, float, float, float] = (0.72, 0.74, 0.26, 0.24),
) -> None:
    """Draw intron-1 and intron-2 100% stacked splice-ratio bars."""
    host = ax.inset_axes(list(bounds))
    host.set_axis_off()
    host.set_title("Intron splice ratio", fontsize=7, pad=2)
    intron_indices = list(intron_summaries)
    n_plots = max(len(intron_indices), 1)
    for plot_idx, intron_idx in enumerate(intron_indices):
        left = plot_idx / n_plots + 0.04 / n_plots
        width = 1.0 / n_plots - 0.10 / n_plots
        bar_ax = host.inset_axes([left, 0.18, width, 0.70])
        summary = intron_summaries[intron_idx]
        x_positions = list(range(len(READTHROUGH_BAR_ORDER)))
        spliced_fracs = []
        retained_fracs = []
        decided_ns = []
        for rt_class in READTHROUGH_BAR_ORDER:
            row = summary.loc[summary["read_class"].eq(rt_class)]
            n_spliced = int(row["n_spliced"].iloc[0]) if not row.empty else 0
            n_retained = int(row["n_retained"].iloc[0]) if not row.empty else 0
            n_decided = n_spliced + n_retained
            decided_ns.append(n_decided)
            if n_decided:
                spliced_fracs.append(n_spliced / n_decided)
                retained_fracs.append(n_retained / n_decided)
            else:
                spliced_fracs.append(0.0)
                retained_fracs.append(0.0)
        bar_ax.bar(
            x_positions,
            spliced_fracs,
            color=INTRON_SPLICED_COLOR,
            width=0.68,
            label="Spliced",
        )
        bar_ax.bar(
            x_positions,
            retained_fracs,
            bottom=spliced_fracs,
            color=INTRON_RETAINED_COLOR,
            width=0.68,
            label="Unspliced",
        )
        for x_pos, spliced_frac, n_decided in zip(x_positions, spliced_fracs, decided_ns):
            if n_decided <= 0:
                bar_ax.text(
                    x_pos, 0.5, "n=0", ha="center", va="center", fontsize=5, color="#666666"
                )
                continue
            label_y = spliced_frac / 2.0 if spliced_frac >= 0.18 else spliced_frac + 0.04
            label_color = "white" if spliced_frac >= 0.18 else "black"
            bar_ax.text(
                x_pos,
                label_y,
                f"{spliced_frac:.0%}",
                ha="center",
                va="center" if spliced_frac >= 0.18 else "bottom",
                fontsize=6,
                color=label_color,
                fontweight="bold",
            )
        tick_labels = [
            f"{READTHROUGH_BAR_TICKS[rt_class]}\nn={n_decided}"
            for rt_class, n_decided in zip(READTHROUGH_BAR_ORDER, decided_ns)
        ]
        bar_ax.set_xticks(x_positions)
        bar_ax.set_xticklabels(tick_labels, fontsize=5)
        for tick, rt_class in zip(bar_ax.get_xticklabels(), READTHROUGH_BAR_ORDER):
            tick.set_color(READTHROUGH_COLORS[rt_class])
        bar_ax.set_ylim(0, 1.05)
        bar_ax.set_yticks([0, 0.5, 1.0])
        if plot_idx == 0:
            bar_ax.set_yticklabels(["0", "0.5", "1"], fontsize=5)
            bar_ax.set_ylabel("Splice ratio", fontsize=6)
        else:
            bar_ax.set_yticklabels([])
        bar_ax.set_title(f"Intron {intron_idx}", fontsize=7, pad=1)
        bar_ax.spines["top"].set_visible(False)
        bar_ax.spines["right"].set_visible(False)
        bar_ax.tick_params(length=2, pad=1)
    handles = [
        Patch(facecolor=INTRON_SPLICED_COLOR, label="Spliced"),
        Patch(facecolor=INTRON_RETAINED_COLOR, label="Unspliced"),
    ]
    host.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=2,
        frameon=False,
        fontsize=5,
    )


# ============================================================
# ANALYSIS
# ============================================================

def plot_classified_gene_reads(
    gene_id: str,
    bam_path: str | Path,
    classify_path: str | Path,
    *,
    bed_path: str | Path = "data/refs/exon_intron_pos.chr.bed",
    pas_bed_path: str | Path = "data/refs/Ath.PAS.completed.bed",
    out_dir: str | Path = "results/gene_reads",
    sample_label: str | None = None,
    color_by: str = "splicing_class",
    min_readthrough_distance: int = 50,
    boundary_bp: int = 200,
    width: float = 10.0,
    height: float = 6.0,
    height_ratio: float = 0.06,
    dpi: int = 600,
    read_exon_height: float = 0.08,
    read_intron_height: float = 0.012,
    read_row_gap: float = 0.10,
) -> tuple[Path, Path, Path]:
    """Plot classified reads and return PNG, PDF, and summary paths."""
    if color_by not in COLOR_BY_CHOICES:
        raise ValueError(f"Unsupported color_by={color_by!r}")
    class_order, class_colors, class_labels, pie_title, stem_suffix = color_scheme(color_by)
    bam_path = Path(bam_path)
    classify_path = Path(classify_path)
    bed_path = Path(bed_path)
    pas_bed_path = Path(pas_bed_path)
    out_dir = Path(out_dir)

    if not bam_path.is_file():
        raise FileNotFoundError(f"BAM not found: {bam_path}")
    if not classify_path.is_file():
        raise FileNotFoundError(f"Classification TSV not found: {classify_path}")

    gene_model = read_gene_model(gene_id, bed_path)
    classes = read_gene_classes(gene_id, classify_path)
    chrom = str(gene_model["chrom"].iloc[0])
    gene_start = int(gene_model["start"].min())
    gene_end = int(gene_model["end"].max())
    strand = str(gene_model["strand"].iloc[0])
    pas_hit = load_pas_hit(gene_id, pas_bed_path)
    fetched = _fetch_reads_for_locus(bam_path, chrom, gene_start, gene_end, strand)
    reads = fetched
    _sort_reads_for_strand(reads, strand)
    if not reads:
        raise ValueError(f"No BAM reads found for {gene_id}")

    if color_by == "readthrough":
        class_by_read = assign_readthrough_classes(
            reads,
            strand,
            pas_3end_coord(strand, pas_hit, gene_start, gene_end),
            min_readthrough_distance,
        )
        highlight_noncanonical = False
    else:
        class_by_read = dict(zip(classes["read_id"], classes["splicing_class"]))
        highlight_noncanonical = True
    noncanonical_by_read = {
        row.read_id: {
            tuple(int(value) for value in junction.split("-", 1))
            for junction in (
                str(row.alternative_junctions).split(",")
                + str(row.unannotated_junctions).split(",")
            )
            if junction and junction.lower() != "nan" and "-" in junction
        }
        for row in classes.itertuples()
    }
    plotted_ids = {read["read_id"] for read in reads}
    missing_count = len(set(class_by_read).difference(plotted_ids))
    if missing_count:
        print(
            f"[WARN] {missing_count} classified reads were not fetched from the gene interval.",
            file=sys.stderr,
        )

    plotted_classes = pd.Series(
        [
            class_by_read.get(read["read_id"], "other")
            if class_by_read.get(read["read_id"], "other") in class_order
            else "other"
            for read in reads
        ],
        dtype="string",
    )
    counts = plotted_classes.value_counts()
    summary = pd.DataFrame({"read_class": class_order})
    summary["count"] = summary["read_class"].map(counts).fillna(0).astype(int)
    informative_total = int(summary["count"].sum())
    if informative_total == 0:
        raise ValueError(f"No {color_by} reads found for {gene_id}; pie chart is undefined")
    summary["percentage"] = 100 * summary["count"] / informative_total
    summary.attrs["other_count"] = int(counts.get("other", 0))
    summary.attrs["noncanonical_count"] = sum(
        bool(noncanonical_by_read.get(read["read_id"]))
        for read in reads
    )

    intron_summaries = None
    if color_by == "readthrough":
        intron_summaries = intron_splice_summaries(reads, class_by_read, classes)

    min_pos = min(gene_start, *(read["start"] for read in reads))
    max_pos = max(gene_end, *(read["end"] for read in reads))
    fig, (read_ax, gene_ax) = plt.subplots(
        nrows=2,
        figsize=(width, height),
        height_ratios=[1, height_ratio],
    )
    draw_classified_reads(
        read_ax,
        reads,
        class_by_read,
        noncanonical_by_read,
        min_pos,
        max_pos,
        boundary_bp,
        read_exon_height,
        read_intron_height,
        read_row_gap,
        class_colors=class_colors,
        highlight_noncanonical=highlight_noncanonical,
    )
    label = sample_label or bam_path.stem
    read_ax.set_title(f"{gene_id} | {label} | n={len(reads)}", loc="left", fontsize=11)
    if color_by == "readthrough":
        if intron_summaries is not None:
            add_intron_splice_stacked_bars(read_ax, intron_summaries)
        add_class_pie(
            read_ax,
            summary,
            class_colors=class_colors,
            class_labels=class_labels,
            pie_title=pie_title,
            show_noncanonical=False,
            show_other=False,
            bounds=(0.76, 0.32, 0.23, 0.24),
            legend_anchor=(0.5, -0.22),
            title_fontsize=8,
            pct_fontsize=7,
            legend_fontsize=6,
        )
    else:
        add_class_pie(
            read_ax,
            summary,
            class_colors=class_colors,
            class_labels=class_labels,
            pie_title=pie_title,
            show_noncanonical=highlight_noncanonical,
        )
    draw_gene_model(gene_ax, gene_model, min_pos, max_pos, boundary_bp)

    if pas_hit is not None:
        _draw_pas_3end_vline((read_ax, gene_ax), *pas_hit)

    if strand == "-":
        read_ax.invert_xaxis()

    fig.subplots_adjust(hspace=0.01)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{gene_id}_{label}_{stem_suffix}"
    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"
    summary_path = out_dir / f"{stem}.summary.tsv"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    export = summary.copy()
    if color_by == "splicing_class":
        export = export.rename(columns={"read_class": "splicing_class"})
    elif intron_summaries is not None:
        intron_exports = []
        for intron_idx, intron_df in intron_summaries.items():
            intron_export = intron_df.copy()
            intron_export["group"] = f"intron{intron_idx}_splice"
            intron_exports.append(intron_export)
        export = export.assign(group="readthrough", n_spliced=pd.NA, n_retained=pd.NA)
        export = pd.concat([export, *intron_exports], ignore_index=True)
        export = export[["group", "read_class", "count", "percentage", "n_spliced", "n_retained"]]
    export.to_csv(summary_path, sep="\t", index=False)

    print(f"[INFO] Plotted reads: {len(reads)}", file=sys.stderr)
    print(f"[INFO] Wrote {png_path}", file=sys.stderr)
    print(f"[INFO] Wrote {pdf_path}", file=sys.stderr)
    print(f"[INFO] Wrote {summary_path}", file=sys.stderr)
    return png_path, pdf_path, summary_path


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """Run the command-line workflow."""
    args = parse_args()
    plot_classified_gene_reads(
        args.gene_id,
        args.bam,
        args.classify,
        bed_path=args.bed,
        pas_bed_path=args.pas_bed,
        out_dir=args.out_dir,
        sample_label=args.label,
        color_by=args.color_by,
        min_readthrough_distance=args.min_readthrough_distance,
        boundary_bp=args.boundary_bp,
        width=args.width,
        height=args.height,
        height_ratio=args.height_ratio,
        dpi=args.dpi,
        read_exon_height=args.read_exon_height,
        read_intron_height=args.read_intron_height,
        read_row_gap=args.read_row_gap,
    )


if __name__ == "__main__":
    main()
