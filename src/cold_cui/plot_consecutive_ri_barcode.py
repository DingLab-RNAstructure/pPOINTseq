#!/usr/bin/env python3
"""Plot a read-by-intron barcode that makes consecutive intron retention visible.

Genome-track plots color a whole molecule as spliced or unspliced. That hides
whether two adjacent annotated introns are both retained on the same read.
This figure uses one row per molecule and one column per intron:

    orange  = retained
    blue    = spliced
    gray    = traversed, but neither spliced nor retained
    blank   = intron downstream of the read 3' end (not drawn)

A black box is drawn around every run of two or more adjacent retained
introns. Warm is stacked above Cold. Molecules are ordered by 3' end
(``p3_coord``) in transcription direction: 5' at the top, 3' at the bottom.

Input is the elongating classify TSV (covered_introns / spliced_introns /
retained_introns / p3_coord / strand). Reads with splicing_class
invalid_p5_start are dropped.

Usage:
    python src/cold_cui/plot_consecutive_ri_barcode.py
    python src/cold_cui/plot_consecutive_ri_barcode.py --genes AT5G57655.2 AT3G09405.1
    python src/cold_cui/plot_consecutive_ri_barcode.py --max-reads 80
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle

_src = next(p for p in Path(__file__).resolve().parents if (p / "project_config.py").is_file())
sys.path.insert(0, str(_src))
import bootstrap
bootstrap.setup(__file__)

from project_config import REF_DIR, RESULT_DIRS, SAMPLES, classify_tsv
from _find_warm_consec_cold_down import EXCLUDED_CLASSES, consecutive_pairs, longest_run
from plot_classified_gene_reads import read_gene_model

# ============================================================
# CONSTANTS
# ============================================================

UNCOVERED = 0
SPLICED = 1
RETAINED = 2

STATE_COLORS = {
    UNCOVERED: "#F2F2F2",
    SPLICED: "#0072B2",
    RETAINED: "#D55E00",
}
STATE_LABELS = {
    UNCOVERED: "Not covered",
    SPLICED: "Spliced",
    RETAINED: "Retained",
}
SAMPLE_GROUPS = {
    "Warm": [s for s in SAMPLES if str(s).startswith("Warm")],
    "Cold": [s for s in SAMPLES if str(s).startswith("Cold")],
}
DEFAULT_GENES = ("AT5G57655.2", "AT3G09405.1", "AT1G80560.1")
CLASSIFY_COLUMNS = (
    "read_id",
    "gene_id",
    "strand",
    "p3_coord",
    "splicing_class",
    "covered_introns",
    "spliced_introns",
    "retained_introns",
)

# ============================================================
# CLI
# ============================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    default_dir = RESULT_DIRS["gene_reads"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--genes",
        nargs="+",
        default=list(DEFAULT_GENES),
        help="Transcript IDs to plot (default: AT5G57655.2 AT3G09405.1 AT1G80560.1)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_dir,
        help=f"Output directory (default: {default_dir})",
    )
    parser.add_argument(
        "--max-reads",
        type=int,
        default=80,
        help="Maximum molecules plotted per group after ranking (default: 80)",
    )
    parser.add_argument(
        "--row-height",
        type=float,
        default=0.016,
        help="Inches per molecule row (default: 0.016)",
    )
    parser.add_argument(
        "--bed",
        type=Path,
        default=REF_DIR / "exon_intron_pos.chr.bed",
        help="Exon/intron BED used to count annotated introns",
    )
    return parser.parse_args()


# ============================================================
# I/O HELPERS
# ============================================================


def parse_intron_list(value: object) -> list[int]:
    """Parse a comma-separated intron-index field."""
    if pd.isna(value):
        return []
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none"}:
        return []
    out = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(int(token))
    return out


def n_intron_for_gene(gene_id: str, bed_path: Path) -> int:
    """Return the annotated intron count for one transcript."""
    model = read_gene_model(gene_id, bed_path)
    n_intron = int(model["feature"].str.contains("_intron", regex=False).sum())
    if n_intron < 1:
        raise SystemExit(f"[ERROR] no intron rows for {gene_id} in {bed_path}")
    return n_intron


def save_pub(fig: plt.Figure, stem: Path) -> None:
    """Write SVG/PDF/TIFF/PNG with editable text."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{stem}.svg", bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.tiff", dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")


# ============================================================
# DATA PROCESSING
# ============================================================


def load_gene_reads(gene_ids: list[str]) -> pd.DataFrame:
    """Load valid classified reads for the requested transcripts."""
    want = set(gene_ids)
    frames = []
    for sample in SAMPLES:
        path = classify_tsv(sample)
        print(f"[INFO] {sample} scanning {path.name}", file=sys.stderr)
        chunks = []
        for chunk in pd.read_csv(
            path,
            sep="\t",
            usecols=list(CLASSIFY_COLUMNS),
            dtype=str,
            chunksize=250_000,
        ):
            chunk = chunk.loc[chunk["gene_id"].isin(want)]
            chunk = chunk.loc[~chunk["splicing_class"].isin(EXCLUDED_CLASSES)]
            if not chunk.empty:
                chunks.append(chunk)
        if not chunks:
            continue
        frame = pd.concat(chunks, ignore_index=True)
        frame["sample"] = sample
        frame["group"] = "Cold" if str(sample).startswith("Cold") else "Warm"
        frames.append(frame)
    if not frames:
        raise SystemExit(f"[ERROR] no classified reads for {', '.join(gene_ids)}")
    return pd.concat(frames, ignore_index=True)


def parse_p3_coord(value: object) -> float:
    """Return a numeric 3' coordinate, or NaN if missing."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def encode_read(row: pd.Series, n_intron: int) -> dict:
    """Return one molecule's intron-state vector and consecutive-RI scores."""
    covered = set(parse_intron_list(row.covered_introns))
    spliced = set(parse_intron_list(row.spliced_introns))
    retained = set(parse_intron_list(row.retained_introns))
    states = np.full(n_intron, UNCOVERED, dtype=np.int8)
    for idx in range(1, n_intron + 1):
        if idx in retained:
            states[idx - 1] = RETAINED
        elif idx in spliced:
            states[idx - 1] = SPLICED
    pairs = consecutive_pairs(sorted(retained))
    run = longest_run(sorted(retained))
    first_run = pairs[0][0] if pairs else (min(retained) if retained else n_intron + 1)
    return {
        "read_id": str(row.read_id),
        "sample": str(row.sample),
        "group": str(row.group),
        "strand": str(row.strand),
        "p3_coord": parse_p3_coord(row.p3_coord),
        "states": states,
        "n_retained": int(len(retained)),
        "n_covered": int(len(covered)),
        "n_consec_pairs": int(len(pairs)),
        "max_run": int(run),
        "first_run": int(first_run),
        "is_consecutive": bool(pairs),
    }


def rank_and_cap(rows: list[dict], max_reads: int) -> list[dict]:
    """Sort informative reads by 3' end in transcription order, then cap."""
    informative = [row for row in rows if row["n_covered"] >= 1]
    if not informative:
        return []
    plus = informative[0]["strand"] != "-"
    informative.sort(
        key=lambda row: (
            np.isnan(row["p3_coord"]),
            row["p3_coord"] if plus else -row["p3_coord"],
        )
    )
    if len(informative) <= max_reads:
        return informative
    idx = np.linspace(0, len(informative) - 1, max_reads).round().astype(int)
    return [informative[int(i)] for i in idx]


# ============================================================
# PLOTTING
# ============================================================


def apply_publication_style() -> None:
    """Apply compact Nature-style rcParams."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def visible_state_matrix(rows: list[dict], reached_key: str = "n_covered") -> np.ma.MaskedArray:
    """Hide introns downstream of each read's 3' end."""
    matrix = np.vstack([row["states"] for row in rows])
    mask = np.zeros(matrix.shape, dtype=bool)
    for i, row in enumerate(rows):
        reached = int(row[reached_key])
        if reached < matrix.shape[1]:
            mask[i, reached:] = True
    return np.ma.array(matrix, mask=mask)


def consecutive_spans(states: np.ndarray) -> list[tuple[int, int]]:
    """Return 0-based (start, end_inclusive) spans of adjacent retained introns."""
    spans = []
    start = None
    for i, value in enumerate(states):
        if value == RETAINED:
            if start is None:
                start = i
        elif start is not None:
            if i - 1 > start:
                spans.append((start, i - 1))
            start = None
    if start is not None and len(states) - 1 > start:
        spans.append((start, len(states) - 1))
    return spans


def draw_barcode(axis, rows: list[dict], n_intron: int, title: str) -> None:
    """Draw one group's read-by-intron barcode and outline consecutive runs."""
    if not rows:
        axis.set_axis_off()
        axis.set_title(f"{title}\nno reads", loc="left", fontsize=8)
        return
    matrix = visible_state_matrix(rows)
    cmap = ListedColormap([STATE_COLORS[UNCOVERED], STATE_COLORS[SPLICED], STATE_COLORS[RETAINED]])
    x_edges = np.arange(n_intron + 1)
    y_edges = np.arange(len(rows) + 1)
    axis.pcolormesh(
        x_edges,
        y_edges,
        matrix,
        cmap=cmap,
        vmin=UNCOVERED,
        vmax=RETAINED,
        edgecolors="white",
        linewidths=0.15,
        shading="flat",
    )
    for y, row in enumerate(rows):
        for x0, x1 in consecutive_spans(row["states"]):
            # consecutive_spans returns inclusive (start, end); left/right
            # edges flush with the intron cell edges.
            axis.add_patch(
                Rectangle(
                    (x0, y + 0.12),
                    (x1 - x0) + 1.0,
                    0.76,
                    fill=False,
                    edgecolor="#272727",
                    linewidth=0.4,
                    zorder=3,
                )
            )
    n_consec = sum(row["is_consecutive"] for row in rows)
    axis.set_title(f"{title}   {n_consec}/{len(rows)} consecutive-RI reads", loc="left", fontsize=7, pad=2)
    axis.set_xlim(0, n_intron)
    axis.set_ylim(len(rows), 0)
    axis.set_xticks(np.arange(n_intron) + 0.5)
    axis.set_xticklabels([str(i) for i in range(1, n_intron + 1)], fontsize=6)
    axis.set_yticks([])
    axis.set_ylabel("Molecules (by 3′ end)", fontsize=7)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.tick_params(length=0)


def plot_gene_barcode(
    gene_id: str,
    encoded: list[dict],
    n_intron: int,
    max_reads: int,
    out_dir: Path,
    row_height: float,
) -> pd.DataFrame:
    """Write Warm/Cold barcodes and a per-read source table."""
    by_group = {name: [row for row in encoded if row["group"] == name] for name in ("Warm", "Cold")}
    plotted = {name: rank_and_cap(rows, max_reads) for name, rows in by_group.items()}
    n_warm = max(len(plotted["Warm"]), 1)
    n_cold = max(len(plotted["Cold"]), 1)
    width = min(7.2, max(3.6, 0.28 * n_intron + 1.6))
    height = min(4.2, max(2.0, row_height * (n_warm + n_cold) + 1.05))
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(width, height),
        gridspec_kw={"height_ratios": [n_warm, n_cold], "hspace": 0.12},
        sharex=True,
    )
    draw_barcode(axes[0], plotted["Warm"], n_intron, f"{gene_id}   Warm")
    draw_barcode(axes[1], plotted["Cold"], n_intron, "Cold")
    axes[1].set_xlabel("Intron index (5′ → 3′)", fontsize=7)
    handles = [
        Patch(facecolor=STATE_COLORS[RETAINED], edgecolor="none", label="Retained"),
        Patch(facecolor=STATE_COLORS[SPLICED], edgecolor="none", label="Spliced"),
        Patch(facecolor=STATE_COLORS[UNCOVERED], edgecolor="#BDBDBD", label="Not covered"),
        Patch(facecolor="none", edgecolor="#272727", label="Consecutive RI"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.01), fontsize=7)
    stem = out_dir / f"{gene_id}_Warm_Cold.consecutive_ri_barcode"
    fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.16, hspace=0.18)
    save_pub(fig, stem)
    plt.close(fig)
    print(f"[INFO] wrote {stem}.pdf/.svg/.tiff/.png", file=sys.stderr)

    records = []
    for group, rows in plotted.items():
        for rank, row in enumerate(rows, start=1):
            records.append(
                {
                    "gene_id": gene_id,
                    "group": group,
                    "plot_rank": rank,
                    "read_id": row["read_id"],
                    "sample": row["sample"],
                    "p3_coord": row["p3_coord"],
                    "n_covered": row["n_covered"],
                    "n_retained": row["n_retained"],
                    "n_consec_pairs": row["n_consec_pairs"],
                    "max_run": row["max_run"],
                    "is_consecutive": row["is_consecutive"],
                    "states": ",".join("R" if s == RETAINED else "S" if s == SPLICED else "." for s in row["states"]),
                }
            )
    table = pd.DataFrame(records)
    table_path = out_dir / f"{gene_id}_Warm_Cold.consecutive_ri_barcode.tsv"
    table.to_csv(table_path, sep="\t", index=False)
    print(f"[INFO] wrote {table_path}", file=sys.stderr)
    return table


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    """Build consecutive-RI barcodes for the requested transcripts."""
    args = parse_args()
    apply_publication_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_all = load_gene_reads(args.genes)
    for gene_id in args.genes:
        n_intron = n_intron_for_gene(gene_id, args.bed)
        raw = raw_all.loc[raw_all["gene_id"].eq(gene_id)]
        if raw.empty:
            print(f"[WARN] no classified reads for {gene_id}", file=sys.stderr)
            continue
        encoded = [encode_read(row, n_intron) for row in raw.itertuples(index=False)]
        n_consec_warm = sum(row["is_consecutive"] and row["group"] == "Warm" for row in encoded)
        n_consec_cold = sum(row["is_consecutive"] and row["group"] == "Cold" for row in encoded)
        print(
            f"[INFO] {gene_id} introns={n_intron} "
            f"Warm={sum(r['group']=='Warm' for r in encoded)} (consec {n_consec_warm}) "
            f"Cold={sum(r['group']=='Cold' for r in encoded)} (consec {n_consec_cold})",
            file=sys.stderr,
        )
        plot_gene_barcode(
            gene_id,
            encoded,
            n_intron,
            args.max_reads,
            args.output_dir,
            args.row_height,
        )


if __name__ == "__main__":
    main()
