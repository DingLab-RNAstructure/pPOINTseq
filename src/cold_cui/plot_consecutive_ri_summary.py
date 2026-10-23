#!/usr/bin/env python3
"""Publication figure for consecutive intron-retention tests.

The claim is that Cold reduces consecutive intron retention relative to Warm.
Panel a shows what consecutive retention looks like on single molecules.
Panel b quantifies the library-level rate. Panel c tests same-molecule
coupling of adjacent retention. Panel d maps the gene-level GLM.

Inputs are the barcode TSV, consecutive_ri.by_sample.tsv,
consecutive_ri.coupling_2x2.by_sample.tsv, consecutive_ri.library_test.tsv,
and consecutive_ri.gene_glm.tsv.

Usage:
    python src/cold_cui/plot_consecutive_ri_summary.py
    python src/cold_cui/plot_consecutive_ri_summary.py --gene AT3G09405.1
    python src/cold_cui/plot_consecutive_ri_summary.py --gene AT5G57655.2
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

from plot_consecutive_ri_barcode import (  # noqa: E402
    RETAINED,
    SPLICED,
    STATE_COLORS,
    UNCOVERED,
    consecutive_spans,
    visible_state_matrix,
)
from project_config import RESULT_DIRS  # noqa: E402

# ============================================================
# CONSTANTS
# ============================================================

Warm_COLOR = "#0072B2"
Cold_COLOR = "#D55E00"
NS_COLOR = "#BDBDBD"
STATE_CHAR = {".": UNCOVERED, "S": SPLICED, "R": RETAINED}
VOLCANO_LABELS = ("AT3G09405.1", "AT5G57655.2", "AT1G58180.2", "AT5G38420.1")

# ============================================================
# CLI
# ============================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    as_dir = RESULT_DIRS["alternative_splicing"]
    gene_dir = RESULT_DIRS["gene_reads"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gene",
        default="AT3G09405.1",
        help="Transcript used for the barcode hero panel (default: AT3G09405.1)",
    )
    parser.add_argument(
        "--barcode-tsv",
        type=Path,
        default=None,
        help="Barcode source TSV; default is <gene>_Warm_Cold.consecutive_ri_barcode.tsv",
    )
    parser.add_argument(
        "--sample-table",
        type=Path,
        default=as_dir / "consecutive_ri.by_sample.tsv",
        help="Per-sample consecutive-RI rates",
    )
    parser.add_argument(
        "--library-test",
        type=Path,
        default=as_dir / "consecutive_ri.library_test.tsv",
        help="Library-level Welch/GLM table",
    )
    parser.add_argument(
        "--coupling-table",
        type=Path,
        default=as_dir / "consecutive_ri.coupling_2x2.by_sample.tsv",
        help="Per-sample intron-pair odds ratios",
    )
    parser.add_argument(
        "--gene-glm",
        type=Path,
        default=as_dir / "consecutive_ri.gene_glm.tsv",
        help="Gene-level GLM table",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=as_dir,
        help=f"Output directory (default: {as_dir})",
    )
    parser.add_argument(
        "--gene-dir",
        type=Path,
        default=gene_dir,
        help="Directory that holds barcode TSVs",
    )
    parser.add_argument(
        "--panel-a-genes",
        nargs="+",
        default=None,
        help="Write extra panel-a-only barcode figures for these transcripts",
    )
    parser.add_argument(
        "--write-volcano",
        action="store_true",
        help="Also write a standalone volcano plot (consecutive_ri.volcano.*)",
    )
    return parser.parse_args()


# ============================================================
# I/O HELPERS
# ============================================================


def required_columns(frame: pd.DataFrame, names: set[str], path: Path) -> None:
    """Raise if required columns are missing."""
    missing = sorted(names - set(frame.columns))
    if missing:
        raise SystemExit(f"[ERROR] {path} missing columns: {', '.join(missing)}")


def save_pub(fig: plt.Figure, stem: Path) -> None:
    """Write SVG/PDF/TIFF/PNG with editable text."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{stem}.svg", bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.tiff", dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")


def parse_state_string(value: object) -> np.ndarray:
    """Decode a comma-separated R/S/. barcode string."""
    tokens = str(value).split(",")
    return np.array([STATE_CHAR.get(token.strip(), UNCOVERED) for token in tokens], dtype=np.int8)


def welch_p(library_tests: pd.DataFrame, test_name: str) -> float:
    """Return one Welch p-value from the library test table."""
    hit = library_tests.loc[library_tests["test"].eq(test_name)]
    if hit.empty:
        raise SystemExit(f"[ERROR] missing test {test_name}")
    return float(hit["p"].iloc[0])


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


def add_panel_label(axis, label: str) -> None:
    """Place a bold panel letter outside the top-left corner."""
    axis.text(
        -0.12,
        1.08,
        label,
        transform=axis.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def barcode_rows(frame: pd.DataFrame) -> list[dict]:
    """Convert one group's barcode TSV into draw-ready rows."""
    rows = []
    ordered = frame.sort_values("plot_rank")
    for rec in ordered.itertuples(index=False):
        states = parse_state_string(rec.states)
        rows.append(
            {
                "states": states,
                "n_covered": int(rec.n_covered),
                "is_consecutive": bool(rec.is_consecutive),
            }
        )
    return rows


def draw_barcode_panel(axis, rows: list[dict], n_intron: int, group: str) -> None:
    """Draw one group's molecule-by-intron barcode."""
    matrix = visible_state_matrix(rows)
    cmap = ListedColormap([STATE_COLORS[UNCOVERED], STATE_COLORS[SPLICED], STATE_COLORS[RETAINED]])
    axis.pcolormesh(
        np.arange(n_intron + 1),
        np.arange(len(rows) + 1),
        matrix,
        cmap=cmap,
        vmin=UNCOVERED,
        vmax=RETAINED,
        edgecolors="white",
        linewidths=0.12,
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
                    linewidth=0.35,
                    zorder=3,
                )
            )
    axis.set_xlim(0, n_intron)
    axis.set_ylim(len(rows), 0)
    axis.set_yticks([])
    axis.set_ylabel(group, fontsize=7)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.tick_params(length=0)


def write_panel_a(gene_id: str, barcode: pd.DataFrame, output_dir: Path) -> Path:
    """Write an Warm/Cold barcode figure in the same layout as summary panel a."""
    nv_rows = barcode_rows(barcode.loc[barcode["group"].eq("Warm")])
    c5_rows = barcode_rows(barcode.loc[barcode["group"].eq("Cold")])
    if not nv_rows and not c5_rows:
        raise SystemExit(f"[ERROR] no barcode rows for {gene_id}")
    n_intron = int(len((nv_rows or c5_rows)[0]["states"]))
    fig = plt.figure(figsize=(3.6, 5.5))
    grid = fig.add_gridspec(2, 1, hspace=0.18)
    ax_nv = fig.add_subplot(grid[0])
    ax_c5 = fig.add_subplot(grid[1], sharex=ax_nv)
    if nv_rows:
        draw_barcode_panel(ax_nv, nv_rows, n_intron, "Warm")
    else:
        ax_nv.set_axis_off()
    if c5_rows:
        draw_barcode_panel(ax_c5, c5_rows, n_intron, "Cold")
    else:
        ax_c5.set_axis_off()
    ax_nv.set_xticks(np.arange(n_intron) + 0.5)
    ax_nv.tick_params(labelbottom=False)
    ax_c5.set_xticks(np.arange(n_intron) + 0.5)
    ax_c5.set_xticklabels([str(i) for i in range(1, n_intron + 1)], fontsize=5.5)
    ax_c5.set_xlabel("Intron (5′ → 3′)")
    ax_nv.set_title(
        f"{gene_id}   Warm {sum(r['is_consecutive'] for r in nv_rows)}/{len(nv_rows)}",
        loc="left",
        fontsize=7,
        pad=2,
    )
    ax_c5.set_title(
        f"Cold {sum(r['is_consecutive'] for r in c5_rows)}/{len(c5_rows)}",
        loc="left",
        fontsize=7,
        pad=2,
    )
    ax_nv.text(-0.18, 1.08, "a", transform=ax_nv.transAxes, fontsize=8, fontweight="bold", ha="left", va="bottom")
    handles = [
        Patch(facecolor=STATE_COLORS[RETAINED], edgecolor="none", label="Retained"),
        Patch(facecolor=STATE_COLORS[SPLICED], edgecolor="none", label="Spliced"),
        Patch(facecolor=STATE_COLORS[UNCOVERED], edgecolor="#BDBDBD", label="Not covered"),
        Patch(facecolor="none", edgecolor="#272727", label="Consecutive RI"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.01), fontsize=6.5)
    fig.subplots_adjust(left=0.16, right=0.96, top=0.93, bottom=0.12)
    stem = output_dir / f"consecutive_ri.panel_a.{gene_id}"
    save_pub(fig, stem)
    plt.close(fig)
    print(f"[INFO] wrote {stem}.pdf/.svg/.tiff/.png", file=sys.stderr)
    return stem


def plot_library(axis, sample_tbl: pd.DataFrame, library_tests: pd.DataFrame) -> None:
    """Plot replicate consecutive-RI rates for two denominators."""
    metrics = (
        ("pct_valid_consec", "of valid P5", "Welch_pct_valid_consec"),
        ("pct_ri_consec", "of RI reads", "Welch_pct_ri_consec"),
    )
    rng = np.random.default_rng(1)
    for x0, (column, _label, test_name) in enumerate(metrics):
        p_value = welch_p(library_tests, test_name)
        for group, color, dx in (("Warm", Warm_COLOR, -0.12), ("Cold", Cold_COLOR, 0.12)):
            values = sample_tbl.loc[sample_tbl["group"].eq(group), column].to_numpy(dtype=float)
            jitter = rng.uniform(-0.03, 0.03, size=values.size)
            axis.scatter(np.full(values.size, x0 + dx) + jitter, values, s=18, color=color, zorder=3)
            axis.hlines(float(np.mean(values)), x0 + dx - 0.08, x0 + dx + 0.08, color=color, lw=1.3)
        y_max = float(sample_tbl[column].max())
        axis.plot([x0 - 0.12, x0 + 0.12], [y_max + 1.6, y_max + 1.6], color="#272727", lw=0.6)
        axis.text(x0, y_max + 2.1, f"p = {p_value:.3g}", ha="center", va="bottom", fontsize=6)
    axis.set_xticks([0, 1], ["of valid P5", "of RI reads"])
    axis.set_ylabel("Consecutive RI (%)")
    axis.set_xlim(-0.55, 1.55)
    ymax = max(sample_tbl["pct_valid_consec"].max(), sample_tbl["pct_ri_consec"].max())
    axis.set_ylim(0, ymax + 6)
    add_panel_label(axis, "b")


def plot_coupling(axis, pair_tbl: pd.DataFrame) -> None:
    """Plot same-molecule retention odds ratio versus intron distance."""
    jitter = {"Warm": -0.12, "Cold": 0.12}
    colors = {"Warm": Warm_COLOR, "Cold": Cold_COLOR}
    for _, row in pair_tbl.iterrows():
        x = float(row["distance"]) + jitter[row["group"]]
        axis.plot([x, x], [row["or_ci95_lo"], row["or_ci95_hi"]], color=colors[row["group"]], lw=0.7)
        axis.scatter(x, row["odds_ratio"], s=16, color=colors[row["group"]], zorder=3)
    axis.axhline(1.0, color="#767676", lw=0.6, ls="--")
    axis.set_xticks([1, 2, 3], ["adjacent", "skip 1", "skip 2"])
    axis.set_ylabel("Odds ratio")
    axis.set_ylim(0, 42)
    axis.set_xlim(0.5, 3.5)
    add_panel_label(axis, "c")


def plot_volcano(axis, gene_tbl: pd.DataFrame, panel_label: str = "d", point_size: float = 7) -> None:
    """Plot gene-level Δ consecutive-RI rate versus BH-adjusted p."""
    y = -np.log10(np.clip(gene_tbl["padj"].to_numpy(dtype=float), 1e-40, 1.0))
    colors = np.full(len(gene_tbl), NS_COLOR)
    colors[gene_tbl["direction"].eq("higher_consecutive_in_Warm")] = Warm_COLOR
    colors[gene_tbl["direction"].eq("higher_consecutive_in_Cold")] = Cold_COLOR
    axis.scatter(gene_tbl["delta_rate"], y, s=point_size, c=colors, linewidths=0, rasterized=False, zorder=2)
    axis.axvline(-0.1, color="#767676", lw=0.5, ls=":")
    axis.axvline(0.1, color="#767676", lw=0.5, ls=":")
    axis.axhline(-np.log10(0.05), color="#767676", lw=0.5, ls=":")
    offsets = {
        "AT3G09405.1": (-4, 10),
        "AT5G57655.2": (7, -11),
        "AT1G58180.2": (8, 8),
        "AT5G38420.1": (8, -8),
    }
    labeled = gene_tbl.loc[gene_tbl["gene_id"].isin(VOLCANO_LABELS)]
    for rec in labeled.itertuples(index=False):
        yy = -np.log10(max(float(rec.padj), 1e-40))
        dx, dy = offsets.get(rec.gene_id, (6, 4))
        axis.annotate(
            rec.gene_id,
            xy=(rec.delta_rate, yy),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=5.5,
            color="#272727",
            arrowprops={"arrowstyle": "-", "lw": 0.4, "color": "#767676"},
        )
    n_nv = int((gene_tbl["direction"] == "higher_consecutive_in_Warm").sum())
    n_c5 = int((gene_tbl["direction"] == "higher_consecutive_in_Cold").sum())
    axis.text(0.98, 0.96, f"Warm higher {n_nv}\nCold higher {n_c5}", transform=axis.transAxes, ha="right", va="top", fontsize=6)
    axis.set_xlim(-0.30, 0.68)
    axis.set_xlabel("Δ consecutive RI (Warm − Cold)")
    axis.set_ylabel(r"−log$_{10}$ padj")
    if panel_label:
        add_panel_label(axis, panel_label)


def write_volcano(gene_tbl: pd.DataFrame, output_dir: Path) -> Path:
    """Write a standalone gene-level consecutive-RI volcano plot."""
    fig, axis = plt.subplots(figsize=(3.6, 3.3))
    plot_volcano(axis, gene_tbl, panel_label="", point_size=12)
    handles = [
        Patch(facecolor=Warm_COLOR, edgecolor="none", label="Higher in Warm"),
        Patch(facecolor=Cold_COLOR, edgecolor="none", label="Higher in Cold"),
        Patch(facecolor=NS_COLOR, edgecolor="none", label="Not called"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.55, -0.02), fontsize=6.5)
    fig.subplots_adjust(left=0.16, right=0.97, top=0.96, bottom=0.18)
    stem = output_dir / "consecutive_ri.volcano"
    save_pub(fig, stem)
    plt.close(fig)
    print(f"[INFO] wrote {stem}.pdf/.svg/.tiff/.png", file=sys.stderr)
    return stem


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    """Assemble the four-panel consecutive-RI summary figure."""
    args = parse_args()
    apply_publication_style()
    barcode_path = args.barcode_tsv or (args.gene_dir / f"{args.gene}_Warm_Cold.consecutive_ri_barcode.tsv")
    if not barcode_path.is_file():
        raise SystemExit(f"[ERROR] missing barcode table {barcode_path}")

    barcode = pd.read_csv(barcode_path, sep="\t")
    required_columns(barcode, {"group", "plot_rank", "states", "is_consecutive", "n_covered"}, barcode_path)
    sample_tbl = pd.read_csv(args.sample_table, sep="\t")
    required_columns(sample_tbl, {"sample", "group", "pct_valid_consec", "pct_ri_consec"}, args.sample_table)
    library_tests = pd.read_csv(args.library_test, sep="\t")
    pair_tbl = pd.read_csv(args.coupling_table, sep="\t")
    required_columns(pair_tbl, {"group", "distance", "odds_ratio", "or_ci95_lo", "or_ci95_hi"}, args.coupling_table)
    gene_tbl = pd.read_csv(args.gene_glm, sep="\t")
    required_columns(gene_tbl, {"gene_id", "delta_rate", "padj", "direction"}, args.gene_glm)

    nv_rows = barcode_rows(barcode.loc[barcode["group"].eq("Warm")])
    c5_rows = barcode_rows(barcode.loc[barcode["group"].eq("Cold")])
    n_intron = int(len(nv_rows[0]["states"])) if nv_rows else int(len(c5_rows[0]["states"]))

    fig = plt.figure(figsize=(7.2, 5.5))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.34)
    left = outer[0].subgridspec(2, 1, hspace=0.18)
    right = outer[1].subgridspec(3, 1, hspace=0.55)
    ax_nv = fig.add_subplot(left[0])
    ax_c5 = fig.add_subplot(left[1], sharex=ax_nv)
    ax_b = fig.add_subplot(right[0])
    ax_c = fig.add_subplot(right[1])
    ax_d = fig.add_subplot(right[2])

    draw_barcode_panel(ax_nv, nv_rows, n_intron, "Warm")
    draw_barcode_panel(ax_c5, c5_rows, n_intron, "Cold")
    n_nv_consec = sum(row["is_consecutive"] for row in nv_rows)
    n_c5_consec = sum(row["is_consecutive"] for row in c5_rows)
    ax_nv.set_xticks(np.arange(n_intron) + 0.5)
    ax_nv.tick_params(labelbottom=False)
    ax_c5.set_xticks(np.arange(n_intron) + 0.5)
    ax_c5.set_xticklabels([str(i) for i in range(1, n_intron + 1)], fontsize=5.5)
    ax_c5.set_xlabel("Intron (5′ → 3′)")
    ax_nv.set_title(f"{args.gene}   Warm {n_nv_consec}/{len(nv_rows)}", loc="left", fontsize=7, pad=2)
    ax_c5.set_title(f"Cold {n_c5_consec}/{len(c5_rows)}", loc="left", fontsize=7, pad=2)
    ax_nv.text(-0.18, 1.08, "a", transform=ax_nv.transAxes, fontsize=8, fontweight="bold", ha="left", va="bottom")

    plot_library(ax_b, sample_tbl, library_tests)
    plot_coupling(ax_c, pair_tbl)
    plot_volcano(ax_d, gene_tbl)

    state_handles = [
        Patch(facecolor=STATE_COLORS[RETAINED], edgecolor="none", label="Retained"),
        Patch(facecolor=STATE_COLORS[SPLICED], edgecolor="none", label="Spliced"),
        Patch(facecolor=STATE_COLORS[UNCOVERED], edgecolor="#BDBDBD", label="Not covered"),
        Patch(facecolor="none", edgecolor="#272727", label="Consecutive RI"),
    ]
    group_handles = [
        Patch(facecolor=Warm_COLOR, edgecolor="none", label="Warm"),
        Patch(facecolor=Cold_COLOR, edgecolor="none", label="Cold"),
    ]
    fig.legend(handles=state_handles, loc="lower left", ncol=4, bbox_to_anchor=(0.08, 0.01), fontsize=6.5)
    fig.legend(handles=group_handles, loc="lower right", ncol=2, bbox_to_anchor=(0.98, 0.01), fontsize=6.5)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.93, bottom=0.12)
    stem = args.output_dir / "consecutive_ri.summary_figure"
    save_pub(fig, stem)
    plt.close(fig)
    print(f"[INFO] wrote {stem}.pdf/.svg/.tiff/.png", file=sys.stderr)

    extra_genes = args.panel_a_genes or []
    for gene_id in extra_genes:
        extra_path = args.gene_dir / f"{gene_id}_Warm_Cold.consecutive_ri_barcode.tsv"
        if not extra_path.is_file():
            print(f"[WARN] skip {gene_id}: missing {extra_path}", file=sys.stderr)
            continue
        extra = pd.read_csv(extra_path, sep="\t")
        required_columns(extra, {"group", "plot_rank", "states", "is_consecutive", "n_covered"}, extra_path)
        write_panel_a(gene_id, extra, args.output_dir)
    if args.write_volcano:
        write_volcano(gene_tbl, args.output_dir)


if __name__ == "__main__":
    main()
