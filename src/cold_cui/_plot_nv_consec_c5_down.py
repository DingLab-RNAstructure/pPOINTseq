#!/usr/bin/env python3
"""Plot consecutive-RI gene-reads for Warm-high / Cold-down example genes."""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator

_src = next(p for p in Path(__file__).resolve().parents if (p / "project_config.py").is_file())
sys.path.insert(0, str(_src))
import bootstrap
bootstrap.setup(__file__)

from plot_classified_gene_reads import load_pas_hit, read_gene_model
from plot_gene_reads import (
    _draw_pas_3end_vline,
    _fetch_reads_for_locus,
    _plot_reads_on_ax,
    _sort_reads_for_strand,
)
from project_config import BAM_BY_SAMPLE, REF_DIR, RESULT_DIRS, SAMPLES, classify_tsv
from _find_nv_consec_c5_down import (
    EXCLUDED_CLASSES,
    RI_TOKEN,
    consecutive_pairs,
    parse_ri_introns,
)

SAMPLE_GROUPS = {
    "Warm": [s for s in SAMPLES if str(s).startswith("Warm")],
    "Cold": [s for s in SAMPLES if str(s).startswith("Cold")],
}
GROUP_BY_SAMPLE = {s: g for g, reps in SAMPLE_GROUPS.items() for s in reps}
PLOT_DPI = 600
RI_MAX_READS = 50
GENES = [
    "AT3G09405.1",
    "AT5G57655.2",
    "AT1G80560.1",
    "AT3G48360.1",
    "AT2G04690.5",
]


def _draw_ri_gene_model(ax, gene_model: pd.DataFrame, min_pos: int, max_pos: int, boundary_bp: int) -> None:
    gene_start = int(gene_model["start"].min())
    gene_end = int(gene_model["end"].max())
    strand = str(gene_model["strand"].iloc[0])
    exon_h = 0.1
    ax.plot([gene_start, gene_end], [0, 0], color="black", linewidth=0.5)
    for row in gene_model.itertuples():
        if "exon" in str(row.feature):
            ax.add_patch(
                Rectangle(
                    (row.start, -exon_h / 2),
                    row.end - row.start,
                    exon_h,
                    facecolor="#4C78A8",
                    edgecolor="none",
                    linewidth=0,
                )
            )
    arrow_len = min((max_pos - min_pos) * 0.03, boundary_bp * 0.8)
    anchor = gene_end if strand == "+" else gene_start
    target = anchor + arrow_len if strand == "+" else anchor - arrow_len
    origin = anchor - arrow_len * 0.1 if strand == "+" else anchor + arrow_len * 0.1
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


def collect_ids(genes: list[str]) -> dict[str, dict[str, set[str]]]:
    want = set(genes)
    ids = {g: {s: set() for s in SAMPLES} for g in genes}
    for sample in SAMPLES:
        path = classify_tsv(sample)
        print(f"[{sample}] collecting consecutive-RI ids", flush=True)
        for chunk in pd.read_csv(
            path,
            sep="\t",
            usecols=["read_id", "gene_id", "splicing_class", "alternative_splicing_details"],
            dtype=str,
            chunksize=250_000,
        ):
            chunk = chunk.loc[~chunk["splicing_class"].isin(EXCLUDED_CLASSES)]
            chunk = chunk.loc[chunk["gene_id"].isin(want)]
            if chunk.empty:
                continue
            details = chunk["alternative_splicing_details"].fillna("")
            ri_mask = details.str.contains(RI_TOKEN, regex=False)
            sub = chunk.loc[ri_mask]
            for gene_id, read_id, det in zip(sub["gene_id"], sub["read_id"], sub["alternative_splicing_details"]):
                if consecutive_pairs(parse_ri_introns(det)):
                    ids[str(gene_id)][sample].add(str(read_id))
    return ids


def fetch_group_reads(gene_id, group, id_map, chrom, start, end, strand):
    rows = []
    for sample in SAMPLE_GROUPS[group]:
        keep = id_map.get(sample, set())
        if not keep:
            continue
        fetched = _fetch_reads_for_locus(BAM_BY_SAMPLE[sample], chrom, start, end, strand)
        for rec in fetched:
            if rec["read_id"] in keep:
                rec = dict(rec)
                rec["sample"] = sample
                rows.append(rec)
    _sort_reads_for_strand(rows, strand)
    if len(rows) <= RI_MAX_READS:
        return rows
    idx = np.linspace(0, len(rows) - 1, RI_MAX_READS).round().astype(int)
    picked = [rows[int(i)] for i in idx]
    _sort_reads_for_strand(picked, strand)
    return picked


def plot_gene(gene_id: str, id_map: dict[str, set[str]]) -> None:
    bed_path = REF_DIR / "exon_intron_pos.chr.bed"
    pas_path = REF_DIR / "Ath.PAS.completed.bed"
    gene_model = read_gene_model(gene_id, bed_path)
    chrom = str(gene_model["chrom"].iloc[0])
    gene_start = int(gene_model["start"].min())
    gene_end = int(gene_model["end"].max())
    strand = str(gene_model["strand"].iloc[0])
    nv_reads = fetch_group_reads(gene_id, "Warm", id_map, chrom, gene_start, gene_end, strand)
    c5_reads = fetch_group_reads(gene_id, "Cold", id_map, chrom, gene_start, gene_end, strand)
    print(
        f"{gene_id}  strand={strand}  "
        f"Warm consec ids={sum(len(id_map[s]) for s in SAMPLE_GROUPS['Warm']):,} plotted={len(nv_reads)}  "
        f"Cold consec ids={sum(len(id_map[s]) for s in SAMPLE_GROUPS['Cold']):,} plotted={len(c5_reads)}",
        flush=True,
    )
    all_plotted = nv_reads + c5_reads
    min_pos = gene_start
    max_pos = gene_end
    if all_plotted:
        min_pos = min(min_pos, min(int(r["start"]) for r in all_plotted))
        max_pos = max(max_pos, max(int(r["end"]) for r in all_plotted))
    gene_segments = [
        (int(r.start), int(r.end), str(r.feature), "exon" if "exon" in str(r.feature) else "intron")
        for r in gene_model.itertuples()
        if ("exon" in str(r.feature) or "intron" in str(r.feature))
    ]
    boundary_bp = 200
    n_nv, n_c5 = max(len(nv_reads), 1), max(len(c5_reads), 1)
    fig_h = min(12.0, 2.2 + 0.09 * (len(nv_reads) + len(c5_reads)))
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10.0, fig_h),
        gridspec_kw={"height_ratios": [n_nv, n_c5, 8], "hspace": 0.04},
        sharex=True,
    )
    _plot_reads_on_ax(
        axes[0], nv_reads, gene_segments, min_pos, max_pos, boundary_bp,
        title=f"{gene_id}   Warm  consecutive intron-retention reads",
    )
    _plot_reads_on_ax(
        axes[1], c5_reads, gene_segments, min_pos, max_pos, boundary_bp,
        title="Cold  consecutive intron-retention reads",
    )
    _draw_ri_gene_model(axes[2], gene_model, min_pos, max_pos, boundary_bp)
    pas_hit = load_pas_hit(gene_id, pas_path)
    if pas_hit is not None:
        _draw_pas_3end_vline(list(axes), pas_hit[0], pas_hit[1], pas_hit[2])
    if strand == "-":
        for ax in axes:
            ax.invert_xaxis()
    fig.subplots_adjust(hspace=0.04)
    stem = RESULT_DIRS["gene_reads"] / f"{gene_id}_Warm_Cold.consecutive_ri_gene_reads"
    fig.savefig(f"{stem}.png", dpi=PLOT_DPI, bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {stem.name}.png/.pdf", flush=True)


def main() -> None:
    RESULT_DIRS["gene_reads"].mkdir(parents=True, exist_ok=True)
    ids = collect_ids(GENES)
    for gene_id in GENES:
        plot_gene(gene_id, ids[gene_id])


if __name__ == "__main__":
    main()
