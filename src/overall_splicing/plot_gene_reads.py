#!/usr/bin/env python3
import argparse
import re
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pysam
from matplotlib import font_manager
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator

available_fonts = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams["font.family"] = "Arial" if "Arial" in available_fonts else "DejaVu Sans"

# Read colors: same_feature = start/end in one exon or intron;
# spliced = CIGAR has N; unspliced = no N.
DEFAULT_READ_COLORS = {
    "same_feature": "#7F7F7F",
    "spliced": "#D62728",
    "unspliced": "#1F77B4",
}
DEFAULT_GENE_EXON_COLOR = "#1F77B4"
DEFAULT_PAS_COLOR = "#9467BD"


def _default_work_dir() -> Path:
    """Project root: the directory that contains ``src/project_config.py``."""
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "src" / "project_config.py").is_file():
            return candidate
    return here.parents[1]


def _default_bed_path() -> Path:
    return _default_work_dir() / "data/refs/exon_intron_pos.chr.bed"


def _default_pas_bed_path() -> Path:
    return _default_work_dir() / "data/refs/Ath.PAS.completed.bed"


_BED_COORD_RE = r"^\d+$"


def _filter_bed_interval_rows(
    df: pd.DataFrame, start_col: str = "start", end_col: str = "end"
) -> pd.DataFrame:
    """Drop header or malformed rows (e.g. ``start`` / ``end`` column names)."""
    s = df[start_col].astype(str)
    e = df[end_col].astype(str)
    mask = s.str.fullmatch(_BED_COORD_RE, na=False) & e.str.fullmatch(
        _BED_COORD_RE, na=False
    )
    return df.loc[mask].copy()


def _base_gene_id(gene_id: str) -> str:
    return str(gene_id).split(".", 1)[0]


def _lookup_pas_interval(
    gene_id: str, pas_bed: pd.DataFrame
) -> tuple[int, int, str] | None:
    for key in (gene_id, _base_gene_id(gene_id)):
        sub = pas_bed[pas_bed["gene_id"] == key]
        if not sub.empty:
            row = sub.iloc[0]
            return int(row["start"]), int(row["end"]), str(row["strand"])
    return None


def _draw_pas_3end_vline(axes, pas_start: int, pas_end: int, strand: str) -> None:
    """Vertical dashed line at PAS 3' end only (+ strand: pas_end; - strand: pas_start)."""
    pas_3end = pas_end if strand == "+" else pas_start
    for ax in axes:
        ax.axvline(
            pas_3end,
            color=DEFAULT_PAS_COLOR,
            linestyle="--",
            linewidth=0.9,
            zorder=10,
        )


def _resolve_bam_path(bam: str | Path, work_dir: Path, bam_dir: Path) -> Path:
    """Resolve an absolute path, a path relative to the working directory, or a file name under data/bams."""
    p = Path(bam)
    if p.is_file():
        return p.resolve()
    cand = bam_dir / p.name
    if cand.is_file():
        return cand.resolve()
    cand2 = work_dir / p
    if cand2.is_file():
        return cand2.resolve()
    if p.exists():
        return p.resolve()
    raise FileNotFoundError(
        f"BAM not found: {bam!r}\n  tried: {p.resolve()}\n  {cand}\n  {cand2}"
    )


def _normalize_bam_paths(bam_paths: str | Path | Sequence[str | Path]) -> list[Path]:
    if isinstance(bam_paths, (str, Path)):
        return [Path(bam_paths)]
    return [Path(p) for p in bam_paths]


def parse_cigar(cigar, start_pos=None, end_pos=None):
    """
    Parse SAM CIGAR (e.g. "50M200N30M") into reference coordinates.

    - blocks: list of (ref_start, ref_end) for each M/=/X segment (aligned to genome).
    - junctions: list of (ref_start, ref_end) for each N (skipped region on reference, intron).
    """
    if start_pos is None:
        start_pos = 0
    ref_pos = int(start_pos)
    query_pos = 0
    blocks = []
    junctions = []
    num_blocks = 0
    num_introns = 0
    last_block_end = None

    tokens = re.findall(r"(\d+)([MIDNSHP=X])", str(cigar).upper())
    for length_str, op in tokens:
        length = int(length_str)
        if op in "M=X":  # alignment to reference (match / sequence match / mismatch)
            blocks.append((ref_pos, ref_pos + length))
            num_blocks += 1
            last_block_end = ref_pos + length
            ref_pos += length
            query_pos += length
        elif op == "I":  # insertion vs reference: consumes query only
            query_pos += length
        elif op == "D":  # deletion: consumes reference only
            ref_pos += length
        elif op == "N":  # skipped region on reference (splice / intron)
            if last_block_end is not None:
                junctions.append((ref_pos, ref_pos + length))
                num_introns += 1
            ref_pos += length
        elif op == "S":  # soft clip: consumes query
            query_pos += length
        elif op in "HP":  # hard clip / padding
            pass

    computed_end = ref_pos
    if end_pos is None:
        end_pos = computed_end
    return {
        "start": start_pos,
        "end": end_pos,
        "blocks": blocks,
        "junctions": junctions,
        "exon_length": query_pos,
        "num_exons": num_blocks,
        "num_introns": num_introns,
    }


def _intervals_overlap(
    a_start: int, a_end: int, b_start: int, b_end: int
) -> bool:
    """True if half-open intervals [a_start, a_end) and [b_start, b_end) overlap."""
    return a_start < b_end and a_end > b_start


def _fetch_reads_for_locus(
    bam_path: Path,
    chrom: str,
    gene_start: int,
    gene_end: int,
    gene_strand: str,
) -> list[dict]:
    """
    Reads assigned to the gene when:
    - alignment strand matches ``gene_strand``;
    - [reference_start, reference_end) overlaps [gene_start, gene_end) (BED span).
    """
    read_rows: list[dict] = []
    with pysam.AlignmentFile(str(bam_path), "rb") as bam:
        for read in bam.fetch(chrom, gene_start, gene_end):
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue
            read_strand = "-" if read.is_reverse else "+"
            if read_strand != gene_strand:
                continue
            rs = int(read.reference_start)
            re = int(read.reference_end)
            if not _intervals_overlap(rs, re, gene_start, gene_end):
                continue
            cig = read.cigarstring
            if not cig:
                continue
            read_rows.append(
                {
                    "read_id": read.query_name,
                    "start": rs,
                    "end": re,
                    "read_cigar": cig,
                }
            )
    return read_rows


def _sort_reads_for_strand(read_rows: list[dict], gene_strand: str) -> None:
    if gene_strand == "+":
        read_rows.sort(key=lambda r: r["end"])
    else:
        read_rows.sort(key=lambda r: r["start"], reverse=True)


def _find_feature_at_pos(
    pos: int,
    segments: list[tuple[int, int, str, str]],
) -> tuple[str, str] | None:
    """Return (segment_name, segment_type) if pos falls in one exon/intron segment."""
    for s, e, name, seg_type in segments:
        if s <= pos < e:
            return name, seg_type
    return None


def _plot_reads_on_ax(
    ax,
    read_rows: list[dict],
    gene_segments: list[tuple[int, int, str, str]],
    min_pos: int,
    max_pos: int,
    boundary_bp: int,
    title: str | None = None,
    *,
    read_exon_h: float = 0.08,
    read_intron_h: float | None = None,
    read_row_gap: float = 0.05,
    read_colors: dict[str, str] | None = None,
) -> tuple[int, int, int]:
    """
    Draw reads on one axes; return junction/no-junction alignment counts.

    These CIGAR-only counts are visualization diagnostics, not splicing
    classifications. Biological splicing analyses must use the classify TSV.

    Each read occupies one horizontal band. Adjacent bands are separated by ``read_row_gap``
    (clear space between the bottom of one read's exon strip and the top of the next), so rows
    do not overlap. Rectangles use ``linewidth=0`` to avoid stroke bleed between rows.
    """
    exon_h = float(read_exon_h)
    intron_h = float(read_intron_h) if read_intron_h is not None else max(0.008, exon_h * 0.12)
    # Distance between row centers = exon_h + read_row_gap, so the gap between bands is read_row_gap.
    y_step = exon_h + float(read_row_gap)
    y_position = 0.0
    spliced_count = 0
    unspliced_count = 0
    # Antialiased edges blend into the background and look grey. Turn antialiasing off so the fill stays solid.
    patch_kw = dict(linewidth=0.01, edgecolor="none", antialiased=False)
    colors = {**DEFAULT_READ_COLORS, **(read_colors or {})}

    for row in read_rows:
        read_cigar_info = parse_cigar(row["read_cigar"], start_pos=row["start"], end_pos=row["end"])
        read_start = int(row["start"])
        read_end = int(row["end"])
        # pysam reference_end is exclusive; use end-1 as read's last covered reference position.
        read_last = max(read_start, read_end - 1)
        start_feature = _find_feature_at_pos(read_start, gene_segments)
        end_feature = _find_feature_at_pos(read_last, gene_segments)
        in_same_feature = (
            start_feature is not None
            and end_feature is not None
            and start_feature == end_feature
            and start_feature[1] in {"exon", "intron"}
        )
        if in_same_feature:
            color = colors["same_feature"]
        elif read_cigar_info["num_introns"] > 0:
            color = colors["spliced"]
            spliced_count += 1
        else:
            color = colors["unspliced"]
            unspliced_count += 1
        for block in read_cigar_info["blocks"]:
            ax.add_patch(
                Rectangle(
                    (block[0], y_position - exon_h / 2),
                    block[1] - block[0],
                    exon_h,
                    facecolor=color,
                    **patch_kw,
                )
            )
        for junction in read_cigar_info["junctions"]:
            ax.add_patch(
                Rectangle(
                    (junction[0], y_position - intron_h / 2),
                    junction[1] - junction[0],
                    intron_h,
                    facecolor=color,
                    **patch_kw,
                )
            )
        y_position -= y_step

    n_reads = len(read_rows)
    ax.set_xlim(min_pos - boundary_bp, max_pos + boundary_bp)
    ax.set_ylim(y_position, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["top"].set_visible(False)
    if title:
        ax.set_title(title, loc="left", fontsize=10)
    ax.text(
        0.99,
        0.98,
        f"Reads: {n_reads}\nWith junction: {spliced_count}\nNo junction: {unspliced_count}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
    )
    return n_reads, spliced_count, unspliced_count


def plot_gene_reads(
    gene_id: str,
    bam_paths: str | Path | Sequence[str | Path],
    *,
    bed_path: str | Path | None = None,
    pas_bed_path: str | Path | None = None,
    out_dir: str | Path | None = None,
    sample_labels: Sequence[str] | None = None,
    boundary_bp: int = 200,
    plot_width: float = 8,
    plot_height: float = 6,
    plot_height_ratio: float = 0.05,
    dpi: int = 600,
    read_exon_h: float = 0.08,
    read_intron_h: float | None = None,
    read_row_gap: float = 0.1,
    read_colors: dict[str, str] | None = None,
    gene_exon_color: str | None = None,
    stack_bams: bool = False,
    pool_groups: Sequence[tuple[str, int]] | None = None,
    **kwargs,
) -> tuple[Path, Path]:
    """
    Plot reads overlapping a gene from one or more BAMs and the gene exon/intron model from BED.

    Only reads on the same strand as the gene (BED ``strand``) and overlapping the gene
    span (min/max of that transcript's BED rows, half-open coordinates) are shown.

    By default all BAMs are merged into a single read track. With ``stack_bams=True``,
    each BAM gets its own read panel (first BAM on top, last BAM above the gene model).

    Notebook aliases: ``width`` to ``plot_width``, ``height`` to ``plot_height``,
    ``height_ratio`` to ``plot_height_ratio``. ``dpi`` may be overridden through ``**kwargs``.
    ``labels`` maps to ``sample_labels`` only when ``sample_labels`` was not passed.

    Parameters
    ----------
    gene_id : str
        Transcript / mRNA id (e.g. AT1G01010.1), must match the prefix before '_' in BED name column.
    bam_paths : path-like or sequence of path-like
        One or more indexed BAM files (each requires .bai).
    bed_path : path-like, optional
        Exon/intron BED (name like AT1G01010.1_exon1). Defaults to data/refs/exon_intron_pos.chr.bed.
    pas_bed_path : path-like, optional
        PAS BED for vertical dashed markers. Defaults to data/refs/Ath.PAS.completed.bed.
    out_dir : path-like, optional
        Figure output directory. Default: first BAM's parent / gene_reads_figures
    sample_labels : sequence of str, optional
        One label per BAM, shown in the read panel title joined by \" + \". Default: file stems.
    boundary_bp : int
        Padding (bp) on both sides of the locus.
    plot_width : float
        Figure width in inches (matplotlib ``figsize``).
    plot_height : float
        Figure height in inches (read panel + gene track).
    plot_height_ratio : float
        Height ratio of the gene track panel vs read panel (second row / first row).
    dpi : int
        Resolution for PNG/PDF output (``savefig``).
    read_exon_h : float
        Height of the exon block in a read band, in y data units, independent of genomic x.
    read_intron_h : float, optional
        Height of the junction bar. Defaults to slightly less than ``read_exon_h``.
    read_row_gap : float
        Clear gap between adjacent read bands, from the bottom of one exon strip to the top of the next.
    read_colors : dict, optional
        Override read fills. Keys: ``same_feature``, ``spliced``, ``unspliced``.
    gene_exon_color : str, optional
        Fill color for exons on the gene model track.
    stack_bams : bool, default=False
        If True and multiple BAMs are given, draw one read panel per BAM
        (first on top, last above the gene model). If False, merge all BAMs
        into a single read track.

    Returns
    -------
    (png_path, pdf_path)
    """
    if kwargs:
        kw = dict(kwargs)
        if "width" in kw:
            plot_width = float(kw.pop("width"))
        if "height" in kw:
            plot_height = float(kw.pop("height"))
        if "height_ratio" in kw:
            plot_height_ratio = float(kw.pop("height_ratio"))
        if "dpi" in kw:
            dpi = int(kw.pop("dpi"))
        if "read_exon_h" in kw:
            read_exon_h = float(kw.pop("read_exon_h"))
        if "read_intron_h" in kw:
            v = kw.pop("read_intron_h")
            read_intron_h = None if v is None else float(v)
        if "read_row_gap" in kw:
            read_row_gap = float(kw.pop("read_row_gap"))
        if "labels" in kw and sample_labels is None:
            sample_labels = kw.pop("labels")
        elif "labels" in kw:
            kw.pop("labels")
        if "read_colors" in kw:
            read_colors = kw.pop("read_colors")
        if "gene_exon_color" in kw:
            gene_exon_color = kw.pop("gene_exon_color")
        if "stack_bams" in kw:
            stack_bams = bool(kw.pop("stack_bams"))
        if kw:
            bad = ", ".join(sorted(kw))
            raise TypeError(f"plot_gene_reads() got unexpected keyword argument(s): {bad}")

    paths = _normalize_bam_paths(bam_paths)
    if not paths:
        raise ValueError("At least one BAM path is required.")

    if bed_path is None:
        bed_path = _default_bed_path()
    bed_path = Path(bed_path)
    if pas_bed_path is None:
        pas_bed_path = _default_pas_bed_path()
    pas_bed_path = Path(pas_bed_path)
    if out_dir is None:
        out_dir = paths[0].parent / "gene_reads_figures"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if sample_labels is not None:
        labels = list(sample_labels)
        if len(labels) != len(paths):
            raise ValueError(
                f"sample_labels length ({len(labels)}) must match number of BAMs ({len(paths)})."
            )
    else:
        labels = [p.stem for p in paths]

    gene_bed = pd.read_csv(bed_path, header=None, sep="\t")
    gene_bed.columns = ["chrom", "start", "end", "gene", "score", "strand"]
    gene_bed = _filter_bed_interval_rows(gene_bed)
    gene_bed["start"] = gene_bed["start"].astype(int)
    gene_bed["end"] = gene_bed["end"].astype(int)
    gene_bed["gene_id"] = gene_bed["gene"].str.split("_").str[0]
    gene_bed_selected = gene_bed[gene_bed["gene_id"] == gene_id]
    if gene_bed_selected.empty:
        raise ValueError(f"No rows for gene_id={gene_id!r} in {bed_path}")

    gene_region_start = int(gene_bed_selected["start"].min())
    gene_region_end = int(gene_bed_selected["end"].max())
    chrom = str(gene_bed_selected["chrom"].iloc[0])
    gene_strand = str(gene_bed_selected["strand"].iloc[0])

    reads_by_bam: list[list[dict]] = []
    for bam_p in paths:
        rows = _fetch_reads_for_locus(
            bam_p, chrom, gene_region_start, gene_region_end, gene_strand
        )
        _sort_reads_for_strand(rows, gene_strand)
        reads_by_bam.append(rows)

    if pool_groups:
        panel_reads = []
        panel_labels = []
        offset = 0
        for label, n_bams in pool_groups:
            if n_bams < 1:
                raise ValueError(f"pool group {label!r} must include at least one BAM.")
            chunk: list[dict] = []
            for rows in reads_by_bam[offset : offset + n_bams]:
                chunk.extend(rows)
            _sort_reads_for_strand(chunk, gene_strand)
            panel_reads.append(chunk)
            panel_labels.append(label)
            offset += n_bams
        if offset != len(reads_by_bam):
            raise ValueError(
                f"pool_groups cover {offset} BAMs, but {len(reads_by_bam)} were provided."
            )
    elif stack_bams and len(paths) > 1:
        panel_reads = reads_by_bam
        panel_labels = list(labels)
    else:
        merged_reads: list[dict] = []
        for rows in reads_by_bam:
            merged_reads.extend(rows)
        _sort_reads_for_strand(merged_reads, gene_strand)
        panel_reads = [merged_reads]
        panel_labels = [" + ".join(labels) if len(labels) > 1 else labels[0]]

    all_reads = [row for rows in panel_reads for row in rows]
    if all_reads:
        locus_min_pos = min(int(r["start"]) for r in all_reads)
        locus_max_pos = max(int(r["end"]) for r in all_reads)
    else:
        locus_min_pos = gene_region_start
        locus_max_pos = gene_region_end

    n_read = len(panel_reads)
    fig, axes = plt.subplots(
        figsize=(plot_width, plot_height),
        ncols=1,
        nrows=n_read + 1,
        height_ratios=[1] * n_read + [plot_height_ratio],
        squeeze=False,
    )
    axes = list(axes[:, 0])
    ax_gene = axes[-1]

    exon_h = 0.1
    intron_h = 0.001
    gene_segments = [
        (int(r.start), int(r.end), str(r.gene), "exon" if "exon" in str(r.gene) else "intron")
        for r in gene_bed_selected.itertuples()
        if ("exon" in str(r.gene) or "intron" in str(r.gene))
    ]
    for ax_reads, rows, title in zip(axes[:-1], panel_reads, panel_labels):
        _plot_reads_on_ax(
            ax_reads,
            rows,
            gene_segments,
            locus_min_pos,
            locus_max_pos,
            boundary_bp,
            title=title,
            read_exon_h=read_exon_h,
            read_intron_h=read_intron_h,
            read_row_gap=read_row_gap,
            read_colors=read_colors,
        )
    if n_read > 1:
        for ax in axes[:-2]:
            ax.spines["bottom"].set_visible(True)
            ax.spines["bottom"].set_color("black")
            ax.spines["bottom"].set_linewidth(1.0)
    arrow_len = min((locus_max_pos - locus_min_pos) * 0.03, boundary_bp * 0.8)
    # The gene-track baseline covers only the annotated gene, so it does not look longer than the gene.
    ax_gene.plot(
        [gene_region_start, gene_region_end], [0, 0], color="black", linewidth=0.4, zorder=0
    )
    for row in gene_bed_selected.itertuples():
        start = int(row.start)
        end = int(row.end)
        gname = str(row.gene)
        if "exon" in gname:
            ax_gene.add_patch(
                Rectangle(
                    (start, -exon_h / 2),
                    end - start,
                    exon_h,
                    facecolor=gene_exon_color or DEFAULT_GENE_EXON_COLOR,
                    linewidth=0,
                    edgecolor="none",
                    antialiased=False,
                    zorder=2,
                )
            )
        elif "intron" in gname:
            ax_gene.add_patch(
                Rectangle(
                    (start, -intron_h / 2),
                    end - start,
                    intron_h,
                    facecolor="black",
                    linewidth=0,
                    edgecolor="none",
                    antialiased=False,
                    zorder=1,
                )
            )

    # The strand arrow is always on the right of the panel.
    # On an unflipped plus strand that is gene_region_end; after a minus-strand flip it is gene_region_start.
    anchor_x = gene_region_end if gene_strand == "+" else gene_region_start
    if gene_strand == "+":
        ax_gene.annotate(
            "",
            xy=(anchor_x + arrow_len, 0),
            xytext=(anchor_x - arrow_len * 0.1, 0),
            arrowprops=dict(arrowstyle="->", lw=1),
        )
    else:
        ax_gene.annotate(
            "",
            xy=(anchor_x - arrow_len, 0),
            xytext=(anchor_x + arrow_len * 0.1, 0),
            arrowprops=dict(arrowstyle="->", lw=1),
        )

    if pas_bed_path.is_file():
        pas_bed = pd.read_csv(
            pas_bed_path,
            sep="\t",
            header=None,
            names=["chr", "start", "end", "gene_id", "score", "strand"],
        )
        pas_bed = _filter_bed_interval_rows(pas_bed)
        pas_bed["start"] = pas_bed["start"].astype(int)
        pas_bed["end"] = pas_bed["end"].astype(int)
        pas_hit = _lookup_pas_interval(gene_id, pas_bed)
        if pas_hit is not None:
            pas_start, pas_end, pas_strand = pas_hit
            _draw_pas_3end_vline(axes, pas_start, pas_end, pas_strand)

    ax_gene.set_xlim(locus_min_pos - boundary_bp, locus_max_pos + boundary_bp)
    ax_gene.set_ylim(-(0.7 * exon_h), 0.7 * exon_h)
    ax_gene.xaxis.set_major_locator(MaxNLocator(nbins=8))
    ax_gene.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x / 1000:.2f}"))
    ax_gene.xaxis.set_ticks_position("bottom")
    ax_gene.xaxis.set_label_position("bottom")
    ax_gene.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    ax_gene.set_xlabel("Position (kb)")
    ax_gene.set_yticks([])
    ax_gene.spines["left"].set_visible(False)
    ax_gene.spines["right"].set_visible(False)
    ax_gene.spines["top"].set_visible(False)

    if gene_strand == "-":
        for ax in axes:
            ax.invert_xaxis()

    fig.subplots_adjust(hspace=0.01)

    name_parts = list(panel_labels)
    sample_key = "__".join(name_parts)
    stem = f"{gene_id}_{sample_key}_gene_reads"
    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"
    plt.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.savefig(pdf_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return png_path, pdf_path


def plot_reads_genes(
    gene_id: str,
    *bams: str | Path,
    work_dir: str | Path | None = None,
    bam_dir: str | Path | None = None,
    labels: Sequence[str] | None = None,
    bed_path: str | Path | None = None,
    pas_bed_path: str | Path | None = None,
    out_dir: str | Path | None = None,
    width: float = 8,
    height: float = 6,
    plot_width: float | None = None,
    plot_height: float | None = None,
    boundary_bp: int = 200,
    height_ratio: float = 0.05,
    dpi: int = 600,
    read_exon_h: float = 0.08,
    read_intron_h: float | None = None,
    read_row_gap: float = 0.12,
    read_colors: dict[str, str] | None = None,
    gene_exon_color: str | None = None,
    stack_bams: bool = False,
    pool_groups: Sequence[tuple[str, int]] | None = None,
) -> tuple[Path, Path]:
    """
    Short wrapper for a Jupyter or IPython cell.

    Parameters
    ----------
    gene_id : str
        Gene or transcript ID.
    *bams : str | Path
        One or more BAMs. A full path or a file name resolved under ``bam_dir``.
    work_dir, bam_dir, labels, bed_path, out_dir
        Paths and legend labels. See the defaults in the source.
    width, height : float
        Figure size in inches (``figsize``). ``plot_width`` and ``plot_height`` override these when both are set.
    plot_width, plot_height : float, optional
        Same as ``width`` and ``height``. Names match ``plot_gene_reads``.
    boundary_bp : int
        Extra bases shown on each side of the gene.
    height_ratio : float
        Height ratio of the gene-model panel to the read panel (second axes : first axes).
    dpi : int
        PNG and PDF resolution.
    read_exon_h, read_intron_h, read_row_gap
        Same band height and row gap as in ``plot_gene_reads``.
    stack_bams : bool
        If True, each BAM gets its own read panel, with the first BAM on top. If False, BAMs share one track.

    Examples
    --------
    >>> from plot_gene_reads import plot_reads_genes
    >>> plot_reads_genes("AT1G01040.2", "Warm_Ser5_rep1.elongating.bam", "Warm_Ser5_rep2.elongating.bam")
    >>> plot_reads_genes(
    ...     "AT1G01040.2", "a.bam", "b.bam",
    ...     width=10, height=8, height_ratio=0.06, dpi=300,
    ... )
    """
    if not bams:
        raise ValueError("Pass at least one BAM path or file name.")
    wd = Path(work_dir) if work_dir is not None else _default_work_dir()
    bd = Path(bam_dir) if bam_dir is not None else wd / "data" / "bams"
    resolved = [_resolve_bam_path(b, wd, bd) for b in bams]
    if bed_path is None:
        bed_path = wd / "data/refs/exon_intron_pos.chr.bed"
    if pas_bed_path is None:
        pas_bed_path = wd / "data/refs/Ath.PAS.completed.bed"
    if out_dir is None:
        out_dir = wd / "results" / "gene_reads"
    pw = float(plot_width) if plot_width is not None else float(width)
    ph = float(plot_height) if plot_height is not None else float(height)
    return plot_gene_reads(
        gene_id,
        resolved,
        bed_path=bed_path,
        pas_bed_path=pas_bed_path,
        out_dir=out_dir,
        sample_labels=labels,
        boundary_bp=boundary_bp,
        plot_width=pw,
        plot_height=ph,
        plot_height_ratio=height_ratio,
        dpi=dpi,
        read_exon_h=read_exon_h,
        read_intron_h=read_intron_h,
        read_row_gap=read_row_gap,
        read_colors=read_colors,
        gene_exon_color=gene_exon_color,
        stack_bams=stack_bams,
        pool_groups=pool_groups,
    )


DEFAULT_WORK_DIR = _default_work_dir()

__all__ = [
    "DEFAULT_WORK_DIR",
    "plot_gene_reads",
    "plot_reads_genes",
    "parse_cigar",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot gene reads from one or more BAMs + exon/intron BED."
    )
    parser.add_argument("--gene-id", required=True, help="Gene / mRNA id, e.g. AT5G57380.1")
    parser.add_argument(
        "--bam",
        type=Path,
        nargs="+",
        required=True,
        help="One or more indexed BAM files.",
    )
    parser.add_argument(
        "--bed",
        type=Path,
        default=None,
        help=f"Exon/intron BED (default: {_default_bed_path()})",
    )
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory for figures.")
    parser.add_argument(
        "--label",
        dest="sample_labels",
        nargs="*",
        default=None,
        help="Optional label per BAM (same order as --bam); shown in read panel title joined by ' + '.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    png, pdf = plot_gene_reads(
        args.gene_id,
        args.bam,
        bed_path=args.bed,
        out_dir=args.out_dir,
        sample_labels=args.sample_labels,
    )
    print(f"Wrote {png}\n{pdf}")


if __name__ == "__main__":
    main()
