#!/usr/bin/env python3
"""Classify RNA-seq reads by splicing status and readthrough from BAM + annotation.

Uses only pysam and the Python standard library (no project-specific imports).

Annotation (auto-detect):
  - exon/intron BED6 (default): data/refs/exon_intron_pos.chr.bed
      chrom start end TX_exonN|TX_intronN . strand
  - BED12 (one transcript per line)
  - GTF/GFF (exon features)
PAS: BED6 (chrom, start, end, name, score, strand).

Also maps strand-relative 5'/3' endpoints (P5/P3) to gene features, with
relative distances within each feature.

Only primary, same-strand alignments passing the mapping-quality filters are
kept.

Reads whose P5 endpoint is neither promoter nor exon_1 are retained in the
output for auditing but assigned splicing_class=invalid_p5_start. Downstream
splicing and readthrough summaries should exclude that class.

Splicing status is based on CIGAR evidence, plus one P5-in-intron rule. An
intron is SPLICED when a
CIGAR N operation covers at least a configurable fraction of the annotated
intron (default: 80%). The junction is canonical when both splice-site
coordinates are within a configurable tolerance of the annotation (default:
5 bp); otherwise it is additionally reported as alternative splicing. Both
canonical and alternative junctions count as spliced evidence. An intron is RETAINED
when one continuous aligned block covers the full intron plus a configurable
anchor on both sides, or when P5 falls inside that intron more than a
configurable distance (default: 10 bp) past the intron 5' splice site and
CIGAR has not already called the intron spliced. A 3' end inside an intron
is reported separately and never changes the CIGAR-based splicing class.

covered_introns is defined only by P5/P3 feature span (same range as
expected_introns): introns that the read is expected to have traversed
between its 5' and 3' endpoints. It is not limited to introns with direct
CIGAR spliced/retained calls. spliced_introns / retained_introns still come
from CIGAR (+ P5-in-intron) evidence.

A read is fully_spliced when every covered (P5–P3) intron is spliced, none
are retained, and there is no relevant noncanonical junction. A read is
fully_unspliced when every covered intron is retained and none are spliced.
If covered and spliced/retained do not match (e.g. covered=1,2,3,4 but
spliced=2,3,4), the class is partial_spliced. Alternative splice coordinates
are retained for auditing but do not by themselves change the read class.
Alternative splicing events are labelled SE, RI, MXE, A5SS, or A3SS in
alternative_splicing_events / alternative_splicing_details. Otherwise
unannotated junctions produce partial_spliced when relevant to the expected
annotated introns. Reads with only unannotated junctions and no
expected/covered annotated intron are reported as unannotated_junction_only.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pysam

EXON_INTRON_NAME_RE = re.compile(r"^(?P<tx>.+)_(?P<kind>exon|intron)(?P<num>\d+)$", re.IGNORECASE)

DEFAULT_INTRON_END_THRESHOLD = 10
DEFAULT_MIN_ANCHOR = 10
DEFAULT_P5_INTRON_RETAIN_BP = 10
DEFAULT_MIN_MAPQ = 20
VALID_P5_FEATURES = frozenset({"promoter", "exon_1"})
INVALID_P5_CLASS = "invalid_p5_start"

OUTPUT_FIELDS = [
    "read_id",
    "gene_id",
    "chrom",
    "read_start",
    "read_end",
    "strand",
    "n_introns",
    "covered_introns",
    "spliced_introns",
    "alternative_spliced_introns",
    "retained_introns",
    "expected_introns",
    "alternative_junctions",
    "unannotated_junctions",
    "alternative_splicing_events",
    "alternative_splicing_details",
    "splicing_class",
    "is_readthrough",
    "readthrough_dist",
    "pas_site",
    "p5_coord",
    "p5_feat",
    "p5_relative_dist",
    "p3_coord",
    "p3_feat",
    "p3_relative_dist",
    "intron_terminated",
    "intron_terminated_id",
]


@dataclass
class GeneModel:
    gene_id: str
    chrom: str
    start: int  # 0-based inclusive
    end: int  # 0-based exclusive
    strand: str
    exons: list[tuple[int, int]] = field(default_factory=list)  # sorted, merged, tx order

    @property
    def introns(self) -> list[tuple[int, int]]:
        """Intron intervals (0-based half-open, genomic ascending) in transcription order."""
        if len(self.exons) < 2:
            return []
        out: list[tuple[int, int]] = []
        for i in range(len(self.exons) - 1):
            a_start, a_end = self.exons[i]
            b_start, b_end = self.exons[i + 1]
            # Genomic interval between consecutive exons in transcription order.
            # + strand: (a_end, b_start); − strand: (b_end, a_start) — unified as:
            intron_start = min(a_end, b_end)
            intron_end = max(a_start, b_start)
            if intron_end > intron_start:
                out.append((intron_start, intron_end))
        return out


def parse_gtf_attributes(attr_field: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in attr_field.strip().split(";"):
        part = part.strip()
        if not part:
            continue
        if " " not in part:
            continue
        key, val = part.split(" ", 1)
        out[key] = val.strip().strip('"')
    return out


def merge_exons(exons: list[tuple[int, int]], strand: str) -> list[tuple[int, int]]:
    """Sort exons in transcription order and merge overlaps/adjacent blocks."""
    if not exons:
        return []
    exons = sorted(exons, key=lambda x: x[0])
    merged: list[list[int]] = [[exons[0][0], exons[0][1]]]
    for s, e in exons[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    if strand == "-":
        merged = [list(x) for x in reversed(merged)]
    return [(s, e) for s, e in merged]


def load_genes_from_gtf(path: str) -> list[GeneModel]:
    """Build gene models from GTF exon rows grouped by gene_id."""
    exons_by_gene: dict[str, list[tuple[str, int, int, str]]] = defaultdict(list)
    gene_strand: dict[str, str] = {}

    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9:
                continue
            chrom, _src, feature, start, end, _score, strand, _frame, attrs = fields[:9]
            if feature != "exon":
                continue
            attr = parse_gtf_attributes(attrs)
            gene_id = attr.get("gene_id") or attr.get("transcript_id") or attr.get("gene")
            if not gene_id:
                continue
            s, e = int(start) - 1, int(end)  # GTF 1-based closed -> BED half-open
            exons_by_gene[gene_id].append((chrom, s, e, strand))
            gene_strand[gene_id] = strand

    genes: list[GeneModel] = []
    for gene_id, rows in exons_by_gene.items():
        chrom = rows[0][0]
        strand = gene_strand[gene_id]
        raw_exons = [(s, e) for c, s, e, st in rows if c == chrom]
        exons = merge_exons(raw_exons, strand)
        if not exons:
            continue
        genes.append(
            GeneModel(
                gene_id=gene_id,
                chrom=chrom,
                start=min(s for s, _ in exons),
                end=max(e for _, e in exons),
                strand=strand,
                exons=exons,
            )
        )
    return genes


def load_genes_from_bed12(path: str) -> list[GeneModel]:
    genes: list[GeneModel] = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            fields = [x.strip() for x in line.rstrip("\n").split("\t")]
            if len(fields) < 12:
                continue
            chrom = fields[0]
            try:
                chrom_start = int(fields[1])
                chrom_end = int(fields[2])
                block_count = int(fields[9])
            except ValueError:
                continue
            gene_id = fields[3]
            strand = fields[5] if fields[5] in ("+", "-") else "+"
            block_sizes = [int(x) for x in fields[10].rstrip(",").split(",") if x]
            block_starts = [int(x) for x in fields[11].rstrip(",").split(",") if x]
            if block_count != len(block_sizes) or block_count != len(block_starts):
                continue
            exons: list[tuple[int, int]] = []
            for i in range(block_count):
                s = chrom_start + block_starts[i]
                e = s + block_sizes[i]
                exons.append((s, e))
            exons = merge_exons(exons, strand)
            genes.append(
                GeneModel(
                    gene_id=gene_id,
                    chrom=chrom,
                    start=chrom_start,
                    end=chrom_end,
                    strand=strand,
                    exons=exons,
                )
            )
    return genes


def _default_work_dir() -> Path:
    """Return the project root, the directory that contains ``src/project_config.py``."""
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "src" / "project_config.py").is_file():
            return candidate
    return here.parents[2]


def _default_annotation() -> Path:
    return _default_work_dir() / "data/refs/exon_intron_pos.chr.bed"


def _default_pas() -> Path:
    return _default_work_dir() / "data/refs/Ath.PAS.completed.bed"


def parse_exon_intron_name(name: str) -> tuple[str, str, int] | None:
    """Parse TX_exon3 / TX_intron2 → (transcript_id, kind, number)."""
    m = EXON_INTRON_NAME_RE.match(name.strip())
    if not m:
        return None
    return m.group("tx"), m.group("kind").lower(), int(m.group("num"))


def load_genes_from_exon_intron_bed(path: str) -> list[GeneModel]:
    """Build gene models from FLEPSeq-style exon/intron BED (one feature per line)."""
    # (chrom, transcript_id, strand) -> {exon_num: (start, end)}
    exons_by_tx: dict[tuple[str, str, str], dict[int, tuple[int, int]]] = defaultdict(dict)

    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            fields = [x.strip() for x in line.rstrip("\n").split("\t")]
            if len(fields) < 6:
                continue
            chrom, name, strand = fields[0], fields[3], fields[5]
            if strand not in ("+", "-"):
                continue
            parsed = parse_exon_intron_name(name)
            if parsed is None:
                continue
            tx_id, kind, feat_num = parsed
            if kind != "exon":
                continue
            try:
                start, end = int(fields[1]), int(fields[2])
            except ValueError:
                continue
            if end <= start:
                continue
            key = (chrom, tx_id, strand)
            exons_by_tx[key][feat_num] = (start, end)

    genes: list[GeneModel] = []
    for (chrom, tx_id, strand), exon_dict in exons_by_tx.items():
        if not exon_dict:
            continue
        exons = [exon_dict[n] for n in sorted(exon_dict)]
        genes.append(
            GeneModel(
                gene_id=tx_id,
                chrom=chrom,
                start=min(s for s, _ in exons),
                end=max(e for _, e in exons),
                strand=strand,
                exons=exons,
            )
        )
    genes.sort(key=lambda g: (g.chrom, g.start, g.gene_id))
    return genes


def _detect_annotation_format(path: str) -> str:
    lower = path.lower()
    if lower.endswith(".gtf") or lower.endswith(".gff") or lower.endswith(".gff3"):
        return "gtf"
    if "exon_intron" in lower:
        return "exon_intron"

    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) >= 12:
                try:
                    int(fields[9])
                    return "bed12"
                except ValueError:
                    pass
            if len(fields) >= 4:
                parsed = parse_exon_intron_name(fields[3])
                if parsed is not None:
                    return "exon_intron"
            break
    return "bed12"


def load_annotation(path: str) -> list[GeneModel]:
    fmt = _detect_annotation_format(path)
    if fmt == "gtf":
        return load_genes_from_gtf(path)
    if fmt == "exon_intron":
        return load_genes_from_exon_intron_bed(path)
    return load_genes_from_bed12(path)


@dataclass
class PasRecord:
    chrom: str
    start: int
    end: int
    name: str
    strand: str


def load_pas_bed(path: str) -> list[PasRecord]:
    records: list[PasRecord] = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            f = [x.strip() for x in line.rstrip("\n").split("\t")]
            if len(f) < 6:
                continue
            try:
                start, end = int(f[1]), int(f[2])
            except ValueError:
                continue
            records.append(
                PasRecord(
                    chrom=f[0],
                    start=start,
                    end=end,
                    name=f[3],
                    strand=f[5],
                )
            )
    return records


def index_pas_by_chrom(pas_records: list[PasRecord]) -> dict[str, list[PasRecord]]:
    by_chrom: dict[str, list[PasRecord]] = defaultdict(list)
    for rec in pas_records:
        by_chrom[rec.chrom].append(rec)
    for chrom in by_chrom:
        by_chrom[chrom].sort(key=lambda r: r.start)
    return by_chrom


def find_downstream_pas(gene: GeneModel, pas_by_chrom: dict[str, list[PasRecord]]) -> PasRecord | None:
    """Nearest downstream PAS on the same chromosome."""
    candidates = pas_by_chrom.get(gene.chrom, [])
    if not candidates:
        return None
    if gene.strand == "+":
        downstream = [p for p in candidates if p.start >= gene.end]
        if not downstream:
            return None
        return min(downstream, key=lambda p: p.start)
    downstream = [p for p in candidates if p.end <= gene.start]
    if not downstream:
        return None
    return max(downstream, key=lambda p: p.end)


def read_alignment_strand(read: pysam.AlignedSegment) -> str:
    """Alignment strand on reference (+ / -)."""
    return "-" if read.is_reverse else "+"


def read_matches_gene_strand(read: pysam.AlignedSegment, gene_strand: str) -> bool:
    """Keep reads whose alignment strand matches the gene strand."""
    return read_alignment_strand(read) == gene_strand


def read_matches_library_strand(read: pysam.AlignedSegment, gene_strand: str, library: str) -> bool:
    if library == "unstranded":
        return True
    is_rev = read.is_reverse
    if library == "fr-firststrand":
        return (gene_strand == "+" and not is_rev) or (gene_strand == "-" and is_rev)
    if library == "fr-secondstrand":
        return (gene_strand == "+" and is_rev) or (gene_strand == "-" and not is_rev)
    raise ValueError(f"Unknown library type: {library}")


def overlaps_gene(read: pysam.AlignedSegment, gene: GeneModel) -> bool:
    return read.reference_start < gene.end and read.reference_end > gene.start


def parse_cigar_blocks_and_junctions(
    read: pysam.AlignedSegment,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Return reference-coverage blocks and N-skipped regions (0-based half-open).

    Match/equal/mismatch and deletions cover the reference. Intronic skips (N)
    are junctions and break coverage. A 1-bp D inside an intron must not prevent
    calling that intron retained.
    """
    if read.cigartuples is None:
        return [], []
    ref_pos = read.reference_start
    blocks: list[tuple[int, int]] = []
    junctions: list[tuple[int, int]] = []
    for op, length in read.cigartuples:
        if op in (0, 2, 7, 8):  # M, D, =, X
            blocks.append((ref_pos, ref_pos + length))
            ref_pos += length
        elif op == 3:  # N
            junctions.append((ref_pos, ref_pos + length))
            ref_pos += length
        elif op in (1, 4, 5, 6):  # I, S, H, P
            continue
    return blocks, junctions


def merge_blocks(blocks: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not blocks:
        return []
    blocks = sorted(blocks)
    merged = [[blocks[0][0], blocks[0][1]]]
    for s, e in blocks[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def covers_interval(blocks: list[tuple[int, int]], start: int, end: int) -> bool:
    """True if merged blocks continuously cover [start, end)."""
    for bs, be in blocks:
        if bs <= start and be >= end:
            return True
        if be <= start:
            continue
        if bs > start:
            return False
        # extend coverage
        start = be
        if start >= end:
            return True
    return False


def junction_matches_intron(
    junction: tuple[int, int],
    intron: tuple[int, int],
    tolerance: int,
) -> bool:
    j_start, j_end = junction
    i_start, i_end = intron
    return abs(j_start - i_start) <= tolerance and abs(j_end - i_end) <= tolerance


def junction_intron_overlap_fraction(
    junction: tuple[int, int],
    intron: tuple[int, int],
) -> float:
    """Return the fraction of the annotated intron covered by one junction."""
    j_start, j_end = junction
    i_start, i_end = intron
    intron_length = i_end - i_start
    if intron_length <= 0:
        return 0.0
    overlap = max(0, min(j_end, i_end) - max(j_start, i_start))
    return overlap / intron_length


def junction_splices_intron(
    junction: tuple[int, int],
    intron: tuple[int, int],
    min_junction_fraction: float,
) -> bool:
    """Return whether a junction removes enough of an annotated intron."""
    j_start, j_end = junction
    i_start, i_end = intron
    intron_length = i_end - i_start
    junction_length = j_end - j_start
    return (
        intron_length > 0
        and junction_length >= intron_length * min_junction_fraction
        and junction_intron_overlap_fraction(junction, intron)
        >= min_junction_fraction
    )


def junction_overlaps_intron_region(
    junction: tuple[int, int],
    intron: tuple[int, int],
) -> bool:
    j_start, j_end = junction
    i_start, i_end = intron
    return j_start < i_end and j_end > i_start


AS_EVENT_ORDER = ("SE", "RI", "MXE", "A5SS", "A3SS")


def intron_donor_acceptor(intron: tuple[int, int], strand: str) -> tuple[int, int]:
    """Return donor and acceptor splice coordinates for one intron."""
    i_start, i_end = intron
    if strand == "-":
        return i_end, i_start
    return i_start, i_end


def interval_overlaps_any(
    interval: tuple[int, int],
    blocks: list[tuple[int, int]],
) -> bool:
    start, end = interval
    for bs, be in blocks:
        if bs < end and be > start:
            return True
    return False


def classify_junction_event(
    junction: tuple[int, int],
    introns: list[tuple[int, int]],
    strand: str,
    tolerance: int,
) -> tuple[set[str], list[str]]:
    """Label A5SS/A3SS for a junction overlapping one annotated intron.

    Junctions that span two or more annotated introns are handled as SE/MXE
    from exon inclusion, not here.
    """
    j_start, j_end = junction
    events: set[str] = set()
    details: list[str] = []
    overlapped = [
        idx
        for idx, intron in enumerate(introns, start=1)
        if junction_overlaps_intron_region(junction, intron)
    ]
    if len(overlapped) != 1:
        return events, details

    idx = overlapped[0]
    intron = introns[idx - 1]
    donor, acceptor = intron_donor_acceptor(intron, strand)
    junction_donor = j_end if strand == "-" else j_start
    junction_acceptor = j_start if strand == "-" else j_end
    donor_matches = abs(junction_donor - donor) <= tolerance
    acceptor_matches = abs(junction_acceptor - acceptor) <= tolerance
    if donor_matches and not acceptor_matches:
        events.add("A3SS")
        details.append(f"A3SS:junction={j_start}-{j_end}:intron={idx}")
    elif acceptor_matches and not donor_matches:
        events.add("A5SS")
        details.append(f"A5SS:junction={j_start}-{j_end}:intron={idx}")
    elif not donor_matches and not acceptor_matches:
        events.update(("A5SS", "A3SS"))
        details.append(f"A5SS+A3SS:junction={j_start}-{j_end}:intron={idx}")
    return events, details


def classify_se_mxe_events(
    exons: list[tuple[int, int]],
    blocks: list[tuple[int, int]],
    read_start: int,
    read_end: int,
) -> tuple[set[str], list[str], set[int]]:
    """Call SE and MXE from which annotated exons the read includes.

    Between two consecutive included exons, skipped fully spanned exons are SE
    unless the local structure is a 4-exon MXE: exactly one skipped exon, and
    only one side of that cassette has an extra included exon (C-A-D or C-B-D).
    A cassette with extra exons on both sides is SE, not MXE.
    """
    events: set[str] = set()
    details: list[str] = []
    n_exons = len(exons)
    if n_exons < 3:
        return events, details, set()

    included = [interval_overlaps_any(exon, blocks) for exon in exons]
    spanned = [
        read_start <= exon[0] and read_end >= exon[1]
        for exon in exons
    ]
    mxe_skipped: set[int] = set()
    included_idx = [i for i, flag in enumerate(included) if flag]
    for left, right in zip(included_idx, included_idx[1:]):
        if right == left + 1:
            continue
        skipped = list(range(left + 1, right))
        if not all(spanned[idx] for idx in skipped):
            continue
        skipped_ids = [idx + 1 for idx in skipped]
        left_ext = left > 0 and included[left - 1]
        right_ext = right + 1 < n_exons and included[right + 1]
        if len(skipped_ids) == 1 and left_ext != right_ext:
            included_alt = (left + 1) if left_ext else (right + 1)
            skipped_exon = skipped_ids[0]
            mxe_skipped.add(skipped_exon)
            events.add("MXE")
            details.append(
                f"MXE:included_exon={included_alt}:skipped_exon={skipped_exon}"
            )
            continue
        events.add("SE")
        details.append(f"SE:skipped_exons={','.join(map(str, skipped_ids))}")

    return events, details, mxe_skipped


def join_as_events(events: set[str]) -> str:
    return ",".join(name for name in AS_EVENT_ORDER if name in events)


def classify_alternative_splicing_events(
    retained: list[int],
    alternative_junctions: list[tuple[int, int]],
    unannotated_junctions: list[tuple[int, int]],
    introns: list[tuple[int, int]],
    exons: list[tuple[int, int]],
    coverage_blocks: list[tuple[int, int]],
    read_start: int,
    read_end: int,
    strand: str,
    tolerance: int,
) -> tuple[str, str]:
    """Return read-level AS event labels: SE, RI, MXE, A5SS, A3SS."""
    events: set[str] = set()
    details: list[str] = []
    for intron_idx in retained:
        events.add("RI")
        details.append(f"RI:intron={intron_idx}")
    se_events, se_details, _mxe_skipped = classify_se_mxe_events(
        exons,
        coverage_blocks,
        read_start,
        read_end,
    )
    events.update(se_events)
    details.extend(se_details)
    for junction in alternative_junctions + unannotated_junctions:
        junction_events, junction_details = classify_junction_event(
            junction,
            introns,
            strand,
            tolerance,
        )
        events.update(junction_events)
        details.extend(junction_details)
    return join_as_events(events), ";".join(details)


def classify_intron_for_read(
    read: pysam.AlignedSegment,
    intron: tuple[int, int],
    tolerance: int,
    min_junction_fraction: float,
    min_anchor: int,
) -> str:
    """Return SPLICED, ALTERNATIVE_SPLICED, RETAINED, or NOT_COVERED."""
    i_start, i_end = intron
    r_start = read.reference_start
    r_end = read.reference_end
    if r_start > i_start or r_end < i_end:
        return "NOT_COVERED"

    blocks, junctions = parse_cigar_blocks_and_junctions(read)
    merged = merge_blocks(blocks)

    intron_junctions = [
        junction
        for junction in junctions
        if junction_splices_intron(junction, intron, min_junction_fraction)
    ]
    for junc in intron_junctions:
        if junction_matches_intron(junc, intron, tolerance):
            return "SPLICED"
    if intron_junctions:
        return "ALTERNATIVE_SPLICED"

    retained_start = max(0, i_start - min_anchor)
    retained_end = i_end + min_anchor
    if covers_interval(merged, retained_start, retained_end):
        # No matching N and continuous coverage across both splice-site anchors.
        for junc in junctions:
            if junction_splices_intron(junc, intron, min_junction_fraction):
                return "NOT_COVERED"
        return "RETAINED"

    return "NOT_COVERED"


def p5_distance_from_intron_5ss(
    p5_coord: int,
    intron: tuple[int, int],
    strand: str,
) -> int | None:
    """Distance from the intron 5' splice site to P5 (transcriptional)."""
    i_start, i_end = intron
    if i_end <= i_start:
        return None
    if strand == "+":
        dist = p5_coord - i_start
    else:
        dist = (i_end - 1) - p5_coord
    if dist < 0:
        return None
    return dist


def classify_read_splicing(
    read: pysam.AlignedSegment,
    gene: GeneModel,
    p5_feat: str,
    p3_feat: str,
    p5_coord: int,
    tolerance: int,
    min_junction_fraction: float,
    min_anchor: int,
    min_p5_intron_bp: int = DEFAULT_P5_INTRON_RETAIN_BP,
) -> tuple[str, str, str, str, str, str, str, str, str, str, str]:
    """Classify splicing; covered_introns come from P5/P3, evidence from CIGAR."""
    introns = gene.introns
    n_introns = len(introns)
    if n_introns == 0:
        return "0", "", "", "", "", "", "", "", "", "", "no_intron_covered"

    spliced: list[int] = []
    alternative_spliced: list[int] = []
    retained: list[int] = []
    p5_intron_idx = (
        int(p5_feat.split("_", 1)[1]) if p5_feat.startswith("intron_") else None
    )

    for idx, intron in enumerate(introns, start=1):
        status = classify_intron_for_read(
            read,
            intron,
            tolerance,
            min_junction_fraction,
            min_anchor,
        )
        if (
            p5_intron_idx == idx
            and status not in {"SPLICED", "ALTERNATIVE_SPLICED"}
        ):
            dist = p5_distance_from_intron_5ss(p5_coord, intron, gene.strand)
            if dist is not None and dist > min_p5_intron_bp:
                status = "RETAINED"
        if status == "SPLICED":
            spliced.append(idx)
        elif status == "ALTERNATIVE_SPLICED":
            spliced.append(idx)
            alternative_spliced.append(idx)
        elif status == "RETAINED":
            retained.append(idx)

    _blocks, junctions = parse_cigar_blocks_and_junctions(read)
    canonical_junctions = {
        junction
        for junction in junctions
        if any(
            junction_splices_intron(junction, intron, min_junction_fraction)
            and junction_matches_intron(junction, intron, tolerance)
            for intron in introns
        )
    }
    alternative_junctions = [
        junction
        for junction in junctions
        if junction not in canonical_junctions
        and any(
            junction_splices_intron(junction, intron, min_junction_fraction)
            for intron in introns
        )
    ]
    unannotated_junctions = [
        junction
        for junction in junctions
        if junction not in canonical_junctions
        and junction not in alternative_junctions
    ]

    def first_intron_after_p5(feature: str) -> int | None:
        if feature == "promoter":
            return 1
        if feature.startswith("exon_"):
            return int(feature.split("_", 1)[1])
        if feature.startswith("intron_"):
            return int(feature.split("_", 1)[1])
        return None

    def last_intron_before_p3(feature: str) -> int | None:
        if feature == "termination":
            return n_introns
        if feature.startswith("exon_"):
            return int(feature.split("_", 1)[1]) - 1
        if feature.startswith("intron_"):
            return int(feature.split("_", 1)[1]) - 1
        return None

    # covered_introns = introns spanned by P5→P3 (not CIGAR evidence alone)
    covered: list[int] = []
    first_covered = first_intron_after_p5(p5_feat)
    last_covered = last_intron_before_p3(p3_feat)
    if (
        first_covered is not None
        and last_covered is not None
        and first_covered <= last_covered
    ):
        covered = list(
            range(
                max(1, first_covered),
                min(n_introns, last_covered) + 1,
            )
        )

    expected = list(covered)
    expected_set = set(expected)
    covered_set = set(covered)
    spliced_set = set(spliced)
    retained_set = set(retained)
    relevant_noncanonical = [
        junction
        for junction in unannotated_junctions
        if any(
            junction_overlaps_intron_region(junction, introns[idx - 1])
            for idx in expected
        )
    ]

    # fully_spliced: every P5–P3 covered intron must be spliced (sets must match)
    if (
        covered
        and covered_set <= spliced_set
        and not retained
        and not relevant_noncanonical
    ):
        splicing_class = "fully_spliced"
    elif (
        covered
        and covered_set <= retained_set
        and not spliced
        and not relevant_noncanonical
    ):
        splicing_class = "fully_unspliced"
    elif expected and (
        expected_set.intersection(spliced_set | retained_set)
        or relevant_noncanonical
        or (covered_set - spliced_set - retained_set)
    ):
        splicing_class = "partial_spliced"
    elif not expected and not covered and not unannotated_junctions:
        splicing_class = "no_intron_covered"
    elif not expected and not covered and unannotated_junctions:
        splicing_class = "unannotated_junction_only"
    elif (
        retained
        and not spliced
        and not unannotated_junctions
    ):
        splicing_class = "fully_unspliced"
    elif (
        spliced
        or alternative_spliced
        or retained
        or alternative_junctions
        or unannotated_junctions
    ):
        splicing_class = "partial_spliced"
    else:
        splicing_class = "no_intron_covered"

    def join(nums: list[int]) -> str:
        return ",".join(str(x) for x in nums)

    def join_junctions(items: list[tuple[int, int]]) -> str:
        return ",".join(f"{start}-{end}" for start, end in items)

    alternative_splicing_events, alternative_splicing_details = (
        classify_alternative_splicing_events(
            retained,
            alternative_junctions,
            unannotated_junctions,
            introns,
            gene.exons,
            merge_blocks(_blocks),
            read.reference_start,
            read.reference_end,
            gene.strand,
            tolerance,
        )
    )

    return (
        str(n_introns),
        join(covered),
        join(spliced),
        join(alternative_spliced),
        join(retained),
        join(expected),
        join_junctions(alternative_junctions),
        join_junctions(unannotated_junctions),
        alternative_splicing_events,
        alternative_splicing_details,
        splicing_class,
    )


def classify_intron_termination(
    p3_feat: str,
    p3_relative_dist: int | None,
) -> tuple[bool, str]:
    """Report a read ending inside an intron without changing splicing status."""
    if (
        p3_relative_dist is None
        or p3_relative_dist <= DEFAULT_INTRON_END_THRESHOLD
        or not p3_feat.startswith("intron_")
    ):
        return False, ""
    return True, p3_feat.split("_", 1)[1]


def apply_p5_start_rule(splicing_class: str, p5_feat: str) -> str:
    """Assign a separate class when P5 is not in promoter or exon_1."""
    if p5_feat not in VALID_P5_FEATURES:
        return INVALID_P5_CLASS
    return splicing_class


def read_three_prime_end(read: pysam.AlignedSegment, gene_strand: str) -> int:
    if gene_strand == "+":
        return read.reference_end
    return read.reference_start


def annotate_readthrough(
    read: pysam.AlignedSegment,
    gene: GeneModel,
    pas: PasRecord | None,
) -> tuple[bool, int, str]:
    if pas is None:
        return False, 0, "NA"

    rt_end = read_three_prime_end(read, gene.strand)
    if gene.strand == "+":
        pas_site = pas.start
        is_rt = rt_end > pas.start
        dist = abs(rt_end - pas.start) if is_rt else 0
    else:
        pas_site = pas.end
        is_rt = rt_end < pas.end
        dist = abs(rt_end - pas.end) if is_rt else 0

    return is_rt, dist, str(pas_site)


def read_strand_label(read: pysam.AlignedSegment) -> str:
    return read_alignment_strand(read)


def _closed_half_open(start: int, end_exclusive: int) -> tuple[int, int] | None:
    """Convert 0-based half-open [start, end) to closed [start, end-1]; None if empty."""
    if end_exclusive <= start:
        return None
    return start, end_exclusive - 1


def promoter_termination_regions(gene: GeneModel, flank: int) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
    """Return (promoter, termination) as closed intervals in genomic coordinates."""
    if gene.strand == "+":
        promoter = _closed_half_open(max(0, gene.start - flank), gene.start)
        termination = _closed_half_open(gene.end, gene.end + flank)
    else:
        # promoter: (gene_end, gene_end + flank]
        prom_start = gene.end + 1
        prom_end = gene.end + flank
        promoter = (prom_start, prom_end) if prom_start <= prom_end else None
        # termination: (gene_start - flank, gene_start]
        term_start = max(0, gene.start - flank + 1)
        term_end = gene.start
        termination = (term_start, term_end) if term_start <= term_end else None
    return promoter, termination


def _in_closed(pos: int, interval: tuple[int, int] | None) -> bool:
    if interval is None:
        return False
    start, end = interval
    return start <= pos <= end


def classify_endpoint(pos: int, gene: GeneModel, flank: int) -> str:
    """Map a genomic coordinate to a gene feature label (closed-interval logic)."""
    # 1. exons (transcription order)
    for idx, (ex_start, ex_end) in enumerate(gene.exons, start=1):
        ex_closed = _closed_half_open(ex_start, ex_end)
        if ex_closed and _in_closed(pos, ex_closed):
            return f"exon_{idx}"

    # 2. introns
    for idx, (in_start, in_end) in enumerate(gene.introns, start=1):
        in_closed = _closed_half_open(in_start, in_end)
        if in_closed and _in_closed(pos, in_closed):
            return f"intron_{idx}"

    promoter, termination = promoter_termination_regions(gene, flank)

    # 3. promoter
    if _in_closed(pos, promoter):
        return "promoter"

    # 4. termination
    if _in_closed(pos, termination):
        return "termination"

    # 5. intergenic
    return "intergenic"


def feature_closed_interval(
    feat: str,
    gene: GeneModel,
    flank: int,
) -> tuple[int, int] | None:
    """Return closed [start, end] for exon_i, intron_i, promoter, or termination."""
    if feat.startswith("exon_"):
        idx = int(feat.split("_", 1)[1])
        if 1 <= idx <= len(gene.exons):
            ex_start, ex_end = gene.exons[idx - 1]
            return _closed_half_open(ex_start, ex_end)
        return None
    if feat.startswith("intron_"):
        idx = int(feat.split("_", 1)[1])
        if 1 <= idx <= len(gene.introns):
            in_start, in_end = gene.introns[idx - 1]
            return _closed_half_open(in_start, in_end)
        return None
    promoter, termination = promoter_termination_regions(gene, flank)
    if feat == "promoter":
        return promoter
    if feat == "termination":
        return termination
    return None


def _non_negative_dist(value: int) -> int:
    return max(0, value)


def compute_p5_relative_dist(
    p5: int,
    feat: str,
    gene: GeneModel,
    flank: int,
) -> int | None:
    if feat == "intergenic":
        return None
    if feat == "promoter":
        if not gene.exons:
            return None
        ex1_start, ex1_end = gene.exons[0]
        ex1_end_closed = ex1_end - 1
        if gene.strand == "+":
            return _non_negative_dist(ex1_start - p5)
        return _non_negative_dist(p5 - ex1_end_closed)

    interval = feature_closed_interval(feat, gene, flank)
    if interval is None:
        return None
    feat_start, feat_end = interval
    if gene.strand == "+":
        return _non_negative_dist(feat_end - p5)
    return _non_negative_dist(p5 - feat_start)


def compute_p3_relative_dist(
    p3: int,
    feat: str,
    gene: GeneModel,
    flank: int,
) -> int | None:
    if feat == "intergenic":
        return None
    if feat == "termination":
        if not gene.exons:
            return None
        last_start, last_end = gene.exons[-1]
        last_end_closed = last_end - 1
        if gene.strand == "+":
            return _non_negative_dist(p3 - last_end_closed)
        return _non_negative_dist(last_start - p3)

    interval = feature_closed_interval(feat, gene, flank)
    if interval is None:
        return None
    feat_start, feat_end = interval
    if gene.strand == "+":
        return _non_negative_dist(p3 - feat_start)
    return _non_negative_dist(feat_end - p3)


def read_endpoint_annotations(
    read: pysam.AlignedSegment,
    gene: GeneModel,
    flank: int,
) -> dict[str, int | str | None]:
    """Step 5–6: strand-relative P5/P3 labels with relative distances."""
    left_coord = read.reference_start
    right_coord = read.reference_end - 1

    if read.is_reverse:
        p5_coord = right_coord
        p3_coord = left_coord
    else:
        p5_coord = left_coord
        p3_coord = right_coord
    p5_feat = classify_endpoint(p5_coord, gene, flank)
    p3_feat = classify_endpoint(p3_coord, gene, flank)

    return {
        "p5_coord": p5_coord,
        "p5_feat": p5_feat,
        "p5_relative_dist": compute_p5_relative_dist(p5_coord, p5_feat, gene, flank),
        "p3_coord": p3_coord,
        "p3_feat": p3_feat,
        "p3_relative_dist": compute_p3_relative_dist(p3_coord, p3_feat, gene, flank),
    }


def process_gene(
    bam: pysam.AlignmentFile,
    gene: GeneModel,
    pas: PasRecord | None,
    library: str,
    flank: int,
    tolerance: int,
    min_junction_fraction: float,
    min_anchor: int,
    min_mapq: int,
    min_p5_intron_bp: int = DEFAULT_P5_INTRON_RETAIN_BP,
) -> list[dict]:
    win_start = max(0, gene.start - flank)
    win_end = gene.end + flank
    rows: list[dict] = []
    seen: set[str] = set()

    try:
        fetched = bam.fetch(gene.chrom, win_start, win_end)
    except ValueError:
        return rows

    for read in fetched:
        if (
            read.is_unmapped
            or read.is_secondary
            or read.is_supplementary
            or read.is_duplicate
            or read.is_qcfail
            or read.mapping_quality < min_mapq
        ):
            continue
        if read.query_name in seen:
            continue
        if not overlaps_gene(read, gene):
            continue
        if not read_matches_gene_strand(read, gene.strand):
            continue
        if not read_matches_library_strand(read, gene.strand, library):
            continue
        seen.add(read.query_name)

        is_rt, rt_dist, pas_site = annotate_readthrough(read, gene, pas)
        endpoints = read_endpoint_annotations(read, gene, flank)
        (
            n_introns,
            covered,
            spliced,
            alternative_spliced,
            retained,
            expected,
            alternative_junctions,
            unannotated_junctions,
            alternative_splicing_events,
            alternative_splicing_details,
            splicing_class,
        ) = classify_read_splicing(
            read,
            gene,
            endpoints["p5_feat"],
            endpoints["p3_feat"],
            endpoints["p5_coord"],
            tolerance,
            min_junction_fraction,
            min_anchor,
            min_p5_intron_bp,
        )
        intron_terminated, intron_terminated_id = classify_intron_termination(
            endpoints["p3_feat"],
            endpoints["p3_relative_dist"],
        )
        splicing_class = apply_p5_start_rule(
            splicing_class,
            endpoints["p5_feat"],
        )

        rows.append(
            {
                "read_id": read.query_name,
                "gene_id": gene.gene_id,
                "chrom": gene.chrom,
                "read_start": read.reference_start,
                "read_end": read.reference_end,
                "strand": read_strand_label(read),
                "n_introns": n_introns,
                "covered_introns": covered,
                "spliced_introns": spliced,
                "alternative_spliced_introns": alternative_spliced,
                "retained_introns": retained,
                "expected_introns": expected,
                "alternative_junctions": alternative_junctions,
                "unannotated_junctions": unannotated_junctions,
                "alternative_splicing_events": alternative_splicing_events,
                "alternative_splicing_details": alternative_splicing_details,
                "splicing_class": splicing_class,
                "is_readthrough": is_rt,
                "readthrough_dist": rt_dist,
                "pas_site": pas_site,
                **endpoints,
                "intron_terminated": intron_terminated,
                "intron_terminated_id": intron_terminated_id,
            }
        )
    return rows


def write_rows(path: str, rows: list[dict]) -> None:
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_FIELDS, delimiter="\t")
        writer.writeheader()
        for row in rows:
            out = dict(row)
            for key in ("is_readthrough", "intron_terminated"):
                out[key] = str(out[key]).lower()
            for key in ("p5_relative_dist", "p3_relative_dist"):
                if out[key] is None:
                    out[key] = "NA"
            writer.writerow(out)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify BAM reads by splicing status and readthrough per gene.",
    )
    parser.add_argument("--bam", required=True, help="Sorted, indexed BAM file")
    parser.add_argument(
        "--annotation",
        default=str(_default_annotation()),
        help="Gene annotation: exon/intron BED6 (default), BED12, or GTF/GFF",
    )
    parser.add_argument(
        "--pas",
        default=str(_default_pas()),
        help="PAS annotation BED6 (default: data/refs/Ath.PAS.completed.bed)",
    )
    parser.add_argument(
        "--library",
        choices=("fr-firststrand", "fr-secondstrand", "unstranded"),
        default="unstranded",
        help="Stranded library type (default: unstranded)",
    )
    parser.add_argument(
        "--flank",
        type=int,
        default=10000,
        help="Flank size (bp) around gene for fetch window (default: 10000)",
    )
    parser.add_argument(
        "--tolerance",
        type=int,
        default=5,
        help="Canonical splice-site coordinate tolerance in bp (default: 5)",
    )
    parser.add_argument(
        "--min-junction-fraction",
        type=float,
        default=0.8,
        help="Minimum fraction of an intron removed by a junction (default: 0.8)",
    )
    parser.add_argument(
        "--min-anchor",
        type=int,
        default=DEFAULT_MIN_ANCHOR,
        help="Minimum continuous aligned bases on each side for retention (default: 10)",
    )
    parser.add_argument(
        "--min-p5-intron-bp",
        type=int,
        default=DEFAULT_P5_INTRON_RETAIN_BP,
        help=(
            "If P5 is in intron k and more than this many bp past the intron "
            "5'SS, count intron k as retained (default: 10)"
        ),
    )
    parser.add_argument(
        "--min-mapq",
        type=int,
        default=DEFAULT_MIN_MAPQ,
        help="Minimum alignment MAPQ (default: 20)",
    )
    parser.add_argument("--output", required=True, help="Output TSV path")
    args = parser.parse_args()

    for name in ("tolerance", "min_anchor", "min_p5_intron_bp", "min_mapq"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be >= 0")
    if not 0 < args.min_junction_fraction <= 1:
        parser.error("--min-junction-fraction must be > 0 and <= 1")

    if not os.path.isfile(args.bam):
        sys.exit(f"Error: BAM not found: {args.bam}")
    if not os.path.isfile(args.annotation):
        sys.exit(f"Error: annotation not found: {args.annotation}")
    if not os.path.isfile(args.pas):
        sys.exit(f"Error: PAS BED not found: {args.pas}")

    ann_fmt = _detect_annotation_format(args.annotation)
    print(f"Loading annotation ({ann_fmt}): {args.annotation}", file=sys.stderr)
    genes = load_annotation(args.annotation)
    print(f"Loaded {len(genes):,} gene models", file=sys.stderr)

    print(f"Loading PAS: {args.pas}", file=sys.stderr)
    pas_by_chrom = index_pas_by_chrom(load_pas_bed(args.pas))

    all_rows: list[dict] = []
    bam = pysam.AlignmentFile(args.bam, "rb")

    try:
        for i, gene in enumerate(genes, start=1):
            pas = find_downstream_pas(gene, pas_by_chrom)
            rows = process_gene(
                bam,
                gene,
                pas,
                args.library,
                args.flank,
                args.tolerance,
                args.min_junction_fraction,
                args.min_anchor,
                args.min_mapq,
                args.min_p5_intron_bp,
            )
            all_rows.extend(rows)
            if i % 100 == 0 or i == len(genes):
                print(
                    f"Processed {i:,}/{len(genes):,} genes | "
                    f"{len(all_rows):,} read-gene rows",
                    file=sys.stderr,
                )
    finally:
        bam.close()

    write_rows(args.output, all_rows)
    print(f"Wrote {len(all_rows):,} rows to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
