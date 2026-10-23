#!/usr/bin/env python3
"""Consecutive intron retention counts and figures.

A read is consecutive intron retention when alternative_splicing_details
contains neighboring RI:intron=k and RI:intron=k+1. Reads with
splicing_class invalid_p5_start are excluded. The numerator is labelled
valid-P5 reads (at least one AS event type). Pair and run-length percentages
among consecutive-RI reads use that numerator. The library bars use all
valid-P5 reads as the denominator.

Outputs go to results/alternative_splicing/ and results/gene_reads/.

Usage:
    python src/cold_cui/count_consecutive_intron_retention.py
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

import _provenance as _prov  # noqa: E402

_prov.start(__file__)


# %% cell 1
_prov.mark(1)
from pathlib import Path
import sys

SRC_DIR = Path.cwd()
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import importlib
import project_config
importlib.reload(project_config)
from project_config import (
    REF_DIR,
    RESULTS_DIR,
    SAMPLES_TSV,
    load_sample_table,
    validate_inputs,
    SAMPLES,
    BAM_BY_SAMPLE,
    RESULT_DIRS,
    ensure_result_dirs,
    classify_tsv,
)

ensure_result_dirs()
samples_df = load_sample_table(SAMPLES_TSV)
input_errors = validate_inputs()
if input_errors:
    raise FileNotFoundError("\n".join(input_errors))
print("Samples:")
print(samples_df.to_string(index=False))
print(f"Results: {RESULTS_DIR}")

import numpy as np
import pandas as pd
from plotnine import *

try:
    from IPython.display import display
except ImportError:
    def display(obj=None, **kwargs):
        return obj

AS_DIR = RESULT_DIRS["alternative_splicing"]
AS_DIR.mkdir(parents=True, exist_ok=True)
GENE_READS_DIR = RESULT_DIRS["gene_reads"]
GENE_READS_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_GROUPS = {
    "Warm": [s for s in SAMPLES if str(s).startswith("Warm")],
    "Cold": [s for s in SAMPLES if str(s).startswith("Cold")],
}
GROUP_BY_SAMPLE = {s: g for g, reps in SAMPLE_GROUPS.items() for s in reps}
Warm_COLOR = "#29A2C6"
Cold_COLOR = "#FFCB18"
AS_EVENT_ORDER = ("SE", "RI", "MXE", "A5SS", "A3SS")
EVENT_MAP = {"IR": "RI", "ES": "SE"}
EXCLUDED_CLASSES = frozenset({"invalid_p5_start"})
PLOT_DPI = 600

theme_common = (
    theme_bw()
    + theme(
        axis_text=element_text(size=10, color="black"),
        axis_title=element_text(size=12, color="black"),
        panel_grid=element_blank(),
        legend_title=element_text(size=10),
        legend_text=element_text(size=9),
        plot_title=element_text(size=11),
        strip_background=element_rect(fill="white", color="black"),
    )
)


def save_gg(p, stem, width, height):
    stem = Path(stem)
    p.save(Path(str(stem) + ".png"), width=width, height=height, dpi=PLOT_DPI, verbose=False)
    p.save(Path(str(stem) + ".pdf"), width=width, height=height, verbose=False)
    print(f"saved {stem.name}.png/.pdf")
    return p


CLASSIFY_COLS = [
    "read_id",
    "gene_id",
    "splicing_class",
    "alternative_splicing_events",
    "alternative_splicing_details",
]


def parse_event_types(value) -> list[str]:
    if pd.isna(value):
        return []
    seen = []
    for part in str(value).split(","):
        token = EVENT_MAP.get(part.strip(), part.strip())
        if token in AS_EVENT_ORDER and token not in seen:
            seen.append(token)
    return seen


def load_labelled_reads(sample: str) -> pd.DataFrame:
    """Valid-P5 reads that carry at least one alternative-splicing type."""
    path = classify_tsv(sample)
    df = pd.read_csv(path, sep="\t", usecols=CLASSIFY_COLS, dtype=str)
    df = df.loc[~df["splicing_class"].isin(EXCLUDED_CLASSES)].copy()
    df["gene_id"] = df["gene_id"].fillna("").astype(str)
    df["event_types"] = df["alternative_splicing_events"].map(parse_event_types)
    df["sample"] = sample
    df["group"] = GROUP_BY_SAMPLE[sample]
    valid_n_by_sample[sample] = len(df)
    labelled = df.loc[df["event_types"].map(len) > 0].copy()
    print(f"[{sample}] valid-P5 reads={len(df):,}; labelled={len(labelled):,}")
    return labelled


valid_n_by_sample = {}
labelled_by_sample = {sample: load_labelled_reads(sample) for sample in SAMPLES}


# ## 2. Consecutive intron retention
#
# Combo plots collapse every `RI:intron=` token into one type `RI`, so they hide whether two **adjacent** introns are both retained on the same molecule.
#
# Here a read has consecutive RI if `alternative_splicing_details` contains neighbouring indices (`k` and `k+1`). That matches `retained_introns`. Counts exclude `invalid_p5_start`. Pair and run-length percentages are among consecutive-RI reads; a second run-length plot and the P3-offset plot use the valid-P5 library as the denominator. The summary also reports the share of valid-P5 reads and of RI reads.


# %% cell 2
_prov.mark(2)
# =============================================================================
# Consecutive intron retention: neighboring introns retained on the same molecule
# =============================================================================
import re
from collections import Counter

RI_INTRON_RE = re.compile(r"RI:intron=(\d+)")
PAIR_KEEP = [(i, i + 1) for i in range(1, 9)]
RUN_BINS = ("2", "3", "4", "5", "≥6")


def parse_ri_introns(value) -> list[int]:
    if pd.isna(value):
        return []
    return sorted({int(idx) for idx in RI_INTRON_RE.findall(str(value))})


def consecutive_pairs(idxs) -> list[tuple[int, int]]:
    ordered = sorted(set(idxs))
    return [(a, b) for a, b in zip(ordered, ordered[1:]) if b == a + 1]


def longest_run(idxs) -> int:
    ordered = sorted(set(idxs))
    if not ordered:
        return 0
    best = run = 1
    for a, b in zip(ordered, ordered[1:]):
        if b == a + 1:
            run += 1
            best = max(best, run)
        else:
            run = 1
    return best


def run_bin(length: int) -> str:
    if length >= 6:
        return "≥6"
    return str(length)


def count_valid_p5(sample: str) -> int:
    n_valid = 0
    for chunk in pd.read_csv(
        classify_tsv(sample),
        sep="\t",
        usecols=["splicing_class"],
        dtype=str,
        chunksize=200_000,
    ):
        n_valid += int((~chunk["splicing_class"].isin(EXCLUDED_CLASSES)).sum())
    return n_valid


summary_rows = []
pair_rows = []
run_rows = []
example_rows = []

valid_by_sample = {}
for sample in SAMPLES:
    if sample in valid_n_by_sample:
        n_valid = int(valid_n_by_sample[sample])
    else:
        n_valid = count_valid_p5(sample)
    valid_by_sample[sample] = n_valid
    lab = labelled_by_sample[sample]
    n_ri = n_ri_multi = n_consec = max_run = 0
    pair_n = Counter()
    run_n = Counter()
    for gene_id, details in zip(lab["gene_id"], lab["alternative_splicing_details"]):
        ri_idxs = parse_ri_introns(details)
        if not ri_idxs:
            continue
        n_ri += 1
        if len(ri_idxs) >= 2:
            n_ri_multi += 1
        pairs = consecutive_pairs(ri_idxs)
        if not pairs:
            continue
        n_consec += 1
        run = longest_run(ri_idxs)
        max_run = max(max_run, run)
        run_n[run] += 1
        for pair in pairs:
            pair_n[pair] += 1
        if len(example_rows) < 12:
            example_rows.append(
                {
                    "sample": sample,
                    "gene_id": gene_id,
                    "ri_introns": ",".join(str(i) for i in ri_idxs),
                    "n_consecutive_pairs": len(pairs),
                    "longest_run": run,
                    "alternative_splicing_details": str(details),
                }
            )
    group = GROUP_BY_SAMPLE[sample]
    summary_rows.append(
        {
            "sample": sample,
            "group": group,
            "n_valid": n_valid,
            "n_ri": n_ri,
            "n_ri_multi": n_ri_multi,
            "n_consec": n_consec,
            "max_run": max_run,
            "pct_valid_consec": 100.0 * n_consec / n_valid if n_valid else np.nan,
            "pct_ri_consec": 100.0 * n_consec / n_ri if n_ri else np.nan,
            "pct_multi_consec": 100.0 * n_consec / n_ri_multi if n_ri_multi else np.nan,
        }
    )
    for pair, n in pair_n.items():
        pair_rows.append(
            {
                "sample": sample,
                "group": group,
                "intron_a": pair[0],
                "intron_b": pair[1],
                "pair": f"{pair[0]}-{pair[1]}",
                "n_reads": n,
                "percent": 100.0 * n / n_consec if n_consec else np.nan,
            }
        )
    for length, n in run_n.items():
        run_rows.append(
            {
                "sample": sample,
                "group": group,
                "run_length": length,
                "run_bin": run_bin(length),
                "n_reads": n,
                "percent": 100.0 * n / n_consec if n_consec else np.nan,
            }
        )
    print(
        f"[{sample}] valid={n_valid:,}; RI={n_ri:,}; RI≥2={n_ri_multi:,}; "
        f"consecutive={n_consec:,} ({100.0 * n_consec / n_valid:.1f}% valid, "
        f"{100.0 * n_consec / n_ri:.1f}% RI); max run={max_run}"
    )

consec_summary = pd.DataFrame(summary_rows)
consec_pairs = pd.DataFrame(pair_rows)
consec_runs = pd.DataFrame(run_rows)
consec_examples = pd.DataFrame(example_rows)

consec_summary.to_csv(AS_DIR / "consecutive_ri.by_sample.tsv", sep="\t", index=False)
consec_pairs.to_csv(AS_DIR / "consecutive_ri.pairs.by_replicate.tsv", sep="\t", index=False)
consec_runs.to_csv(AS_DIR / "consecutive_ri.run_length.by_replicate.tsv", sep="\t", index=False)
consec_examples.to_csv(AS_DIR / "consecutive_ri.examples.tsv", sep="\t", index=False)
print("\nPer-sample consecutive RI:")
print(consec_summary.to_string(index=False))
display(consec_examples)


# %% cell 3
_prov.mark(3)
pct_long = consec_summary.melt(
    id_vars=["sample", "group"],
    value_vars=["pct_valid_consec", "pct_ri_consec"],
    var_name="metric",
    value_name="percent",
)
pct_long["metric"] = pct_long["metric"].map(
    {"pct_valid_consec": "of valid P5", "pct_ri_consec": "of RI reads"}
)
pct_mean = (
    pct_long.groupby(["group", "metric"], as_index=False)["percent"].mean()
)
pct_mean["group"] = pd.Categorical(pct_mean["group"], categories=("Warm", "Cold"), ordered=True)
pct_mean["metric"] = pd.Categorical(
    pct_mean["metric"], categories=("of valid P5", "of RI reads"), ordered=True
)
pct_mean.to_csv(AS_DIR / "consecutive_ri.percent.mean.tsv", sep="\t", index=False)

p_pct = (
    ggplot(pct_mean, aes(x="metric", y="percent", fill="group"))
    + geom_col(position=position_dodge(width=0.8), width=0.7, color="black", size=0.2)
    + scale_fill_manual(values={"Warm": Warm_COLOR, "Cold": Cold_COLOR})
    + labs(x="", y="Consecutive-RI reads (%)", fill="")
    + theme_common
    + theme(legend_position="right")
)
save_gg(p_pct, AS_DIR / "consecutive_ri.percent.bar", width=4.4, height=3.4)
display(p_pct)

pair_keep_labels = [f"{a}-{b}" for a, b in PAIR_KEEP]
pair_filled_rows = []
for sample in SAMPLES:
    group = GROUP_BY_SAMPLE[sample]
    sub = consec_pairs.loc[consec_pairs["sample"].eq(sample)]
    for pair in pair_keep_labels:
        hit = sub.loc[sub["pair"].eq(pair)]
        pair_filled_rows.append(
            {
                "sample": sample,
                "group": group,
                "pair": pair,
                "percent": float(hit["percent"].sum()) if len(hit) else 0.0,
                "n_reads": int(hit["n_reads"].sum()) if len(hit) else 0,
            }
        )
pair_filled = pd.DataFrame(pair_filled_rows)
pair_mean = (
    pair_filled.groupby(["group", "pair"], as_index=False)
    .agg(percent=("percent", "mean"), n_reads=("n_reads", "sum"))
)
pair_mean["pair"] = pd.Categorical(pair_mean["pair"], categories=pair_keep_labels, ordered=True)
pair_mean["group"] = pd.Categorical(pair_mean["group"], categories=("Warm", "Cold"), ordered=True)
pair_mean.to_csv(AS_DIR / "consecutive_ri.pairs.mean.tsv", sep="\t", index=False)

p_pair = (
    ggplot(pair_mean, aes(x="pair", y="percent", fill="group"))
    + geom_col(position=position_dodge(width=0.8), width=0.7, color="black", size=0.2)
    + scale_fill_manual(values={"Warm": Warm_COLOR, "Cold": Cold_COLOR})
    + labs(x="Adjacent retained introns", y="Consecutive-RI reads (%)", fill="")
    + theme_common
    + theme(legend_position="right")
)
save_gg(p_pair, AS_DIR / "consecutive_ri.pairs.bar", width=6.2, height=3.5)
display(p_pair)

run_filled_rows = []
for sample in SAMPLES:
    group = GROUP_BY_SAMPLE[sample]
    sub = consec_runs.loc[consec_runs["sample"].eq(sample)]
    for bin_name in RUN_BINS:
        run_filled_rows.append(
            {
                "sample": sample,
                "group": group,
                "run_bin": bin_name,
                "percent": float(sub.loc[sub["run_bin"].eq(bin_name), "percent"].sum()),
            }
        )
run_filled = pd.DataFrame(run_filled_rows)
run_mean = run_filled.groupby(["group", "run_bin"], as_index=False)["percent"].mean()
run_mean["run_bin"] = pd.Categorical(run_mean["run_bin"], categories=RUN_BINS, ordered=True)
run_mean["group"] = pd.Categorical(run_mean["group"], categories=("Warm", "Cold"), ordered=True)
run_mean.to_csv(AS_DIR / "consecutive_ri.run_length.mean.tsv", sep="\t", index=False)

p_run = (
    ggplot(run_mean, aes(x="run_bin", y="percent", fill="group"))
    + geom_col(position=position_dodge(width=0.8), width=0.7, color="black", size=0.2)
    + scale_fill_manual(values={"Warm": Warm_COLOR, "Cold": Cold_COLOR})
    + labs(x="Longest consecutive RI run", y="Consecutive-RI reads (%)", fill="")
    + theme_common
    + theme(legend_position="right")
)
save_gg(p_run, AS_DIR / "consecutive_ri.run_length.bar", width=4.6, height=3.4)
display(p_run)

run_lib_rows = []
for sample in SAMPLES:
    group = GROUP_BY_SAMPLE[sample]
    n_valid = float(consec_summary.loc[consec_summary["sample"].eq(sample), "n_valid"].iloc[0])
    sub = consec_runs.loc[consec_runs["sample"].eq(sample)]
    for bin_name in RUN_BINS:
        n_reads = float(sub.loc[sub["run_bin"].eq(bin_name), "n_reads"].sum())
        run_lib_rows.append(
            {
                "sample": sample,
                "group": group,
                "run_bin": bin_name,
                "n_reads": n_reads,
                "n_valid": n_valid,
                "percent": 100.0 * n_reads / n_valid if n_valid else float("nan"),
            }
        )
run_lib = pd.DataFrame(run_lib_rows)
run_lib_mean = run_lib.groupby(["group", "run_bin"], as_index=False)["percent"].mean()
run_lib_mean["run_bin"] = pd.Categorical(run_lib_mean["run_bin"], categories=RUN_BINS, ordered=True)
run_lib_mean["group"] = pd.Categorical(run_lib_mean["group"], categories=("Warm", "Cold"), ordered=True)
run_lib.to_csv(AS_DIR / "consecutive_ri.run_length.library.by_replicate.tsv", sep="\t", index=False)
run_lib_mean.to_csv(AS_DIR / "consecutive_ri.run_length.library.mean.tsv", sep="\t", index=False)

p_run_lib = (
    ggplot(run_lib_mean, aes(x="run_bin", y="percent", fill="group"))
    + geom_col(position=position_dodge(width=0.8), width=0.7, color="black", size=0.2)
    + scale_fill_manual(values={"Warm": Warm_COLOR, "Cold": Cold_COLOR})
    + scale_y_continuous(expand=(0, 0), limits=(0, 6))
    + labs(x="Longest consecutive RI run", y="Valid P5 reads (%)", fill="")
    + theme_common
    + theme(legend_position="right")
)
save_gg(p_run_lib, AS_DIR / "consecutive_ri.run_length.library.bar", width=4.6, height=3.4)
display(p_run_lib)

import plot_consecutive_ri_p3_offset as p3_offset

p3_lib_mean = p3_offset.write_outputs(
    [(sample, classify_tsv(sample)) for sample in SAMPLES],
    AS_DIR,
)
display(p3_lib_mean)

ri_lib = consec_summary[["sample", "group", "n_valid", "n_ri", "n_consec"]].copy()
ri_lib["RI reads"] = 100.0 * ri_lib["n_ri"] / ri_lib["n_valid"]
ri_lib["consecutive RI"] = 100.0 * ri_lib["n_consec"] / ri_lib["n_valid"]
ri_lib_long = ri_lib.melt(
    id_vars=["sample", "group", "n_valid", "n_ri", "n_consec"],
    value_vars=["RI reads", "consecutive RI"],
    var_name="metric",
    value_name="percent",
)
ri_lib_mean = ri_lib_long.groupby(["group", "metric"], as_index=False)["percent"].mean()
ri_lib_mean["metric"] = pd.Categorical(
    ri_lib_mean["metric"], categories=("RI reads", "consecutive RI"), ordered=True
)
ri_lib_mean["group"] = pd.Categorical(ri_lib_mean["group"], categories=("Warm", "Cold"), ordered=True)
ri_lib_long.to_csv(AS_DIR / "consecutive_ri.ri_reads.library.by_sample.tsv", sep="\t", index=False)
ri_lib_mean.to_csv(AS_DIR / "consecutive_ri.ri_reads.library.mean.tsv", sep="\t", index=False)

p_ri_lib = (
    ggplot(ri_lib_mean, aes(x="metric", y="percent", fill="group"))
    + geom_col(position=position_dodge(width=0.8), width=0.7, color="black", size=0.2)
    + scale_fill_manual(values={"Warm": Warm_COLOR, "Cold": Cold_COLOR})
    + scale_y_continuous(expand=(0, 0), limits=(0, 30))
    + labs(x="", y="Valid P5 reads (%)", fill="")
    + theme_common
    + theme(legend_position="right")
)
save_gg(p_ri_lib, AS_DIR / "consecutive_ri.ri_reads.library.bar", width=4.4, height=3.4)
display(p_ri_lib)
display(ri_lib_mean)


# ## 4. Gene-reads track: intron-retention molecules only
#
# Warm on top, Cold below. Only classified reads that carry `RI:intron=` (exclude `invalid_p5_start`). Each group pools the three replicates, then keeps up to `RI_MAX_READS` molecules. Color follows the gene-reads convention: blue = no CIGAR junction, red = at least one junction.


# %% cell 4
_prov.mark(4)
# RI-only gene reads: Warm on top, Cold below
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator

from plot_classified_gene_reads import load_pas_hit, read_gene_model
from plot_gene_reads import (
    _draw_pas_3end_vline,
    _fetch_reads_for_locus,
    _plot_reads_on_ax,
    _sort_reads_for_strand,
)

RI_EXAMPLE_GENE = None  # None picks the gene with the most RI reads in both Warm and Cold
RI_MAX_READS = 50
RI_MIN_BOTH_GROUPS = 20


def _has_ri(details) -> bool:
    return (not pd.isna(details)) and ("RI:intron=" in str(details))


def _ri_counts_by_group() -> pd.DataFrame:
    rows = []
    for sample, df in labelled_by_sample.items():
        group = GROUP_BY_SAMPLE[sample]
        ri = df.loc[df["alternative_splicing_details"].map(_has_ri)]
        if ri.empty:
            continue
        vc = ri.groupby("gene_id").size()
        for gene_id, n in vc.items():
            rows.append({"gene_id": gene_id, "group": group, "n_ri": int(n)})
    if not rows:
        return pd.DataFrame(columns=["gene_id", "Warm", "Cold"])
    tall = pd.DataFrame(rows).groupby(["gene_id", "group"], as_index=False)["n_ri"].sum()
    return tall.pivot(index="gene_id", columns="group", values="n_ri").fillna(0).astype(int)


def _pick_ri_gene() -> str:
    if RI_EXAMPLE_GENE:
        return str(RI_EXAMPLE_GENE)
    if "EXAMPLE_GENE" in globals() and EXAMPLE_GENE:
        return str(EXAMPLE_GENE)
    counts = _ri_counts_by_group()
    if counts.empty or "Warm" not in counts.columns or "Cold" not in counts.columns:
        raise ValueError("No RI reads found in labelled_by_sample")
    both = counts.loc[(counts["Warm"] >= RI_MIN_BOTH_GROUPS) & (counts["Cold"] >= RI_MIN_BOTH_GROUPS)].copy()
    both["total"] = both["Warm"] + both["Cold"]
    return str(both.sort_values("total", ascending=False).index[0])


def _ri_ids_for_gene(gene_id: str) -> dict[str, dict[str, set[str]]]:
    """Return {group: {sample: set(read_id)}} for RI reads on this gene."""
    out = {"Warm": {}, "Cold": {}}
    for sample, df in labelled_by_sample.items():
        group = GROUP_BY_SAMPLE[sample]
        sub = df.loc[df["gene_id"].eq(gene_id) & df["alternative_splicing_details"].map(_has_ri)]
        out[group][sample] = set(sub["read_id"].astype(str))
    return out


def _fetch_group_ri_reads(gene_id: str, group: str, id_map: dict[str, set[str]], chrom, start, end, strand):
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


ri_gene = _pick_ri_gene()
bed_path = REF_DIR / "exon_intron_pos.chr.bed"
pas_path = REF_DIR / "Ath.PAS.completed.bed"
gene_model = read_gene_model(ri_gene, bed_path)
chrom = str(gene_model["chrom"].iloc[0])
gene_start = int(gene_model["start"].min())
gene_end = int(gene_model["end"].max())
strand = str(gene_model["strand"].iloc[0])
ri_ids = _ri_ids_for_gene(ri_gene)
nv_reads = _fetch_group_ri_reads(ri_gene, "Warm", ri_ids["Warm"], chrom, gene_start, gene_end, strand)
c5_reads = _fetch_group_ri_reads(ri_gene, "Cold", ri_ids["Cold"], chrom, gene_start, gene_end, strand)
print(
    f"{ri_gene}  strand={strand}  "
    f"Warm RI ids={sum(len(v) for v in ri_ids['Warm'].values()):,} plotted={len(nv_reads)}  "
    f"Cold RI ids={sum(len(v) for v in ri_ids['Cold'].values()):,} plotted={len(c5_reads)}"
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
    title=f"{ri_gene}   Warm  intron-retention reads",
)
_plot_reads_on_ax(
    axes[1], c5_reads, gene_segments, min_pos, max_pos, boundary_bp,
    title="Cold  intron-retention reads",
)
_draw_ri_gene_model(axes[2], gene_model, min_pos, max_pos, boundary_bp)
pas_hit = load_pas_hit(ri_gene, pas_path)
if pas_hit is not None:
    _draw_pas_3end_vline(list(axes), pas_hit[0], pas_hit[1], pas_hit[2])
if strand == "-":
    for ax in axes:
        ax.invert_xaxis()
fig.subplots_adjust(hspace=0.04)
stem = GENE_READS_DIR / f"{ri_gene}_Warm_Cold.ri_gene_reads"
fig.savefig(f"{stem}.png", dpi=PLOT_DPI, bbox_inches="tight")
fig.savefig(f"{stem}.pdf", bbox_inches="tight")
print(f"saved {stem.name}.png/.pdf")
plt.show()

plotted_rows = []
for group, recs in (("Warm", nv_reads), ("Cold", c5_reads)):
    for rec in recs:
        plotted_rows.append(
            {"group": group, "sample": rec["sample"], "read_id": rec["read_id"], "start": rec["start"], "end": rec["end"]}
        )
pd.DataFrame(plotted_rows).to_csv(AS_DIR / f"{ri_gene}_Warm_Cold.ri_gene_reads.tsv", sep="\t", index=False)
