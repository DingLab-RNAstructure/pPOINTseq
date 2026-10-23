#!/usr/bin/env python3
"""First-intron splicing ratio versus distance past the first 3' splice site.

Run from anywhere: ``python src/first_intron/first_splicing_dynamics.py``. Outputs go to ``results/``;
``logs/provenance/first_splicing_dynamics.tsv`` records which cell wrote each result file.
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
from IPython.display import display  # noqa: E402,F401

_prov.start(__file__)


# First-intron splicing ratio versus distance past the first 3' splice site.


# %% cell 1
_prov.mark(1)
# =============================================================================
# Project paths and sample sheet
# 1) To change BAMs or samples, edit configs/samples.tsv.
#    POINTSEQ_SAMPLE_SHEET / POINTSEQ_RESULTS_DIR switch the analysis batch.
# 2) Thresholds: edit the notebook parameters in the next cell.
# =============================================================================
from pathlib import Path
import sys

SRC_DIR = Path.cwd()
if not (SRC_DIR / "project_config.py").exists():
    SRC_DIR = SRC_DIR / "src_newsplicing" if (SRC_DIR / "src_newsplicing" / "project_config.py").exists() else SRC_DIR
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import importlib
import project_config
importlib.reload(project_config)
from project_config import (
    WORK_DIR,
    CONFIG_DIR,
    DATA_DIR,
    BAM_DIR,
    REF_DIR,
    RESULTS_DIR,
    LOG_DIR,
    SAMPLES_TSV,
    load_sample_table,
    validate_inputs,
    SAMPLES,
    OUT_NAMES,
    REPLICATE_LABELS,
    BAM_BY_SAMPLE,
    out_name,
    RESULT_DIRS,
    ensure_result_dirs,
    classify_tsv,
    bw_3end_prefix,
    read_info_tsv,
    existing_read_info_tsv,
    expression_gene_tsv,
    pausing_index_tsv,
    splicing_ratio_tsv,
    splicing_ratio_legacy_tsv,
)

ensure_result_dirs()
samples_df = load_sample_table(SAMPLES_TSV)
input_errors = validate_inputs()
if input_errors:
    raise FileNotFoundError("\n".join(input_errors))
print("Samples:")
print(samples_df.to_string(index=False))
print(f"Results: {RESULTS_DIR}")

# Directory aliases
BAM_SAMPLES = dict(BAM_BY_SAMPLE)
REPLICATES = {s: classify_tsv(s) for s in SAMPLES}
READ_THROUGH_DIR = RESULT_DIRS["read_through"]
SPLICING_DIR = RESULT_DIRS["splicing"]
EXPRESSION_DIR = RESULT_DIRS["expression"]
QUANT_DIR = RESULT_DIRS["quant"]
ORDER_DIR = RESULT_DIRS["splicing_order"]
EXON_DIR = RESULT_DIRS["exon_intron"]
PAUSING_DIR = RESULT_DIRS["pausing"]
GENE_PAUSING_DIR = RESULT_DIRS["gene_pausing"]
BW_DIR = RESULT_DIRS["bw"]
ENDS_DIR = RESULT_DIRS["ends"]
DISTRIBUTION_DIR = RESULT_DIRS["distribution"]
READ_INFO_DIR = RESULT_DIRS["read_info"]
SCRIPT_DIR = SRC_DIR
CHROM_SIZES = REF_DIR / "Ath.chrom.sizes"

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from plotnine import *

theme_common = theme(
    axis_text=element_text(size=12, color="black"),
    axis_title=element_text(size=14, color="black"),
    panel_grid=element_blank(),
)

# --- Notebook parameters (first-intron coupling) ---
GENE_READ_DIR = RESULT_DIRS.get("gene_reads", WORK_DIR / "results" / "gene_reads")
RESULT_DIRS.setdefault("gene_reads", GENE_READ_DIR)
DEFAULT_SAMPLE = SAMPLES[1]
DIST_BINS = np.arange(0, 5001, 100)
SLIDING_WINDOW = 100
SLIDING_STEP = 10
MAX_DIST = 5000
MIN_READS_SS1 = 10
MIN_READS_GROUP = 5
N_FIX_INTRONS = 5
DISPERSION_MIN_READS = 30
DISPERSION_N_PERM = 1000
VALID_P5_FEATURES = ("promoter", "exon_1")


# %% cell 2
_prov.mark(2)
# Load per-sample classify tables (notebook 4)
classify_by_sample = {}
for sample in SAMPLES:
    path = classify_tsv(sample)
    if not path.exists():
        print(f'Skip {sample}: {path} not found')
        continue
    df = pd.read_csv(path, sep='\t')
    df = df[df['splicing_class'] != 'no_intron_covered'].copy()
    classify_by_sample[sample] = df
    print(sample, df.shape)
# default alias for downstream cells that expect a single dataframe
classify_df = classify_by_sample.get(DEFAULT_SAMPLE, next(iter(classify_by_sample.values()), pd.DataFrame()))


# %% cell 3
_prov.mark(3)
gene_df = pd.read_csv(REF_DIR / "exon_intron_pos.chr.bed", sep="\t", header=None)
gene_df.columns = ["chr", "start", "end", "name", "score", "strand"]
gene_df["gene_id"] = gene_df["name"].str.rsplit("_", n=1).str[0]
gene_df["feature_name"] = gene_df["name"].str.rsplit("_", n=1).str[1]
gene_df["feature_type"] = gene_df["feature_name"].str.extract(r"(exon|intron)", expand=False)
gene_df["feature_indx"] = gene_df["feature_name"].str.extract(r"(?:exon|intron)(\d+)", expand=False).astype(int)
gene_df["feature_length"] = gene_df["end"] - gene_df["start"]

exons = gene_df[gene_df["feature_type"] == "exon"].copy()
introns = gene_df[gene_df["feature_type"] == "intron"].copy()

# Per-gene annotation
gene_stats = (
    gene_df.groupby("gene_id", observed=True)
    .agg(
        chr=("chr", "first"),
        strand=("strand", "first"),
        gene_start=("start", "min"),
        gene_end=("end", "max"),
        n_exon=("feature_type", lambda x: (x == "exon").sum()),
        n_intron=("feature_type", lambda x: (x == "intron").sum()),
    )
    .reset_index()
)
gene_stats["gene_length"] = gene_stats["gene_end"] - gene_stats["gene_start"]

# Length of exon 1 and of the last exon
exon1_info = (
    exons.loc[exons["feature_indx"] == 1, ["gene_id", "start", "end", "feature_length"]]
    .rename(columns={
        "start": "exon1_start",
        "end": "exon1_end",
        "feature_length": "exon1_length",
    })
)
last_exon_idx = exons.groupby("gene_id", observed=True)["feature_indx"].max().reset_index(name="last_exon_idx")
last_exon_len = (
    exons.merge(last_exon_idx, on="gene_id")
    .query("feature_indx == last_exon_idx")[["gene_id", "feature_length"]]
    .rename(columns={"feature_length": "last_exon_length"})
)

# Intron 1 coordinates and length, and the length of the last intron
intron1_info = (
    introns.loc[
        introns["feature_indx"] == 1,
        ["gene_id", "start", "end", "feature_length"],
    ]
    .rename(columns={
        "start": "intron1_start",
        "end": "intron1_end",
        "feature_length": "intron1_length",
    })
)
last_intron_idx = introns.groupby("gene_id", observed=True)["feature_indx"].max().reset_index(name="last_intron_idx")
last_intron_len = (
    introns.merge(last_intron_idx, on="gene_id")
    .query("feature_indx == last_intron_idx")[["gene_id", "feature_length"]]
    .rename(columns={"feature_length": "last_intron_length"})
)

gene_feature_df = (
    gene_stats
    .merge(exon1_info, on="gene_id", how="left")
    .merge(last_exon_len, on="gene_id", how="left")
    .merge(intron1_info, on="gene_id", how="left")
    .merge(last_intron_len, on="gene_id", how="left")
)
print(gene_feature_df.head())
print(gene_feature_df.shape)

gene_feature_df.to_csv(EXON_DIR / "gene_feature_df.csv", index=False)


# %% cell 4
_prov.mark(4)
# Merge the true first-intron 3' splice-site coordinates.
gene_intron1_df = gene_feature_df[[
    "gene_id", "intron1_start", "intron1_end", "intron1_length"
]]
Warm_SS_df = classify_df.merge(gene_intron1_df, on="gene_id", how="left")
print(Warm_SS_df.head())

# Distance transcribed after the first intron 3'SS (acceptor).
# BED coordinates are half-open: + strand acceptor=intron1_end;
# - strand acceptor=intron1_start. Positive values are downstream in transcription.
Warm_SS_df["dist_after_first_3ss"] = np.where(
    Warm_SS_df["strand"] == "+",
    Warm_SS_df["p3_coord"] - Warm_SS_df["intron1_end"],
    Warm_SS_df["intron1_start"] - Warm_SS_df["p3_coord"],
)
print(Warm_SS_df[["gene_id", "strand", "intron1_length", "dist_after_first_3ss"]].head())


# %% cell 5
_prov.mark(5)
# Bin by distance of the read 3' end past the first-intron 3' splice site: 100 bp bins, 0–5 kb
_dist_bins = np.arange(0, 5000 + 100, 100)
_dist_labels = [f"{i}-{i + 100}" for i in range(0, 5000, 100)]

Warm_SS_df["dist_group"] = pd.cut(
    Warm_SS_df["dist_after_first_3ss"],
    bins=_dist_bins,
    labels=_dist_labels,
    include_lowest=True,
)
print(Warm_SS_df[["dist_after_first_3ss", "dist_group"]].head())


# %% cell 6
_prov.mark(6)
def _intron_set(val) -> set[int]:
    if pd.isna(val) or str(val).strip() == "":
        return set()
    return {int(x) for x in str(val).split(",")}


# Keep a read in the total only when it covers intron 1 (1 is in covered_introns).
# Then 1 in spliced_introns means spliced; otherwise the intron is unspliced (retained).
_covered_sets = Warm_SS_df["covered_introns"].map(_intron_set)
_spliced_sets = Warm_SS_df["spliced_introns"].map(_intron_set)

_covers_intron1 = _covered_sets.map(lambda introns: 1 in introns)
Warm_SS_intron1_df = Warm_SS_df[
    _covers_intron1
    & Warm_SS_df["dist_after_first_3ss"].ge(0)
].copy()
Warm_SS_intron1_df["intron1_status"] = np.where(
    [1 in s for s in Warm_SS_intron1_df["spliced_introns"].map(_intron_set)],
    "spliced",
    "unspliced",
)
print(f"reads covering intron1: {len(Warm_SS_intron1_df):,} / {len(Warm_SS_df):,}")

# Fixed-bin table (optional; inspect the raw bins)
intron1_splice_df = (
    Warm_SS_intron1_df.dropna(subset=["dist_group"])
    .groupby(["gene_id", "dist_group"], observed=True)["intron1_status"]
    .value_counts()
    .unstack(fill_value=0)
    .reset_index()
)
intron1_splice_df["total"] = (
    intron1_splice_df.get("spliced", 0) + intron1_splice_df.get("unspliced", 0)
)
intron1_splice_df["splicing_ratio"] = (
    intron1_splice_df.get("spliced", 0) / intron1_splice_df["total"]
)
print(intron1_splice_df.head())
print(intron1_splice_df.shape)


# %% cell 7
_prov.mark(7)
intron1_splice_df2 = intron1_splice_df[intron1_splice_df["total"]>10]
print(intron1_splice_df2.head())
print(intron1_splice_df2.shape)


# %% cell 8
_prov.mark(8)
# Left-aligned sliding window [x, x+WINDOW), plotted at the left edge, so the first point is at 0 bp.
# Warm1, Warm2, and Warm3 each get a gene-mean splicing ratio. The plot shows the mean of the three replicates and the SD across replicates.
WINDOW = 100   # bp
STEP = 10      # bp
MIN_READS = 10
MAX_DIST_ALL = 2500
Warm_SS1_REPS = [s for s in SAMPLES if str(s).startswith("Warm") and s in classify_by_sample]


def _intron1_window_table(classify_one: pd.DataFrame) -> pd.DataFrame:
    """Reads covering intron 1, with distance after the first 3'SS."""
    df = classify_one.merge(gene_intron1_df, on="gene_id", how="left")
    df["dist_after_first_3ss"] = np.where(
        df["strand"].eq("+"),
        df["p3_coord"] - df["intron1_end"],
        df["intron1_start"] - df["p3_coord"],
    )
    covers = df["covered_introns"].map(_intron_set).map(lambda introns: 1 in introns)
    df = df.loc[
        covers & df["dist_after_first_3ss"].ge(0),
        ["gene_id", "dist_after_first_3ss", "spliced_introns", "intron1_length"],
    ].copy()
    df["spliced"] = df["spliced_introns"].map(_intron_set).map(lambda introns: 1 in introns)
    return df


def _window_gene_means(reads: pd.DataFrame, starts: np.ndarray) -> list[dict]:
    """Unweighted mean of per-gene intron-1 ratios in each left-aligned window."""
    dist = reads["dist_after_first_3ss"].to_numpy()
    gene = reads["gene_id"].to_numpy()
    spliced = reads["spliced"].to_numpy()
    rows = []
    for start in starts:
        mask = (dist >= start) & (dist < start + WINDOW)
        if not mask.any():
            continue
        sub = pd.DataFrame({"gene_id": gene[mask], "spliced": spliced[mask]})
        agg = sub.groupby("gene_id", sort=False)["spliced"].agg(n_spliced="sum", n_reads="size")
        agg = agg.loc[agg["n_reads"] >= MIN_READS]
        if agg.empty:
            continue
        rows.append({
            "dist_mid": int(start),
            "mean": float((agg["n_spliced"] / agg["n_reads"]).mean()),
            "n_genes": int(len(agg)),
        })
    return rows


# Left edges run through MAX_DIST. The last window is [MAX_DIST, MAX_DIST+WINDOW) and extends one window past the axis.
_starts = np.arange(0, MAX_DIST_ALL + 1, STEP)
_rep_rows = []
for _sample in Warm_SS1_REPS:
    _reads = _intron1_window_table(classify_by_sample[_sample])
    print(f"[{_sample}] intron1 reads for SS1 window: {len(_reads):,}")
    for _row in _window_gene_means(_reads, _starts):
        _row["sample"] = _sample
        _rep_rows.append(_row)

rep_df = pd.DataFrame(_rep_rows)
rep_wide = rep_df.pivot(index="dist_mid", columns="sample", values="mean")
plot_df = pd.DataFrame({
    "dist_mid": rep_wide.index.astype(int),
    "mean": rep_wide.mean(axis=1).to_numpy(),
    "sd": rep_wide.std(axis=1, ddof=1).to_numpy(),
    "n_reps": rep_wide.notna().sum(axis=1).to_numpy(),
})
plot_df["ymin"] = plot_df["mean"] - plot_df["sd"]
plot_df["ymax"] = plot_df["mean"] + plot_df["sd"]
rep_df.to_csv(EXON_DIR / "SS1.splicing.by_replicate.tsv", sep="\t", index=False)
plot_df.to_csv(EXON_DIR / "SS1.splicing.mean.tsv", sep="\t", index=False)
print(
    f"window={WINDOW} bp, step={STEP} bp, align=left, reps={Warm_SS1_REPS}, "
    f"points={len(plot_df)}"
)
print(plot_df.head())

p1 = (
    ggplot(plot_df, aes(x="dist_mid", y="mean"))
    + geom_ribbon(aes(ymin="ymin", ymax="ymax"), fill="#1F77B4", alpha=0.3)
    + geom_line(color="#1F77B4", size=1)
    + scale_y_continuous(breaks=np.arange(0, 1.01, 0.2))
    + coord_cartesian(ylim=(0, 1))
    + theme_bw()
    + theme_common
    + labs(
        x="Distance after first intron 3'SS (bp)",
        y="Mean splicing ratio",
        title=(
            f"Warm mean ± SD ({WINDOW} bp, step {STEP} bp; "
            f"{len(Warm_SS1_REPS)} replicates)"
        ),
    )
)
p1.save(EXON_DIR / "SS1.splicing.pdf", width=6, height=5, dpi=600)
p1.save(EXON_DIR / "SS1.splicing.png", width=6, height=5, dpi=600)
p1


# %% cell 9
_prov.mark(9)
# Two intron-1 length groups: <100 bp and >300 bp. Lengths of 100–300 bp are left out.
# Warm1, Warm2, and Warm3 are computed separately. The plot shows the mean of the three replicates and the SD across replicates.
WINDOW = 100
STEP = 10
MIN_READS = 10
MAX_DIST = 1000
LENGTH_GROUP_ORDER = (
    "short (<100 bp)",
    "long (>300 bp)",
)
LENGTH_GROUP_COLORS = {
    "short (<100 bp)": "#332288",
    "long (>300 bp)": "#CC6677",
}


def _intron1_length_group(length: pd.Series) -> pd.Series:
    """Bin intron-1 length. Lengths of 100–300 bp are left unassigned."""
    grouped = np.select(
        [length < 100, length > 300],
        list(LENGTH_GROUP_ORDER),
        default=pd.NA,
    )
    return pd.Series(grouped, index=length.index)


_starts = np.arange(0, MAX_DIST + 1, STEP)
_rep_rows = []
for _sample in Warm_SS1_REPS:
    _reads = _intron1_window_table(classify_by_sample[_sample])
    _reads["intron1_len_group"] = _intron1_length_group(_reads["intron1_length"])
    _reads = _reads.dropna(subset=["intron1_len_group"])
    print(
        f"[{_sample}] genes by intron1 length:\n"
        + _reads.groupby("intron1_len_group", observed=True)["gene_id"].nunique().to_string()
    )
    for _group, _gdf in _reads.groupby("intron1_len_group", observed=True):
        for _row in _window_gene_means(_gdf, _starts):
            _row["sample"] = _sample
            _row["intron1_len_group"] = _group
            _rep_rows.append(_row)

rep_df = pd.DataFrame(_rep_rows)
_mean_rows = []
for _group, _gdf in rep_df.groupby("intron1_len_group", observed=True):
    _wide = _gdf.pivot(index="dist_mid", columns="sample", values="mean")
    _mean_rows.append(pd.DataFrame({
        "intron1_len_group": _group,
        "dist_mid": _wide.index.astype(int),
        "mean": _wide.mean(axis=1).to_numpy(),
        "sd": _wide.std(axis=1, ddof=1).to_numpy(),
        "n_reps": _wide.notna().sum(axis=1).to_numpy(),
    }))
plot_df = pd.concat(_mean_rows, ignore_index=True)
plot_df["ymin"] = plot_df["mean"] - plot_df["sd"]
plot_df["ymax"] = plot_df["mean"] + plot_df["sd"]
plot_df["intron1_len_group"] = pd.Categorical(
    plot_df["intron1_len_group"], categories=LENGTH_GROUP_ORDER, ordered=True
)
plot_df = plot_df.sort_values(["intron1_len_group", "dist_mid"])
rep_df.to_csv(EXON_DIR / "SS1.splicing.group.by_replicate.tsv", sep="\t", index=False)
plot_df.to_csv(EXON_DIR / "SS1.splicing.group.length.mean.tsv", sep="\t", index=False)
print(
    f"window={WINDOW} bp, step={STEP} bp, reps={Warm_SS1_REPS}, "
    f"groups={list(LENGTH_GROUP_ORDER)}"
)
print(plot_df.groupby("intron1_len_group", observed=True).size())

p1 = (
    ggplot(plot_df, aes(x="dist_mid", y="mean", color="intron1_len_group", fill="intron1_len_group"))
    + geom_ribbon(aes(ymin="ymin", ymax="ymax"), alpha=0.2, color=None)
    + geom_line(size=1)
    + scale_color_manual(values=LENGTH_GROUP_COLORS, breaks=list(LENGTH_GROUP_ORDER))
    + scale_fill_manual(values=LENGTH_GROUP_COLORS, breaks=list(LENGTH_GROUP_ORDER))
    + scale_y_continuous(breaks=np.arange(0, 1.01, 0.2))
    + coord_cartesian(xlim=(0, 1000), ylim=(0, 1))
    + theme_bw()
    + theme_common
    + labs(
        x="Distance after first intron 3'SS (bp)",
        y="Mean splicing ratio",
        title=(
            f"Warm mean ± SD ({WINDOW} bp, step {STEP} bp; "
            f"{len(Warm_SS1_REPS)} replicates)"
        ),
        color="Intron1 length",
        fill="Intron1 length",
    )
)
p1.save(EXON_DIR / "SS1.splicing.group.pdf", width=6.2, height=4.2, dpi=600)
p1.save(EXON_DIR / "SS1.splicing.group.png", width=6.2, height=4.2, dpi=600)
p1


# %% cell 10
_prov.mark(10)
# Replicate-mean version of the intron1 sliding-window plot (Warm.mean / Cold.mean).
# Left-aligned windows [x, x+WINDOW) plotted at x, so the first point is at 0 bp.
# Per window: gene ratio in each replicate (≥ MIN_READS), then mean across ≥2 reps,
# then mean ± SE across genes. Same window / intron1-length groups as the single-rep plot.
WINDOW = 100
STEP = 10
MIN_READS = 10
MAX_DIST = 1000
SMOOTH_N = 15
MIN_REPS_FOR_MEAN = 2
SAMPLE_GROUPS = {
    "Warm": [s for s in SAMPLES if str(s).startswith("Warm") and s in classify_by_sample],
    "Cold": [s for s in SAMPLES if str(s).startswith("Cold") and s in classify_by_sample],
}


def build_intron1_read_table(classify_one: pd.DataFrame) -> pd.DataFrame:
    """Reads covering intron1, with distance after the first 3'SS and length group."""
    df = classify_one.merge(gene_intron1_df, on="gene_id", how="left")
    df["dist_after_first_3ss"] = np.where(
        df["strand"] == "+",
        df["p3_coord"] - df["intron1_end"],
        df["intron1_start"] - df["p3_coord"],
    )
    covers = df["covered_introns"].map(_intron_set).map(lambda s: 1 in s)
    df = df.loc[covers & df["dist_after_first_3ss"].ge(0)].copy()
    spliced = df["spliced_introns"].map(_intron_set).map(lambda s: 1 in s)
    df["intron1_status"] = np.where(spliced, "spliced", "unspliced")
    df["intron1_len_group"] = np.select(
        [df["intron1_length"] < 100, df["intron1_length"] > 300],
        ["short (<100 bp)", "long (>300 bp)"],
        default=pd.NA,
    )
    return df.dropna(subset=["intron1_len_group"]).copy()


intron1_by_sample = {
    sample: build_intron1_read_table(sdf) for sample, sdf in classify_by_sample.items()
}

# Left edge runs to MAX_DIST; the last window [MAX_DIST, MAX_DIST+WINDOW)
# extends one window past the axis so 800-1000 is not empty.
_starts = np.arange(0, MAX_DIST + 1, STEP)
_mean_rows = []
for group, reps in SAMPLE_GROUPS.items():
    if len(reps) < MIN_REPS_FOR_MEAN:
        print(f"Skip {group}.mean: only {len(reps)} replicate(s)")
        continue
    label = f"{group}.mean"
    stacked = pd.concat(
        [intron1_by_sample[s].assign(sample=s) for s in reps],
        ignore_index=True,
    )
    print(f"{label}: reps={reps}; intron1 reads={len(stacked):,}")
    for len_group, gdf in stacked.groupby("intron1_len_group", observed=True):
        for start in _starts:
            lo, hi = start, start + WINDOW
            sub = gdf.loc[
                gdf["dist_after_first_3ss"].ge(lo) & gdf["dist_after_first_3ss"].lt(hi)
            ]
            if sub.empty:
                continue
            gene_rep = (
                sub.groupby(["gene_id", "sample"], observed=True)
                .agg(
                    n_reads=("intron1_status", "size"),
                    spliced=("intron1_status", lambda s: (s == "spliced").sum()),
                )
                .reset_index()
            )
            gene_rep = gene_rep.loc[gene_rep["n_reads"] >= MIN_READS].copy()
            gene_rep["ratio"] = gene_rep["spliced"] / gene_rep["n_reads"]
            gene_mean = (
                gene_rep.groupby("gene_id", observed=True)
                .agg(ratio=("ratio", "mean"), n_reps=("sample", "nunique"))
                .reset_index()
            )
            gene_mean = gene_mean.loc[gene_mean["n_reps"] >= MIN_REPS_FOR_MEAN]
            if gene_mean.empty:
                continue
            n = len(gene_mean)
            sd = float(gene_mean["ratio"].std(ddof=1)) if n > 1 else 0.0
            _mean_rows.append(
                {
                    "sample": label,
                    "intron1_len_group": len_group,
                    "dist_mid": start,
                    "mean": float(gene_mean["ratio"].mean()),
                    "sd": sd,
                    "se": sd / np.sqrt(n) if n > 0 else 0.0,
                    "n_genes": n,
                }
            )

mean_plot_df = pd.DataFrame(_mean_rows).sort_values(
    ["sample", "intron1_len_group", "dist_mid"]
)
_g = mean_plot_df.groupby(["sample", "intron1_len_group"], observed=True)
mean_plot_df["mean_smooth"] = _g["mean"].transform(
    lambda s: s.rolling(SMOOTH_N, center=True, min_periods=1).mean()
)
mean_plot_df["se_smooth"] = _g["se"].transform(
    lambda s: s.rolling(SMOOTH_N, center=True, min_periods=1).mean()
)
mean_plot_df["ymin"] = mean_plot_df["mean_smooth"] - mean_plot_df["se_smooth"]
mean_plot_df["ymax"] = mean_plot_df["mean_smooth"] + mean_plot_df["se_smooth"]
mean_plot_df.to_csv(EXON_DIR / "SS1.splicing.group.mean.tsv", sep="\t", index=False)
print(mean_plot_df.groupby(["sample", "intron1_len_group"], observed=True).size())

mean_plots = {}
for label, sdf in mean_plot_df.groupby("sample", observed=True):
    p_one = (
        ggplot(
            sdf,
            aes(
                x="dist_mid",
                y="mean_smooth",
                color="intron1_len_group",
                fill="intron1_len_group",
            ),
        )
        + geom_ribbon(aes(ymin="ymin", ymax="ymax"), alpha=0.2, color=None)
        + geom_line(size=1)
        + coord_cartesian(xlim=(0, 1000))
        + theme_bw()
        + theme_common
        + labs(
            x="Distance after first intron 3'SS (bp)",
            y="Mean splicing ratio",
            title=(
                f"{label}: sliding window "
                f"({WINDOW} bp, step {STEP} bp; left-aligned; ≥{MIN_REPS_FOR_MEAN} reps; smooth={SMOOTH_N}; ±SE)"
            ),
            color="Intron1 length",
            fill="Intron1 length",
        )
    )
    out = EXON_DIR / f"SS1.splicing.group.{label}"
    p_one.save(Path(str(out) + ".pdf"), width=6, height=4, dpi=600)
    p_one.save(Path(str(out) + ".png"), width=6, height=4, dpi=600)
    print(f"saved {out.name}.pdf/.png")
    mean_plots[label] = p_one
mean_plots
