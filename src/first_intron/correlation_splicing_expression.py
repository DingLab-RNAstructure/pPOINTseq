#!/usr/bin/env python3
"""Step 02_splicing_expression: plain-script conversion of notebook ``2.splicing_expression.ipynb``.

Run from anywhere: ``python src/first_intron/correlation_splicing_expression.py``. Outputs go to ``results/``;
``logs/provenance/02_splicing_expression.tsv`` records which cell wrote each result file.
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


# # 2. Splicing vs expression (per-sample + group mean)
#
# Relate per-gene splicing ratio to expression, gene length, and exon/intron architecture.
#
# For each genotype group (Warm, Cold), make **4** plot sets:
# 1–3. individual replicates
# 4. gene-level mean across replicates (`Warm.mean` / `Cold.mean`)
#
# Mean rule: keep genes with valid values in **≥2** replicates; average over the available 2 or 3 replicates.


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

import pandas as pd
import numpy as np
import subprocess
import matplotlib.pyplot as plt
from scipy import stats
from plotnine import *
import seaborn as sns

# --- Notebook parameters (splicing vs expression) ---
EXPRESSION_POOLED = EXPRESSION_DIR / "Warm_1st_intron_df.tsv"
EXPRESSION_BY_SAMPLE = EXPRESSION_DIR / "Warm_1st_intron_by_rep.tsv"
BUILD_EXPRESSION_SCRIPT = _ROOT / "src" / "first_intron" / "build_first_intron_expression.py"
# Three replicates plus the group mean (keep a gene with at least 2 replicates; average those replicates)
SAMPLE_GROUPS = {
    "Warm": [s for s in SAMPLES if str(s).startswith("Warm")],
    "Cold": [s for s in SAMPLES if str(s).startswith("Cold")],
}
MIN_REPS_FOR_MEAN = 2
PLOT_SAMPLES = []
for _group, _reps in SAMPLE_GROUPS.items():
    PLOT_SAMPLES.extend(_reps)
    PLOT_SAMPLES.append(f"{_group}.mean")
GENE_ARCH_BED = REF_DIR / "exon_intron_pos.chr.bed"
PLOT_DPI = 600
PLOT_WIDTH, PLOT_HEIGHT = 4, 3.5


# %% cell 2
_prov.mark(2)
# Load per-sample expression/splicing, then build group means (≥2 reps).
classify_inputs = [classify_tsv(sample) for sample in SAMPLES]
missing_classify = [path for path in classify_inputs if not path.is_file()]
if missing_classify:
    raise FileNotFoundError(f"Missing classify TSVs: {missing_classify}")

expression_is_stale = (
    not EXPRESSION_BY_SAMPLE.is_file()
    or not EXPRESSION_POOLED.is_file()
    or any(path.stat().st_mtime > EXPRESSION_BY_SAMPLE.stat().st_mtime for path in classify_inputs)
)
if expression_is_stale:
    print(f"Building classify-derived inputs via {BUILD_EXPRESSION_SCRIPT.name}")
    subprocess.run([sys.executable, str(BUILD_EXPRESSION_SCRIPT)], check=True)

expression_by_sample = pd.read_csv(EXPRESSION_BY_SAMPLE, sep="\t")
expression_by_sample["sample"] = expression_by_sample["sample"].astype(str)


def build_group_mean(
    df: pd.DataFrame,
    reps: list[str],
    label: str,
    min_reps: int = MIN_REPS_FOR_MEAN,
) -> pd.DataFrame:
    """Gene-level mean TPM / spliced_ratio over available replicates (≥ min_reps)."""
    sub = df.loc[df["sample"].isin(reps), ["gene_id", "TPM", "spliced_ratio"]].copy()
    sub = sub.dropna(subset=["TPM", "spliced_ratio"])
    sub = sub.loc[sub["TPM"] > 0]
    if sub.empty:
        return pd.DataFrame(columns=["sample", "gene_id", "TPM", "spliced_ratio", "n_reps"])
    grouped = sub.groupby("gene_id", observed=True)
    n_reps = grouped.size()
    keep = n_reps[n_reps >= min_reps].index
    out = (
        grouped.agg(TPM=("TPM", "mean"), spliced_ratio=("spliced_ratio", "mean"))
        .loc[keep]
        .reset_index()
    )
    out["sample"] = label
    out["n_reps"] = out["gene_id"].map(n_reps).astype(int)
    return out[["sample", "gene_id", "TPM", "spliced_ratio", "n_reps"]]


mean_frames = []
for group, reps in SAMPLE_GROUPS.items():
    mean_df = build_group_mean(expression_by_sample, reps, f"{group}.mean")
    mean_frames.append(mean_df)
    n2 = int((mean_df["n_reps"] == 2).sum()) if len(mean_df) else 0
    n3 = int((mean_df["n_reps"] == 3).sum()) if len(mean_df) else 0
    print(f"{group}.mean: genes={len(mean_df):,} (n_reps=2: {n2:,}; n_reps=3: {n3:,}); reps={reps}")

expression_mean = pd.concat(mean_frames, ignore_index=True) if mean_frames else pd.DataFrame()
expression_mean.to_csv(EXPRESSION_DIR / "group_mean_1st_intron.tsv", sep="\t", index=False)

# Keep per-rep rows + mean rows for plotting
rep_cols = ["sample", "gene_id", "TPM", "spliced_ratio"]
expression_plot = pd.concat(
    [
        expression_by_sample[rep_cols].assign(n_reps=1),
        expression_mean[rep_cols + ["n_reps"]],
    ],
    ignore_index=True,
)
expression_plot["sample"] = pd.Categorical(
    expression_plot["sample"], categories=PLOT_SAMPLES, ordered=True
)
print("plot labels:", PLOT_SAMPLES)
print(expression_plot.groupby("sample", observed=True).size().to_string())

# Quick TPM vs spliced_ratio check per plot label
for sample, plot_df in expression_plot.groupby("sample", observed=True):
    plot_df = plot_df.dropna(subset=["TPM", "spliced_ratio"]).copy()
    plot_df = plot_df.loc[plot_df["TPM"] > 0]
    if len(plot_df) < 3:
        print(f"{sample}: skip TPM check (n={len(plot_df)})")
        continue
    r_p, p_p = stats.pearsonr(np.log10(plot_df["TPM"]), plot_df["spliced_ratio"])
    print(f"{sample}: Pearson r={r_p:.4f}, p={p_p:.4e}, n={len(plot_df)}")


# %% cell 3
_prov.mark(3)
gene_df = pd.read_csv(WORK_DIR / "data" / "refs" / "exon_intron_pos.chr.bed", sep="\t", header=None)
gene_df.columns = ["chr", "start", "end", "name", "score","strand"]
gene_df["gene_id"] = gene_df["name"].str.split("_", expand=True)[0]
gene_df["feature_name"] = gene_df["name"].str.split("_", expand=True)[1]
gene_df["feature_length"] = gene_df["end"] - gene_df["start"]
gene_df["feature_type"] = gene_df["feature_name"].str.extract(r'(exon|intron)', expand=False)
gene_df["feature_indx"] = gene_df["feature_name"].str.extract(r"(?:exon|intron)(\d+)", expand=False).astype(int)
print(gene_df.head())

# Per-gene summary: genomic span, exon/intron totals and means, counts
# gene_span_bp: annotated transcript span on genome = max(end) - min(start)
# exon_total_bp / intron_total_bp: sum of segment lengths for that transcript model

gene_span = (
    gene_df.groupby("gene_id")
    .agg(chr=("chr", "first"), strand=("strand", "first"), _lo=("start", "min"), _hi=("end", "max"))
    .assign(gene_span_bp=lambda d: (d["_hi"] - d["_lo"]).astype(int))
    .drop(columns=["_lo", "_hi"])
    .reset_index()
)


def _agg_features(ft: str) -> pd.DataFrame:
    sub = gene_df.loc[gene_df["feature_type"] == ft].copy()
    if sub.empty:
        return pd.DataFrame(columns=["gene_id", f"{ft}_total_bp", f"n_{ft}s", f"{ft}_mean_bp"])
    g = sub.groupby("gene_id")["feature_length"].agg(["sum", "count"]).rename(
        columns={"sum": f"{ft}_total_bp", "count": f"n_{ft}s"}
    )
    g[f"{ft}_mean_bp"] = g[f"{ft}_total_bp"] / g[f"n_{ft}s"]
    return g.reset_index()


exon_stats = _agg_features("exon")
intron_stats = _agg_features("intron")

# First intron length: row intron1 only — feature_type=intron and index 1 (matches *_intron1 in BED name)
intron1_df = (
    gene_df.loc[
        (gene_df["feature_type"] == "intron") & (gene_df["feature_indx"] == 1),
        ["gene_id", "feature_length"],
    ]
    .drop_duplicates(subset=["gene_id"])
    .rename(columns={"feature_length": "intron1_bp"})
)
# Second exon length (exon2 only)
exon2_df = (
    gene_df.loc[
        (gene_df["feature_type"] == "exon") & (gene_df["feature_indx"] == 2),
        ["gene_id", "feature_length"],
    ]
    .drop_duplicates(subset=["gene_id"])
    .rename(columns={"feature_length": "exon2_bp"})
)
intron2_df = (
    gene_df.loc[
        (gene_df["feature_type"] == "intron") & (gene_df["feature_indx"] == 2),
        ["gene_id", "feature_length"],
    ]
    .drop_duplicates(subset=["gene_id"])
    .rename(columns={"feature_length": "intron2_bp"})
)

gene_summary = (
    gene_span.merge(exon_stats, on="gene_id", how="left")
    .merge(intron_stats, on="gene_id", how="left")
    .merge(intron1_df, on="gene_id", how="left")
    .merge(exon2_df, on="gene_id", how="left")
    .merge(intron2_df, on="gene_id", how="left")
)

gene_summary["exon_total_bp"] = gene_summary["exon_total_bp"].fillna(0)
gene_summary["intron_total_bp"] = gene_summary["intron_total_bp"].fillna(0)
gene_summary["n_exons"] = gene_summary["n_exons"].fillna(0).astype(int)
gene_summary["n_introns"] = gene_summary["n_introns"].fillna(0).astype(int)

gene_summary.head(10)


# %% cell 4
_prov.mark(4)
# Merge per-rep + group-mean expression/splicing with transcript architecture.
expression_df2 = pd.merge(
    expression_plot[["sample", "gene_id", "TPM", "spliced_ratio", "n_reps"]],
    gene_summary,
    on="gene_id",
    how="inner",
).dropna(subset=["TPM", "spliced_ratio"])
expression_df2 = expression_df2.loc[expression_df2["TPM"] > 0].copy()
expression_df2["sample"] = pd.Categorical(
    expression_df2["sample"].astype(str), categories=PLOT_SAMPLES, ordered=True
)
print(expression_df2.groupby("sample", observed=True).size().to_string())

theme_scatter = (
    theme_bw()
    + theme(
        axis_text=element_text(size=10, color="black"),
        axis_title=element_text(size=12, color="black"),
        panel_grid=element_blank(),
        plot_title=element_text(size=11),
    )
)


def save_corr_scatter(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    stem: str,
    xlab: str,
    ylab: str,
    x_log10_for_r: bool = False,
    y_log10_for_r: bool = False,
    use_x_log10: bool = False,
    use_y_log10: bool = False,
    y_limits=(None, 1.15),
    y_breaks=None,
    width: float = PLOT_WIDTH,
    height: float = PLOT_HEIGHT,
):
    """Draw and save one scatter+lm plot per sample / group-mean label."""
    if y_breaks is None and not use_y_log10:
        y_breaks = np.arange(0, 1.2, 0.2)
    last = None
    for sample in PLOT_SAMPLES:
        sdf = df.loc[df["sample"] == sample, [x, y]].dropna().copy()
        if x_log10_for_r or use_x_log10:
            sdf = sdf.loc[sdf[x] > 0]
        if y_log10_for_r or use_y_log10:
            sdf = sdf.loc[sdf[y] > 0]
        if len(sdf) < 3:
            print(f"Skip {sample} {stem}: n={len(sdf)}")
            continue
        xx = np.log10(sdf[x]) if x_log10_for_r else sdf[x]
        yy = np.log10(sdf[y]) if y_log10_for_r else sdf[y]
        cor_r, cor_p = stats.pearsonr(xx, yy)
        p_txt = "p < 1e-300" if (not np.isfinite(cor_p) or cor_p <= 0) else f"p = {cor_p:.3e}"
        p = (
            ggplot(sdf, aes(x=x, y=y))
            + geom_point(alpha=0.25, size=1, color="grey")
            + geom_smooth(method="lm", se=False, color="black", size=0.5)
            + theme_scatter
            + labs(x=xlab, y=ylab)
            + ggtitle(f"{sample}: Pearson r = {cor_r:.3f} ({p_txt})")
        )
        if use_x_log10:
            p = p + scale_x_log10()
        if use_y_log10:
            p = p + scale_y_log10()
        elif y_limits is not None:
            p = p + scale_y_continuous(limits=y_limits, breaks=y_breaks)
        out = EXPRESSION_DIR / f"{sample}.elongating.{stem}"
        p.save(Path(str(out) + ".png"), width=width, height=height, dpi=PLOT_DPI)
        p.save(Path(str(out) + ".pdf"), width=width, height=height)
        print(f"saved {out.name}.png/.pdf  n={len(sdf)}")
        last = p
    return last


# %% cell 5
_prov.mark(5)
# gene_length_vs_spliced_ratio — one plot per sample
save_corr_scatter(
    expression_df2,
    x='gene_span_bp',
    y='spliced_ratio',
    stem='gene_length_vs_spliced_ratio',
    xlab='Gene Length (bp) (log10)',
    ylab='Spliced Ratio',
    x_log10_for_r=True,
    use_x_log10=True,
)


# Weighted gene-level PSI vs gene length. PSI is the read-weighted mean across annotated introns (`weighted_gene_splicing`: sum 3' end bins, then `Σ PSI_i w_i / Σ w_i`, `w_i = max(intron reads, 1)`). Plots cover the three replicates and `Warm.mean` / `Cold.mean` (≥2 replicates).


# %% cell 6
_prov.mark(6)
# gene_length_vs_weighted_psi — three replicates + group mean
WEIGHTED_PSI_MIN_READS = 10
WEIGHTED_PSI_TSV = EXPRESSION_DIR / "by_sample.elongating_gene_length_vs_weighted_psi.tsv"


def load_all_p5_splicing(sample: str) -> pd.DataFrame:
    """Load all-P5 intron splicing counts for one replicate."""
    for suffix in (
        "elongating.all_p5.splicing_ratio.filtered.tsv",
        "elongating.all_p5.splicing_ratio.tsv",
    ):
        path = SPLICING_DIR / f"{out_name(sample)}.{suffix}"
        if path.is_file():
            return pd.read_csv(path, sep="\t")
    raise FileNotFoundError(
        f"Missing splicing table for {sample}. Run notebook 1 splicing QC first."
    )


def weighted_gene_splicing(
    splicing_df: pd.DataFrame,
    min_reads: int = WEIGHTED_PSI_MIN_READS,
) -> pd.DataFrame:
    """Collapse intron PSI to one gene-level ratio, weighted by intron reads."""
    required = {"gene_id", "intron_id", "total_reads", "spliced_reads"}
    missing = required.difference(splicing_df.columns)
    if missing:
        raise KeyError(f"Splicing table missing columns: {sorted(missing)}")

    intron = (
        splicing_df.groupby(["gene_id", "intron_id"], as_index=False, observed=True)
        .agg(total_reads=("total_reads", "sum"), spliced_reads=("spliced_reads", "sum"))
    )
    intron = intron.loc[intron["total_reads"] > min_reads].copy()
    if intron.empty:
        return pd.DataFrame(columns=["gene_id", "spliced_ratio", "n_introns", "total_reads"])

    intron["w"] = intron["total_reads"].clip(lower=1).astype(float)
    intron["PSI"] = intron["spliced_reads"] / intron["total_reads"]
    intron["num"] = intron["PSI"] * intron["w"]
    gene = intron.groupby("gene_id", as_index=False).agg(
        spliced_ratio=("num", "sum"),
        w=("w", "sum"),
        n_introns=("intron_id", "size"),
        total_reads=("total_reads", "sum"),
    )
    gene["spliced_ratio"] = gene["spliced_ratio"] / gene["w"]
    return gene[["gene_id", "spliced_ratio", "n_introns", "total_reads"]]


def build_weighted_psi_mean(
    df: pd.DataFrame,
    reps: list[str],
    label: str,
    min_reps: int = MIN_REPS_FOR_MEAN,
) -> pd.DataFrame:
    """Average gene-level weighted PSI across replicates with ≥ min_reps."""
    sub = df.loc[df["sample"].isin(reps), ["gene_id", "spliced_ratio"]].dropna()
    if sub.empty:
        return pd.DataFrame(columns=["sample", "gene_id", "spliced_ratio", "n_reps"])
    grouped = sub.groupby("gene_id", observed=True)
    n_reps = grouped.size()
    keep = n_reps[n_reps >= min_reps].index
    out = grouped.agg(spliced_ratio=("spliced_ratio", "mean")).loc[keep].reset_index()
    out["sample"] = label
    out["n_reps"] = out["gene_id"].map(n_reps).astype(int)
    return out[["sample", "gene_id", "spliced_ratio", "n_reps"]]


weighted_rows = []
for sample in SAMPLES:
    psi = weighted_gene_splicing(load_all_p5_splicing(sample))
    psi["sample"] = sample
    psi["n_reps"] = 1
    weighted_rows.append(psi)
    print(f"[{sample}] genes with weighted PSI: {len(psi):,}")

weighted_by_sample = pd.concat(weighted_rows, ignore_index=True)
mean_frames = []
for group, reps in SAMPLE_GROUPS.items():
    label = f"{group}.mean"
    mean_df = build_weighted_psi_mean(weighted_by_sample, reps, label)
    mean_frames.append(mean_df)
    n2 = int((mean_df["n_reps"] == 2).sum()) if len(mean_df) else 0
    n3 = int((mean_df["n_reps"] >= 3).sum()) if len(mean_df) else 0
    print(f"[{label}] genes={len(mean_df):,} (n_reps=2: {n2:,}; n_reps≥3: {n3:,}); reps={reps}")

weighted_plot = pd.concat(
    [
        weighted_by_sample[["sample", "gene_id", "spliced_ratio", "n_reps"]],
        pd.concat(mean_frames, ignore_index=True) if mean_frames else pd.DataFrame(),
    ],
    ignore_index=True,
)
weighted_length_df = weighted_plot.merge(
    gene_summary[["gene_id", "gene_span_bp"]],
    on="gene_id",
    how="inner",
).dropna(subset=["spliced_ratio", "gene_span_bp"])
weighted_length_df = weighted_length_df.loc[weighted_length_df["gene_span_bp"] > 0].copy()
weighted_length_df["sample"] = pd.Categorical(
    weighted_length_df["sample"].astype(str), categories=PLOT_SAMPLES, ordered=True
)
weighted_length_df.to_csv(WEIGHTED_PSI_TSV, sep="\t", index=False)

save_corr_scatter(
    weighted_length_df,
    x="gene_span_bp",
    y="spliced_ratio",
    stem="gene_length_vs_weighted_psi",
    xlab="Gene Length (bp) (log10)",
    ylab="Weighted PSI",
    x_log10_for_r=True,
    use_x_log10=True,
)


# Gene TPM vs the fraction of elongating reads whose 3' end falls in an annotated exon (`p3_feat` starts with `exon_`). Denominator is all classified reads assigned to the gene. Keep only genes with **≥1 annotated intron** (`gene_summary.n_introns`). Genes need more than 10 such reads. Plots cover the three replicates and `Warm.mean` / `Cold.mean` (≥2 replicates).


# %% cell 7
_prov.mark(7)
# TPM vs 3' end-in-exon proportion — three replicates + group mean
EXON_3END_MIN_READS = 10
EXON_3END_TSV = EXPRESSION_DIR / "by_sample.elongating_TPM_vs_exon_3end_prop.tsv"


def gene_exon_3end_prop(sample: str, min_reads: int = EXON_3END_MIN_READS) -> pd.DataFrame:
    """Per-gene fraction of classified reads with 3' end in an annotated exon."""
    path = classify_tsv(sample)
    if not path.is_file():
        raise FileNotFoundError(path)
    reads = pd.read_csv(path, sep="\t", usecols=["gene_id", "p3_feat"])
    reads = reads.dropna(subset=["gene_id"]).copy()
    reads["gene_id"] = reads["gene_id"].astype(str)
    p3 = reads["p3_feat"].fillna("").astype(str)
    # classify TSV is one row per read; is_exon marks reads whose 3' end is in an exon.
    reads["is_exon"] = p3.str.startswith("exon_")
    gene = reads.groupby("gene_id", as_index=False, observed=True).agg(
        n_reads=("is_exon", "size"),
        n_exon_reads=("is_exon", "sum"),
    )
    gene = gene.loc[gene["n_reads"] > min_reads].copy()
    gene["exon_3end_prop"] = gene["n_exon_reads"] / gene["n_reads"]
    gene["sample"] = sample
    return gene[["sample", "gene_id", "n_reads", "n_exon_reads", "exon_3end_prop"]]


def load_gene_tpm() -> pd.DataFrame:
    """Gene TPM from this notebook's per-sample table, or the saved TSV."""
    if "expression_by_sample" in globals():
        tpm = expression_by_sample[["sample", "gene_id", "TPM"]].copy()
    else:
        tpm = pd.read_csv(EXPRESSION_BY_SAMPLE, sep="\t", usecols=["sample", "gene_id", "TPM"])
    tpm["gene_id"] = tpm["gene_id"].astype(str)
    tpm = tpm.dropna(subset=["TPM"])
    return tpm.loc[tpm["TPM"] > 0].copy()


def build_tpm_exon_mean(
    df: pd.DataFrame,
    reps: list[str],
    label: str,
    min_reps: int = MIN_REPS_FOR_MEAN,
) -> pd.DataFrame:
    """Average TPM and exon 3' proportion across replicates with ≥ min_reps."""
    sub = df.loc[df["sample"].isin(reps), ["gene_id", "TPM", "exon_3end_prop"]].dropna()
    sub = sub.loc[sub["TPM"] > 0]
    if sub.empty:
        return pd.DataFrame(columns=["sample", "gene_id", "TPM", "exon_3end_prop", "n_reps"])
    grouped = sub.groupby("gene_id", observed=True)
    n_reps = grouped.size()
    keep = n_reps[n_reps >= min_reps].index
    out = grouped.agg(TPM=("TPM", "mean"), exon_3end_prop=("exon_3end_prop", "mean")).loc[keep].reset_index()
    out["sample"] = label
    out["n_reps"] = out["gene_id"].map(n_reps).astype(int)
    return out[["sample", "gene_id", "TPM", "exon_3end_prop", "n_reps"]]


tpm_df = load_gene_tpm()
if "gene_summary" not in globals():
    raise RuntimeError("Run the gene-architecture cell first so gene_summary exists.")
intron_genes = set(gene_summary.loc[gene_summary["n_introns"] >= 1, "gene_id"].astype(str))
print(
    f"Intron-containing genes (n_introns ≥ 1): {len(intron_genes):,} / {len(gene_summary):,}"
)

exon_rows = []
for sample in SAMPLES:
    prop = gene_exon_3end_prop(sample)
    n_before = len(prop)
    prop = prop.loc[prop["gene_id"].isin(intron_genes)].copy()
    merged = prop.merge(tpm_df.loc[tpm_df["sample"] == sample, ["gene_id", "TPM"]], on="gene_id", how="inner")
    merged["n_reps"] = 1
    exon_rows.append(merged)
    print(
        f"[{sample}] intron genes with TPM and exon 3' prop: {len(merged):,} "
        f"(dropped intronless {n_before - len(prop):,})"
    )

exon_by_sample = pd.concat(exon_rows, ignore_index=True)
mean_frames = []
for group, reps in SAMPLE_GROUPS.items():
    label = f"{group}.mean"
    mean_df = build_tpm_exon_mean(exon_by_sample, reps, label)
    mean_frames.append(mean_df)
    n2 = int((mean_df["n_reps"] == 2).sum()) if len(mean_df) else 0
    n3 = int((mean_df["n_reps"] >= 3).sum()) if len(mean_df) else 0
    print(f"[{label}] genes={len(mean_df):,} (n_reps=2: {n2:,}; n_reps≥3: {n3:,}); reps={reps}")

exon_plot = pd.concat(
    [
        exon_by_sample[["sample", "gene_id", "TPM", "exon_3end_prop", "n_reps"]],
        pd.concat(mean_frames, ignore_index=True) if mean_frames else pd.DataFrame(),
    ],
    ignore_index=True,
)
exon_plot["sample"] = pd.Categorical(exon_plot["sample"].astype(str), categories=PLOT_SAMPLES, ordered=True)
exon_plot.to_csv(EXON_3END_TSV, sep="\t", index=False)
print(f"Wrote {EXON_3END_TSV}")

save_corr_scatter(
    exon_plot,
    x="TPM",
    y="exon_3end_prop",
    stem="TPM_vs_exon_3end_prop",
    xlab="TPM (log10)",
    ylab="3' end in exon (proportion)",
    x_log10_for_r=True,
    use_x_log10=True,
)


# %% cell 8
_prov.mark(8)
# exon_number_vs_spliced_ratio — one plot per sample
save_corr_scatter(
    expression_df2,
    x='n_exons',
    y='spliced_ratio',
    stem='exon_number_vs_spliced_ratio',
    xlab='Exon Number',
    ylab='Spliced Ratio',
)


# %% cell 9
_prov.mark(9)
# exon_mean_bp_vs_spliced_ratio — one plot per sample
save_corr_scatter(
    expression_df2,
    x='exon_mean_bp',
    y='spliced_ratio',
    stem='exon_mean_bp_vs_spliced_ratio',
    xlab='exon_mean_bp',
    ylab='Spliced Ratio',
)


# %% cell 10
_prov.mark(10)
# intron_mean_bp_vs_spliced_ratio — one plot per sample
save_corr_scatter(
    expression_df2,
    x='intron_mean_bp',
    y='spliced_ratio',
    stem='intron_mean_bp_vs_spliced_ratio',
    xlab='intron_mean_bp',
    ylab='Spliced Ratio',
)


# %% cell 11
_prov.mark(11)
# exon_total_bp_vs_spliced_ratio — one plot per sample
save_corr_scatter(
    expression_df2,
    x='exon_total_bp',
    y='spliced_ratio',
    stem='exon_total_bp_vs_spliced_ratio',
    xlab='exon_total_bp',
    ylab='Spliced Ratio',
    x_log10_for_r=True,
    use_x_log10=True,
)


# %% cell 12
_prov.mark(12)
# intron_total_bp_vs_spliced_ratio — one plot per sample
save_corr_scatter(
    expression_df2,
    x='intron_total_bp',
    y='spliced_ratio',
    stem='intron_total_bp_vs_spliced_ratio',
    xlab='intron_total_bp',
    ylab='Spliced Ratio',
    x_log10_for_r=True,
    use_x_log10=True,
)


# %% cell 13
_prov.mark(13)
# intron_total_bp_vs_TPM — one plot per sample
save_corr_scatter(
    expression_df2,
    x='intron_total_bp',
    y='TPM',
    stem='intron_total_bp_vs_TPM',
    xlab='intron_total_bp',
    ylab='TPM',
    x_log10_for_r=True,
    y_log10_for_r=True,
    use_x_log10=True,
    use_y_log10=True,
    y_limits=None,
)


# %% cell 14
_prov.mark(14)
# intron1_bp_vs_spliced_ratio — one plot per sample
save_corr_scatter(
    expression_df2,
    x='intron1_bp',
    y='spliced_ratio',
    stem='intron1_bp_vs_spliced_ratio',
    xlab='Intron1 Length (bp)',
    ylab='Spliced Ratio',
    use_x_log10=True,
    use_y_log10=False,
    y_limits=None,
)


# %% cell 15
_prov.mark(15)
# exon2_bp_vs_spliced_ratio — one plot per sample
save_corr_scatter(
    expression_df2,
    x='exon2_bp',
    y='spliced_ratio',
    stem='exon2_bp_vs_spliced_ratio',
    xlab='Exon2 Length (bp)',
    ylab='Spliced Ratio',
    use_x_log10=True,
    use_y_log10=True,
    y_limits=None,
)


# %% cell 16
_prov.mark(16)
# intron2_bp_vs_spliced_ratio — one plot per sample
save_corr_scatter(
    expression_df2,
    x='intron2_bp',
    y='spliced_ratio',
    stem='intron2_bp_vs_spliced_ratio',
    xlab='Intron2 Length (bp)',
    ylab='Spliced Ratio',
    use_x_log10=True,
    use_y_log10=True,
    y_limits=None,
)


# ## Per-intron splicing vs nearest downstream exon
#
# Each intron is paired with the immediately 3' exon of the same transcript (`intronN` → `exonN+1`).
# Splicing ratios are collapsed across 3' end bins, then plotted against that downstream exon length.


# %% cell 17
_prov.mark(17)
# Per-intron spliced ratio vs nearest downstream exon length (intronN → exonN+1).
MIN_INTRON_READS = 10

downstream_exon = (
    gene_df.loc[gene_df["feature_type"] == "exon", ["gene_id", "feature_indx", "feature_length"]]
    .rename(columns={"feature_indx": "exon_indx", "feature_length": "downstream_exon_bp"})
)
intron_to_exon = (
    gene_df.loc[gene_df["feature_type"] == "intron", ["gene_id", "feature_indx"]]
    .rename(columns={"feature_indx": "intron_indx"})
    .assign(exon_indx=lambda d: d["intron_indx"] + 1)
    .merge(downstream_exon, on=["gene_id", "exon_indx"], how="inner")
)
intron_to_exon["intron_id"] = "intron" + intron_to_exon["intron_indx"].astype(str)
intron_to_exon = intron_to_exon[["gene_id", "intron_id", "downstream_exon_bp"]].drop_duplicates()


def intron_level_splicing_ratio(splicing_df: pd.DataFrame, min_reads: int = MIN_INTRON_READS) -> pd.DataFrame:
    """Collapse 3' end bins to one spliced ratio per gene intron."""
    grouped = splicing_df.groupby(["gene_id", "intron_id"], as_index=False, observed=True).agg(
        total_reads=("total_reads", "sum"),
        spliced_reads=("spliced_reads", "sum"),
    )
    grouped = grouped.loc[grouped["total_reads"] >= min_reads].copy()
    grouped["spliced_ratio"] = grouped["spliced_reads"] / grouped["total_reads"]
    return grouped


intron_frames = []
for sample in SAMPLES:
    splicing_path = SPLICING_DIR / f"{out_name(sample)}.elongating.all_p5.splicing_ratio.tsv"
    if not splicing_path.is_file():
        splicing_path = splicing_ratio_tsv(sample)
    if not splicing_path.is_file():
        raise FileNotFoundError(f"Missing splicing-ratio table for {sample}")
    sample_introns = intron_level_splicing_ratio(pd.read_csv(splicing_path, sep="\t"))
    sample_introns["sample"] = str(sample)
    intron_frames.append(sample_introns)
    print(f"{sample}: intron-level rows={len(sample_introns):,} (min_reads>={MIN_INTRON_READS})")

intron_ratio_by_sample = pd.concat(intron_frames, ignore_index=True)


def build_intron_group_mean(
    df: pd.DataFrame,
    reps: list[str],
    label: str,
    min_reps: int = MIN_REPS_FOR_MEAN,
) -> pd.DataFrame:
    """Mean intron spliced ratio over available replicates (≥ min_reps)."""
    sub = df.loc[df["sample"].isin(reps), ["gene_id", "intron_id", "spliced_ratio"]].dropna()
    if sub.empty:
        return pd.DataFrame(columns=["sample", "gene_id", "intron_id", "spliced_ratio", "n_reps"])
    grouped = sub.groupby(["gene_id", "intron_id"], observed=True)
    n_reps = grouped.size()
    keep = n_reps[n_reps >= min_reps].index
    out = grouped.agg(spliced_ratio=("spliced_ratio", "mean")).loc[keep].reset_index()
    out["sample"] = label
    out["n_reps"] = out.set_index(["gene_id", "intron_id"]).index.map(n_reps).astype(int)
    return out[["sample", "gene_id", "intron_id", "spliced_ratio", "n_reps"]]


mean_intron_frames = []
for group, reps in SAMPLE_GROUPS.items():
    mean_df = build_intron_group_mean(intron_ratio_by_sample, reps, f"{group}.mean")
    mean_intron_frames.append(mean_df)
    print(f"{group}.mean: introns={len(mean_df):,}; reps={reps}")

intron_plot = pd.concat(
    [
        intron_ratio_by_sample[["sample", "gene_id", "intron_id", "spliced_ratio"]],
        pd.concat(mean_intron_frames, ignore_index=True)[["sample", "gene_id", "intron_id", "spliced_ratio"]],
    ],
    ignore_index=True,
)
intron_plot = intron_plot.merge(intron_to_exon, on=["gene_id", "intron_id"], how="inner")
intron_plot["sample"] = pd.Categorical(intron_plot["sample"].astype(str), categories=PLOT_SAMPLES, ordered=True)
intron_plot.to_csv(
    SPLICING_DIR / "intron_spliced_ratio_vs_downstream_exon.tsv",
    sep="\t",
    index=False,
)
print(intron_plot.groupby("sample", observed=True).size().to_string())

save_corr_scatter(
    intron_plot,
    x="downstream_exon_bp",
    y="spliced_ratio",
    stem="downstream_exon_bp_vs_spliced_ratio",
    xlab="Downstream exon length (bp)",
    ylab="Intron spliced ratio",
    x_log10_for_r=True,
    use_x_log10=True,
)


# %% cell 18
_prov.mark(18)
# Split the per-intron downstream-exon scatter by gene intron number, then boxplot.
INTRON_N_GROUP_ORDER = ("1", "2", "3", "4", "5", "6", "7", "8", "9", "10+")


def intron_number_group(n) -> str | None:
    """Bin genes by annotated intron count: 1–9 separately, 10+ together."""
    if pd.isna(n):
        return None
    n = int(n)
    if n <= 0:
        return None
    if n >= 10:
        return "10+"
    return str(n)


intron_n_plot = intron_plot.merge(
    gene_summary[["gene_id", "n_introns"]],
    on="gene_id",
    how="left",
)
intron_n_plot["n_intron_group"] = pd.Categorical(
    intron_n_plot["n_introns"].map(intron_number_group),
    categories=INTRON_N_GROUP_ORDER,
    ordered=True,
)
intron_n_plot = intron_n_plot.dropna(subset=["n_intron_group", "spliced_ratio"]).copy()
intron_n_plot.to_csv(
    SPLICING_DIR / "intron_spliced_ratio_vs_downstream_exon.by_intron_number.tsv",
    sep="\t",
    index=False,
)

for sample in PLOT_SAMPLES:
    sdf = intron_n_plot.loc[intron_n_plot["sample"] == sample].copy()
    if sdf.empty:
        print(f"Skip {sample} intron-number boxplot: no rows")
        continue
    counts = (
        sdf.groupby("n_intron_group", observed=True)
        .agg(n=("spliced_ratio", "size"), n_genes=("gene_id", "nunique"))
        .reset_index()
    )
    counts["label"] = "n=" + counts["n"].astype(str)
    counts["y_label"] = 1.04
    print(f"[{sample}]")
    print(counts.to_string(index=False))
    p = (
        ggplot(sdf, aes(x="n_intron_group", y="spliced_ratio"))
        + geom_boxplot(fill="#4F88C6", outlier_shape=None, width=0.6)
        + geom_text(
            data=counts,
            mapping=aes(x="n_intron_group", y="y_label", label="label"),
            inherit_aes=False,
            size=6,
        )
        + theme_bw()
        + theme(
            axis_text=element_text(size=10, color="black"),
            axis_title=element_text(size=12, color="black"),
            panel_grid=element_blank(),
            plot_title=element_text(size=11),
        )
        + labs(
            x="Introns per gene",
            y="Intron spliced ratio",
            title=f"{sample}: downstream-exon set, by gene intron number",
        )
        + scale_y_continuous(limits=(0, 1.12), breaks=np.arange(0, 1.2, 0.2))
    )
    out = EXPRESSION_DIR / f"{sample}.elongating.downstream_exon.spliced_ratio_by_intron_number_boxplot"
    p.save(Path(str(out) + ".png"), width=5.2, height=4.0, dpi=PLOT_DPI)
    p.save(Path(str(out) + ".pdf"), width=5.2, height=4.0)
    print(f"saved {out.name}.png/.pdf")


# Per-intron scatter of **intron number** vs spliced ratio, same style as `exon_number_vs_spliced_ratio`. Each point is one intron (`min_reads ≥ 10`); *x* is the gene’s annotated intron count.


# %% cell 19
_prov.mark(19)
# intron_number_vs_spliced_ratio — intron-level, one plot per sample
BY_INTRON_N_TSV = SPLICING_DIR / "intron_spliced_ratio_vs_downstream_exon.by_intron_number.tsv"
if "intron_n_plot" not in globals():
    intron_n_plot = pd.read_csv(BY_INTRON_N_TSV, sep="\t")

intron_number_scatter = intron_n_plot[["sample", "n_introns", "spliced_ratio"]].copy()
intron_number_scatter = intron_number_scatter.loc[intron_number_scatter["n_introns"] >= 1]
save_corr_scatter(
    intron_number_scatter,
    x="n_introns",
    y="spliced_ratio",
    stem="intron_number_vs_spliced_ratio",
    xlab="Intron Number",
    ylab="Spliced Ratio",
)


# ## Per-intron correlation heatmap: intron number vs spliced ratio
#
# Each row is one intron (`min_reads ≥ 10`). **Intron position** is the integer in `intron_id` (`intron1`, `intron2`, …). A pooled Spearman of position vs spliced ratio is confounded by gene intron count (later positions exist only in multi-intron genes). The main heatmap therefore computes Spearman **within genes of the same intron count**.


# %% cell 20
_prov.mark(20)
# Intron-level Spearman heatmap: intron position vs spliced ratio.
CORR_MIN_N = 30
INTRON_INDEX_GROUP_ORDER = ("1", "2", "3", "4", "5", "6", "7", "8", "9", "10+")
FEATURE_LABELS = {
    "intron_index": "Intron position",
    "n_introns": "Introns per gene",
}
BY_INTRON_N_TSV = SPLICING_DIR / "intron_spliced_ratio_vs_downstream_exon.by_intron_number.tsv"
if "intron_n_plot" not in globals():
    intron_n_plot = pd.read_csv(BY_INTRON_N_TSV, sep="\t")

heat_df = intron_n_plot.copy()
heat_df["intron_index"] = heat_df["intron_id"].str.extract(r"intron(\d+)", expand=False).astype("Int64")
heat_df = heat_df.dropna(subset=["intron_index", "spliced_ratio", "n_introns"]).copy()
heat_df["intron_index"] = heat_df["intron_index"].astype(int)
SAMPLE_HEAT_ORDER = list(reversed(PLOT_SAMPLES))
heat_df["sample"] = pd.Categorical(heat_df["sample"].astype(str), categories=SAMPLE_HEAT_ORDER, ordered=True)


def intron_index_group(n) -> str | None:
    if pd.isna(n):
        return None
    n = int(n)
    if n <= 0:
        return None
    if n >= 10:
        return "10+"
    return str(n)


def spearman_row(sub: pd.DataFrame, x: str, min_n: int = CORR_MIN_N) -> dict:
    pair = sub[[x, "spliced_ratio"]].dropna()
    n = int(len(pair))
    n_x = int(pair[x].nunique()) if n else 0
    out = {
        "n": n,
        "spearman_rho": np.nan,
        "spearman_p": np.nan,
        "pearson_r": np.nan,
        "pearson_p": np.nan,
    }
    if n < min_n or n_x < 2:
        return out
    spearman = stats.spearmanr(pair[x], pair["spliced_ratio"])
    pearson = stats.pearsonr(pair[x], pair["spliced_ratio"])
    out.update(
        spearman_rho=float(spearman.statistic),
        spearman_p=float(spearman.pvalue),
        pearson_r=float(pearson.statistic),
        pearson_p=float(pearson.pvalue),
    )
    return out


overall_rows = []
for sample in PLOT_SAMPLES:
    sdf = heat_df.loc[heat_df["sample"].eq(sample)]
    for feature in ("intron_index", "n_introns"):
        row = spearman_row(sdf, feature)
        row.update(sample=sample, stratum="all_introns", feature=feature, n_intron_group="all")
        overall_rows.append(row)

strat_rows = []
for sample in PLOT_SAMPLES:
    sdf = heat_df.loc[heat_df["sample"].eq(sample)]
    for group in INTRON_N_GROUP_ORDER:
        gdf = sdf.loc[sdf["n_intron_group"].astype(str).eq(group)]
        row = spearman_row(gdf, "intron_index")
        row.update(sample=sample, stratum="by_n_introns", feature="intron_index", n_intron_group=group)
        strat_rows.append(row)

corr_df = pd.concat([pd.DataFrame(overall_rows), pd.DataFrame(strat_rows)], ignore_index=True)
corr_df["sample"] = pd.Categorical(corr_df["sample"], categories=SAMPLE_HEAT_ORDER, ordered=True)
corr_path = SPLICING_DIR / "intron_index_vs_spliced_ratio.correlation.tsv"
corr_df.to_csv(corr_path, sep="\t", index=False)
print(f"wrote {corr_path.name}")

theme_heat = (
    theme_bw()
    + theme(
        axis_text=element_text(size=10, color="black"),
        axis_title=element_text(size=12, color="black"),
        panel_grid=element_blank(),
        plot_title=element_text(size=11),
        legend_title=element_text(size=10),
        legend_text=element_text(size=9),
    )
)


def save_heatmap(p, stem: str, width: float, height: float) -> None:
    out = EXPRESSION_DIR / stem
    p.save(Path(str(out) + ".png"), width=width, height=height, dpi=PLOT_DPI, verbose=False)
    p.save(Path(str(out) + ".pdf"), width=width, height=height, verbose=False)
    print(f"saved {out.name}.png/.pdf")


strat_heat = corr_df.loc[
    corr_df["stratum"].eq("by_n_introns") & corr_df["n_intron_group"].ne("1")
].copy()
strat_heat["n_intron_group"] = pd.Categorical(
    strat_heat["n_intron_group"],
    categories=[g for g in INTRON_N_GROUP_ORDER if g != "1"],
    ordered=True,
)
strat_heat["label"] = strat_heat["spearman_rho"].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
p_strat = (
    ggplot(strat_heat, aes(x="n_intron_group", y="sample", fill="spearman_rho"))
    + geom_tile(color="white", size=0.4)
    + geom_text(aes(label="label"), size=8)
    + scale_fill_gradient2(
        low="#B2182B",
        mid="white",
        high="#2166AC",
        midpoint=0,
        limits=(-0.30, 0.30),
        name="Spearman ρ",
    )
    + labs(
        x="Introns per gene",
        y="",
        title="Intron position vs spliced ratio (within intron-count class)",
    )
    + theme_heat
)
save_heatmap(p_strat, "intron_index_vs_spliced_ratio.correlation.heatmap", 7.2, 3.8)

overall_heat = corr_df.loc[corr_df["stratum"].eq("all_introns")].copy()
overall_heat["feature_label"] = pd.Categorical(
    overall_heat["feature"].map(FEATURE_LABELS),
    categories=["Intron position", "Introns per gene"],
    ordered=True,
)
overall_heat["label"] = overall_heat["spearman_rho"].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
p_overall = (
    ggplot(overall_heat, aes(x="feature_label", y="sample", fill="spearman_rho"))
    + geom_tile(color="white", size=0.4)
    + geom_text(aes(label="label"), size=9)
    + scale_fill_gradient2(
        low="#B2182B",
        mid="white",
        high="#2166AC",
        midpoint=0,
        limits=(-0.60, 0.60),
        name="Spearman ρ",
    )
    + labs(
        x="",
        y="",
        title="Pooled intron-level correlation vs spliced ratio",
    )
    + theme_heat
)
save_heatmap(p_overall, "intron_number_vs_spliced_ratio.correlation.heatmap", 4.6, 3.8)

mean_heat = heat_df.copy()
mean_heat["intron_index_group"] = pd.Categorical(
    mean_heat["intron_index"].map(intron_index_group),
    categories=INTRON_INDEX_GROUP_ORDER,
    ordered=True,
)
mean_heat = (
    mean_heat.dropna(subset=["intron_index_group"])
    .groupby(["sample", "intron_index_group"], observed=True, as_index=False)
    .agg(mean_spliced_ratio=("spliced_ratio", "mean"), n=("spliced_ratio", "size"))
)
mean_heat["label"] = mean_heat["mean_spliced_ratio"].map(lambda v: f"{v:.2f}")
mean_path = SPLICING_DIR / "intron_index_vs_spliced_ratio.mean.tsv"
mean_heat.to_csv(mean_path, sep="\t", index=False)
p_mean = (
    ggplot(mean_heat, aes(x="intron_index_group", y="sample", fill="mean_spliced_ratio"))
    + geom_tile(color="white", size=0.4)
    + geom_text(aes(label="label"), size=8)
    + scale_fill_gradient(low="#F7FBFF", high="#08306B", limits=(0.4, 1.0), name="Mean spliced ratio")
    + labs(x="Intron position", y="", title="Mean intron spliced ratio by intron position")
    + theme_heat
)
save_heatmap(p_mean, "intron_index_vs_spliced_ratio.mean.heatmap", 7.2, 3.8)
display(p_strat)


# %% cell 21
_prov.mark(21)
# intron_total_bp × TPM bins vs spliced_ratio — one boxplot panel per sample
for sample in PLOT_SAMPLES:
    sdf = expression_df2.loc[expression_df2["sample"] == sample].copy()
    sdf = sdf.dropna(subset=["intron_total_bp", "TPM", "spliced_ratio"])
    if len(sdf) < 25:
        print(f"Skip {sample} intron_total boxplot: n={len(sdf)}")
        continue
    sdf["intron_total_bp_bin"] = pd.qcut(
        sdf["intron_total_bp"], q=5, labels=["I1", "I2", "I3", "I4", "I5"], duplicates="drop"
    )
    sdf["TPM_bin"] = pd.qcut(
        sdf["TPM"], q=5, labels=["T1", "T2", "T3", "T4", "T5"], duplicates="drop"
    )
    box_counts = (
        sdf.groupby(["intron_total_bp_bin", "TPM_bin"], observed=False)
        .agg(n_genes=("spliced_ratio", "size"), y_top=("spliced_ratio", "max"))
        .reset_index()
    )
    box_counts["label"] = box_counts["n_genes"].astype(str)
    box_counts["y_label"] = np.minimum(box_counts["y_top"] + 0.04, 1.14)
    p1 = (
        ggplot(sdf, aes(x="TPM_bin", y="spliced_ratio"))
        + geom_boxplot(fill="#4F88C6")
        + geom_text(
            data=box_counts,
            mapping=aes(x="TPM_bin", y="y_label", label="label"),
            size=7,
            inherit_aes=False,
        )
        + facet_wrap("intron_total_bp_bin", ncol=5)
        + theme_bw()
        + theme(
            axis_text=element_text(size=10, color="black"),
            axis_title=element_text(size=12, color="black"),
            panel_grid=element_blank(),
            plot_title=element_text(size=11),
        )
        + labs(x="TPM bin", y="Spliced Ratio", title=sample)
        + scale_y_continuous(limits=(0, 1.1), breaks=np.arange(0, 1.2, 0.2))
    )
    out = EXPRESSION_DIR / f"{sample}.elongating.boxplot.intron_total_bp_vs_spliced_ratio"
    p1.save(Path(str(out) + ".png"), width=6, height=3.5, dpi=PLOT_DPI)
    p1.save(Path(str(out) + ".pdf"), width=6, height=3.5)
    print(f"saved {out.name}")


# %% cell 22
_prov.mark(22)
# exon2_bp bins × TPM / exonN — one boxplot panel per sample
def _bin_n_exons(n):
    if pd.isna(n):
        return pd.NA
    n = int(n)
    if 2 <= n <= 3:
        return "EN1"
    if 4 <= n <= 6:
        return "EN2"
    if 8 <= n <= 10:
        return "EN3"
    if 11 <= n <= 15:
        return "EN4"
    if n > 15:
        return "EN5"
    return pd.NA


for sample in PLOT_SAMPLES:
    sdf = expression_df2.loc[expression_df2["sample"] == sample].copy()
    sdf = sdf.dropna(subset=["exon2_bp", "TPM", "spliced_ratio", "n_exons"])
    if len(sdf) < 25:
        print(f"Skip {sample} exon2 boxplots: n={len(sdf)}")
        continue
    sdf["exon2_bp_bin"] = pd.qcut(
        sdf["exon2_bp"], q=5, labels=["E21", "E22", "E23", "E24", "E25"], duplicates="drop"
    )
    sdf["TPM_bin"] = pd.qcut(
        sdf["TPM"], q=5, labels=["T1", "T2", "T3", "T4", "T5"], duplicates="drop"
    )
    sdf["exonN_bin"] = pd.Categorical(
        sdf["n_exons"].apply(_bin_n_exons),
        categories=["EN1", "EN2", "EN3", "EN4", "EN5"],
        ordered=True,
    )

    TPM_box_counts = (
        sdf.groupby(["exon2_bp_bin", "TPM_bin"], observed=False)
        .agg(n_genes=("spliced_ratio", "size"), y_top=("spliced_ratio", "max"))
        .reset_index()
    )
    TPM_box_counts["label"] = TPM_box_counts["n_genes"].astype(str)
    TPM_box_counts["y_label"] = np.minimum(TPM_box_counts["y_top"] + 0.04, 1.14)

    exonN_box_counts = (
        sdf.groupby(["exon2_bp_bin", "exonN_bin"], observed=False)
        .agg(n_genes=("spliced_ratio", "size"), y_top=("spliced_ratio", "max"))
        .reset_index()
    )
    exonN_box_counts["label"] = exonN_box_counts["n_genes"].astype(str)
    exonN_box_counts["y_label"] = np.minimum(exonN_box_counts["y_top"] + 0.04, 1.14)

    p1 = (
        ggplot(sdf, aes(x="TPM_bin", y="spliced_ratio"))
        + geom_boxplot(fill="#4F88C6")
        + geom_text(
            data=TPM_box_counts,
            mapping=aes(x="TPM_bin", y="y_label", label="label"),
            size=7,
            inherit_aes=False,
        )
        + facet_wrap("exon2_bp_bin", ncol=5)
        + theme_bw()
        + theme(
            axis_text=element_text(size=10, color="black"),
            axis_title=element_text(size=12, color="black"),
            panel_grid=element_blank(),
            plot_title=element_text(size=11),
        )
        + labs(x="TPM bin", y="Spliced Ratio", title=sample)
        + scale_y_continuous(limits=(0, 1.1), breaks=np.arange(0, 1.2, 0.2))
    )
    out1 = EXPRESSION_DIR / f"{sample}.elongating.boxplot.exon2_bp_vs_spliced_ratio"
    p1.save(Path(str(out1) + ".png"), width=6, height=3.5, dpi=PLOT_DPI)
    p1.save(Path(str(out1) + ".pdf"), width=6, height=3.5)

    p2 = (
        ggplot(sdf.dropna(subset=["exonN_bin"]), aes(x="exonN_bin", y="spliced_ratio"))
        + geom_boxplot(fill="#4F88C6")
        + geom_text(
            data=exonN_box_counts,
            mapping=aes(x="exonN_bin", y="y_label", label="label"),
            size=7,
            inherit_aes=False,
        )
        + facet_wrap("exon2_bp_bin", ncol=5)
        + theme_bw()
        + theme(
            axis_text=element_text(size=10, color="black"),
            axis_title=element_text(size=12, color="black"),
            panel_grid=element_blank(),
            plot_title=element_text(size=11),
        )
        + labs(x="Exon-number bin", y="Spliced Ratio", title=sample)
        + scale_y_continuous(limits=(0, 1.1), breaks=np.arange(0, 1.2, 0.2))
    )
    out2 = EXPRESSION_DIR / f"{sample}.elongating.boxplot.exonN_vs_spliced_ratio"
    p2.save(Path(str(out2) + ".png"), width=6, height=3.5, dpi=PLOT_DPI)
    p2.save(Path(str(out2) + ".pdf"), width=6, height=3.5)
    print(f"saved {out1.name} / {out2.name}")


# %% cell 23
_prov.mark(23)
# Draft Bayesian intron-coupling model (not wired to this notebook's tables).
# Needs per-intron spliced/retained counts: k_spliced, k_retained.
# Set RUN_PYMC_DEMO = True only if you want a toy demo fit.

RUN_PYMC_DEMO = False

if not RUN_PYMC_DEMO:
    print(
        "Skip PyMC cell: define k_spliced / k_retained from multi-intron "
        "classify counts, or set RUN_PYMC_DEMO = True for a toy demo."
    )
else:
    import pymc as pm

    # Toy observed counts for intron 1/2/3 (replace with real data)
    k_spliced = np.array([80, 65, 70])
    k_retained = np.array([20, 35, 30])

    with pm.Model() as co_tx_model:
        mu = pm.Normal("mu", 0, 1, shape=3)
        beta = pm.Normal("beta_coupling", 0, 0.5, shape=2)
        psi1 = pm.Deterministic("psi1", pm.math.sigmoid(mu[0]))
        psi2 = pm.Deterministic("psi2", pm.math.sigmoid(mu[1] + beta[0] * psi1))
        psi3 = pm.Deterministic("psi3", pm.math.sigmoid(mu[2] + beta[1] * psi2))
        phi = pm.HalfNormal("phi", 10)
        for i, (psi, k_sp, k_ret) in enumerate(
            zip([psi1, psi2, psi3], k_spliced, k_retained)
        ):
            alpha_bb = psi * phi
            beta_bb = (1 - psi) * phi
            pm.BetaBinomial(
                f"R{i+1}",
                n=int(k_sp + k_ret),
                alpha=alpha_bb,
                beta=beta_bb,
                observed=int(k_sp),
            )
        trace = pm.sample(2000, tune=1000)
