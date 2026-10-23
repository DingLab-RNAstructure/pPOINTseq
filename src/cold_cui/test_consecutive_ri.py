#!/usr/bin/env python3
"""Statistical tests for consecutive intron retention (Warm vs Cold).

Three tests:

1. Library rate. Replicate-level Welch t-test and quasi-binomial GLM on
   consecutive-RI / valid-P5 (and / RI) using consecutive_ri.by_sample.tsv.
2. Gene rate. Per-transcript quasi-binomial GLM of n_consec / n_valid on
   group, BH-adjusted, from consecutive_ri.gene_nv_vs_c5.tsv.
3. Same-molecule coupling. For every read that has a spliced/retained call
   on both members of an intron pair, fill a 2x2 table. Adjacent pairs
   (distance = 1) are the consecutive test; distances 2 and 3 are controls.
   Odds ratio > 1 means retaining intron k makes retaining k+d more likely
   than independent retention. Observed vs expected consecutive pairs given
   the number of retained calls on the same molecule is a clustering check.

Usage:
    python src/cold_cui/test_consecutive_ri.py
    python src/cold_cui/test_consecutive_ri.py --skip-classify
    python src/cold_cui/test_consecutive_ri.py --min-gene-reads 20 --delta-rate 0.1
"""
from __future__ import annotations

import argparse
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning

_src = next(p for p in Path(__file__).resolve().parents if (p / "project_config.py").is_file())
sys.path.insert(0, str(_src))
import bootstrap
bootstrap.setup(__file__)

from project_config import RESULT_DIRS, SAMPLES, classify_tsv  # noqa: E402

# ============================================================
# CONSTANTS
# ============================================================

EXCLUDED_CLASSES = frozenset({"invalid_p5_start"})
SAMPLE_GROUPS = {
    "Warm": [s for s in SAMPLES if str(s).startswith("Warm")],
    "Cold": [s for s in SAMPLES if str(s).startswith("Cold")],
}
GROUP_BY_SAMPLE = {s: g for g, reps in SAMPLE_GROUPS.items() for s in reps}
PAIR_DISTANCES = (1, 2, 3)
CLASSIFY_COLUMNS = ("splicing_class", "spliced_introns", "retained_introns")

# ============================================================
# CLI
# ============================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    default_dir = RESULT_DIRS["alternative_splicing"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-table",
        type=Path,
        default=default_dir / "consecutive_ri.by_sample.tsv",
        help="Per-sample consecutive-RI counts",
    )
    parser.add_argument(
        "--gene-table",
        type=Path,
        default=default_dir / "consecutive_ri.gene_nv_vs_c5.tsv",
        help="Per-gene consecutive-RI counts by replicate",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_dir,
        help=f"Output directory (default: {default_dir})",
    )
    parser.add_argument(
        "--min-gene-reads",
        type=int,
        default=20,
        help="Minimum mean valid reads per group for gene GLM (default: 20)",
    )
    parser.add_argument(
        "--min-gene-events",
        type=int,
        default=10,
        help="Minimum total consecutive-RI reads across samples (default: 10)",
    )
    parser.add_argument(
        "--fdr",
        type=float,
        default=0.05,
        help="BH FDR threshold for gene tests (default: 0.05)",
    )
    parser.add_argument(
        "--delta-rate",
        type=float,
        default=0.1,
        help="Minimum |Warm-Cold| consecutive-RI rate for a gene call (default: 0.1)",
    )
    parser.add_argument(
        "--skip-classify",
        action="store_true",
        help="Skip the classify-TSV coupling scan",
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


def parse_intron_list(value: object) -> set[int]:
    """Parse a comma-separated intron-index field."""
    if not isinstance(value, str) or value == "":
        return set()
    if "," not in value:
        try:
            return {int(value)}
        except ValueError:
            return set()
    return {int(token) for token in value.split(",") if token}


def save_pub(fig: plt.Figure, stem: Path) -> None:
    """Write PDF/PNG with editable text."""
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")


# ============================================================
# STATISTICS
# ============================================================


def welch_group_test(values, groups, label: str) -> dict:
    """Welch t-test of Warm versus Cold replicate values."""
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    nv = values[groups == "Warm"]
    c5 = values[groups == "Cold"]
    t_stat, p_value = stats.ttest_ind(nv, c5, equal_var=False)
    return {
        "test": label,
        "n_Warm": int(nv.size),
        "n_Cold": int(c5.size),
        "mean_Warm": float(np.mean(nv)),
        "mean_Cold": float(np.mean(c5)),
        "delta_Warm_minus_Cold": float(np.mean(nv) - np.mean(c5)),
        "t": float(t_stat),
        "p": float(p_value),
    }


def ttest_vs_value(values, mu: float, label: str) -> dict:
    """One-sample t-test against a fixed null value."""
    values = np.asarray(values, dtype=float)
    t_stat, p_value = stats.ttest_1samp(values, mu)
    return {
        "test": label,
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "mu": mu,
        "t": float(t_stat),
        "p": float(p_value),
    }


def quasi_binomial_group(
    n_success: np.ndarray,
    n_total: np.ndarray,
    group: np.ndarray,
    df_prior: float = 10.0,
) -> dict:
    """Quasi-binomial GLM: success/total ~ Warm vs Cold, with scale floored at 1."""
    n_fail = n_total - n_success
    endog = np.column_stack([n_success, n_fail])
    nv = (np.asarray(group) == "Warm").astype(float)
    design = sm.add_constant(nv)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PerfectSeparationWarning)
        fit = sm.GLM(endog, design, family=sm.families.Binomial()).fit()
    coef = float(fit.params[1])
    if not np.isfinite(coef) or abs(coef) > 15:
        raise ValueError("unstable GLM coefficient")
    df_resid = max(float(fit.df_resid), 1.0)
    x2_scale = float(fit.pearson_chi2 / df_resid)
    scale = max((df_prior + df_resid * x2_scale) / (df_prior + df_resid), 1.0)
    se = float(fit.bse[1]) * np.sqrt(scale)
    z_stat = coef / se if se else np.nan
    p_value = float(2 * stats.norm.sf(abs(z_stat))) if np.isfinite(z_stat) else np.nan
    return {
        "coef_Warm": coef,
        "se_Warm": se,
        "p": p_value,
        "or_Warm_vs_Cold": float(np.exp(coef)),
        "scale": scale,
    }


def odds_ratio_ci(ss: float, sr: float, rs: float, rr: float) -> tuple[float, float, float, float]:
    """Haldane-Anscombe odds ratio and 95% CI. Table: i\\j spliced/retained."""
    a, b, c, d = ss + 0.5, sr + 0.5, rs + 0.5, rr + 0.5
    log_or = np.log((a * d) / (b * c))
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    or_hat = float(np.exp(log_or))
    lo = float(np.exp(log_or - 1.96 * se))
    hi = float(np.exp(log_or + 1.96 * se))
    chi = ((ss * rr - sr * rs) ** 2) * (ss + sr + rs + rr) / max(
        (ss + sr) * (rs + rr) * (ss + rs) * (sr + rr), 1.0
    )
    p_chi = float(stats.chi2.sf(chi, df=1))
    return or_hat, lo, hi, p_chi


# ============================================================
# LIBRARY AND GENE TESTS
# ============================================================


def test_library_rates(sample_tbl: pd.DataFrame) -> pd.DataFrame:
    """Replicate-level tests of consecutive-RI rates."""
    rows = []
    for column, label in (
        ("pct_valid_consec", "consecutive_rate_of_valid_P5"),
        ("pct_ri_consec", "consecutive_rate_of_RI_reads"),
    ):
        rows.append(welch_group_test(sample_tbl[column], sample_tbl["group"], f"Welch_{column}"))
        glm = quasi_binomial_group(
            sample_tbl["n_consec"].to_numpy(dtype=float),
            sample_tbl["n_valid" if "valid" in column else "n_ri"].to_numpy(dtype=float),
            sample_tbl["group"].to_numpy(),
        )
        rows.append(
            {
                "test": f"quasi_binomial_{label}",
                "n_Warm": 3,
                "n_Cold": 3,
                "mean_Warm": float(sample_tbl.loc[sample_tbl["group"].eq("Warm"), column].mean()),
                "mean_Cold": float(sample_tbl.loc[sample_tbl["group"].eq("Cold"), column].mean()),
                "delta_Warm_minus_Cold": float(
                    sample_tbl.loc[sample_tbl["group"].eq("Warm"), column].mean()
                    - sample_tbl.loc[sample_tbl["group"].eq("Cold"), column].mean()
                ),
                "t": glm["coef_Warm"] / glm["se_Warm"] if glm["se_Warm"] else np.nan,
                "p": glm["p"],
                "or_Warm_vs_Cold": glm["or_Warm_vs_Cold"],
                "scale": glm["scale"],
            }
        )
    return pd.DataFrame(rows)


def gene_long_table(gene_tbl: pd.DataFrame) -> pd.DataFrame:
    """Melt per-sample n_consec / n_valid into one row per gene-sample."""
    rows = []
    for sample in SAMPLES:
        n_valid = gene_tbl[f"n_valid.{sample}"]
        n_consec = gene_tbl[f"n_consec.{sample}"]
        part = pd.DataFrame(
            {
                "gene_id": gene_tbl["gene_id"],
                "sample": sample,
                "group": GROUP_BY_SAMPLE[sample],
                "n_valid": n_valid,
                "n_consec": n_consec,
            }
        )
        rows.append(part)
    long = pd.concat(rows, ignore_index=True)
    long["n_valid"] = pd.to_numeric(long["n_valid"], errors="coerce")
    long["n_consec"] = pd.to_numeric(long["n_consec"], errors="coerce").fillna(0.0)
    long = long.dropna(subset=["n_valid"])
    long = long.loc[long["n_valid"] > 0].copy()
    long["n_consec"] = long["n_consec"].clip(upper=long["n_valid"])
    return long


def test_genes(
    gene_tbl: pd.DataFrame,
    min_reads: int,
    min_events: int,
    fdr: float,
    delta_rate: float,
) -> pd.DataFrame:
    """Per-gene quasi-binomial GLM of consecutive-RI rate."""
    keep = gene_tbl.copy()
    if "n_intron" in keep.columns:
        keep = keep.loc[keep["n_intron"].fillna(0) >= 2]
    if {"nv_mean_n_valid", "c5_mean_n_valid"}.issubset(keep.columns):
        keep = keep.loc[
            (keep["nv_mean_n_valid"] >= min_reads)
            & (keep["c5_mean_n_valid"] >= min_reads)
        ]
    long = gene_long_table(keep)
    records = []
    for gene_id, frame in long.groupby("gene_id", sort=False):
        nv = frame.loc[frame["group"].eq("Warm")]
        c5 = frame.loc[frame["group"].eq("Cold")]
        if nv["sample"].nunique() < 2 or c5["sample"].nunique() < 2:
            continue
        if nv["n_valid"].mean() < min_reads or c5["n_valid"].mean() < min_reads:
            continue
        if nv["n_consec"].sum() + c5["n_consec"].sum() < min_events:
            continue
        try:
            glm = quasi_binomial_group(
                frame["n_consec"].to_numpy(dtype=float),
                frame["n_valid"].to_numpy(dtype=float),
                frame["group"].to_numpy(),
            )
        except Exception:
            continue
        if not np.isfinite(glm["p"]):
            continue
        rate_nv = float(nv["n_consec"].sum() / max(nv["n_valid"].sum(), 1.0))
        rate_c5 = float(c5["n_consec"].sum() / max(c5["n_valid"].sum(), 1.0))
        records.append(
            {
                "gene_id": gene_id,
                "n_reps_Warm": int(nv["sample"].nunique()),
                "n_reps_Cold": int(c5["sample"].nunique()),
                "mean_valid_Warm": float(nv["n_valid"].mean()),
                "mean_valid_Cold": float(c5["n_valid"].mean()),
                "n_consec_Warm": float(nv["n_consec"].sum()),
                "n_consec_Cold": float(c5["n_consec"].sum()),
                "rate_Warm": rate_nv,
                "rate_Cold": rate_c5,
                "delta_rate": rate_nv - rate_c5,
                "coef_Warm": glm["coef_Warm"],
                "or_Warm_vs_Cold": glm["or_Warm_vs_Cold"],
                "p": glm["p"],
                "scale": glm["scale"],
            }
        )
    if not records:
        return pd.DataFrame()
    out = pd.DataFrame(records)
    out["padj"] = multipletests(out["p"].to_numpy(dtype=float), method="fdr_bh")[1]
    out["significant"] = (out["padj"] < fdr) & (out["delta_rate"].abs() >= delta_rate)
    out["direction"] = np.select(
        [out["significant"] & (out["delta_rate"] > 0), out["significant"] & (out["delta_rate"] < 0)],
        ["higher_consecutive_in_Warm", "higher_consecutive_in_Cold"],
        default="not_significant",
    )
    return out.sort_values(["padj", "p", "gene_id"]).reset_index(drop=True)


# ============================================================
# COUPLING FROM CLASSIFY TABLES
# ============================================================


def scan_coupling(sample: str) -> dict:
    """Accumulate adjacent-pair 2x2 tables and clustering totals for one sample."""
    pair_counts = {
        distance: {"ss": 0, "sr": 0, "rs": 0, "rr": 0} for distance in PAIR_DISTANCES
    }
    n_reads = 0
    n_multi = 0
    obs_adj = 0.0
    exp_adj = 0.0
    path = classify_tsv(sample)
    print(f"[INFO] {sample} coupling scan {path.name}", file=sys.stderr, flush=True)
    for chunk in pd.read_csv(
        path,
        sep="\t",
        usecols=list(CLASSIFY_COLUMNS),
        dtype=str,
        chunksize=250_000,
    ):
        chunk = chunk.loc[~chunk["splicing_class"].isin(EXCLUDED_CLASSES)]
        spliced_col = chunk["spliced_introns"].fillna("").to_numpy()
        retained_col = chunk["retained_introns"].fillna("").to_numpy()
        for spliced_text, retained_text in zip(spliced_col, retained_col):
            if not spliced_text and not retained_text:
                continue
            spliced = parse_intron_list(spliced_text)
            retained = parse_intron_list(retained_text)
            called = spliced | retained
            n_called = len(called)
            if n_called == 0:
                continue
            n_reads += 1
            if n_called < 2:
                continue
            n_ret = len(retained)
            if n_ret >= 2:
                n_multi += 1
            for distance in PAIR_DISTANCES:
                counts = pair_counts[distance]
                for idx in called:
                    other = idx + distance
                    if other not in called:
                        continue
                    i_ret = idx in retained
                    j_ret = other in retained
                    if i_ret and j_ret:
                        counts["rr"] += 1
                    elif i_ret:
                        counts["rs"] += 1
                    elif j_ret:
                        counts["sr"] += 1
                    else:
                        counts["ss"] += 1
            adj_slots = sum(1 for idx in called if (idx + 1) in called)
            obs = sum(1 for idx in retained if (idx + 1) in retained)
            if n_ret >= 2 and adj_slots:
                exp = adj_slots * n_ret * (n_ret - 1) / (n_called * (n_called - 1))
                obs_adj += obs
                exp_adj += exp
    return {
        "sample": sample,
        "group": GROUP_BY_SAMPLE[sample],
        "n_reads_called": n_reads,
        "n_multi_ri": n_multi,
        "obs_adjacent_pairs": obs_adj,
        "exp_adjacent_pairs": exp_adj,
        "obs_over_exp": obs_adj / exp_adj if exp_adj > 0 else np.nan,
        "pairs": pair_counts,
    }


def coupling_tables(scans: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-sample 2x2 summaries and clustering ratios."""
    pair_rows = []
    cluster_rows = []
    for scan in scans:
        cluster_rows.append({k: scan[k] for k in scan if k != "pairs"})
        for distance, counts in scan["pairs"].items():
            or_hat, lo, hi, p_chi = odds_ratio_ci(
                counts["ss"], counts["sr"], counts["rs"], counts["rr"]
            )
            pair_rows.append(
                {
                    "sample": scan["sample"],
                    "group": scan["group"],
                    "distance": distance,
                    "ss": counts["ss"],
                    "sr": counts["sr"],
                    "rs": counts["rs"],
                    "rr": counts["rr"],
                    "n_pairs": counts["ss"] + counts["sr"] + counts["rs"] + counts["rr"],
                    "odds_ratio": or_hat,
                    "or_ci95_lo": lo,
                    "or_ci95_hi": hi,
                    "p_chi2": p_chi,
                    "log2_or": float(np.log2(or_hat)),
                }
            )
    return pd.DataFrame(pair_rows), pd.DataFrame(cluster_rows)


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


def plot_tests(
    sample_tbl: pd.DataFrame,
    library_tests: pd.DataFrame,
    pair_tbl: pd.DataFrame | None,
    cluster_tbl: pd.DataFrame | None,
    stem: Path,
) -> None:
    """Plot library rates and adjacent-pair odds ratios."""
    apply_publication_style()
    n_panels = 2 if pair_tbl is None else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(3.1 * n_panels, 2.8), squeeze=False)
    axes = axes[0]
    colors = {"Warm": "#0072B2", "Cold": "#D55E00"}

    axis = axes[0]
    for i, sample in enumerate(sample_tbl["sample"]):
        row = sample_tbl.iloc[i]
        axis.scatter(
            0 if row["group"] == "Warm" else 1,
            row["pct_valid_consec"],
            color=colors[row["group"]],
            s=22,
            zorder=3,
        )
    means = sample_tbl.groupby("group")["pct_valid_consec"].mean()
    axis.hlines(means["Warm"], -0.15, 0.15, color=colors["Warm"], lw=1.4)
    axis.hlines(means["Cold"], 0.85, 1.15, color=colors["Cold"], lw=1.4)
    welch = library_tests.loc[library_tests["test"].eq("Welch_pct_valid_consec")].iloc[0]
    axis.set_xticks([0, 1], ["Warm", "Cold"])
    axis.set_ylabel("Consecutive RI (% of valid P5)")
    axis.set_title(f"Library rate\nWelch p = {welch['p']:.3g}", fontsize=7)

    axis = axes[1]
    jitter = {"Warm": -0.12, "Cold": 0.12}
    if pair_tbl is None:
        axis.set_axis_off()
    else:
        for _, row in pair_tbl.iterrows():
            x = row["distance"] + jitter[row["group"]]
            axis.scatter(x, row["odds_ratio"], color=colors[row["group"]], s=18, zorder=3)
            axis.plot(
                [x, x],
                [row["or_ci95_lo"], row["or_ci95_hi"]],
                color=colors[row["group"]],
                lw=0.8,
            )
        axis.axhline(1.0, color="#767676", lw=0.6, ls="--")
        axis.set_xticks(list(PAIR_DISTANCES))
        axis.set_xlabel("Intron distance")
        axis.set_ylabel("Odds ratio (retain i and i+d)")
        axis.set_title("Same-molecule coupling", fontsize=7)

    if n_panels == 3 and cluster_tbl is not None:
        axis = axes[2]
        for _, row in cluster_tbl.iterrows():
            axis.scatter(
                0 if row["group"] == "Warm" else 1,
                row["obs_over_exp"],
                color=colors[row["group"]],
                s=22,
                zorder=3,
            )
        axis.axhline(1.0, color="#767676", lw=0.6, ls="--")
        axis.set_xticks([0, 1], ["Warm", "Cold"])
        axis.set_ylabel("Observed / expected adjacent RI")
        cluster_test = welch_group_test(
            cluster_tbl["obs_over_exp"], cluster_tbl["group"], "cluster"
        )
        axis.set_title(f"Clustering vs independence\nWelch p = {cluster_test['p']:.3g}", fontsize=7)

    fig.tight_layout()
    save_pub(fig, stem)
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    """Run library, gene, and coupling tests for consecutive RI."""
    args = parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_tbl = pd.read_csv(args.sample_table, sep="\t")
    required_columns(
        sample_tbl,
        {"sample", "group", "n_valid", "n_ri", "n_consec", "pct_valid_consec", "pct_ri_consec"},
        args.sample_table,
    )
    library_tests = test_library_rates(sample_tbl)
    library_path = out_dir / "consecutive_ri.library_test.tsv"
    library_tests.to_csv(library_path, sep="\t", index=False)
    print(f"[INFO] wrote {library_path}", file=sys.stderr)
    for row in library_tests.itertuples(index=False):
        print(
            f"[INFO] {row.test}: Warm={row.mean_Warm:.3g} Cold={row.mean_Cold:.3g} "
            f"delta={row.delta_Warm_minus_Cold:.3g} p={row.p:.3g}",
            file=sys.stderr,
        )

    gene_tbl = pd.read_csv(args.gene_table, sep="\t")
    required_columns(gene_tbl, {"gene_id"}, args.gene_table)
    gene_tests = test_genes(
        gene_tbl,
        args.min_gene_reads,
        args.min_gene_events,
        args.fdr,
        args.delta_rate,
    )
    gene_path = out_dir / "consecutive_ri.gene_glm.tsv"
    gene_tests.to_csv(gene_path, sep="\t", index=False)
    n_sig = int(gene_tests["significant"].sum()) if not gene_tests.empty else 0
    n_nv = int((gene_tests["direction"] == "higher_consecutive_in_Warm").sum()) if not gene_tests.empty else 0
    n_c5 = int((gene_tests["direction"] == "higher_consecutive_in_Cold").sum()) if not gene_tests.empty else 0
    print(f"[INFO] wrote {gene_path}", file=sys.stderr)
    print(
        f"[INFO] gene GLM tested={len(gene_tests):,} significant={n_sig:,} "
        f"higher_in_Warm={n_nv:,} higher_in_Cold={n_c5:,}",
        file=sys.stderr,
    )

    pair_tbl = None
    cluster_tbl = None
    pair_path = out_dir / "consecutive_ri.coupling_2x2.by_sample.tsv"
    cluster_path = out_dir / "consecutive_ri.clustering.by_sample.tsv"
    if not args.skip_classify:
        with ProcessPoolExecutor(max_workers=min(6, len(SAMPLES))) as pool:
            scans = list(pool.map(scan_coupling, list(SAMPLES)))
        pair_tbl, cluster_tbl = coupling_tables(scans)
        pair_tbl.to_csv(pair_path, sep="\t", index=False)
        cluster_tbl.to_csv(cluster_path, sep="\t", index=False)
        print(f"[INFO] wrote {pair_path}", file=sys.stderr)
        print(f"[INFO] wrote {cluster_path}", file=sys.stderr)
        adj = pair_tbl.loc[pair_tbl["distance"].eq(1)]
        nv_or = adj.loc[adj["group"].eq("Warm"), "odds_ratio"]
        c5_or = adj.loc[adj["group"].eq("Cold"), "odds_ratio"]
        nv_cl = cluster_tbl.loc[cluster_tbl["group"].eq("Warm"), "obs_over_exp"]
        c5_cl = cluster_tbl.loc[cluster_tbl["group"].eq("Cold"), "obs_over_exp"]
        extra_rows = [
            welch_group_test(np.log(adj["odds_ratio"]), adj["group"], "log_OR_adjacent_Warm_vs_Cold"),
            welch_group_test(cluster_tbl["obs_over_exp"], cluster_tbl["group"], "obs_over_exp_Warm_vs_Cold"),
            ttest_vs_value(np.log(nv_or), 0.0, "log_OR_adjacent_Warm_vs_0"),
            ttest_vs_value(np.log(c5_or), 0.0, "log_OR_adjacent_Cold_vs_0"),
            ttest_vs_value(nv_cl, 1.0, "obs_over_exp_Warm_vs_1"),
            ttest_vs_value(c5_cl, 1.0, "obs_over_exp_Cold_vs_1"),
        ]
        extra = pd.DataFrame(extra_rows)
        extra_path = out_dir / "consecutive_ri.coupling_group_test.tsv"
        extra.to_csv(extra_path, sep="\t", index=False)
        print(f"[INFO] wrote {extra_path}", file=sys.stderr)
        print(
            f"[INFO] adjacent OR: Warm={nv_or.mean():.3g} Cold={c5_or.mean():.3g} "
            f"Welch logOR p={extra_rows[0]['p']:.3g}",
            file=sys.stderr,
        )
        print(
            f"[INFO] clustering obs/exp: Warm={nv_cl.mean():.3g} Cold={c5_cl.mean():.3g} "
            f"Welch p={extra_rows[1]['p']:.3g}",
            file=sys.stderr,
        )
    elif pair_path.is_file() and cluster_path.is_file():
        pair_tbl = pd.read_csv(pair_path, sep="\t")
        cluster_tbl = pd.read_csv(cluster_path, sep="\t")
        print(f"[INFO] reused {pair_path.name} and {cluster_path.name}", file=sys.stderr)

    plot_tests(
        sample_tbl,
        library_tests,
        pair_tbl,
        cluster_tbl,
        out_dir / "consecutive_ri.statistical_tests",
    )
    print(f"[INFO] wrote {out_dir / 'consecutive_ri.statistical_tests'}.pdf/.png", file=sys.stderr)


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()
