#!/usr/bin/env python3
"""Replicate-level tests of first–last intron splicing coupling.

Input:
    results_p5_newsplicing/exon_intron/first_last_intron_coupling.by_rep.tsv
    from notebook 8. Each row is one replicate 2x2 table of first vs last
    intron state (n11 both spliced, n10 first spliced/last retained,
    n01 first retained/last spliced, n00 both retained).

Core logic:
    Fisher exact tests on pooled reads are not reported. Tests use the three
    biological replicates per genotype:
    1. one-sample t of logOR, residual, and forward delta-P against the
       independence null (OR = 1, residual = 0, delta-P = 0);
    2. Welch t of those statistics between Warm and Cold.
    Odds ratios use the Haldane–Anscombe +0.5 correction.

Output:
    first_last_intron_coupling.test.tsv
    first_last_intron_coupling.or.{pdf,png}

Usage:
    python src/overall_splicing/test_first_last_coupling.py
    python src/overall_splicing/test_first_last_coupling.py --input path/to/by_rep.tsv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

_src = next(p for p in Path(__file__).resolve().parents if (p / "project_config.py").is_file())
sys.path.insert(0, str(_src))
import bootstrap
bootstrap.setup(__file__)

from project_config import RESULT_DIRS  # noqa: E402

# ============================================================
# CONSTANTS
# ============================================================

REQUIRED_COLUMNS = {
    "sample",
    "n11",
    "n10",
    "n01",
    "n00",
    "phi",
    "odds_ratio",
    "fwd_deltaP",
    "residual",
}
Warm_COLOR = "#29A2C6"
Cold_COLOR = "#FFCB18"

# ============================================================
# CLI
# ============================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    default_dir = RESULT_DIRS["exon_intron"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=default_dir / "first_last_intron_coupling.by_rep.tsv",
        help="Per-replicate first–last 2x2 table",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_dir,
        help="Directory for test table and plot",
    )
    return parser.parse_args()


# ============================================================
# I/O HELPERS
# ============================================================


def required_columns(frame: pd.DataFrame, names: set[str], path: Path) -> None:
    """Exit if required columns are missing."""
    missing = names - set(frame.columns)
    if missing:
        raise SystemExit(f"[ERROR] {path} missing columns: {', '.join(sorted(missing))}")


def sample_group(sample: str) -> str:
    """Return Warm or Cold from a sample name."""
    return "Cold" if str(sample).startswith("Cold") else "Warm"


# ============================================================
# DATA PROCESSING
# ============================================================


def haldane_or_ci(n11: float, n10: float, n01: float, n00: float) -> tuple[float, float, float]:
    """Return Haldane–Anscombe OR and 95% CI for a 2x2 table."""
    a, b, c, d = n11 + 0.5, n10 + 0.5, n01 + 0.5, n00 + 0.5
    log_or = np.log((a * d) / (b * c))
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    or_hat = float(np.exp(log_or))
    return or_hat, float(np.exp(log_or - 1.96 * se)), float(np.exp(log_or + 1.96 * se))


def annotate_replicates(frame: pd.DataFrame) -> pd.DataFrame:
    """Add genotype and Haldane–Anscombe OR columns."""
    out = frame.copy()
    out["group"] = out["sample"].map(sample_group)
    ors = [
        haldane_or_ci(row.n11, row.n10, row.n01, row.n00)
        for row in out.itertuples(index=False)
    ]
    out["or_ha"] = [item[0] for item in ors]
    out["or_ci95_lo"] = [item[1] for item in ors]
    out["or_ci95_hi"] = [item[2] for item in ors]
    out["log_or"] = np.log(out["or_ha"])
    return out


def welch_group_test(values: np.ndarray, groups: np.ndarray, label: str) -> dict:
    """Welch t-test of Warm versus Cold replicate values."""
    warm = values[groups == "Warm"]
    cold = values[groups == "Cold"]
    t_stat, p_value = stats.ttest_ind(warm, cold, equal_var=False)
    return {
        "test": label,
        "n_Warm": int(warm.size),
        "n_Cold": int(cold.size),
        "mean_Warm": float(np.mean(warm)),
        "mean_Cold": float(np.mean(cold)),
        "delta_Warm_minus_Cold": float(np.mean(warm) - np.mean(cold)),
        "t": float(t_stat),
        "p": float(p_value),
        "n": np.nan,
        "mean": np.nan,
        "mu": np.nan,
    }


def ttest_vs_value(values: np.ndarray, mu: float, label: str) -> dict:
    """One-sample t-test against a fixed null value."""
    t_stat, p_value = stats.ttest_1samp(values, mu)
    return {
        "test": label,
        "n_Warm": np.nan,
        "n_Cold": np.nan,
        "mean_Warm": np.nan,
        "mean_Cold": np.nan,
        "delta_Warm_minus_Cold": np.nan,
        "t": float(t_stat),
        "p": float(p_value),
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "mu": mu,
    }


def run_tests(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the replicate-level first–last coupling tests."""
    groups = frame["group"].to_numpy()
    rows = [
        ttest_vs_value(frame.loc[frame["group"].eq("Warm"), "log_or"].to_numpy(), 0.0, "log_OR_Warm_vs_0"),
        ttest_vs_value(frame.loc[frame["group"].eq("Cold"), "log_or"].to_numpy(), 0.0, "log_OR_Cold_vs_0"),
        welch_group_test(frame["log_or"].to_numpy(), groups, "log_OR_Warm_vs_Cold"),
        ttest_vs_value(frame.loc[frame["group"].eq("Warm"), "residual"].to_numpy(), 0.0, "residual_Warm_vs_0"),
        ttest_vs_value(frame.loc[frame["group"].eq("Cold"), "residual"].to_numpy(), 0.0, "residual_Cold_vs_0"),
        welch_group_test(frame["residual"].to_numpy(), groups, "residual_Warm_vs_Cold"),
        ttest_vs_value(frame.loc[frame["group"].eq("Warm"), "fwd_deltaP"].to_numpy(), 0.0, "fwd_deltaP_Warm_vs_0"),
        ttest_vs_value(frame.loc[frame["group"].eq("Cold"), "fwd_deltaP"].to_numpy(), 0.0, "fwd_deltaP_Cold_vs_0"),
        welch_group_test(frame["fwd_deltaP"].to_numpy(), groups, "fwd_deltaP_Warm_vs_Cold"),
        ttest_vs_value(frame.loc[frame["group"].eq("Warm"), "phi"].to_numpy(), 0.0, "phi_Warm_vs_0"),
        ttest_vs_value(frame.loc[frame["group"].eq("Cold"), "phi"].to_numpy(), 0.0, "phi_Cold_vs_0"),
        welch_group_test(frame["phi"].to_numpy(), groups, "phi_Warm_vs_Cold"),
    ]
    return pd.DataFrame(rows)


# ============================================================
# WRITING OUTPUTS
# ============================================================


def plot_odds_ratios(frame: pd.DataFrame, stem: Path) -> None:
    """Plot per-replicate first–last odds ratios with 95% CI."""
    plot_df = frame.sort_values(["group", "sample"]).reset_index(drop=True)
    colors = [Warm_COLOR if group == "Warm" else Cold_COLOR for group in plot_df["group"]]
    fig, axis = plt.subplots(figsize=(4.4, 3.4))
    x = np.arange(len(plot_df))
    y = plot_df["or_ha"].to_numpy()
    lo = y - plot_df["or_ci95_lo"].to_numpy()
    hi = plot_df["or_ci95_hi"].to_numpy() - y
    axis.axhline(1.0, color="#666666", ls="--", lw=0.8, zorder=1)
    axis.errorbar(x, y, yerr=[lo, hi], fmt="none", ecolor="#272727", elinewidth=0.8, capsize=2, zorder=2)
    axis.scatter(x, y, c=colors, s=36, zorder=3, edgecolors="#272727", linewidths=0.4)
    axis.set_xticks(x)
    axis.set_xticklabels(plot_df["sample"].astype(str), rotation=45, ha="right")
    axis.set_ylabel("First–last odds ratio")
    axis.set_yscale("log")
    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=Warm_COLOR, markeredgecolor="#272727", label="Warm"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=Cold_COLOR, markeredgecolor="#272727", label="Cold"),
    ]
    axis.legend(handles=handles, frameon=False)
    fig.tight_layout()
    for suffix in (".pdf", ".png"):
        fig.savefig(Path(str(stem) + suffix), dpi=300, bbox_inches="tight")
    plt.close(fig)


def log_test_summary(tests: pd.DataFrame) -> None:
    """Print the main coupling tests to stderr."""
    for _, row in tests.iterrows():
        if str(row["test"]).endswith("_vs_0"):
            print(
                f"[INFO] {row['test']}: mean={row['mean']:.4g}  t={row['t']:.3g}  p={row['p']:.3g}",
                file=sys.stderr,
            )
        else:
            print(
                f"[INFO] {row['test']}: Warm={row['mean_Warm']:.4g}  Cold={row['mean_Cold']:.4g}  "
                f"t={row['t']:.3g}  p={row['p']:.3g}",
                file=sys.stderr,
            )


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    """Test first–last intron coupling at the replicate level."""
    args = parse_args()
    if not args.input.is_file():
        raise SystemExit(f"[ERROR] missing {args.input}; run notebook 8 coupling cell first")
    raw = pd.read_csv(args.input, sep="\t")
    required_columns(raw, REQUIRED_COLUMNS, args.input)
    frame = annotate_replicates(raw)
    tests = run_tests(frame)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table_path = args.output_dir / "first_last_intron_coupling.test.tsv"
    tests.to_csv(table_path, sep="\t", index=False)
    annotated_path = args.output_dir / "first_last_intron_coupling.by_rep.annotated.tsv"
    frame.to_csv(annotated_path, sep="\t", index=False)
    plot_odds_ratios(frame, args.output_dir / "first_last_intron_coupling.or")
    print(f"[INFO] wrote {table_path}", file=sys.stderr)
    print(f"[INFO] wrote {annotated_path}", file=sys.stderr)
    print(
        f"[INFO] wrote {args.output_dir / 'first_last_intron_coupling.or'}.pdf/.png",
        file=sys.stderr,
    )
    log_test_summary(tests)


if __name__ == "__main__":
    main()
