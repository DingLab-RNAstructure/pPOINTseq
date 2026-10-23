#!/usr/bin/env python3
"""Read-sampling test of the Warm versus Cold consecutive-RI rate.

Input tables, written by the consecutive-RI summary:
  consecutive_ri.by_sample.tsv
    sample, group, n_valid, n_consec
  consecutive_ri.run_length.library.by_replicate.tsv
    sample, group, run_bin, n_reads, n_valid

Each valid-P5 read is one draw, and a read is a success when it carries a
consecutive retained-intron run. Sampling those reads is a hypergeometric
draw from the per-sample counts, so the classify TSV is not read again.

Every replicate is reduced to the same number of reads, the smallest valid-P5
library. In each repeat the statistic is the mean Warm rate minus the mean Cold
rate. The null redraws every library from one shared rate, the pooled
consecutive-RI proportion. A one-sided p-value is the fraction of null
differences at least as large as the observed difference. The percentile
interval is the spread of differences when reads are redrawn from the real
libraries.

Output: a TSV with one row per metric, the null-draw distribution for the
overall rate, and a histogram of that null against the observed difference.

Usage:
    python src/cold_cui/test_consecutive_ri_read_sample.py
    python src/cold_cui/test_consecutive_ri_read_sample.py --n-sim 10000 --seed 1
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

from project_config import RESULT_DIRS


RUN_BIN_ORDER = ("2", "3", "4", "5", "≥6")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    default_dir = RESULT_DIRS["alternative_splicing"]
    parser = argparse.ArgumentParser(
        description="Equal-depth read-sampling test of consecutive RI (Warm > Cold)."
    )
    parser.add_argument(
        "--sample-table",
        type=Path,
        default=default_dir / "consecutive_ri.by_sample.tsv",
        help="Per-sample consecutive-RI counts (default: consecutive_ri.by_sample.tsv)",
    )
    parser.add_argument(
        "--run-table",
        type=Path,
        default=default_dir / "consecutive_ri.run_length.library.by_replicate.tsv",
        help="Per-sample run-length counts with a valid-P5 denominator",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=default_dir / "consecutive_ri.read_sample",
        help="Output path prefix (default: results/alternative_splicing/consecutive_ri.read_sample)",
    )
    parser.add_argument(
        "--n-sim",
        type=int,
        default=10000,
        help="Number of random read draws (default: 10000)",
    )
    parser.add_argument("--seed", type=int, default=20260712, help="RNG seed (default: 20260712)")
    return parser.parse_args()


def log(message: str) -> None:
    """Print a progress message to stderr."""
    print(f"[INFO] {message}", file=sys.stderr)


def load_counts(sample_path: Path, run_path: Path) -> pd.DataFrame:
    """Return one row per sample and metric, with event and library sizes."""
    samples = pd.read_csv(sample_path, sep="\t")
    required = {"sample", "group", "n_valid", "n_consec"}
    missing = required - set(samples.columns)
    if missing:
        raise SystemExit(f"[ERROR] {sample_path} missing columns: {', '.join(sorted(missing))}")
    overall = samples[["sample", "group", "n_valid", "n_consec"]].copy()
    overall["metric"] = "consecutive"
    overall = overall.rename(columns={"n_consec": "n_event"})

    runs = pd.read_csv(run_path, sep="\t")
    required = {"sample", "group", "run_bin", "n_reads", "n_valid"}
    missing = required - set(runs.columns)
    if missing:
        raise SystemExit(f"[ERROR] {run_path} missing columns: {', '.join(sorted(missing))}")
    runs = runs.rename(columns={"n_reads": "n_event", "run_bin": "metric"})
    runs = runs[["sample", "group", "n_valid", "n_event", "metric"]]
    counts = pd.concat([overall, runs], ignore_index=True)
    counts["n_valid"] = counts["n_valid"].astype(int)
    counts["n_event"] = counts["n_event"].astype(int)
    return counts


def metric_order(metrics: list[str]) -> list[str]:
    """Place the overall rate before the run-length bins."""
    ordered = ["consecutive"]
    ordered.extend(bin_name for bin_name in RUN_BIN_ORDER if bin_name in metrics)
    ordered.extend(metric for metric in metrics if metric not in ordered)
    return ordered


def sample_differences(
    n_event: np.ndarray,
    n_valid: np.ndarray,
    is_nv: np.ndarray,
    n_draw: int,
    n_sim: int,
    rng: np.random.Generator,
    shared_rate: float | None,
) -> np.ndarray:
    """Return Warm-minus-Cold rate differences from equal-sized read draws.

    A shared rate draws every library from the same binomial probability.
    Otherwise each library is sampled without replacement from its own reads.
    """
    if shared_rate is None:
        hits = np.column_stack(
            [
                rng.hypergeometric(event, valid - event, n_draw, size=n_sim)
                for event, valid in zip(n_event, n_valid)
            ]
        )
    else:
        hits = rng.binomial(n_draw, shared_rate, size=(n_sim, len(n_event)))
    rates = hits / n_draw
    return rates[:, is_nv].mean(axis=1) - rates[:, ~is_nv].mean(axis=1)


def analytical_p(delta: float, n_draw: int, n_nv: int, n_c5: int, shared_rate: float) -> tuple[float, float]:
    """Return the normal z and one-sided p-value for a shared read rate."""
    variance = shared_rate * (1.0 - shared_rate) / n_draw
    se = np.sqrt(variance / n_nv + variance / n_c5)
    if se == 0:
        return np.nan, np.nan
    z_stat = delta / se
    return float(z_stat), float(stats.norm.sf(z_stat))


def test_metric(
    counts: pd.DataFrame,
    metric: str,
    n_draw: int,
    n_sim: int,
    rng: np.random.Generator,
) -> tuple[dict[str, object], np.ndarray]:
    """Test one event rate and return the summary plus the null differences."""
    sub = counts.loc[counts["metric"].eq(metric)].sort_values("sample")
    n_event = sub["n_event"].to_numpy()
    n_valid = sub["n_valid"].to_numpy()
    is_nv = sub["group"].eq("Warm").to_numpy()
    observed_rate = n_event / n_valid
    delta = float(observed_rate[is_nv].mean() - observed_rate[~is_nv].mean())
    shared_rate = float(n_event.sum() / n_valid.sum())
    observed_draws = sample_differences(n_event, n_valid, is_nv, n_draw, n_sim, rng, None)
    null_draws = sample_differences(
        n_event, n_valid, is_nv, n_draw, n_sim, rng, shared_rate
    )
    z_stat, analytical = analytical_p(delta, n_draw, int(is_nv.sum()), int((~is_nv).sum()), shared_rate)
    monte_carlo = float((np.sum(null_draws >= delta) + 1) / (n_sim + 1))
    summary = {
        "metric": metric,
        "n_draw": n_draw,
        "n_sim": n_sim,
        "n_event_Warm": int(n_event[is_nv].sum()),
        "n_valid_Warm": int(n_valid[is_nv].sum()),
        "n_event_Cold": int(n_event[~is_nv].sum()),
        "n_valid_Cold": int(n_valid[~is_nv].sum()),
        "mean_Warm": float(observed_rate[is_nv].mean()),
        "mean_Cold": float(observed_rate[~is_nv].mean()),
        "delta_Warm_minus_Cold": delta,
        "resample_ci_low": float(np.quantile(observed_draws, 0.025)),
        "resample_ci_high": float(np.quantile(observed_draws, 0.975)),
        "shared_rate": shared_rate,
        "null_mean": float(null_draws.mean()),
        "p_read_one_sided": monte_carlo,
        "z_read": z_stat,
        "p_analytical_one_sided": analytical,
    }
    return summary, null_draws


def plot_null(null_draws: np.ndarray, delta: float, n_draw: int, path: Path) -> None:
    """Draw the null distribution of the equal-depth rate difference."""
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.hist(null_draws * 100, bins=40, color="#BDBDBD", edgecolor="white", linewidth=0.4)
    ax.axvline(delta * 100, color="#B2182B", linewidth=1.4)
    ax.set_xlabel("Warm − Cold consecutive RI (percentage points)")
    ax.set_ylabel("Null draws")
    ax.set_title(f"Shared-rate null, {n_draw:,} reads per replicate")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    log(f"Wrote {path}")


def main() -> None:
    """Draw equal numbers of reads and test the Warm minus Cold rate."""
    args = parse_args()
    counts = load_counts(args.sample_table, args.run_table)
    library_size = (
        counts.loc[counts["metric"].eq("consecutive"), ["sample", "n_valid"]]
        .drop_duplicates()
    )
    n_draw = int(library_size["n_valid"].min())
    limiting = library_size.loc[library_size["n_valid"].eq(n_draw), "sample"].iloc[0]
    log(f"Drawing {n_draw:,} valid-P5 reads per replicate, limited by {limiting}")

    rng = np.random.default_rng(args.seed)
    rows = []
    overall_null = None
    overall_delta = None
    for metric in metric_order(counts["metric"].drop_duplicates().tolist()):
        summary, null_draws = test_metric(counts, metric, n_draw, args.n_sim, rng)
        rows.append(summary)
        log(
            f"{metric}: Warm {100 * summary['mean_Warm']:.2f}% , "
            f"Cold {100 * summary['mean_Cold']:.2f}% , "
            f"delta {100 * summary['delta_Warm_minus_Cold']:.2f} pp, "
            f"read p {summary['p_read_one_sided']:.3g}"
        )
        if metric == "consecutive":
            overall_null = null_draws
            overall_delta = float(summary["delta_Warm_minus_Cold"])

    result = pd.DataFrame(rows)
    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_path = prefix.with_name(prefix.name + ".summary.tsv")
    result.to_csv(summary_path, sep="\t", index=False)
    log(f"Wrote {summary_path}")
    null_path = prefix.with_name(prefix.name + ".null_draws.tsv")
    pd.DataFrame({"delta_Warm_minus_Cold": overall_null}).to_csv(null_path, sep="\t", index=False)
    log(f"Wrote {null_path}")
    plot_null(
        overall_null,
        overall_delta,
        n_draw,
        prefix.with_name(prefix.name + ".null.pdf"),
    )
    result.to_csv(sys.stdout, sep="\t", index=False)


if __name__ == "__main__":
    main()
