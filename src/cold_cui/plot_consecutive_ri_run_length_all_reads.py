"""
Consecutive-RI run length as a percent of all classified reads
----------------------------------------------------------------
Input: per-sample ``*.elongating.classify.tsv``
  alternative_splicing_details  splicing_class

Logic:
  A read has consecutive intron retention when ``RI:intron=`` indices include
  at least one neighbouring pair (k, k+1). The bar uses the longest such run,
  binned as 2, 3, 4, 5, or >=6. Reads with splicing_class ``invalid_p5_start``
  stay in both the numerator and the denominator. The denominator is every
  row in the classify table. Group bars are the mean of the three replicate
  percentages.

Output:
  consecutive_ri.run_length.all_reads.by_replicate.tsv
  consecutive_ri.run_length.all_reads.mean.tsv
  consecutive_ri.run_length.all_reads.bar.pdf
  consecutive_ri.run_length.all_reads.bar.png

Usage:
    python src/cold_cui/plot_consecutive_ri_run_length_all_reads.py
    python src/cold_cui/plot_consecutive_ri_run_length_all_reads.py \\
        --classify-dir results/read_through \\
        -o results/alternative_splicing
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from plotnine import (
    aes,
    element_blank,
    element_rect,
    element_text,
    geom_col,
    ggplot,
    labs,
    position_dodge,
    scale_fill_manual,
    scale_y_continuous,
    theme,
    theme_bw,
)

RI_INTRON_RE = re.compile(r"RI:intron=(\d+)")
RUN_BINS = ("2", "3", "4", "5", "≥6")
GROUPS = ("Warm", "Cold")
Warm_COLOR = "#29A2C6"
Cold_COLOR = "#FFCB18"
INVALID_CLASS = "invalid_p5_start"
CHUNK_SIZE = 200_000


def parse_args() -> argparse.Namespace:
    """Return command-line arguments."""
    root = next(p for p in Path(__file__).resolve().parents if (p / "src" / "project_config.py").is_file())
    parser = argparse.ArgumentParser(
        description=(
            "Plot consecutive-RI run length as a percent of all classified "
            "reads, including invalid_p5_start."
        )
    )
    parser.add_argument(
        "--classify-dir",
        type=Path,
        default=root / "results" / "read_through",
        help="Directory of *.elongating.classify.tsv (default: results/read_through)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=root / "results" / "alternative_splicing",
        help="Output directory (default: results/alternative_splicing)",
    )
    parser.add_argument(
        "--width",
        type=float,
        default=4.6,
        help="Figure width in inches (default: 4.6)",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=3.4,
        help="Figure height in inches (default: 3.4)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="PNG resolution (default: 600)",
    )
    return parser.parse_args()


def sample_group(sample: str) -> str:
    """Return Warm or Cold from a sample id."""
    if sample.startswith("Warm"):
        return "Warm"
    if sample.startswith("Cold"):
        return "Cold"
    raise ValueError(f"Cannot assign group for sample {sample!r}")


def run_bin(length: int) -> str:
    """Return the longest-run bin label for a consecutive RI length."""
    if length >= 6:
        return "≥6"
    return str(length)


def longest_consecutive_run(details: str) -> int:
    """Return the longest neighbouring RI run, or 0 when no pair is consecutive."""
    ordered = sorted({int(idx) for idx in RI_INTRON_RE.findall(details)})
    if len(ordered) < 2:
        return 0
    best = 1
    run = 1
    for left, right in zip(ordered, ordered[1:]):
        if right == left + 1:
            run += 1
            best = max(best, run)
        else:
            run = 1
    if best < 2:
        return 0
    return best


def classify_paths(classify_dir: Path) -> list[tuple[str, Path]]:
    """Return (sample, path) pairs for elongating classify tables."""
    paths = sorted(classify_dir.glob("*.elongating.classify.tsv"))
    if not paths:
        raise FileNotFoundError(f"No *.elongating.classify.tsv in {classify_dir}")
    samples = [(path.name.split(".", 1)[0], path) for path in paths]
    unknown = [sample for sample, _path in samples if not sample.startswith(("Warm", "Cold"))]
    if unknown:
        raise ValueError(f"Unrecognised sample prefixes: {', '.join(unknown)}")
    return samples


def count_run_lengths(path: Path) -> tuple[int, int, int, Counter]:
    """Return total reads, invalid reads, invalid consecutive-RI reads, and run counts."""
    n_reads = 0
    n_invalid = 0
    n_invalid_consec = 0
    run_n: Counter = Counter()
    for chunk in pd.read_csv(
        path,
        sep="\t",
        usecols=["alternative_splicing_details", "splicing_class"],
        dtype=str,
        chunksize=CHUNK_SIZE,
        keep_default_na=False,
    ):
        n_reads += len(chunk)
        invalid = chunk["splicing_class"].eq(INVALID_CLASS)
        n_invalid += int(invalid.sum())
        for value, is_invalid in zip(chunk["alternative_splicing_details"], invalid):
            if "RI:intron=" not in value:
                continue
            length = longest_consecutive_run(value)
            if not length:
                continue
            run_n[length] += 1
            if is_invalid:
                n_invalid_consec += 1
    return n_reads, n_invalid, n_invalid_consec, run_n


def replicate_rows(sample: str, n_reads: int, run_n: Counter) -> list[dict]:
    """Return one percentage row per run-length bin for a sample."""
    binned = Counter()
    for length, count in run_n.items():
        binned[run_bin(length)] += count
    rows = []
    for bin_name in RUN_BINS:
        count = int(binned[bin_name])
        rows.append(
            {
                "sample": sample,
                "group": sample_group(sample),
                "run_bin": bin_name,
                "n_reads": count,
                "n_all": n_reads,
                "percent": 100.0 * count / n_reads if n_reads else float("nan"),
            }
        )
    return rows


def mean_by_group(by_replicate: pd.DataFrame) -> pd.DataFrame:
    """Return the mean replicate percentage for each group and run-length bin."""
    mean = by_replicate.groupby(["group", "run_bin"], as_index=False, observed=True)["percent"].mean()
    mean["run_bin"] = pd.Categorical(mean["run_bin"], categories=RUN_BINS, ordered=True)
    mean["group"] = pd.Categorical(mean["group"], categories=GROUPS, ordered=True)
    return mean.sort_values(["group", "run_bin"])


def write_table(frame: pd.DataFrame, path: Path) -> None:
    """Write a tab-separated table with a header."""
    frame.to_csv(path, sep="\t", index=False)
    print(f"[INFO] wrote {path}", file=sys.stderr)


def y_limit(mean: pd.DataFrame) -> float:
    """Return a y-axis ceiling that keeps the tallest bar inside the panel."""
    peak = float(mean["percent"].max())
    if not math.isfinite(peak) or peak <= 0:
        return 6.0
    return float(math.ceil(peak * 1.15))


def plot_run_length(mean: pd.DataFrame, stem: Path, width: float, height: float, dpi: int) -> None:
    """Write the grouped bar chart as PDF and PNG."""
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
    plot = (
        ggplot(mean, aes(x="run_bin", y="percent", fill="group"))
        + geom_col(position=position_dodge(width=0.8), width=0.7, color="black", size=0.2)
        + scale_fill_manual(values={"Warm": Warm_COLOR, "Cold": Cold_COLOR})
        + scale_y_continuous(expand=(0, 0), limits=(0, y_limit(mean)))
        + labs(x="Longest consecutive RI run", y="All reads (%)", fill="")
        + theme_common
        + theme(legend_position="right")
    )
    plot.save(Path(str(stem) + ".png"), width=width, height=height, dpi=dpi, verbose=False)
    plot.save(Path(str(stem) + ".pdf"), width=width, height=height, verbose=False)
    print(f"[INFO] wrote {stem.name}.pdf/.png", file=sys.stderr)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for sample, path in classify_paths(args.classify_dir):
        print(f"[INFO] counting {path.name}", file=sys.stderr)
        n_reads, n_invalid, n_invalid_consec, run_n = count_run_lengths(path)
        n_consec = sum(run_n.values())
        print(
            f"[INFO] {sample}: reads={n_reads:,}; invalid_p5_start={n_invalid:,} "
            f"({100.0 * n_invalid / n_reads if n_reads else 0:.1f}%); "
            f"consecutive RI={n_consec:,}; "
            f"consecutive RI on invalid_p5_start={n_invalid_consec:,}",
            file=sys.stderr,
        )
        rows.extend(replicate_rows(sample, n_reads, run_n))

    by_replicate = pd.DataFrame(rows)
    by_replicate["run_bin"] = pd.Categorical(
        by_replicate["run_bin"], categories=RUN_BINS, ordered=True
    )
    by_replicate = by_replicate.sort_values(["sample", "run_bin"])
    mean = mean_by_group(by_replicate)

    write_table(by_replicate, args.output / "consecutive_ri.run_length.all_reads.by_replicate.tsv")
    write_table(mean, args.output / "consecutive_ri.run_length.all_reads.mean.tsv")
    plot_run_length(
        mean,
        args.output / "consecutive_ri.run_length.all_reads.bar",
        args.width,
        args.height,
        args.dpi,
    )


if __name__ == "__main__":
    main()
