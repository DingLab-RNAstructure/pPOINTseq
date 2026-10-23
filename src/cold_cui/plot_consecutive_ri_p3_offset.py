"""
3' end of consecutive-RI reads relative to the last retained intron
-------------------------------------------------------------------
Input: per-sample ``*.elongating.classify.tsv``
  alternative_splicing_details  splicing_class  p3_feat

Logic:
  A read is consecutive RI when ``RI:intron=`` indices include at least one
  neighbouring pair. The anchor is the 3' end of the longest such run; ties
  use the downstream-most run. ``p3_feat`` is then placed relative to that intron:

    next_exon            exon immediately downstream          (+1 exon)
    next_intron          intron immediately downstream        (+1 intron)
    further_downstream   a later exon or intron               (Downstream)
    termination          annotated termination                (Termination)
    still_in_or_before_run   3' end still inside or upstream of the run
    other                promoter, intergenic, or unparsed

  ``invalid_p5_start`` is excluded. The bar chart uses each replicate's
  percentage of valid P5 reads, then the unweighted mean of the three
  replicates. ``still_in_or_before_run`` and ``other`` stay in the tables
  and are omitted from the bar chart.

Output:
  consecutive_ri.p3_offset.by_sample.tsv
  consecutive_ri.p3_offset.library.by_sample.tsv
  consecutive_ri.p3_offset.library.mean.tsv
  consecutive_ri.p3_offset.library.bar.pdf
  consecutive_ri.p3_offset.library.bar.png

Usage:
  python src/cold_cui/plot_consecutive_ri_p3_offset.py
  python src/cold_cui/plot_consecutive_ri_p3_offset.py \\
      --classify-dir results/read_through \\
      -o results/alternative_splicing
"""

from __future__ import annotations

import argparse
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
    scale_x_discrete,
    scale_y_continuous,
    theme,
    theme_bw,
)

RI_INTRON_RE = re.compile(r"RI:intron=(\d+)")
P3_FEAT_RE = re.compile(r"^(exon|intron)_(\d+)$")
INVALID_CLASS = "invalid_p5_start"
CHUNK_SIZE = 200_000
GROUPS = ("Warm", "Cold")
Warm_COLOR = "#29A2C6"
Cold_COLOR = "#FFCB18"
ALL_CLASSES = (
    "next_exon",
    "next_intron",
    "further_downstream",
    "termination",
    "still_in_or_before_run",
    "other",
)
PLOT_CLASSES = (
    "next_exon",
    "next_intron",
    "further_downstream",
    "termination",
)
PLOT_LABELS = {
    "next_exon": "+1 exon",
    "next_intron": "+1 intron",
    "further_downstream": "Downstream",
    "termination": "Termination",
}


def parse_args() -> argparse.Namespace:
    """Return command-line arguments."""
    root = next(p for p in Path(__file__).resolve().parents if (p / "src" / "project_config.py").is_file())
    parser = argparse.ArgumentParser(
        description=(
            "Plot where the 3' end of consecutive-RI reads sits relative to "
            "the last intron in the consecutive run."
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
        default=4.4,
        help="Figure width in inches (default: 4.4)",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=4.4,
        help="Figure height in inches (default: 4.4)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="PNG resolution (default: 600)",
    )
    parser.add_argument(
        "--y-max",
        type=float,
        default=6.0,
        help="Y-axis upper limit in percent of valid P5 (default: 6)",
    )
    return parser.parse_args()


def sample_group(sample: str) -> str:
    """Return Warm or Cold from a sample id."""
    if sample.startswith("Warm"):
        return "Warm"
    if sample.startswith("Cold"):
        return "Cold"
    raise ValueError(f"Cannot assign group for sample {sample!r}")


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


def last_consecutive_intron(details: str) -> int | None:
    """Return the 3' end of the longest neighbouring RI run.

    When several runs share that length, use the downstream-most run.
    """
    ordered = sorted({int(idx) for idx in RI_INTRON_RE.findall(details)})
    best_length = 1
    best_end = None
    run = 1
    for left, right in zip(ordered, ordered[1:]):
        if right == left + 1:
            run += 1
            if run >= best_length:
                best_length = run
                best_end = right
        else:
            run = 1
    return best_end


def p3_vs_last_ri(p3_feat: str, last_ri: int) -> str:
    """Classify a 3' feature by how far it sits past the last consecutive RI."""
    if p3_feat == "termination":
        return "termination"
    match = P3_FEAT_RE.match(p3_feat)
    if match is None:
        return "other"
    kind = match.group(1)
    # Exon k+1 and intron k+1 are the features immediately downstream of intron k.
    offset = int(match.group(2)) - last_ri
    if offset <= 0:
        return "still_in_or_before_run"
    if offset == 1 and kind == "exon":
        return "next_exon"
    if offset == 1 and kind == "intron":
        return "next_intron"
    return "further_downstream"


def count_p3_offset(path: Path) -> tuple[int, Counter]:
    """Return valid-P5 reads and consecutive-RI counts by 3' class."""
    n_valid = 0
    counts: Counter = Counter()
    for chunk in pd.read_csv(
        path,
        sep="\t",
        usecols=["alternative_splicing_details", "splicing_class", "p3_feat"],
        dtype=str,
        chunksize=CHUNK_SIZE,
        keep_default_na=False,
    ):
        valid = chunk["splicing_class"] != INVALID_CLASS
        n_valid += int(valid.sum())
        kept = chunk.loc[valid]
        for details, p3_feat in zip(kept["alternative_splicing_details"], kept["p3_feat"]):
            if "RI:intron=" not in details:
                continue
            last_ri = last_consecutive_intron(details)
            if last_ri is None:
                continue
            counts[p3_vs_last_ri(p3_feat, last_ri)] += 1
    return n_valid, counts


def by_sample_rows(sample: str, n_valid: int, counts: Counter) -> tuple[list[dict], list[dict]]:
    """Return consecutive-RI percentages and valid-P5 percentages for one sample."""
    n_consec = sum(counts.values())
    group = sample_group(sample)
    of_consec = []
    of_library = []
    for label in ALL_CLASSES:
        n_reads = int(counts[label])
        of_consec.append(
            {
                "sample": sample,
                "group": group,
                "p3_vs_last_ri": label,
                "n_reads": n_reads,
                "percent": 100.0 * n_reads / n_consec if n_consec else float("nan"),
            }
        )
        of_library.append(
            {
                "sample": sample,
                "group": group,
                "p3_vs_last_ri": label,
                "n_reads": n_reads,
                "n_valid": n_valid,
                "percent": 100.0 * n_reads / n_valid if n_valid else float("nan"),
            }
        )
    return of_consec, of_library


def mean_of_replicates(library: pd.DataFrame) -> pd.DataFrame:
    """Return the unweighted mean of replicate percentages for the plotted classes."""
    plotted = library.loc[library["p3_vs_last_ri"].isin(PLOT_CLASSES)].copy()
    mean = plotted.groupby(["group", "p3_vs_last_ri"], as_index=False, observed=True).agg(
        percent=("percent", "mean"),
        n_reads=("n_reads", "sum"),
    )
    mean["p3_vs_last_ri"] = pd.Categorical(mean["p3_vs_last_ri"], categories=PLOT_CLASSES, ordered=True)
    mean["group"] = pd.Categorical(mean["group"], categories=GROUPS, ordered=True)
    return mean.sort_values(["group", "p3_vs_last_ri"])


def write_table(frame: pd.DataFrame, path: Path) -> None:
    """Write a tab-separated table with a header."""
    frame.to_csv(path, sep="\t", index=False)
    print(f"[INFO] wrote {path}", file=sys.stderr)


def plot_library(mean: pd.DataFrame, stem: Path, width: float, height: float, dpi: int, y_max: float) -> None:
    """Write the grouped bar chart as PDF and PNG."""
    plot_df = mean.copy()
    plot_df["position"] = plot_df["p3_vs_last_ri"].astype(str).map(PLOT_LABELS)
    plot_df["position"] = pd.Categorical(
        plot_df["position"],
        categories=[PLOT_LABELS[label] for label in PLOT_CLASSES],
        ordered=True,
    )
    theme_common = (
        theme_bw()
        + theme(
            axis_text=element_text(size=10, color="black"),
            axis_title=element_text(size=12, color="black"),
            plot_title=element_text(size=12, color="black", ha="center"),
            panel_grid=element_blank(),
            legend_title=element_text(size=10),
            legend_text=element_text(size=9),
            strip_background=element_rect(fill="white", color="black"),
            axis_text_x=element_text(rotation=40, ha="right"),
            legend_position="right",
        )
    )
    plot = (
        ggplot(plot_df, aes(x="position", y="percent", fill="group"))
        + geom_col(position=position_dodge(width=0.8), width=0.7, color="black", size=0.2)
        + scale_fill_manual(values={"Warm": Warm_COLOR, "Cold": Cold_COLOR})
        + scale_x_discrete(drop=False)
        + scale_y_continuous(expand=(0, 0), limits=(0, y_max))
        + labs(x="", y="Valid P5 reads (%)", fill="", title="Pol II position")
        + theme_common
    )
    plot.save(Path(str(stem) + ".png"), width=width, height=height, dpi=dpi, verbose=False)
    plot.save(Path(str(stem) + ".pdf"), width=width, height=height, verbose=False)
    print(f"[INFO] wrote {stem.name}.pdf/.png", file=sys.stderr)


def write_outputs(
    samples: list[tuple[str, Path]],
    output_dir: Path,
    width: float = 4.4,
    height: float = 4.4,
    dpi: int = 600,
    y_max: float = 6.0,
) -> pd.DataFrame:
    """Count 3' classes, write the tables, and save the library bar chart."""
    output_dir.mkdir(parents=True, exist_ok=True)
    consec_rows = []
    library_rows = []
    ordered = sorted(samples, key=lambda item: (0 if item[0].startswith("Warm") else 1, item[0]))
    for sample, path in ordered:
        print(f"[INFO] counting {path.name}", file=sys.stderr)
        n_valid, counts = count_p3_offset(path)
        n_consec = sum(counts.values())
        print(
            f"[INFO] {sample}: valid P5={n_valid:,}; consecutive RI={n_consec:,} "
            f"({100.0 * n_consec / n_valid if n_valid else 0:.2f}%)",
            file=sys.stderr,
        )
        of_consec, of_library = by_sample_rows(sample, n_valid, counts)
        consec_rows.extend(of_consec)
        library_rows.extend(of_library)

    by_sample = pd.DataFrame(consec_rows)
    library = pd.DataFrame(library_rows)
    sample_order = [sample for sample, _path in ordered]
    library["sample"] = pd.Categorical(library["sample"], categories=sample_order, ordered=True)
    library["p3_vs_last_ri"] = pd.Categorical(
        library["p3_vs_last_ri"], categories=ALL_CLASSES, ordered=True
    )
    library = library.sort_values(["sample", "p3_vs_last_ri"])
    mean = mean_of_replicates(library)

    write_table(by_sample, output_dir / "consecutive_ri.p3_offset.by_sample.tsv")
    write_table(library, output_dir / "consecutive_ri.p3_offset.library.by_sample.tsv")
    write_table(mean, output_dir / "consecutive_ri.p3_offset.library.mean.tsv")
    plot_library(
        mean,
        output_dir / "consecutive_ri.p3_offset.library.bar",
        width,
        height,
        dpi,
        y_max,
    )
    return mean


def main() -> None:
    args = parse_args()
    write_outputs(
        classify_paths(args.classify_dir),
        args.output,
        width=args.width,
        height=args.height,
        dpi=args.dpi,
        y_max=args.y_max,
    )


if __name__ == "__main__":
    main()
