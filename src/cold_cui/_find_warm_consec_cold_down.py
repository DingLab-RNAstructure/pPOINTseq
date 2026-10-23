#!/usr/bin/env python3
"""Find genes with consecutive intron retention in Warm that drop in Cold."""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

_src = next(p for p in Path(__file__).resolve().parents if (p / "project_config.py").is_file())
sys.path.insert(0, str(_src))
import bootstrap
bootstrap.setup(__file__)

from project_config import (  # noqa: E402
    REF_DIR,
    RESULT_DIRS,
    SAMPLES,
    classify_tsv,
)

EXCLUDED_CLASSES = frozenset({"invalid_p5_start"})
RI_TOKEN = "RI:intron="
SAMPLE_GROUPS = {
    "Warm": [s for s in SAMPLES if str(s).startswith("Warm")],
    "Cold": [s for s in SAMPLES if str(s).startswith("Cold")],
}
GROUP_BY_SAMPLE = {s: g for g, reps in SAMPLE_GROUPS.items() for s in reps}
Warm_SAMPLES = SAMPLE_GROUPS["Warm"]
Cold_SAMPLES = SAMPLE_GROUPS["Cold"]


def parse_ri_introns(value) -> list[int]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return []
    found = []
    for part in text.split(";"):
        token = part.strip()
        if not token.startswith(RI_TOKEN):
            continue
        try:
            found.append(int(token.split("=", 1)[1]))
        except ValueError:
            continue
    return sorted(set(found))


def consecutive_pairs(introns: list[int]) -> list[tuple[int, int]]:
    if len(introns) < 2:
        return []
    s = set(introns)
    return [(i, i + 1) for i in sorted(s) if (i + 1) in s]


def longest_run(introns: list[int]) -> int:
    if not introns:
        return 0
    ordered = sorted(set(introns))
    best = run = 1
    for a, b in zip(ordered, ordered[1:]):
        if b == a + 1:
            run += 1
            best = max(best, run)
        else:
            run = 1
    return best


def n_intron_from_bed(bed_path: Path) -> pd.DataFrame:
    bed = pd.read_csv(
        bed_path,
        sep="\t",
        header=None,
        usecols=[0, 3, 5],
        names=["chrom", "feature", "strand"],
        dtype=str,
    )
    intron = bed.loc[bed["feature"].str.contains("_intron", regex=False)].copy()
    intron["gene_id"] = intron["feature"].str.replace(r"_intron\d+$", "", regex=True)
    n_tbl = intron.groupby("gene_id").size().rename("n_intron").reset_index()
    loc = intron.groupby("gene_id").agg(chrom=("chrom", "first"), strand=("strand", "first")).reset_index()
    return n_tbl.merge(loc, on="gene_id", how="left")


def count_sample(sample: str) -> pd.DataFrame:
    path = classify_tsv(sample)
    n_valid: dict[str, int] = defaultdict(int)
    n_ri: dict[str, int] = defaultdict(int)
    n_consec: dict[str, int] = defaultdict(int)
    max_run: dict[str, int] = defaultdict(int)
    print(f"[{sample}] scanning {path}", flush=True)
    for chunk in pd.read_csv(
        path,
        sep="\t",
        usecols=["gene_id", "splicing_class", "alternative_splicing_details"],
        dtype=str,
        chunksize=250_000,
    ):
        chunk = chunk.loc[~chunk["splicing_class"].isin(EXCLUDED_CLASSES)].copy()
        chunk["gene_id"] = chunk["gene_id"].fillna("").astype(str)
        chunk = chunk.loc[chunk["gene_id"] != ""]
        for gid, n in chunk["gene_id"].value_counts().items():
            n_valid[gid] += int(n)
        details = chunk["alternative_splicing_details"].fillna("")
        ri_mask = details.str.contains(RI_TOKEN, regex=False)
        if not ri_mask.any():
            continue
        ri = chunk.loc[ri_mask]
        for gid, n in ri["gene_id"].value_counts().items():
            n_ri[gid] += int(n)
        for gid, det in zip(ri["gene_id"], ri["alternative_splicing_details"]):
            rii = parse_ri_introns(det)
            if not consecutive_pairs(rii):
                continue
            n_consec[gid] += 1
            max_run[gid] = max(max_run[gid], longest_run(rii))
    rows = []
    for gid, warm in n_valid.items():
        rows.append(
            {
                "sample": sample,
                "group": GROUP_BY_SAMPLE[sample],
                "gene_id": gid,
                "n_valid": warm,
                "n_ri": n_ri[gid],
                "n_consec": n_consec[gid],
                "max_run": max_run[gid],
            }
        )
    print(
        f"[{sample}] genes={len(rows):,} consec_genes={sum(n_consec[g] > 0 for g in n_consec):,} "
        f"consec_reads={sum(n_consec.values()):,}",
        flush=True,
    )
    return pd.DataFrame(rows)


def mean_cols(wide: pd.DataFrame, prefix: str, samples: list[str]) -> pd.Series:
    cols = [f"{prefix}.{s}" for s in samples]
    return wide[cols].mean(axis=1)


def main() -> None:
    sample_tbl = pd.concat([count_sample(s) for s in SAMPLES], ignore_index=True)
    sample_tbl["rate"] = sample_tbl["n_consec"] / sample_tbl["n_valid"].clip(lower=1)

    wide = sample_tbl.pivot_table(
        index="gene_id",
        columns="sample",
        values=["n_valid", "n_consec", "n_ri", "rate", "max_run"],
        aggfunc="first",
    )
    wide.columns = [f"{metric}.{sample}" for metric, sample in wide.columns]
    wide = wide.reset_index()

    wide["warm_mean_n_valid"] = mean_cols(wide, "n_valid", Warm_SAMPLES)
    wide["cold_mean_n_valid"] = mean_cols(wide, "n_valid", Cold_SAMPLES)
    wide["warm_mean_n_consec"] = mean_cols(wide, "n_consec", Warm_SAMPLES)
    wide["cold_mean_n_consec"] = mean_cols(wide, "n_consec", Cold_SAMPLES)
    wide["warm_mean_n_ri"] = mean_cols(wide, "n_ri", Warm_SAMPLES)
    wide["cold_mean_n_ri"] = mean_cols(wide, "n_ri", Cold_SAMPLES)
    wide["warm_mean_rate"] = mean_cols(wide, "rate", Warm_SAMPLES)
    wide["cold_mean_rate"] = mean_cols(wide, "rate", Cold_SAMPLES)
    wide["warm_max_run"] = pd.concat([wide[f"max_run.{s}"] for s in Warm_SAMPLES], axis=1).max(axis=1)
    wide["cold_max_run"] = pd.concat([wide[f"max_run.{s}"] for s in Cold_SAMPLES], axis=1).max(axis=1)
    wide["warm_n_reps_consec"] = sum((wide[f"n_consec.{s}"].fillna(0) > 0).astype(int) for s in Warm_SAMPLES)
    wide["cold_n_reps_consec"] = sum((wide[f"n_consec.{s}"].fillna(0) > 0).astype(int) for s in Cold_SAMPLES)
    wide["rate_diff"] = wide["warm_mean_rate"] - wide["cold_mean_rate"]
    wide["rate_ratio"] = wide["cold_mean_rate"] / wide["warm_mean_rate"].clip(lower=1e-9)
    wide["n_consec_diff"] = wide["warm_mean_n_consec"] - wide["cold_mean_n_consec"]
    wide["locus"] = wide["gene_id"].str.replace(r"\.\d+$", "", regex=True)

    bed_ann = n_intron_from_bed(REF_DIR / "exon_intron_pos.chr.bed")
    wide = wide.merge(bed_ann, on="gene_id", how="left")

    outdir = RESULT_DIRS["alternative_splicing"]
    outdir.mkdir(parents=True, exist_ok=True)
    gene_path = outdir / "consecutive_ri.gene_warm_vs_cold.tsv"
    wide.sort_values(["rate_diff", "n_consec_diff"], ascending=False).to_csv(
        gene_path, sep="\t", index=False
    )

    cand = wide.loc[
        (wide["warm_n_reps_consec"] == 3)
        & (wide["warm_mean_n_valid"] >= 30)
        & (wide["cold_mean_n_valid"] >= 20)
        & (wide["warm_mean_n_consec"] >= 8)
        & (wide["warm_mean_rate"] >= 0.05)
        & (wide["rate_ratio"] <= 0.4)
        & (wide["n_intron"].between(3, 15))
    ].copy()
    cand = cand.sort_values(["rate_diff", "n_consec_diff"], ascending=False)
    cand_path = outdir / "consecutive_ri.warm_present_cold_down.plottable.tsv"
    cand.to_csv(cand_path, sep="\t", index=False)

    show_cols = [
        "gene_id",
        "n_intron",
        "warm_mean_n_valid",
        "cold_mean_n_valid",
        "warm_mean_n_consec",
        "cold_mean_n_consec",
        "warm_mean_rate",
        "cold_mean_rate",
        "rate_diff",
        "rate_ratio",
        "warm_max_run",
        "cold_n_reps_consec",
    ]
    print(f"saved {gene_path}", flush=True)
    print(f"saved {cand_path}  n={len(cand)}", flush=True)
    print(cand[show_cols].head(20).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    hit = wide.loc[wide["gene_id"] == "AT2G11225.1", show_cols]
    print("\nAT2G11225.1")
    print(hit.to_string(index=False) if not hit.empty else "not found")


if __name__ == "__main__":
    main()
