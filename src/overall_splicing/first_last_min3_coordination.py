#!/usr/bin/env python3
"""First–last intron coupling for genes with n_introns >= 3.

A read enters the 2×2 only when the first and last introns both have a call
in spliced_introns or retained_introns, and the read 5' end is in promoter
or exon_1. Genes with two introns are left out, so first and last are not
an adjacent pair. Replicate tests and the odds-ratio plot call
``src/test_first_last_coupling.py``.

Outputs go to ``results/exon_intron/min3/``:
    first_last_intron_coupling.by_rep.tsv
    first_last_intron_coupling.by_rep.annotated.tsv
    first_last_intron_coupling.test.tsv
    first_last_intron_coupling.or.{pdf,png}
    first_last_conditional_prob.{pdf,png}
    first_last_conditional_flow.{pdf,png}
"""
import os
import subprocess
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

import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch, PathPatch, Rectangle  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402
from plotnine import *  # noqa: E402,F401,F403
from scipy import stats  # noqa: E402

import project_config  # noqa: E402
from project_config import RESULT_DIRS, SAMPLES, classify_tsv  # noqa: E402

MIN_N = 3
VALID_P5_FEATURES = ("promoter", "exon_1")
EXON_DIR = RESULT_DIRS["exon_intron"]
OUT_DIR = EXON_DIR / "min3"
REPLICATES = {s: classify_tsv(s) for s in SAMPLES}

theme_common = theme(
    axis_text=element_text(size=12, color="black"),
    axis_title=element_text(size=14, color="black"),
    panel_grid=element_blank(),
)

FLOW_COLORS = {
    "n11": "#2E8B57",  # first spliced -> last spliced
    "n10": "#C44E52",  # first spliced -> last retained
    "n01": "#4C78A8",  # first retained -> last spliced
    "n00": "#9E9E9E",  # first retained -> last retained
}


# %% cell 1
_prov.mark(1)
# --- Coupling helpers. Reads with n_introns < MIN_N are skipped. ------------


def _panel_intron_set(value) -> set[int]:
    """Parse a comma-separated intron-index field."""
    if pd.isna(value) or str(value).strip() == "":
        return set()
    return {int(item) for item in str(value).split(",")}


def _phi(n11, n10, n01, n00):
    n11, n10, n01, n00 = map(float, (n11, n10, n01, n00))
    num = n11 * n00 - n10 * n01
    den = float(np.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)))
    return num / den if den > 0 else np.nan


def compute_first_last_coupling(df: pd.DataFrame, min_n: int) -> dict:
    """2x2 of first-intron vs last-intron state, over reads deciding both."""
    n11 = n10 = n01 = n00 = 0
    for row in df.itertuples(index=False):
        n = getattr(row, "n_introns")
        if pd.isna(n) or int(n) < min_n:
            continue
        n = int(n)
        spl = _panel_intron_set(getattr(row, "spliced_introns"))
        ret = _panel_intron_set(getattr(row, "retained_introns"))
        if 1 in spl:
            sf = 1
        elif 1 in ret:
            sf = 0
        else:
            continue
        if n in spl:
            sl = 1
        elif n in ret:
            sl = 0
        else:
            continue
        if sf and sl:
            n11 += 1
        elif sf and not sl:
            n10 += 1
        elif not sf and sl:
            n01 += 1
        else:
            n00 += 1

    total = n11 + n10 + n01 + n00
    out = {"n11": n11, "n10": n10, "n01": n01, "n00": n00, "n_reads": total}
    if total == 0:
        return out
    out["P_last|first_spliced"] = n11 / (n11 + n10) if (n11 + n10) else np.nan
    out["P_last|first_retained"] = n01 / (n01 + n00) if (n01 + n00) else np.nan
    out["P_first|last_spliced"] = n11 / (n11 + n01) if (n11 + n01) else np.nan
    out["P_first|last_retained"] = n10 / (n10 + n00) if (n10 + n00) else np.nan
    out["fwd_deltaP"] = out["P_last|first_spliced"] - out["P_last|first_retained"]
    out["rev_deltaP"] = out["P_first|last_spliced"] - out["P_first|last_retained"]
    out["phi"] = _phi(n11, n10, n01, n00)
    odds, p_value = stats.fisher_exact([[n11, n10], [n01, n00]])
    out["odds_ratio"] = odds
    out["fisher_p"] = p_value
    p_first = (n11 + n10) / total
    p_last = (n11 + n01) / total
    out["P_both_spliced_obs"] = n11 / total
    out["P_both_spliced_exp"] = p_first * p_last
    out["residual"] = n11 / total - p_first * p_last
    return out


# %% cell 2
_prov.mark(2)
# Run the coupling test per replicate. P5 must be promoter or exon_1.
OUT_DIR.mkdir(parents=True, exist_ok=True)
coupling_rows = []
for sample, classify_path in REPLICATES.items():
    rep_df = pd.read_csv(classify_path, sep="\t", low_memory=False)
    if "p5_feat" in rep_df.columns:
        rep_df = rep_df.loc[rep_df["p5_feat"].isin(VALID_P5_FEATURES)].copy()
    res = compute_first_last_coupling(rep_df, MIN_N)
    res["sample"] = sample
    coupling_rows.append(res)
    print(f"[{sample}] n_introns>={MIN_N}: n_reads={res['n_reads']:,} phi={res.get('phi', float('nan')):.3f}")

col_order = [
    "sample", "n_reads", "n11", "n10", "n01", "n00",
    "P_last|first_spliced", "P_last|first_retained",
    "fwd_deltaP", "rev_deltaP", "phi", "odds_ratio", "fisher_p",
    "P_both_spliced_obs", "P_both_spliced_exp", "residual",
]
coupling_df = pd.DataFrame(coupling_rows)[col_order]
coupling_df.to_csv(OUT_DIR / "first_last_intron_coupling.by_rep.tsv", sep="\t", index=False)
print(coupling_df.to_string(index=False))


# %% cell 3
_prov.mark(3)
# Replicate-level tests and the odds-ratio forest plot.
command = [
    sys.executable, str(_ROOT / "src" / "overall_splicing" / "test_first_last_coupling.py"),
    "--input", str(OUT_DIR / "first_last_intron_coupling.by_rep.tsv"),
    "--output-dir", str(OUT_DIR),
]
print("Running:", " ".join(command), flush=True)
subprocess.run(command, check=True)


# %% cell 4
_prov.mark(4)
# Conditional-probability bar chart (P(last spliced) | first state), per replicate.
cond_long = []
for _, r in coupling_df.iterrows():
    cond_long.append({"sample": r["sample"], "first_intron": "spliced", "p_last_spliced": r["P_last|first_spliced"]})
    cond_long.append({"sample": r["sample"], "first_intron": "retained", "p_last_spliced": r["P_last|first_retained"]})
cond_long = pd.DataFrame(cond_long)
cond_long["first_intron"] = pd.Categorical(cond_long["first_intron"], categories=["retained", "spliced"], ordered=True)

p_cond = (
    ggplot(cond_long, aes(x="first_intron", y="p_last_spliced", fill="sample"))
    + geom_col(position="dodge", width=0.7)
    + scale_y_continuous(limits=(0, 1))
    + theme_bw()
    + theme_common
    + labs(
        x="First intron status", y="P(last intron spliced)", fill="Replicate",
        title=f"First-intron state predicts last-intron splicing (n_introns >= {MIN_N})",
    )
)
p_cond.save(OUT_DIR / "first_last_conditional_prob.pdf", width=4.5, height=4.5, dpi=600)
p_cond.save(OUT_DIR / "first_last_conditional_prob.png", width=4.5, height=4.5, dpi=600)


# %% cell 5
_prov.mark(5)
# Alluvial (first -> last) diagram, Warm and Cold panels, replicates summed.
# Alluvial: Warm and Cold panels, replicates summed.
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "font.size": 9,
    "axes.spines.right": False,
    "axes.spines.top": False,
})


def _bezier_ribbon(x0, x1, y0a, y0b, y1a, y1b):
    verts = [
        (x0, y0a), ((x0 + x1) / 2, y0a), ((x0 + x1) / 2, y1a), (x1, y1a),
        (x1, y1b), ((x0 + x1) / 2, y1b), ((x0 + x1) / 2, y0b), (x0, y0b), (x0, y0a),
    ]
    codes = [
        MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.CLOSEPOLY,
    ]
    return PathPatch(MplPath(verts, codes), lw=0, ec="none")


def _draw_gene_cartoon(ax, y=1.08):
    """Tiny first-intron / last-intron schematic above the alluvial."""
    boxes = [
        (0.08, 0.07, "#4F88C6", "E1"),
        (0.16, 0.10, "#F4C430", "intron 1"),
        (0.27, 0.07, "#4F88C6", ""),
        (0.38, 0.18, "#D9D9D9", "···"),
        (0.57, 0.07, "#4F88C6", ""),
        (0.65, 0.12, "#F4C430", "last intron"),
        (0.78, 0.10, "#4F88C6", "last E"),
    ]
    h = 0.045
    for x, w, color, lab in boxes:
        ax.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor="black", lw=0.4, clip_on=False))
        if lab:
            ax.text(x + w / 2, y + h / 2, lab, ha="center", va="center", fontsize=7, clip_on=False)


def draw_first_last_alluvial(ax, n11, n10, n01, n00, title):
    total = n11 + n10 + n01 + n00
    if total <= 0:
        ax.set_axis_off()
        return
    left_spl = (n11 + n10) / total
    left_ret = (n01 + n00) / total
    right_spl = (n11 + n01) / total
    right_ret = (n10 + n00) / total
    gap = 0.04
    y_top = 0.98
    left_spl_top, left_spl_bot = y_top, y_top - left_spl * (0.9 - gap)
    left_ret_top = left_spl_bot - gap
    left_ret_bot = left_ret_top - left_ret * (0.9 - gap)
    right_spl_top, right_spl_bot = y_top, y_top - right_spl * (0.9 - gap)
    right_ret_top = right_spl_bot - gap
    right_ret_bot = right_ret_top - right_ret * (0.9 - gap)

    x0, x1 = 0.22, 0.78
    flows = [("n11", n11, True, True), ("n10", n10, True, False), ("n01", n01, False, True), ("n00", n00, False, False)]
    ls, lr = left_spl_top, left_ret_top
    rs, rr = right_spl_top, right_ret_top
    for key, n, from_spl, to_spl in flows:
        h = n / total * (0.9 - gap)
        if from_spl:
            y0b, y0a = ls, ls - h
            ls -= h
        else:
            y0b, y0a = lr, lr - h
            lr -= h
        if to_spl:
            y1b, y1a = rs, rs - h
            rs -= h
        else:
            y1b, y1a = rr, rr - h
            rr -= h
        patch = _bezier_ribbon(x0, x1, y0a, y0b, y1a, y1b)
        patch.set_facecolor(FLOW_COLORS[key])
        patch.set_alpha(0.55)
        ax.add_patch(patch)

    def node(x, yb, yt, facecolor, label, sub):
        ax.add_patch(FancyBboxPatch(
            (x - 0.07, yb), 0.14, max(yt - yb, 0.02),
            boxstyle="round,pad=0.008,rounding_size=0.02", facecolor=facecolor, edgecolor="black", lw=0.6,
        ))
        ax.text(x, (yb + yt) / 2, f"{label}\n{sub}", ha="center", va="center", fontsize=8,
                color="white" if facecolor == "#2E8B57" else "#222222", fontweight="bold")

    node(0.12, left_spl_bot, left_spl_top, "#2E8B57", "spliced", f"{left_spl:.0%}")
    node(0.12, left_ret_bot, left_ret_top, "#9E9E9E", "retained", f"{left_ret:.0%}")
    node(0.88, right_spl_bot, right_spl_top, "#2E8B57", "spliced", f"{right_spl:.0%}")
    node(0.88, right_ret_bot, right_ret_top, "#9E9E9E", "retained", f"{right_ret:.0%}")
    ax.text(0.12, 1.02, "First intron", ha="center", fontsize=10, fontweight="bold")
    ax.text(0.88, 1.02, "Last intron", ha="center", fontsize=10, fontweight="bold")
    _draw_gene_cartoon(ax)

    p_last_given_spl = n11 / (n11 + n10) if (n11 + n10) else float("nan")
    p_last_given_ret = n01 / (n01 + n00) if (n01 + n00) else float("nan")
    ax.text(
        0.50, 0.02,
        f"P(last spliced | first spliced) = {p_last_given_spl:.2f}\n"
        f"P(last spliced | first retained) = {p_last_given_ret:.2f}"
        f"   Delta = {p_last_given_spl - p_last_given_ret:+.2f}",
        ha="center", va="bottom", fontsize=8,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1.16)
    ax.set_axis_off()
    ax.set_title(f"{title}   n = {total:,.0f} molecules", loc="left", fontsize=11)


group_counts = []
for group, prefix in (("Warm", "Warm"), ("Cold", "Cold")):
    sub = coupling_df[coupling_df["sample"].astype(str).str.startswith(prefix)]
    group_counts.append({
        "group": f"{group}.mean (summed 2x2)",
        "n11": int(sub["n11"].sum()), "n10": int(sub["n10"].sum()),
        "n01": int(sub["n01"].sum()), "n00": int(sub["n00"].sum()),
    })

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
for ax, rec in zip(axes, group_counts):
    draw_first_last_alluvial(ax, rec["n11"], rec["n10"], rec["n01"], rec["n00"], rec["group"])
legend_handles = [
    Rectangle((0, 0), 1, 1, facecolor=FLOW_COLORS["n11"], alpha=0.55, label="first spliced -> last spliced"),
    Rectangle((0, 0), 1, 1, facecolor=FLOW_COLORS["n10"], alpha=0.55, label="first spliced -> last unspliced"),
    Rectangle((0, 0), 1, 1, facecolor=FLOW_COLORS["n01"], alpha=0.55, label="first unspliced -> last spliced"),
    Rectangle((0, 0), 1, 1, facecolor=FLOW_COLORS["n00"], alpha=0.55, label="first unspliced -> last unspliced"),
]
fig.suptitle(f"n_introns >= {MIN_N}", fontsize=10, y=1.02)
fig.legend(handles=legend_handles, loc="lower center", ncol=2, frameon=False, fontsize=8)
fig.subplots_adjust(bottom=0.22, wspace=0.12)
for suffix in (".pdf", ".png"):
    fig.savefig(Path(str(OUT_DIR / "first_last_conditional_flow") + suffix), dpi=300, bbox_inches="tight")
plt.close(fig)


print(f"\n[INFO] wrote {OUT_DIR}")
