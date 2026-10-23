#!/usr/bin/env python3
"""Tests and summary figures for consecutive intron retention.

Run after ``src/cold_cui/count_consecutive_intron_retention.py``, which writes
``consecutive_ri.by_sample.tsv`` and the other per-read tables. Everything
lands in ``results/alternative_splicing/`` and ``results/gene_reads/``. Each
cell below is one helper script; the provenance table records which cell
wrote which file.
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

# Example transcripts used for the barcode figures (Warm-high, Cold-down).
BARCODE_GENES = [
    "AT1G58180.2", "AT1G76450.1", "AT1G80560.1", "AT2G04690.5", "AT2G47490.1",
    "AT3G09405.1", "AT3G48360.1", "AT3G49620.1", "AT5G07440.1", "AT5G57655.2",
]
# Transcripts shown as panel a of the summary figure.
PANEL_A_GENES = ["AT3G09405.1", "AT1G58180.2", "AT1G80560.1", "AT2G04690.5", "AT3G48360.1", "AT5G57655.2"]


def run(script: str, *args: str) -> None:
    """Run one helper script from src/ and fail loudly on a non-zero exit."""
    command = [sys.executable, str(_ROOT / "src" / "cold_cui" / script), *args]
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


# %% cell 1
_prov.mark(1)
# Per-gene Warm vs Cold consecutive-RI table and Warm-present / Cold-down candidate list.
run("_find_nv_consec_c5_down.py")

# %% cell 2
_prov.mark(2)
# Library-rate test, gene GLM, same-molecule coupling (2x2 / odds ratios), clustering.
run("test_consecutive_ri.py")

# %% cell 3
_prov.mark(3)
# Read-by-intron barcode figures for the example transcripts.
run("plot_consecutive_ri_barcode.py", "--genes", *BARCODE_GENES)

# %% cell 4
_prov.mark(4)
# Summary figure (panels a-d) + standalone volcano; the first gene is the hero
# panel, the rest get panel-a-only barcode figures.
run("plot_consecutive_ri_summary.py", "--gene", PANEL_A_GENES[0], "--write-volcano",
    "--panel-a-genes", *PANEL_A_GENES[1:])

# %% cell 5
_prov.mark(5)
# Gene-reads tracks for Warm-high / Cold-down example genes.
run("_plot_nv_consec_c5_down.py")
