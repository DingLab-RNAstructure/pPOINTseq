#!/usr/bin/env python3
"""Attribute every result file to the step script and cell that wrote it.

Each converted step script calls ``mark(cell)`` at the top of every cell. A
mark scans ``results/`` and assigns files that are new or changed since the
previous mark to the cell that just finished. Rows are appended to
``logs/provenance/<step>.tsv`` (result_file, step_script, cell, line).
"""
from __future__ import annotations

import atexit
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
PROV_DIR = ROOT / "logs" / "provenance"

_state = {"script": None, "cell": None, "line": None, "snap": {}, "out": None}


def _scan() -> dict[str, tuple[int, int]]:
    snap = {}
    stack = [str(RESULTS)]
    while stack:
        with os.scandir(stack.pop()) as it:
            for entry in it:
                if entry.is_dir(follow_symlinks=False):
                    stack.append(entry.path)
                elif entry.is_file(follow_symlinks=False):
                    st = entry.stat(follow_symlinks=False)
                    snap[entry.path] = (st.st_mtime_ns, st.st_size)
    return snap


def _flush() -> None:
    """Assign files changed since the last scan to the current cell."""
    if _state["script"] is None:
        return
    snap = _scan()
    old = _state["snap"]
    changed = sorted(p for p, v in snap.items() if old.get(p) != v)
    if changed and _state["cell"] is not None:
        with _state["out"].open("a", encoding="utf-8") as handle:
            for path in changed:
                rel = os.path.relpath(path, RESULTS)
                handle.write(f"{rel}\t{_state['script']}\t{_state['cell']}\t{_state['line']}\n")
    _state["snap"] = snap


def start(script_path: str) -> None:
    """Begin tracking for one step script."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    PROV_DIR.mkdir(parents=True, exist_ok=True)
    name = Path(script_path).name
    _state["script"] = name
    _state["out"] = PROV_DIR / f"{Path(name).stem}.tsv"
    _state["out"].write_text("result_file\tstep_script\tcell\tline\n", encoding="utf-8")
    _state["snap"] = _scan()
    atexit.register(_flush)


def mark(cell: int) -> None:
    """Close the previous cell and open a new one."""
    _flush()
    _state["cell"] = cell
    _state["line"] = sys._getframe(1).f_lineno
