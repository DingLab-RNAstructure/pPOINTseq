"""Put ``src/`` and each analysis folder on ``sys.path``.

Analysis scripts live in subfolders. They still import each other by module
name (``project_config``, ``plot_gene_reads``, ``classify_splicing_tables``).
Call ``setup`` before those imports.
"""
from __future__ import annotations

import sys
from pathlib import Path


def src_dir(script_file: str | Path) -> Path:
    """Return the ``src`` directory that contains ``project_config.py``."""
    here = Path(script_file).resolve()
    for candidate in here.parents:
        if (candidate / "project_config.py").is_file():
            return candidate
    raise SystemExit(f"project_config.py not found above {script_file}")


def setup(script_file: str | Path) -> Path:
    """Add ``src`` and its analysis folders to ``sys.path``. Return the project root."""
    src = src_dir(script_file)
    folders = [src]
    for folder in sorted(src.iterdir()):
        if not folder.is_dir() or folder.name.startswith((".", "_")):
            continue
        if any(folder.glob("*.py")):
            folders.append(folder)
    for folder in reversed(folders):
        text = str(folder)
        if text in sys.path:
            sys.path.remove(text)
        sys.path.insert(0, text)
    return src.parent
