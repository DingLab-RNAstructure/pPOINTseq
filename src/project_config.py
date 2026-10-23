"""Central project and sample configuration for the POINTSeq workflow.

Inputs are defined in a tab-separated sample sheet. BAM paths may be absolute,
project-relative, or bare filenames resolved under ``data/bams``. Output paths
are generated consistently for notebooks and command-line scripts.

Environment overrides:
    POINTSEQ_SAMPLE_SHEET: sample sheet path.
    POINTSEQ_RESULTS_DIR: results root for the current analysis run.

Usage:
    python src/validate_project_config.py
    POINTSEQ_SAMPLE_SHEET=configs/samples.test.tsv python src/validate_project_config.py
    POINTSEQ_RESULTS_DIR=results/run_02 jupyter lab
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path

# ============================================================
# PROJECT PATHS
# ============================================================

WORK_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent
CONFIG_DIR = WORK_DIR / "configs"
DATA_DIR = WORK_DIR / "data"
BAM_DIR = DATA_DIR / "bams"
REF_DIR = DATA_DIR / "refs"
LOG_DIR = WORK_DIR / "logs"

DEFAULT_SAMPLES_TSV = CONFIG_DIR / "samples.tsv"
DEFAULT_RESULTS_DIR = WORK_DIR / "results"

REQUIRED_SAMPLE_COLUMNS = ("sample", "bam_file", "out_name", "replicate")
OPTIONAL_SAMPLE_COLUMNS = ("enabled",)
TRUE_VALUES = frozenset({"1", "true", "yes", "y"})
FALSE_VALUES = frozenset({"0", "false", "no", "n", ""})


def _environment_path(variable: str, default: Path) -> Path:
    """Return an absolute path from an environment override or a default."""
    value = os.environ.get(variable, "").strip()
    if not value:
        return default
    path = Path(value).expanduser()
    return path if path.is_absolute() else WORK_DIR / path


SAMPLES_TSV = _environment_path("POINTSEQ_SAMPLE_SHEET", DEFAULT_SAMPLES_TSV)
RESULTS_DIR = _environment_path("POINTSEQ_RESULTS_DIR", DEFAULT_RESULTS_DIR)

RESULT_SUBDIRS = (
    "quant",
    "expression",
    "splicing",
    "alternative_splicing",
    "read_through",
    "splicing_order",
    "exon_intron",
    "pausing",
    "gene_pausing",
    "gene_reads",
    "distribution",
    "ends",
    "splice_sites",
    "strength",
    "bw",
    "intronless",
    "read_info",
)

RESULT_DIRS = {name: RESULTS_DIR / name for name in RESULT_SUBDIRS}
ORIGINAL_RESULT_DIRS = {name: RESULTS_DIR / name for name in RESULT_SUBDIRS}
P5_RESULTS_DIR = RESULTS_DIR / "p5"
P5_RESULT_DIRS = {name: P5_RESULTS_DIR / name for name in RESULT_SUBDIRS}


# ============================================================
# SAMPLE CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class Sample:
    """One active sample and its resolved BAM path."""

    sample: str
    bam: Path
    out_name: str
    replicate: str


@dataclass(frozen=True)
class SampleTable:
    """Small dependency-free table used for manifest display and validation."""

    columns: tuple[str, ...]
    rows: tuple[dict[str, str | Path], ...]

    def to_string(self, index: bool = False) -> str:
        """Render a compact table compatible with the notebook display call."""
        del index
        visible_columns = tuple(
            column for column in self.columns if column != "bam_path"
        )
        widths = {
            column: max(
                len(column),
                *(len(str(row.get(column, ""))) for row in self.rows),
            )
            for column in visible_columns
        }

        def render(row: dict[str, str | Path]) -> str:
            return "  ".join(
                str(row.get(column, "")).ljust(widths[column])
                for column in visible_columns
            ).rstrip()

        header = render({column: column for column in visible_columns})
        return "\n".join((header, *(render(row) for row in self.rows)))


@dataclass(frozen=True)
class ProjectConfig:
    """Validated paths and active samples for one analysis run."""

    work_dir: Path
    sample_sheet: Path
    results_dir: Path
    samples: tuple[Sample, ...]

    @property
    def sample_ids(self) -> tuple[str, ...]:
        """Return active sample identifiers in manifest order."""
        return tuple(sample.sample for sample in self.samples)

    @property
    def by_sample(self) -> dict[str, Sample]:
        """Return active samples keyed by sample identifier."""
        return {sample.sample: sample for sample in self.samples}


def resolve_project_path(path_text: str, base_dir: Path = WORK_DIR) -> Path:
    """Resolve an absolute or project-relative path without requiring it to exist."""
    path = Path(path_text.strip()).expanduser()
    return path if path.is_absolute() else base_dir / path


def resolve_bam_path(bam_file: str) -> Path:
    """Resolve a BAM filename, project-relative path, or absolute path."""
    path = Path(bam_file.strip()).expanduser()
    if path.is_absolute():
        return path
    if len(path.parts) == 1:
        return BAM_DIR / path
    return WORK_DIR / path


def sort_bam_path(bam_path: Path | str) -> Path:
    """Return the full sorted BAM for one sample-configured analysis BAM.

    For example, ``Warm1.nonpolya.bam`` or ``Warm1_tagged_elongating.bam`` resolves to
    ``data/raw/Warm1_sort.bam``. If that file is missing, the expected path is still
    returned so the caller can report it.
    """
    path = Path(bam_path)
    sample_id = path.stem.split("_")[0].split(".")[0]
    candidates = (
        DATA_DIR / "raw" / f"{sample_id}_sort.bam",
        BAM_DIR / f"{sample_id}_sort.bam",
    )
    return next((candidate for candidate in candidates if candidate.is_file()), candidates[0])


def _is_enabled(value: str, path: Path, row_number: int) -> bool:
    """Parse one optional enabled value."""
    normalized = value.strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    raise ValueError(
        f"{path}: invalid enabled value {value!r} on row {row_number}; "
        "use true/false, yes/no, or 1/0"
    )


def load_sample_table(path: Path | str | None = None) -> SampleTable:
    """Load, normalize, and validate the active rows in a sample sheet."""
    sheet = resolve_project_path(str(path)) if path is not None else SAMPLES_TSV
    if not sheet.is_file():
        raise FileNotFoundError(f"Sample sheet not found: {sheet}")

    with sheet.open(newline="", encoding="utf-8") as handle:
        lines = (line for line in handle if line.strip() and not line.lstrip().startswith("#"))
        reader = csv.DictReader(lines, delimiter="\t")
        columns = tuple(reader.fieldnames or ())
        raw_rows = list(reader)

    missing = sorted(set(REQUIRED_SAMPLE_COLUMNS) - set(columns))
    if missing:
        raise ValueError(f"{sheet}: missing columns {missing}")

    rows: list[dict[str, str | Path]] = []
    for row_number, raw_row in enumerate(raw_rows, start=2):
        row = {column: (raw_row.get(column) or "").strip() for column in columns}
        empty = [column for column in REQUIRED_SAMPLE_COLUMNS if not row[column]]
        if empty:
            raise ValueError(
                f"{sheet}: empty required values {empty} on row {row_number}"
            )
        if "enabled" in columns and not _is_enabled(
            str(row["enabled"]), sheet, row_number
        ):
            continue
        row["bam_path"] = resolve_bam_path(str(row["bam_file"]))
        rows.append(row)

    if not rows:
        raise ValueError(f"{sheet}: no enabled samples")

    for column in ("sample", "out_name"):
        values = [str(row[column]) for row in rows]
        duplicated = sorted({value for value in values if values.count(value) > 1})
        if duplicated:
            raise ValueError(f"{sheet}: duplicate {column} values {duplicated}")

    return SampleTable(columns + ("bam_path",), tuple(rows))


def load_project_config(path: Path | str | None = None) -> ProjectConfig:
    """Build a project configuration from a sample sheet."""
    sheet = resolve_project_path(str(path)) if path is not None else SAMPLES_TSV
    table = load_sample_table(sheet)
    samples = tuple(
        Sample(
            sample=str(row["sample"]),
            bam=Path(row["bam_path"]),
            out_name=str(row["out_name"]),
            replicate=str(row["replicate"]),
        )
        for row in table.rows
    )
    return ProjectConfig(WORK_DIR, sheet, RESULTS_DIR, samples)


def bam_index_path(bam_path: Path) -> Path | None:
    """Return an existing BAI path for a BAM, if present."""
    candidates = (Path(f"{bam_path}.bai"), bam_path.with_suffix(".bai"))
    return next((path for path in candidates if path.is_file()), None)


def validate_inputs(config: ProjectConfig | None = None) -> list[str]:
    """Return human-readable BAM and BAM-index validation errors."""
    current = config or PROJECT
    errors: list[str] = []
    for sample in current.samples:
        if not sample.bam.is_file():
            errors.append(f"{sample.sample}: BAM not found: {sample.bam}")
        elif bam_index_path(sample.bam) is None:
            errors.append(
                f"{sample.sample}: BAM index not found "
                f"({sample.bam}.bai or {sample.bam.with_suffix('.bai')})"
            )
    return errors


# ============================================================
# OUTPUT PATHS
# ============================================================

def sample_config(sample: str) -> Sample:
    """Return one sample configuration or raise a useful error."""
    try:
        return PROJECT.by_sample[sample]
    except KeyError as exc:
        raise KeyError(
            f"Unknown sample {sample!r}; available: {list(PROJECT.sample_ids)}"
        ) from exc


def out_name(sample: str) -> str:
    """Return the configured output prefix for a sample."""
    return sample_config(sample).out_name


def ensure_project_dirs() -> None:
    """Create writable output and log directories."""
    for path in (*RESULT_DIRS.values(), LOG_DIR):
        path.mkdir(parents=True, exist_ok=True)


def ensure_p5_result_dirs() -> None:
    """Create the parallel ``results/p5/<subdir>`` output tree."""
    for path in (*P5_RESULT_DIRS.values(), LOG_DIR):
        path.mkdir(parents=True, exist_ok=True)


def ensure_result_dirs() -> None:
    """Backward-compatible alias for ensure_project_dirs."""
    ensure_project_dirs()


def classify_tsv(sample: str) -> Path:
    return RESULT_DIRS["read_through"] / f"{out_name(sample)}.elongating.classify.tsv"


def bw_3end_prefix(sample: str) -> Path:
    return RESULT_DIRS["bw"] / f"{out_name(sample)}.elongating.3end"


def read_info_tsv(sample: str) -> Path:
    return RESULT_DIRS["read_info"] / f"{out_name(sample)}.sort.read_info.tsv"


def existing_read_info_tsv(sample: str) -> Path:
    """Return current-run read info, falling back to original results then legacy."""
    configured = read_info_tsv(sample)
    original = ORIGINAL_RESULT_DIRS["read_info"] / configured.name
    legacy = DATA_DIR / "read_info" / configured.name
    for path in (configured, original, legacy):
        if path.is_file():
            return path
    return configured


def expression_gene_tsv(sample: str) -> Path:
    return RESULT_DIRS["expression"] / f"{out_name(sample)}.elongating_read_info.tsv"


def pausing_index_tsv(sample: str) -> Path:
    return RESULT_DIRS["pausing"] / f"{out_name(sample)}.elongating.pausing_index.tsv"


def splicing_ratio_tsv(sample: str) -> Path:
    return RESULT_DIRS["splicing"] / f"{out_name(sample)}.elongating.splicing_ratio.tsv"


def splicing_ratio_legacy_tsv(sample: str) -> Path:
    """Return the legacy filename still used by notebook 4."""
    return RESULT_DIRS["splicing"] / f"{out_name(sample)}_elongating_splicing_ratio.tsv"


def original_classify_tsv(sample: str) -> Path:
    """Return the BAM-derived classify TSV (not the P5-rewritten copy)."""
    return ORIGINAL_RESULT_DIRS["read_through"] / f"{out_name(sample)}.elongating.classify.tsv"


def p5_classify_tsv(sample: str) -> Path:
    """Return the P5-in-intron rewritten classify TSV."""
    return P5_RESULT_DIRS["read_through"] / f"{out_name(sample)}.elongating.classify.tsv"


def splicing_ratio_p5_tsv(sample: str) -> Path:
    """Return the P5-intron splicing-ratio table under results/p5/splicing."""
    return P5_RESULT_DIRS["splicing"] / f"{out_name(sample)}.elongating.splicing_ratio.tsv"


def splicing_ratio_p5_legacy_tsv(sample: str) -> Path:
    """Return the legacy-named P5 splicing-ratio table."""
    return P5_RESULT_DIRS["splicing"] / f"{out_name(sample)}_elongating_splicing_ratio.tsv"


# ============================================================
# BACKWARD-COMPATIBLE NOTEBOOK GLOBALS
# ============================================================

PROJECT = load_project_config()
SAMPLE_TABLE = load_sample_table(PROJECT.sample_sheet)
SAMPLES = PROJECT.sample_ids
OUT_NAMES = tuple(sample.out_name for sample in PROJECT.samples)
REPLICATE_LABELS = {
    sample.sample: sample.replicate for sample in PROJECT.samples
}
BAM_BY_SAMPLE = {sample.sample: sample.bam for sample in PROJECT.samples}
OUT_NAME_BY_SAMPLE = {
    sample.sample: sample.out_name for sample in PROJECT.samples
}
