"""Canonical project paths. Everything resolves relative to the repo root so the
tree can be relocated without editing configs."""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

CONFIGS = REPO_ROOT / "configs"
DATA = REPO_ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
MANIFESTS = DATA / "manifests"
SPLITS = DATA / "splits"

OUTPUTS = REPO_ROOT / "outputs"
LOGS = OUTPUTS / "logs"
CHECKPOINTS = OUTPUTS / "checkpoints"
PREDICTIONS = OUTPUTS / "predictions"
METRICS = OUTPUTS / "metrics"
FIGURES = OUTPUTS / "figures"
ERROR_ANALYSIS = OUTPUTS / "error_analysis"

REPORTS = REPO_ROOT / "reports"

_ALL = [
    RAW, INTERIM, PROCESSED, MANIFESTS, SPLITS,
    LOGS, CHECKPOINTS, PREDICTIONS, METRICS, FIGURES, ERROR_ANALYSIS, REPORTS,
]


def ensure_dirs() -> None:
    for d in _ALL:
        d.mkdir(parents=True, exist_ok=True)


def rel_to_repo(p: str | Path) -> str:
    """Repo-relative POSIX path. Used everywhere a path is persisted so that no
    absolute private path ever reaches a manifest, figure or report."""
    p = Path(p).resolve()
    try:
        return p.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return p.name
