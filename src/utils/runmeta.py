"""Per-run provenance capture.

Section 24 of the specification requires every experiment to record enough state
that its numbers can be traced and regenerated. This module stamps each run with
config, seed, git commit, manifest/split hashes, environment, hardware and timing,
and appends one row per completed run to outputs/metrics/experiment_registry.csv.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.utils.paths import METRICS, REPO_ROOT

REGISTRY = METRICS / "experiment_registry.csv"


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                             capture_output=True, text=True, timeout=15)
        if out.returncode == 0:
            return out.stdout.strip()
        return "NOT_A_GIT_REPO"
    except Exception:  # noqa: BLE001
        return "GIT_UNAVAILABLE"


CODE_PATHS = ("src", "scripts", "configs", "tests", "pyproject.toml", "requirements-lock.txt")


def git_dirty(paths: tuple[str, ...] = CODE_PATHS, cwd: Path = REPO_ROOT) -> bool:
    """True when the TRAINING CODE differs from HEAD (modified or untracked files
    under `paths`). Scoped on purpose: run artefacts under outputs/ are written while
    a batch is running, so an unscoped `git status` flagged every run after the first
    (77 of 78 round-1 rows). The scope is stamped into each row as git_dirty_scope."""
    try:
        out = subprocess.run(["git", "status", "--porcelain", "--", *paths], cwd=cwd,
                             capture_output=True, text=True, timeout=15)
        return bool(out.stdout.strip()) if out.returncode == 0 else True
    except Exception:  # noqa: BLE001
        return True


def environment_info() -> dict:
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = torch.version.cuda
        info["cudnn"] = torch.backends.cudnn.version()
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["gpu_total_mem_gb"] = round(props.total_memory / 1e9, 2)
            info["gpu_capability"] = f"{props.major}.{props.minor}"
    except Exception as e:  # noqa: BLE001
        info["torch_error"] = str(e)
    for mod in ("timm", "torchvision", "numpy", "pandas", "sklearn", "albumentations"):
        try:
            m = __import__(mod)
            info[f"{mod}_version"] = getattr(m, "__version__", "unknown")
        except Exception:  # noqa: BLE001
            info[f"{mod}_version"] = "not_installed"
    try:
        import os
        info["cpu_count"] = os.cpu_count()
    except Exception:  # noqa: BLE001
        pass
    return info


@dataclass
class RunMeta:
    experiment: str
    run_name: str
    seed: int
    config: dict
    manifest_hash: str = ""
    split_hash: str = ""
    protocol: str = ""
    fold: int = -1
    started_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    finished_utc: str = ""
    duration_sec: float = 0.0
    git_commit: str = field(default_factory=git_commit)
    git_dirty: bool = field(default_factory=git_dirty)
    git_dirty_scope: str = ",".join(CODE_PATHS)
    environment: dict = field(default_factory=environment_info)
    best_epoch: int = -1
    best_val_metric: float = float("nan")
    best_val_metric_name: str = ""
    status: str = "running"
    error: str = ""
    checkpoint_path: str = ""
    predictions_path: str = ""
    metrics_path: str = ""
    thresholds: list[float] = field(default_factory=list)
    curve_path: str = ""
    notes: str = ""

    def finish(self, status: str = "completed", error: str = "") -> None:
        self.finished_utc = datetime.now(timezone.utc).isoformat()
        t0 = datetime.fromisoformat(self.started_utc)
        self.duration_sec = round((datetime.fromisoformat(self.finished_utc) - t0).total_seconds(), 2)
        self.status = status
        self.error = error

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, default=str), encoding="utf-8")

    def register(self) -> None:
        """Append one row per completed run to the machine-readable registry."""
        row = {
            "experiment": self.experiment, "run_name": self.run_name, "seed": self.seed,
            "protocol": self.protocol, "fold": self.fold, "status": self.status,
            "started_utc": self.started_utc, "finished_utc": self.finished_utc,
            "duration_sec": self.duration_sec,
            "best_epoch": self.best_epoch,
            "best_val_metric_name": self.best_val_metric_name,
            "best_val_metric": self.best_val_metric,
            "git_commit": self.git_commit, "git_dirty": self.git_dirty,
            "git_dirty_scope": self.git_dirty_scope,
            "manifest_hash": self.manifest_hash, "split_hash": self.split_hash,
            "backbone": self.config.get("model", {}).get("backbone", ""),
            "gpu": self.environment.get("gpu_name", ""),
            "torch": self.environment.get("torch", ""),
            "checkpoint_path": self.checkpoint_path,
            "predictions_path": self.predictions_path,
            "metrics_path": self.metrics_path,
            "thresholds": json.dumps([round(float(t), 4) for t in self.thresholds]),
            "error": self.error, "notes": self.notes,
        }
        REGISTRY.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame([row])
        if REGISTRY.exists():
            prev = pd.read_csv(REGISTRY)
            # a re-run of the same run_name replaces its previous row
            prev = prev[~((prev["run_name"] == self.run_name) & (prev["seed"] == self.seed))]
            df = pd.concat([prev, df], ignore_index=True)
        df.to_csv(REGISTRY, index=False)


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    import os
    import random
    import numpy as np
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
