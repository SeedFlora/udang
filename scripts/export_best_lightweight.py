"""Find the best lightweight (MobileNetV3-Large) checkpoint by VALIDATION metric
and export it via scripts/export_model.py.

Selection uses the experiment registry's recorded validation metric only -- never
test metrics.

Usage:
    python scripts/export_best_lightweight.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.paths import METRICS, REPO_ROOT  # noqa: E402


def main() -> int:
    reg = METRICS / "experiment_registry.csv"
    if not reg.exists():
        print("BLOCKER: experiment_registry.csv not found - no runs recorded.")
        return 2
    df = pd.read_csv(reg)
    light = df[(df["status"] == "completed")
               & (df["backbone"].astype(str) == "mobilenetv3_large")]
    if light.empty:
        print("BLOCKER: no completed MobileNetV3-Large run found in the registry.")
        return 2

    best = light.sort_values("best_val_metric", ascending=False).iloc[0]
    print(f"best lightweight run by VALIDATION metric: {best['run_name']} "
          f"({best['best_val_metric_name']}={best['best_val_metric']:.4f}, "
          f"epoch {best['best_epoch']})")
    ck = best["checkpoint_path"]
    if not isinstance(ck, str) or not ck:
        print("BLOCKER: selected run has no checkpoint recorded.")
        return 2
    if not (REPO_ROOT / ck).exists():
        print(f"BLOCKER: checkpoint file missing: {ck}")
        return 2

    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "export_model.py"),
           "--checkpoint", ck, "--backbone", "mobilenetv3_large",
           "--n-samples", "64"]
    print("running:", " ".join(cmd))
    return subprocess.run(cmd, cwd=REPO_ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
