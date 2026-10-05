"""Run the whole post-training chain in dependency order.

Each stage is allowed to fail without aborting the rest -- a missing optional
artefact should not prevent the report from being written, and the report states
which stages failed rather than silently omitting them.

Usage:
    python scripts/finalize.py [--n-boot 2000] [--skip-export] [--skip-explain]
                               [--skip-views] [--skip-selection]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.paths import METRICS, REPO_ROOT  # noqa: E402


def run(name: str, argv: list[str], required: bool = True) -> dict:
    print(f"\n{'=' * 74}\n### {name}\n{'=' * 74}", flush=True)
    t0 = time.time()
    r = subprocess.run([sys.executable, *argv], cwd=REPO_ROOT)
    dt = round(time.time() - t0, 1)
    ok = r.returncode == 0
    print(f"--- {name}: {'OK' if ok else f'EXIT {r.returncode}'} ({dt}s)", flush=True)
    return {"stage": name, "argv": argv, "returncode": r.returncode,
            "ok": ok, "seconds": dt, "required": required}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--skip-export", action="store_true")
    ap.add_argument("--skip-explain", action="store_true")
    ap.add_argument("--skip-views", action="store_true")
    ap.add_argument("--skip-selection", action="store_true",
                    help="do not re-rank models (the revision fixes the reported configs)")
    args = ap.parse_args()

    stages: list[dict] = []
    S = "scripts"

    if not args.skip_selection:
        stages.append(run("model selection (validation only)", [f"{S}/select_models.py"]))
    stages.append(run("evaluation + bootstrap CIs + paired tests",
                      [f"{S}/evaluate.py", "--n-boot", str(args.n_boot)]))
    if not args.skip_views:
        stages.append(run("biomass view-count sweep",
                          [f"{S}/eval_view_counts.py", "--repeats", "3"],
                          required=False))
    stages.append(run("figures", [f"{S}/make_figures.py"]))
    stages.append(run("error analysis + montages", [f"{S}/error_analysis.py"]))
    if not args.skip_export:
        stages.append(run("export lightweight model (TorchScript + ONNX)",
                          [f"{S}/export_best_lightweight.py"], required=False))
    if not args.skip_explain:
        stages.append(run("explainability (optional)", [f"{S}/explainability.py"],
                          required=False))
    stages.append(run("write reports", [f"{S}/write_report.py"]))

    METRICS.mkdir(parents=True, exist_ok=True)
    (METRICS / "finalize_log.json").write_text(
        json.dumps(stages, indent=2), encoding="utf-8")

    print(f"\n{'=' * 74}\nFINALIZE SUMMARY\n{'=' * 74}")
    for s in stages:
        flag = "OK " if s["ok"] else ("FAIL" if s["required"] else "skip")
        print(f"  [{flag}] {s['stage']:52s} {s['seconds']:>7.1f}s")
    failed_required = [s for s in stages if s["required"] and not s["ok"]]
    if failed_required:
        print(f"\n{len(failed_required)} REQUIRED stage(s) failed: "
              f"{[s['stage'] for s in failed_required]}")
        return 1
    print("\nall required stages completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
