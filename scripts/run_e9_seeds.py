"""E9: repeat the selected best models at three seeds (42, 2026, 3407).

Model selection MUST already have happened on validation performance; this script
only repeats the chosen configurations. It never inspects test metrics to choose.

Usage:
    python scripts/run_e9_seeds.py [--profile standard] [--best-json outputs/metrics/model_selection.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.registry import SEEDS, build_e9  # noqa: E402
from src.training.run_matrix import run_configs  # noqa: E402
from src.utils.paths import METRICS  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="standard", choices=["smoke", "standard", "full"])
    ap.add_argument("--best-json", default=str(METRICS / "model_selection.json"))
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    best = None
    p = Path(args.best_json)
    if p.exists():
        sel = json.loads(p.read_text(encoding="utf-8"))
        best = sel.get("selected")
        print(f"model selection loaded from {p.name}:")
        for k, v in (best or {}).items():
            print(f"  {k:18s} -> {v}")
        print(f"selection criterion: {sel.get('criterion', 'validation metric')}")
    else:
        print(f"{p.name} not found; using registry defaults for E9 selection")

    cfgs = build_e9(args.profile, best=best)
    print(f"\nE9: {len(cfgs)} runs across seeds {SEEDS}")
    for c in cfgs:
        print(f"  {c.name} (seed {c.seed}, folds {c.folds})")
    df = run_configs(cfgs, verbose=not args.quiet)
    print(f"\ncompleted {len(df)} E9 fold-runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
