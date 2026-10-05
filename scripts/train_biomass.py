"""Thin wrapper around src.training.run_matrix for experiments E4, E5.

Usage:
    python scripts/train_biomass.py [--profile standard] [--only E2_pooled_bce_ordinary] [--seed 42]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.run_matrix import run_configs, select_matrix  # noqa: E402

DEFAULT_EXPERIMENTS = ["E4", "E5"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="standard", choices=["smoke", "standard", "full"])
    ap.add_argument("--only", nargs="*", default=None,
                    help="restrict to these experiment ids or config names")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    only = args.only or DEFAULT_EXPERIMENTS
    cfgs = select_matrix(args.profile, only, seed=args.seed)
    if not cfgs:
        print(f"no configurations matched {only}")
        return 1
    print(f"profile={args.profile}  {len(cfgs)} configuration(s): "
          + ", ".join(c.name for c in cfgs))
    df = run_configs(cfgs, verbose=not args.quiet)
    print(f"\ncompleted {len(df)} fold-runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
