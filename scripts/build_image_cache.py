"""Build the pre-resized image cache used by all training runs.

Usage:
    python scripts/build_image_cache.py [--max-side 448] [--workers 12]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.cache import CACHE_ROOT, build_cache  # noqa: E402
from src.utils.paths import MANIFESTS, METRICS, ensure_dirs  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-side", type=int, default=448)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    ensure_dirs()

    man = pd.read_parquet(MANIFESTS / "unified_manifest.parquet")
    print(f"caching {int((~man['exclude'].astype(bool)).sum())} images "
          f"at max side {args.max_side} -> {CACHE_ROOT}")
    stats = build_cache(man, max_side=args.max_side, workers=args.workers)
    (METRICS / "image_cache.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in stats.items() if k != "failures"}, indent=2))
    if stats["failed"]:
        print(f"WARNING: {stats['failed']} images failed to cache; training falls "
              f"back to the original file for those.")
        for f in stats["failures"]:
            print("   ", f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
