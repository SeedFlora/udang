"""Build the unified manifest across all three datasets.

Usage:
    python scripts/build_manifests.py [--workers 10] [--phash-threshold 6]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.manifest import build_unified_manifest  # noqa: E402
from src.utils.hashing import hash_dataframe  # noqa: E402
from src.utils.paths import MANIFESTS, RAW, ensure_dirs, rel_to_repo  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--phash-threshold", type=int, default=6)
    args = ap.parse_args()
    ensure_dirs()

    versions = {}
    rec_path = MANIFESTS / "acquisition_record.json"
    if rec_path.exists():
        rec = json.loads(rec_path.read_text(encoding="utf-8"))
        for _k, v in rec.get("datasets", {}).items():
            versions[v["name"]] = str(v["resolved_version"])
    print(f"dataset versions from acquisition record: {versions}", flush=True)

    print("scanning + hashing images (this reads every file) ...", flush=True)
    df, diag = build_unified_manifest(RAW, versions, workers=args.workers,
                                      phash_threshold=args.phash_threshold)

    pq = MANIFESTS / "unified_manifest.parquet"
    csv = MANIFESTS / "unified_manifest.csv"
    df.to_parquet(pq, index=False)
    df.to_csv(csv, index=False)

    # duplicate clusters: only clusters with more than one member are interesting
    dup = (df[~df["exclude"].astype(bool)]
           .groupby("duplicate_cluster_id")
           .agg(n=("record_id", "count"),
                record_ids=("record_id", lambda s: ";".join(sorted(s))),
                datasets=("dataset", lambda s: ";".join(sorted(set(s)))),
                paths=("relative_image_path", lambda s: ";".join(sorted(s))))
           .reset_index())
    dup = dup[dup["n"] > 1].sort_values("n", ascending=False)
    dup.to_csv(MANIFESTS / "duplicate_clusters.csv", index=False)

    exc = df[df["exclude"].astype(bool)][
        ["record_id", "dataset", "relative_image_path", "exclude_reason"]]
    exc.to_csv(MANIFESTS / "excluded_records.csv", index=False)

    (MANIFESTS / "manifest_diagnostics.json").write_text(
        json.dumps(diag, indent=2, default=str), encoding="utf-8")

    mh = hash_dataframe(df, ["record_id", "sha256"])
    (MANIFESTS / "manifest_hash.txt").write_text(mh, encoding="utf-8")

    print(f"\nmanifest rows: {len(df)}")
    print(df.groupby(["dataset", "task", "record_role"]).size().to_string())
    print(f"\nexcluded (unreadable/corrupt): {len(exc)}")
    print(f"exact-duplicate clusters with >1 member: {len(dup)}")
    print(f"fused groups: {df['group_id'].nunique()}")
    print(f"manifest hash: {mh}")
    for p in (pq, csv, MANIFESTS / 'duplicate_clusters.csv',
              MANIFESTS / 'excluded_records.csv'):
        print(f"  -> {rel_to_repo(p)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
