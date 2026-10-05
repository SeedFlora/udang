"""Run the non-neural biomass baselines (specification section 12).

These run before any deep model so the image-based results have a reference.
Note that the allometric and tabular baselines consume ground-truth manual
measurements that an image-only model never sees; every result records that
information advantage explicitly.

The revision evaluates three biomass protocols (leave-one-DB-out, leave-one-
site-out, within-database CV); the leave-one-DB-out files keep their original
names so downstream readers are unaffected.

Usage:
    python scripts/run_biomass_baselines.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.biomass_baselines import (  # noqa: E402
    individual_table, run_all_baselines, summarize_baselines,
)
from src.utils.paths import MANIFESTS, METRICS, SPLITS, ensure_dirs  # noqa: E402

PROTOCOLS = ("biomass_lodo", "biomass_loso", "biomass_within_db")


def suffix_for(protocol: str) -> str:
    """Output-file suffix: leave-one-DB-out keeps the historical bare names."""
    return "" if protocol == "biomass_lodo" else f"_{protocol.split('biomass_')[1]}"


def main() -> int:
    ensure_dirs()
    man = pd.read_parquet(MANIFESTS / "unified_manifest.parquet")
    spl = pd.read_parquet(SPLITS / "splits.parquet")

    ind = individual_table(man)
    if len(ind) == 0:
        print("BLOCKER: no biomass individuals in the manifest.")
        return 2
    print(f"{len(ind)} individuals across {ind['pond_or_db_id'].nunique()} DBs")
    print(ind.groupby("pond_or_db_id").agg(
        n=("individual_id", "count"),
        mean_weight_g=("weight_g", "mean"),
        mean_length_cm=("length_cm", "mean"),
        has_ceph=("cephalothorax_cm", lambda s: int(s.notna().sum()))).to_string())

    available = set(spl["protocol"])
    written = 0
    for proto in PROTOCOLS:
        if proto not in available:
            print(f"\nskip {proto}: not in split table")
            continue
        res = run_all_baselines(man, spl, protocol=proto)
        suffix = suffix_for(proto)
        (METRICS / f"biomass_baselines{suffix}.json").write_text(
            json.dumps(res, indent=2, default=str), encoding="utf-8")

        tbl = summarize_baselines(res)
        tbl.to_csv(METRICS / f"biomass_baselines{suffix}.csv", index=False)
        written += 1

        cols = [c for c in ("fold", "held_out_db", "model", "weight_mae", "weight_rmse",
                            "weight_mape", "weight_r2") if c in tbl.columns]
        print(f"\n=== {proto}: per-fold (held-out unit) ===")
        print(tbl[cols].round(3).to_string(index=False))

        print(f"\n=== {proto}: mean across held-out units ===")
        agg = tbl.groupby("model")[[c for c in tbl.columns
                                    if c.startswith("weight_")]].mean(numeric_only=True)
        print(agg.round(3).to_string())

        print(f"\n{proto}: allometric fits per fold:")
        for f, d in res["folds"].items():
            a = d["allometric"]
            if "a" in a:
                print(f"  fold {f} (held out {d['held_out_db']}): "
                      f"W = {a['a']:.5f} * L^{a['b']:.3f}  "
                      f"(log-space fit R2={a['fit_r2_log_space']:.3f}, "
                      f"n_train={a['n_train']})")
            else:
                print(f"  fold {f}: {a.get('error')}")
        print(f"-> outputs/metrics/biomass_baselines{suffix}.{{json,csv}}")

    if written == 0:
        print("BLOCKER: no biomass protocol present in the split table.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
