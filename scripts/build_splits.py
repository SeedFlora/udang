"""Build all leakage-safe split tables and verify them.

Usage:
    python scripts/build_splits.py [--seed 42] [--folds 5]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.splits import (  # noqa: E402
    leakage_report, make_biomass_lodo, make_biomass_loso, make_biomass_within_db,
    make_cross_dataset, make_disease_cv, make_disease_cv_single_dataset,
    make_disease_cv_withaug, make_disease_naive, naive_test_composition,
    verify_no_leakage,
)
from src.utils.hashing import hash_dataframe  # noqa: E402
from src.utils.paths import MANIFESTS, SPLITS, ensure_dirs, rel_to_repo  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    ensure_dirs()

    man = pd.read_parquet(MANIFESTS / "unified_manifest.parquet")

    parts = []
    print("building disease_cv (primary: originals only, annotated copies excluded) ...")
    base = make_disease_cv(man, n_splits=args.folds, seed=args.seed,
                           include_published_aug=False, protocol_name="disease_cv")
    parts.append(base)

    print("building disease_cv_withaug (ABLATION: augmentations added to TRAIN only, "
          "folds inherited from disease_cv so the test set is identical) ...")
    parts.append(make_disease_cv_withaug(man, base))

    print("building per-dataset disease CV (E1 single-dataset baselines) ...")
    for ds, proto in (("ShrimpDiseaseImageBD", "disease_cv_sdibd"),
                      ("TigerShrimpBD", "disease_cv_tiger")):
        parts.append(make_disease_cv_single_dataset(man, ds, proto,
                                                    n_splits=args.folds, seed=args.seed))

    print("building cross-dataset protocols (both directions) ...")
    parts.append(make_cross_dataset(man, seed=args.seed))

    print("building biomass leave-one-DB-out ...")
    parts.append(make_biomass_lodo(man, seed=args.seed))

    print("building biomass leave-one-SITE-out (DB1 | DB2+DB3 | DB4+DB5) ...")
    parts.append(make_biomass_loso(man, seed=args.seed))

    print("building biomass within-database grouped CV (protocol control) ...")
    parts.append(make_biomass_within_db(man, n_splits=args.folds, seed=args.seed))

    safe = pd.concat(parts, ignore_index=True)

    print("\nverifying no group crosses train/val/test in any protocol/fold ...")
    report = verify_no_leakage(safe, man)   # raises on violation -- the gate is intact
    cc = report.get("manifest_cross_check") or {}
    print(f"LEAKAGE CHECK PASSED "
          f"(self-consistency + manifest cross-check over "
          f"{len(cc.get('checked_protocols', []))} protocol/fold combinations)")

    # ---- the naive control arm: MEASURED, deliberately NOT gated ------------
    # It exists to show what an image-level random split places in test. Running
    # verify_no_leakage on it would (correctly) abort the build, so its leakage is
    # recorded by the same leakage_report the gate uses and written to the report.
    print("\nbuilding disease_cv_naive_withaug (CONTROL ARM: image-level random split, "
          "no grouping; leakage is measured below, not prevented) ...")
    naive = make_disease_naive(man, n_splits=args.folds, seed=args.seed)
    naive_rep = leakage_report(naive, man)
    # The ONLY violations a naive arm may show are the ones it exists to show: true
    # groups / animals straddling splits. An empty split or a duplicated record
    # would be a bug, not a finding, so those still abort the build.
    unexpected = [v for v in naive_rep["violations"]
                  if v.get("key") not in ("manifest_group", "manifest_individual")]
    if unexpected:
        raise AssertionError(f"naive control arm is malformed, not merely leaky: {unexpected[:3]}")
    comp = naive_test_composition(naive, man)
    report["naive_control_arm"] = {
        "protocol": "disease_cv_naive_withaug",
        "gated": False,
        "n_measured_violations": len(naive_rep["violations"]),
        # keep the per-protocol/fold detail (n_straddling + examples), not just a
        # count: the arm exists to be reported, and split_report.json is the report
        "measured_cross_check_violations": (naive_rep.get("manifest_cross_check") or {}
                                            ).get("violations", []),
        "test_composition_per_fold": comp,
    }
    for c in comp:
        print(f"  naive fold {c['fold']}: test={c['n_test']} "
              f"group-leaked={c['frac_test_group_leaked']:.1%} "
              f"synthetic={c['frac_test_synthetic']:.1%}")

    splits = pd.concat([safe, naive], ignore_index=True)

    # the ablation must be scored on exactly the baseline's test records
    a = set(splits[(splits.protocol == "disease_cv") & (splits.split == "test")]["record_id"])
    b = set(splits[(splits.protocol == "disease_cv_withaug") & (splits.split == "test")]["record_id"])
    print(f"ablation test-set identity: disease_cv test={len(a)}, withaug test={len(b)}, "
          f"identical={a == b}")
    if a != b:
        raise AssertionError(
            "disease_cv_withaug must be scored on the SAME test records as disease_cv, "
            f"otherwise the ablation confounds train and test changes "
            f"({len(a ^ b)} records differ)")

    out = SPLITS / "splits.parquet"
    splits.to_parquet(out, index=False)
    splits.to_csv(SPLITS / "splits.csv", index=False)
    for proto, g in splits.groupby("protocol"):
        g.to_csv(SPLITS / f"{proto}.csv", index=False)

    sh = hash_dataframe(splits, ["record_id", "protocol", "fold", "split"])
    report["protocol_membership_hashes"] = {
        proto: hash_dataframe(g, ["record_id", "fold", "split"])
        for proto, g in splits.groupby("protocol")
    }
    gh = hash_dataframe(man[~man["exclude"].astype(bool)],
                        ["record_id", "group_id", "individual_id"])
    (SPLITS / "grouping_hash.txt").write_text(gh, encoding="utf-8")
    report["grouping_hash"] = gh
    print(f"grouping hash: {gh}")
    (SPLITS / "splits_hash.txt").write_text(sh, encoding="utf-8")
    (SPLITS / "split_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")

    print(f"\nsplit hash: {sh}")
    for proto, folds in report["protocols"].items():
        print(f"\n{proto}:")
        for f, info in sorted(folds.items()):
            bs = info["by_split"]
            gs = info["groups_by_split"]
            print(f"  fold {f}: " +
                  " ".join(f"{k}={bs.get(k, 0)}({gs.get(k, 0)}g)"
                           for k in ("train", "val", "test")))
    print(f"\n-> {rel_to_repo(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
