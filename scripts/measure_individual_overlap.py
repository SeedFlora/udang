"""How many TEST images share a physical animal with train/val of the same fold?

Derives animal identity from ShrimpDiseaseImageBD filenames INDEPENDENTLY of the
manifest's individual_id column, so it can be run against split tables that were
built before that column existed (the submitted round-1 splits) and against the
corrected ones. TigerShrimpBD carries no animal identity and is reported as
"not identifiable" rather than assumed clean.

Usage:
    python scripts/measure_individual_overlap.py [--splits-dir data/splits]
        [--manifest data/manifests/unified_manifest.parquet]
        [--out outputs/metrics/individual_overlap.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.manifest import parse_sdibd_name, sdibd_individual_id  # noqa: E402
from src.utils.paths import MANIFESTS, METRICS, SPLITS, rel_to_repo  # noqa: E402

DISEASE_PROTOCOL_PREFIXES = ("disease_cv", "cross_")

TABLE_COLUMNS = ["protocol", "fold", "n_test", "n_test_identifiable",
                  "n_test_unmatched", "n_test_shared_animal", "frac_shared"]


def animal_key(row: pd.Series) -> str | None:
    if row["dataset"] != "ShrimpDiseaseImageBD":
        return None
    _tok, ind, _view = parse_sdibd_name(str(row["original_filename"]))
    return sdibd_individual_id(str(row["raw_class"]), ind) if ind is not None else None


def overlap_table(splits: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    # NOTE: built as a plain Python list via iterrows(), not manifest.apply(axis=1).
    # Under pandas 3.0's default string-dtype inference, .apply(axis=1) over a
    # function that sometimes returns None silently coerces those Nones into
    # float('nan') in the resulting Series -- which then survives `is not None`
    # and `- {None}` checks below, over-counting TigerShrimpBD as identifiable.
    keys = [animal_key(row) for _, row in manifest.iterrows()]
    manifest_ids = set(manifest["record_id"])
    key_of = dict(zip(manifest["record_id"], keys, strict=True))
    rows = []
    for (proto, fold), g in splits.groupby(["protocol", "fold"]):
        te = g[g["split"] == "test"]
        trv = g[g["split"].isin(["train", "val"])]
        trv_animals = {key_of.get(r) for r in trv["record_id"]} - {None}
        te_ids = list(te["record_id"])
        unmatched = [r for r in te_ids if r not in manifest_ids]
        te_keys = [key_of.get(r) for r in te_ids]
        ident = [k for k in te_keys if k is not None]
        shared = [k for k in ident if k in trv_animals]
        rows.append({
            "protocol": proto, "fold": int(fold), "n_test": int(len(te)),
            "n_test_identifiable": len(ident), "n_test_unmatched": len(unmatched),
            "n_test_shared_animal": len(shared),
            "frac_shared": (len(shared) / len(ident)) if ident else None,
        })
    t = pd.DataFrame(rows, columns=TABLE_COLUMNS)
    # Keep genuine Python None (-> JSON null) for "unmeasurable", never NaN: a
    # naive numeric column would collapse None to NaN, and NaN prints as the
    # bare (invalid-JSON) token `NaN`, not `null` -- and reads too easily as a
    # measured 0.0 ("no leakage") instead of "nothing to measure".
    t["frac_shared"] = t["frac_shared"].astype(object).where(t["frac_shared"].notna(), None)
    return t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits-dir", default=str(SPLITS))
    ap.add_argument("--manifest", default=str(MANIFESTS / "unified_manifest.parquet"))
    ap.add_argument("--out", default=str(METRICS / "individual_overlap.json"))
    args = ap.parse_args()

    spl = pd.read_parquet(Path(args.splits_dir) / "splits.parquet")
    spl = spl[spl["protocol"].str.startswith(DISEASE_PROTOCOL_PREFIXES)]
    man = pd.read_parquet(args.manifest)
    t = overlap_table(spl, man)
    print(t.to_string(index=False))

    n_unmatched = int(t["n_test_unmatched"].sum())
    if n_unmatched > 0:
        raise SystemExit(
            f"{n_unmatched} split records are missing from the manifest; "
            "the join is broken and this table cannot be trusted"
        )

    measurable = t.dropna(subset=["frac_shared"])
    summary = (measurable.groupby("protocol")["frac_shared"]
               .agg(["min", "max", "mean"]).astype(float).round(4)
               .to_dict(orient="index"))
    for proto in t["protocol"].unique():
        if proto not in summary:
            summary[proto] = {"min": None, "max": None, "mean": None,
                               "note": "no identifiable test images"}

    out = {"splits_dir": rel_to_repo(args.splits_dir),
           "per_fold": t.to_dict(orient="records"),
           "per_protocol_frac_shared": summary}
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\n-> {rel_to_repo(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
