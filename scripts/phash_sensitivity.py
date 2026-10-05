"""Group count as a function of the perceptual-hash Hamming threshold.

Answers reviewer A10 ("thresholds should be justified") with a curve instead of
an assertion. Uses exactly the fusion call build_unified_manifest makes.

Usage:  python scripts/phash_sensitivity.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.audit.dedup import build_fused_groups  # noqa: E402
from src.data.manifest import forced_individual_links  # noqa: E402
from src.utils.paths import MANIFESTS, METRICS  # noqa: E402

THRESHOLDS = (0, 2, 4, 6, 8, 10)


def main() -> int:
    man = pd.read_parquet(MANIFESTS / "unified_manifest.parquet")
    usable = man[~man["exclude"].astype(bool)].copy()
    # pandas 3 string inference turns None into NaN on the round trip through
    # parquet, and NaN is truthy -- build_fused_groups would key one giant family
    # on it. Only a non-empty str counts as a family key.
    fam = [x if isinstance(x, str) and x else None
           for x in usable["augmentation_family_id"].tolist()]
    blocks = np.array([f"{r['dataset']}|{r['raw_class'] if r['task'] == 'disease' else r['individual_id']}"
                       for _, r in usable.iterrows()])
    ph = np.array([np.uint64(int(str(x), 16)) for x in usable["phash"].fillna("0")], dtype=np.uint64)
    extra = forced_individual_links(usable)
    pipeline_threshold = 6
    manifest_n_groups = int(man.loc[~man["exclude"].astype(bool), "group_id"].nunique())
    out = {"pipeline_threshold": pipeline_threshold, "manifest_n_groups": manifest_n_groups}
    for t in THRESHOLDS:
        _m, d = build_fused_groups(list(usable["record_id"]), list(usable["sha256"].fillna("")),
                                   ph, fam, phash_threshold=t, near_dup_blocks=blocks, extra_links=extra)
        out[str(t)] = {"n_groups": d["n_groups"], "n_near_dup_links": d["n_near_dup_links"],
                       "largest_group": d["largest_group"], "n_singletons": d["n_singleton_groups"]}
        print(f"H<={t:2d}: groups={d['n_groups']:5d} near_links={d['n_near_dup_links']:6d} "
              f"largest={d['largest_group']:3d}")
        if t == pipeline_threshold:
            match = d["n_groups"] == manifest_n_groups
            print(f"  H<={pipeline_threshold} n_groups {'==' if match else '!='} "
                  f"manifest group_id nunique ({d['n_groups']} vs {manifest_n_groups})")
    out_path = METRICS / "phash_sensitivity.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
