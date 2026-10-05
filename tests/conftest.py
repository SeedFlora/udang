import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.manifest import MANIFEST_COLUMNS  # noqa: E402


@pytest.fixture
def synthetic_manifest() -> pd.DataFrame:
    """A small manifest exercising every structural case the splitters must handle:
    an augmentation family, an exact duplicate pair, partial-label masks, and a
    multi-view biomass individual."""
    rows = []

    def base(**kw):
        r = {c: np.nan for c in MANIFEST_COLUMNS}
        r.update({"exclude": False, "exclude_reason": "", "split": "", "fold": -1,
                  "is_published_augmentation": False, "record_role": "raw",
                  "quality_flags": "", "annotation_path": None})
        r.update(kw)
        return r

    # TigerShrimpBD: 4 originals per class + an augmentation family for one of them
    for cls, n in [("Healthy", 6), ("Black Gill", 6), ("WSSV", 6), ("Yellow Head", 6)]:
        for i in range(n):
            rid = f"TG-{cls[:2]}{i:02d}"
            gid = f"grp_tg_{cls[:2]}_{i}"
            rows.append(base(
                record_id=rid, dataset="TigerShrimpBD", dataset_version="1",
                task="disease", raw_class=cls, group_id=gid,
                relative_image_path=f"data/raw/tg/{cls}/orig/{rid}.jpg",
                original_filename=f"{rid}.jpg", sha256=f"sha_{rid}",
                phash="0000000000000000", width=640, height=480, file_size=1000,
                health_status="healthy" if cls == "Healthy" else "diseased",
                black_gill=1 if cls == "Black Gill" else 0,
                wssv=1 if cls == "WSSV" else 0,
                yellow_head=1 if cls == "Yellow Head" else 0,
                label_mask_black_gill=1, label_mask_wssv=1, label_mask_yellow_head=1,
                duplicate_cluster_id=f"dup_{rid}",
                augmentation_family_id=f"tiger::{cls}::{rid}",
                source_domain="TigerShrimpBD", source_subdomain="original",
            ))
            # two published augmented copies share the SAME group id
            for k in range(2):
                arid = f"{rid}_aug{k}"
                rows.append(base(
                    record_id=arid, dataset="TigerShrimpBD", dataset_version="1",
                    task="disease", raw_class=cls, group_id=gid,
                    relative_image_path=f"data/raw/tg/{cls}/aug/{arid}.jpg",
                    original_filename=f"aug_{k}_{rid}.jpg", sha256=f"sha_{arid}",
                    phash="0000000000000000", width=640, height=480, file_size=1000,
                    health_status="healthy" if cls == "Healthy" else "diseased",
                    black_gill=1 if cls == "Black Gill" else 0,
                    wssv=1 if cls == "WSSV" else 0,
                    yellow_head=1 if cls == "Yellow Head" else 0,
                    label_mask_black_gill=1, label_mask_wssv=1, label_mask_yellow_head=1,
                    is_published_augmentation=True, record_role="augmented",
                    duplicate_cluster_id=f"dup_{arid}",
                    augmentation_family_id=f"tiger::{cls}::{rid}",
                    source_domain="TigerShrimpBD", source_subdomain="augmented",
                ))

    # ShrimpDiseaseImageBD: yellow_head UNKNOWN (mask 0) everywhere
    for cls, bg, wssv in [("Healthy", 0, 0), ("BG", 1, 0), ("WSSV", 0, 1),
                          ("BG_WSSV", 1, 1)]:
        for i in range(6):
            rid = f"SD-{cls[:3]}{i:02d}"
            rows.append(base(
                record_id=rid, dataset="ShrimpDiseaseImageBD", dataset_version="3",
                task="disease", raw_class=cls,
                group_id=f"grp_sd_{cls}_{i // 3}",                 # one group per ANIMAL
                individual_id=f"sdibd::{cls}::{i // 3:03d}",       # class-scoped animal key
                relative_image_path=f"data/raw/sd/{cls}/{rid}.jpg",
                original_filename=f"{cls}-{i // 3 + 1}-img-{i % 3 + 1}.jpg",
                sha256=f"sha_{rid}",
                phash="0000000000000000", width=800, height=600, file_size=2000,
                health_status="healthy" if cls == "Healthy" else "diseased",
                black_gill=bg, wssv=wssv, yellow_head=0,
                label_mask_black_gill=1, label_mask_wssv=1,
                label_mask_yellow_head=0,          # <- UNKNOWN, not negative
                duplicate_cluster_id=f"dup_{rid}",
                source_domain="ShrimpDiseaseImageBD", source_subdomain="raw",
            ))

    # Biomass: 5 individuals per DB, 4 views each
    for db_i, db in enumerate(["DB1", "DB2", "DB3", "DB4", "DB5"]):
        for s in range(5):
            ind = f"{db}_s{s:03d}"
            for v in range(4):
                rid = f"BM-{db}-{s}-{v}"
                rows.append(base(
                    record_id=rid, dataset="BiomassVannamei", dataset_version="2",
                    task="biomass", group_id=ind, individual_id=ind,
                    pond_or_db_id=db,
                    relative_image_path=f"data/raw/bm/{db}/s{s}_{v}.png",
                    original_filename=f"s{s}_{v}.png", sha256=f"sha_{rid}",
                    phash="0000000000000000", width=512, height=512, file_size=3000,
                    weight_g=10.0 + s + db_i, length_cm=9.0 + 0.3 * s,
                    cephalothorax_cm=np.nan if db == "DB1" else 3.0 + 0.1 * s,
                    complete_body=1.0,
                    source_domain="BiomassVannamei", source_subdomain=db,
                ))
    return pd.DataFrame(rows)[MANIFEST_COLUMNS]
