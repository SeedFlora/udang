"""Visual sanity-check grids for every augmentation pipeline (specification section 9).

The point is to be able to LOOK at what the network is trained on and confirm that
no augmentation destroys the signal it is supposed to learn:

  disease  - Black Gill, WSSV and Yellow Head are colour- and texture-dependent, so
             the default pipeline avoids hue shifts and aggressive saturation. The
             `strong_color` grid shows the ablation-only pipeline for comparison.
  biomass  - length and weight depend on the whole body being visible and
             undistorted, so nothing may crop the rostrum/tail or change aspect ratio.

Usage:
    python scripts/make_augmentation_grids.py [--n-images 4] [--n-variants 6]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.cache import resolve_image_path  # noqa: E402
from src.data.transforms import (  # noqa: E402
    build_biomass_transforms, build_disease_transforms, save_augmentation_grid,
)
from src.utils.paths import FIGURES, MANIFESTS, REPO_ROOT, ensure_dirs  # noqa: E402


def pick(man: pd.DataFrame, query: str, n: int, seed: int = 0) -> list[str]:
    sub = man.query(query) if query else man
    sub = sub[~sub["exclude"].astype(bool)]
    if len(sub) == 0:
        return []
    sub = sub.sample(n=min(n, len(sub)), random_state=seed)
    return [str(resolve_image_path(p, REPO_ROOT))
            for p in sub["relative_image_path"]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-images", type=int, default=4)
    ap.add_argument("--n-variants", type=int, default=6)
    ap.add_argument("--image-size", type=int, default=320)
    args = ap.parse_args()
    ensure_dirs()
    out_dir = FIGURES / "augmentation_checks"
    out_dir.mkdir(parents=True, exist_ok=True)

    man = pd.read_parquet(MANIFESTS / "unified_manifest.parquet")
    made: list[str] = []

    grids = [
        ("disease_train_safe",
         build_disease_transforms(args.image_size, train=True, strong_color=False),
         "task=='disease' and record_role in ['raw','original']",
         "Disease TRAIN pipeline (default, disease-safe): flips, rotation, mild "
         "shift/scale, mild brightness/contrast, mild blur/noise, letterbox. "
         "No hue shift - disease signs are colour-dependent."),
        ("disease_train_strong_color",
         build_disease_transforms(args.image_size, train=True, strong_color=True),
         "task=='disease' and record_role in ['raw','original']",
         "Disease TRAIN pipeline with ABLATION-ONLY aggressive colour transforms. "
         "Used solely to measure how much colour distortion costs; not a default."),
        ("disease_eval",
         build_disease_transforms(args.image_size, train=False),
         "task=='disease' and record_role in ['raw','original']",
         "Disease EVAL pipeline: letterbox + normalise only (deterministic)."),
        ("biomass_train",
         build_biomass_transforms(args.image_size, train=True),
         "task=='biomass'",
         "Biomass TRAIN pipeline: rotation, flips, mild illumination, mild blur, "
         "letterbox. No random crop (would remove rostrum/tail/cephalothorax) and "
         "no aspect-ratio distortion."),
        ("biomass_eval",
         build_biomass_transforms(args.image_size, train=False),
         "task=='biomass'",
         "Biomass EVAL pipeline: letterbox + normalise only (deterministic)."),
    ]

    for name, tf, query, caption in grids:
        paths = pick(man, query, args.n_images, seed=7)
        if not paths:
            print(f"skip {name}: no matching images")
            continue
        out = out_dir / f"{name}.png"
        try:
            save_augmentation_grid(tf, paths, str(out), n_variants=args.n_variants,
                                   seed=7)
            made.append(out.name)
            print(f"[ok] {name}: {caption[:70]}")
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")

    # per-class disease grids so each disease sign can be inspected separately
    for cls in ["Healthy", "BG", "WSSV", "BG_WSSV", "Black Gill", "Yellow Head"]:
        paths = pick(man, f"task=='disease' and raw_class=='{cls}' and "
                          f"record_role in ['raw','original']", 3, seed=11)
        if not paths:
            continue
        safe = cls.replace(" ", "_")
        out = out_dir / f"disease_class_{safe}.png"
        try:
            save_augmentation_grid(
                build_disease_transforms(args.image_size, train=True),
                paths, str(out), n_variants=5, seed=11)
            made.append(out.name)
            print(f"[ok] class grid: {cls}")
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] class grid {cls}: {e}")

    (out_dir / "README.md").write_text(
        "# Augmentation sanity-check grids\n\n"
        "Each grid shows source images (rows) and augmented variants (columns) as the "
        "network actually receives them, de-normalised back to displayable RGB.\n\n"
        + "\n".join(f"- `{n}`" for n in sorted(made))
        + "\n\nDesign constraints these grids exist to verify:\n\n"
          "1. Disease augmentation must not erase or fabricate a disease sign. Black "
          "Gill, WSSV and Yellow Head are colour- and texture-dependent, so hue "
          "shifts, channel shuffling, grayscale and heavy saturation are excluded "
          "from the default pipeline and appear only in the `strong_color` ablation.\n"
          "2. Biomass augmentation must never crop body endpoints (rostrum, tail, "
          "cephalothorax) and must never distort aspect ratio, because both would "
          "change the apparent size the model regresses on. Letterbox padding is used "
          "instead of resize-to-square.\n"
          "3. Eval pipelines must be deterministic.\n",
        encoding="utf-8")
    print(f"\nwrote {len(made)} grids to outputs/figures/augmentation_checks/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
