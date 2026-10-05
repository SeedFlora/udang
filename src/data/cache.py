"""Pre-resized image cache.

ShrimpDiseaseImageBD ships every image at 2048x2048 (4.19 MP), so one pass over
the primary disease pool decodes ~5.8 gigapixels. That decode, not the GPU,
dominates epoch time. Caching a max-side-448 copy cuts it by roughly 20x.

This is faithful, not lossy in any way that matters: training letterboxes to 320,
so a 448-max-side source already carries more detail than the network consumes,
plus headroom for rotation and mild scale augmentation. Images already smaller
than the target are copied through untouched rather than upscaled.
"""
from __future__ import annotations

import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from src.utils.paths import PROCESSED, REPO_ROOT

CACHE_ROOT = PROCESSED / "cache"
DEFAULT_MAX_SIDE = 448
JPEG_QUALITY = 95


def cache_path_for(rel: str, cache_root: Path = CACHE_ROOT) -> Path:
    """Cache location mirroring the source-relative path."""
    return cache_root / rel


def resolve_image_path(rel: str, root: Path = REPO_ROOT,
                       cache_root: Path = CACHE_ROOT) -> Path:
    """Cached copy when present, otherwise the original file."""
    c = cache_path_for(rel, cache_root)
    if c.exists():
        return c
    return root / rel


def _resize_one(args: tuple[str, str, str, int]) -> tuple[str, bool, str]:
    rel, root, cache_root, max_side = args
    from PIL import Image
    src = Path(root) / rel
    dst = Path(cache_root) / rel
    if dst.exists():
        return rel, True, "cached"
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src) as im:
            im = im.convert("RGB")
            w, h = im.size
            if max(w, h) <= max_side:
                # Already small enough: copy verbatim so no resampling is applied.
                if src.suffix.lower() in (".jpg", ".jpeg"):
                    shutil.copyfile(src, dst)
                else:
                    im.save(dst, quality=JPEG_QUALITY, subsampling=0)
                return rel, True, "copied"
            scale = max_side / float(max(w, h))
            im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                           Image.LANCZOS)
            im.save(dst, format="JPEG", quality=JPEG_QUALITY, subsampling=0)
        return rel, True, "resized"
    except Exception as e:  # noqa: BLE001
        return rel, False, f"{type(e).__name__}: {e}"


CACHE_META = "_cache_meta.json"


def read_cache_meta(cache_root: Path = CACHE_ROOT) -> dict:
    import json
    p = cache_root / CACHE_META
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def build_cache(manifest: pd.DataFrame, max_side: int = DEFAULT_MAX_SIDE,
                workers: int = 10, root: Path = REPO_ROOT,
                cache_root: Path = CACHE_ROOT, force: bool = False) -> dict:
    """Build (or extend) the cache.

    _resize_one short-circuits on dst.exists(), which makes a second run with a
    DIFFERENT --max-side a silent no-op: the files stay at the old resolution while
    the run reports the new number. The cache therefore records the max_side it was
    built with and refuses a conflicting request instead of misreporting provenance.
    """
    import json

    rels = [r for r in manifest.loc[~manifest["exclude"].astype(bool),
                                    "relative_image_path"].tolist()]
    cache_root.mkdir(parents=True, exist_ok=True)

    prev = read_cache_meta(cache_root)
    if prev and prev.get("max_side") != max_side and not force:
        raise ValueError(
            f"cache at {cache_root} was built with max_side={prev.get('max_side')}, "
            f"but max_side={max_side} was requested. Existing files would be reused "
            f"unchanged and the reported resolution would be wrong. Delete the cache "
            f"directory or pass force=True to rebuild."
        )
    args = [(r, str(root), str(cache_root), max_side) for r in rels]
    stats = {"resized": 0, "copied": 0, "cached": 0, "failed": 0}
    failures: list[str] = []
    chunk = max(1, len(args) // (workers * 8))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for rel, ok, how in ex.map(_resize_one, args, chunksize=chunk):
            if ok:
                stats[how] = stats.get(how, 0) + 1
            else:
                stats["failed"] += 1
                failures.append(f"{rel}: {how}")
    stats["n_total"] = len(rels)
    stats["max_side"] = max_side
    stats["jpeg_quality"] = JPEG_QUALITY
    stats["failures"] = failures[:20]
    size = sum(p.stat().st_size for p in cache_root.rglob("*") if p.is_file())
    stats["cache_size_mb"] = round(size / 1e6, 1)
    (cache_root / CACHE_META).write_text(
        json.dumps({"max_side": max_side, "jpeg_quality": JPEG_QUALITY,
                    "n_files": len(rels)}, indent=2), encoding="utf-8")
    return stats
