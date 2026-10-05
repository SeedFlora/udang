"""Torch datasets and samplers built on the unified manifest + split table.

Every dataset returns supervision MASKS alongside targets. A mask of 0 means the
source dataset never assessed that label; the loss must skip it. Nothing in this
module ever substitutes a zero for an unknown.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from src.data.cache import resolve_image_path
from src.data.labels import DISEASE_ATTRS, N_ATTRS
from src.utils.hashing import stable_seed
from src.utils.paths import REPO_ROOT

BIOMASS_TARGETS = ("log_weight", "length_cm", "cephalothorax_cm")


def _load_rgb(path: Path) -> np.ndarray:
    from PIL import Image
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


class DiseaseDataset(Dataset):
    """One image -> (health target, 3 attribute targets, 3 attribute masks)."""

    def __init__(self, manifest: pd.DataFrame, record_ids, transform,
                 domain_map: dict[str, int] | None = None, root: Path = REPO_ROOT):
        idx = manifest.set_index("record_id")
        self.df = idx.loc[list(record_ids)].reset_index()
        self.transform = transform
        self.root = Path(root)
        doms = sorted(self.df["dataset"].unique())
        self.domain_map = domain_map or {d: i for i, d in enumerate(doms)}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> dict:
        r = self.df.iloc[i]
        img = _load_rgb(resolve_image_path(r["relative_image_path"], self.root))
        x = self.transform(img)
        attrs = np.array([float(r[a]) for a in DISEASE_ATTRS], dtype=np.float32)
        masks = np.array([float(r[f"label_mask_{a}"]) for a in DISEASE_ATTRS], dtype=np.float32)
        return {
            "image": x,
            "health": torch.tensor(1.0 if r["health_status"] == "diseased" else 0.0),
            "attrs": torch.from_numpy(attrs),
            "attr_mask": torch.from_numpy(masks),
            "domain": torch.tensor(self.domain_map.get(r["dataset"], 0), dtype=torch.long),
            "record_id": r["record_id"],
            "dataset": r["dataset"],
            "raw_class": str(r["raw_class"]),
        }


class BiomassImageDataset(Dataset):
    """One image -> (log_weight, length_cm, cephalothorax_cm) + masks.

    Cephalothorax is absent for DB1, so its mask is 0 there. Weight is modelled in
    natural-log space; `biomass_pool` guarantees weight > 0 before this point.
    """

    def __init__(self, manifest: pd.DataFrame, record_ids, transform,
                 root: Path = REPO_ROOT, domain_map: dict[str, int] | None = None):
        idx = manifest.set_index("record_id")
        self.df = idx.loc[list(record_ids)].reset_index()
        self.transform = transform
        self.root = Path(root)
        dbs = sorted(self.df["pond_or_db_id"].dropna().unique())
        self.domain_map = domain_map or {d: i for i, d in enumerate(dbs)}

    def __len__(self) -> int:
        return len(self.df)

    def _targets(self, r) -> tuple[np.ndarray, np.ndarray]:
        w, ln, ce = r["weight_g"], r["length_cm"], r["cephalothorax_cm"]
        t = np.zeros(3, dtype=np.float32)
        m = np.zeros(3, dtype=np.float32)
        if pd.notna(w) and w > 0:
            t[0], m[0] = math.log(float(w)), 1.0
        if pd.notna(ln) and ln > 0:
            t[1], m[1] = float(ln), 1.0
        if pd.notna(ce) and ce > 0:
            t[2], m[2] = float(ce), 1.0
        return t, m

    def __getitem__(self, i: int) -> dict:
        r = self.df.iloc[i]
        img = _load_rgb(resolve_image_path(r["relative_image_path"], self.root))
        x = self.transform(img)
        t, m = self._targets(r)
        return {
            "image": x,
            "biomass": torch.from_numpy(t),
            "biomass_mask": torch.from_numpy(m),
            "individual_id": r["individual_id"],
            "db": r["pond_or_db_id"],
            "domain": torch.tensor(self.domain_map.get(r["pond_or_db_id"], 0), dtype=torch.long),
            "record_id": r["record_id"],
            "weight_g": float(r["weight_g"]) if pd.notna(r["weight_g"]) else float("nan"),
            "length_cm": float(r["length_cm"]) if pd.notna(r["length_cm"]) else float("nan"),
            "complete_body": float(r["complete_body"]) if pd.notna(r["complete_body"]) else float("nan"),
        }


class BiomassIndividualDataset(Dataset):
    """One INDIVIDUAL -> up to `max_views` images + one target vector.

    View selection is deterministic given (seed, epoch, individual) so runs are
    reproducible, and evaluation can request an exact number of views to produce
    the 1 / 3 / 5 / all-view comparison the specification asks for.
    """

    def __init__(self, manifest: pd.DataFrame, record_ids, transform,
                 max_views: int = 4, root: Path = REPO_ROOT, seed: int = 0,
                 deterministic_first_n: bool = False):
        idx = manifest.set_index("record_id")
        df = idx.loc[list(record_ids)].reset_index()
        self.df = df
        self.transform = transform
        self.root = Path(root)
        self.max_views = max_views
        self.seed = seed
        self.epoch = 0
        self.deterministic_first_n = deterministic_first_n
        self.groups: list[str] = sorted(df["individual_id"].dropna().unique())
        self._by_ind = {k: g.sort_values("relative_image_path")
                        for k, g in df.groupby("individual_id")}

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.groups)

    def _pick(self, ind: str, n_avail: int) -> np.ndarray:
        k = min(self.max_views, n_avail)
        if self.deterministic_first_n:
            return np.arange(k)
        # stable_seed (SHA-256), NOT builtin hash(): hash() salts string hashing
        # per process, so seeding off hash((..., ind)) would pick a different view
        # subset on every fresh interpreter and make multi-view runs unreproducible.
        rng = np.random.default_rng(stable_seed(self.seed, self.epoch, ind))
        return rng.permutation(n_avail)[:k]

    def __getitem__(self, i: int) -> dict:
        ind = self.groups[i]
        g = self._by_ind[ind]
        sel = self._pick(ind, len(g))
        # resolve_image_path, not root/rel: the multi-view path must read the SAME
        # pixels as the single-view path, and it must not forfeit the cache. Reading
        # originals here measured 18.0 ms/image and 67.9 MB for 200 images versus
        # 2.1 ms and 6.6 MB through the cache -- 8.5x slower on every batch, in the
        # one training mode that loads several images per sample.
        imgs = [self.transform(
            _load_rgb(resolve_image_path(g.iloc[j]["relative_image_path"], self.root)))
            for j in sel]

        v = self.max_views
        c, h, w = imgs[0].shape
        stack = torch.zeros(v, c, h, w, dtype=imgs[0].dtype)
        vm = torch.zeros(v)
        for j, im in enumerate(imgs[:v]):
            stack[j] = im
            vm[j] = 1.0

        r = g.iloc[0]
        t = np.zeros(3, dtype=np.float32)
        m = np.zeros(3, dtype=np.float32)
        if pd.notna(r["weight_g"]) and r["weight_g"] > 0:
            t[0], m[0] = math.log(float(r["weight_g"])), 1.0
        if pd.notna(r["length_cm"]) and r["length_cm"] > 0:
            t[1], m[1] = float(r["length_cm"]), 1.0
        if pd.notna(r["cephalothorax_cm"]) and r["cephalothorax_cm"] > 0:
            t[2], m[2] = float(r["cephalothorax_cm"]), 1.0

        return {
            "views": stack,
            "view_mask": vm,
            "biomass": torch.from_numpy(t),
            "biomass_mask": torch.from_numpy(m),
            "individual_id": ind,
            "db": r["pond_or_db_id"],
            "n_available_views": len(g),
            "weight_g": float(r["weight_g"]) if pd.notna(r["weight_g"]) else float("nan"),
            "length_cm": float(r["length_cm"]) if pd.notna(r["length_cm"]) else float("nan"),
        }


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
class SourceBalancedSampler(Sampler[int]):
    """Draw each source dataset with equal probability.

    Pooled disease training is dominated by whichever dataset is larger; without
    rebalancing the model can score well by specialising on the majority source.
    This sampler makes the source-balanced-vs-ordinary comparison a controlled
    one-variable ablation.
    """

    def __init__(self, sources: list[str], num_samples: int | None = None,
                 seed: int = 42, replacement: bool = True):
        self.sources = np.asarray(sources)
        self.uniq = sorted(set(sources))
        self.by_source = {s: np.nonzero(self.sources == s)[0] for s in self.uniq}
        self.num_samples = num_samples or len(sources)
        self.seed = seed
        self.replacement = replacement
        self.epoch = 0
        self._iterations = 0

    def set_epoch(self, epoch: int) -> None:
        """Optional explicit control. Not required: __iter__ advances its own
        counter, so the sampler stays correct even if nobody calls this."""
        self.epoch = int(epoch)
        self._iterations = int(epoch)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        # The epoch counter advances HERE rather than relying on an external
        # set_epoch call. Depending on a caller to advance it is fragile: nothing
        # in the training loop called set_epoch, so every epoch replayed one
        # identical index multiset and the source-balanced arm never saw roughly
        # half of its training rows -- silently turning the sampling ablation into
        # a fixed-subsample ablation. DataLoader calls __iter__ once per epoch in
        # the parent process, so this counter is both correct and self-contained.
        rng = np.random.default_rng(self.seed + self._iterations)
        self._iterations += 1
        per = int(math.ceil(self.num_samples / len(self.uniq)))
        idx: list[int] = []
        for s in self.uniq:
            pool = self.by_source[s]
            take = rng.choice(pool, size=per, replace=self.replacement or per > len(pool))
            idx.extend(take.tolist())
        idx = np.asarray(idx)
        idx = rng.permutation(idx)[: self.num_samples]
        return iter(idx.tolist())


def class_balance_report(df: pd.DataFrame) -> dict:
    out = {
        "n": int(len(df)),
        "by_dataset": df["dataset"].value_counts().to_dict(),
        "by_raw_class": df["raw_class"].value_counts(dropna=False).to_dict(),
        "health": df["health_status"].value_counts(dropna=False).to_dict(),
    }
    for a in DISEASE_ATTRS:
        sup = df[f"label_mask_{a}"].fillna(0) > 0
        out[f"{a}_supervised"] = int(sup.sum())
        out[f"{a}_positive"] = int((df.loc[sup, a] > 0).sum())
    return out
