"""Grad-CAM for disease models, saliency for biomass, attention weights for multi-view.

Specification section 17.3 (OPTIONAL - run only after the mandatory matrix).

These are DIAGNOSTIC visualizations. A bright region means the network's output was
sensitive to that region; it is not evidence of a biological mechanism. Read them
to spot shortcut learning (background, hands, container edges), not to claim the
model "found the lesion".

With no flags the panel is drawn from the whole eligible manifest, which is fine
for a quick look but says nothing about generalisation: some of those images were
in the run's training set. Pass `--config <run-name prefix>` to scope the panel to
the rows that run actually held out (its protocol comes from the run's
`outputs/logs/<run>_meta.json`, its fold from the checkpoint name).

Usage:
    python scripts/explainability.py [--n-images 8] [--config E2_pooled_bce_ordinary]
    python scripts/explainability.py --config E1_disease_single_tiger_f0_s42 \
        --split test --out outputs/figures/gradcam_e1.png --npz outputs/figures/gradcam_e1.npz
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.cache import resolve_image_path  # noqa: E402
from src.data.labels import DISEASE_ATTRS  # noqa: E402
from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD, build_disease_transforms  # noqa: E402
from src.models.multitask import ModelConfig, ShrimpMultiTaskModel  # noqa: E402
from src.utils.paths import CHECKPOINTS, FIGURES, LOGS, MANIFESTS, REPO_ROOT, SPLITS  # noqa: E402

DEFAULT_CONFIG = "E2_pooled_bce_ordinary"
_FOLD_RE = re.compile(r"_f(\d+)_")


class GradCAM:
    """Gradient-weighted class activation mapping on the last conv stage."""

    def __init__(self, model: torch.nn.Module, layer: torch.nn.Module):
        self.model = model
        self.acts: torch.Tensor | None = None
        self.grads: torch.Tensor | None = None
        layer.register_forward_hook(self._fwd)
        layer.register_full_backward_hook(self._bwd)

    def _fwd(self, _m, _i, o):
        self.acts = o.detach() if isinstance(o, torch.Tensor) else o[0].detach()

    def _bwd(self, _m, _gi, go):
        self.grads = go[0].detach()

    def __call__(self, x: torch.Tensor, score: torch.Tensor) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        score.backward(retain_graph=True)
        if self.acts is None or self.grads is None:
            return np.zeros((x.shape[-2], x.shape[-1]), dtype=np.float32)
        w = self.grads.mean(dim=(2, 3), keepdim=True)          # GAP over spatial dims
        cam = F.relu((w * self.acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam


def find_last_conv_stage(model: ShrimpMultiTaskModel) -> torch.nn.Module:
    """Last module producing a 4-D feature map, discovered by inspection rather
    than by hard-coding timm internals."""
    candidates = [m for m in model.encoder.modules()
                  if isinstance(m, (torch.nn.Conv2d, torch.nn.BatchNorm2d,
                                    torch.nn.LayerNorm))]
    convs = [m for m in model.encoder.modules() if isinstance(m, torch.nn.Conv2d)]
    return convs[-1] if convs else candidates[-1]


def denorm(t: torch.Tensor) -> np.ndarray:
    m = np.array(IMAGENET_MEAN).reshape(3, 1, 1)
    s = np.array(IMAGENET_STD).reshape(3, 1, 1)
    x = t.cpu().numpy() * s + m
    return np.clip(x.transpose(1, 2, 0), 0, 1)


# --------------------------------------------------------------------------- #
def fold_from_run_name(run_name: str) -> int | None:
    """Fold index encoded in a run name as `_f<k>_`; None when it is absent."""
    m = _FOLD_RE.search(str(run_name))
    return int(m.group(1)) if m else None


def resolve_checkpoint(config: str | None) -> Path | None:
    """First (sorted) `outputs/checkpoints/<config>*_best.pt`, with the same
    fallbacks the flagless invocation has always used."""
    cks = sorted(CHECKPOINTS.glob(f"{config or DEFAULT_CONFIG}*_best.pt"))
    if not cks and config is None:
        cks = (sorted(CHECKPOINTS.glob("E*disease*_best.pt"))
               or sorted(CHECKPOINTS.glob("E2_*_best.pt")))
    return cks[0] if cks else None


def run_protocol(run_name: str) -> str | None:
    """The protocol the run trained under, read from its run meta."""
    p = LOGS / f"{run_name}_meta.json"
    if not p.exists():
        return None
    meta = json.loads(p.read_text(encoding="utf-8"))
    proto = meta.get("protocol") or (meta.get("config") or {}).get("protocol")
    return str(proto) if isinstance(proto, str) and proto else None


def pick_examples(man: pd.DataFrame, splits: pd.DataFrame | None,
                  protocol: str | None, fold: int | None, split: str,
                  n: int) -> pd.DataFrame:
    """One example per (dataset, raw_class), capped at `n`.

    With a split table and a protocol the pool is restricted to the records that
    run assigned to `split` in `fold` -- so the panel shows images the run held
    out. With `splits=None` (or no protocol) the pool is the whole eligible
    manifest, which is the pre-flag behaviour. Published augmentations, excluded
    records and non-disease records are never eligible either way, and the draw
    is seeded so the same run always yields the same panel.
    """
    pool = man[(man["task"] == "disease") & (~man["exclude"].astype(bool))
               & (man["record_role"].isin(["raw", "original"]))]
    if splits is not None and protocol is not None:
        sel = splits[(splits["protocol"].astype(str) == str(protocol))
                     & (splits["fold"].astype(int) == int(fold))
                     & (splits["split"].astype(str) == str(split))]
        pool = pool[pool["record_id"].isin(set(sel["record_id"]))]
    if pool.empty:
        return pool
    # groupby.sample keeps all columns (groupby.apply drops the keys in pandas 2.x).
    return (pool.groupby(["dataset", "raw_class"], group_keys=False)
            .sample(n=1, random_state=3).head(n))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None,
                    help="run-name prefix; picks outputs/checkpoints/<config>*_best.pt "
                         "(first match). Default: the first disease checkpoint found.")
    ap.add_argument("--split", default="test", choices=["test", "val", "train"],
                    help="which split of the run's protocol/fold to draw from")
    ap.add_argument("--fold", type=int, default=None,
                    help="fold to draw from; default: parsed from the checkpoint name")
    ap.add_argument("--out", default=None, help="PNG path (default outputs/figures/gradcam_disease.png)")
    ap.add_argument("--npz", default=None,
                    help="also dump images/cams/p_diseased/labels/record_ids to this .npz")
    ap.add_argument("--n-images", type=int, default=8)
    ap.add_argument("--image-size", type=int, default=320)
    args = ap.parse_args()

    ck_path = resolve_checkpoint(args.config)
    if ck_path is None:
        print("BLOCKER: no disease checkpoint found. Run E1/E2 first.")
        return 2
    run_name = ck_path.name[: -len("_best.pt")]
    print(f"using {ck_path.name}")
    print(f"checkpoint: {ck_path.relative_to(REPO_ROOT).as_posix()}")

    # Naming a run scopes the panel to the rows that run held out. Without
    # --config there is no run to scope to, and the panel is drawn from the whole
    # eligible manifest exactly as it always was.
    protocol: str | None = None
    fold: int | None = None
    splits: pd.DataFrame | None = None
    if args.config is not None:
        protocol = run_protocol(run_name)
        fold = args.fold if args.fold is not None else fold_from_run_name(run_name)
        sp = SPLITS / "splits.parquet"
        if protocol is None:
            print(f"BLOCKER: cannot scope --config {args.config} to a held-out split: "
                  f"outputs/logs/{run_name}_meta.json has no protocol")
            return 2
        if fold is None:
            print(f"BLOCKER: no fold in run name {run_name!r} and no --fold given")
            return 2
        if not sp.exists():
            print("BLOCKER: data/splits/splits.parquet not found")
            return 2
        splits = pd.read_parquet(sp, columns=["record_id", "protocol", "fold", "split"])
        print(f"run {run_name}: protocol={protocol} fold={fold} split={args.split}")
    else:
        print(f"run {run_name}: no --config given -- drawing from the whole manifest "
              f"(images this run may have trained on)")

    ck = torch.load(ck_path, map_location="cpu", weights_only=True)
    mcfg = ModelConfig(pretrained=False, enable_biomass=False)
    raw = ck.get("config", {}).get("model") if isinstance(ck.get("config"), dict) else None
    if isinstance(raw, dict):
        for k, v in raw.items():
            if hasattr(mcfg, k) and k != "pretrained":
                setattr(mcfg, k, v)
        mcfg.pretrained = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ShrimpMultiTaskModel(mcfg).to(device)
    model.load_state_dict(ck["model_state"], strict=False)
    model.eval()

    cam = GradCAM(model, find_last_conv_stage(model))
    tf = build_disease_transforms(args.image_size, train=False)

    man = pd.read_parquet(MANIFESTS / "unified_manifest.parquet")
    picks = pick_examples(man, splits, protocol, fold, args.split, args.n_images)
    if picks.empty:
        print(f"BLOCKER: no eligible images for protocol={protocol} fold={fold} "
              f"split={args.split}")
        return 2
    print(f"{len(picks)} examples: " + ", ".join(picks["record_id"].astype(str)))

    from PIL import Image
    n = len(picks)
    dump: dict[str, list] = {"images": [], "cams": [], "p_diseased": [],
                             "labels": [], "record_ids": []}
    fig, axes = plt.subplots(2, n, figsize=(2.9 * n, 6.2))
    axes = np.atleast_2d(axes)
    for j, (_, r) in enumerate(picks.iterrows()):
        p = resolve_image_path(r["relative_image_path"], REPO_ROOT)
        with Image.open(p) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
        x = tf(arr).unsqueeze(0).to(device)
        x.requires_grad_(True)
        out = model(x)
        heat = cam(x, out["health_logit"][0])
        img = denorm(x[0].detach())

        axes[0, j].imshow(img)
        axes[0, j].set_title(f"{r['dataset'][:11]}\n{r['raw_class']}", fontsize=7)
        axes[0, j].axis("off")
        axes[1, j].imshow(img)
        axes[1, j].imshow(heat, cmap="jet", alpha=0.45)
        pd_ = float(torch.sigmoid(out["health_logit"])[0])
        pa = torch.sigmoid(out["attr_logits"])[0].detach().cpu().numpy()
        axes[1, j].set_title(
            f"p(dis)={pd_:.2f}\n" +
            " ".join(f"{a[:2].upper()}={v:.2f}" for a, v in zip(DISEASE_ATTRS, pa)),
            fontsize=6)
        axes[1, j].axis("off")

        dump["images"].append(img.astype(np.float32))
        dump["cams"].append(heat.astype(np.float32))
        dump["p_diseased"].append(pd_)
        dump["labels"].append(int(str(r.get("health_status", "")) == "diseased"))
        dump["record_ids"].append(str(r["record_id"]))

    scope = (f"{protocol} fold {fold} {args.split}" if splits is not None
             else "whole manifest")
    fig.suptitle(f"Grad-CAM on the health head, {scope} (diagnostic only - a bright "
                 f"region means output sensitivity, not a biological mechanism)",
                 fontsize=9)
    fig.tight_layout()
    out_p = Path(args.out) if args.out else FIGURES / "gradcam_disease.png"
    if not out_p.is_absolute():
        out_p = REPO_ROOT / out_p
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_p, dpi=125, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out_p.relative_to(REPO_ROOT).as_posix()}")

    if args.npz:
        npz_p = Path(args.npz)
        if not npz_p.is_absolute():
            npz_p = REPO_ROOT / npz_p
        npz_p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            npz_p,
            images=np.stack(dump["images"]),                      # [N, H, W, 3]
            cams=np.stack(dump["cams"]),                          # [N, H, W]
            p_diseased=np.array(dump["p_diseased"], dtype=np.float32),
            labels=np.array(dump["labels"], dtype=np.int64),
            record_ids=np.array(dump["record_ids"]),              # unicode, no pickle
            run_name=run_name, protocol=str(protocol), split=args.split,
            fold=-1 if fold is None else int(fold))
        print(f"-> {npz_p.relative_to(REPO_ROOT).as_posix()}")

    # multi-view attention weights, if an attention-pooling run exists
    attn_ck = sorted(CHECKPOINTS.glob("E5_biomass_multiview_attention_f*_best.pt"))
    if attn_ck:
        print(f"attention-pooling weights available in the E5 predictions "
              f"(view_weights column of outputs/predictions/"
              f"E5_biomass_multiview_attention_*_biomass_test.parquet)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
