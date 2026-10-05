"""Export the lightweight model to TorchScript and ONNX, and verify parity.

Parity is checked against the PyTorch model on >= 50 real samples drawn from the
manifest (not random noise), because export bugs frequently only appear on the
input distribution the model was trained on.

Usage:
    python scripts/export_model.py --checkpoint outputs/checkpoints/<run>_best.pt \
        [--backbone mobilenetv3_large] [--n-samples 64] [--image-size 320]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.transforms import build_disease_transforms  # noqa: E402
from src.models.multitask import ModelConfig, ShrimpMultiTaskModel, count_parameters  # noqa: E402
from src.utils.paths import CHECKPOINTS, MANIFESTS, METRICS, REPO_ROOT  # noqa: E402


class ExportWrapper(torch.nn.Module):
    """Dict outputs are awkward for TorchScript/ONNX; emit a single concatenated
    tensor [health_logit | attr_logits | biomass] with a documented layout."""

    def __init__(self, model: ShrimpMultiTaskModel):
        super().__init__()
        self.model = model
        self.has_attrs = model.attr_head is not None
        self.has_biomass = model.biomass_head is not None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.model.encoder(x)
        parts = [self.model.health_head(f)]
        if self.has_attrs:
            parts.append(self.model.attr_head(f))
        if self.has_biomass:
            parts.append(self.model.biomass_head(f))
        return torch.cat(parts, dim=1)


def load_sample_batch(n: int, image_size: int) -> torch.Tensor:
    man = pd.read_parquet(MANIFESTS / "unified_manifest.parquet")
    ok = man[(~man["exclude"].astype(bool)) & (man["task"] == "disease")]
    ok = ok.sample(n=min(n, len(ok)), random_state=42)
    tf = build_disease_transforms(image_size, train=False)
    from PIL import Image
    xs = []
    for p in ok["relative_image_path"]:
        with Image.open(REPO_ROOT / p) as im:
            xs.append(tf(np.asarray(im.convert("RGB"), dtype=np.uint8)))
    return torch.stack(xs)


def bench(fn, x: torch.Tensor, warmup: int = 5, iters: int = 30) -> dict:
    for _ in range(warmup):
        fn(x)
    if x.is_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(x)
    if x.is_cuda:
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    return {"latency_ms_per_batch": round(dt * 1000, 3),
            "latency_ms_per_image": round(dt * 1000 / len(x), 3),
            "batch_size": int(len(x)),
            "throughput_img_per_s": round(len(x) / dt, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--backbone", default="mobilenetv3_large")
    ap.add_argument("--n-samples", type=int, default=64)
    ap.add_argument("--image-size", type=int, default=320)
    ap.add_argument("--out-dir", default="outputs/checkpoints/export")
    args = ap.parse_args()

    ck_path = Path(args.checkpoint)
    if not ck_path.is_absolute():
        ck_path = REPO_ROOT / ck_path
    if not ck_path.exists():
        print(f"BLOCKER: checkpoint not found: {ck_path}")
        return 2

    ck = torch.load(ck_path, map_location="cpu", weights_only=True)
    saved = ck.get("config", {})
    mcfg_raw = saved.get("model") if isinstance(saved, dict) else None
    mcfg = ModelConfig(backbone=args.backbone, pretrained=False)
    if isinstance(mcfg_raw, dict):
        for k, v in mcfg_raw.items():
            if hasattr(mcfg, k) and k != "pretrained":
                setattr(mcfg, k, v)
        mcfg.pretrained = False
    model = ShrimpMultiTaskModel(mcfg)
    model.load_state_dict(ck["model_state"], strict=False)
    model.eval()

    wrapper = ExportWrapper(model).eval()
    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = ck_path.stem.replace("_best", "")

    x = load_sample_batch(args.n_samples, args.image_size)
    print(f"verification batch: {tuple(x.shape)} real images from the manifest")

    with torch.no_grad():
        ref = wrapper(x)

    report: dict = {
        "checkpoint": str(ck_path.relative_to(REPO_ROOT).as_posix()),
        "backbone": mcfg.backbone,
        "image_size": args.image_size,
        "n_verification_samples": int(len(x)),
        "output_layout": "[health_logit(1) | attr_logits(3) | biomass(3)]",
        "parameters": count_parameters(model),
    }

    # ---------------- TorchScript ----------------
    ts_path = out_dir / f"{stem}_torchscript.pt"
    try:
        ts = torch.jit.trace(wrapper, x[:2])
        ts = torch.jit.freeze(ts)
        torch.jit.save(ts, ts_path)
        loaded = torch.jit.load(ts_path)
        with torch.no_grad():
            got = loaded(x)
        diff = (got - ref).abs()
        report["torchscript"] = {
            "path": str(ts_path.relative_to(REPO_ROOT).as_posix()),
            "file_size_mb": round(ts_path.stat().st_size / 1e6, 3),
            "max_abs_diff": float(diff.max()),
            "mean_abs_diff": float(diff.mean()),
            "allclose_atol_1e-4": bool(torch.allclose(got, ref, atol=1e-4)),
        }
        print(f"TorchScript OK  max|diff|={float(diff.max()):.3e}")
    except Exception as e:  # noqa: BLE001
        report["torchscript"] = {"error": f"{type(e).__name__}: {e}"}
        print(f"TorchScript FAILED: {e}")

    # ---------------- ONNX ----------------
    onnx_path = out_dir / f"{stem}.onnx"
    try:
        torch.onnx.export(
            wrapper, x[:2], onnx_path,
            input_names=["image"], output_names=["predictions"],
            dynamic_axes={"image": {0: "batch"}, "predictions": {0: "batch"}},
            opset_version=17,
        )
        import onnxruntime as ort
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        got = torch.from_numpy(
            sess.run(None, {"image": x.numpy()})[0])
        diff = (got - ref).abs()
        report["onnx"] = {
            "path": str(onnx_path.relative_to(REPO_ROOT).as_posix()),
            "file_size_mb": round(onnx_path.stat().st_size / 1e6, 3),
            "max_abs_diff": float(diff.max()),
            "mean_abs_diff": float(diff.mean()),
            "allclose_atol_1e-4": bool(torch.allclose(got, ref, atol=1e-4)),
            "opset": 17,
        }
        print(f"ONNX OK  max|diff|={float(diff.max()):.3e}")

        report["latency"] = {
            "pytorch_cpu": bench(lambda t: wrapper(t), x[:16]),
            "onnxruntime_cpu": bench(
                lambda t: sess.run(None, {"image": t.numpy()}), x[:16]),
        }
    except Exception as e:  # noqa: BLE001
        report["onnx"] = {"error": f"{type(e).__name__}: {e}"}
        print(f"ONNX FAILED: {e}")

    # ---------------- GPU latency ----------------
    if torch.cuda.is_available():
        try:
            gm = wrapper.cuda().eval()
            gx = x[:16].cuda()
            with torch.no_grad():
                report.setdefault("latency", {})["pytorch_gpu"] = bench(
                    lambda t: gm(t), gx)
            report["gpu_name"] = torch.cuda.get_device_name(0)
            wrapper.cpu()
        except Exception as e:  # noqa: BLE001
            report.setdefault("latency", {})["pytorch_gpu"] = {"error": str(e)}

    p = METRICS / f"export_report_{stem}.json"
    p.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nexport report -> {p.relative_to(REPO_ROOT).as_posix()}")
    print(json.dumps({k: v for k, v in report.items()
                      if k in ("torchscript", "onnx", "latency")}, indent=2)[:1800])

    ok = (report.get("torchscript", {}).get("allclose_atol_1e-4") and
          report.get("onnx", {}).get("allclose_atol_1e-4"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
