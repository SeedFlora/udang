"""Structured error analysis with image montages (specification section 22).

Produces, per model:
  disease  - false-healthy, false-diseased, BG/WSSV confusion, missed
             co-infections, Yellow Head errors, per-source failures
  biomass  - incomplete bodies, extreme weights, blurred/low-contrast images,
             per-DB failures, and individuals whose views disagree most

Montages show ground truth, prediction, confidence, dataset, fold and error
magnitude. No absolute filesystem path is ever drawn on a figure -- only the
record id and basename.

Usage:
    python scripts/error_analysis.py [--top-k 16]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.labels import DISEASE_ATTRS  # noqa: E402
from src.utils.paths import (  # noqa: E402
    ERROR_ANALYSIS, MANIFESTS, PREDICTIONS, REPO_ROOT, ensure_dirs,
)


def load_image(rel: str, max_side: int = 300) -> np.ndarray | None:
    from PIL import Image
    p = REPO_ROOT / rel
    if not p.exists():
        return None
    try:
        with Image.open(p) as im:
            im = im.convert("RGB")
            im.thumbnail((max_side, max_side))
            return np.asarray(im)
    except Exception:  # noqa: BLE001
        return None


def montage(rows: pd.DataFrame, man: pd.DataFrame, out: Path, title: str,
            caption_fn, cols: int = 4) -> str | None:
    if len(rows) == 0:
        return None
    paths = man.set_index("record_id")["relative_image_path"]
    rows = rows.head(cols * 4)
    n = len(rows)
    r = int(np.ceil(n / cols))
    fig, axes = plt.subplots(r, cols, figsize=(3.5 * cols, 3.7 * r))
    axes = np.atleast_1d(axes).ravel()
    drawn = 0
    for ax, (_, row) in zip(axes, rows.iterrows()):
        rel = paths.get(row["record_id"])
        img = load_image(rel) if isinstance(rel, str) else None
        if img is None:
            ax.text(.5, .5, "image unavailable", ha="center", va="center", fontsize=8)
        else:
            ax.imshow(img)
            drawn += 1
        ax.set_title(caption_fn(row), fontsize=7, loc="left")
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes[n:]:
        ax.set_axis_off()
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=115, bbox_inches="tight")
    plt.close(fig)
    return out.name if drawn else None


# --------------------------------------------------------------------------- #
def disease_errors(pred: pd.DataFrame, man: pd.DataFrame, tag: str,
                   top_k: int) -> dict:
    out: dict = {"model": tag, "n": int(len(pred)), "categories": {}}
    p = pred.copy()
    p["pred_health"] = (p["p_health"] >= .5).astype(int)
    p["health_err"] = (p["pred_health"] != p["y_health"]).astype(int)
    p["health_conf"] = np.where(p["pred_health"] == 1, p["p_health"], 1 - p["p_health"])

    def cap(r):
        return (f"{r['record_id']}  [{str(r['dataset'])[:12]}]\n"
                f"true={'diseased' if r['y_health'] else 'healthy'}  "
                f"pred={'diseased' if r['pred_health'] else 'healthy'}\n"
                f"p(diseased)={r['p_health']:.3f}  class={r['raw_class']}")

    # false healthy: a diseased shrimp called healthy -- the costly error
    fh = p[(p["y_health"] == 1) & (p["pred_health"] == 0)].sort_values("p_health")
    out["categories"]["false_healthy"] = {
        "n": int(len(fh)), "description":
        "diseased shrimp predicted healthy (missed disease; the operationally costly error)",
        "montage": montage(fh, man, ERROR_ANALYSIS / f"{tag}_false_healthy.png",
                           f"{tag}: FALSE HEALTHY (missed disease), most confident first",
                           cap)}
    fd = p[(p["y_health"] == 0) & (p["pred_health"] == 1)].sort_values(
        "p_health", ascending=False)
    out["categories"]["false_diseased"] = {
        "n": int(len(fd)), "description": "healthy shrimp predicted diseased (false alarm)",
        "montage": montage(fd, man, ERROR_ANALYSIS / f"{tag}_false_diseased.png",
                           f"{tag}: FALSE DISEASED (false alarm), most confident first",
                           cap)}

    # BG vs WSSV confusion, restricted to rows where both are supervised
    sup = (p["m_black_gill"] > 0) & (p["m_wssv"] > 0)
    bg_as_wssv = p[sup & (p["y_black_gill"] == 1) & (p["y_wssv"] == 0)
                   & (p["p_wssv"] > p["p_black_gill"])]
    wssv_as_bg = p[sup & (p["y_wssv"] == 1) & (p["y_black_gill"] == 0)
                   & (p["p_black_gill"] > p["p_wssv"])]

    def cap_attr(r):
        return (f"{r['record_id']}  [{str(r['dataset'])[:12]}]\n"
                f"true={r['raw_class']}\n"
                f"p(BG)={r['p_black_gill']:.3f} p(WSSV)={r['p_wssv']:.3f} "
                f"p(YH)={r['p_yellow_head']:.3f}")

    out["categories"]["bg_predicted_as_wssv"] = {
        "n": int(len(bg_as_wssv)),
        "description": "true Black Gill scored higher for WSSV",
        "montage": montage(bg_as_wssv, man, ERROR_ANALYSIS / f"{tag}_bg_as_wssv.png",
                           f"{tag}: Black Gill confused with WSSV", cap_attr)}
    out["categories"]["wssv_predicted_as_bg"] = {
        "n": int(len(wssv_as_bg)),
        "description": "true WSSV scored higher for Black Gill",
        "montage": montage(wssv_as_bg, man, ERROR_ANALYSIS / f"{tag}_wssv_as_bg.png",
                           f"{tag}: WSSV confused with Black Gill", cap_attr)}

    # missed co-infection: both BG and WSSV true, at most one predicted
    co = p[sup & (p["y_black_gill"] == 1) & (p["y_wssv"] == 1)]
    missed_co = co[(co["p_black_gill"] < .5) | (co["p_wssv"] < .5)]
    out["categories"]["missed_co_infection"] = {
        "n": int(len(missed_co)), "n_co_infection_total": int(len(co)),
        "description": "BG+WSSV co-infection where at least one attribute was missed",
        "montage": montage(missed_co, man, ERROR_ANALYSIS / f"{tag}_missed_coinfection.png",
                           f"{tag}: missed co-infections (BG_WSSV)", cap_attr)}

    yh = p[(p["m_yellow_head"] > 0)]
    yh_err = yh[((yh["y_yellow_head"] == 1) & (yh["p_yellow_head"] < .5))
                | ((yh["y_yellow_head"] == 0) & (yh["p_yellow_head"] >= .5))]
    out["categories"]["yellow_head_errors"] = {
        "n": int(len(yh_err)), "n_supervised": int(len(yh)),
        "description": "Yellow Head errors among rows where Yellow Head is supervised "
                       "(only TigerShrimpBD annotates it)",
        "montage": montage(yh_err, man, ERROR_ANALYSIS / f"{tag}_yellow_head_errors.png",
                           f"{tag}: Yellow Head errors", cap_attr)}

    by_src = {}
    for src, g in p.groupby("dataset"):
        by_src[str(src)] = {
            "n": int(len(g)),
            "health_error_rate": round(float(g["health_err"].mean()), 4),
            "n_false_healthy": int(((g["y_health"] == 1) & (g["pred_health"] == 0)).sum()),
            "n_false_diseased": int(((g["y_health"] == 0) & (g["pred_health"] == 1)).sum()),
        }
    out["by_source"] = by_src

    by_cls = {}
    for cls, g in p.groupby("raw_class"):
        by_cls[str(cls)] = {"n": int(len(g)),
                            "health_error_rate": round(float(g["health_err"].mean()), 4)}
    out["by_raw_class"] = by_cls

    # confidently wrong: high confidence, wrong answer
    conf_wrong = p[(p["health_err"] == 1) & (p["health_conf"] > .9)]
    out["categories"]["confidently_wrong"] = {
        "n": int(len(conf_wrong)),
        "description": "health prediction wrong with confidence > 0.9",
        "montage": montage(conf_wrong.sort_values("health_conf", ascending=False),
                           man, ERROR_ANALYSIS / f"{tag}_confidently_wrong.png",
                           f"{tag}: confidently wrong health predictions", cap)}
    return out


def biomass_errors(pred: pd.DataFrame, man: pd.DataFrame, tag: str,
                   top_k: int) -> dict:
    p = pred.copy()
    p["pred_weight_g"] = np.exp(p["pred_log_weight"])
    p["abs_err_g"] = (p["pred_weight_g"] - p["y_weight_g"]).abs()
    p["rel_err"] = p["abs_err_g"] / p["y_weight_g"].replace(0, np.nan)

    ind = p.groupby("individual_id").agg(
        pred_weight_g=("pred_weight_g", "median"),
        pred_spread_g=("pred_weight_g", lambda s: float(s.max() - s.min())),
        y_weight_g=("y_weight_g", "first"), y_length_cm=("y_length_cm", "first"),
        db=("db", "first"), n_views=("pred_weight_g", "count")).reset_index()
    ind["abs_err_g"] = (ind["pred_weight_g"] - ind["y_weight_g"]).abs()
    ind["rel_err"] = ind["abs_err_g"] / ind["y_weight_g"].replace(0, np.nan)

    out: dict = {"model": tag, "n_images": int(len(p)),
                 "n_individuals": int(len(ind)), "categories": {}}

    def cap(r):
        return (f"{r['record_id']}  [{r['db']}]\n"
                f"true={r['y_weight_g']:.2f} g  pred={r['pred_weight_g']:.2f} g\n"
                f"abs err={r['abs_err_g']:.2f} g "
                f"({100 * r['rel_err']:.1f}%)" if np.isfinite(r.get("rel_err", np.nan))
                else f"{r['record_id']}")

    worst = p.sort_values("abs_err_g", ascending=False)
    out["categories"]["worst_image_level"] = {
        "n": int(len(worst)),
        "description": "images with the largest absolute weight error",
        "montage": montage(worst, man, ERROR_ANALYSIS / f"{tag}_worst_images.png",
                           f"{tag}: largest image-level weight errors", cap)}

    if "complete_body" in p.columns and p["complete_body"].notna().any():
        inc = p[p["complete_body"] == 0].sort_values("abs_err_g", ascending=False)
        comp = p[p["complete_body"] == 1]
        out["categories"]["incomplete_body"] = {
            "n": int(len(inc)),
            "description": "individuals flagged as lacking a complete-body image",
            "mae_incomplete_g": round(float(inc["abs_err_g"].mean()), 3) if len(inc) else None,
            "mae_complete_g": round(float(comp["abs_err_g"].mean()), 3) if len(comp) else None,
            "montage": montage(inc, man, ERROR_ANALYSIS / f"{tag}_incomplete_body.png",
                               f"{tag}: incomplete-body cases", cap)}

    # individuals whose per-view predictions disagree most
    dis = ind.sort_values("pred_spread_g", ascending=False).head(top_k)
    out["categories"]["view_disagreement"] = {
        "description": "individuals whose different views produced the most divergent "
                       "weight predictions (a proxy for view-dependent uncertainty)",
        "top": dis[["individual_id", "db", "n_views", "y_weight_g", "pred_weight_g",
                    "pred_spread_g", "abs_err_g"]].round(3).to_dict("records"),
    }

    q = ind["y_weight_g"].quantile([.1, .9])
    small, large = ind[ind["y_weight_g"] <= q.iloc[0]], ind[ind["y_weight_g"] >= q.iloc[1]]
    out["categories"]["size_extremes"] = {
        "description": "error on the smallest and largest 10% of individuals",
        "small_n": int(len(small)), "large_n": int(len(large)),
        "small_mae_g": round(float(small["abs_err_g"].mean()), 3) if len(small) else None,
        "large_mae_g": round(float(large["abs_err_g"].mean()), 3) if len(large) else None,
        "mid_mae_g": round(float(ind["abs_err_g"].median()), 3),
    }

    out["by_db"] = {str(db): {"n": int(len(g)),
                              "mae_g": round(float(g["abs_err_g"].mean()), 3),
                              "medae_g": round(float(g["abs_err_g"].median()), 3),
                              "bias_g": round(float((g["pred_weight_g"] - g["y_weight_g"]).mean()), 3)}
                    for db, g in ind.groupby("db")}
    out["worst_individuals"] = ind.sort_values("abs_err_g", ascending=False).head(top_k)[
        ["individual_id", "db", "n_views", "y_weight_g", "pred_weight_g",
         "abs_err_g", "rel_err"]].round(3).to_dict("records")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=16)
    args = ap.parse_args()
    ensure_dirs()

    man = pd.read_parquet(MANIFESTS / "unified_manifest.parquet")
    report: dict = {"disease": {}, "biomass": {}}

    dis_files: dict[str, list[pd.DataFrame]] = {}
    for f in sorted(PREDICTIONS.glob("*_test.parquet")):
        if "biomass" in f.name:
            continue
        df = pd.read_parquet(f)
        if "p_health" in df.columns:
            dis_files.setdefault(f.name.rsplit("_f", 1)[0], []).append(df)
    for tag, parts in dis_files.items():
        print(f"disease error analysis: {tag}")
        report["disease"][tag] = disease_errors(
            pd.concat(parts, ignore_index=True), man, tag, args.top_k)

    bio_files: dict[str, list[pd.DataFrame]] = {}
    for f in sorted(PREDICTIONS.glob("*_biomass_test.parquet")):
        bio_files.setdefault(f.name.rsplit("_f", 1)[0], []).append(pd.read_parquet(f))
    for tag, parts in bio_files.items():
        print(f"biomass error analysis: {tag}")
        report["biomass"][tag] = biomass_errors(
            pd.concat(parts, ignore_index=True), man, tag, args.top_k)

    if not report["disease"] and not report["biomass"]:
        print("BLOCKER: no prediction files found.")
        return 2

    (ERROR_ANALYSIS / "error_analysis.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")

    rows = []
    for tag, r in report["disease"].items():
        for cat, v in r["categories"].items():
            rows.append({"task": "disease", "model": tag, "category": cat,
                         "n": v.get("n"), "montage": v.get("montage") or ""})
    for tag, r in report["biomass"].items():
        for cat, v in r["categories"].items():
            rows.append({"task": "biomass", "model": tag, "category": cat,
                         "n": v.get("n"), "montage": v.get("montage") or ""})
    pd.DataFrame(rows).to_csv(ERROR_ANALYSIS / "error_summary.csv", index=False)
    print(f"\n-> outputs/error_analysis/ ({len(rows)} categories)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
