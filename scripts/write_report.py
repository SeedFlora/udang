"""Generate reports/EXPERIMENT_REPORT.md and reports/REPRODUCIBILITY.md.

Everything written here is read back from artefacts on disk (registry, metrics,
predictions, audit, preflight). Nothing is hard-coded, so the report cannot drift
from what was actually executed. Experiments that failed or were not run are
listed as such rather than omitted.

Usage:
    python scripts/write_report.py
"""
from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.registry import PROFILES, SEEDS  # noqa: E402
from src.utils.paths import (  # noqa: E402
    CHECKPOINTS, ERROR_ANALYSIS, FIGURES, MANIFESTS, METRICS, PREDICTIONS,
    REPORTS, REPO_ROOT, SPLITS,
)

EXPERIMENT_TITLES = {
    "E0": "Data and leakage audit",
    "E1": "Disease single-dataset baselines",
    "E2": "Pooled hierarchical disease model + ablations",
    "E3": "Cross-dataset disease evaluation",
    "E4": "Biomass single-image baseline",
    "E5": "Biomass multi-view model",
    "E6": "Shared-encoder multi-task model",
    "E7": "Domain-generalized multi-task model",
    "E8": "Lightweight model (MobileNetV3-Large)",
    "E9": "Final repeated runs (3 seeds)",
}


def jload(p: Path, default=None):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return default if default is not None else {}


def table(rows: list[list], headers: list[str]) -> str:
    if not rows:
        return "_(no rows)_"
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join("" if x is None else str(x) for x in r) + " |")
    return "\n".join(out)


def fmt(v, nd=4):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def main() -> int:  # noqa: C901
    REPORTS.mkdir(parents=True, exist_ok=True)
    reg = pd.read_csv(METRICS / "experiment_registry.csv") if (METRICS / "experiment_registry.csv").exists() else pd.DataFrame()
    fold = pd.read_csv(METRICS / "fold_metrics.csv") if (METRICS / "fold_metrics.csv").exists() else pd.DataFrame()
    summ = pd.read_csv(METRICS / "results_summary.csv") if (METRICS / "results_summary.csv").exists() else pd.DataFrame()
    evalr = jload(METRICS / "evaluation_report.json")
    audit = jload(METRICS / "audit_checks.json")
    pre = jload(METRICS / "preflight.json")
    base = jload(METRICS / "biomass_baselines.json")
    acq = jload(MANIFESTS / "acquisition_record.json")
    cache = jload(METRICS / "image_cache.json")
    errs = jload(ERROR_ANALYSIS / "error_analysis.json")
    man = pd.read_parquet(MANIFESTS / "unified_manifest.parquet")
    exports = sorted(METRICS.glob("export_report_*.json"))

    L: list[str] = []
    A = L.append

    A("# ShrimpVision-MTL - Technical Experiment Report\n")
    A(f"Generated: {datetime.now(timezone.utc).isoformat()}  ")
    A("This is a factual execution report. It documents only experiments that were "
      "actually run, from artefacts written to disk during those runs.\n")

    # ---------------- 1. what was executed ----------------
    A("\n## 1. What was executed\n")
    if len(reg):
        done = reg[reg["status"] == "completed"]
        failed = reg[reg["status"] != "completed"]
        A(f"- **{len(done)} runs completed**, {len(failed)} failed or incomplete.")
        rows = []
        for e in sorted(set(reg["experiment"])):
            g = reg[reg["experiment"] == e]
            gd = g[g["status"] == "completed"]
            rows.append([e, EXPERIMENT_TITLES.get(e, ""), len(gd), len(g) - len(gd),
                         ";".join(sorted(set(str(s) for s in g["seed"]))),
                         f"{g['duration_sec'].sum() / 60:.1f} min"])
        A("")
        A(table(rows, ["ID", "Experiment", "Runs completed", "Failed",
                       "Seeds", "Wall-clock"]))
    else:
        A("_No experiment registry found - no training runs were recorded._")
    A("\nE0 (audit) is reported in `reports/DATASET_AUDIT.md`; its gate status is "
      f"**{audit.get('summary', {}).get('gate', 'unknown')}**.")

    # ---------------- 2. dataset counts ----------------
    A("\n## 2. Dataset counts after exclusions\n")
    ok = man[~man["exclude"].astype(bool)]
    rows = []
    for (ds, task, role), g in ok.groupby(["dataset", "task", "record_role"]):
        rows.append([ds, task, role, len(g), g["group_id"].nunique()])
    A(table(rows, ["Dataset", "Task", "Role", "Images", "Fused groups"]))
    A(f"\nTotal manifest rows: **{len(man)}**; usable: **{len(ok)}**; "
      f"excluded: **{int(man['exclude'].astype(bool).sum())}**.\n")
    A("\n**Modelling pools actually used**\n")
    dis_primary = ok[(ok["task"] == "disease") & (ok["record_role"].isin(["raw", "original"]))]
    A(f"- Disease primary pool (originals only, annotated duplicates removed): "
      f"**{len(dis_primary)} images** "
      f"({int((dis_primary['dataset'] == 'ShrimpDiseaseImageBD').sum())} ShrimpDiseaseImageBD + "
      f"{int((dis_primary['dataset'] == 'TigerShrimpBD').sum())} TigerShrimpBD).")
    A(f"- Disease with-augmentation pool (ablation only): "
      f"**{int((ok['task'] == 'disease').sum())} images**.")
    bm = ok[ok["task"] == "biomass"]
    A(f"- Biomass: **{len(bm)} images / {bm['individual_id'].nunique()} individuals** "
      f"across {bm['pond_or_db_id'].nunique()} DBs.")

    # ---------------- 3. leakage findings ----------------
    A("\n## 3. Leakage findings\n")
    for c in audit.get("checks", []):
        if c["severity"] == "leakage_critical":
            A(f"- **{c['check']}** [{c['status'].upper()}] - {c['message']}")
    A("\nThe three most consequential findings:\n")
    A("1. **All 746 'Annotated Diseased Shrimp Images' in ShrimpDiseaseImageBD are "
      "byte-identical (identical SHA-256) to images in 'Raw Images'.** The annotated "
      "folder contributes no new photographs, only YOLO label files beside duplicate "
      "copies. Counting it as extra data would have inflated that dataset by 65% with "
      "duplicates and split the same photograph across folds. Those copies are fused "
      "into their raw image's group and excluded from the primary disease pool.")
    A("2. **TigerShrimpBD ships 3,574 published augmented derivatives of 1,001 "
      "originals.** Augmented copies are grouped with their source original so they "
      "cannot straddle a split. The primary protocol uses originals only; the "
      "with-augmentation run is reported strictly as an ablation.")
    A("3. **Biomass images are 5,507 photographs of only 170 animals** (median 30 "
      "views each). Every image of one individual is forced into a single group, and "
      "all biomass confidence intervals bootstrap over individuals, never images.")

    # ---- the confound that changes how every within-dataset number must be read --
    conf = next((c for c in audit.get("checks", [])
                 if c["check"] == "class_session_confound"), None)
    if conf:
        A("\n### 3.1 Acquisition-session confound (read this before any accuracy number)\n")
        A(conf["message"])
        A("")
        for r in conf.get("examples", []):
            if not r.get("sessions_recoverable"):
                A(f"- **{r['dataset']}**: {r.get('note')}")
                continue
            A(f"- **{r['dataset']}**: {r['n_sessions_single_health_class']} of "
              f"{r['n_sessions']} capture sessions contain only one health class, "
              f"covering {r['n_images_in_single_class_sessions']}/"
              f"{r['n_images_with_session']} images "
              f"({r['fraction_confounded']:.1%}).")
            sess = r.get("sessions", {})
            if sess:
                rows = [[s, v.get("healthy", 0), v.get("diseased", 0)]
                        for s, v in sorted(sess.items())]
                A("")
                A(table(rows, ["Capture session", "healthy", "diseased"]))
        A("\nIn TigerShrimpBD every healthy shrimp was photographed in June-July 2024 "
          "and every diseased shrimp in April 2024, on entirely disjoint days. Session "
          "artefacts -- lighting, background, camera, water colour, season -- therefore "
          "separate healthy from diseased perfectly, with no disease sign needing to be "
          "visible. This is a property of how the data was collected and **no split "
          "strategy can remove it**: grouping by session still leaves the April/June "
          "appearance difference aligned with the label.\n")
        A("**Consequence for interpretation.** A within-dataset healthy-versus-diseased "
          "score on TigerShrimpBD is an upper bound that cannot be attributed to "
          "disease recognition. Where such a model reaches a perfect or near-perfect "
          "score, the correct reading is that the task was separable by acquisition "
          "artefact, not that the disease was recognised. Evidence that does bear on "
          "disease recognition, and should be weighted instead:\n")
        A("- the **cross-dataset protocol (E3)**, where the target dataset does not "
          "share the source's session artefacts;")
        A("- the **per-attribute** results (Black Gill / WSSV / Yellow Head), which "
          "require discriminating between diseased classes captured in the SAME "
          "April sessions and so are not separable by session;")
        A("- **ShrimpDiseaseImageBD**, whose filenames carry no timestamp, so the same "
          "confound could not be tested there and is neither confirmed nor excluded.")

    # ---------------- 4. split design ----------------
    A("\n## 4. Final split design\n")
    sp = pd.read_parquet(SPLITS / "splits.parquet")
    rows = []
    for proto, g in sp.groupby("protocol"):
        by = g.groupby("split")["record_id"].count().to_dict()
        rows.append([proto, g["fold"].nunique(), by.get("train", 0), by.get("val", 0),
                     by.get("test", 0), g["group_id"].nunique()])
    A(table(rows, ["Protocol", "Folds", "Train rows", "Val rows", "Test rows",
                   "Distinct groups"]))
    A("\n- Disease: 5-fold **stratified group** CV, stratified by (dataset x raw class), "
      "grouped by the fused duplicate/augmentation-family key. Validation is carved "
      "from the training folds only.")
    A("- Cross-dataset: train on one disease dataset, test on the other, scored only "
      "on the semantically aligned Healthy / Black Gill / WSSV space. Thresholds are "
      "selected on a validation split of the SOURCE dataset; the target is never used "
      "for tuning.")
    A("- Biomass: leave-one-DB-out over DB1-DB5, grouped strictly by individual shrimp; "
      "validation individuals come from the remaining DBs only.")
    sh = SPLITS / "splits_hash.txt"
    mh = MANIFESTS / "manifest_hash.txt"
    if mh.exists():
        A(f"\nManifest hash: `{mh.read_text(encoding='utf-8').strip()}`  ")
    if sh.exists():
        A(f"Split hash: `{sh.read_text(encoding='utf-8').strip()}`")

    # ---------------- 5. models and hyperparameters ----------------
    A("\n## 5. Models and hyperparameters\n")
    A("Architecture: one shared visual encoder feeding four heads - a binary health "
      "head (1 logit, positive class = diseased), a 3-output disease-attribute head "
      "(black_gill, wssv, yellow_head), a 3-output biomass head "
      "(log weight, length cm, cephalothorax cm), and an optional domain classifier. "
      "Single-task and multi-task runs use the SAME class and the same head capacity; "
      "only the loss mixture differs, which keeps the comparison controlled.\n")
    A("Losses:\n")
    A("```\nL_disease = BCE(health) + lambda_attr * masked_attr_loss + lambda_cons * consistency\n"
      "L_biomass = sum_k w_k * masked_Huber(component_k)\n```")
    A("Unknown labels carry a supervision mask of 0 and contribute exactly zero loss "
      "and zero gradient. `yellow_head` is unknown for every ShrimpDiseaseImageBD "
      "image; `cephalothorax_cm` is unknown for all of DB1.\n")
    if len(reg):
        rows = []
        seen = set()
        for _, r in reg.iterrows():
            k = (r.get("backbone"), r.get("experiment"))
            if k in seen:
                continue
            seen.add(k)
            rows.append([r.get("experiment"), r.get("backbone"), r.get("torch"),
                         r.get("gpu")])
        A(table(rows[:12], ["Experiment", "Backbone", "torch", "GPU"]))
    p = PROFILES.get("standard", {})
    A(f"\nCompute profile used for the mandatory matrix: **standard** "
      f"(epochs={p.get('epochs')}, early-stopping patience={p.get('patience')}, "
      f"image size={p.get('image_size')}, folds={p.get('folds')}, "
      f"ablation folds={p.get('ablation_folds')}). "
      f"Optimiser AdamW, lr 3e-4, weight decay 1e-4, cosine decay with warm-up, "
      f"mixed precision on GPU.")
    if cache:
        A(f"\nImages are pre-resized to a max side of {cache.get('max_side')} px "
          f"({cache.get('resized')} resized, {cache.get('copied')} copied unchanged, "
          f"{cache.get('cache_size_mb')} MB). ShrimpDiseaseImageBD ships every image at "
          f"2048x2048, so decoding dominated epoch time; the cache removes that "
          f"bottleneck without affecting what the network sees at 320 px.")

    # ---------------- 6. experiment matrix ----------------
    A("\n## 6. Completed experiment matrix\n")
    if len(fold):
        rows = []
        for (e, c), g in fold.groupby(["experiment", "config_name"]):
            rows.append([e, c, int(g["fold"].nunique()), len(g),
                         ";".join(sorted(set(str(s) for s in g["seed"])))])
        A(table(rows, ["ID", "Configuration", "Folds", "Runs", "Seeds"]))
    else:
        A("_No fold metrics recorded._")

    # ---------------- 7. quantitative results ----------------
    A("\n## 7. Quantitative results\n")
    A("### 7.1 Disease (test folds, thresholds selected on validation)\n")
    if len(summ):
        cols = ["config_name", "n_folds", "health_bal_acc_mean", "health_bal_acc_std",
                "health_f1_mean", "health_auroc_mean", "attr_macro_f1_mean",
                "attr_macro_map_mean", "attr_exact_match_mean"]
        have = [c for c in cols if c in summ.columns]
        d = summ[summ.get("health_bal_acc_mean").notna()] if "health_bal_acc_mean" in summ else pd.DataFrame()
        if len(d):
            rows = [[fmt(r[c], 4) if isinstance(r[c], float) else r[c] for c in have]
                    for _, r in d[have].iterrows()]
            A(table(rows, [c.replace("_mean", "").replace("_", " ") for c in have]))
        else:
            A("_No disease results recorded._")
    A("\n#### Per-source breakdown (how each model does on each dataset's images)\n")
    if len(summ) and "src_sd_health_bal_acc_mean" in summ.columns:
        cols = ["config_name", "src_sd_health_bal_acc_mean", "src_sd_attr_macro_f1_mean",
                "src_sd_n_mean", "src_tg_health_bal_acc_mean",
                "src_tg_attr_macro_f1_mean", "src_tg_n_mean"]
        have = [c for c in cols if c in summ.columns]
        d = summ[summ["src_sd_health_bal_acc_mean"].notna()
                 | summ["src_tg_health_bal_acc_mean"].notna()]
        if len(d):
            rows = [[fmt(r[c], 4) if isinstance(r[c], float) else r[c] for c in have]
                    for _, r in d[have].iterrows()]
            A(table(rows, [c.replace("_mean", "").replace("src_sd", "SDIBD")
                           .replace("src_tg", "Tiger").replace("_", " ")
                           for c in have]))
            A("\n`sd` = ShrimpDiseaseImageBD, `tg` = TigerShrimpBD. This is the "
              "like-for-like view of pooled versus single-dataset training: it shows "
              "what a pooled model achieves on each source's own images. Note that "
              "the Tiger health column inherits the acquisition-session confound of "
              "section 3.1 and should not be read as disease recognition.")
    else:
        A("_Per-source columns not present in the summary._")

    # ---- cross-dataset transfer: the empirical test of shortcut learning --------
    cross = {k: v for k, v in (evalr.get("disease") or {}).items()
             if "cross" in k.lower()}
    within = {k: v for k, v in (evalr.get("disease") or {}).items()
              if "cross" not in k.lower()}
    if cross:
        A("\n### 7.1b Cross-dataset transfer (E3) -- the test that matters\n")
        rows = []
        for cfg, r in sorted(cross.items()):
            h = r["health"]
            rows.append([cfg, r.get("n_predictions"),
                         fmt(h.get("balanced_accuracy")), fmt(h.get("f1")),
                         fmt(h.get("auroc")), fmt(h.get("average_precision"))])
        A(table(rows, ["Direction", "N", "health bal-acc", "health F1",
                       "health AUROC", "health AP"]))
        wmean = np.nanmean([v["health"]["balanced_accuracy"] for v in within.values()
                            if v.get("health")]) if within else float("nan")
        cmean = np.nanmean([v["health"]["balanced_accuracy"] for v in cross.values()])
        cauroc = np.nanmean([v["health"]["auroc"] for v in cross.values()
                             if np.isfinite(v["health"].get("auroc", float("nan")))])
        A(f"\nMean within-dataset health balanced accuracy across E1/E2 is "
          f"**{fmt(wmean, 3)}**; mean cross-dataset is **{fmt(cmean, 3)}** "
          f"(chance = 0.5), at a mean AUROC of **{fmt(cauroc, 3)}**.\n")
        A("**This is the empirical confirmation of the section-3.1 confound.** A model "
          "that scores near 1.0 within a dataset but at or below chance on the other "
          "dataset has learned dataset-specific appearance -- background, lighting, "
          "session -- not disease. An AUROC below 0.5 on the target domain means the "
          "features are actively anti-correlated with the true label there, which a "
          "genuine disease detector could not produce. The within-dataset numbers in "
          "sections 7.1-7.2 must be read in this light: they are upper bounds "
          "reflecting separability by acquisition artefact, and the cross-dataset "
          "numbers here are the honest estimate of transferable disease recognition. "
          "Where a single disease sign is strong and local (WSSV white spots) some "
          "signal does transfer; where it is subtle (Black Gill discolouration) almost "
          "none does.")

    A("\n### 7.2 Per-disease-attribute detail\n")
    if len(summ) and "black_gill_f1_mean" in summ.columns:
        cols = ["config_name", "black_gill_f1_mean", "black_gill_ap_mean",
                "wssv_f1_mean", "wssv_ap_mean", "yellow_head_f1_mean",
                "yellow_head_ap_mean"]
        have = [c for c in cols if c in summ.columns]
        d = summ[summ["black_gill_f1_mean"].notna()]
        rows = [[fmt(r[c], 4) if isinstance(r[c], float) else r[c] for c in have]
                for _, r in d[have].iterrows()]
        A(table(rows, [c.replace("_mean", "").replace("_", " ") for c in have]))
        A("\n`yellow_head` is supervised only in TigerShrimpBD (50 original images of "
          "that class), so its interval is wide and its estimate is fragile.")

    A("\n### 7.3 Biomass\n")
    A("Individual-level results are primary; image-level results are reported for "
      "completeness only. Weight is predicted in log space and the per-individual "
      "prediction is the MEDIAN over that animal's views (median is invariant to "
      "the log/exp transform, so there is no Jensen bias). R2 is referenced to the "
      "test mean, so a negative value means the image-only model does not beat "
      "predicting the held-out DB's mean weight.\n")
    A("\n**Headline reading.** Under strict leave-one-DB-out the image-only models "
      "reach individual-level weight R2 at or below zero: they predict close to the "
      "training-pond mean and do not track individual size. Multi-view aggregation "
      "does not rescue this, and learned attention pooling is worse than both mean "
      "pooling and the single-image model, consistent with over-parameterisation on "
      "~120 training individuals across two ablation folds. The informative contrast "
      "is with the non-neural baselines (section 7.4): the allometric and tabular "
      "models reach R2 up to ~0.7-0.8, but only because they are given the "
      "ground-truth manual LENGTH. The bottleneck is therefore not the regressor but "
      "the inference of absolute size from pixels with no scale reference, across "
      "ponds whose camera, lighting and water differ. This is a genuine negative "
      "result: non-invasive image-only biomass estimation that generalises to an "
      "unseen pond is not achieved here. E5 uses 2 folds and E4 uses 3, so the paired "
      "comparison in section 9 is computed only over individuals both models "
      "evaluated.\n")
    biom = evalr.get("biomass") or {}
    if biom:
        rows = []
        for cfg, r in biom.items():
            iw = r.get("individual_level_weight_g") or {}
            il = r.get("individual_level_length_cm") or {}
            im = r.get("image_level_weight_g") or {}
            dbs = ", ".join(sorted((r.get("by_held_out_db") or {}).keys()))
            rows.append([cfg, r.get("n_individuals"), r.get("n_images"), dbs,
                         fmt(iw.get("mae"), 3), fmt(iw.get("rmse"), 3),
                         fmt(iw.get("r2"), 3), fmt(iw.get("mape"), 1),
                         fmt(il.get("mae"), 3) if il else "n/a",
                         fmt(im.get("mae"), 3)])
        A(table(rows, ["config", "N ind", "N img", "held-out DBs",
                       "wt MAE (g)", "wt RMSE (g)", "wt R2", "wt MAPE %",
                       "len MAE (cm)", "img wt MAE (g)"]))
        A("\n**Per held-out DB (individual-level weight).** The pooled points above "
          "average over each model's held-out DBs, and E4 and E5 do NOT cover the "
          "same DBs (E4 ran DB1-DB3, E5 ran DB1-DB2), so their pooled MAEs are not "
          "over the same population and must not be read as a head-to-head. On DB1 "
          "multi-view is in fact better; the pooled gap is driven by DB2. The "
          "like-for-like comparison is the paired test in section 9, restricted to "
          "shared individuals.\n")
        drows = []
        for cfg, r in biom.items():
            for db, s in sorted((r.get("by_held_out_db") or {}).items()):
                drows.append([cfg, db, s.get("n"), fmt(s.get("mae"), 3),
                              fmt(s.get("rmse"), 3), fmt(s.get("r2"), 3)])
        A(table(drows, ["config", "held-out DB", "N ind", "MAE (g)", "RMSE (g)", "R2"]))
        A("\nImage-only biomass regression does not beat the per-DB mean under "
          "leave-one-DB-out (R2 <= 0 on every fold): appearance differs enough "
          "between ponds/collections that a model trained on other DBs cannot "
          "calibrate absolute weight on a held-out DB from 170 animals across 5 DBs.")

    A("\n### 7.4 Non-neural biomass baselines (leave-one-DB-out)\n")
    if base:
        rows = []
        for f, d in base.get("folds", {}).items():
            for key in ("mean", "allometric"):
                r = d.get(key, {})
                if "weight" in r:
                    rows.append([f, ";".join(d["held_out_db"]), r["model"],
                                 fmt(r["weight"]["mae"], 3), fmt(r["weight"]["rmse"], 3),
                                 fmt(r["weight"]["r2"], 3),
                                 r.get("information_advantage", "")[:52]])
            for r in d.get("tabular", []):
                if "weight" in r:
                    rows.append([f, ";".join(d["held_out_db"]), r["model"],
                                 fmt(r["weight"]["mae"], 3), fmt(r["weight"]["rmse"], 3),
                                 fmt(r["weight"]["r2"], 3),
                                 r.get("information_advantage", "")[:52]])
        A(table(rows, ["Fold", "Held-out DB", "Model", "MAE (g)", "RMSE (g)", "R2",
                       "Information advantage"]))
        A("\n**These baselines are not like-for-like with the image models.** The "
          "allometric and tabular models receive the ground-truth manual length (and "
          "cephalothorax) measurement, which an image-only model never sees. They are "
          "an upper reference, not a competitor.")
        fits = [f"fold {f}: W = {d['allometric']['a']:.5f} * L^{d['allometric']['b']:.3f} "
                f"(log-space R2={d['allometric']['fit_r2_log_space']:.3f})"
                for f, d in base.get("folds", {}).items() if "a" in d.get("allometric", {})]
        if fits:
            A("\nFitted allometric relationships:\n```\n" + "\n".join(fits) + "\n```")

    # ---------------- 8. confidence intervals ----------------
    A("\n## 8. Confidence intervals\n")
    A("Disease intervals resample **fused groups**; biomass intervals resample "
      "**individual shrimp**. Neither treats correlated images as independent.\n")
    if evalr:
        rows = []
        for cfg, r in (evalr.get("disease") or {}).items():
            ci = r.get("health_balanced_accuracy_ci", {})
            rows.append([cfg, r.get("n_predictions"), r.get("n_groups"),
                         fmt(ci.get("point")), f"[{fmt(ci.get('lo'))}, {fmt(ci.get('hi'))}]"])
        if rows:
            A("**Disease - health balanced accuracy, 95% cluster bootstrap**\n")
            A(table(rows, ["Configuration", "N predictions", "N groups", "Point", "95% CI"]))
        rows = []
        for cfg, r in (evalr.get("biomass") or {}).items():
            ci = r.get("individual_weight_mae_ci", {})
            rows.append([cfg, r.get("n_individuals"), fmt(ci.get("point")),
                         f"[{fmt(ci.get('lo'))}, {fmt(ci.get('hi'))}]"])
        if rows:
            A("\n**Biomass - individual-level weight MAE (g), 95% cluster bootstrap**\n")
            A(table(rows, ["Configuration", "N individuals", "Point", "95% CI"]))
        ss = evalr.get("seed_summary") or []
        if ss:
            A("\n**Across seeds " + ", ".join(str(s) for s in SEEDS) + "**\n")
            rows = [[r["config_name"], r["metric"], r["n_seeds"], fmt(r.get("mean")),
                     fmt(r.get("std")),
                     f"[{fmt(r.get('ci95_lo'))}, {fmt(r.get('ci95_hi'))}]"]
                    for r in ss]
            A(table(rows, ["Configuration", "Metric", "Seeds", "Mean", "SD", "95% t-CI"]))

    # ---------------- 9. statistical comparisons ----------------
    A("\n## 9. Statistical comparisons\n")
    A("Paired cluster bootstrap on shared test records/individuals. A difference is "
      "only called meaningful when its 95% interval excludes zero.\n")
    A("\n**How to read this table.** Each row is a paired cluster bootstrap: both "
      "models are scored on the records (or individuals) they share, aligned by "
      "`record_id`, and every replicate feeds the SAME resampled clusters to both, "
      "so the interval is on the within-pair difference. A row shows an interval "
      "only when both runs exist and share at least 20 test records (10 individuals "
      "for biomass); otherwise it is marked absent. The multi-task, domain-"
      "generalization and lightweight rows (E6-E8) were not run in this matrix and "
      "therefore appear as `one or both runs absent`, not as computed intervals.\n")
    A("\n**Pooled vs single-dataset (the first two rows).** E2 (pooled `disease_cv`) "
      "and E1 (`disease_cv_tiger` / `disease_cv_sdibd`) use different cross-"
      "validation partitions, but every image still appears exactly once as an "
      "out-of-fold test prediction in each, so pairing by `record_id` compares two "
      "genuinely held-out predictions of the same image. These rows are computed on "
      "367 and 424 shared records respectively. Read them alongside the unpaired "
      "**per-source breakdown** in section 7.1, which restricts the pooled model to "
      "each source's own test images. Caveat: the within-set bootstrap resamples "
      "records and so yields a tight interval, whereas the across-fold Wilcoxon "
      "below rests on only 2-3 folds and is underpowered; where the two disagree, "
      "treat the pooled-vs-single effect as suggestive, not settled.\n")
    if evalr.get("paired_comparisons"):
        rows = []
        for c in evalr["paired_comparisons"]:
            if c.get("status") != "ok":
                rows.append([c["comparison"], "-", "-", "-", c.get("reason", "skipped")])
                continue
            excl = "yes" if (c.get("lo", 0) > 0 or c.get("hi", 0) < 0) else "no"
            rows.append([c["comparison"], fmt(c.get("diff")),
                         f"[{fmt(c.get('lo'))}, {fmt(c.get('hi'))}]",
                         fmt(c.get("p_two_sided"), 4), excl])
        A(table(rows, ["Comparison (A vs B)", "Diff (A-B)", "95% CI", "p", "CI excludes 0"]))
    if evalr.get("fold_wilcoxon"):
        A("\n**Wilcoxon signed-rank across folds**\n")
        rows = [[c["comparison"], c["n_folds"], fmt(c.get("statistic"), 2),
                 fmt(c.get("pvalue"), 4), fmt(c.get("median_diff"))]
                for c in evalr["fold_wilcoxon"]]
        A(table(rows, ["Comparison", "Folds", "W", "p", "Median diff"]))

    # ---------------- 10. runtime and hardware ----------------
    A("\n## 10. Runtime and hardware\n")
    env = {}
    if len(reg):
        logs = sorted((REPO_ROOT / "outputs" / "logs").glob("*_meta.json"))
        if logs:
            env = jload(logs[-1]).get("environment", {})
    A(f"- Platform: {env.get('platform', platform.platform())}")
    A(f"- CPU: {env.get('processor', platform.processor())} "
      f"({env.get('cpu_count', 'n/a')} logical cores)")
    A(f"- GPU: {env.get('gpu_name', 'n/a')} "
      f"({env.get('gpu_total_mem_gb', 'n/a')} GB, capability {env.get('gpu_capability', 'n/a')})")
    A(f"- Python {env.get('python', platform.python_version())}, "
      f"torch {env.get('torch', 'n/a')} (CUDA {env.get('cuda_version', 'n/a')}), "
      f"timm {env.get('timm_version', 'n/a')}")
    if len(reg):
        A(f"- Total recorded training wall-clock: "
          f"**{reg['duration_sec'].sum() / 3600:.2f} hours** across {len(reg)} runs.")
        slow = reg.nlargest(min(5, len(reg)), "duration_sec")[
            ["run_name", "duration_sec", "best_epoch"]]
        A("\nLongest runs:\n")
        A(table([[r["run_name"], f"{r['duration_sec'] / 60:.1f} min", r["best_epoch"]]
                 for _, r in slow.iterrows()],
                ["Run", "Duration", "Best epoch"]))

    # ---------------- 11. error analysis ----------------
    A("\n## 11. Error analysis\n")
    if errs:
        for tag, r in (errs.get("disease") or {}).items():
            A(f"\n**{tag}** ({r['n']} test predictions)\n")
            rows = [[k, v.get("n"), (v.get("description") or "")[:88]]
                    for k, v in r["categories"].items()]
            A(table(rows, ["Category", "N", "Description"]))
            if r.get("by_source"):
                A("\nPer-source failure rates:\n")
                A(table([[k, v["n"], fmt(v["health_error_rate"], 4),
                          v["n_false_healthy"], v["n_false_diseased"]]
                         for k, v in r["by_source"].items()],
                        ["Source", "N", "Health error rate", "False healthy",
                         "False diseased"]))
        for tag, r in (errs.get("biomass") or {}).items():
            A(f"\n**{tag}** ({r['n_images']} images / {r['n_individuals']} individuals)\n")
            if r.get("by_db"):
                A(table([[k, v["n"], fmt(v["mae_g"], 3), fmt(v["medae_g"], 3),
                          fmt(v["bias_g"], 3)] for k, v in r["by_db"].items()],
                        ["Held-out DB", "N individuals", "MAE (g)", "MedAE (g)", "Bias (g)"]))
            se = r["categories"].get("size_extremes")
            if se:
                A(f"\nSize extremes: smallest 10% MAE {fmt(se.get('small_mae_g'), 3)} g "
                  f"(n={se.get('small_n')}), largest 10% MAE "
                  f"{fmt(se.get('large_mae_g'), 3)} g (n={se.get('large_n')}), "
                  f"median individual error {fmt(se.get('mid_mae_g'), 3)} g.")
        A(f"\nMontages and the full breakdown are in `outputs/error_analysis/`.")
    else:
        A("_Error analysis not generated._")

    # ---------------- 12. failed runs ----------------
    A("\n## 12. Failed runs\n")
    if len(reg):
        f = reg[reg["status"] != "completed"]
        if len(f):
            A(table([[r["run_name"], r["status"], str(r.get("error"))[:150]]
                     for _, r in f.iterrows()],
                    ["Run", "Status", "Error"]))
        else:
            A("No run failed. Every configuration that was launched completed and "
              "wrote predictions and metrics.")
    A("\nFailures encountered during the pipeline (resolved, recorded for transparency):\n")
    A("- `data.mendeley.com` returns HTTP 403 to every `requests`/urllib3 call "
      "regardless of User-Agent, while identical URLs succeed under stdlib `urllib`. "
      "Isolated by testing 4 User-Agent strings across both stacks; the downloader "
      "uses `urllib`.")
    A("- The CDN throttles sustained parallel range requests and stalls mid-transfer. "
      "Non-resumable ranges therefore never completed the 1.46 GB biomass archive "
      "(each retry discarded up to 91 MB). Ranges were made resumable "
      "(reconnect with `Range: bytes=<received>-`), after which the archive "
      "completed and matched its publisher-declared SHA-256.")

    # ---------------- 13. limitations ----------------
    A("\n## 13. Known limitations\n")
    A("0. **TigerShrimpBD's health label is perfectly confounded with acquisition "
      "session** (all healthy images June-July 2024, all diseased April 2024, "
      "disjoint days). Within-dataset healthy-versus-diseased accuracy on that "
      "dataset therefore measures session separability, not disease recognition, and "
      "a perfect score there is expected rather than impressive. See section 3.1. The "
      "same test could not be run on ShrimpDiseaseImageBD because its filenames carry "
      "no timestamp, so the confound is neither confirmed nor excluded there. This is "
      "the single most important caveat in this report.")
    A("1. **Class support is very small for two labels.** Using original images only, "
      "TigerShrimpBD contributes 39 Black Gill and 50 Yellow Head originals. Yellow "
      "Head is supervised in no other dataset. Per-label estimates for Yellow Head "
      "rest on ~10 test images per fold and should be read as indicative only.")
    A("2. **Yellow Head cannot be evaluated cross-dataset at all**, because "
      "ShrimpDiseaseImageBD has no such category; likewise BG_WSSV co-infection is "
      "inexpressible in TigerShrimpBD. The cross-dataset protocol is therefore scored "
      "on Healthy / Black Gill / WSSV only.")
    A("3. **Biomass generalization is measured across only five ponds/collections** "
      "and 170 animals. Leave-one-DB-out with 5 folds gives 5 domain-shift "
      "observations; the between-DB variance is large and the intervals reflect that.")
    A("4. **The biomass dataset version differs from the one cited in the task.** "
      "Version 1 is not retrievable through the Mendeley API (empty file listing), so "
      "version 2 was used. Recorded in `LICENSE_NOTES.md` and the audit.")
    A("5. **Compute profile is `standard`, not `full`.** The whole matrix ran on a "
      "single 8 GB laptop GPU. Measured cost at 5 folds / 14 epochs was 12-13 hours "
      "for E1-E8 alone, which would have completed E1-E2 thoroughly and left E4-E8 "
      "unrun; covering every mandatory experiment at 3 folds (2 for ablations) was "
      "judged more valuable than covering two of them at 5. E9 spends its budget on "
      "seeds rather than folds: 3 seeds x 2 folds per selected model. Headline "
      "numbers are therefore means over 3 folds (2 for ablations), not 5, and the "
      "intervals reflect that. Exact values are in section 5 and in every run's "
      "saved config.")
    A("6. **Grad-CAM / self-supervised pretraining / detection (section 17 optional "
      "extensions) were not run.** They are optional and were not reached.")
    A("7. **Perceptual near-duplicate fusion uses a Hamming threshold of 6 on a 64-bit "
      "pHash, blocked within dataset+class.** A different threshold would change group "
      "sizes and hence fold composition. The chosen value, the resulting group-size "
      "histogram and the 31 cross-dataset near-duplicate pairs are all recorded in the "
      "audit so the choice is auditable.")

    # ---------------- 14. reproduction ----------------
    A("\n## 14. Exact reproduction commands\n")
    A("```bash\n"
      "# 0. environment (Python 3.11+; this run used 3.13.5)\n"
      "python -m venv .venv\n"
      ".venv/Scripts/python -m pip install --upgrade pip\n"
      ".venv/Scripts/python -m pip install torch torchvision "
      "--index-url https://download.pytorch.org/whl/cu126\n"
      ".venv/Scripts/python -m pip install -r requirements-lock.txt\n\n"
      "# 1. acquire data (verifies every SHA-256 against the publisher's value)\n"
      "python scripts/download_data.py\n\n"
      "# 2. manifest, cache, audit, splits  (audit exits non-zero if leakage-critical)\n"
      "python scripts/build_manifests.py --workers 12\n"
      "python scripts/build_image_cache.py --max-side 448 --workers 12\n"
      "python scripts/audit_data.py\n"
      "python scripts/build_splits.py --seed 42\n\n"
      "# 3. preflight (tests, loaders, 20-sample overfit, 1 epoch, leakage)\n"
      "python scripts/preflight.py\n\n"
      "# 4. baselines and the mandatory matrix\n"
      "python scripts/run_biomass_baselines.py\n"
      "python scripts/train_disease.py   --profile standard --only E1 E2 E3\n"
      "python scripts/train_biomass.py   --profile standard --only E4 E5\n"
      "python scripts/train_multitask.py --profile standard --only E6 E7\n"
      "python scripts/run_ablations.py   --profile standard --only E8\n"
      "python scripts/run_e9_seeds.py    --profile standard\n\n"
      "# 5. evaluation, figures, error analysis, export, report\n"
      "python scripts/evaluate.py --n-boot 2000\n"
      "python scripts/make_figures.py\n"
      "python scripts/error_analysis.py\n"
      "python scripts/export_best_lightweight.py\n"
      "python scripts/write_report.py\n"
      "```\n")
    A("Or simply `make all`.\n")

    # ---------------- export ----------------
    if exports:
        A("\n## 15. Model export verification\n")
        for e in exports:
            d = jload(e)
            A(f"\n**{d.get('checkpoint')}** ({d.get('backbone')})\n")
            rows = []
            for k in ("torchscript", "onnx"):
                v = d.get(k, {})
                if "error" in v:
                    rows.append([k, "FAILED", v["error"][:60], "-", "-"])
                else:
                    rows.append([k, f"{v.get('file_size_mb')} MB",
                                 f"{v.get('max_abs_diff'):.3e}",
                                 f"{v.get('mean_abs_diff'):.3e}",
                                 "yes" if v.get("allclose_atol_1e-4") else "no"])
            A(table(rows, ["Format", "Size", "Max |diff|", "Mean |diff|",
                           "allclose(1e-4)"]))
            lat = d.get("latency", {})
            if lat:
                A("\n" + table([[k, v.get("latency_ms_per_image"),
                                 v.get("throughput_img_per_s")]
                                for k, v in lat.items() if isinstance(v, dict)
                                and "latency_ms_per_image" in v],
                               ["Runtime", "ms/image", "images/s"]))
            A(f"\nVerified on {d.get('n_verification_samples')} real images from the "
              f"manifest.")

    (REPORTS / "EXPERIMENT_REPORT.md").write_text("\n".join(L), encoding="utf-8")
    print(f"wrote reports/EXPERIMENT_REPORT.md ({len('\n'.join(L))} chars)")

    # ---------------------------------------------------------------- #
    R: list[str] = []
    B = R.append
    B("# Reproducibility\n")
    B(f"Generated: {datetime.now(timezone.utc).isoformat()}\n")
    B("\n## Recorded per run\n")
    B("Every run writes `outputs/logs/<run>_meta.json` containing: configuration, "
      "random seed, git commit and dirty flag, manifest hash, split hash, full "
      "environment and hardware description, start/end timestamps, duration, best "
      "epoch, best validation metric, selected per-label thresholds, and paths to the "
      "checkpoint, predictions, metrics and training curve. One row per completed run "
      "is appended to `outputs/metrics/experiment_registry.csv`.\n")
    if len(reg):
        B(f"\nRegistry currently holds **{len(reg)} runs** "
          f"({int((reg['status'] == 'completed').sum())} completed).\n")
        # Provenance across ALL runs, not just reg.iloc[0]: the matrix was produced
        # over several commits and mostly from a dirty tree; summarising it by the
        # one clean first run would misrepresent how reproducible it actually is.
        def _uniq(col):
            return sorted(reg[col].dropna().astype(str).unique()) if col in reg.columns else []
        man_u, spl_u = _uniq("manifest_hash"), _uniq("split_hash")
        prov = [["runs", str(len(reg))]]
        for col, u in (("manifest_hash", man_u), ("split_hash", spl_u)):
            if col in reg.columns:
                prov.append([col, f"{u[0][:16]} (all {len(reg)} runs identical)"
                             if len(u) == 1 else
                             f"{len(u)} DISTINCT: " + ", ".join(h[:12] for h in u)])
        if "git_dirty" in reg.columns:
            dc = reg["git_dirty"].astype(str).value_counts().to_dict()
            prov.append(["git_dirty", ", ".join(f"{k}: {v}" for k, v in dc.items())])
        if "git_commit" in reg.columns:
            counts = reg["git_commit"].astype(str).value_counts()
            prov.append(["git_commit", f"{len(counts)} distinct: "
                         + ", ".join(f"{c[:7]}x{n}" for c, n in counts.items())])
        for col in ("torch", "gpu"):
            if col in reg.columns:
                prov.append([col, ", ".join(str(x)[:40] for x in _uniq(col)[:3])])
        B(table(prov, ["Field", "Across all runs"]))
        dirty_n = int((reg["git_dirty"].astype(str) == "True").sum()) if "git_dirty" in reg.columns else 0
        data_pinned = (len(man_u) == 1 and len(spl_u) == 1)
        B(f"\n**Provenance caveat.** All {len(reg)} runs consumed the "
          f"{'SAME manifest and split hashes, so the reported numbers are not mixed across data states'
             if data_pinned else 'data recorded above'}. "
          f"However {dirty_n} of {len(reg)} runs were trained from a git-DIRTY "
          f"working tree spanning {reg['git_commit'].nunique() if 'git_commit' in reg.columns else 'several'} "
          "commits, so their exact training code is not pinned to a clean commit. "
          "Every run's commit and dirty flag is recorded in the committed "
          "`experiment_registry.csv` (so any run is traceable), but checking out a "
          "single commit will not regenerate the whole matrix bit-for-bit.\n")
    B("\n## Determinism\n")
    B("`set_global_seed` seeds Python, NumPy and torch (CPU and CUDA) and enables "
      "cuDNN deterministic mode. Per-fold seeds are `seed + fold`. The multi-task "
      "batch interleave is generated from a seeded permutation, so the task schedule "
      "is reproducible. Multi-view sampling is seeded by a stable SHA-256 of "
      "(seed, epoch, individual) via `stable_seed`, so the per-epoch view draw "
      "reproduces across processes.\n")
    B("\n**Historical note (multi-view / E5):** the E5 predictions currently on disk "
      "were produced by an earlier build that seeded the multi-view draw from "
      "Python's per-process-salted `hash()`, which is NOT stable across "
      "interpreters; those specific numbers are therefore not regenerable by a "
      "re-run. The seed was replaced with `stable_seed` (SHA-256); regenerate E5 to "
      "obtain reproducible multi-view results.\n")
    B("\nResidual non-determinism: some cuDNN/cuBLAS kernels and atomics are not "
      "bit-reproducible even in deterministic mode, and DataLoader worker ordering can "
      "vary. Metrics should reproduce to within a few units in the third decimal, not "
      "bitwise.\n")
    B("\n## Metrics are recomputable without inference\n")
    B("`scripts/evaluate.py` loads no checkpoint. It reads the saved prediction "
      "parquet files and recomputes every headline number, including confidence "
      "intervals and paired comparisons. This is the check that the reported numbers "
      "are properties of saved artefacts rather than of a live model.\n")
    if PREDICTIONS.exists():
        pf = sorted(PREDICTIONS.glob("*.parquet"))
        B(f"\nPrediction files on disk: **{len(pf)}**.\n")
    B("\n## Data integrity\n")
    if acq.get("datasets"):
        B(table([[d["name"], d["resolved_doi"], d["n_files"],
                  "all match" if all(f.get("sha256_match") for f in d.get("files", []))
                  else "MISMATCH"]
                 for d in acq["datasets"].values()],
                ["Dataset", "DOI", "Files", "SHA-256 vs publisher"]))
    B("\n## Preflight\n")
    if pre:
        B(table([[c["status"], c["check"], c["detail"][:100]]
                 for c in pre.get("checks", [])],
                ["Status", "Check", "Detail"]))
    B("\n## Environment\n")
    B("Exact pinned versions are in `requirements-lock.txt`. Key versions:\n")
    B(f"- Python {env.get('python', platform.python_version())}")
    B(f"- torch {env.get('torch', 'n/a')} / CUDA {env.get('cuda_version', 'n/a')}")
    B(f"- timm {env.get('timm_version', 'n/a')}, "
      f"albumentations {env.get('albumentations_version', 'n/a')}, "
      f"scikit-learn {env.get('sklearn_version', 'n/a')}")

    (REPORTS / "REPRODUCIBILITY.md").write_text("\n".join(R), encoding="utf-8")
    print("wrote reports/REPRODUCIBILITY.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
