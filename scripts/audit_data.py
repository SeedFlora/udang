"""Run the mandatory data audit and emit reports/DATASET_AUDIT.md.

Exits non-zero when a leakage-critical check fails, so the pipeline (and the
Makefile) refuses to advance to training on unsound data.

Usage:
    python scripts/audit_data.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.audit.checks import run_all_checks  # noqa: E402
from src.data.labels import DISEASE_ATTRS  # noqa: E402
from src.utils.paths import MANIFESTS, METRICS, REPORTS, ensure_dirs  # noqa: E402

STATUS_ICON = {"pass": "PASS", "warn": "WARN", "fail": "FAIL"}


def fmt_table(rows: list[list], headers: list[str]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def main() -> int:
    ensure_dirs()
    mpath = MANIFESTS / "unified_manifest.parquet"
    if not mpath.exists():
        print("BLOCKER: unified manifest not found. Run scripts/build_manifests.py first.")
        return 2
    df = pd.read_parquet(mpath)
    diag = json.loads((MANIFESTS / "manifest_diagnostics.json").read_text(encoding="utf-8"))
    acq = {}
    ap = MANIFESTS / "acquisition_record.json"
    if ap.exists():
        acq = json.loads(ap.read_text(encoding="utf-8"))

    checks, summary = run_all_checks(df, diag)
    (METRICS / "audit_checks.json").write_text(
        json.dumps({"summary": summary, "checks": checks}, indent=2, default=str),
        encoding="utf-8")

    ok = df[~df["exclude"].astype(bool)]
    L: list[str] = []
    A = L.append

    A("# ShrimpVision-MTL - Dataset Audit (E0)\n")
    A(f"Generated: {datetime.now(timezone.utc).isoformat()}  ")
    A(f"Manifest: `data/manifests/unified_manifest.parquet` "
      f"({len(df)} rows, {len(ok)} usable)  ")
    mh = (MANIFESTS / "manifest_hash.txt")
    if mh.exists():
        A(f"Manifest content hash: `{mh.read_text(encoding='utf-8').strip()}`\n")

    A(f"\n**Leakage gate: {summary['gate']}** "
      f"({summary['n_pass']} pass, {summary['n_warn']} warn, {summary['n_fail']} fail "
      f"of {summary['n_checks']} checks)\n")
    if summary["leakage_critical_failures"]:
        A(f"\n> BLOCKED by: {', '.join(summary['leakage_critical_failures'])}\n")

    # ---------------- acquisition ----------------
    A("\n## 1. Acquisition, provenance and licence\n")
    if acq.get("datasets"):
        rows = []
        for _k, d in acq["datasets"].items():
            rows.append([
                d["name"], d["resolved_version"], f"`{d['resolved_doi']}`",
                d["retrieval_date_utc"][:19].replace("T", " "),
                d["n_files"], f"{(d.get('api_total_size_bytes') or 0) / 1e6:.1f} MB",
                d.get("declared_licence", "?"),
            ])
        A(fmt_table(rows, ["Dataset", "Ver", "DOI", "Retrieved (UTC)", "Files",
                           "Size", "Licence"]))
        A("\n### Version conflicts\n")
        any_conf = False
        for _k, d in acq["datasets"].items():
            if d.get("version_conflict"):
                any_conf = True
                A(f"- **{d['name']}**: requested `{d['requested_doi']}` "
                  f"(v{d['requested_version']}), resolved to `{d['resolved_doi']}` "
                  f"(v{d['resolved_version']}). Reason: {d.get('version_conflict_reason')}")
        if not any_conf:
            A("- None.")

        A("\n### Integrity verification\n")
        A("Every downloaded file's SHA-256 was recomputed locally and compared with "
          "the publisher-declared hash from the Mendeley API.\n")
        rows = []
        for _k, d in acq["datasets"].items():
            n = len(d.get("files", []))
            good = sum(1 for f in d.get("files", []) if f.get("sha256_match"))
            rows.append([d["name"], n, good, "ALL MATCH" if good == n else "MISMATCH"])
        A(fmt_table(rows, ["Dataset", "Files hashed", "SHA-256 matches", "Verdict"]))

        A("\n### Archive checksums (top-level files)\n")
        rows = []
        for _k, d in acq["datasets"].items():
            for f in d.get("files", [])[:6]:
                if f["filename"].lower().endswith((".zip", ".xlsx", ".docx")):
                    rows.append([d["name"], f["filename"], f"{f['size_bytes'] / 1e6:.2f} MB",
                                 f"`{(f['sha256_local'] or '')[:32]}...`"])
        if rows:
            A(fmt_table(rows, ["Dataset", "File", "Size", "SHA-256 (truncated)"]))
    else:
        A("_No acquisition record found._")

    if acq.get("licence_conflicts"):
        A("\n### Licence conflicts\n")
        for k, v in acq["licence_conflicts"].items():
            A(f"- **{k}**: repository field says `{v['mendeley_declared']}`, "
              f"description text says \"{v['description_claim']}\". "
              f"{v['resolution']} See `LICENSE_NOTES.md`.")

    # ---------------- counts ----------------
    A("\n## 2. Verified counts after exclusions\n")
    rows = []
    for (ds, task, role), g in ok.groupby(["dataset", "task", "record_role"]):
        rows.append([ds, task, role, len(g), g["group_id"].nunique()])
    A(fmt_table(rows, ["Dataset", "Task", "Role", "Images", "Fused groups"]))

    A("\n### Disease class distribution (usable images)\n")
    dis = ok[ok["task"] == "disease"]
    rows = []
    for (ds, cls, role), g in dis.groupby(["dataset", "raw_class", "record_role"]):
        rows.append([ds, cls, role, len(g)])
    A(fmt_table(rows, ["Dataset", "Raw class", "Role", "Images"]))

    A("\n### Harmonized label supervision (masked, unknown never coerced to negative)\n")
    rows = []
    for a in DISEASE_ATTRS:
        sup = dis[f"label_mask_{a}"].fillna(0) > 0
        pos = ((dis[a].fillna(0) > 0) & sup).sum()
        rows.append([a, int(sup.sum()), int(len(dis) - sup.sum()), int(pos),
                     int(sup.sum() - pos)])
    A(fmt_table(rows, ["Attribute", "Supervised", "Unknown (masked)", "Positive",
                       "Negative"]))
    A("\n`yellow_head` is unknown for every ShrimpDiseaseImageBD image because that "
      "dataset has no Yellow Head category. Those rows carry mask=0 and contribute "
      "zero loss; they are **not** treated as yellow_head-negative.\n")

    A("\n### Biomass structure\n")
    bm = ok[ok["task"] == "biomass"]
    if len(bm):
        rows = []
        for db, g in bm.groupby("pond_or_db_id"):
            rows.append([db, len(g), g["individual_id"].nunique(),
                         round(len(g) / max(1, g["individual_id"].nunique()), 1),
                         int(g["weight_g"].notna().sum()),
                         int(g["cephalothorax_cm"].notna().sum())])
        A(fmt_table(rows, ["DB", "Images", "Individuals", "Views/individual",
                           "With weight", "With cephalothorax"]))
        A(f"\nTotal: **{len(bm)} images**, **{bm['individual_id'].nunique()} individuals**.")
        w = bm.drop_duplicates("individual_id")["weight_g"].dropna()
        ln = bm.drop_duplicates("individual_id")["length_cm"].dropna()
        if len(w):
            A(f"\nWeight (g) per individual: min {w.min():.2f}, median {w.median():.2f}, "
              f"max {w.max():.2f}, mean {w.mean():.2f}, sd {w.std():.2f}  ")
        if len(ln):
            A(f"Length (cm) per individual: min {ln.min():.2f}, median {ln.median():.2f}, "
              f"max {ln.max():.2f}\n")

    # ---------------- checks ----------------
    A("\n## 3. Audit checks\n")
    rows = [[STATUS_ICON[c["status"]], c["check"], c["severity"], c["count"],
             c["message"][:160]] for c in checks]
    A(fmt_table(rows, ["Status", "Check", "Severity", "Count", "Detail"]))

    A("\n### Check detail\n")
    for c in checks:
        A(f"\n#### {STATUS_ICON[c['status']]} - `{c['check']}` ({c['severity']})\n")
        A(c["message"])
        if c["extra"]:
            A("\n```json\n" + json.dumps(c["extra"], indent=2, default=str)[:2200] + "\n```")
        if c["examples"]:
            A("\nExamples:\n```\n" +
              "\n".join(str(e)[:220] for e in c["examples"][:8]) + "\n```")

    # ---------------- grouping ----------------
    A("\n## 4. Grouping and leakage control\n")
    d = diag.get("dedup", {})
    A(f"- records grouped: **{d.get('n_records')}**")
    A(f"- fused groups: **{d.get('n_groups')}**")
    A(f"- exact-duplicate links: {d.get('n_exact_dup_links')}")
    A(f"- augmentation-family links: {d.get('n_family_links')}")
    A(f"- perceptual near-duplicate links (within-block): {d.get('n_near_dup_links')}")
    A(f"- forced links (individual identity (biomass shrimp; SDIBD animals)): "
      f"{d.get('n_forced_links')}")
    A(f"- largest group: {d.get('largest_group')} images")
    A("\nThe grouping key is the transitive closure of exact-duplicate, "
      "augmentation-family and within-block perceptual near-duplicate relations, "
      "plus forced links binding every image of one individual (biomass shrimp or "
      "SDIBD animal). All splits "
      "are constructed over this key, so an augmented copy can never appear in a "
      "different split from its source original, and no shrimp can appear in two folds.\n")
    A("\nPerceptual fusion is deliberately **blocked** by dataset+class (disease) and "
      "by individual (biomass). Unblocked fusion would chain the 170 biomass shrimp "
      "photographed on one rig into a single component and collapse the splits.\n")

    A("\n## 5. Excluded records\n")
    exc = df[df["exclude"].astype(bool)]
    A(f"{len(exc)} records excluded. See `data/manifests/excluded_records.csv`.")
    if len(exc):
        A("\n```\n" + exc[["record_id", "relative_image_path", "exclude_reason"]]
          .head(20).to_string(index=False) + "\n```")

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "DATASET_AUDIT.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(f"[{STATUS_ICON[c['status']]}] {c['check']}: {c['message'][:110]}"
                    for c in checks))
    print(f"\nGATE: {summary['gate']}")
    print("wrote reports/DATASET_AUDIT.md and outputs/metrics/audit_checks.json")
    return 1 if summary["gate"] == "BLOCKED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
