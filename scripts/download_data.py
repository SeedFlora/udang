"""Acquire the three ShrimpVision-MTL source datasets from Mendeley Data.

Fail-closed by design. Every structural expectation declared in
configs/data/sources.yaml is asserted against the live API *before* any file is
written or any class label is attached to a folder. A contract breach aborts the
run instead of producing plausibly-labelled but wrong data.

For every dataset this records: name, version, DOI, retrieval date, archive
filename, file size, SHA-256 (both publisher-declared and locally recomputed),
declared licence, licence source and citation requirement.

Usage:
    python scripts/download_data.py                 # all datasets
    python scripts/download_data.py --only tiger_shrimp_bd
    python scripts/download_data.py --verify-only   # re-verify local files
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.hashing import sha256_file  # noqa: E402
from src.utils.paths import MANIFESTS, RAW, CONFIGS, ensure_dirs, rel_to_repo  # noqa: E402

UA = {"User-Agent": "ShrimpVision-MTL/1.0 (research data acquisition)"}
TIMEOUT = 180

# NOTE: this module deliberately uses stdlib urllib rather than `requests`.
# data.mendeley.com answers every `requests`/urllib3 call with HTTP 403 regardless
# of User-Agent, while the identical URLs succeed under urllib. Verified by
# isolating the variables (4 UA strings x both stacks) during acquisition.


def http_get(url: str, retries: int = 5) -> bytes:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET failed after {retries} attempts: {url} :: {last}")


def http_get_json(url: str) -> dict:
    return json.loads(http_get(url).decode("utf-8"))


def http_stream_to_file(url: str, dest: Path, retries: int = 5) -> None:
    last: Exception | None = None
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r, open(tmp, "wb") as fh:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
            tmp.replace(dest)
            return
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"download failed after {retries} attempts: {url} :: {last}")


# The Mendeley download URL 302-redirects to S3, which throttles a single
# connection to roughly 0.05-0.25 MB/s -- hours for the 1.46 GB biomass archive.
# S3 advertises `Accept-Ranges: bytes` and answers 206, so a large file is pulled
# as N concurrent byte ranges written into one preallocated file via seek().
RANGED_MIN_BYTES = 16 << 20
RANGED_PART_BYTES = 24 << 20     # target size of one range
RANGED_MAX_PARTS = 96
RANGED_CONCURRENCY = 6           # simultaneous connections to the CDN

# Sizing rationale: a range that stalls is retried FROM SCRATCH, so large ranges
# make a slow tail extremely expensive (a stalled 91 MB range discards 91 MB on
# every attempt). Small ranges bound that loss, and capping concurrency at 6
# avoids the sustained-throughput throttling that appeared with 16 open
# connections during acquisition of the 1.46 GB biomass archive.


RANGE_TIMEOUT = 60          # detect a stalled socket quickly
RANGE_MAX_ATTEMPTS = 40     # reconnects, not restarts -- each resumes where it stopped


def _fetch_range(url: str, dest: Path, start: int, end: int,
                 retries: int = RANGE_MAX_ATTEMPTS) -> int:
    """Download bytes [start, end] inclusive into dest at offset start, RESUMABLY.

    Chunks are written straight to their final offset and a running `written`
    counter is kept, so a stalled connection is resumed with
    `Range: bytes=start+written-end` instead of discarding the range and starting
    over. Without this, the CDN's habit of stalling mid-transfer means a large
    range can never complete: every retry loses all of its progress.

    Ranges are disjoint by construction, so concurrent writers need no locking.
    """
    written = 0
    last: Exception | None = None
    want = end - start + 1
    for attempt in range(retries):
        if written >= want:
            break
        try:
            lo = start + written
            req = urllib.request.Request(
                url, headers={**UA, "Range": f"bytes={lo}-{end}"})
            with urllib.request.urlopen(req, timeout=RANGE_TIMEOUT) as r:
                if r.status not in (206, 200):
                    raise RuntimeError(f"expected 206, got {r.status}")
                with open(dest, "r+b") as fh:
                    fh.seek(lo)
                    while True:
                        chunk = r.read(1 << 19)
                        if not chunk:
                            break
                        fh.write(chunk)
                        written += len(chunk)
                        if written >= want:
                            break
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(min(8.0, 1.0 + 0.5 * attempt))
    if written < want:
        raise RuntimeError(
            f"range {start}-{end} incomplete: {written}/{want} bytes after "
            f"{retries} attempts :: {last}")
    return want


def http_download_ranged(url: str, dest: Path, total: int, parts: int | None = None,
                         label: str = "") -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "wb") as fh:  # preallocate so every worker can seek independently
        fh.truncate(total)

    if parts is None:
        parts = max(4, min(RANGED_MAX_PARTS,
                           (total + RANGED_PART_BYTES - 1) // RANGED_PART_BYTES))
    size = (total + parts - 1) // parts
    ranges = [(i * size, min(total - 1, (i + 1) * size - 1)) for i in range(parts)]
    ranges = [(s, e) for s, e in ranges if s <= e]

    got = 0
    done = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=min(RANGED_CONCURRENCY, len(ranges))) as ex:
        futs = {ex.submit(_fetch_range, url, tmp, s, e): (s, e) for s, e in ranges}
        for fut in as_completed(futs):
            got += fut.result()
            done += 1
            el = max(1e-6, time.time() - t0)
            if done % 4 == 0 or done == len(ranges):
                print(f"      {label} {got / 1e6:.0f}/{total / 1e6:.0f} MB "
                      f"[{done}/{len(ranges)} ranges] ({got / 1e6 / el:.2f} MB/s)",
                      flush=True)

    if tmp.stat().st_size != total:
        raise RuntimeError(f"assembled size {tmp.stat().st_size} != expected {total}")
    tmp.replace(dest)


class ContractError(RuntimeError):
    """A declared expectation about the source data did not hold."""


# --------------------------------------------------------------------------- #
# Mendeley API
# --------------------------------------------------------------------------- #
def fetch_dataset_meta(api_base: str, mendeley_id: str) -> dict:
    return http_get_json(f"{api_base}/{mendeley_id}")


def licence_block(meta: dict) -> dict:
    lic = meta.get("data_licence") or {}
    contributors = "; ".join(
        f"{c.get('first_name', '').strip()} {c.get('last_name', '').strip()}".strip()
        for c in meta.get("contributors", [])
    )
    return {
        "declared_licence": lic.get("short_name"),
        "declared_licence_full": lic.get("full_name"),
        "licence_url": lic.get("url"),
        "licence_source": f"Mendeley Data public API field data_licence for dataset {meta.get('id')}",
        "citation_required": True,  # CC BY 4.0 mandates attribution
        "commercial_use_restriction": (
            "None imposed by CC BY 4.0 (attribution required)."
            if (lic.get("short_name") or "").upper().startswith("CC BY")
            else "UNKNOWN - inspect licence text before any non-research use."
        ),
        "authors": contributors,
        "publish_date": meta.get("publish_date"),
        "title": meta.get("name"),
    }


# --------------------------------------------------------------------------- #
# Contract verification
# --------------------------------------------------------------------------- #
def verify_contract(key: str, cfg: dict, meta: dict) -> list[str]:
    """Assert declared expectations. Returns notes; raises ContractError on breach."""
    notes: list[str] = []
    v = cfg.get("verify", {})
    files = meta.get("files", [])

    if "version" in v and meta.get("version") != v["version"]:
        raise ContractError(
            f"{key}: expected version {v['version']}, API serves {meta.get('version')}"
        )
    if "n_files" in v and len(files) != v["n_files"]:
        raise ContractError(f"{key}: expected {v['n_files']} files, API lists {len(files)}")
    if "total_size_bytes" in v and meta.get("size") != v["total_size_bytes"]:
        raise ContractError(
            f"{key}: expected total size {v['total_size_bytes']}, API reports {meta.get('size')}"
        )
    if "filenames" in v:
        got = {f["filename"] for f in files}
        missing = set(v["filenames"]) - got
        if missing:
            raise ContractError(f"{key}: missing expected files {sorted(missing)}")

    # TigerShrimpBD folder -> class contract
    if "folders" in cfg:
        by_folder: dict[str, int] = {}
        for f in files:
            by_folder[f.get("folder_id")] = by_folder.get(f.get("folder_id"), 0) + 1
        declared = {fo["folder_id"]: fo for fo in cfg["folders"]}
        unknown = set(by_folder) - set(declared)
        if unknown:
            raise ContractError(f"{key}: undeclared folder ids present: {sorted(unknown)}")
        for fid, fo in declared.items():
            got_n = by_folder.get(fid, 0)
            if got_n != fo["expect_n"]:
                raise ContractError(
                    f"{key}: folder {fid} ({fo['class']}/{fo['role']}) expected "
                    f"{fo['expect_n']} files, API lists {got_n}. Class mapping is "
                    f"derived from these counts, so labelling is unsafe. Aborting."
                )
            notes.append(f"folder {fid[:8]} -> {fo['class']}/{fo['role']} : {got_n} files (OK)")

        # Cross-check: every augmented file's base name must exist as an original
        # in the SAME class. This independently confirms the folder pairing.
        for cls in sorted({fo["class"] for fo in cfg["folders"]}):
            aug_ids = {fid for fid, fo in declared.items() if fo["class"] == cls and fo["role"] == "augmented"}
            org_ids = {fid for fid, fo in declared.items() if fo["class"] == cls and fo["role"] == "original"}
            org_names = {
                Path(f["filename"]).stem.lower()
                for f in files if f.get("folder_id") in org_ids
            }
            aug_names = [f["filename"] for f in files if f.get("folder_id") in aug_ids]
            matched = sum(1 for a in aug_names if _strip_aug(a).lower() in org_names)
            frac = matched / max(1, len(aug_names))
            notes.append(
                f"class {cls}: {matched}/{len(aug_names)} augmented files trace to an "
                f"original in the paired folder ({frac:.1%})"
            )
            if frac < 0.50:
                raise ContractError(
                    f"{key}/{cls}: only {frac:.1%} of augmented files trace back to a "
                    f"paired original. Augmentation families cannot be grouped safely."
                )
    return notes


def _strip_aug(filename: str) -> str:
    """Reduce a TigerShrimpBD filename to its presumed original stem."""
    import re
    stem = Path(filename).stem
    stem = re.sub(r"^aug_\d+_", "", stem, flags=re.I)
    stem = re.sub(r"^aug_", "", stem, flags=re.I)
    stem = re.sub(r"[-_ ]?resized$", "", stem, flags=re.I)
    stem = re.sub(r"[-_ ]?copy$", "", stem, flags=re.I)
    stem = re.sub(r"\s*\(\d+\)$", "", stem)
    return stem


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
def download_one(url: str, dest: Path, expect_size: int | None) -> tuple[bool, str]:
    """Skip when a byte-exact copy is already on disk, else fetch with retries.

    Large files go through the ranged/parallel path; small ones use one request.
    """
    if dest.exists() and expect_size and dest.stat().st_size == expect_size:
        return True, "cached"
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        if expect_size and expect_size >= RANGED_MIN_BYTES:
            http_download_ranged(url, dest, expect_size, label=dest.name[:38])
        else:
            http_stream_to_file(url, dest)
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"
    if expect_size and dest.stat().st_size != expect_size:
        return False, f"size {dest.stat().st_size} != expected {expect_size}"
    return True, "downloaded"


def acquire_dataset(key: str, cfg: dict, api_base: str, workers: int,
                    verify_only: bool) -> dict:
    print(f"\n{'=' * 78}\n### {key} :: {cfg['name']}\n{'=' * 78}", flush=True)
    meta = fetch_dataset_meta(api_base, cfg["mendeley_id"])

    notes = verify_contract(key, cfg, meta)
    for n in notes:
        print(f"  [contract] {n}", flush=True)
    print(f"  [contract] all declared expectations hold for {key}", flush=True)

    lic = licence_block(meta)
    print(f"  licence: {lic['declared_licence']}  ({lic['licence_url']})", flush=True)

    out_root = RAW / cfg["raw_subdir"]
    out_root.mkdir(parents=True, exist_ok=True)
    files = meta["files"]

    # Where does each file land locally?
    if cfg["layout"] == "loose_files":
        folder_map = {fo["folder_id"]: fo for fo in cfg["folders"]}
        targets = {
            f["id"]: out_root / folder_map[f["folder_id"]]["class"]
            / folder_map[f["folder_id"]]["role"] / f["filename"]
            for f in files
        }
    else:
        targets = {f["id"]: out_root / f["filename"] for f in files}

    records: list[dict] = []
    if not verify_only:
        todo = [(f, targets[f["id"]]) for f in files]
        done = 0
        t0 = time.time()
        # Archives fan out internally over byte ranges, so keep the outer pool
        # small; loose-file datasets need the wide pool instead.
        outer = 2 if cfg["layout"] == "archives" else workers
        with ThreadPoolExecutor(max_workers=outer) as ex:
            futs = {
                ex.submit(
                    download_one,
                    f["content_details"]["download_url"], dest, f["size"],
                ): (f, dest)
                for f, dest in todo
            }
            for fut in as_completed(futs):
                f, dest = futs[fut]
                ok, status = fut.result()
                done += 1
                if not ok:
                    raise ContractError(f"{key}: failed to download {f['filename']}: {status}")
                if done % 250 == 0 or done == len(todo):
                    rate = done / max(1e-6, time.time() - t0)
                    print(f"    {done}/{len(todo)} files  ({rate:.1f}/s)", flush=True)

    # Recompute SHA-256 locally and compare against publisher-declared value.
    print("  verifying local SHA-256 against publisher-declared hashes ...", flush=True)
    mismatches = []
    with ThreadPoolExecutor(max_workers=min(16, workers)) as ex:
        futs = {ex.submit(sha256_file, targets[f["id"]]): f for f in files}
        for fut in as_completed(futs):
            f = futs[fut]
            local = fut.result()
            declared = f["content_details"].get("sha256_hash")
            if declared and local != declared:
                mismatches.append(f["filename"])
            records.append({
                "filename": f["filename"],
                "relative_path": rel_to_repo(targets[f["id"]]),
                "size_bytes": f["size"],
                "sha256_local": local,
                "sha256_declared": declared,
                "sha256_match": (declared is None) or (local == declared),
                "folder_id": f.get("folder_id"),
            })
    if mismatches:
        raise ContractError(
            f"{key}: SHA-256 mismatch on {len(mismatches)} file(s): {mismatches[:5]} - archive corrupt"
        )
    print(f"  [integrity] {len(records)} files, all SHA-256 match publisher-declared values", flush=True)

    # Extract archives
    extracted = []
    for arc in cfg.get("extract", []):
        src = out_root / arc
        dst = out_root / "extracted"
        marker = dst / (Path(arc).stem + ".extracted.ok")
        if marker.exists():
            print(f"  [extract] {arc} already extracted", flush=True)
            extracted.append(arc)
            continue
        print(f"  [extract] {arc} -> {rel_to_repo(dst)}", flush=True)
        dst.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(src) as zf:
                bad = zf.testzip()
                if bad is not None:
                    raise ContractError(f"{key}: corrupt entry {bad} in {arc}")
                zf.extractall(dst)
        except zipfile.BadZipFile as e:
            raise ContractError(f"{key}: {arc} is not a readable zip archive: {e}") from e
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("ok", encoding="utf-8")
        extracted.append(arc)

    return {
        "key": key,
        "name": cfg["name"],
        "long_name": cfg.get("long_name"),
        "mendeley_id": cfg["mendeley_id"],
        "requested_doi": cfg["requested_doi"],
        "requested_version": cfg["requested_version"],
        "resolved_doi": meta["doi"]["id"],
        "resolved_version": meta["version"],
        "version_conflict": bool(cfg.get("version_conflict", False)),
        "version_conflict_reason": cfg.get("version_conflict_reason"),
        "retrieval_date_utc": datetime.now(timezone.utc).isoformat(),
        "api_total_size_bytes": meta.get("size"),
        "n_files": len(files),
        "task": cfg["task"],
        "raw_subdir": cfg["raw_subdir"],
        "extracted_archives": extracted,
        "contract_notes": notes,
        **lic,
        "files": records,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    ensure_dirs()
    cfg = yaml.safe_load((CONFIGS / "data" / "sources.yaml").read_text(encoding="utf-8"))
    keys = args.only or list(cfg["datasets"].keys())

    out: dict = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "api_base": cfg["api_base"],
        "licence_conflicts": cfg.get("licence_conflicts", {}),
        "datasets": {},
    }
    record_path = MANIFESTS / "acquisition_record.json"
    if record_path.exists():  # keep previously acquired datasets when running --only
        try:
            out["datasets"] = json.loads(record_path.read_text(encoding="utf-8")).get("datasets", {})
        except Exception:  # noqa: BLE001
            pass

    failures = []
    for k in keys:
        try:
            out["datasets"][k] = acquire_dataset(
                k, cfg["datasets"][k], cfg["api_base"], args.workers, args.verify_only
            )
        except ContractError as e:
            print(f"\n!!! BLOCKER [{k}]: {e}", flush=True)
            failures.append({"dataset": k, "error": str(e)})
        except Exception as e:  # noqa: BLE001
            print(f"\n!!! FAILURE [{k}]: {type(e).__name__}: {e}", flush=True)
            failures.append({"dataset": k, "error": f"{type(e).__name__}: {e}"})

    out["failures"] = failures
    record_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nacquisition record -> {rel_to_repo(record_path)}", flush=True)
    if failures:
        print(f"FAILED datasets: {[f['dataset'] for f in failures]}", flush=True)
        return 1
    print("ALL DATASETS ACQUIRED AND VERIFIED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
