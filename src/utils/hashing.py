"""Content hashing helpers used by acquisition, audit and split integrity checks."""
from __future__ import annotations

import hashlib
from pathlib import Path

_CHUNK = 1 << 20  # 1 MiB


def sha256_file(path: str | Path, chunk: int = _CHUNK) -> str:
    """Streamed SHA-256 of a file (never loads the whole file into memory)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_seed(*parts: object, bits: int = 32) -> int:
    """Deterministic, process-independent integer seed from arbitrary parts.

    Python's built-in ``hash()`` salts string hashing per process
    (``PYTHONHASHSEED``), so seeding an RNG from ``hash((..., some_str))`` yields
    different draws on every fresh interpreter -- silently breaking any claim that
    a run is reproducible. This routes the parts through SHA-256, which is stable
    across processes and machines. Use it wherever a seed must reproduce.
    """
    text = "\x1f".join(str(p) for p in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[: max(1, bits // 8)], "big")


def hash_dataframe(df, cols: list[str] | None = None) -> str:
    """Order-independent content hash of a dataframe.

    Used to stamp every run with the manifest / split identity it consumed, so a
    metric file can always be traced back to the exact data partition.
    """
    import pandas as pd  # local import keeps this module importable without pandas

    sub = df[cols] if cols else df
    canonical = sub.sort_values(list(sub.columns)).to_csv(index=False)
    return sha256_text(canonical)
