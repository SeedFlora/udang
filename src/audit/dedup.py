"""Duplicate / near-duplicate / augmentation-family clustering.

Everything that governs leakage safety flows through this module. Four distinct
relations are computed and then FUSED into a single grouping key:

  1. exact duplicates          - identical SHA-256
  2. augmentation families     - a published augmented copy and its source original
  3. near duplicates           - perceptual-hash Hamming distance <= threshold
  4. forced individual links   - every view of one animal

The fused key is what the splitter groups on. Fusing matters: an augmented copy
may be near-duplicate of a *different* original than the one its filename claims,
and a raw image may reappear inside an "annotated" folder under a new name. Only
the transitive closure of all four relations is safe to split on.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np


class UnionFind:
    """Disjoint-set with path compression; keys are arbitrary hashables."""

    def __init__(self) -> None:
        self._parent: dict = {}
        self._rank: dict = {}

    def add(self, x) -> None:
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x] = 0

    def find(self, x):
        self.add(x)
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:  # path compression
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a, b) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self._rank[ra] < self._rank[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        if self._rank[ra] == self._rank[rb]:
            self._rank[ra] += 1

    def groups(self) -> dict:
        out: dict = {}
        for x in self._parent:
            out.setdefault(self.find(x), []).append(x)
        return out

    def n_components(self) -> int:
        return len({self.find(x) for x in list(self._parent)})


# --------------------------------------------------------------------------- #
# Perceptual hashing
# --------------------------------------------------------------------------- #
def phash_to_uint64(phash_hex: str) -> np.uint64:
    """Pack a 16-hex-char (64-bit) perceptual hash into a uint64 for fast XOR."""
    return np.uint64(int(phash_hex, 16))


_POPCOUNT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _popcount64(x: np.ndarray) -> np.ndarray:
    """Vectorised popcount over a uint64 array."""
    v = x.view(np.uint8).reshape(*x.shape, 8)
    return _POPCOUNT8[v].sum(axis=-1).astype(np.int16)


def near_duplicate_pairs(hashes: np.ndarray, threshold: int = 6,
                         block: int = 2048) -> list[tuple[int, int, int]]:
    """All (i, j, distance) pairs with i<j and Hamming distance <= threshold.

    Blocked XOR/popcount over the full matrix. At ~11k images this is ~1.2e8
    comparisons, which numpy handles in seconds -- an approximate index (BK-tree,
    LSH) would add failure modes for no practical gain at this scale.
    """
    n = len(hashes)
    out: list[tuple[int, int, int]] = []
    for i0 in range(0, n, block):
        i1 = min(n, i0 + block)
        left = hashes[i0:i1, None]
        for j0 in range(i0, n, block):
            j1 = min(n, j0 + block)
            d = _popcount64(np.bitwise_xor(left, hashes[None, j0:j1]))
            ii, jj = np.nonzero(d <= threshold)
            for a, b in zip(ii, jj):
                gi, gj = i0 + int(a), j0 + int(b)
                if gi < gj:
                    out.append((gi, gj, int(d[a, b])))
    return out


# --------------------------------------------------------------------------- #
# TigerShrimpBD augmentation-family reconstruction
# --------------------------------------------------------------------------- #
_AUG_PATTERNS = [
    (re.compile(r"^aug_(\d+)_(?P<base>.+)$", re.I), "aug_indexed"),
    (re.compile(r"^aug_(?P<base>.+)$", re.I), "aug_plain"),
]
_SUFFIX_PATTERNS = [
    # Separators are consumed greedily: real names include
    # 'aug_0_IMG-20240421-WA0079 - Copy.jpg', where a single optional separator
    # would leave a trailing ' -' and split the family away from its siblings.
    (re.compile(r"^(?P<base>.+?)[\s\-_]*resized$", re.I), "resized"),
    (re.compile(r"^(?P<base>.+?)[\s\-_]*copy$", re.I), "copy"),
    (re.compile(r"^(?P<base>.+?)[\s\-_]*\(\d+\)$", re.I), "paren_copy"),
]
_TRAILING_SEPARATORS = re.compile(r"[\s\-_]+$")


def normalize_original_stem(filename: str) -> tuple[str, list[str]]:
    """Reduce a filename to its presumed source-original stem.

    Returns (base_stem, transforms_stripped). Applied repeatedly because published
    names stack markers, e.g. 'aug_3_IMG-1234-resized.jpg'.
    """
    stem = Path(filename).stem
    applied: list[str] = []
    changed = True
    while changed:
        changed = False
        for pat, tag in _AUG_PATTERNS:
            m = pat.match(stem)
            if m:
                stem, changed = m.group("base"), True
                applied.append(tag)
                break
        if changed:
            continue
        for pat, tag in _SUFFIX_PATTERNS:
            m = pat.match(stem)
            if m:
                stem, changed = m.group("base"), True
                applied.append(tag)
                break
    return _TRAILING_SEPARATORS.sub("", stem), applied


def is_derivative(filename: str) -> bool:
    _, applied = normalize_original_stem(filename)
    return bool(applied)


def near_duplicate_pairs_blocked(hashes: np.ndarray, blocks: np.ndarray,
                                 threshold: int = 6) -> list[tuple[int, int, int]]:
    """Near-duplicate pairs restricted to within-block comparisons.

    Blocking is a correctness requirement, not an optimisation. The biomass
    dataset photographs 170 different shrimp against the same rig and background,
    so unblocked perceptual-hash fusion would chain distinct individuals into one
    giant component and collapse the splits. Cross-block similarity is still
    computed separately for the audit -- it is reported, never silently fused.
    """
    out: list[tuple[int, int, int]] = []
    for b in np.unique(blocks):
        idx = np.nonzero(blocks == b)[0]
        if len(idx) < 2:
            continue
        sub = near_duplicate_pairs(hashes[idx], threshold=threshold)
        out.extend((int(idx[i]), int(idx[j]), d) for i, j, d in sub)
    return out


def cross_block_near_duplicates(hashes: np.ndarray, blocks: np.ndarray,
                                threshold: int = 6,
                                limit: int = 5000) -> list[tuple[int, int, int]]:
    """Near-duplicate pairs that CROSS block boundaries (e.g. the same photo
    appearing in two different datasets). Reported in the audit; never auto-fused."""
    pairs = near_duplicate_pairs(hashes, threshold=threshold)
    out = [(i, j, d) for i, j, d in pairs if blocks[i] != blocks[j]]
    return out[:limit]


def build_fused_groups(record_ids: list[str],
                       sha256s: list[str],
                       phash_u64: np.ndarray | None,
                       family_keys: list[str | None],
                       phash_threshold: int = 6,
                       near_dup_blocks: np.ndarray | None = None,
                       extra_links: list[tuple[str, str]] | None = None,
                       ) -> tuple[dict[str, str], dict[str, dict]]:
    """Fuse exact-dup, augmentation-family, near-dup and forced-individual-link
    relations into one key.

    near_dup_blocks restricts perceptual fusion to within-block pairs (see
    near_duplicate_pairs_blocked). extra_links forces additional unions, used to
    bind every record that shares an individual_id together -- every view of one
    biomass shrimp AND, since the round-1 revision, every view of one
    ShrimpDiseaseImageBD animal.

    Returns (record_id -> group_id, diagnostics).

    Only the terminal count is order-invariant; the per-stage intermediates
    depend on the fixed order exact -> family -> near -> forced and are reported
    for the audit funnel.
    """
    uf = UnionFind()
    for rid in record_ids:
        uf.add(rid)

    n_after: dict[str, int] = {"seeded": len(record_ids)}

    # 1) exact duplicates
    by_sha: dict[str, list[str]] = {}
    for rid, sha in zip(record_ids, sha256s):
        by_sha.setdefault(sha, []).append(rid)
    n_exact = 0
    for sha, rids in by_sha.items():
        for other in rids[1:]:
            uf.union(rids[0], other)
            n_exact += 1
    n_after["after_exact_duplicates"] = uf.n_components()

    # 2) declared augmentation families
    by_family: dict[str, list[str]] = {}
    for rid, fam in zip(record_ids, family_keys):
        if fam:
            by_family.setdefault(fam, []).append(rid)
    n_family_links = 0
    for fam, rids in by_family.items():
        for other in rids[1:]:
            uf.union(rids[0], other)
            n_family_links += 1
    n_after["after_augmentation_families"] = uf.n_components()

    # 3) perceptual near-duplicates (blocked when a blocking key is supplied)
    n_near = 0
    near_pairs: list[tuple[int, int, int]] = []
    if phash_u64 is not None and len(phash_u64) == len(record_ids):
        if near_dup_blocks is not None:
            near_pairs = near_duplicate_pairs_blocked(
                phash_u64, near_dup_blocks, threshold=phash_threshold)
        else:
            near_pairs = near_duplicate_pairs(phash_u64, threshold=phash_threshold)
        for i, j, _d in near_pairs:
            uf.union(record_ids[i], record_ids[j])
            n_near += 1
    n_after["after_perceptual_near_duplicates"] = uf.n_components()

    # 4) caller-forced links (every view of one individual: biomass shrimp or
    # ShrimpDiseaseImageBD animal)
    n_extra = 0
    for a, b in (extra_links or []):
        uf.union(a, b)
        n_extra += 1
    n_after["after_forced_individuals"] = uf.n_components()

    groups = uf.groups()
    # stable, human-readable group ids ordered by size then first member
    ordered = sorted(groups.values(), key=lambda g: (-len(g), min(g)))
    mapping: dict[str, str] = {}
    for k, members in enumerate(ordered):
        gid = f"grp_{k:06d}"
        for m in members:
            mapping[m] = gid

    sizes = np.array([len(g) for g in ordered])
    diagnostics = {
        "n_records": len(record_ids),
        "n_groups": len(ordered),
        "n_exact_dup_links": n_exact,
        "n_family_links": n_family_links,
        "n_near_dup_links": n_near,
        "n_forced_links": n_extra,
        "largest_group": int(sizes.max()) if len(sizes) else 0,
        "n_singleton_groups": int((sizes == 1).sum()),
        "group_size_histogram": {int(k): int(v) for k, v in
                                 zip(*np.unique(sizes, return_counts=True))},
        "near_pairs_sample": near_pairs[:50],
        "stage_components": n_after,
        "stage_links": {"exact": n_exact, "family": n_family_links,
                        "near": n_near, "forced": n_extra},
    }
    return mapping, diagnostics
