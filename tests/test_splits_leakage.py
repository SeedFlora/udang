"""Split integrity: grouping, and the hard no-overlap guarantee.

These are the tests that protect every number in the final report. If an
augmentation family or a shrimp individual can straddle train and test, the
reported metrics are inflated and the whole experiment is void.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.audit.checks import check_sdibd_individual_names
from src.audit.dedup import (
    UnionFind, build_fused_groups, is_derivative, near_duplicate_pairs,
    normalize_original_stem,
)
from src.data.manifest import forced_individual_links
from src.data.splits import (
    BIOMASS_SITES, SPLIT_COLUMNS, leakage_report, make_biomass_lodo, make_biomass_loso,
    make_biomass_within_db, make_cross_dataset, make_disease_cv,
    make_disease_cv_withaug, make_disease_naive, naive_test_composition,
    verify_no_leakage,
)


class TestUnionFind:
    def test_transitive_closure(self):
        uf = UnionFind()
        uf.union("a", "b")
        uf.union("b", "c")
        uf.union("x", "y")
        g = uf.groups()
        sizes = sorted(len(v) for v in g.values())
        assert sizes == [2, 3]
        assert uf.find("a") == uf.find("c")
        assert uf.find("a") != uf.find("x")


class TestAugmentationFamilyParsing:
    @pytest.mark.parametrize("name,expected", [
        ("aug_0_IMG-20240414-WA0014.jpg", "IMG-20240414-WA0014"),
        ("aug_12_IMG-20240414-WA0014.jpg", "IMG-20240414-WA0014"),
        ("aug_IMG-20240624-WA0005.jpg", "IMG-20240624-WA0005"),
        ("IMG-20240414-WA0014.jpg", "IMG-20240414-WA0014"),
        ("IMG-20240414-WA0014-resized.jpg", "IMG-20240414-WA0014"),
        ("aug_3_IMG-1234_resized.jpg", "IMG-1234"),
        ("IMG-9 (1).jpg", "IMG-9"),
    ])
    def test_normalisation_recovers_original_stem(self, name, expected):
        assert normalize_original_stem(name)[0] == expected

    def test_derivative_detection(self):
        assert is_derivative("aug_0_IMG-1.jpg")
        assert is_derivative("IMG-1-resized.jpg")
        assert not is_derivative("IMG-1.jpg")


class TestFusedGrouping:
    def test_exact_duplicates_fuse(self):
        m, d = build_fused_groups(["a", "b", "c"], ["s1", "s1", "s2"], None, [None] * 3)
        assert m["a"] == m["b"] != m["c"]
        assert d["n_exact_dup_links"] == 1

    def test_family_keys_fuse(self):
        m, _ = build_fused_groups(["a", "b", "c"], ["s1", "s2", "s3"], None,
                                  ["fam1", "fam1", "fam2"])
        assert m["a"] == m["b"] != m["c"]

    def test_near_duplicates_blocked_by_block_key(self):
        """Identical hashes in DIFFERENT blocks must NOT fuse."""
        h = np.array([np.uint64(0), np.uint64(0), np.uint64(0)], dtype=np.uint64)
        blocks = np.array(["A", "A", "B"])
        m, _ = build_fused_groups(["a", "b", "c"], ["s1", "s2", "s3"], h, [None] * 3,
                                  near_dup_blocks=blocks)
        assert m["a"] == m["b"], "same-block identical hashes should fuse"
        assert m["c"] != m["a"], "cross-block identical hashes must NOT fuse"

    def test_forced_links(self):
        m, d = build_fused_groups(["a", "b", "c"], ["s1", "s2", "s3"], None, [None] * 3,
                                  extra_links=[("a", "b")])
        assert m["a"] == m["b"] != m["c"]
        assert d["n_forced_links"] == 1

    def test_near_duplicate_distance(self):
        h = np.array([np.uint64(0b0), np.uint64(0b111), np.uint64((1 << 40) - 1)],
                     dtype=np.uint64)
        pairs = near_duplicate_pairs(h, threshold=3)
        assert (0, 1, 3) in pairs
        assert not any(p[:2] == (0, 2) for p in pairs)


class TestDiseaseSplits:
    def test_no_group_crosses_splits(self, synthetic_manifest):
        s = make_disease_cv(synthetic_manifest, n_splits=3, seed=42)
        verify_no_leakage(s)          # raises on violation

    def test_augmentations_stay_with_their_original(self, synthetic_manifest):
        """The ablation protocol includes published augmentations; each must land in
        the same fold as the original it was derived from."""
        s = make_disease_cv(synthetic_manifest, n_splits=3, seed=42,
                            include_published_aug=True,
                            protocol_name="disease_cv_withaug")
        verify_no_leakage(s)
        for fold, g in s.groupby("fold"):
            for gid, gg in g.groupby("group_id"):
                assert gg["split"].nunique() == 1, (
                    f"group {gid} spans splits {set(gg['split'])} in fold {fold}")

    def test_primary_protocol_excludes_published_augmentations(self, synthetic_manifest):
        s = make_disease_cv(synthetic_manifest, n_splits=3, seed=42)
        ids = set(s["record_id"])
        aug_ids = set(synthetic_manifest[
            synthetic_manifest["is_published_augmentation"].astype(bool)]["record_id"])
        assert not (ids & aug_ids), "primary protocol must use original images only"

    def test_every_record_appears_once_per_fold(self, synthetic_manifest):
        s = make_disease_cv(synthetic_manifest, n_splits=3, seed=42)
        for _fold, g in s.groupby("fold"):
            assert not g["record_id"].duplicated().any()

    def test_all_three_splits_present(self, synthetic_manifest):
        s = make_disease_cv(synthetic_manifest, n_splits=3, seed=42)
        for _fold, g in s.groupby("fold"):
            assert set(g["split"]) == {"train", "val", "test"}

    def test_deterministic_given_seed(self, synthetic_manifest):
        a = make_disease_cv(synthetic_manifest, n_splits=3, seed=7)
        b = make_disease_cv(synthetic_manifest, n_splits=3, seed=7)
        assert a.equals(b)


class TestAugmentationAblationIsControlled:
    """The published-augmentation ablation must change exactly one variable.

    Re-running CV over the augmented pool would put synthetic images into TEST,
    so the ablation would be scored on different data from its baseline and the
    comparison would confound three changes at once.
    """

    @pytest.fixture
    def pair(self, synthetic_manifest):
        base = make_disease_cv(synthetic_manifest, n_splits=3, seed=42)
        ab = make_disease_cv_withaug(synthetic_manifest, base)
        return synthetic_manifest, base, ab

    def test_test_sets_are_identical(self, pair):
        _m, base, ab = pair
        for fold in sorted(base["fold"].unique()):
            a = set(base[(base["fold"] == fold) & (base["split"] == "test")]["record_id"])
            b = set(ab[(ab["fold"] == fold) & (ab["split"] == "test")]["record_id"])
            assert a == b, f"fold {fold}: test sets differ by {len(a ^ b)} records"

    def test_val_sets_are_identical(self, pair):
        _m, base, ab = pair
        for fold in sorted(base["fold"].unique()):
            a = set(base[(base["fold"] == fold) & (base["split"] == "val")]["record_id"])
            b = set(ab[(ab["fold"] == fold) & (ab["split"] == "val")]["record_id"])
            assert a == b

    def test_no_augmented_image_in_val_or_test(self, pair):
        m, _base, ab = pair
        idx = m.set_index("record_id")["is_published_augmentation"].astype(bool)
        for split in ("val", "test"):
            rec = ab[ab["split"] == split]["record_id"]
            assert not idx.loc[rec].any(), \
                f"{split} contains published augmented copies"

    def test_train_actually_gains_augmentations(self, pair):
        m, base, ab = pair
        idx = m.set_index("record_id")["is_published_augmentation"].astype(bool)
        for fold in sorted(base["fold"].unique()):
            tb = base[(base["fold"] == fold) & (base["split"] == "train")]["record_id"]
            ta = ab[(ab["fold"] == fold) & (ab["split"] == "train")]["record_id"]
            assert idx.loc[tb].sum() == 0
            assert idx.loc[ta].sum() > 0, "ablation train must contain augmentations"
            assert len(ta) > len(tb)

    def test_still_leakage_free(self, pair):
        m, _base, ab = pair
        verify_no_leakage(ab, m)


class TestManifestCrossCheck:
    """verify_no_leakage must not be circular: a protocol that writes a WRONG
    grouping key would pass a self-consistency check while leaking badly."""

    def test_detects_wrong_grouping_key(self, synthetic_manifest):
        """Simulate a protocol that grouped per-record instead of per-family: an
        augmented copy ends up in test while its source original is in train."""
        s = make_disease_cv(synthetic_manifest, n_splits=3, seed=42,
                            include_published_aug=True,
                            protocol_name="disease_cv_withaug")
        bad = s.copy()
        bad["group_id"] = bad["record_id"]      # per-record "grouping"

        # find a real family whose members currently share fold 0, then split it
        f0 = bad[bad["fold"] == 0]
        mg = synthetic_manifest.set_index("record_id")["group_id"]
        f0 = f0.assign(true_group=f0["record_id"].map(mg))
        fam = f0.groupby("true_group").filter(lambda g: len(g) > 1)
        assert len(fam), "fixture must contain a multi-member family"
        gid = fam["true_group"].iloc[0]
        members = bad[(bad["fold"] == 0) & (bad["record_id"].map(mg) == gid)]
        src = members["split"].iloc[0]
        bad.loc[members.index[0], "split"] = "test" if src != "test" else "train"

        # per-record self-consistency cannot see this: every group has one member
        rep = verify_no_leakage(bad, manifest=None)
        assert rep["violations"] == [], \
            "self-consistency should be blind to a per-record grouping key"

        # the manifest cross-check must catch the straddling family
        with pytest.raises(AssertionError, match="LEAKAGE DETECTED"):
            verify_no_leakage(bad, synthetic_manifest)


class TestSplitCompleteness:
    """Absence of overlap is not evidence of a usable split: an empty or truncated
    split trivially satisfies the no-overlap property."""

    def test_empty_test_split_is_rejected(self, synthetic_manifest):
        s = make_disease_cv(synthetic_manifest, n_splits=3, seed=42)
        bad = s[~((s["fold"] == 0) & (s["split"] == "test"))]
        with pytest.raises(AssertionError, match="LEAKAGE DETECTED"):
            verify_no_leakage(bad, synthetic_manifest)

    def test_empty_val_split_is_rejected(self, synthetic_manifest):
        s = make_disease_cv(synthetic_manifest, n_splits=3, seed=42)
        bad = s[~((s["fold"] == 1) & (s["split"] == "val"))]
        with pytest.raises(AssertionError, match="LEAKAGE DETECTED"):
            verify_no_leakage(bad, synthetic_manifest)

    def test_complete_splits_pass(self, synthetic_manifest):
        s = make_disease_cv(synthetic_manifest, n_splits=3, seed=42)
        rep = verify_no_leakage(s, synthetic_manifest)
        assert rep["violations"] == []


class TestCrossDataset:
    def test_train_and_test_are_different_datasets(self, synthetic_manifest):
        s = make_cross_dataset(synthetic_manifest, seed=42)
        for proto, g in s.groupby("protocol"):
            tr = set(g[g["split"].isin(["train", "val"])]["dataset"])
            te = set(g[g["split"] == "test"]["dataset"])
            assert tr and te and not (tr & te), f"{proto}: source and target overlap"

    def test_both_directions_exist(self, synthetic_manifest):
        s = make_cross_dataset(synthetic_manifest, seed=42)
        assert set(s["protocol"]) == {"cross_sd2tg", "cross_tg2sd"}

    def test_target_restricted_to_common_classes(self, synthetic_manifest):
        s = make_cross_dataset(synthetic_manifest, seed=42)
        m = synthetic_manifest.set_index("record_id")
        for proto, g in s.groupby("protocol"):
            te = g[g["split"] == "test"]
            classes = set(m.loc[te["record_id"], "raw_class"])
            assert "Yellow Head" not in classes
            assert "BG_WSSV" not in classes

    def test_no_leakage(self, synthetic_manifest):
        verify_no_leakage(make_cross_dataset(synthetic_manifest, seed=42))


class TestBiomassSplits:
    def test_individual_never_spans_splits(self, synthetic_manifest):
        s = make_biomass_lodo(synthetic_manifest, seed=42)
        verify_no_leakage(s)
        for fold, g in s.groupby("fold"):
            for ind, gg in g.groupby("group_id"):
                assert gg["split"].nunique() == 1, (
                    f"individual {ind} spans {set(gg['split'])} in fold {fold}")

    def test_leave_one_db_out(self, synthetic_manifest):
        s = make_biomass_lodo(synthetic_manifest, seed=42)
        m = synthetic_manifest.set_index("record_id")
        assert s["fold"].nunique() == 5
        for fold, g in s.groupby("fold"):
            te_db = set(m.loc[g[g["split"] == "test"]["record_id"], "pond_or_db_id"])
            tr_db = set(m.loc[g[g["split"] != "test"]["record_id"], "pond_or_db_id"])
            assert len(te_db) == 1, "test must be exactly one held-out DB"
            assert not (te_db & tr_db), "held-out DB leaked into training"

    def test_validation_comes_from_training_dbs_only(self, synthetic_manifest):
        s = make_biomass_lodo(synthetic_manifest, seed=42)
        m = synthetic_manifest.set_index("record_id")
        for _fold, g in s.groupby("fold"):
            te_db = set(m.loc[g[g["split"] == "test"]["record_id"], "pond_or_db_id"])
            va_db = set(m.loc[g[g["split"] == "val"]["record_id"], "pond_or_db_id"])
            assert not (va_db & te_db)

    def test_all_views_of_individual_together(self, synthetic_manifest):
        s = make_biomass_lodo(synthetic_manifest, seed=42)
        m = synthetic_manifest.set_index("record_id")
        for _fold, g in s.groupby("fold"):
            tmp = g.copy()
            tmp["ind"] = m.loc[tmp["record_id"], "individual_id"].values
            per = tmp.groupby("ind")["split"].nunique()
            assert (per == 1).all()


def test_verify_no_leakage_actually_detects_leakage(synthetic_manifest):
    """A guard that never fires is worthless -- prove it fires.

    Uses the with-augmentation protocol, where a group holds an original plus its
    two published copies. Moving ONE member to another split is exactly the
    real-world failure mode (an augmented copy landing opposite its original).
    """
    s = make_disease_cv(synthetic_manifest, n_splits=3, seed=42,
                        include_published_aug=True,
                        protocol_name="disease_cv_withaug")
    bad = s.copy()
    f0 = bad[bad["fold"] == 0]
    multi = f0.groupby("group_id").filter(lambda g: len(g) > 1)
    assert len(multi), "fixture must contain a multi-record group to exercise the guard"
    gid = multi["group_id"].iloc[0]
    members = bad[(bad["fold"] == 0) & (bad["group_id"] == gid)]
    src_split = members["split"].iloc[0]
    other = "test" if src_split != "test" else "train"
    bad.loc[members.index[0], "split"] = other
    with pytest.raises(AssertionError, match="LEAKAGE DETECTED"):
        verify_no_leakage(bad)


def test_biomass_leakage_guard_detects_split_individual(synthetic_manifest):
    """The biomass analogue: one shrimp's views split across train and test."""
    s = make_biomass_lodo(synthetic_manifest, seed=42)
    bad = s.copy()
    f0 = bad[bad["fold"] == 0]
    tr = f0[f0["split"] == "train"]
    gid = tr["group_id"].iloc[0]
    members = bad[(bad["fold"] == 0) & (bad["group_id"] == gid)]
    bad.loc[members.index[0], "split"] = "test"
    with pytest.raises(AssertionError, match="LEAKAGE DETECTED"):
        verify_no_leakage(bad)


class TestForcedIndividualLinks:
    def test_links_every_record_sharing_an_individual_id(self):
        df = pd.DataFrame({
            "record_id": ["a", "b", "c", "d", "e"],
            "individual_id": ["sdibd::BG::001", "sdibd::BG::001", None,
                              "DB1_s003", "DB1_s003"],
        })
        links = forced_individual_links(df)
        assert sorted(links) == [("a", "b"), ("d", "e")]

    def test_no_individual_ids_means_no_links(self):
        df = pd.DataFrame({"record_id": ["a", "b"], "individual_id": [None, None]})
        assert forced_individual_links(df) == []


class TestFunnelDiagnostics:
    def test_stage_components_are_monotone_and_terminal_matches_groups(self):
        # 6 records: a,b exact dups; b,c same family; d,e forced; f alone
        m, d = build_fused_groups(
            record_ids=list("abcdef"),
            sha256s=["s1", "s1", "s2", "s3", "s4", "s5"],
            phash_u64=None,
            family_keys=[None, "fam", "fam", None, None, None],
            extra_links=[("d", "e")],
        )
        sc = d["stage_components"]
        assert sc["seeded"] == 6
        assert sc["after_exact_duplicates"] == 5
        assert sc["after_augmentation_families"] == 4
        assert sc["after_perceptual_near_duplicates"] == 4
        assert sc["after_forced_individuals"] == 3 == d["n_groups"]
        assert d["stage_links"] == {"exact": 1, "family": 1, "near": 0, "forced": 1}
        assert len(set(m.values())) == 3


class TestSdibdNameAudit:
    def test_all_parsed_and_matched_is_clean(self):
        diag = {"ShrimpDiseaseImageBD": {"unparsed_names": [], "class_token_mismatch": []}}
        checks = check_sdibd_individual_names(diag)
        by_name = {c["check"]: c for c in checks}
        assert by_name["sdibd_unparsable_image_names"]["status"] == "pass"
        assert by_name["sdibd_class_token_mismatch"]["status"] == "pass"

    def test_unparsed_name_fails_leakage_critical(self):
        diag = {"ShrimpDiseaseImageBD": {"unparsed_names": ["Raw Images/1. Healthy/IMG_0001.jpg"],
                                         "class_token_mismatch": []}}
        checks = check_sdibd_individual_names(diag)
        by_name = {c["check"]: c for c in checks}
        c = by_name["sdibd_unparsable_image_names"]
        assert c["status"] == "fail"
        assert c["severity"] == "leakage_critical"
        assert c["count"] == 1

    def test_class_token_mismatch_warns(self):
        diag = {"ShrimpDiseaseImageBD": {
            "unparsed_names": [],
            "class_token_mismatch": ["Raw Images/1. Healthy/BG-3-img-1.jpg"],
        }}
        checks = check_sdibd_individual_names(diag)
        by_name = {c["check"]: c for c in checks}
        assert by_name["sdibd_class_token_mismatch"]["status"] == "warn"


class TestLeakageReportIsNonRaisingCore:
    def test_report_lists_violations_without_raising(self, synthetic_manifest):
        s = make_biomass_lodo(synthetic_manifest, seed=42)
        bad = s.copy()
        gid = bad[(bad["fold"] == 0) & (bad["split"] == "train")]["group_id"].iloc[0]
        idx = bad[(bad["fold"] == 0) & (bad["group_id"] == gid)].index[0]
        bad.loc[idx, "split"] = "test"
        rep = leakage_report(bad)
        assert rep["violations"], "report must list the straddling group"
        with pytest.raises(AssertionError, match="LEAKAGE DETECTED"):
            verify_no_leakage(bad)


class TestLeaveOneSiteOut:
    def test_three_folds_one_site_each_and_individuals_intact(self, synthetic_manifest):
        s = make_biomass_loso(synthetic_manifest, seed=42)
        verify_no_leakage(s, synthetic_manifest)
        assert s["fold"].nunique() == 3
        m = synthetic_manifest.set_index("record_id")
        for fold, g in s.groupby("fold"):
            te_sites = {BIOMASS_SITES[d] for d in m.loc[g[g["split"] == "test"]["record_id"], "pond_or_db_id"]}
            tr_sites = {BIOMASS_SITES[d] for d in m.loc[g[g["split"] != "test"]["record_id"], "pond_or_db_id"]}
            assert len(te_sites) == 1 and not (te_sites & tr_sites)

    def test_db2_and_db3_are_held_out_together(self, synthetic_manifest):
        s = make_biomass_loso(synthetic_manifest, seed=42)
        m = synthetic_manifest.set_index("record_id")
        held = [set(m.loc[g[g["split"] == "test"]["record_id"], "pond_or_db_id"])
                for _f, g in s.groupby("fold")]
        assert {"DB2", "DB3"} in held and {"DB4", "DB5"} in held and {"DB1"} in held

    def test_lodo_is_deterministic_and_five_fold(self, synthetic_manifest):
        """make_biomass_lodo now delegates to a shared helper; its assignment must
        be byte-identical to before (kept runs depend on it)."""
        s = make_biomass_lodo(synthetic_manifest, seed=42)
        assert s["fold"].nunique() == 5
        assert set(s["protocol"]) == {"biomass_lodo"}
        # deterministic and every individual intact
        assert s.equals(make_biomass_lodo(synthetic_manifest, seed=42))

    def test_lodo_assignment_is_pinned(self, synthetic_manifest):
        """Golden hash of make_biomass_lodo on the fixture. Twelve trained runs from
        round 1 depend on this assignment being byte-identical; any refactor that
        moves the rng construction or reorders the per-DB permutation loop changes
        this value and must be a conscious decision, not an accident."""
        from src.utils.hashing import hash_dataframe
        s = make_biomass_lodo(synthetic_manifest, seed=42)
        assert hash_dataframe(s, SPLIT_COLUMNS) == \
            "ce54fdda6af599fc43b4ea4df4832fabbbee655150c7db861a776fd50790e95d"


class TestWithinDatabaseCV:
    def test_every_db_appears_in_train_and_test_of_every_fold(self, synthetic_manifest):
        s = make_biomass_within_db(synthetic_manifest, n_splits=5, seed=42)
        verify_no_leakage(s, synthetic_manifest)
        m = synthetic_manifest.set_index("record_id")
        for _fold, g in s.groupby("fold"):
            te = set(m.loc[g[g["split"] == "test"]["record_id"], "pond_or_db_id"])
            tr = set(m.loc[g[g["split"] == "train"]["record_id"], "pond_or_db_id"])
            assert te == tr == {"DB1", "DB2", "DB3", "DB4", "DB5"}

    def test_each_individual_is_tested_exactly_once(self, synthetic_manifest):
        s = make_biomass_within_db(synthetic_manifest, n_splits=5, seed=42)
        te = s[s["split"] == "test"]
        per_ind = te.groupby("group_id")["fold"].nunique()
        assert (per_ind == 1).all()
        assert set(te["group_id"]) == set(s["group_id"])


class TestNaiveControlArm:
    def test_is_measurably_leaky_and_would_fail_the_gate(self, synthetic_manifest):
        n = make_disease_naive(synthetic_manifest, n_splits=3, seed=42)
        assert (n["group_id"] == n["record_id"]).all()
        assert leakage_report(n)["violations"] == [], "self-consistency is blind by design"
        rep = leakage_report(n, synthetic_manifest)
        assert rep["violations"], "manifest cross-check must catch the naive arm"
        with pytest.raises(AssertionError, match="LEAKAGE DETECTED"):
            verify_no_leakage(n, synthetic_manifest)

    def test_composition_reports_synthetic_and_leaked_fractions(self, synthetic_manifest):
        n = make_disease_naive(synthetic_manifest, n_splits=3, seed=42)
        comp = naive_test_composition(n, synthetic_manifest)
        assert len(comp) == 3
        for c in comp:
            assert 0.0 < c["frac_test_synthetic"] < 1.0
            assert 0.0 < c["frac_test_group_leaked"] <= 1.0
            assert c["n_test"] > 0


class TestSdibdAnimalsInSplits:
    """The fixture (Step 3b) now gives every synthetic SDIBD record a class-scoped
    animal id with three views per animal; these tests exercise the path the
    submitted grouping never exercised."""

    def test_all_views_of_an_animal_share_a_split(self, synthetic_manifest):
        s = make_disease_cv(synthetic_manifest, n_splits=3, seed=42)
        animal = synthetic_manifest.set_index("record_id")["individual_id"]
        for fold, g in s.groupby("fold"):
            sd = g[g["dataset"] == "ShrimpDiseaseImageBD"].copy()
            sd["animal"] = sd["record_id"].map(animal)
            assert (sd.groupby("animal")["split"].nunique() == 1).all(), fold

    def test_manifest_cross_check_catches_a_straddling_animal(self, synthetic_manifest):
        s = make_disease_cv(synthetic_manifest, n_splits=3, seed=42)
        bad = s.copy()
        bad["group_id"] = bad["record_id"]            # a protocol that forgot animals
        animal = synthetic_manifest.set_index("record_id")["individual_id"]
        f0 = bad[(bad["fold"] == 0) & (bad["dataset"] == "ShrimpDiseaseImageBD")]
        target = animal.loc[f0["record_id"].iloc[0]]
        members = f0[f0["record_id"].map(animal) == target]
        assert len(members) >= 2, "fixture must give an animal several views"
        bad.loc[members.index[0], "split"] = "test" if members["split"].iloc[0] != "test" else "train"
        assert leakage_report(bad)["violations"] == [], "self-consistency is blind by design"
        with pytest.raises(AssertionError, match="LEAKAGE DETECTED"):
            verify_no_leakage(bad, synthetic_manifest)
