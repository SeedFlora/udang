"""Tests for the round-1 revision tooling (overlap measurement, probes, archive)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]


def _load(script: str):
    spec = importlib.util.spec_from_file_location(script, REPO / "scripts" / f"{script}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[script] = mod
    spec.loader.exec_module(mod)
    return mod


import subprocess  # noqa: E402


def _git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)


class TestGitDirtyScope:
    @pytest.fixture
    def repo(self, tmp_path):
        _git(tmp_path, "init", "-q")
        _git(tmp_path, "config", "user.email", "t@t")
        _git(tmp_path, "config", "user.name", "t")
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "a.py").write_text("x = 1\n")
        _git(tmp_path, "add", "src/a.py")
        _git(tmp_path, "commit", "-q", "-m", "init")
        return tmp_path

    def test_run_artefacts_do_not_make_the_code_dirty(self, repo):
        from src.utils.runmeta import git_dirty
        (repo / "outputs").mkdir()
        (repo / "outputs" / "run_meta.json").write_text("{}")          # untracked artefact
        assert git_dirty(cwd=repo) is False

    def test_modified_code_is_dirty(self, repo):
        from src.utils.runmeta import git_dirty
        (repo / "src" / "a.py").write_text("x = 2\n")
        assert git_dirty(cwd=repo) is True

    def test_untracked_code_is_dirty(self, repo):
        from src.utils.runmeta import git_dirty
        (repo / "src" / "b.py").write_text("y = 1\n")
        assert git_dirty(cwd=repo) is True


class TestIndividualOverlap:
    @pytest.fixture
    def man(self):
        return pd.DataFrame({
            "record_id": ["SD-1", "SD-2", "SD-3", "SD-4", "TG-1"],
            "dataset": ["ShrimpDiseaseImageBD"] * 4 + ["TigerShrimpBD"],
            "raw_class": ["BG", "BG", "Healthy", "Healthy", "WSSV"],
            "original_filename": ["BG-7-img-1.jpg", "BG-7-img-2.jpg",
                                  "Healthy-7-img-1.jpg", "Healthy-9-img-1.jpg",
                                  "IMG-20240414-WA0014.jpg"],
        })

    def test_shared_animal_across_train_and_test_is_counted(self, man):
        mod = _load("measure_individual_overlap")
        spl = pd.DataFrame({
            "record_id": ["SD-1", "SD-2", "SD-3", "SD-4", "TG-1"],
            "protocol": ["p"] * 5, "fold": [0] * 5,
            "split": ["train", "test", "test", "train", "test"],
        })
        t = mod.overlap_table(spl, man)
        row = t.iloc[0]
        # test = SD-2 (animal BG-7, also in train), SD-3 (Healthy-7, not in train:
        # Healthy-7 != BG-7), TG-1 (not identifiable)
        assert row["n_test"] == 3
        assert row["n_test_identifiable"] == 2
        assert row["n_test_shared_animal"] == 1
        assert row["frac_shared"] == pytest.approx(0.5)

    def test_clean_split_reports_zero(self, man):
        mod = _load("measure_individual_overlap")
        spl = pd.DataFrame({
            "record_id": ["SD-1", "SD-2", "SD-3", "SD-4"],
            "protocol": ["p"] * 4, "fold": [0] * 4,
            "split": ["test", "test", "train", "train"],
        })
        t = mod.overlap_table(spl, man)
        assert t.iloc[0]["n_test_shared_animal"] == 0
        assert t.iloc[0]["frac_shared"] == 0.0

    def test_animal_in_val_counts_as_shared(self, man):
        mod = _load("measure_individual_overlap")
        spl = pd.DataFrame({
            "record_id": ["SD-1", "SD-2", "SD-3", "SD-4"],
            "protocol": ["p"] * 4, "fold": [0] * 4,
            "split": ["val", "test", "train", "train"],
        })
        t = mod.overlap_table(spl, man)
        # SD-1 (animal BG-7) sits in val, not train; SD-2 (same animal) is test.
        assert t.iloc[0]["n_test_shared_animal"] == 1

    def test_unmatched_record_is_counted_not_hidden(self, man):
        mod = _load("measure_individual_overlap")
        spl = pd.DataFrame({
            "record_id": ["SD-1", "SD-4", "SD-99"],
            "protocol": ["p"] * 3, "fold": [0] * 3,
            "split": ["train", "train", "test"],
        })
        t = mod.overlap_table(spl, man)
        row = t.iloc[0]
        # SD-99 is absent from the manifest entirely -- must be counted as
        # unmatched, not silently folded into "not identifiable".
        assert row["n_test_unmatched"] == 1
        assert row["n_test_identifiable"] == 0

    def test_no_identifiable_images_reports_none(self, man):
        mod = _load("measure_individual_overlap")
        spl = pd.DataFrame({
            "record_id": ["SD-1", "TG-1"],
            "protocol": ["p"] * 2, "fold": [0] * 2,
            "split": ["train", "test"],
        })
        t = mod.overlap_table(spl, man)
        # test = TG-1 only, which is not identifiable -> nothing measurable.
        assert t.iloc[0]["frac_shared"] is None


from src.training.registry import PROFILES, build_matrix  # noqa: E402


class TestRevisionRegistry:
    def test_leveled_profiles_only_change_fold_counts(self):
        std = PROFILES["standard"]
        for name, folds in (("standard3", 3), ("standard5", 5)):
            p = PROFILES[name]
            assert (p["folds"], p["ablation_folds"]) == (folds, folds)
            assert (p["epochs"], p["patience"], p["image_size"], p["limit"]) == \
                   (std["epochs"], std["patience"], std["image_size"], std["limit"])

    def test_new_experiments_exist_with_correct_protocols(self):
        cfgs = {c.name: c for c in build_matrix("standard3", seed=42)}
        for bb in ("resnet50", "efficientnet_b0", "swin_t"):
            for d, proto in (("sd2tg", "cross_sd2tg"), ("tg2sd", "cross_tg2sd")):
                c = cfgs[f"E10_{bb}_{d}"]
                assert (c.experiment, c.protocol, c.folds, c.model.backbone) == ("E10", proto, (0,), bb)
                assert c.train.attr_loss == "bce" and c.tasks == ("disease",)
        n = cfgs["E11_naive_withaug"]
        assert (n.experiment, n.protocol, n.folds) == ("E11", "disease_cv_naive_withaug", (0, 1, 2))

    def test_biomass_configs_set_both_protocol_fields(self):
        cfgs = {c.name: c for c in build_matrix("standard5", seed=42)}
        loso, within = cfgs["E12_biomass_loso"], cfgs["E13_biomass_within_db"]
        assert loso.protocol == loso.biomass_protocol == "biomass_loso" and loso.folds == (0, 1, 2)
        assert within.protocol == within.biomass_protocol == "biomass_within_db"
        assert within.folds == (0, 1, 2, 3, 4)
        assert cfgs["E4_biomass_single_image"].folds == (0, 1, 2, 3, 4)
        # E5 and E8-biomass use _folds(AF); under standard5 AF == 5 too
        for name in ("E5_biomass_multiview_mean", "E5_biomass_multiview_attention",
                     "E8_lightweight_biomass"):
            assert cfgs[name].folds == (0, 1, 2, 3, 4), name
        for c in cfgs.values():
            if "biomass" in c.tasks and "disease" not in c.tasks:
                assert c.protocol == c.biomass_protocol, c.name

    def test_swin_config_is_created_at_training_resolution(self):
        c = {c.name: c for c in build_matrix("standard3", seed=42)}["E10_swin_t_sd2tg"]
        assert c.model.img_size == c.image_size == 320

    def test_disease_grid_is_three_folds_under_standard3(self):
        for c in build_matrix("standard3", seed=42):
            if "disease" in c.tasks and not c.protocol.startswith("cross"):
                assert c.folds == (0, 1, 2), c.name


class TestSupersededMask:
    """A run is superseded only if it was trained on a *different* grouping than
    the one the splits currently carry -- otherwise the B->C interlock flags the
    freshly completed Phase B runs as file-grouping-era artefacts."""

    NEW = "n" * 64
    OLD = "o" * 64

    def _reg(self) -> pd.DataFrame:
        return pd.DataFrame({
            "run_name": ["old_cv", "new_cv", "old_cross", "new_cross",
                         "E9_new_cv", "new_tiger", "old_tiger", "nan_cv"],
            "protocol": ["disease_cv", "disease_cv", "cross_sd2tg", "cross_sd2tg",
                         "disease_cv", "disease_cv_tiger", "disease_cv_tiger", "disease_cv"],
            "split_hash": [self.OLD, self.NEW, self.OLD, self.NEW,
                           self.NEW, self.NEW, self.OLD, float("nan")],
        })

    def test_only_old_hash_rows_of_a_superseded_protocol_are_masked(self):
        from src.training.superseded import superseded_mask
        reg = self._reg()
        got = set(reg.loc[superseded_mask(reg, current_split_hash=self.NEW), "run_name"])
        assert got == {"old_cv", "old_cross", "E9_new_cv", "nan_cv"}

    def test_retired_prefix_is_masked_even_on_the_current_hash(self):
        from src.training.superseded import superseded_mask
        reg = self._reg()
        m = superseded_mask(reg, current_split_hash=self.NEW)
        assert bool(m[reg["run_name"] == "E9_new_cv"].iloc[0]) is True

    def test_missing_split_hash_counts_as_superseded(self):
        from src.training.superseded import superseded_mask
        reg = self._reg().drop(columns=["split_hash"])
        m = superseded_mask(reg, current_split_hash=self.NEW)
        # every superseded-protocol row now has an unknown hash
        assert set(reg.loc[m, "run_name"]) == {"old_cv", "new_cv", "old_cross",
                                               "new_cross", "E9_new_cv", "nan_cv"}

    def test_unsuperseded_protocol_is_never_masked_by_hash_alone(self):
        from src.training.superseded import superseded_mask
        reg = self._reg()
        m = superseded_mask(reg, current_split_hash=self.NEW)
        assert not bool(m[reg["run_name"] == "old_tiger"].iloc[0])

    def test_current_hash_is_read_from_the_splits_file_when_not_given(self, tmp_path, monkeypatch):
        import src.training.superseded as sup
        f = tmp_path / "splits_hash.txt"
        f.write_text(self.NEW + "\n", encoding="utf-8")     # trailing newline must not matter
        monkeypatch.setattr(sup, "SPLITS_HASH_FILE", f)
        reg = self._reg()
        assert set(reg.loc[sup.superseded_mask(reg), "run_name"]) == \
               {"old_cv", "old_cross", "E9_new_cv", "nan_cv"}

    def test_absent_hash_file_falls_back_to_protocol_and_prefix(self, tmp_path, monkeypatch):
        import src.training.superseded as sup
        monkeypatch.setattr(sup, "SPLITS_HASH_FILE", tmp_path / "nope.txt")
        reg = self._reg()
        # nothing to compare against -> every superseded-protocol row is suspect
        assert set(reg.loc[sup.superseded_mask(reg), "run_name"]) == \
               {"old_cv", "new_cv", "old_cross", "new_cross", "E9_new_cv", "nan_cv"}

    def test_live_registry_has_no_stale_rows(self):
        """The real guard: after the Phase B rebuild every superseded-protocol row
        carries the current split hash, so the interlock must select nothing."""
        from src.training.superseded import superseded_mask
        reg = pd.read_csv(REPO / "outputs" / "metrics" / "experiment_registry.csv")
        assert superseded_mask(reg).sum() == 0


class TestArchiveSelection:
    def test_selects_by_protocol_and_prefix_only(self):
        mod = _load("archive_superseded_runs")
        reg = pd.DataFrame({
            "run_name": ["E1_disease_single_tiger_f0_s42", "E1_disease_single_sdibd_f0_s42",
                         "E9_E1_disease_single_tiger_f0_s2026", "E4_biomass_single_image_f0_s42"],
            "protocol": ["disease_cv_tiger", "disease_cv_sdibd", "disease_cv_tiger", "biomass_lodo"],
            "checkpoint_path": ["outputs/checkpoints/a_best.pt"] * 4,
            "predictions_path": ["outputs/predictions/a_test.parquet"] * 4,
            "metrics_path": ["outputs/metrics/a.json"] * 4,
        })
        sel = mod.select_superseded(reg, {"disease_cv_sdibd"}, ("E9_",))
        assert sorted(sel["run_name"]) == ["E1_disease_single_sdibd_f0_s42",
                                          "E9_E1_disease_single_tiger_f0_s2026"]

    def test_artefact_paths_cover_every_per_run_file(self):
        mod = _load("archive_superseded_runs")
        row = pd.Series({"checkpoint_path": "outputs/checkpoints/r_best.pt",
                         "predictions_path": "outputs/predictions/r_test.parquet;outputs/predictions/r_biomass_test.parquet",
                         "metrics_path": "outputs/metrics/r.json"})
        got = {p.as_posix() for p in mod.artefact_paths(row, "r")}
        assert got == {
            "outputs/checkpoints/r_best.pt", "outputs/checkpoints/r_history.json",
            "outputs/predictions/r_test.parquet", "outputs/predictions/r_biomass_test.parquet",
            "outputs/predictions/r_val.parquet", "outputs/metrics/r.json",
            "outputs/logs/r_meta.json", "outputs/logs/r_FAILED.log",
        }

    def test_artefact_paths_tolerates_nan_cells(self):
        mod = _load("archive_superseded_runs")
        row = pd.Series({"checkpoint_path": float("nan"), "predictions_path": float("nan"),
                         "metrics_path": "outputs/metrics/r.json"})
        got = {p.as_posix() for p in mod.artefact_paths(row, "r")}
        assert got == {
            "outputs/checkpoints/r_history.json", "outputs/predictions/r_val.parquet",
            "outputs/logs/r_meta.json", "outputs/logs/r_FAILED.log", "outputs/metrics/r.json",
        }


class TestArchiveMainDryRun:
    def test_dry_run_does_not_touch_disk_or_registry(self, tmp_path, monkeypatch):
        mod = _load("archive_superseded_runs")
        registry = tmp_path / "experiment_registry.csv"
        fold_metrics = tmp_path / "fold_metrics.csv"
        monkeypatch.setattr(mod, "REGISTRY", registry)
        monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
        monkeypatch.setattr(mod, "FOLD_METRICS", fold_metrics)

        (tmp_path / "outputs" / "checkpoints").mkdir(parents=True)
        (tmp_path / "outputs" / "predictions").mkdir(parents=True)
        (tmp_path / "outputs" / "metrics").mkdir(parents=True)
        ckpt = tmp_path / "outputs" / "checkpoints" / "S_f0_s2026_best.pt"
        pred = tmp_path / "outputs" / "predictions" / "S_f0_s2026_test.parquet"
        met = tmp_path / "outputs" / "metrics" / "S_f0_s2026.json"
        for f in (ckpt, pred, met):
            f.write_bytes(b"x")

        reg_df = pd.DataFrame({
            "run_name": ["S_f0_s2026", "K1_f0_s42", "K2_f0_s42"],
            "protocol": ["disease_cv", "disease_cv_tiger", "biomass_lodo"],
            "checkpoint_path": ["outputs/checkpoints/S_f0_s2026_best.pt", "", ""],
            "predictions_path": ["outputs/predictions/S_f0_s2026_test.parquet", "", ""],
            "metrics_path": ["outputs/metrics/S_f0_s2026.json", "", ""],
        })
        reg_df.to_csv(registry, index=False)
        fold_metrics.write_text("run_name,status\nS_f0_s2026,completed\n")
        before = registry.read_bytes()

        monkeypatch.setattr(sys, "argv",
                            ["archive_superseded_runs.py", "--dry-run", "--delete-checkpoints"])
        rc = mod.main()

        assert rc == 0
        assert ckpt.exists() and pred.exists() and met.exists()
        assert registry.read_bytes() == before
        assert not (tmp_path / "outputs" / "_archive").exists()


class TestRunRevisionPlan:
    def test_phase_b_enumerates_seed42_disease_grid(self, tmp_path, monkeypatch):
        mod = _load("run_revision")
        monkeypatch.setattr(mod, "CHECKPOINTS", tmp_path)
        monkeypatch.setattr(mod, "METRICS", tmp_path)
        monkeypatch.setattr(mod, "_family_seconds", lambda: {})
        monkeypatch.setattr(mod, "_split_pairs", lambda: {
            ("disease_cv", 0), ("disease_cv", 1), ("disease_cv", 2),
            ("disease_cv_withaug", 0), ("disease_cv_withaug", 1), ("disease_cv_withaug", 2),
            ("disease_cv_sdibd", 0), ("disease_cv_sdibd", 1), ("disease_cv_sdibd", 2),
            ("disease_cv_tiger", 0), ("disease_cv_tiger", 1), ("disease_cv_tiger", 2),
            ("cross_sd2tg", 0), ("cross_tg2sd", 0),
            ("disease_cv_naive_withaug", 0), ("disease_cv_naive_withaug", 1), ("disease_cv_naive_withaug", 2),
            ("biomass_lodo", 0), ("biomass_lodo", 1), ("biomass_lodo", 2),
        })
        df = mod.plan_runs("B")
        assert set(df["seed"]) == {42}
        assert "E1_disease_single_tiger_f2_s42" in set(df["run_name"])
        assert "E11_naive_withaug_f2_s42" in set(df["run_name"])
        assert not df["run_name"].str.startswith("E9_").any()
        assert (df["exists"] == False).all()          # noqa: E712  (empty tmp dirs)

    def test_phase_c_biomass_five_folds_and_both_controls(self, tmp_path, monkeypatch):
        mod = _load("run_revision")
        monkeypatch.setattr(mod, "CHECKPOINTS", tmp_path)
        monkeypatch.setattr(mod, "METRICS", tmp_path)
        monkeypatch.setattr(mod, "_family_seconds", lambda: {})
        monkeypatch.setattr(mod, "_split_pairs", lambda: {
            ("biomass_lodo", 0), ("biomass_lodo", 1), ("biomass_lodo", 2),
            ("biomass_lodo", 3), ("biomass_lodo", 4),
            ("biomass_loso", 0), ("biomass_loso", 1), ("biomass_loso", 2),
            ("biomass_within_db", 0), ("biomass_within_db", 1), ("biomass_within_db", 2),
            ("biomass_within_db", 3), ("biomass_within_db", 4),
        })
        names = set(mod.plan_runs("C")["run_name"])
        assert {"E4_biomass_single_image_f4_s42", "E12_biomass_loso_f2_s42",
                "E13_biomass_within_db_f4_s42"} <= names
        assert "E12_biomass_loso_f3_s42" not in names

    def test_missing_split_rows_are_flagged(self, tmp_path, monkeypatch):
        mod = _load("run_revision")
        monkeypatch.setattr(mod, "CHECKPOINTS", tmp_path)
        monkeypatch.setattr(mod, "METRICS", tmp_path)
        monkeypatch.setattr(mod, "_family_seconds", lambda: {})
        monkeypatch.setattr(mod, "_split_pairs", lambda: set())
        df = mod.plan_runs("B")
        assert (df["split_ok"] == False).all()          # noqa: E712


class TestRunRevisionMainGuards:
    def _forbid_run_configs(self, mod, monkeypatch):
        def _boom(*a, **kw):
            raise AssertionError("run_configs must not be called")
        monkeypatch.setattr(mod, "run_configs", _boom)

    def test_split_guard_blocks_before_run_configs(self, monkeypatch):
        mod = _load("run_revision")
        df = pd.DataFrame({"run_name": ["E1_x_f0_s42"], "exists": [False],
                           "split_ok": [False], "est_sec": [100.0]})
        monkeypatch.setattr(mod, "plan_runs", lambda phase: df)
        monkeypatch.setattr(mod, "_stale_planned", lambda df: [])   # hermetic: no live registry
        self._forbid_run_configs(mod, monkeypatch)

        monkeypatch.setattr(sys, "argv", ["run_revision.py", "--phase", "B"])
        with pytest.raises(SystemExit):
            mod.main()

        monkeypatch.setattr(sys, "argv", ["run_revision.py", "--phase", "B", "--dry-run"])
        assert mod.main() == 1

    def test_stale_artefact_guard_blocks_before_run_configs(self, monkeypatch):
        mod = _load("run_revision")
        df = pd.DataFrame({"run_name": ["E1_x_f0_s42"], "exists": [True],
                           "split_ok": [True], "est_sec": [100.0]})
        monkeypatch.setattr(mod, "plan_runs", lambda phase: df)
        monkeypatch.setattr(mod, "_stale_planned", lambda df: ["E1_x_f0_s42"])
        self._forbid_run_configs(mod, monkeypatch)

        monkeypatch.setattr(sys, "argv", ["run_revision.py", "--phase", "B"])
        with pytest.raises(SystemExit):
            mod.main()

        monkeypatch.setattr(sys, "argv", ["run_revision.py", "--phase", "B", "--dry-run"])
        assert mod.main() == 1


class TestShortcutProbe:
    def test_perfectly_separable_feature_gives_ba_one_and_noise_gives_chance(self):
        mod = _load("shortcut_probes")
        rng = np.random.default_rng(0)
        y = np.array([0] * 100 + [1] * 100)
        sep = np.c_[y * 10.0 + rng.normal(0, 0.1, 200), rng.normal(0, 1, 200)]
        noise = rng.normal(0, 1, (200, 2))
        tr = np.arange(200) % 2 == 0
        te = ~tr
        good = mod.probe(sep, y, tr, te)
        bad = mod.probe(noise, y, tr, te)
        assert good["balanced_accuracy"] == pytest.approx(1.0)
        assert good["auroc"] == pytest.approx(1.0)
        assert 0.3 < bad["balanced_accuracy"] < 0.7
        assert good["n_train"] == 100 and good["n_test"] == 100

    def test_metadata_features_are_width_height_aspect_logsize(self):
        mod = _load("shortcut_probes")
        man = pd.DataFrame({"width": [1040, 1200], "height": [780, 1600], "file_size": [1000, 8000]})
        f = mod.metadata_features(man)
        assert f.shape == (2, 4)
        assert f[0].tolist() == pytest.approx([1040, 780, 1040 / 780, np.log(1000)])


class TestShortcutProbeProtocol:
    def test_run_protocol_degenerate_features_and_index_dropping(self):
        mod = _load("shortcut_probes")
        # 6-row manifest; columns beyond record_id are dummy for protocol "p"
        # (features are passed in precomputed, geometry_rule is not invoked).
        man = pd.DataFrame({
            "record_id": ["a", "b", "c", "d", "e", "f"],
            "width": [0] * 6, "height": [0] * 6, "file_size": [0] * 6,
            "relative_image_path": ["x"] * 6,
        })
        # idx omits "f" on purpose: "f" appears in the split table but is not a
        # manifest row the run has features for, and must be dropped.
        idx = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4}
        # Columns 0,1,2 (width, height, aspect) are constant across fold 0's
        # train+val rows (a, b, c); column 3 (log size) varies. Fold 1's train+val
        # rows (d, e) vary on every column.
        F = np.array([
            [10.0, 10.0, 1.0, 0.0],   # a
            [10.0, 10.0, 1.0, 1.0],   # b
            [10.0, 10.0, 1.0, 2.0],   # c
            [20.0, 5.0, 4.0, 3.0],    # d
            [5.0, 20.0, 0.25, 4.0],   # e
        ])
        target = np.array([0, 1, 0, 1, 0])
        # Fold 0 test is requested as d, e, f; "f" is absent from idx and dropped,
        # leaving n_test == 2 (both classes present, so balanced_accuracy_score
        # never sees a predicted class absent from y_true). Fold 1's test (b, c)
        # is likewise two-class for the same reason.
        spl = pd.DataFrame({
            "record_id": ["a", "b", "c", "d", "e", "f",   "d", "e", "b", "c"],
            "protocol":  ["p"] * 10,
            "fold":      [0, 0, 0, 0, 0, 0,               1, 1, 1, 1],
            "split":     ["train", "train", "val", "test", "test", "test",
                          "train", "train", "test", "test"],
        })

        res = mod.run_protocol(spl, man, "p", target, {"metadata_whasize": F}, idx)

        fold0 = res["per_fold"]["0"]["metadata_whasize"]
        assert fold0["degenerate_train_features"] == [0, 1, 2]
        assert fold0["n_test"] == 2

        fold1 = res["per_fold"]["1"]["metadata_whasize"]
        assert fold1["degenerate_train_features"] == []

    def test_geometry_rule_perfect_and_chance(self):
        mod = _load("shortcut_probes")
        man = pd.DataFrame({"width": [780, 1040, 1280, 720],
                            "height": [1040, 780, 720, 1280]})
        test_rows = np.arange(4)

        # Two portrait rows (0, 1) labelled diseased, two landscape rows (2, 3)
        # labelled healthy: the rule matches every row.
        y_perfect = np.array([1, 1, 0, 0])
        perfect = mod.geometry_rule(man, y_perfect, test_rows)
        assert perfect["balanced_accuracy"] == pytest.approx(1.0)
        assert perfect["n_test"] == 4
        assert perfect["protocol"] == mod.GEOMETRY_RULE_NOTE

        # Flip one label in each class (row 0 diseased->healthy, row 2
        # healthy->diseased) so the rule is wrong for exactly one row of each
        # class: recall is 0.5 in both classes, so balanced accuracy is chance.
        y_flipped = np.array([0, 1, 1, 0])
        chance = mod.geometry_rule(man, y_flipped, test_rows)
        assert chance["balanced_accuracy"] == pytest.approx(0.5)
        assert chance["protocol"] == mod.GEOMETRY_RULE_NOTE

    def test_run_protocol_disease_cv_tiger_geometry_rule_entry(self):
        mod = _load("shortcut_probes")
        man = pd.DataFrame({
            "record_id": ["r0", "r1", "r2", "r3", "r4", "r5"],
            "width":  [780, 1040, 1280, 720, 1040, 720],
            "height": [1040, 780, 720, 1280, 780, 1280],
            "file_size": [1000, 1000, 2000, 2000, 1500, 2500],
            "relative_image_path": ["x"] * 6,
        })
        idx = {rid: i for i, rid in enumerate(man["record_id"])}
        feats = {"metadata_whasize": mod.metadata_features(man)}
        target = np.array([1, 1, 0, 0, 1, 0])
        spl = pd.DataFrame({
            "record_id": ["r0", "r1", "r2", "r3", "r4", "r5"],
            "protocol": ["disease_cv_tiger"] * 6,
            "fold": [0] * 6,
            "split": ["train", "train", "train", "train", "test", "test"],
        })

        res = mod.run_protocol(spl, man, "disease_cv_tiger", target, feats, idx)

        fold0 = res["per_fold"]["0"]
        assert "geometry_rule_780x1040" in fold0
        assert fold0["geometry_rule_780x1040"]["balanced_accuracy"] == pytest.approx(1.0)
        assert fold0["geometry_rule_780x1040"]["protocol"] == mod.GEOMETRY_RULE_NOTE

        assert "geometry_rule_780x1040" in res["mean"]
        assert res["mean"]["geometry_rule_780x1040"]["balanced_accuracy"] == pytest.approx(1.0)
        assert res["mean"]["geometry_rule_780x1040"]["protocol"] == mod.GEOMETRY_RULE_NOTE


class TestExplainabilityPicks:
    """Grad-CAM panels must be drawn from the *held-out* rows of the run they
    illustrate, never from images the run trained on."""

    @pytest.fixture
    def splits(self) -> pd.DataFrame:
        rows = [
            # protocol p, fold 0, test: two Tiger Healthy, one Tiger Black Gill,
            # one SDIBD Healthy, one SDIBD BG -- plus a published augmentation
            # that must never be drawn, and a biomass record that is not disease.
            ("TG-He00", "p", 0, "test"), ("TG-He01", "p", 0, "test"),
            ("TG-Bl00", "p", 0, "test"), ("SD-Hea00", "p", 0, "test"),
            ("SD-BG00", "p", 0, "test"), ("TG-He00_aug0", "p", 0, "test"),
            ("BM-DB1-0-0", "p", 0, "test"),
            # same protocol/fold, other split
            ("TG-WS00", "p", 0, "train"), ("TG-Ye00", "p", 0, "val"),
            # same protocol, other fold
            ("TG-Ye01", "p", 1, "test"),
            # other protocol
            ("SD-WSS00", "q", 0, "test"),
        ]
        return pd.DataFrame(rows, columns=["record_id", "protocol", "fold", "split"])

    def test_picks_one_per_dataset_class_from_that_protocol_fold_split(
            self, synthetic_manifest, splits):
        mod = _load("explainability")
        got = mod.pick_examples(synthetic_manifest, splits, "p", 0, "test", 8)
        assert set(zip(got["dataset"], got["raw_class"])) == {
            ("TigerShrimpBD", "Healthy"), ("TigerShrimpBD", "Black Gill"),
            ("ShrimpDiseaseImageBD", "Healthy"), ("ShrimpDiseaseImageBD", "BG"),
        }
        assert len(got) == 4                      # one row per (dataset, class)

    def test_never_draws_augmented_other_folds_splits_or_protocols(
            self, synthetic_manifest, splits):
        mod = _load("explainability")
        got = set(mod.pick_examples(synthetic_manifest, splits, "p", 0, "test", 8)["record_id"])
        assert "TG-He00_aug0" not in got         # published augmentation
        assert "TG-WS00" not in got              # train rows of the same fold
        assert "TG-Ye00" not in got              # val rows of the same fold
        assert "TG-Ye01" not in got              # another fold
        assert "SD-WSS00" not in got             # another protocol
        assert "BM-DB1-0-0" not in got           # biomass, not a disease record
        assert got <= {"TG-He00", "TG-He01", "TG-Bl00", "SD-Hea00", "SD-BG00"}

    def test_n_caps_the_panel_and_the_choice_is_deterministic(
            self, synthetic_manifest, splits):
        mod = _load("explainability")
        assert len(mod.pick_examples(synthetic_manifest, splits, "p", 0, "test", 2)) == 2
        a = mod.pick_examples(synthetic_manifest, splits, "p", 0, "test", 8)
        b = mod.pick_examples(synthetic_manifest, splits, "p", 0, "test", 8)
        assert list(a["record_id"]) == list(b["record_id"])

    def test_no_splits_keeps_the_whole_eligible_manifest(self, synthetic_manifest, splits):
        mod = _load("explainability")
        got = mod.pick_examples(synthetic_manifest, None, None, None, "test", 8)
        # 2 datasets x 4 raw classes, originals only
        assert len(got) == 8
        assert not got["record_id"].str.contains("_aug").any()
        assert set(got["task"]) == {"disease"}

    def test_empty_selection_returns_empty_not_a_silent_fallback(
            self, synthetic_manifest, splits):
        mod = _load("explainability")
        assert len(mod.pick_examples(synthetic_manifest, splits, "p", 7, "test", 8)) == 0

    def test_fold_is_parsed_from_the_checkpoint_run_name(self):
        mod = _load("explainability")
        assert mod.fold_from_run_name("E1_disease_single_tiger_f0_s42") == 0
        assert mod.fold_from_run_name("E13_biomass_within_db_f4_s42") == 4
        assert mod.fold_from_run_name("no_fold_here") is None


class TestArchiveHashGuard:
    def test_main_refuses_without_a_current_split_hash(self, tmp_path, monkeypatch):
        """Fail-closed for the archiver: with no current hash every superseded-protocol
        row would count as stale, so the tool must refuse instead of moving the live grid."""
        import src.training.superseded as sup
        mod = _load("archive_superseded_runs")
        registry = tmp_path / "experiment_registry.csv"
        pd.DataFrame({"run_name": ["S_f0_s42"], "protocol": ["disease_cv"],
                      "split_hash": ["abc"], "checkpoint_path": [""],
                      "predictions_path": [""], "metrics_path": [""]}).to_csv(registry, index=False)
        monkeypatch.setattr(mod, "REGISTRY", registry)
        monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
        monkeypatch.setattr(sup, "SPLITS_HASH_FILE", tmp_path / "missing_hash.txt")
        monkeypatch.setattr(sys, "argv", ["archive_superseded_runs.py", "--dry-run"])
        with pytest.raises(SystemExit, match="refusing to archive"):
            mod.main()
        assert registry.exists() and not (tmp_path / "outputs" / "_archive").exists()


class TestEvaluateSeedSummary:
    def test_any_multiseed_config_is_summarised(self, tmp_path, monkeypatch):
        """Round 1 keyed the seed table on experiment == "E9"; the revision names
        multi-seed runs plainly, so the summary must key on seed count instead."""
        mod = _load("evaluate")
        fm = tmp_path / "fold_metrics.csv"
        pd.DataFrame({"config_name": ["A"] * 4 + ["B"] * 2, "experiment": ["E2"] * 4 + ["E4"] * 2,
                      "seed": [42, 42, 2026, 2026, 42, 42], "fold": [0, 1, 0, 1, 0, 1],
                      "health_bal_acc": [.9, .8, .85, .75, None, None],
                      "ind_weight_mae_g": [None] * 4 + [4.0, 5.0]}).to_csv(fm, index=False)
        monkeypatch.setattr(mod, "METRICS", tmp_path)
        rows = mod.seed_summary()
        assert {r["config_name"] for r in rows} == {"A"}       # B has one seed
        assert rows[0]["n_seeds"] == 2
