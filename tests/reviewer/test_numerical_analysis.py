"""Numerical safeguards for seed-aware and individual-level analyses."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "reviewer"))
import biomass_analysis as bio
import stats_rehab as stats


def seed_result(seed, delta, lo, hi, p):
    return {"seed": seed, "delta": delta, "lo": lo, "hi": hi,
            "p_two_sided": p, "n_shared": 100, "n_groups": 50}


def test_holm_adjustment_is_monotone_in_ordered_raw_p_values():
    assert stats.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_directional_seed_test_rejects_inconsistent_effect_signs():
    result = stats.aggregate_seeds([
        seed_result(42, 0.10, 0.06, 0.14, 0.001),
        seed_result(2026, -0.08, -0.12, -0.04, 0.002),
        seed_result(3407, 0.09, 0.05, 0.13, 0.001),
    ])
    assert result["sign_consistent"] is False
    assert result["p_raw"] == pytest.approx(1.0)
    assert result["ci_union_lo"] < 0 < result["ci_union_hi"]


def test_directional_seed_test_uses_least_favourable_consistent_seed():
    result = stats.aggregate_seeds([
        seed_result(42, 0.10, 0.05, 0.15, 0.001),
        seed_result(2026, 0.02, -0.01, 0.06, 0.2),
        seed_result(3407, 0.06, 0.01, 0.11, 0.04),
    ])
    assert result["p_raw"] == pytest.approx(0.2)
    assert result["delta_mean"] == pytest.approx(0.06)


def test_identical_models_have_zero_paired_effect_and_interval():
    y = np.repeat([0.0, 1.0] * 8, 3)
    groups = np.repeat(np.arange(16), 3)
    pred = np.where(y, 0.9, 0.1)
    result = stats.paired_bootstrap(y, pred, pred, groups, stats.balanced_accuracy,
                                    n_boot=100, seed=42)
    assert result["delta"] == pytest.approx(0)
    assert result["lo"] == pytest.approx(0)
    assert result["hi"] == pytest.approx(0)
    assert result["n_groups"] == 16
    assert result["p_two_sided"] == pytest.approx(1)


def test_finite_bootstrap_resolution_prevents_zero_p_value():
    y = np.repeat([0.0, 1.0] * 20, 2)
    groups = np.repeat(np.arange(40), 2)
    better = np.where(y, 0.9, 0.1)
    result = stats.paired_bootstrap(y, better, 1 - better, groups,
                                    stats.balanced_accuracy, n_boot=200, seed=42)
    assert result["delta"] == pytest.approx(1)
    assert result["p_two_sided"] == pytest.approx(1 / 200)


def test_database_mean_predictor_explains_no_within_database_variance():
    y = np.array([1.0, 3.0, 101.0, 103.0])
    db = np.array(["DB1", "DB1", "DB2", "DB2"])
    assert bio.r2_centred(y, [2.0, 2.0, 102.0, 102.0], db) == pytest.approx(0)
    assert bio.r2_centred(y, y, db) == pytest.approx(1)


def test_view_aggregation_exponentiates_weight_before_median():
    frame = pd.DataFrame({
        "individual_id": ["a", "a"], "pred_log_weight": np.log([1.0, 9.0]),
        "pred_length_cm": [2.0, 4.0], "y_weight_g": [5.0, 5.0],
        "y_length_cm": [3.0, 3.0], "db": ["DB1", "DB1"], "fold": [0, 0],
    })
    result = bio.individuals_from_preds(frame).iloc[0]
    assert result["pred_weight_g"] == pytest.approx(5)
    assert result["n_views"] == 2


def test_out_of_fold_features_block_training_animal_leakage():
    split = pd.DataFrame({"individual_id": ["a", "b"], "fold": [0, 0],
                          "split": ["train", "test"]})
    with pytest.raises(SystemExit, match="join leaks"):
        bio.check_no_feature_leak(split, pd.DataFrame({"individual_id": ["a"],
                                                      "feature_fold": [0]}))
    bio.check_no_feature_leak(split, pd.DataFrame({"individual_id": ["b"],
                                                 "feature_fold": [0]}))


def test_allometric_fit_recovers_known_length_weight_relation():
    train = pd.DataFrame({"length_cm": [1.0, 2.0, 3.0, 4.0],
                          "weight_g": [2.0, 16.0, 54.0, 128.0]})
    test = pd.DataFrame({"length_cm": [1.5, 5.0]})
    fit = bio.allometric_reference(train, test)
    assert fit["a"] == pytest.approx(2)
    assert fit["b"] == pytest.approx(3)
    assert fit["pred"].values == pytest.approx([6.75, 250.0])
