"""Model shapes, masked losses in situ, multi-view batching, checkpoint round-trip."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.data.labels import DISEASE_ATTRS, N_ATTRS
from src.evaluation.metrics import (
    biomass_metrics, disease_attribute_metrics, disease_health_metrics, select_thresholds,
)
from src.models.multitask import (
    AttentionPool, ModelConfig, ShrimpMultiTaskModel, count_parameters, masked_mean_pool,
)
from src.training.losses import masked_bce_with_logits, masked_huber
from src.training.trainer import MultiTaskTrainer, TrainConfig, cosine_warmup


def tiny_cfg(**kw) -> ModelConfig:
    # pretrained=False keeps the tests offline and fast
    return ModelConfig(backbone="convnext_tiny", pretrained=False, **kw)


class TestModelShapes:
    @pytest.mark.parametrize("backbone", ["convnext_tiny", "mobilenetv3_large"])
    def test_forward_output_dims(self, backbone):
        m = ShrimpMultiTaskModel(ModelConfig(backbone=backbone, pretrained=False))
        x = torch.randn(3, 3, 96, 96)
        o = m(x)
        assert o["health_logit"].shape == (3,)
        assert o["attr_logits"].shape == (3, N_ATTRS)
        assert o["biomass"].shape == (3, 3)
        assert o["features"].shape[0] == 3

    def test_heads_can_be_disabled(self):
        m = ShrimpMultiTaskModel(tiny_cfg(enable_biomass=False, enable_attrs=False))
        o = m(torch.randn(2, 3, 96, 96))
        assert "health_logit" in o
        assert "attr_logits" not in o and "biomass" not in o

    def test_domain_head_optional(self):
        m = ShrimpMultiTaskModel(tiny_cfg(enable_domain=True, n_domains=5))
        o = m(torch.randn(2, 3, 96, 96), domain_lambda=0.5)
        assert o["domain_logits"].shape == (2, 5)

    def test_parameter_count_reported(self):
        m = ShrimpMultiTaskModel(tiny_cfg())
        c = count_parameters(m)
        assert c["total_params"] > 1e6 and c["size_mb_fp32"] > 0

    def test_mobilenet_is_smaller_than_convnext(self):
        a = count_parameters(ShrimpMultiTaskModel(
            ModelConfig(backbone="convnext_tiny", pretrained=False)))
        b = count_parameters(ShrimpMultiTaskModel(
            ModelConfig(backbone="mobilenetv3_large", pretrained=False)))
        assert b["total_params"] < a["total_params"]


class TestMultiView:
    def test_forward_multiview_shapes(self):
        m = ShrimpMultiTaskModel(tiny_cfg())
        x = torch.randn(2, 4, 3, 96, 96)
        vm = torch.tensor([[1., 1., 1., 0.], [1., 1., 0., 0.]])
        o = m.forward_multiview(x, vm)
        assert o["biomass"].shape == (2, 3)
        assert o["view_features"].shape == (2, 4, m.feat_dim)

    def test_attention_pool_weights_sum_to_one_over_valid_views(self):
        pool = AttentionPool(8)
        f = torch.randn(2, 5, 8)
        vm = torch.tensor([[1., 1., 0., 0., 0.], [1., 1., 1., 1., 0.]])
        out, w = pool(f, vm)
        assert out.shape == (2, 8)
        assert torch.allclose(w.sum(dim=1), torch.ones(2), atol=1e-5)
        # padded views must receive zero weight
        assert torch.allclose(w[0, 2:], torch.zeros(3), atol=1e-6)

    def test_attention_pool_handles_zero_valid_views(self):
        pool = AttentionPool(8)
        f = torch.randn(1, 3, 8)
        vm = torch.zeros(1, 3)
        out, w = pool(f, vm)
        assert torch.isfinite(out).all(), "empty view mask must not produce NaN"
        assert torch.allclose(w, torch.zeros_like(w))

    def test_masked_mean_pool_ignores_padding(self):
        f = torch.stack([torch.ones(4, 6), torch.ones(4, 6) * 3])
        vm = torch.tensor([[1., 1., 0., 0.], [1., 0., 0., 0.]])
        out = masked_mean_pool(f, vm)
        assert torch.allclose(out[0], torch.ones(6))
        assert torch.allclose(out[1], torch.ones(6) * 3)


class TestMaskedLossesInModelContext:
    def test_unknown_labels_contribute_zero_loss(self):
        logits = torch.randn(4, N_ATTRS, requires_grad=True)
        targets = torch.zeros(4, N_ATTRS)
        mask = torch.ones(4, N_ATTRS)
        mask[:, 2] = 0.0                      # yellow_head unknown, as in SDIBD
        loss = masked_bce_with_logits(logits, targets, mask)
        loss.backward()
        assert torch.allclose(logits.grad[:, 2], torch.zeros(4)), \
            "masked-out attribute must receive exactly zero gradient"

    def test_all_zero_mask_is_finite_zero(self):
        for fn, args in ((masked_bce_with_logits, (torch.randn(3, 3), torch.zeros(3, 3))),
                         (masked_huber, (torch.randn(3, 3), torch.zeros(3, 3)))):
            out = fn(*args, torch.zeros(3, 3))
            assert torch.isfinite(out) and float(out) == 0.0

    def test_non_finite_prediction_at_masked_slot_cannot_poison(self):
        """DB1 has no cephalothorax measurement, so that column is always masked.
        A diverged value there must not destroy the batch loss."""
        pred = torch.tensor([[2.4, 11.5, float("inf")]])
        tgt = torch.tensor([[2.5, 11.0, 0.0]])
        mask = torch.tensor([[1.0, 1.0, 0.0]])
        out = masked_huber(pred, tgt, mask)
        assert torch.isfinite(out), "inf at a masked slot must not propagate"


class TestMetrics:
    def test_perfect_health_predictions(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.01, 0.02, 0.98, 0.99])
        m = disease_health_metrics(y, p)
        assert m["f1"] == pytest.approx(1.0)
        assert m["balanced_accuracy"] == pytest.approx(1.0)

    def test_fully_masked_attribute_excluded_from_macro(self):
        y = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=float)
        p = np.array([[.9, .1, .5], [.1, .9, .5], [.8, .8, .5], [.1, .1, .5]])
        m = np.array([[1, 1, 0]] * 4, dtype=float)
        r = disease_attribute_metrics(y, p, m, np.array([.5, .5, .5]), DISEASE_ATTRS)
        assert np.isnan(r["yellow_head"]["f1"])
        assert np.isfinite(r["macro_f1"])

    def test_threshold_selection_recovers_separation(self):
        rng = np.random.default_rng(0)
        y = np.concatenate([np.zeros(200), np.ones(200)])[:, None]
        p = np.concatenate([rng.uniform(0, .4, 200), rng.uniform(.6, 1, 200)])[:, None]
        m = np.ones_like(y)
        t = select_thresholds(y, p, m, objective="f1")
        assert .35 < t[0] < .65

    def test_biomass_metrics_perfect(self):
        y = np.array([10., 20., 30.])
        r = biomass_metrics(y, y.copy(), unit="g", mape_floor=0.5)
        assert r["mae"] == pytest.approx(0.0)
        assert r["r2"] == pytest.approx(1.0)


class TestSamplingReproducibility:
    """Guards two bugs where an epoch counter failed to advance, silently shrinking
    the effective training set without any error."""

    def test_source_balanced_sampler_varies_across_epochs(self):
        """The sampler must advance its own epoch. Relying on an external
        set_epoch call meant every epoch replayed one identical index multiset,
        so the balanced arm never saw roughly half its training rows."""
        from src.data.datasets import SourceBalancedSampler
        src = ["A"] * 100 + ["B"] * 20
        s = SourceBalancedSampler(src, seed=42)
        epochs = [list(iter(s)) for _ in range(3)]
        assert len({tuple(e) for e in epochs}) == 3, \
            "sampler replayed identical indices across epochs"

    def test_source_balanced_sampler_actually_balances(self):
        from src.data.datasets import SourceBalancedSampler
        src = ["A"] * 100 + ["B"] * 20
        s = SourceBalancedSampler(src, seed=42)
        idx = list(iter(s))
        a = sum(1 for i in idx if src[i] == "A")
        b = sum(1 for i in idx if src[i] == "B")
        # minority source must be drawn far above its 1:5 natural rate
        assert 0.35 < b / (a + b) < 0.65, f"not balanced: A={a} B={b}"

    def test_source_balanced_sampler_is_seed_reproducible(self):
        from src.data.datasets import SourceBalancedSampler
        src = ["A"] * 50 + ["B"] * 10
        a = list(iter(SourceBalancedSampler(src, seed=7)))
        b = list(iter(SourceBalancedSampler(src, seed=7)))
        assert a == b

    def test_multiview_dataset_resamples_per_epoch(self, synthetic_manifest, tmp_path):
        """BiomassIndividualDataset must select different views per epoch once
        set_epoch is called (the loader-side fix disables persistent workers so
        this mutation actually reaches the sampling code)."""
        from src.data.datasets import BiomassIndividualDataset
        import torch as _t

        ids = list(synthetic_manifest[
            synthetic_manifest["task"] == "biomass"]["record_id"])

        def fake_tf(img):
            return _t.zeros(3, 8, 8)

        ds = BiomassIndividualDataset(synthetic_manifest, ids, fake_tf,
                                      max_views=2, seed=0)
        picks = []
        for e in range(4):
            ds.set_epoch(e)
            picks.append(tuple(ds._pick(ds.groups[0], 4).tolist()))
        assert len(set(picks)) > 1, "view selection did not vary across epochs"

    def test_multiview_eval_selection_is_deterministic(self, synthetic_manifest):
        from src.data.datasets import BiomassIndividualDataset
        import torch as _t

        ids = list(synthetic_manifest[
            synthetic_manifest["task"] == "biomass"]["record_id"])
        ds = BiomassIndividualDataset(synthetic_manifest, ids, lambda i: _t.zeros(3, 8, 8),
                                      max_views=2, seed=0, deterministic_first_n=True)
        ds.set_epoch(0)
        a = tuple(ds._pick(ds.groups[0], 4).tolist())
        ds.set_epoch(3)
        b = tuple(ds._pick(ds.groups[0], 4).tolist())
        assert a == b == (0, 1), "eval view selection must be the first n, always"

    def test_stable_seed_is_process_independent(self):
        """Regression guard for the multi-view reproducibility bug: the training
        view draw must be seeded by a stable hash, never Python's per-process
        salted builtin hash(). These golden values are fixed by SHA-256 and would
        differ every interpreter run under the old ``hash((...))`` seeding.
        """
        from src.utils.hashing import stable_seed
        assert stable_seed(42, 0, "DB1_s010") == 1114328125
        assert stable_seed("seed_str", 0, "ind_00042") == 3821521238
        # order/content sensitivity
        assert stable_seed(42, 1, "DB1_s010") != stable_seed(42, 0, "DB1_s010")

    def test_multiview_training_draw_is_reproducible(self, synthetic_manifest):
        """Two independently constructed datasets with the same seed must select
        the SAME training views for the same (epoch, individual) -- the property
        the per-process hash() seeding silently violated across interpreters.
        """
        from src.data.datasets import BiomassIndividualDataset
        import torch as _t

        ids = list(synthetic_manifest[
            synthetic_manifest["task"] == "biomass"]["record_id"])
        mk = lambda: BiomassIndividualDataset(
            synthetic_manifest, ids, lambda i: _t.zeros(3, 8, 8),
            max_views=2, seed=7, deterministic_first_n=False)
        d1, d2 = mk(), mk()
        d1.set_epoch(2)
        d2.set_epoch(2)
        assert tuple(d1._pick(d1.groups[0], 4).tolist()) == \
               tuple(d2._pick(d2.groups[0], 4).tolist())


class TestTrainerMechanics:
    def test_cosine_warmup_shape(self):
        assert cosine_warmup(0, 100, 10, 0.01) < cosine_warmup(9, 100, 10, 0.01)
        assert cosine_warmup(10, 100, 10, 0.01) == pytest.approx(1.0, abs=1e-6)
        assert cosine_warmup(99, 100, 10, 0.01) < 0.1

    def test_checkpoint_round_trip(self, tmp_path):
        m = ShrimpMultiTaskModel(tiny_cfg())
        tr = MultiTaskTrainer(m, TrainConfig(epochs=1), torch.device("cpu"),
                              tmp_path, "rt")
        tr.best_metric = 0.42
        p = tr.save_checkpoint(0)
        assert p.exists()

        m2 = ShrimpMultiTaskModel(tiny_cfg())
        tr2 = MultiTaskTrainer(m2, TrainConfig(epochs=1), torch.device("cpu"),
                               tmp_path, "rt2")
        tr2.load_checkpoint(p)
        x = torch.randn(2, 3, 96, 96)
        m.eval(); m2.eval()
        with torch.no_grad():
            a, b = m(x)["health_logit"], m2(x)["health_logit"]
        assert torch.allclose(a, b, atol=1e-6), "reloaded weights must reproduce outputs"
        assert tr2.best_metric == pytest.approx(0.42)

    def test_deterministic_task_schedule(self):
        m = ShrimpMultiTaskModel(tiny_cfg())
        tr = MultiTaskTrainer(m, TrainConfig(seed=7), torch.device("cpu"),
                              __import__("pathlib").Path("."), "sched")

        class L:
            def __init__(self, n): self.n = n
            def __len__(self): return self.n

        loaders = {"disease": L(5), "biomass": L(3)}
        a = tr._schedule(loaders, 0)
        b = tr._schedule(loaders, 0)
        c = tr._schedule(loaders, 1)
        assert a == b, "same epoch must give the same schedule"
        assert a != c, "different epochs must vary the interleave"
        assert sorted(a) == sorted(["disease"] * 5 + ["biomass"] * 3)


class TestLossComponentLogging:
    """The per-step component dict is logged, never back-propagated: reading it
    must not warn about converting a graph tensor to a scalar."""

    def _trainer(self, tmp_path):
        model = ShrimpMultiTaskModel(ModelConfig(pretrained=False))
        return MultiTaskTrainer(model, TrainConfig(epochs=1), torch.device("cpu"),
                                tmp_path, "t")

    def test_disease_parts_are_plain_floats_without_a_grad_warning(self, tmp_path):
        import warnings
        tr = self._trainer(tmp_path)
        out = tr.model(torch.randn(2, 3, 96, 96))
        batch = {"health": torch.tensor([0.0, 1.0]),
                 "attrs": torch.zeros(2, N_ATTRS), "attr_mask": torch.ones(2, N_ATTRS)}
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            total, parts = tr.disease_loss(out, batch)
        assert total.requires_grad                        # the graph is untouched
        assert all(isinstance(v, float) for v in parts.values())

    def test_biomass_parts_are_plain_floats_without_a_grad_warning(self, tmp_path):
        import warnings
        tr = self._trainer(tmp_path)
        out = tr.model(torch.randn(2, 3, 96, 96))
        batch = {"biomass": torch.ones(2, 3), "biomass_mask": torch.ones(2, 3)}
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            total, parts = tr.biomass_loss(out, batch)
        assert total.requires_grad
        assert all(isinstance(v, float) for v in parts.values())


class TestCheckpointSlimming:
    def _trainer(self, tmp_path, **kw):
        cfg = ModelConfig(pretrained=False, enable_biomass=False)
        model = ShrimpMultiTaskModel(cfg)
        tc = TrainConfig(epochs=1, **kw)
        return MultiTaskTrainer(model, tc, torch.device("cpu"), tmp_path, "t")

    def test_default_checkpoint_has_no_optimizer_state(self, tmp_path):
        tr = self._trainer(tmp_path)
        p = tr.save_checkpoint(0)
        ck = torch.load(p, map_location="cpu", weights_only=True)
        assert "model_state" in ck and "weighter_state" in ck
        assert "optimizer_state" not in ck and "scaler_state" not in ck
        tr.load_checkpoint(p)   # must still round-trip

    def test_opt_in_keeps_optimizer_state(self, tmp_path):
        tr = self._trainer(tmp_path, save_optimizer_state=True)
        ck = torch.load(tr.save_checkpoint(0), map_location="cpu", weights_only=True)
        assert "optimizer_state" in ck and "scaler_state" in ck


class TestRevisionBackbones:
    """Cross-dataset backbone sweep (reviewer A7). Offline: pretrained=False."""

    @pytest.mark.parametrize("name,dim", [
        ("resnet50", 2048), ("efficientnet_b0", 1280), ("mobilenetv3_large", 1280),
        ("convnext_tiny", 768),
    ])
    def test_feature_dims_measured_by_forward(self, name, dim):
        m = ShrimpMultiTaskModel(ModelConfig(backbone=name, pretrained=False,
                                             enable_biomass=False))
        assert m.feat_dim == dim
        out = m(torch.zeros(2, 3, 64, 64))
        assert out["health_logit"].shape == (2,) and out["attr_logits"].shape == (2, N_ATTRS)

    def test_swin_requires_and_honours_img_size(self):
        with pytest.raises(Exception):
            ShrimpMultiTaskModel(ModelConfig(backbone="swin_t", pretrained=False,
                                             enable_biomass=False))       # 64x64 probe fails
        m = ShrimpMultiTaskModel(ModelConfig(backbone="swin_t", pretrained=False,
                                             enable_biomass=False, img_size=224))
        assert m.feat_dim == 768
        assert m(torch.zeros(1, 3, 224, 224))["health_logit"].shape == (1,)

    def test_unknown_backbone_is_refused(self):
        with pytest.raises(KeyError):
            ShrimpMultiTaskModel(ModelConfig(backbone="vit_b_16", pretrained=False))
