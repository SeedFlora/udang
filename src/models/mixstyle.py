"""MixStyle for domain generalization.

Implements the feature-statistic mixing regulariser of Zhou et al., "Domain
Generalization with MixStyle" (ICLR 2021), plus helpers that graft it onto the
intermediate stages of ``timm`` backbones via forward hooks (the backbone code
itself is never modified).

The core idea: instance-level channel statistics (mean and standard deviation
over the spatial dimensions) carry a large part of the *style* / domain
information of a feature map, while the normalised content carries the
semantics. Mixing the statistics of a sample with those of another sample --
ideally from a different domain -- synthesises novel styles during training and
discourages the network from latching onto domain-specific appearance cues.

Notes
-----
MixStyle is a *training-time* regulariser only. It is a strict no-op in eval
mode, and can be hard-disabled with :meth:`MixStyle.set_active`.

Module names used by the ``apply_mixstyle_to_*`` helpers were verified against
the installed timm 1.0.28 by introspecting ``named_modules()`` of real models
rather than assumed; both the classifier naming (``stages.0``, ``blocks.0``)
and the ``features_only=True`` naming (``stages_0``) are supported.
"""

from __future__ import annotations

import re
from typing import Sequence

import torch
import torch.nn as nn
from torch import Tensor

__all__ = [
    "MixStyle",
    "apply_mixstyle_to_convnext",
    "apply_mixstyle_to_mobilenetv3",
    "set_mixstyle_domain_labels",
    "remove_mixstyle",
]


class MixStyle(nn.Module):
    """Mix instance-level feature statistics across a batch.

    For an input ``x`` of shape ``(B, C, H, W)`` the per-sample, per-channel
    mean ``mu`` and standard deviation ``sigma`` are computed over the spatial
    dimensions. ``x`` is instance-normalised with these statistics and then
    re-scaled with statistics linearly interpolated towards those of another
    sample in the batch::

        lambda   ~ Beta(alpha, alpha)
        mu_mix    = lambda * mu    + (1 - lambda) * mu[perm]
        sigma_mix = lambda * sigma + (1 - lambda) * sigma[perm]
        out       = (x - mu) / sigma * sigma_mix + mu_mix

    Following the reference implementation, ``mu`` and ``sigma`` are detached,
    so gradients flow only through the normalised content.

    Parameters
    ----------
    p : float, optional
        Probability of applying MixStyle on any given forward pass. Sampled
        once per forward (per batch), not per sample, from torch's RNG, so
        ``torch.manual_seed`` alone reproduces a run. Default 0.5.
    alpha : float, optional
        Both parameters of the symmetric ``Beta(alpha, alpha)`` distribution the
        mixing coefficient is drawn from. Must be strictly positive. Small
        values (the paper uses 0.1) push ``lambda`` towards 0 or 1, i.e. towards
        one of the two styles rather than a bland average. Default 0.1.
    eps : float, optional
        Constant added to the variance before the square root, for numerical
        stability. Default 1e-6.
    mix : {"random", "crossdomain"}, optional
        Pairing strategy. ``"random"`` shuffles the batch uniformly.
        ``"crossdomain"`` pairs each sample with one from a *different* domain,
        using ``domain_labels`` when supplied. Default ``"random"``.

    Attributes
    ----------
    active : bool
        Read-only view of the hard enable/disable switch. See
        :meth:`set_active`.

    Raises
    ------
    ValueError
        If ``p`` is outside ``[0, 1]``, ``alpha`` is not positive, or ``mix`` is
        not a recognised strategy.

    Examples
    --------
    >>> import torch
    >>> ms = MixStyle(p=1.0, alpha=0.1)
    >>> x = torch.randn(4, 8, 16, 16)
    >>> ms.train()
    MixStyle(p=1.0, alpha=0.1, eps=1e-06, mix=random, active=True)
    >>> ms(x).shape
    torch.Size([4, 8, 16, 16])
    >>> ms.eval()
    MixStyle(p=1.0, alpha=0.1, eps=1e-06, mix=random, active=True)
    >>> bool(torch.equal(ms(x), x))  # exact no-op in eval mode
    True
    """

    _VALID_MIX = ("random", "crossdomain")

    def __init__(
        self,
        p: float = 0.5,
        alpha: float = 0.1,
        eps: float = 1e-6,
        mix: str = "random",
    ) -> None:
        super().__init__()
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")
        if alpha <= 0.0:
            raise ValueError(f"alpha must be > 0, got {alpha}")
        if mix not in self._VALID_MIX:
            raise ValueError(f"mix must be one of {self._VALID_MIX}, got {mix!r}")

        self.p = float(p)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.mix = mix
        self._active = True
        self._pending_domain_labels: Tensor | None = None
        self._beta = torch.distributions.Beta(self.alpha, self.alpha)

    # ------------------------------------------------------------------ #
    # configuration
    # ------------------------------------------------------------------ #
    @property
    def active(self) -> bool:
        """bool: Whether MixStyle may fire at all (independent of train mode)."""
        return self._active

    def set_active(self, flag: bool) -> "MixStyle":
        """Hard-enable or hard-disable MixStyle.

        When disabled the layer is an exact no-op even in training mode. This is
        the switch to use for deterministic evaluation, calibration passes, or
        for turning the regulariser off partway through a schedule.

        Parameters
        ----------
        flag : bool
            ``True`` to allow MixStyle to fire, ``False`` to force a no-op.

        Returns
        -------
        MixStyle
            ``self``, so calls can be chained.
        """
        self._active = bool(flag)
        return self

    def set_domain_labels(self, domain_labels: Tensor | None) -> "MixStyle":
        """Stash domain labels for the next forward pass.

        When MixStyle is driven by a forward hook there is no way to pass
        ``domain_labels`` through the backbone's call signature. Calling this
        just before the backbone forward lets ``mix="crossdomain"`` still see
        them. The stash is consumed (cleared) by the next forward pass, so a
        stale batch's labels can never leak into a later one.

        Parameters
        ----------
        domain_labels : torch.Tensor or None
            Integer tensor of shape ``(B,)``, or ``None`` to clear the stash.

        Returns
        -------
        MixStyle
            ``self``, so calls can be chained.
        """
        self._pending_domain_labels = domain_labels
        return self

    def extra_repr(self) -> str:
        """str: Compact description used by ``print(model)``."""
        return (
            f"p={self.p}, alpha={self.alpha}, eps={self.eps}, "
            f"mix={self.mix}, active={self._active}"
        )

    # ------------------------------------------------------------------ #
    # permutation strategies
    # ------------------------------------------------------------------ #
    def _crossdomain_perm(self, batch_size: int, domain_labels: Tensor | None,
                          device: torch.device) -> Tensor:
        """Build a permutation pairing samples across domains.

        With ``domain_labels`` each index is paired with a uniformly random
        index carrying a *different* label. Without labels, the reference
        implementation's heuristic is used: reverse the batch order and shuffle
        within each half, which assumes the batch is laid out domain-by-domain.

        Parameters
        ----------
        batch_size : int
            Number of samples in the batch.
        domain_labels : torch.Tensor or None
            Integer tensor of shape ``(B,)`` giving each sample's domain, or
            ``None``.
        device : torch.device
            Device the returned index tensor should live on.

        Returns
        -------
        torch.Tensor
            Long tensor of shape ``(B,)`` of source indices.

        Notes
        -----
        If ``domain_labels`` contains only a single distinct domain, no
        cross-domain pairing exists; the method falls back to a uniformly random
        permutation so that MixStyle still performs (within-domain) style
        mixing rather than silently doing nothing.
        """
        if domain_labels is None:
            perm = torch.arange(batch_size - 1, -1, -1, device=device)
            if batch_size % 2 == 0:
                half = batch_size // 2
                first, second = perm[:half], perm[half:]
                first = first[torch.randperm(half, device=device)]
                second = second[torch.randperm(half, device=device)]
                perm = torch.cat([first, second], dim=0)
            return perm

        labels = domain_labels.detach().to(device).reshape(-1)
        if labels.numel() != batch_size:
            raise ValueError(
                f"domain_labels has {labels.numel()} entries but batch size is "
                f"{batch_size}"
            )
        if labels.unique().numel() < 2:
            # Only one domain present: cross-domain pairing is impossible.
            return torch.randperm(batch_size, device=device)

        # (B, B) boolean: eligible[i, j] is True iff j is in a different domain.
        eligible = labels.unsqueeze(0) != labels.unsqueeze(1)
        # Every row has >= 1 eligible partner because >= 2 domains are present.
        return torch.multinomial(eligible.float(), num_samples=1).squeeze(1)

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #
    def forward(self, x: Tensor, domain_labels: Tensor | None = None) -> Tensor:
        """Apply MixStyle to a feature map.

        Parameters
        ----------
        x : torch.Tensor
            Feature map of shape ``(B, C, H, W)``.
        domain_labels : torch.Tensor or None, optional
            Integer tensor of shape ``(B,)`` used by ``mix="crossdomain"``. If
            ``None``, any labels previously supplied via
            :meth:`set_domain_labels` are used instead.

        Returns
        -------
        torch.Tensor
            Tensor of the same shape, dtype and device as ``x``. When MixStyle
            does not fire, ``x`` itself is returned unchanged (identical object,
            hence bitwise identical).

        Notes
        -----
        MixStyle is skipped -- and ``x`` returned untouched -- when any of the
        following hold:

        * the module is in eval mode, or has been disabled via
          :meth:`set_active`;
        * the Bernoulli(``p``) draw for this forward pass comes up negative;
        * the batch size is 1 (there is no other sample to mix with);
        * the input is not a 4D tensor.
        """
        stashed = self._pending_domain_labels
        self._pending_domain_labels = None
        if domain_labels is None:
            domain_labels = stashed

        if not self.training or not self._active:
            return x
        if x.dim() != 4:
            return x
        batch_size = x.shape[0]
        if batch_size < 2:
            # Nothing to mix with; a single sample would be paired with itself.
            return x
        # Drawn from torch's RNG (not python's ``random``) so that a single
        # ``torch.manual_seed`` reproduces a run: every other stochastic choice
        # in this module (the Beta sample, the permutation) already uses it.
        if float(torch.rand(())) > self.p:
            return x

        # Instance statistics over the spatial dims. correction=0 keeps this
        # well defined when H * W == 1.
        mu = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x - mu) / sig

        lmda = self._beta.sample((batch_size, 1, 1, 1)).to(device=x.device, dtype=x.dtype)

        if self.mix == "crossdomain":
            perm = self._crossdomain_perm(batch_size, domain_labels, x.device)
        else:
            perm = torch.randperm(batch_size, device=x.device)

        mu_perm, sig_perm = mu[perm], sig[perm]
        mu_mix = mu * lmda + mu_perm * (1.0 - lmda)
        sig_mix = sig * lmda + sig_perm * (1.0 - lmda)
        return x_normed * sig_mix + mu_mix


# ---------------------------------------------------------------------- #
# backbone instrumentation
# ---------------------------------------------------------------------- #
class _MixStyleHook:
    """Forward hook that routes a stage's output through a MixStyle layer.

    Defined at module scope rather than as a closure so that instrumented
    models stay picklable: ``torch.save(model)`` serialises every entry of each
    sub-module's ``_forward_hooks``, and pickle cannot serialise a function
    defined inside another function.

    Parameters
    ----------
    mixstyle : MixStyle
        Layer to apply to the hooked module's output.
    """

    __slots__ = ("mixstyle",)

    def __init__(self, mixstyle: MixStyle) -> None:
        self.mixstyle = mixstyle

    def __call__(self, _module: nn.Module, _inputs: tuple, output: object) -> object:
        """Replace a stage's output with its MixStyle-perturbed version."""
        if torch.is_tensor(output) and output.dim() == 4:
            # Call the module (not .forward) so its own hooks still fire.
            return self.mixstyle(output)
        return output

    def __getstate__(self) -> dict[str, object]:
        """dict: Pickle state (``__slots__`` classes need this explicitly)."""
        return {"mixstyle": self.mixstyle}

    def __setstate__(self, state: dict[str, object]) -> None:
        """Restore from :meth:`__getstate__` output."""
        self.mixstyle = state["mixstyle"]  # type: ignore[assignment]


def _discover_indexed_modules(model: nn.Module, container: str) -> dict[int, str]:
    """Find the indexed sub-modules of a backbone container.

    timm exposes stage/block containers under two naming schemes depending on
    how the model was created: ``"stages.0"`` for a normal classifier model and
    ``"stages_0"`` for a ``features_only=True`` feature extractor. Both are
    matched here, so callers never have to care which they hold.

    Parameters
    ----------
    model : torch.nn.Module
        Model to introspect.
    container : str
        Container attribute name, e.g. ``"stages"`` or ``"blocks"``.

    Returns
    -------
    dict of int to str
        Mapping from stage/block index to the fully-qualified module name, e.g.
        ``{0: "stages.0", 1: "stages.1", ...}``.

    Raises
    ------
    ValueError
        If no indexed modules matching ``container`` are found.
    """
    pattern = re.compile(rf"^{re.escape(container)}[._](\d+)$")
    found: dict[int, str] = {}
    for name, _module in model.named_modules():
        match = pattern.match(name)
        if match is not None:
            found[int(match.group(1))] = name
    if not found:
        raise ValueError(
            f"could not find any {container!r} sub-modules on "
            f"{type(model).__name__}; expected names like {container}.0 or "
            f"{container}_0"
        )
    return found


def _attach_mixstyle_hooks(
    model: nn.Module,
    container: str,
    indices: Sequence[int],
    p: float,
    alpha: float,
    mix: str,
) -> list[str]:
    """Register MixStyle forward hooks on selected container children.

    The MixStyle instances are registered on ``model`` as a real sub-module list
    (``model.mixstyle_layers``) so that ``model.train()`` / ``model.eval()``
    propagate to them. MixStyle holds no parameters or buffers, so the model's
    ``state_dict`` is unaffected.

    Parameters
    ----------
    model : torch.nn.Module
        Model to instrument, modified in place.
    container : str
        Container attribute name, e.g. ``"stages"``.
    indices : sequence of int
        Indices to insert MixStyle *after*. The final index is rejected.
    p, alpha : float
        Forwarded to :class:`MixStyle`.
    mix : str
        Forwarded to :class:`MixStyle`.

    Returns
    -------
    list of str
        Names of the modules that were hooked, in ascending index order.

    Raises
    ------
    ValueError
        If an index is out of range, or names the last stage/block (mixing
        styles that late corrupts semantics rather than appearance).
    """
    if mix not in MixStyle._VALID_MIX:
        # Validate before touching the model, so a bad argument can never leave
        # it half-instrumented.
        raise ValueError(f"mix must be one of {MixStyle._VALID_MIX}, got {mix!r}")

    available = _discover_indexed_modules(model, container)
    last_index = max(available)

    requested = sorted(set(int(i) for i in indices))
    for idx in requested:
        if idx not in available:
            raise ValueError(
                f"{container} index {idx} does not exist; available indices are "
                f"{sorted(available)}"
            )
        if idx == last_index:
            raise ValueError(
                f"refusing to insert MixStyle after the final {container} index "
                f"{idx}: late-layer style mixing degrades semantics. Choose from "
                f"{sorted(i for i in available if i != last_index)}"
            )

    modules_by_name = dict(model.named_modules())
    layers: list[MixStyle] = []
    handles: list[torch.utils.hooks.RemovableHandle] = []
    hooked: list[str] = []

    for idx in requested:
        name = available[idx]
        target = modules_by_name[name]
        mixstyle = MixStyle(p=p, alpha=alpha, mix=mix)
        handles.append(target.register_forward_hook(_MixStyleHook(mixstyle)))
        layers.append(mixstyle)
        hooked.append(name)

    # Remove any instrumentation from a previous call before re-attaching.
    remove_mixstyle(model)
    model.mixstyle_layers = nn.ModuleList(layers)
    model._mixstyle_handles = handles
    model.mixstyle_layers.train(model.training)
    return hooked


def apply_mixstyle_to_convnext(
    model: nn.Module,
    p: float = 0.5,
    alpha: float = 0.1,
    after_stages: Sequence[int] = (0, 1),
    mix: str = "random",
) -> list[str]:
    """Insert MixStyle after selected stages of a timm ConvNeXt.

    Verified against timm 1.0.28, where ``convnext_tiny`` exposes four stages
    named ``stages.0`` .. ``stages.3`` (or ``stages_0`` .. ``stages_3`` when
    built with ``features_only=True``). Instrumentation is done with forward
    hooks, so the backbone's own code is untouched and the model remains
    ``torch.save``-able and swappable.

    Parameters
    ----------
    model : torch.nn.Module
        A timm ConvNeXt model. Modified in place.
    p : float, optional
        MixStyle firing probability. Default 0.5.
    alpha : float, optional
        Beta distribution parameter. Default 0.1.
    after_stages : sequence of int, optional
        Stage indices to insert MixStyle after. Default ``(0, 1)`` -- the early,
        style-dominated stages. The final stage may not be selected.
    mix : {"random", "crossdomain"}, optional
        Pairing strategy forwarded to :class:`MixStyle`. With ``"crossdomain"``,
        call :func:`set_mixstyle_domain_labels` before each backbone forward so
        the hooked layers can see the batch's domain labels. Default
        ``"random"``.

    Returns
    -------
    list of str
        Names of the stage modules that were hooked.

    Raises
    ------
    ValueError
        If the stages cannot be discovered, or an index is invalid or names the
        final stage, or ``mix`` is not a recognised strategy.

    Examples
    --------
    >>> import timm  # doctest: +SKIP
    >>> m = timm.create_model("convnext_tiny", pretrained=False, num_classes=0)  # doctest: +SKIP
    >>> apply_mixstyle_to_convnext(m, after_stages=(0, 1))  # doctest: +SKIP
    ['stages.0', 'stages.1']
    """
    return _attach_mixstyle_hooks(
        model, container="stages", indices=after_stages, p=p, alpha=alpha,
        mix=mix,
    )


def apply_mixstyle_to_mobilenetv3(
    model: nn.Module,
    p: float = 0.5,
    alpha: float = 0.1,
    after_blocks: Sequence[int] = (1, 2),
    mix: str = "random",
) -> list[str]:
    """Insert MixStyle after selected blocks of a timm MobileNetV3.

    Verified against timm 1.0.28, where ``mobilenetv3_large_100`` exposes seven
    block groups named ``blocks.0`` .. ``blocks.6`` (identical under
    ``features_only=True``). As with ConvNeXt, forward hooks are used so the
    backbone is untouched.

    Parameters
    ----------
    model : torch.nn.Module
        A timm MobileNetV3 model. Modified in place.
    p : float, optional
        MixStyle firing probability. Default 0.5.
    alpha : float, optional
        Beta distribution parameter. Default 0.1.
    after_blocks : sequence of int, optional
        Block indices to insert MixStyle after. Default ``(1, 2)`` -- resolution
        1/4 and 1/8, early enough to be style-dominated. The final block may not
        be selected.
    mix : {"random", "crossdomain"}, optional
        Pairing strategy forwarded to :class:`MixStyle`. With ``"crossdomain"``,
        call :func:`set_mixstyle_domain_labels` before each backbone forward so
        the hooked layers can see the batch's domain labels. Default
        ``"random"``.

    Returns
    -------
    list of str
        Names of the block modules that were hooked.

    Raises
    ------
    ValueError
        If the blocks cannot be discovered, or an index is invalid or names the
        final block, or ``mix`` is not a recognised strategy.
    """
    return _attach_mixstyle_hooks(
        model, container="blocks", indices=after_blocks, p=p, alpha=alpha,
        mix=mix,
    )


def set_mixstyle_domain_labels(
    model: nn.Module, domain_labels: Tensor | None
) -> int:
    """Broadcast a batch's domain labels to every MixStyle layer on ``model``.

    Hook-driven MixStyle cannot receive ``domain_labels`` through the backbone's
    call signature, so ``mix="crossdomain"`` needs them stashed on each layer
    beforehand. Call this immediately before the backbone forward; each layer
    consumes its stash on that single forward pass, so stale labels cannot leak
    into a later batch.

    Parameters
    ----------
    model : torch.nn.Module
        Model previously instrumented by an ``apply_mixstyle_to_*`` helper.
        Calling this on an uninstrumented model is a no-op.
    domain_labels : torch.Tensor or None
        Integer tensor of shape ``(B,)``, or ``None`` to clear the stashes.

    Returns
    -------
    int
        Number of MixStyle layers the labels were set on.

    Examples
    --------
    >>> apply_mixstyle_to_convnext(m, mix="crossdomain")  # doctest: +SKIP
    >>> set_mixstyle_domain_labels(m, batch["domain"])  # doctest: +SKIP
    >>> logits = m(images)  # doctest: +SKIP
    """
    layers = getattr(model, "mixstyle_layers", None)
    if layers is None:
        return 0
    count = 0
    for layer in layers:
        layer.set_domain_labels(domain_labels)
        count += 1
    return count


def remove_mixstyle(model: nn.Module) -> int:
    """Remove MixStyle instrumentation previously added to ``model``.

    Deregisters every forward hook installed by the ``apply_mixstyle_to_*``
    helpers and drops the attached :class:`MixStyle` modules, restoring the
    original forward behaviour. Safe to call on a model that was never
    instrumented.

    Parameters
    ----------
    model : torch.nn.Module
        Model to clean up, modified in place.

    Returns
    -------
    int
        Number of hooks that were removed.
    """
    handles = getattr(model, "_mixstyle_handles", None)
    removed = 0
    if handles:
        for handle in handles:
            handle.remove()
            removed += 1
    if hasattr(model, "_mixstyle_handles"):
        del model._mixstyle_handles
    if hasattr(model, "mixstyle_layers"):
        del model.mixstyle_layers
    return removed


# ---------------------------------------------------------------------- #
# self-test
# ---------------------------------------------------------------------- #
def _self_test() -> None:
    """Run assertions covering the documented MixStyle contract."""
    torch.manual_seed(0)

    exact = dict(rtol=0.0, atol=0.0)
    x = torch.randn(8, 16, 12, 12)

    # -- 1. eval mode is an exact no-op --------------------------------- #
    ms = MixStyle(p=1.0, alpha=0.1)
    ms.eval()
    out = ms(x)
    assert torch.allclose(out, x, **exact), "eval mode must be an exact no-op"
    assert out.shape == x.shape
    print("[ok] eval mode is an exact no-op (rtol=0, atol=0)")

    # -- 2. set_active(False) is an exact no-op in train mode ----------- #
    ms.train()
    ms.set_active(False)
    assert ms.active is False
    out = ms(x)
    assert torch.allclose(out, x, **exact), "set_active(False) must be an exact no-op"
    print("[ok] set_active(False) is an exact no-op in train mode (rtol=0, atol=0)")

    # -- 3. train mode with p=1.0 changes the tensor -------------------- #
    ms.set_active(True)
    ms.train()
    out = ms(x)
    assert not torch.allclose(out, x, **exact), "p=1.0 in train mode must change x"
    assert out.shape == x.shape, "output shape must equal input shape"
    assert torch.isfinite(out).all(), "output must be finite"
    print("[ok] train mode with p=1.0 changes the tensor; shape preserved")

    # -- 4. output shape always equals input shape ---------------------- #
    for shape in [(2, 3, 8, 8), (4, 64, 1, 1), (16, 8, 5, 7), (3, 32, 4, 4)]:
        probe = torch.randn(*shape)
        result = MixStyle(p=1.0, alpha=0.1).train()(probe)
        assert result.shape == probe.shape, f"shape mismatch for {shape}"
        assert torch.isfinite(result).all(), f"non-finite output for {shape}"
    print("[ok] output shape equals input shape across 4 shapes (incl. H=W=1)")

    # -- 5. batch size 1 is a no-op and does not crash ------------------ #
    single = torch.randn(1, 16, 12, 12)
    out = MixStyle(p=1.0, alpha=0.1).train()(single)
    assert torch.allclose(out, single, **exact), "batch size 1 must be a no-op"
    assert out.shape == single.shape
    print("[ok] batch size 1 is an exact no-op and does not crash")

    # -- 5b. crossdomain pairing respects domain labels ----------------- #
    cd = MixStyle(p=1.0, alpha=0.1, mix="crossdomain").train()
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    out = cd(x, domain_labels=labels)
    assert out.shape == x.shape and torch.isfinite(out).all()
    perms = [
        cd._crossdomain_perm(8, labels, torch.device("cpu")) for _ in range(50)
    ]
    assert all(
        bool((labels[perm] != labels).all()) for perm in perms
    ), "crossdomain must pair samples from different domains"
    # single-domain batches fall back to a random permutation, not a crash
    one_domain = torch.zeros(8, dtype=torch.long)
    assert cd(x, domain_labels=one_domain).shape == x.shape
    # label-free crossdomain (reverse-order heuristic) must still work
    assert cd(x).shape == x.shape
    print("[ok] crossdomain pairs across domains; degenerate cases handled")

    # -- 5c. set_domain_labels stash is consumed exactly once ----------- #
    cd.set_domain_labels(labels)
    assert cd._pending_domain_labels is not None
    cd(x)
    assert cd._pending_domain_labels is None, "stash must be cleared after forward"
    print("[ok] set_domain_labels stash is consumed by one forward pass")

    # -- 6. hooks actually fire on a real timm convnext_tiny ------------ #
    import timm

    model = timm.create_model("convnext_tiny", pretrained=False, num_classes=0)
    model.eval()
    inp = torch.randn(4, 3, 64, 64)
    with torch.no_grad():
        baseline = model(inp)

    hooked = apply_mixstyle_to_convnext(model, p=1.0, alpha=0.1, after_stages=(0, 1))
    assert hooked == ["stages.0", "stages.1"], f"unexpected hooked names: {hooked}"
    print(f"[ok] convnext hooked modules discovered by introspection: {hooked}")

    calls = {"n": 0}

    def counter(_m: nn.Module, _i: tuple, _o: object) -> None:
        calls["n"] += 1

    for layer in model.mixstyle_layers:
        layer.register_forward_hook(counter)

    model.train()
    out_train = model(inp)
    assert calls["n"] == 2, f"expected 2 MixStyle invocations, got {calls['n']}"
    assert out_train.shape == baseline.shape
    print(f"[ok] MixStyle hooks fired {calls['n']}x on a real convnext_tiny forward")

    # -- 7. hooked model output shape unchanged vs unhooked ------------- #
    model.eval()
    with torch.no_grad():
        hooked_out = model(inp)
    assert hooked_out.shape == baseline.shape, "hooked output shape changed"
    assert torch.allclose(hooked_out, baseline, **exact), (
        "hooked model in eval mode must reproduce the unhooked output exactly"
    )
    print("[ok] hooked model output shape unchanged; eval output bitwise identical")

    # train-mode eval-propagation check: model.eval() must silence MixStyle
    calls["n"] = 0
    with torch.no_grad():
        model(inp)
    assert calls["n"] == 2, "MixStyle still runs in eval, but as a no-op"
    assert all(not layer.training for layer in model.mixstyle_layers), (
        "model.eval() must propagate to MixStyle layers"
    )
    print("[ok] model.eval() propagates to MixStyle layers (registered as submodules)")

    # -- 8. mobilenetv3 --------------------------------------------------#
    mnv3 = timm.create_model("mobilenetv3_large_100", pretrained=False, num_classes=0)
    mnv3.eval()
    with torch.no_grad():
        mnv3_base = mnv3(inp)
    mnv3_hooked = apply_mixstyle_to_mobilenetv3(mnv3, p=1.0, alpha=0.1, after_blocks=(1, 2))
    assert mnv3_hooked == ["blocks.1", "blocks.2"], f"unexpected: {mnv3_hooked}"
    mnv3.train()
    assert mnv3(inp).shape == mnv3_base.shape
    print(f"[ok] mobilenetv3 hooked modules: {mnv3_hooked}; shape preserved")

    # -- 9. guard rails --------------------------------------------------#
    for bad_kwargs in [dict(p=1.5), dict(alpha=0.0), dict(mix="nope")]:
        try:
            MixStyle(**bad_kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad_kwargs}")
    try:
        apply_mixstyle_to_convnext(model, after_stages=(3,))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when hooking the final stage")
    print("[ok] invalid p/alpha/mix and final-stage selection all raise ValueError")

    # -- 9b. an instrumented model must stay picklable -------------------#
    # A separate model, because the counter hooks registered above are local
    # closures belonging to this test and would themselves defeat pickling.
    import io

    pick = timm.create_model("convnext_tiny", pretrained=False, num_classes=0)
    pick.eval()
    with torch.no_grad():
        pick_baseline = pick(inp)
    apply_mixstyle_to_convnext(pick, p=1.0, alpha=0.1, after_stages=(0, 1))

    buffer = io.BytesIO()
    torch.save(pick, buffer)  # raises if a hook is an unpicklable closure
    buffer.seek(0)
    # weights_only=False is required to round-trip a whole nn.Module (that is
    # exactly what is under test here) and is safe only because this buffer was
    # produced in-process two lines above. Never use it on an untrusted file.
    reloaded = torch.load(buffer, weights_only=False)
    reloaded.eval()
    with torch.no_grad():
        assert torch.allclose(reloaded(inp), pick_baseline, **exact), (
            "a save/load round trip must preserve eval-mode behaviour"
        )
    assert len(reloaded.mixstyle_layers) == 2, "MixStyle layers lost in round trip"
    # The reloaded hooks must still point at the reloaded MixStyle layers, not
    # at orphaned copies: firing them in train mode must perturb the output.
    reloaded.train()
    assert not torch.allclose(reloaded(inp), pick_baseline, **exact), (
        "reloaded MixStyle hooks must still be live"
    )
    assert remove_mixstyle(reloaded) == 2, "reloaded hooks must still be removable"
    reloaded.eval()
    with torch.no_grad():
        assert torch.allclose(reloaded(inp), pick_baseline, **exact)
    print("[ok] instrumented model survives torch.save/torch.load round trip")

    # -- 9c. torch.manual_seed alone reproduces a run --------------------#
    def _sample_run() -> Tensor:
        torch.manual_seed(1234)
        layer = MixStyle(p=0.5, alpha=0.1).train()
        probe = torch.randn(4, 8, 6, 6)
        return torch.stack([layer(probe).clone() for _ in range(20)])

    assert torch.equal(_sample_run(), _sample_run()), (
        "MixStyle must be reproducible under torch.manual_seed alone"
    )
    print("[ok] torch.manual_seed alone reproduces a MixStyle run")

    # -- 9d. crossdomain is reachable through the helper API -------------#
    cd_model = timm.create_model("convnext_tiny", pretrained=False, num_classes=0)
    apply_mixstyle_to_convnext(cd_model, p=1.0, after_stages=(0, 1), mix="crossdomain")
    assert all(layer.mix == "crossdomain" for layer in cd_model.mixstyle_layers)
    n_set = set_mixstyle_domain_labels(cd_model, torch.tensor([0, 0, 1, 1]))
    assert n_set == 2, f"expected labels set on 2 layers, got {n_set}"
    assert all(l._pending_domain_labels is not None for l in cd_model.mixstyle_layers)
    cd_model.train()
    assert cd_model(inp).shape == baseline.shape
    assert all(l._pending_domain_labels is None for l in cd_model.mixstyle_layers), (
        "every layer's stash must be consumed by the forward pass"
    )
    assert set_mixstyle_domain_labels(timm.create_model("convnext_tiny",
                                                       pretrained=False), None) == 0
    try:
        apply_mixstyle_to_convnext(cd_model, mix="bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for an invalid mix strategy")
    print("[ok] crossdomain reachable via helpers; labels broadcast and consumed")

    # -- 10. cleanup -----------------------------------------------------#
    removed = remove_mixstyle(model)
    assert removed == 2, f"expected 2 hooks removed, got {removed}"
    model.train()
    with torch.no_grad():
        model.eval()
        restored = model(inp)
    assert torch.allclose(restored, baseline, **exact), "removal must restore the model"
    print(f"[ok] remove_mixstyle removed {removed} hooks and restored the backbone")

    print("\nALL ASSERTIONS PASSED")


if __name__ == "__main__":
    _self_test()
