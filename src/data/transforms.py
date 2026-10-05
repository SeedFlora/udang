"""Task-specific image augmentation pipelines for ShrimpVision-MTL.

This module provides two families of albumentations (2.x) pipelines:

``build_disease_transforms``
    Colour- and texture-preserving augmentation for the disease
    (health / attribute) branch. The visual signs of the three modelled
    diseases -- ``black_gill``, ``wssv`` and ``yellow_head`` -- are
    *chromatic*: a darkened gill region, white focal spots on the carapace,
    a yellowed cephalothorax. Any augmentation that shifts hue, shuffles
    channels, desaturates or solarises can either destroy a real sign or
    fabricate a fake one, which silently corrupts the labels. Such transforms
    are therefore excluded from the default pipeline and are only reachable
    through the explicitly ablation-only ``strong_color`` flag.

``build_biomass_transforms``
    Geometry-preserving augmentation for the biomass regression branch.
    Biomass targets (``log_weight``, ``length_cm``, ``cephalothorax_cm``)
    are read off the *full* animal: the rostrum tip, the tail fan and the
    cephalothorax boundary must all remain inside the frame, and the
    body proportions must not be distorted. Consequently this pipeline
    contains no random crop of any kind and never changes the aspect ratio.

Both pipelines end in :func:`letterbox`, an aspect-preserving resize with
symmetric constant padding, followed by ImageNet normalisation and a
conversion to ``torch.FloatTensor`` of shape ``(3, H, W)``.

Notes
-----
The returned objects are instances of :class:`ImageTransform` rather than
closures. This is deliberate: on Windows, ``torch.utils.data.DataLoader``
with ``num_workers > 0`` uses the *spawn* start method, which pickles the
dataset (and therefore its transform). A closure is not picklable; a
module-level class is.

Being picklable is necessary but not sufficient for correct multi-worker
augmentation: spawn ships *identical* pickled bytes to every worker, so the
albumentations RNG must be re-seeded per process on unpickle or all workers
emit the same augmented stream. :meth:`ImageTransform.__setstate__` handles
this; see its docstring.

Non-finite pixels (NaN / +-inf, e.g. from a corrupt decode) are mapped to the
neutral ``LETTERBOX_FILL`` grey rather than cast directly to ``uint8``, which
is undefined behaviour that lands on pure black or pure white -- both of which
imitate a real disease sign.
"""

from __future__ import annotations

import os
from typing import Callable, Sequence

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "LETTERBOX_FILL",
    "ImageTransform",
    "letterbox",
    "build_disease_transforms",
    "build_biomass_transforms",
    "denormalize_to_uint8",
    "save_augmentation_grid",
]

#: ImageNet channel means (RGB order), used by every ImageNet-pretrained backbone.
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
#: ImageNet channel standard deviations (RGB order).
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

#: Default padding value for letterboxing (neutral mid-grey, the YOLO convention).
LETTERBOX_FILL: int = 114


# --------------------------------------------------------------------------- #
# Low-level helpers
# --------------------------------------------------------------------------- #
def _ensure_rgb_uint8(image: np.ndarray) -> np.ndarray:
    """Coerce an arbitrary array into a contiguous ``uint8`` RGB image.

    Parameters
    ----------
    image : numpy.ndarray
        Input image. Accepted layouts are ``(H, W)`` grayscale,
        ``(H, W, 1)``, ``(H, W, 3)`` RGB and ``(H, W, 4)`` RGBA.
        Floating point input is assumed to be in ``[0, 1]`` if its maximum
        is ``<= 1.0``, otherwise in ``[0, 255]``.

    Returns
    -------
    numpy.ndarray
        Contiguous ``uint8`` array of shape ``(H, W, 3)``.

    Raises
    ------
    TypeError
        If ``image`` is not a :class:`numpy.ndarray`.
    ValueError
        If the array rank or channel count is unsupported.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"expected numpy.ndarray, got {type(image)!r}")

    if image.ndim == 2:
        image = image[:, :, None]
    if image.ndim != 3:
        raise ValueError(f"expected a 2D or 3D array, got shape {image.shape}")

    n_channels = image.shape[2]
    if n_channels == 1:
        image = np.repeat(image, 3, axis=2)
    elif n_channels == 4:
        image = image[:, :, :3]
    elif n_channels != 3:
        raise ValueError(f"expected 1, 3 or 4 channels, got {n_channels}")

    if image.dtype != np.uint8:
        arr = image.astype(np.float32)
        finite = np.isfinite(arr)
        # Derive the [0, 1] vs [0, 255] scale from the finite pixels only, so a
        # single NaN/inf does not decide the scale for the whole image.
        max_val = float(arr[finite].max()) if finite.any() else 0.0
        scale = 255.0 if max_val <= 1.0 else 1.0
        arr = np.clip(arr * scale, 0.0, 255.0)
        if not finite.all():
            # Non-finite pixels are corrupt data, not content. Mapping them to
            # the neutral LETTERBOX_FILL grey -- the same "no data" colour used
            # for padding -- is deliberate: a plain ``astype(np.uint8)`` cast of
            # NaN is undefined behaviour that lands on 0 (pure black) on x86,
            # which mimics a black_gill darkening, while +inf saturates to 255
            # (pure white), which mimics a wssv focal spot. Either would
            # fabricate a disease sign out of a decode error.
            arr = np.where(finite, arr, float(LETTERBOX_FILL))
        image = arr.astype(np.uint8)

    return np.ascontiguousarray(image)


def letterbox(image: np.ndarray, size: int, fill: int = LETTERBOX_FILL) -> np.ndarray:
    """Aspect-preserving resize followed by symmetric centre padding.

    The image is scaled by ``r = min(size / h, size / w)`` -- the *same*
    factor on both axes -- and the remainder of the ``(size, size)`` canvas
    is filled with the constant ``fill``. The aspect ratio is therefore
    never distorted: a square region in the source stays square in the
    output (up to sub-pixel rounding of at most one pixel per side).

    Parameters
    ----------
    image : numpy.ndarray
        Source image; see :func:`_ensure_rgb_uint8` for accepted layouts.
    size : int
        Side length of the square output canvas, in pixels. Must be positive.
    fill : int, optional
        Constant padding value applied to all three channels. Defaults to
        ``114`` (neutral mid-grey), which is far from both the black and the
        white extremes and so does not mimic a disease sign.

    Returns
    -------
    numpy.ndarray
        ``uint8`` array of shape exactly ``(size, size, 3)``.

    Raises
    ------
    ValueError
        If ``size`` is not a positive integer or the source image is empty.

    Notes
    -----
    Implemented with pure ``numpy``/``cv2`` -- no albumentations -- so that it
    can be reused from dataset code, inference code and visual diagnostics
    without dragging in a pipeline object.
    """
    if int(size) <= 0:
        raise ValueError(f"size must be a positive integer, got {size}")
    size = int(size)

    # cv2 saturates an out-of-range border value instead of complaining, which
    # would silently turn the "no data" padding into pure white (mimicking a
    # wssv spot) or pure black (mimicking black_gill). Reject it up front.
    if not 0 <= int(fill) <= 255:
        raise ValueError(f"fill must be in [0, 255], got {fill}")

    source = image  # keep a handle on the caller's buffer for the alias check
    image = _ensure_rgb_uint8(image)
    height, width = image.shape[:2]
    if height == 0 or width == 0:
        raise ValueError(f"cannot letterbox an empty image of shape {image.shape}")

    ratio = min(size / float(height), size / float(width))
    new_w = max(1, min(size, int(round(width * ratio))))
    new_h = max(1, min(size, int(round(height * ratio))))

    if (new_w, new_h) != (width, height):
        # INTER_AREA is the correct (anti-aliased) choice when shrinking;
        # INTER_LINEAR is cheaper and smoother when enlarging.
        interpolation = cv2.INTER_AREA if ratio < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    else:
        resized = image

    pad_w = size - new_w
    pad_h = size - new_h
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top

    if pad_w or pad_h:
        resized = cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            borderType=cv2.BORDER_CONSTANT,
            value=(int(fill), int(fill), int(fill)),
        )

    out = np.ascontiguousarray(resized, dtype=np.uint8)

    # When the source is already exactly (size, size, 3) uint8 and contiguous,
    # every step above is a no-op and ``out`` would BE the caller's array. A
    # caller that then writes into the result -- or a dataset that caches
    # decoded frames in RAM and letterboxes them per epoch -- would silently
    # corrupt its own source data. Always hand back an independent buffer.
    if np.may_share_memory(out, source):
        out = out.copy()

    # Raised explicitly rather than via ``assert`` so the invariant survives
    # ``python -O``, which strips assert statements.
    if out.shape != (size, size, 3):
        raise AssertionError(f"letterbox produced {out.shape}, expected {(size, size, 3)}")
    return out


def denormalize_to_uint8(
    tensor: torch.Tensor,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> np.ndarray:
    """Invert ImageNet normalisation and return a displayable image.

    Parameters
    ----------
    tensor : torch.Tensor
        Normalised image tensor of shape ``(3, H, W)``.
    mean, std : sequence of float, optional
        Per-channel statistics used at normalisation time.

    Returns
    -------
    numpy.ndarray
        ``uint8`` array of shape ``(H, W, 3)`` suitable for ``imshow``.
    """
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError(f"expected a (3, H, W) tensor, got {tuple(tensor.shape)}")

    arr = tensor.detach().float().cpu().numpy()
    mean_arr = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
    std_arr = np.asarray(std, dtype=np.float32).reshape(3, 1, 1)
    arr = arr * std_arr + mean_arr
    finite = np.isfinite(arr)
    arr = np.clip(arr, 0.0, 1.0) * 255.0
    if not finite.all():
        # Same rule as _ensure_rgb_uint8: casting NaN to uint8 is undefined
        # behaviour. Render non-finite pixels as the neutral "no data" grey so a
        # NaN-producing model shows up as flat grey in the diagnostic figure
        # instead of as plausible-looking black or white image content.
        arr = np.where(finite, arr, float(LETTERBOX_FILL))
    return np.ascontiguousarray(arr.transpose(1, 2, 0).astype(np.uint8))


# --------------------------------------------------------------------------- #
# Callable wrapper
# --------------------------------------------------------------------------- #
class ImageTransform:
    """Picklable callable: ``uint8`` RGB image -> normalised ``(3, S, S)`` tensor.

    The call sequence is intentionally ordered:

    1. coerce the input to contiguous ``uint8`` RGB;
    2. apply the stochastic albumentations pipeline (train only), which
       operates at the source resolution so that padding introduced by
       rotation matches the letterbox fill value;
    3. :func:`letterbox` to ``(image_size, image_size)`` -- the only resize,
       and an aspect-preserving one;
    4. ImageNet normalisation and conversion to ``torch.FloatTensor``.

    Parameters
    ----------
    augment : albumentations.Compose or None
        Stochastic geometric/photometric stage, or ``None`` for eval.
    image_size : int
        Output side length in pixels.
    fill : int, optional
        Letterbox / border padding value.
    name : str, optional
        Human-readable tag used by ``repr``.

    Attributes
    ----------
    is_train : bool
        ``True`` when a stochastic stage is attached.
    """

    def __init__(
        self,
        augment: A.Compose | None,
        image_size: int,
        fill: int = LETTERBOX_FILL,
        name: str = "transform",
    ) -> None:
        # Fail fast at build time. Deferring these to the first ``__call__``
        # means the error surfaces only after the model and the dataloaders are
        # already up, several minutes into a run.
        if int(image_size) <= 0:
            raise ValueError(f"image_size must be a positive integer, got {image_size}")
        if not 0 <= int(fill) <= 255:
            raise ValueError(f"fill must be in [0, 255], got {fill}")

        self.augment = augment
        self.image_size = int(image_size)
        self.fill = int(fill)
        self.name = name
        self.normalize = A.Compose(
            [
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0),
                ToTensorV2(),
            ]
        )

    @property
    def is_train(self) -> bool:
        """bool: whether a stochastic augmentation stage is attached."""
        return self.augment is not None

    def set_seed(self, seed: int) -> None:
        """Seed the underlying albumentations RNG for reproducible variants.

        Parameters
        ----------
        seed : int
            Seed forwarded to ``albumentations.Compose.set_random_seed``.

        Notes
        -----
        Albumentations 2.x owns a private RNG per ``Compose``; seeding
        ``random`` or ``numpy.random`` alone does *not* make it deterministic.
        """
        if self.augment is not None:
            self.augment.set_random_seed(int(seed))

    def _reseed_from_entropy(self) -> None:
        """Draw a fresh per-process seed for the stochastic stage.

        Notes
        -----
        Mixes OS entropy with the process id so that sibling DataLoader
        workers, which are unpickled from *identical* bytes at nearly the same
        instant, cannot collide.
        """
        if self.augment is None:
            return
        seed = (int.from_bytes(os.urandom(4), "little") ^ (os.getpid() * 2_654_435_761))
        self.augment.set_random_seed(seed % (2**31 - 1))

    def __setstate__(self, state: dict) -> None:
        """Restore from a pickle and re-seed the stochastic stage.

        Parameters
        ----------
        state : dict
            Pickled instance ``__dict__``.

        Notes
        -----
        This is load-bearing for ``DataLoader(num_workers > 0)``. On Windows
        (and any spawn/forkserver start method) torch pickles the dataset --
        and therefore this transform -- **once** and ships the *same* bytes to
        every worker. Albumentations 2.x stores a live ``random.Random`` per
        transform, and ``random.Random`` pickles its internal state, so without
        this hook every worker would resume from an identical RNG state and
        emit the *identical* augmentation sequence. Augmentation diversity
        would then silently collapse by a factor of ``num_workers`` -- a defect
        that never raises, never shows up in a shape or finiteness check, and
        only manifests as unexplained overfitting.

        Re-seeding here means a pickle round-trip is deliberately *not*
        reproducible. Call :meth:`set_seed` after unpickling when an exactly
        reproducible stream is wanted (e.g. for figures).
        """
        self.__dict__.update(state)
        self._reseed_from_entropy()

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        """Transform one image.

        Parameters
        ----------
        image : numpy.ndarray
            RGB ``uint8`` image of shape ``(H, W, 3)``.

        Returns
        -------
        torch.Tensor
            ``float32`` tensor of shape ``(3, image_size, image_size)``.
        """
        image = _ensure_rgb_uint8(image)
        if self.augment is not None:
            image = self.augment(image=image)["image"]
            image = _ensure_rgb_uint8(image)
        image = letterbox(image, self.image_size, fill=self.fill)
        tensor = self.normalize(image=image)["image"]
        return tensor.float()

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        mode = "train" if self.is_train else "eval"
        return f"ImageTransform(name={self.name!r}, mode={mode}, size={self.image_size})"


# --------------------------------------------------------------------------- #
# Disease pipeline
# --------------------------------------------------------------------------- #
def build_disease_transforms(
    image_size: int = 320,
    train: bool = True,
    strong_color: bool = False,
) -> Callable[[np.ndarray], torch.Tensor]:
    """Build the augmentation pipeline for the disease branch.

    Parameters
    ----------
    image_size : int, optional
        Output side length in pixels. Default ``320``.
    train : bool, optional
        If ``True`` build the stochastic train pipeline, otherwise the
        deterministic eval pipeline (letterbox + normalise only).
    strong_color : bool, optional
        **Ablation only.** If ``True``, append ``HueSaturationValue``
        (``hue_shift_limit=20``) and ``RandomGamma`` to the train pipeline.
        See the warning below before enabling it. Ignored when
        ``train=False``.

    Returns
    -------
    Callable
        An :class:`ImageTransform` mapping an RGB ``uint8`` array to a
        normalised ``float32`` tensor of shape ``(3, image_size, image_size)``.

    Notes
    -----
    **Train pipeline (disease-safe).** Horizontal and vertical flip; affine
    rotation up to +/- 25 deg with mild translation (<= 6 % of the side) and
    mild isotropic scaling (<= 10 %); mild brightness/contrast jitter
    (<= 0.15); and, with low probability (~0.15), either a mild Gaussian blur
    or mild Gaussian noise. Finally :func:`letterbox` to ``image_size``.

    **Deliberate omissions.** Hue shift, channel shuffle, grayscale,
    solarize and aggressive saturation are *not* used by default. All three
    modelled attributes are colour-and-texture dependent:

    - ``black_gill`` is a localised darkening of the gill region -- an
      aggressive brightness or gamma shift can hide it or invent it;
    - ``wssv`` presents as white focal spots on the carapace -- solarize and
      channel shuffle create high-contrast artefacts indistinguishable from
      real spots;
    - ``yellow_head`` is a hue change of the cephalothorax -- a hue shift is
      literally the label, so rotating hue can turn a healthy shrimp into an
      apparently diseased one and vice versa.

    Applying such transforms therefore corrupts the supervision rather than
    regularising the model.

    **``strong_color=True`` is ablation-only.** It exists so that the
    "does colour augmentation hurt?" experiment can be run and reported, and
    it must not be enabled for the production training runs. Expect
    ``yellow_head`` recall in particular to degrade.

    **Scaling and aspect ratio.** ``Affine`` is configured with
    ``keep_ratio=True`` so the x and y scale factors are identical; combined
    with :func:`letterbox` this guarantees the shrimp is never anisotropically
    stretched, which would alter texture statistics.
    """
    if not train:
        return ImageTransform(
            augment=None,
            image_size=image_size,
            fill=LETTERBOX_FILL,
            name="disease/eval",
        )

    ops: list[A.BasicTransform] = [
        # Orientation: shrimp appear in arbitrary orientations on the tray,
        # and none of the three signs is orientation dependent.
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # Mild rigid jitter. fit_output=False keeps the framing (and hence the
        # apparent lesion scale) stable; the limits below are small enough that
        # corner clipping is negligible.
        A.Affine(
            scale=(0.90, 1.10),
            translate_percent=(-0.06, 0.06),
            rotate=(-25.0, 25.0),
            keep_ratio=True,
            fit_output=False,
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_CONSTANT,
            fill=LETTERBOX_FILL,
            p=0.9,
        ),
        # MILD illumination jitter only: models tank-lighting variation without
        # swamping the low-contrast black-gill darkening.
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.5,
        ),
        # Low-probability sensor degradation: blur OR noise, never both.
        A.OneOf(
            [
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.GaussNoise(std_range=(0.02, 0.08), per_channel=True, p=1.0),
            ],
            p=0.15,
        ),
    ]

    if strong_color:
        # ------------------------------------------------------------------ #
        # ABLATION ONLY -- do not enable for production training runs.
        # Hue rotation can erase yellow_head or fabricate it on a healthy
        # animal; gamma shifts can erase or fabricate black_gill. This branch
        # exists solely to quantify that damage in the colour-augmentation
        # ablation experiment.
        # ------------------------------------------------------------------ #
        ops.extend(
            [
                A.HueSaturationValue(
                    hue_shift_limit=20,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=0.5,
                ),
                A.RandomGamma(gamma_limit=(70, 140), p=0.5),
            ]
        )

    name = "disease/train+strong_color(ablation)" if strong_color else "disease/train"
    return ImageTransform(
        augment=A.Compose(ops),
        image_size=image_size,
        fill=LETTERBOX_FILL,
        name=name,
    )


# --------------------------------------------------------------------------- #
# Biomass pipeline
# --------------------------------------------------------------------------- #
def build_biomass_transforms(
    image_size: int = 320,
    train: bool = True,
) -> Callable[[np.ndarray], torch.Tensor]:
    """Build the augmentation pipeline for the biomass regression branch.

    Parameters
    ----------
    image_size : int, optional
        Output side length in pixels. Default ``320``.
    train : bool, optional
        If ``True`` build the stochastic train pipeline, otherwise the
        deterministic eval pipeline (letterbox + normalise only).

    Returns
    -------
    Callable
        An :class:`ImageTransform` mapping an RGB ``uint8`` array to a
        normalised ``float32`` tensor of shape ``(3, image_size, image_size)``.

    Notes
    -----
    **Two hard constraints, both enforced by construction.**

    1. *No random crop of any kind* -- not ``RandomCrop``,
       ``RandomResizedCrop``, ``RandomSizedCrop``, ``CropAndPad`` or
       ``Cutout``. The regression targets are the total length, the
       cephalothorax length and (through ``log_weight``) the overall body
       volume. All three are measured between the rostrum tip and the tail
       fan, so removing either endpoint makes the target unrecoverable from
       the pixels: the network would be trained to hallucinate a length it
       cannot see. Rotation therefore uses ``fit_output=True``, which grows
       the output canvas to contain the entire rotated image rather than
       clipping its corners.
    2. *No aspect-ratio deformation* -- ``Affine`` uses ``keep_ratio=True``
       (identical x and y scale) and the only resize is :func:`letterbox`,
       which scales both axes by the same factor. An anisotropic stretch
       would change the length-to-width ratio, which is exactly the signal the
       weight regressor reads.

    The isotropic scale jitter (+/- 10 %) is safe and in fact desirable:
    letterboxing to a fixed canvas already removes absolute metric scale, so
    the model must infer size from body proportions, and mild scale jitter
    discourages it from latching onto residual pixel-size cues.
    """
    if not train:
        return ImageTransform(
            augment=None,
            image_size=image_size,
            fill=LETTERBOX_FILL,
            name="biomass/eval",
        )

    ops: list[A.BasicTransform] = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # fit_output=True expands the canvas so that rotation cannot clip the
        # rostrum or the tail fan; letterbox then brings it back to a square.
        A.Affine(
            scale=(0.90, 1.10),
            translate_percent=(-0.05, 0.05),
            rotate=(-30.0, 30.0),
            keep_ratio=True,
            fit_output=True,
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_CONSTANT,
            fill=LETTERBOX_FILL,
            p=0.9,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.5,
        ),
        A.OneOf(
            [
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.GaussNoise(std_range=(0.02, 0.06), per_channel=True, p=1.0),
            ],
            p=0.15,
        ),
    ]

    # Guard against a future edit silently reintroducing a cropping transform
    # or an anisotropic resize; both would invalidate the biomass targets.
    _assert_biomass_safe(ops)

    return ImageTransform(
        augment=A.Compose(ops),
        image_size=image_size,
        fill=LETTERBOX_FILL,
        name="biomass/train",
    )


_FORBIDDEN_BIOMASS_SUBSTRINGS: tuple[str, ...] = (
    "crop",
    "cutout",
    "erasing",
    "dropout",
    "resize",
    "distortion",
    "elastic",
    "gridshuffle",
    "perspective",
)


def _assert_biomass_safe(ops: Sequence[A.BasicTransform]) -> None:
    """Assert that no biomass-invalidating transform is present.

    Parameters
    ----------
    ops : sequence of albumentations.BasicTransform
        Transforms about to be composed for the biomass branch.

    Raises
    ------
    AssertionError
        If any transform (or nested transform) crops, erases or otherwise
        deforms the image in a way that would remove the rostrum / tail /
        cephalothorax endpoints or change the aspect ratio.
    """
    # NOTE: these guards raise ``AssertionError`` explicitly rather than using
    # the ``assert`` statement. ``python -O`` strips ``assert``, which would
    # turn this whole function into a silent no-op -- and it is exactly the
    # guard that stops a future edit from reintroducing a crop that invalidates
    # every biomass target. The exception *type* is kept so existing callers
    # catching AssertionError are unaffected.
    for op in ops:
        cls_name = type(op).__name__
        lowered = cls_name.lower()
        for bad in _FORBIDDEN_BIOMASS_SUBSTRINGS:
            if bad in lowered:
                raise AssertionError(
                    f"{cls_name} is forbidden in the biomass pipeline: cropping or "
                    "deforming the image removes the size cues the regression "
                    "targets are defined by."
                )
        if isinstance(op, A.Affine) and op.keep_ratio is not True:
            raise AssertionError(
                "biomass Affine must use keep_ratio=True; an anisotropic scale "
                "changes the length-to-width ratio the weight head relies on."
            )
        nested = getattr(op, "transforms", None)
        if nested:
            _assert_biomass_safe(nested)


# --------------------------------------------------------------------------- #
# Visual diagnostics
# --------------------------------------------------------------------------- #
def save_augmentation_grid(
    transform: Callable[[np.ndarray], torch.Tensor],
    image_paths: Sequence[str],
    out_path: str,
    n_variants: int = 5,
    seed: int = 0,
) -> str:
    """Render a rows=images x cols=variants augmentation grid to a PNG.

    Parameters
    ----------
    transform : Callable
        Transform to visualise, e.g. the result of
        :func:`build_disease_transforms`.
    image_paths : sequence of str
        Source image paths, one per grid row. Must be non-empty.
    out_path : str
        Destination PNG path. Parent directories are created if needed.
    n_variants : int, optional
        Number of augmented variants per source image (grid columns).
    seed : int, optional
        Seed forwarded to the transform's RNG so the figure is reproducible.

    Returns
    -------
    str
        ``out_path``, for convenient chaining.

    Raises
    ------
    ValueError
        If ``image_paths`` is empty, ``n_variants < 1``, or an image cannot
        be read.

    Notes
    -----
    Uses the non-interactive ``Agg`` matplotlib backend so it is safe on
    headless machines. Column titles show the **basename** only -- absolute
    filesystem paths are never rendered into the figure, since these PNGs get
    pasted into reports and issue threads.
    """
    if len(image_paths) == 0:
        raise ValueError("image_paths must contain at least one path")
    if int(n_variants) < 1:
        raise ValueError(f"n_variants must be >= 1, got {n_variants}")
    n_variants = int(n_variants)

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    set_seed = getattr(transform, "set_seed", None)
    if callable(set_seed):
        set_seed(seed)

    n_rows = len(image_paths)
    fig, axes = plt.subplots(
        n_rows,
        n_variants,
        figsize=(2.2 * n_variants, 2.4 * n_rows),
        squeeze=False,
    )

    try:
        for row, path in enumerate(image_paths):
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"could not read image: {os.path.basename(str(path))}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            # Basename only: never leak absolute filesystem paths into figures.
            label = os.path.basename(str(path))

            for col in range(n_variants):
                tensor = transform(rgb)
                shown = denormalize_to_uint8(tensor)
                ax = axes[row][col]
                ax.imshow(shown)
                ax.set_xticks([])
                ax.set_yticks([])
                if col == 0:
                    ax.set_ylabel(label, fontsize=7)
                if row == 0:
                    ax.set_title(f"variant {col}", fontsize=7)

        fig.suptitle(f"augmentation grid: {getattr(transform, 'name', 'transform')}", fontsize=10)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))

        parent = os.path.dirname(os.path.abspath(out_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        fig.savefig(out_path, dpi=110)
    finally:
        plt.close(fig)

    return out_path


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def _synthetic_image(height: int, width: int, seed: int = 0) -> np.ndarray:
    """Build a synthetic RGB image with a known square and colour blobs.

    Parameters
    ----------
    height, width : int
        Image dimensions in pixels.
    seed : int, optional
        Seed for the background texture.

    Returns
    -------
    numpy.ndarray
        ``uint8`` RGB image of shape ``(height, width, 3)``.
    """
    rng = np.random.default_rng(seed)
    img = rng.integers(20, 70, size=(height, width, 3), dtype=np.uint8)
    side = max(8, int(0.4 * min(height, width)))
    y0 = (height - side) // 2
    x0 = (width - side) // 2
    img[y0 : y0 + side, x0 : x0 + side] = 255  # pure-white square marker
    return img


def _measure_square(canvas: np.ndarray, threshold: int = 128) -> tuple[int, int]:
    """Measure the bounding box of the bright square in a letterboxed canvas.

    Parameters
    ----------
    canvas : numpy.ndarray
        ``uint8`` RGB image.
    threshold : int, optional
        Brightness threshold separating the square from background and padding.

    Returns
    -------
    tuple of int
        ``(width, height)`` of the detected bright region, in pixels.
    """
    gray = canvas.min(axis=2)  # white square is bright in all channels
    mask = gray > threshold
    cols = np.where(mask.any(axis=0))[0]
    rows = np.where(mask.any(axis=1))[0]
    if cols.size == 0 or rows.size == 0:
        return 0, 0
    return int(cols[-1] - cols[0] + 1), int(rows[-1] - rows[0] + 1)


def _self_test() -> None:
    """Run the module self-test; raises AssertionError on any failure."""
    import tempfile

    checks = 0

    def ok(msg: str) -> None:
        nonlocal checks
        checks += 1
        print(f"  [ok] {msg}")

    print("=" * 72)
    print("transforms.py self-test")
    print("=" * 72)

    shapes = [(480, 640), (900, 300), (300, 900), (512, 512)]
    images = {f"{h}x{w}": _synthetic_image(h, w, seed=i) for i, (h, w) in enumerate(shapes)}

    # ---------------- letterbox ---------------- #
    print("\n[1] letterbox: exact shape, aspect preservation, symmetric padding")
    for size in (320, 224, 384):
        for tag, img in images.items():
            out = letterbox(img, size)
            assert out.shape == (size, size, 3), f"{tag}@{size}: got {out.shape}"
            assert out.dtype == np.uint8, f"{tag}@{size}: dtype {out.dtype}"

            h, w = img.shape[:2]
            ratio = min(size / h, size / w)
            exp_w, exp_h = round(w * ratio), round(h * ratio)

            sq_w, sq_h = _measure_square(out)
            assert sq_w > 0 and sq_h > 0, f"{tag}@{size}: square not found"
            assert abs(sq_w - sq_h) <= 1, (
                f"{tag}@{size}: source square became {sq_w}x{sq_h} "
                f"(aspect ratio distorted by {abs(sq_w - sq_h)} px)"
            )

            src_side = max(8, int(0.4 * min(h, w)))
            expected_side = src_side * ratio
            assert abs(sq_w - expected_side) <= 2.0, (
                f"{tag}@{size}: square width {sq_w} vs expected {expected_side:.2f}"
            )

            # padding must be symmetric to within one pixel
            pad_w, pad_h = size - exp_w, size - exp_h
            assert abs((pad_w // 2) - (pad_w - pad_w // 2)) <= 1
            assert abs((pad_h // 2) - (pad_h - pad_h // 2)) <= 1
    ok(f"letterbox exact (size,size,3) + square stays square (<=1 px) for {len(shapes)} aspect ratios x 3 sizes")

    tall = letterbox(images["900x300"], 320)
    assert (tall[:, 0, :] == LETTERBOX_FILL).all(), "expected left column to be pad fill"
    wide = letterbox(images["300x900"], 320)
    assert (wide[0, :, :] == LETTERBOX_FILL).all(), "expected top row to be pad fill"
    ok("extreme aspect ratios (900x300, 300x900) pad on the correct axis with fill=114")

    gray_in = letterbox(images["480x640"][:, :, 0], 256)
    assert gray_in.shape == (256, 256, 3)
    rgba_in = letterbox(np.dstack([images["480x640"], np.full((480, 640, 1), 255, np.uint8)]), 256)
    assert rgba_in.shape == (256, 256, 3)
    ok("grayscale and RGBA inputs coerced to (S,S,3)")

    # ---------------- transform shapes / finiteness ---------------- #
    print("\n[2] transform outputs: dtype, shape, finiteness")
    size = 320
    builders = {
        "disease/train": build_disease_transforms(size, train=True),
        "disease/train+strong_color": build_disease_transforms(size, train=True, strong_color=True),
        "disease/eval": build_disease_transforms(size, train=False),
        "biomass/train": build_biomass_transforms(size, train=True),
        "biomass/eval": build_biomass_transforms(size, train=False),
    }
    for name, tf in builders.items():
        for tag, img in images.items():
            out = tf(img)
            assert isinstance(out, torch.Tensor), f"{name}/{tag}: not a tensor"
            assert out.dtype == torch.float32, f"{name}/{tag}: dtype {out.dtype}"
            assert tuple(out.shape) == (3, size, size), f"{name}/{tag}: shape {tuple(out.shape)}"
            assert torch.isfinite(out).all(), f"{name}/{tag}: non-finite values"
    ok(f"{len(builders)} pipelines x {len(images)} images -> torch.float32 (3,{size},{size}), all finite")

    for isz in (224, 256, 384):
        out = build_disease_transforms(isz, train=True)(images["480x640"])
        assert tuple(out.shape) == (3, isz, isz), f"image_size={isz}: {tuple(out.shape)}"
    ok("image_size argument honoured for 224 / 256 / 384")

    # ---------------- determinism / stochasticity ---------------- #
    print("\n[3] eval determinism and train stochasticity")
    for name in ("disease/eval", "biomass/eval"):
        tf = builders[name]
        for tag, img in images.items():
            a, b = tf(img), tf(img)
            assert torch.equal(a, b), f"{name}/{tag}: eval transform is not deterministic"
    ok("eval transforms bit-identical across repeated calls (all images)")

    for name in ("disease/train", "biomass/train"):
        tf = builders[name]
        img = images["480x640"]
        diffs = [not torch.equal(tf(img), tf(img)) for _ in range(8)]
        assert any(diffs), f"{name}: 8 paired calls produced identical output"
        assert sum(diffs) >= 6, f"{name}: only {sum(diffs)}/8 paired calls differed"
    ok("train transforms stochastic (>=6/8 paired calls differ)")

    dis_train = builders["disease/train"]
    dis_train.set_seed(1234)
    first = [dis_train(images["480x640"]) for _ in range(4)]
    dis_train.set_seed(1234)
    second = [dis_train(images["480x640"]) for _ in range(4)]
    assert all(torch.equal(a, b) for a, b in zip(first, second)), "set_seed did not make train reproducible"
    ok("set_seed makes the train pipeline reproducible")

    # ---------------- disease colour-safety contract ---------------- #
    print("\n[4] disease default pipeline excludes colour-destroying transforms")
    banned = {"HueSaturationValue", "RandomGamma", "ChannelShuffle", "ToGray",
              "Solarize", "RGBShift", "ColorJitter", "InvertImg", "FancyPCA"}

    def op_names(compose: A.Compose) -> set[str]:
        found: set[str] = set()
        stack = list(compose.transforms)
        while stack:
            op = stack.pop()
            found.add(type(op).__name__)
            stack.extend(getattr(op, "transforms", []) or [])
        return found

    default_ops = op_names(builders["disease/train"].augment)
    assert not (default_ops & banned), f"colour-unsafe ops leaked into default: {default_ops & banned}"
    ok(f"default disease train ops = {sorted(default_ops)}")

    ablation_ops = op_names(builders["disease/train+strong_color"].augment)
    assert "HueSaturationValue" in ablation_ops and "RandomGamma" in ablation_ops
    ok("strong_color=True (ablation-only) adds HueSaturationValue + RandomGamma")

    # ---------------- biomass safety contract ---------------- #
    print("\n[5] biomass pipeline forbids crops and aspect deformation")
    bio_ops = op_names(builders["biomass/train"].augment)
    assert not any("crop" in n.lower() or "cutout" in n.lower() for n in bio_ops), bio_ops
    ok(f"no crop/cutout in biomass train ops = {sorted(bio_ops)}")

    try:
        _assert_biomass_safe([A.RandomResizedCrop(size=(320, 320), p=1.0)])
    except AssertionError:
        ok("_assert_biomass_safe rejects RandomResizedCrop")
    else:  # pragma: no cover
        raise AssertionError("_assert_biomass_safe failed to reject RandomResizedCrop")

    try:
        _assert_biomass_safe([A.Affine(scale=(0.9, 1.1), keep_ratio=False, p=1.0)])
    except AssertionError:
        ok("_assert_biomass_safe rejects Affine(keep_ratio=False)")
    else:  # pragma: no cover
        raise AssertionError("_assert_biomass_safe failed to reject anisotropic Affine")

    # `python -O` strips `assert` statements. The biomass guard is the only
    # thing standing between a future edit and silently invalidated regression
    # targets, so it must raise explicitly rather than via `assert`.
    import subprocess
    import sys

    probe = (
        "import sys, albumentations as A;"
        f"sys.path.insert(0, {os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))!r});"
        "from src.data.transforms import _assert_biomass_safe;"
        "\ntry:\n"
        "    _assert_biomass_safe([A.RandomResizedCrop(size=(320,320), p=1.0)])\n"
        "    print('BYPASSED')\n"
        "except AssertionError:\n"
        "    print('HELD')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-O", "-c", probe], capture_output=True, text=True
    )
    assert "HELD" in proc.stdout, (
        "biomass crop guard is stripped by `python -O` "
        f"(stdout={proc.stdout!r}, stderr={proc.stderr[-300:]!r})"
    )
    ok("biomass safety guards survive `python -O` (not bare asserts)")

    # ---------------- picklability (Windows spawn DataLoader) ---------------- #
    print("\n[6] picklability for num_workers>0 on Windows (spawn)")
    # Safe use of pickle: this round-trips objects this process just built, in
    # order to verify they survive the spawn-based DataLoader worker handoff.
    # No externally supplied or untrusted pickle data is ever loaded here.
    import pickle

    for name, tf in builders.items():
        restored = pickle.loads(pickle.dumps(tf))
        out = restored(images["480x640"])
        assert tuple(out.shape) == (3, size, size), name
    ok("all pipelines survive pickle round-trip and still run")

    # Spawn ships the SAME pickled bytes to every worker. If the albumentations
    # RNG state rode along inside the pickle, every worker would replay one
    # identical augmentation stream and diversity would collapse silently.
    small = images["480x640"]
    for name in ("disease/train", "biomass/train"):
        blob = pickle.dumps(builders[name])
        worker_a, worker_b = pickle.loads(blob), pickle.loads(blob)
        collisions = sum(
            bool(torch.equal(worker_a(small), worker_b(small))) for _ in range(6)
        )
        assert collisions <= 1, (
            f"{name}: {collisions}/6 augmentations identical across two unpickled "
            "workers -- DataLoader(num_workers>0) would train on a correlated stream"
        )
    ok("unpickled worker copies are decorrelated (no shared augmentation stream)")

    # ---------------- aliasing / input safety ---------------- #
    print("\n[6b] no aliasing of caller buffers, non-finite input handling")
    already_sized = np.ascontiguousarray(_synthetic_image(size, size, seed=9))
    pristine = already_sized.copy()
    lb_out = letterbox(already_sized, size)
    lb_out[:] = 0
    assert np.array_equal(already_sized, pristine), (
        "letterbox returned a view of its input: writing to the result corrupted "
        "the caller's array"
    )
    ok("letterbox never aliases the caller's buffer (even at exact output size)")

    nan_src = _synthetic_image(120, 160, seed=10).astype(np.float32) / 255.0
    nan_src[10:30, 10:30, :] = np.nan
    nan_src[40:60, 40:60, :] = np.inf
    lb_nan = letterbox(nan_src, 128)
    assert lb_nan.dtype == np.uint8 and lb_nan.shape == (128, 128, 3)
    tensor_nan = builders["disease/eval"](nan_src)
    assert torch.isfinite(tensor_nan).all(), "non-finite input produced a non-finite tensor"
    ok("NaN/inf pixels map to neutral grey, never to fabricated black/white signs")

    for bad_fill in (-1, 256, 999):
        try:
            letterbox(images["480x640"], 64, fill=bad_fill)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f"fill={bad_fill} must be rejected, not saturated by cv2")
    for bad_size in (0, -5):
        try:
            build_disease_transforms(bad_size, train=True)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f"image_size={bad_size} must be rejected at build time")
    ok("out-of-range fill and non-positive image_size rejected eagerly")

    # ---------------- augmentation grid ---------------- #
    print("\n[7] augmentation grid PNG")
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    fig_dir = os.path.join(repo_root, "outputs", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    grid_path = os.path.join(fig_dir, "_selftest_aug_grid.png")

    with tempfile.TemporaryDirectory() as tmp:
        src_paths = []
        for tag in ("480x640", "900x300", "300x900"):
            p = os.path.join(tmp, f"shrimp_{tag}.png")
            cv2.imwrite(p, cv2.cvtColor(images[tag], cv2.COLOR_RGB2BGR))
            src_paths.append(p)

        returned = save_augmentation_grid(
            builders["disease/train"], src_paths, grid_path, n_variants=5, seed=0
        )

    assert returned == grid_path, "save_augmentation_grid must return out_path"
    assert os.path.isfile(grid_path), f"grid PNG not written: {grid_path}"
    assert os.path.getsize(grid_path) > 5_000, "grid PNG suspiciously small"
    ok(f"wrote outputs/figures/_selftest_aug_grid.png ({os.path.getsize(grid_path)} bytes)")

    try:
        save_augmentation_grid(builders["disease/train"], [], grid_path)
    except ValueError:
        ok("save_augmentation_grid raises ValueError on empty image_paths")
    else:  # pragma: no cover
        raise AssertionError("empty image_paths should raise ValueError")

    print("\n" + "=" * 72)
    print(f"ALL {checks} ASSERTION GROUPS PASSED")
    print("=" * 72)


if __name__ == "__main__":  # pragma: no cover
    _self_test()
