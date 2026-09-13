"""Dynamic-content masks: smoke, fire, dust and broadcast graphics.

Registration must only use pixels that show *static, georeferenced* structure.
On 2001-09-11 footage that is a minority of the frame: after 09:03 the plume
covers a large part of most shots, dust fills the streets after each collapse,
and almost every broadcast frame carries a station bug, a lower-third and a
ticker painted over the image.

Everything here is a **heuristic**, deliberately so: it has to run on CPU, in
CI, with no model weights, on 480i footage of unknown colourimetry.  The
interface (:class:`MaskSegmenter`) is the point -- it is shaped so a learned
segmenter can be dropped in later without changing any caller.

Conventions
-----------
* A mask is a boolean array, ``True`` = **exclude this pixel**.
* Images are float in [0, 1], either (H, W) grayscale or (H, W, 3) RGB.
* A *stack* is (T, H, W[, 3]) of frames from the **same shot**, used for the
  temporal cues.  Frames need not be consecutive; sampling every ~10th frame
  over a few seconds works better than consecutive frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

__all__ = [
    "MaskConfig",
    "MaskSegmenter",
    "HeuristicSegmenter",
    "dynamic_mask",
    "fire_mask",
    "rgb_to_hsv",
    "sky_mask",
    "smoke_mask",
    "static_graphics_mask",
    "temporal_variance_mask",
]


# --- small image helpers -------------------------------------------------------
def _as_gray(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img, dtype=np.float64)
    if a.ndim == 3:
        return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    return a


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """(H, W, 3) RGB in [0, 1] -> HSV with H in degrees [0, 360), S, V in [0, 1]."""
    a = np.asarray(rgb, dtype=np.float64)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx = a.max(axis=-1)
    mn = a.min(axis=-1)
    d = mx - mn
    h = np.zeros_like(mx)
    nz = d > 1e-12
    idx = nz & (mx == r)
    h[idx] = (60.0 * ((g - b)[idx] / d[idx])) % 360.0
    idx = nz & (mx == g)
    h[idx] = 60.0 * ((b - r)[idx] / d[idx]) + 120.0
    idx = nz & (mx == b)
    h[idx] = 60.0 * ((r - g)[idx] / d[idx]) + 240.0
    s = np.where(mx > 1e-12, d / np.where(mx > 1e-12, mx, 1.0), 0.0)
    return np.stack([h, s, mx], axis=-1)


def _box_blur(img: np.ndarray, radius: int) -> np.ndarray:
    """Separable box blur via a summed-area table (no SciPy needed)."""
    if radius <= 0:
        return img.astype(np.float64)
    a = np.asarray(img, dtype=np.float64)
    pad = np.pad(a, radius, mode="edge")
    cs = np.cumsum(np.cumsum(pad, axis=0), axis=1)
    cs = np.pad(cs, ((1, 0), (1, 0)))
    k = 2 * radius + 1
    h, w = a.shape
    out = cs[k:, k:] - cs[:h, k:] - cs[k:, :w] + cs[:h, :w]
    return out / (k * k)


def local_std(img: np.ndarray, radius: int = 3) -> np.ndarray:
    """Local standard deviation of a grayscale image (texture strength)."""
    g = _as_gray(img)
    mean = _box_blur(g, radius)
    mean_sq = _box_blur(g * g, radius)
    return np.sqrt(np.clip(mean_sq - mean * mean, 0.0, None))


def _gradient_magnitude(img: np.ndarray) -> np.ndarray:
    g = _as_gray(img)
    gy, gx = np.gradient(g)
    return np.hypot(gx, gy)


def _grow(seed: np.ndarray, allowed: np.ndarray, iters: int) -> np.ndarray:
    """Dilate ``seed`` ``iters`` times, constrained to ``allowed`` (4-connected)."""
    cur = seed & allowed
    for _ in range(max(0, iters)):
        grown = cur.copy()
        grown[1:] |= cur[:-1]
        grown[:-1] |= cur[1:]
        grown[:, 1:] |= cur[:, :-1]
        grown[:, :-1] |= cur[:, 1:]
        cur = grown & allowed
    return cur


def _flood_from_top(seed_ok: np.ndarray, max_iter: int = 400) -> np.ndarray:
    """Connected region of ``seed_ok`` reachable from the top row (4-connected)."""
    cur = np.zeros_like(seed_ok, dtype=bool)
    cur[0] = seed_ok[0]
    for _ in range(max_iter):
        grown = cur.copy()
        grown[1:] |= cur[:-1]
        grown[:-1] |= cur[1:]
        grown[:, 1:] |= cur[:, :-1]
        grown[:, :-1] |= cur[:, 1:]
        grown &= seed_ok
        if grown.sum() == cur.sum():
            return grown
        cur = grown
    return cur


# --- configuration ----------------------------------------------------------------
@dataclass
class MaskConfig:
    """Thresholds for the heuristics.  All are on [0, 1] image values."""

    smoke_min_value: float = 0.40
    """Bright smoke: grey-white plume and dust clouds."""
    smoke_max_saturation: float = 0.22
    """Smoke is nearly colourless; sky is too, hence the texture test below."""
    smoke_max_texture: float = 0.055
    """Smoke is *soft*: little high-frequency detail compared with a facade."""
    dark_smoke_max_value: float = 0.22
    """Dark smoke over the fires; also matches shadow, so it is used only
    where the temporal cue agrees."""
    fire_min_saturation: float = 0.45
    fire_hue_deg: tuple[float, float] = (0.0, 45.0)
    fire_min_value: float = 0.45
    temporal_std_dynamic: float = 0.055
    """Temporal standard deviation above which a pixel is considered moving."""
    graphics_max_temporal_std: float = 0.012
    """Below this a pixel never changes across the shot."""
    graphics_min_gradient: float = 0.12
    """Graphics are sharp: real static scene content in SD footage is not."""
    sky_min_value: float = 0.45
    sky_max_texture: float = 0.05
    dilate_radius: int = 2
    """Grow the final mask slightly: the boundary of a plume is the worst place
    to trust a pixel."""


# --- individual cues ----------------------------------------------------------------
def sky_mask(image: np.ndarray, config: MaskConfig | None = None) -> np.ndarray:
    """Pixels that are open sky.

    Sky is bright, smooth and connected to the top of the frame.  This is *not*
    a dynamic mask -- sky is the most useful background there is, because the
    building/sky boundary is exactly the silhouette
    :mod:`wtc4d.camreg.render_match` matches against.  It is returned
    separately so callers can use it as a cue rather than a rejection.
    """
    cfg = config or MaskConfig()
    g = _as_gray(image)
    radius = 3
    smooth = local_std(g, radius) < cfg.sky_max_texture
    bright = g > cfg.sky_min_value
    core = _flood_from_top(bright & smooth)
    # The texture test rejects a band of sky ``radius`` pixels wide along every
    # silhouette, which would bias the skyline upwards by that much.  Grow the
    # core back into merely-bright pixels to put the boundary where the
    # brightness actually crosses over.
    return _grow(core, bright, radius + 1)


def skyline_profile(image: np.ndarray, config: MaskConfig | None = None) -> np.ndarray:
    """Row index of the first non-sky pixel in each column, as float (W,).

    Three distinct values matter to the caller:

    * a finite row -- the silhouette is here;
    * ``inf`` -- the column is sky all the way down, so there is *nothing* in
      front of the camera in that direction.  This is positive evidence, not
      missing data: a model that puts a building in such a column is wrong;
    * ``nan`` -- unknown, because the topmost pixel is already not sky (a
      building, a helicopter's window frame or smoke fills the top of the
      frame), so the column says nothing either way.
    """
    sky = sky_mask(image, config)
    first_non_sky = np.argmax(~sky, axis=0).astype(np.float64)
    out = np.where(sky.all(axis=0), np.inf, first_non_sky)
    out[~sky[0]] = np.nan  # the column starts on something opaque: no information
    return out


def smoke_mask(
    image: np.ndarray, config: MaskConfig | None = None, sky: np.ndarray | None = None
) -> np.ndarray:
    """Smoke, dust and haze: bright, desaturated, low-texture, *not* sky."""
    cfg = config or MaskConfig()
    g = _as_gray(image)
    tex = local_std(g, 3)
    if np.asarray(image).ndim == 3:
        sat = rgb_to_hsv(np.asarray(image, dtype=np.float64))[..., 1]
    else:
        sat = np.zeros_like(g)  # a grayscale frame tells us nothing about colour
    bright_soft = (g > cfg.smoke_min_value) & (sat < cfg.smoke_max_saturation)
    bright_soft &= tex < cfg.smoke_max_texture
    if sky is None:
        sky = sky_mask(image, cfg)
    return bright_soft & ~sky


def fire_mask(image: np.ndarray, config: MaskConfig | None = None) -> np.ndarray:
    """Flame and glowing debris: saturated red-orange, bright.  RGB only."""
    cfg = config or MaskConfig()
    a = np.asarray(image, dtype=np.float64)
    if a.ndim != 3:
        return np.zeros(a.shape[:2], dtype=bool)
    hsv = rgb_to_hsv(a)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    lo, hi = cfg.fire_hue_deg
    hue_ok = (h >= lo) & (h <= hi)
    return hue_ok & (s > cfg.fire_min_saturation) & (v > cfg.fire_min_value)


def temporal_variance_mask(stack: np.ndarray, config: MaskConfig | None = None) -> np.ndarray:
    """Pixels whose value changes across the shot: smoke, debris, people, traffic.

    Only meaningful for a **static** camera.  For a moving camera every edge
    changes, so this returns the union of "moving content" and "camera motion";
    :func:`dynamic_mask` therefore only uses it when told the camera is static.
    """
    cfg = config or MaskConfig()
    arr = np.asarray(stack, dtype=np.float64)
    if arr.ndim == 4:
        arr = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    if arr.shape[0] < 2:
        return np.zeros(arr.shape[1:], dtype=bool)
    return arr.std(axis=0) > cfg.temporal_std_dynamic


def static_graphics_mask(stack: np.ndarray, config: MaskConfig | None = None) -> np.ndarray:
    """Burnt-in broadcast graphics: station bug, lower-third, ticker, clock.

    A graphic is *perfectly* constant across a shot -- it is composited after
    the camera, so it carries none of the scene's noise, motion or grain -- and
    it is *sharp*, unlike anything in a 480i telecine of a distant building.
    Both tests are needed: on a locked-off tripod shot the whole scene is
    temporally constant, and the sharpness test is then what separates the
    ticker from the skyline.  Expect false negatives on translucent graphics
    and false positives on high-contrast near-field structure (a railing in
    front of a tripod).
    """
    cfg = config or MaskConfig()
    arr = np.asarray(stack, dtype=np.float64)
    if arr.ndim == 4:
        gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    else:
        gray = arr
    if gray.shape[0] < 2:
        return np.zeros(gray.shape[1:], dtype=bool)
    frozen = gray.std(axis=0) < cfg.graphics_max_temporal_std
    sharp = _gradient_magnitude(gray.mean(axis=0)) > cfg.graphics_min_gradient
    return frozen & sharp


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    return _box_blur(mask.astype(np.float64), radius) > 1e-9


# --- the public entry point ----------------------------------------------------------
def dynamic_mask(
    image: np.ndarray,
    stack: np.ndarray | None = None,
    camera_static: bool = False,
    config: MaskConfig | None = None,
) -> np.ndarray:
    """Everything registration must ignore in this frame.

    Parameters
    ----------
    image : the frame to mask, (H, W) or (H, W, 3) in [0, 1].
    stack : frames from the same shot for the temporal cues, or ``None``.
    camera_static : set True for a tripod/locked-off shot, which enables the
        temporal-variance cue (meaningless for a moving camera).

    Returns
    -------
    (H, W) bool, True where the pixel is dynamic or synthetic.
    """
    cfg = config or MaskConfig()
    sky = sky_mask(image, cfg)
    mask = smoke_mask(image, cfg, sky=sky) | fire_mask(image, cfg)
    if stack is not None and len(stack) >= 2:
        mask |= static_graphics_mask(stack, cfg)
        if camera_static:
            mask |= temporal_variance_mask(stack, cfg)
    return _dilate(mask, cfg.dilate_radius)


class MaskSegmenter(Protocol):
    """Swap-in point for a learned segmenter.

    Implementations take the frame (and optionally a stack of frames from the
    same shot) and return a boolean mask, ``True`` = exclude.  A future learned
    version -- a fine-tuned SAM/Segformer trained on a few hundred hand-labelled
    2001 frames -- should satisfy exactly this protocol and be selectable from
    the CLI, so nothing downstream changes.
    """

    def __call__(
        self, image: np.ndarray, stack: np.ndarray | None = None, camera_static: bool = False
    ) -> np.ndarray: ...


@dataclass
class HeuristicSegmenter:
    """The default :class:`MaskSegmenter`, wrapping :func:`dynamic_mask`."""

    config: MaskConfig = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.config is None:
            self.config = MaskConfig()

    def __call__(
        self, image: np.ndarray, stack: np.ndarray | None = None, camera_static: bool = False
    ) -> np.ndarray:
        return dynamic_mask(image, stack, camera_static=camera_static, config=self.config)
