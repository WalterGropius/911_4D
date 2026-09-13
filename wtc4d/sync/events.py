"""Detecting the day's anchor events in a shot, from image evidence alone.

Four moments have times good to a couple of seconds from NIST NCSTAR 1 (see
:mod:`wtc4d.timeline`), and each leaves an unmistakable signature in any shot
that has the towers in frame:

* **impact fireball** -- a one- or two-frame jump in luminance and in the area
  of fire-coloured pixels inside the tower region, decaying over a second or
  two into a black smoke plume;
* **collapse onset** -- the region stops being mostly static and starts moving
  *downwards*: the changed-pixel fraction climbs and the centroid of the
  change mask accelerates down the image.

Both are detected on a caller-supplied region of interest.  Getting that
region from a registered camera is the camreg workstream's job; the interface
is :class:`~wtc4d.sync.types.RoiTrack`, a frame-indexed bounding box, so a
static camera needs one box and a moving one needs a few keyframes.  With no
ROI at all the whole frame is used, which works for tight shots and fails
noisily on wide ones.

What has to be rejected
-----------------------
A shot cut looks exactly like a flash to a naive brightness detector, and
2001 news footage cuts every few seconds.  Every candidate is therefore
checked against a **global** scene-change signal computed outside the ROI: a
real fireball changes the ROI far more than the rest of the frame, a cut
changes both equally.  That check alone is not enough, though: a lower-third
graphic template (globe, ticker, "BREAKING NEWS" strap) keeps the *rest* of
the frame static even during a hard cut or a fade-to-black, because only the
live-video sub-window changes -- exactly like a real event would.  Two more
checks handle that case for the collapse detector, which is the one exposed
to it in practice: **persistence** (a cut or a feed dropout is one bright or
black frame that then goes flat; a real collapse keeps the ROI churning for
seconds) and a **luminance floor** (a fade-to-black transition passes through
near-zero mean luminance; falling debris and an expanding dust cloud are lit
and never do).  Both were tuned against real footage -- see "Validation" in
``wtc4d/sync/README.md`` for the false positives they were built to catch.

Known bias, not just noise
---------------------------
The collapse detector fires when the *visible* debris/dust signal clears its
threshold, which lags the structural onset NIST anchors on: initial buckling
is a subtle motion of already-standing columns, and only the following
seconds of violent debris ejection are strong enough to trigger a
pixel-change threshold reliably.  On the one real collapse window this was
validated against (see the README) that lag was about **6-7 seconds**, one-
directional (always late, never early).  This is why :func:`match_to_anchors`
reports a sigma from the anchor and the detection, not from this lag, and why
event-anchor estimates should be weighted down (or their sigma widened)
relative to a clock-OCR or broadcast-metadata estimate in the fusion rather
than trusted alone for the project's <=1 s target.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel, Field

from wtc4d.schema.time import TimeEstimate, TimeMethod
from wtc4d.sync.types import BBox, EventCandidate, EventKind, RoiTrack
from wtc4d.sync.video import DecodedFrame, FrameSource
from wtc4d.timeline import EVENTS_BY_ID, Event, fmt_local

# Fire-coloured pixels: warm hue, strongly saturated, bright.  OpenCV hue is
# 0-179, so 35 is roughly yellow; a 2001 broadcast fireball lands in 5-30.
FIRE_HUE_MAX = 35
FIRE_SAT_MIN = 80
FIRE_VAL_MIN = 110


@dataclass
class FrameFeatures:
    """Per-frame scalars used by both detectors."""

    index: list[int] = field(default_factory=list)
    t_rel: list[float] = field(default_factory=list)
    roi_mean: list[float] = field(default_factory=list)
    roi_fire: list[float] = field(default_factory=list)
    roi_change: list[float] = field(default_factory=list)
    roi_change_cy: list[float] = field(default_factory=list)
    outside_change: list[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.index)

    def as_arrays(self) -> dict[str, np.ndarray]:
        return {
            k: np.asarray(v, dtype=np.float64)
            for k, v in (
                ("t_rel", self.t_rel),
                ("roi_mean", self.roi_mean),
                ("roi_fire", self.roi_fire),
                ("roi_change", self.roi_change),
                ("roi_change_cy", self.roi_change_cy),
                ("outside_change", self.outside_change),
            )
        }


def _roi_for(roi: RoiTrack | BBox | None, frame_idx: int, w: int, h: int) -> BBox:
    if roi is None:
        return BBox(x=0, y=0, w=w, h=h)
    box = roi if isinstance(roi, BBox) else roi.at(frame_idx)
    if box is None:
        return BBox(x=0, y=0, w=w, h=h)
    return box.clip(w, h)


def compute_features(
    frames: Iterable[DecodedFrame],
    roi: RoiTrack | BBox | None = None,
    *,
    change_threshold: int = 18,
) -> FrameFeatures:
    """Walk a frame sequence once and collect the per-frame scalars.

    ``change_threshold`` is an 8-bit luminance difference; 18 is well above
    MPEG-2 mosquito noise on this material and well below any real motion.
    """
    import cv2

    feats = FrameFeatures()
    prev_gray: np.ndarray | None = None
    prev_shape: tuple[int, int] | None = None
    for fr in frames:
        img = fr.image
        if img.ndim == 2:
            bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            gray = img
        else:
            bgr = img
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        box = _roi_for(roi, fr.index, w, h)
        sl = box.as_slices()

        roi_gray = gray[sl]
        hsv = cv2.cvtColor(bgr[sl], cv2.COLOR_BGR2HSV)
        fire = (
            (hsv[:, :, 0] <= FIRE_HUE_MAX)
            & (hsv[:, :, 1] >= FIRE_SAT_MIN)
            & (hsv[:, :, 2] >= FIRE_VAL_MIN)
        )

        change = 0.0
        change_cy = 0.5
        outside = 0.0
        if prev_gray is not None and prev_shape == (h, w):
            diff = cv2.absdiff(gray, prev_gray)
            mask = diff >= change_threshold
            roi_mask = mask[sl]
            change = float(roi_mask.mean())
            if roi_mask.any():
                # vertical centroid of the change mask, normalised to 0..1
                # within the ROI so it is comparable across ROI sizes
                weights = roi_mask.sum(axis=1).astype(np.float64)
                rows = np.arange(len(weights), dtype=np.float64)
                change_cy = float((rows * weights).sum() / weights.sum() / max(1, len(weights)))
            full = float(mask.mean())
            roi_px = box.w * box.h
            total_px = w * h
            out_px = max(1, total_px - roi_px)
            outside = float((full * total_px - change * roi_px) / out_px)

        feats.index.append(fr.index)
        feats.t_rel.append(fr.t_rel)
        feats.roi_mean.append(float(roi_gray.mean()))
        feats.roi_fire.append(float(fire.mean()))
        feats.roi_change.append(change)
        feats.roi_change_cy.append(change_cy)
        feats.outside_change.append(outside)
        prev_gray = gray
        prev_shape = (h, w)
    return feats


def _robust_z(x: np.ndarray) -> np.ndarray:
    """Median/MAD z-score; robust to the event itself dominating the series."""
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    scale = 1.4826 * mad
    if scale < 1e-9:
        scale = float(x.std()) or 1.0
    return (x - med) / scale


class DetectorConfig(BaseModel):
    """Thresholds, exposed so a caller can loosen them for odd material."""

    flash_z: float = Field(default=6.0, description="robust z of d(fire fraction)/dt")
    flash_min_fire_rise: float = Field(
        default=0.002, description="absolute rise in fire-pixel fraction (0-1)"
    )
    collapse_z: float = Field(default=4.0, description="robust z of the change fraction")
    collapse_min_change: float = Field(default=0.05, description="absolute changed-pixel fraction")
    collapse_persist_s: float = Field(
        default=1.0,
        description=(
            "how long the change must stay elevated after onset, seconds. This is what "
            "actually separates a real collapse (turbulent debris for many seconds) from a "
            "hard cut or a feed dropout (one bright/black frame, then flat): a cut passes the "
            "single-frame z-score test just as easily as a real onset does."
        ),
    )
    collapse_persist_floor: float = Field(
        default=0.35,
        description="fraction of the peak changed-pixel fraction that must be sustained "
        "through the persistence window",
    )
    collapse_min_luma: float = Field(
        default=15.0,
        description=(
            "reject if the ROI's mean luminance dips below this (0-255) anywhere in the "
            "persistence window: a fade-to-black editorial transition produces the same "
            "sustained, downward-biased change signal as a real collapse, but a collapse's "
            "dust and debris are lit and never read as near-black."
        ),
    )
    cut_ratio: float = Field(
        default=0.5,
        description="reject if outside-ROI change exceeds this fraction of ROI change (a cut)",
    )
    min_separation_s: float = Field(default=3.0, description="suppress candidates this close")


def detect_impact_flash(
    feats: FrameFeatures,
    cfg: DetectorConfig | None = None,
    *,
    roi_label: str = "",
) -> list[EventCandidate]:
    """Fireball onsets: a sharp rise in fire-coloured area that is not a cut."""
    cfg = cfg or DetectorConfig()
    n = len(feats)
    if n < 4:
        return []
    a = feats.as_arrays()
    fire = a["roi_fire"]
    d_fire = np.diff(fire, prepend=fire[0])
    z = _robust_z(d_fire)

    out: list[EventCandidate] = []
    for i in range(1, n):
        if z[i] < cfg.flash_z or d_fire[i] < cfg.flash_min_fire_rise:
            continue
        if a["outside_change"][i] > cfg.cut_ratio * max(a["roi_change"][i], 1e-9):
            continue  # whole frame changed: this is an edit, not an explosion
        out.append(
            EventCandidate(
                kind=EventKind.IMPACT_FLASH,
                frame_idx=feats.index[i],
                t_rel=feats.t_rel[i],
                confidence=float(min(1.0, z[i] / (3.0 * cfg.flash_z))),
                sigma_rel=_onset_sigma(feats, i),
                features={
                    "fire_rise": float(d_fire[i]),
                    "fire_after": float(fire[i]),
                    "z": float(z[i]),
                    "roi_change": float(a["roi_change"][i]),
                    "outside_change": float(a["outside_change"][i]),
                },
                roi_label=roi_label,
            )
        )
    return _suppress(out, cfg.min_separation_s)


def detect_collapse_onset(
    feats: FrameFeatures,
    cfg: DetectorConfig | None = None,
    *,
    roi_label: str = "",
) -> list[EventCandidate]:
    """Collapse onsets: sustained change in the ROI moving *downwards*."""
    cfg = cfg or DetectorConfig()
    n = len(feats)
    if n < 8:
        return []
    a = feats.as_arrays()
    change = a["roi_change"]
    cy = a["roi_change_cy"]
    z = _robust_z(change)
    d_cy = np.diff(cy, prepend=cy[0])

    t_rel = a["t_rel"]
    out: list[EventCandidate] = []
    for i in range(2, n - 2):
        if z[i] < cfg.collapse_z or change[i] < cfg.collapse_min_change:
            continue
        # must be a step, not a spike: the next couple of frames stay disturbed too
        if change[i + 1 : i + 3].mean() < 0.6 * change[i]:
            continue
        if a["outside_change"][i] > cfg.cut_ratio * max(change[i], 1e-9):
            continue
        downward = float(d_cy[i : i + 3].mean())
        if downward <= 0:
            continue  # the change is drifting up the frame: not a collapse

        # Persistence: a hard cut or a feed dropout produces exactly the same
        # single-frame spike as a real onset, then the change signal falls back
        # to baseline within a frame or two. A real collapse keeps the ROI
        # churning (falling debris, expanding dust) for seconds. Require the
        # window average over `collapse_persist_s` to stay above a floor set
        # relative to the peak, not just the next couple of frames.
        j = i
        while j < n - 1 and t_rel[j] - t_rel[i] < cfg.collapse_persist_s:
            j += 1
        window = change[i:j]
        if window.size < 3 or window.mean() < cfg.collapse_persist_floor * change[i]:
            continue
        if a["roi_mean"][i:j].min() < cfg.collapse_min_luma:
            continue  # fade-to-black transition, not a lit debris/dust event

        out.append(
            EventCandidate(
                kind=EventKind.COLLAPSE_ONSET,
                frame_idx=feats.index[i],
                t_rel=feats.t_rel[i],
                confidence=float(min(1.0, z[i] / (3.0 * cfg.collapse_z))),
                sigma_rel=_onset_sigma(feats, i),
                features={
                    "change": float(change[i]),
                    "z": float(z[i]),
                    "downward_cy_rate": downward,
                    "outside_change": float(a["outside_change"][i]),
                    "persist_mean": float(window.mean()),
                },
                roi_label=roi_label,
            )
        )
    return _suppress(out, cfg.min_separation_s)


def _onset_sigma(feats: FrameFeatures, i: int) -> float:
    """Half the sampling interval: the onset is between this frame and the last."""
    if i == 0 or len(feats) < 2:
        return 0.0
    return abs(feats.t_rel[i] - feats.t_rel[i - 1]) / 2.0


def _suppress(cands: Sequence[EventCandidate], min_sep_s: float) -> list[EventCandidate]:
    """Collapse each run of candidates into the *earliest* one per cluster.

    A sustained event (debris churning for seconds) clears the threshold on
    many consecutive frames, and its confidence need not decrease
    monotonically as the event continues -- picking the highest-confidence
    frame in that run (as a naive "strongest wins" rule would) can report a
    time seconds after the true onset for no reason but which frame happened
    to score best.  Picking the *first* frame of each run is what "onset"
    means, so that is what survives here.

    Clustering is done by the gap between *consecutive* candidates, not by
    distance from the run's first (kept) member: gating on distance from the
    representative would restart the cluster -- and re-report a fresh "onset"
    -- every ``min_sep_s`` for any event that runs longer than that, which is
    routine for a multi-second collapse.  A run of closely-spaced candidates
    of any length is one cluster; a new cluster starts only where the
    candidates themselves actually thin out.
    """
    ordered = sorted(cands, key=lambda c: c.t_rel)
    kept: list[EventCandidate] = []
    prev_t: float | None = None
    for c in ordered:
        if prev_t is None or c.t_rel - prev_t >= min_sep_s:
            kept.append(c)
        prev_t = c.t_rel
    return kept


def detect_events(
    source: FrameSource,
    *,
    roi: RoiTrack | BBox | None = None,
    start_s: float = 0.0,
    duration_s: float | None = None,
    sample_fps: float | None = None,
    max_width: int | None = 640,
    config: DetectorConfig | None = None,
    kinds: Sequence[EventKind] = (EventKind.IMPACT_FLASH, EventKind.COLLAPSE_ONSET),
) -> tuple[list[EventCandidate], FrameFeatures]:
    """Run both detectors over a window of a media file.

    ``sample_fps=None`` decodes every frame, which is what you want for the
    final pass: the onset sigma is half the sampling interval, so a 4 fps scan
    can never do better than +-0.125 s.  Subsample for a first sweep, then
    re-run at full rate around the hit.
    """
    label = roi.label if isinstance(roi, RoiTrack) else ""
    feats = compute_features(
        source.iter(
            start_s=start_s,
            duration_s=duration_s,
            sample_fps=sample_fps,
            max_width=max_width,
        ),
        roi,
    )
    cands: list[EventCandidate] = []
    if EventKind.IMPACT_FLASH in kinds:
        cands += detect_impact_flash(feats, config, roi_label=label)
    if EventKind.COLLAPSE_ONSET in kinds:
        cands += detect_collapse_onset(feats, config, roi_label=label)
    cands.sort(key=lambda c: c.t_rel)
    return (cands, feats)


# --- turning a candidate into absolute time ---------------------------------


COLLAPSE_DETECTION_LAG_S = 6.5
"""How long the collapse detector lags true structural onset, seconds.

A single-point estimate from the one real collapse window this was validated
against (see "Validation" in ``wtc4d/sync/README.md``): initial buckling is
subtle, and the detector only fires once falling debris clears its change
threshold, several seconds after NIST's onset.  Applied as a correction (the
detection is *later* than the true event, so this is subtracted back out)
with a sigma wide enough that one data point does not overstate how well it
is known.
"""
COLLAPSE_DETECTION_LAG_SIGMA_S = 3.5

# Bias correction applied when matching a candidate of this kind to an
# anchor: (lag, lag_sigma) in seconds.  Zero for the flash, which is prompt.
_DETECTION_LAG_S: dict[EventKind, tuple[float, float]] = {
    EventKind.IMPACT_FLASH: (0.0, 0.0),
    EventKind.COLLAPSE_ONSET: (COLLAPSE_DETECTION_LAG_S, COLLAPSE_DETECTION_LAG_SIGMA_S),
}


def anchor_estimate(
    candidate: EventCandidate,
    anchor_id: str,
    *,
    media_offset_s: float = 0.0,
    channel_delay_s: float = 0.0,
    channel_delay_sigma_s: float = 0.0,
    shot_id: str = "",
    correct_detection_lag: bool = True,
) -> TimeEstimate:
    """Absolute time of **media offset 0** implied by matching a candidate to an anchor.

    The estimate is deliberately attached to offset 0 rather than to the
    candidate frame, because that is what the fusion needs for a shot start.
    ``media_offset_s`` is added when the analysed window did not start at the
    beginning of the file.

    ``channel_delay_s`` shifts the *anchor*, not the detection: a broadcast
    shows the fireball ``delay`` seconds after it happened, so relative to the
    known event time the frame is later by that much.  When
    ``correct_detection_lag`` is set (the default), a collapse candidate also
    gets :data:`COLLAPSE_DETECTION_LAG_S` subtracted back out, with its sigma
    folded in -- see that constant's docstring for why the correction exists
    and how uncertain it still is.
    """
    ev: Event | None = EVENTS_BY_ID.get(anchor_id)
    if ev is None:
        raise KeyError(f"unknown timeline anchor {anchor_id!r}")
    lag_s, lag_sigma_s = (
        _DETECTION_LAG_S.get(candidate.kind, (0.0, 0.0)) if correct_detection_lag else (0.0, 0.0)
    )
    t_event_seen = ev.t + channel_delay_s + lag_s
    t_zero = t_event_seen - (candidate.t_rel + media_offset_s)
    sigma = math.sqrt(
        ev.sigma**2 + candidate.sigma_rel**2 + channel_delay_sigma_s**2 + lag_sigma_s**2
    )
    return TimeEstimate(
        t=t_zero,
        sigma=sigma,
        method=TimeMethod.EVENT_ANCHOR,
        evidence=(
            f"{candidate.kind.value} detected at media offset "
            f"{candidate.t_rel + media_offset_s:.3f} s (frame {candidate.frame_idx}, "
            f"confidence {candidate.confidence:.2f}) matched to {ev.id} "
            f"@ {fmt_local(ev.t)} +-{ev.sigma:g} s [{ev.source}]"
            + (f"; +{channel_delay_s:.2f} s broadcast delay applied" if channel_delay_s else "")
            + (
                f"; -{lag_s:.1f} s detection-lag correction applied (+-{lag_sigma_s:.1f} s)"
                if lag_s
                else ""
            )
        ),
        derived_from=[shot_id] if shot_id else [],
    )


def match_to_anchors(
    candidates: Sequence[EventCandidate],
    *,
    prior: TimeEstimate | None = None,
    media_offset_s: float = 0.0,
    channel_delay_s: float = 0.0,
    channel_delay_sigma_s: float = 0.0,
    shot_id: str = "",
    max_sigma: float = 4.0,
    correct_detection_lag: bool = True,
) -> list[TimeEstimate]:
    """Match candidates to timeline anchors, using a prior time when available.

    Without a prior an impact flash is ambiguous between the two impacts (and
    a collapse between the two collapses), so *every* compatible pairing is
    returned and the fusion is left to sort it out.  With a prior -- typically
    the broadcast-metadata estimate, good to seconds -- only pairings within
    ``max_sigma`` of it survive, which resolves the ambiguity in practice.
    """
    out: list[TimeEstimate] = []
    for cand in candidates:
        for anchor_id in cand.anchor_ids():
            est = anchor_estimate(
                cand,
                anchor_id,
                media_offset_s=media_offset_s,
                channel_delay_s=channel_delay_s,
                channel_delay_sigma_s=channel_delay_sigma_s,
                shot_id=shot_id,
                correct_detection_lag=correct_detection_lag,
            )
            if prior is not None:
                spread = math.hypot(prior.sigma, est.sigma)
                if spread > 0 and abs(est.t - prior.t) / spread > max_sigma:
                    continue
            out.append(est)
    return out


__all__ = [
    "COLLAPSE_DETECTION_LAG_S",
    "COLLAPSE_DETECTION_LAG_SIGMA_S",
    "DetectorConfig",
    "EventCandidate",
    "EventKind",
    "FrameFeatures",
    "anchor_estimate",
    "compute_features",
    "detect_collapse_onset",
    "detect_events",
    "detect_impact_flash",
    "match_to_anchors",
]
