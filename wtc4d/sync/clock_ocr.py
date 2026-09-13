"""Reading an on-screen clock bug and turning it into absolute time.

Many 9/11 broadcasts carry a live clock: a station bug, a ticker, a
"BREAKING NEWS" strap.  When one is present it is the strongest cue available
short of an event anchor, because it is a *direct* statement of wall-clock
time by the broadcaster, sampled at the frame rate.

Pipeline
--------
1. :func:`find_clock_regions` -- sample frames across the shot, run text
   detection on them, keep boxes whose text parses as a time, and cluster
   those boxes by position.  A real clock bug sits still and its surroundings
   never change, so each candidate is additionally scored with a **temporal
   variance** mask: high change inside the box (digits tick) and near-zero
   change in a ring around it (static graphic).  That is what separates a
   clock bug from a time mentioned in a lower-third caption or a timestamp in
   the scene.
2. :func:`read_clock_sequence` -- OCR just that crop over many frames.
3. :func:`fit_clock` -- fit ``t = t0 + (i - i0) / rate`` robustly.

Why the fit is an interval problem, not a least-squares problem
---------------------------------------------------------------
A clock that shows ``9:02`` is not asserting ``09:02:00``; it is asserting
"the time is somewhere in [09:02:00, 09:03:00)".  A clock showing
``09:02:59`` asserts a one-second interval.  So every reading is an
*interval* constraint on ``t0``, and the estimator is a maximum-consensus
interval stab (a RANSAC whose model space is one-dimensional and whose
consensus set can be found exactly by a sweep, rather than by sampling).
Outliers -- an OCR misread, a caption that happens to look like a time -- are
the readings whose intervals miss the consensus point, and they are reported
rather than silently dropped.

The payoff: a shot with minute-only readings spanning a single minute
*transition* is pinned to about the OCR sampling interval, not to a minute,
because the constraint from the last "9:02" and the first "9:03" intersect in
a narrow band.  With seconds on screen the result is frame-accurate.

Timezones
---------
Broadcasters cycled time zones in tickers (CNN's ran ET, CT, MT, PT in turn on
the morning in question).  A zone label found next to the digits is parsed and
converted to EDT; without one, the reading is assumed to already be EDT.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from pydantic import BaseModel, Field

from wtc4d.schema.time import TimeEstimate, TimeMethod
from wtc4d.sync.types import BBox, LinearClock
from wtc4d.sync.video import DecodedFrame, FrameSource
from wtc4d.timeline import fmt_local, hms

# --- OCR engine abstraction --------------------------------------------------


@dataclass(frozen=True)
class OcrBox:
    """One text detection: the string, its confidence, its pixel box."""

    text: str
    confidence: float
    bbox: BBox


class OcrEngine(Protocol):
    """Anything that turns an image into text boxes.

    Kept as a protocol so tests can inject a deterministic fake and so the
    heavy ONNX engine is never imported at module import time.
    """

    def __call__(self, image: np.ndarray) -> list[OcrBox]:  # pragma: no cover - protocol
        ...


class RapidOcrEngine:
    """``rapidocr-onnxruntime``: CPU-only, pip-installable, models in the wheel.

    Chosen over tesseract (needs a system binary) and over easyocr/paddle
    (torch/paddle downloads).  It ships PP-OCR v4 detection+recognition ONNX
    models inside the wheel, so it works with no network access.
    """

    def __init__(self, **kwargs: object) -> None:
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                'clock OCR needs rapidocr-onnxruntime: pip install -e ".[sync]"'
            ) from exc
        self._ocr = RapidOCR(**kwargs)

    def __call__(self, image: np.ndarray) -> list[OcrBox]:
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        result, _elapsed = self._ocr(np.ascontiguousarray(image))
        boxes: list[OcrBox] = []
        for entry in result or []:
            poly, text, conf = entry[0], str(entry[1]), float(entry[2])
            xs = [float(p[0]) for p in poly]
            ys = [float(p[1]) for p in poly]
            x0, y0 = int(math.floor(min(xs))), int(math.floor(min(ys)))
            x1, y1 = int(math.ceil(max(xs))), int(math.ceil(max(ys)))
            boxes.append(
                OcrBox(
                    text=text,
                    confidence=conf,
                    bbox=BBox(x=x0, y=y0, w=max(1, x1 - x0), h=max(1, y1 - y0)),
                )
            )
        return boxes


_DEFAULT_ENGINE: OcrEngine | None = None


def default_engine() -> OcrEngine:
    """Process-wide :class:`RapidOcrEngine` (model load is not cheap)."""
    global _DEFAULT_ENGINE
    if _DEFAULT_ENGINE is None:
        _DEFAULT_ENGINE = RapidOcrEngine()
    return _DEFAULT_ENGINE


# --- parsing -----------------------------------------------------------------


# A colon is the only separator a clock bug actually uses; ';' and "'" are the
# usual OCR misreads of one.  A '.' is accepted too, but only in a string that
# also carries an am/pm marker or a zone label -- without that guard, stock
# tickers ("DOW 10.00", "S&P 0.30") parse as times, and CNN's crawl on the
# morning in question has both a clock and a futures ticker on the same strap.
def _time_re(sep: str) -> re.Pattern[str]:
    """Digits only; the am/pm marker and zone label are parsed separately."""
    return re.compile(
        r"(?<![\w:.])(?P<h>\d{1,2})\s*" + sep + r"\s*(?P<m>\d{2})"
        r"(?:\s*" + sep + r"\s*(?P<s>\d{2}))?"
    )


TIME_RE = _time_re(r"[:;']")
TIME_RE_LOOSE = _time_re(r"[.]")

_AMPM = r"(?P<ampm>[AaPp])\.?\s*[Mm]?\.?"
_ZONE = r"(?P<zone>[ECMPAH])[DS]?T(?![A-Za-z])"
_UTC = r"(?P<utc>UTC|GMT|Z)(?![A-Za-z])"

# Ordered because the suffix is genuinely ambiguous once OCR glues tokens
# together: "7:02aMT" is 7:02 am Mountain, not 7:02 a.m. followed by stray
# letters, and only trying the longest interpretation first gets that right.
SUFFIX_RES = [
    re.compile(r"^\s*" + _AMPM + r"\s*" + _ZONE),
    re.compile(r"^\s*" + _AMPM + r"\s*" + _UTC),
    re.compile(r"^\s*" + _ZONE),
    re.compile(r"^\s*" + _UTC),
    re.compile(r"^\s*" + _AMPM + r"(?![A-Za-z])"),
]
AMPM_RE = re.compile(r"\d\s*[AaPp]\.?\s*[Mm]?(?![A-Za-z])")
TZ_RE = re.compile(r"(?<![A-Za-z0-9])(?:" + _ZONE + r"|" + _UTC + r")")


def parse_clock_suffix(tail: str) -> tuple[str, str | None]:
    """Parse what follows the digits -> ``(ampm, zone_or_None)``."""
    for rx in SUFFIX_RES:
        m = rx.match(tail)
        if m:
            groups = m.groupdict()
            ampm = (groups.get("ampm") or "").lower()
            if groups.get("utc"):
                return (ampm, "UTC")
            if groups.get("zone"):
                return (ampm, groups["zone"].upper())
            return (ampm, None)
    return ("", None)


# Hours east of EDT that must be ADDED to a reading in that zone to get EDT.
_ZONE_TO_EDT_HOURS = {"E": 0, "C": 1, "M": 2, "P": 3, "A": 4, "H": 5}

# Glyph confusions of small, interlaced, heavily-compressed broadcast digits.
_DIGIT_FIXUPS = str.maketrans(
    {
        "O": "0",
        "o": "0",
        "Q": "0",
        "D": "0",
        "U": "0",
        "I": "1",
        "l": "1",
        "|": "1",
        "i": "1",
        "!": "1",
        "]": "1",
        "[": "1",
        "Z": "2",
        "z": "2",
        "S": "5",
        "s": "5",
        "G": "6",
        "b": "6",
        "T": "7",
        "?": "7",
        "B": "8",
        "g": "9",
        "q": "9",
    }
)


class ClockReading(BaseModel):
    """One parsed clock display, with the time *interval* it asserts."""

    frame_idx: int
    t_rel: float = Field(description="seconds from the start of the media file")
    text: str
    hour: int
    minute: int
    second: int | None = None
    zone: str = "E"
    confidence: float = 1.0
    t_display: float = Field(description="project seconds the display asserts (start of bucket)")
    resolution_s: float = Field(description="60 for HH:MM, 1 for HH:MM:SS")

    def interval(self) -> tuple[float, float]:
        """``[lo, hi)`` project seconds consistent with this display."""
        return (self.t_display, self.t_display + self.resolution_s)


def parse_clock_text(
    text: str,
    *,
    hint_t: float | None = None,
    default_zone: str = "E",
) -> tuple[int, int, int | None, str] | None:
    """Parse ``"8:02a CT"`` -> ``(8, 2, None, "C")``.

    ``hint_t`` (project seconds) resolves a 12-hour display with no am/pm
    marker to whichever of the two candidates is nearer.  Returns ``None`` if
    nothing in the string looks like a clock.
    """
    for candidate in (text, text.translate(_DIGIT_FIXUPS)):
        m = TIME_RE.search(candidate)
        if m is None and (AMPM_RE.search(candidate) or TZ_RE.search(candidate)):
            m = TIME_RE_LOOSE.search(candidate)
        if not m:
            continue
        h = int(m.group("h"))
        minute = int(m.group("m"))
        second = int(m.group("s")) if m.group("s") else None
        if minute > 59 or (second is not None and second > 59) or h > 23:
            continue

        # The suffix is always read from the *untranslated* text: the digit
        # fixups are single-character, so offsets carry over, but applying
        # them to a zone label would turn "PT" into "P7" and lose the zone.
        ampm, zone = parse_clock_suffix(text[m.end() :])
        if zone is None:
            # the label sometimes sits before the digits ("ET 9:02")
            zm = TZ_RE.search(text[: m.start()])
            if zm:
                zone = "UTC" if zm.group("utc") else (zm.group("zone") or "").upper()
        zone = zone or default_zone

        if ampm == "a":
            h = 0 if h == 12 else h
        elif ampm == "p":
            h = 12 if h == 12 else h + 12
        elif h <= 12 and hint_t is not None:
            # no marker: pick the 12-hour branch nearer the hint
            alt = h + 12 if h < 12 else h - 12
            if abs(hms(alt, minute, second or 0) - hint_t) < abs(
                hms(h, minute, second or 0) - hint_t
            ):
                h = alt
        if h > 23:
            continue
        return (h, minute, second, zone)
    return None


def reading_to_project_seconds(hour: int, minute: int, second: int | None, zone: str) -> float:
    """Displayed clock -> project seconds (start of the displayed bucket).

    Project time is EDT, so a display in another US zone is shifted forward by
    the zone difference; ``UTC``/``GMT`` is shifted back by 4 hours.
    """
    t = hms(hour, minute, second or 0)
    if zone == "UTC":
        return t - 4 * 3600.0
    return t + _ZONE_TO_EDT_HOURS.get(zone, 0) * 3600.0


def _join_boxes(boxes: Sequence[OcrBox]) -> str:
    """Concatenate detections left-to-right; the zone label is its own box."""
    return " ".join(b.text for b in sorted(boxes, key=lambda b: (b.bbox.y // 12, b.bbox.x)))


def read_clock_image(
    image: np.ndarray,
    engine: OcrEngine,
    *,
    frame_idx: int = 0,
    t_rel: float = 0.0,
    hint_t: float | None = None,
    upscale: int = 3,
    min_confidence: float = 0.3,
) -> ClockReading | None:
    """OCR one (already cropped) clock image into a :class:`ClockReading`.

    ``upscale`` matters: broadcast clock digits are 10-20 px tall on a 480-line
    source, below what the recogniser was trained on.  A plain 3x resize
    roughly doubles the hit rate and costs nothing at these crop sizes.
    """
    img = image
    if upscale > 1:
        import cv2

        img = cv2.resize(
            image,
            (image.shape[1] * upscale, image.shape[0] * upscale),
            interpolation=cv2.INTER_CUBIC,
        )
    boxes = [b for b in engine(img) if b.confidence >= min_confidence]
    if not boxes:
        return None
    text = _join_boxes(boxes)
    parsed = parse_clock_text(text, hint_t=hint_t)
    if parsed is None:
        return None
    h, m, s, zone = parsed
    conf = float(np.mean([b.confidence for b in boxes]))
    return ClockReading(
        frame_idx=frame_idx,
        t_rel=t_rel,
        text=text,
        hour=h,
        minute=m,
        second=s,
        zone=zone,
        confidence=conf,
        t_display=reading_to_project_seconds(h, m, s, zone),
        resolution_s=1.0 if s is not None else 60.0,
    )


# --- region detection --------------------------------------------------------


class ClockRegion(BaseModel):
    """A screen region that repeatedly reads as a clock."""

    bbox: BBox
    n_hits: int
    n_samples: int
    mean_confidence: float
    inner_variance: float = Field(description="mean temporal std of pixels inside the box")
    ring_variance: float = Field(description="mean temporal std of a ring around the box")
    score: float
    sample_texts: list[str] = Field(default_factory=list)


def temporal_std(images: Sequence[np.ndarray]) -> np.ndarray:
    """Per-pixel standard deviation over time of grayscale frames."""
    stack = np.stack([im.astype(np.float32) for im in images], axis=0)
    return stack.std(axis=0)


def _box_ring_stats(std_map: np.ndarray, box: BBox, pad: int = 8) -> tuple[float, float]:
    h, w = std_map.shape[:2]
    b = box.clip(w, h)
    inner = std_map[b.as_slices()]
    outer = BBox(
        x=max(0, b.x - pad),
        y=max(0, b.y - pad),
        w=min(w, b.x1 + pad) - max(0, b.x - pad),
        h=min(h, b.y1 + pad) - max(0, b.y - pad),
    ).clip(w, h)
    ring_sum = float(std_map[outer.as_slices()].sum() - inner.sum())
    ring_px = outer.w * outer.h - b.w * b.h
    ring = ring_sum / ring_px if ring_px > 0 else 0.0
    return (float(inner.mean()), ring)


def _iou(a: BBox, b: BBox) -> float:
    ix0, iy0 = max(a.x, b.x), max(a.y, b.y)
    ix1, iy1 = min(a.x1, b.x1), min(a.y1, b.y1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    return inter / float(a.w * a.h + b.w * b.h - inter)


def _merge(a: BBox, b: BBox) -> BBox:
    x0, y0 = min(a.x, b.x), min(a.y, b.y)
    x1, y1 = max(a.x1, b.x1), max(a.y1, b.y1)
    return BBox(x=x0, y=y0, w=x1 - x0, h=y1 - y0)


def find_clock_regions(
    frames: Iterable[DecodedFrame],
    engine: OcrEngine,
    *,
    hint_t: float | None = None,
    min_hits: int = 2,
    pad: int = 6,
) -> list[ClockRegion]:
    """Locate clock bugs by running OCR over sampled frames and clustering hits.

    ``frames`` should be a modest sample (10-20) spread over the shot: OCR of a
    full 480-line frame costs a good fraction of a second.  Returns regions
    sorted best-first.
    """
    grays: list[np.ndarray] = []
    hits: list[tuple[BBox, float, str]] = []
    n_samples = 0
    for fr in frames:
        n_samples += 1
        img = fr.image
        if img.ndim == 3:
            import cv2

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        grays.append(gray)
        for box in engine(img):
            if parse_clock_text(box.text, hint_t=hint_t) is not None:
                hits.append((box.bbox, box.confidence, box.text))
    if not hits or not grays:
        return []

    shape = grays[0].shape
    grays = [g for g in grays if g.shape == shape]
    std_map = temporal_std(grays) if len(grays) > 1 else np.zeros(shape, np.float32)
    global_std = float(std_map.mean()) or 1e-6

    clusters: list[dict] = []
    for bbox, conf, text in hits:
        for cl in clusters:
            if _iou(cl["bbox"], bbox) > 0.3:
                cl["bbox"] = _merge(cl["bbox"], bbox)
                cl["confs"].append(conf)
                cl["texts"].append(text)
                break
        else:
            clusters.append({"bbox": bbox, "confs": [conf], "texts": [text]})

    regions: list[ClockRegion] = []
    for cl in clusters:
        if len(cl["confs"]) < min_hits:
            continue
        box = BBox(
            x=max(0, cl["bbox"].x - pad),
            y=max(0, cl["bbox"].y - pad),
            w=cl["bbox"].w + 2 * pad,
            h=cl["bbox"].h + 2 * pad,
        ).clip(shape[1], shape[0])
        inner, ring = _box_ring_stats(std_map, box)
        hit_rate = len(cl["confs"]) / max(1, n_samples)
        mean_conf = float(np.mean(cl["confs"]))
        # A clock bug ticks inside a frozen graphic: reward inner change,
        # reward a quiet ring, and reward being read often and confidently.
        ticking = inner / (inner + global_std)
        quiet_ring = global_std / (ring + global_std)
        score = hit_rate * mean_conf * (0.5 + 0.25 * ticking + 0.25 * quiet_ring)
        regions.append(
            ClockRegion(
                bbox=box,
                n_hits=len(cl["confs"]),
                n_samples=n_samples,
                mean_confidence=mean_conf,
                inner_variance=inner,
                ring_variance=ring,
                score=score,
                sample_texts=cl["texts"][:6],
            )
        )
    regions.sort(key=lambda r: r.score, reverse=True)
    return regions


def read_clock_sequence(
    source: FrameSource,
    region: ClockRegion | BBox,
    engine: OcrEngine,
    *,
    start_s: float = 0.0,
    duration_s: float | None = None,
    sample_fps: float = 2.0,
    hint_t: float | None = None,
    max_width: int | None = None,
) -> list[ClockReading]:
    """OCR the clock crop over a window of the media."""
    box = region.bbox if isinstance(region, ClockRegion) else region
    readings: list[ClockReading] = []
    for fr in source.iter(
        start_s=start_s, duration_s=duration_s, sample_fps=sample_fps, max_width=max_width
    ):
        img = fr.image
        h, w = img.shape[:2]
        crop = img[box.clip(w, h).as_slices()]
        if crop.size == 0:
            continue
        r = read_clock_image(crop, engine, frame_idx=fr.index, t_rel=fr.t_rel, hint_t=hint_t)
        if r is not None:
            readings.append(r)
    return readings


# --- robust fit --------------------------------------------------------------


def max_overlap_interval(
    intervals: Sequence[tuple[float, float]],
) -> tuple[float, float, list[int]]:
    """Find the point covered by the most half-open intervals.

    Returns ``(lo, hi, member_indices)`` -- the maximal-overlap *band*, not
    just a point, so the caller can take its width as the residual
    uncertainty.  Sweep over endpoints, O(n log n).
    """
    if not intervals:
        return (0.0, 0.0, [])
    events: list[tuple[float, int, int]] = []
    for i, (lo, hi) in enumerate(intervals):
        events.append((lo, 1, i))
        events.append((hi, -1, i))
    # closes before opens at the same coordinate (half-open intervals)
    events.sort(key=lambda e: (e[0], e[1]))
    best_count = 0
    best_lo = best_hi = intervals[0][0]
    active: set[int] = set()
    best_members: list[int] = []
    for k, (x, kind, idx) in enumerate(events):
        if kind == 1:
            active.add(idx)
            if len(active) > best_count:
                nxt = next((e[0] for e in events[k + 1 :] if e[0] > x), x)
                best_count = len(active)
                best_lo, best_hi = x, nxt
                best_members = sorted(active)
        else:
            active.discard(idx)
    return (best_lo, best_hi, best_members)


class ClockFit(BaseModel):
    """Result of fitting a frame index to the clock readings."""

    clock: LinearClock
    n_readings: int
    n_inliers: int
    inlier_frames: list[int] = Field(default_factory=list)
    outlier_frames: list[int] = Field(default_factory=list)
    band_s: float = Field(description="width of the maximum-consensus band, seconds")
    estimate: TimeEstimate


def fit_clock(
    readings: Sequence[ClockReading],
    *,
    fps: float,
    i0: int = 0,
    rate_search: Sequence[float] | None = None,
    min_inliers: int = 2,
) -> ClockFit | None:
    """Fit ``t = t0 + (i - i0)/rate`` to clock readings by maximum consensus.

    Each reading constrains ``t0`` to an interval; the estimate is the centre
    of the band that the most intervals agree on.  ``rate_search`` (a sequence
    of candidate frame rates) enables a coarse search for a mis-stated or
    transferred frame rate -- it only helps when the readings span enough time
    to separate the candidates, so it is off by default.

    The reported sigma is the consensus band half-width as a uniform
    distribution (``width / sqrt(12)``) combined with a floor of one sampling
    interval, because the display could have ticked between two OCR'd frames.
    """
    readings = [r for r in readings if r.resolution_s > 0]
    if len(readings) < min_inliers:
        return None

    rates = list(rate_search) if rate_search else [fps]
    best: tuple[int, float, float, float, list[int]] | None = None
    for rate in rates:
        intervals = [
            (lo - (r.frame_idx - i0) / rate, hi - (r.frame_idx - i0) / rate)
            for r, (lo, hi) in ((r, r.interval()) for r in readings)
        ]
        lo, hi, members = max_overlap_interval(intervals)
        if best is None or len(members) > best[0]:
            best = (len(members), rate, lo, hi, members)
    assert best is not None
    n_inliers, rate, lo, hi, members = best
    if n_inliers < min_inliers:
        return None

    band = max(hi - lo, 0.0)
    t0 = 0.5 * (lo + hi)
    inlier_set = set(members)
    inlier_frames = [readings[i].frame_idx for i in members]
    outlier_frames = [r.frame_idx for i, r in enumerate(readings) if i not in inlier_set]

    # sampling floor: the clock may have ticked between consecutive OCR'd frames
    frames_sorted = sorted(r.frame_idx for r in readings)
    gaps = [b - a for a, b in zip(frames_sorted, frames_sorted[1:], strict=False)]
    sample_gap_s = (min(gaps) / rate) if gaps else 1.0 / rate
    sigma = math.sqrt((band / math.sqrt(12.0)) ** 2 + (sample_gap_s / math.sqrt(12.0)) ** 2)
    sigma = max(sigma, 1.0 / rate)

    finest = min(r.resolution_s for r in readings)
    clock = LinearClock(t0=t0, i0=i0, rate=rate, sigma_t0=sigma, sigma_rate=0.0)
    est = TimeEstimate(
        t=t0,
        sigma=sigma,
        method=TimeMethod.ONSCREEN_CLOCK,
        evidence=(
            f"on-screen clock: {n_inliers}/{len(readings)} readings agree "
            f"({finest:g} s display resolution); consensus band {band:.3f} s at "
            f"{rate:.4f} fps gives frame {i0} = {fmt_local(t0)}; "
            f"{len(outlier_frames)} reading(s) rejected"
        ),
    )
    return ClockFit(
        clock=clock,
        n_readings=len(readings),
        n_inliers=n_inliers,
        inlier_frames=inlier_frames,
        outlier_frames=outlier_frames,
        band_s=band,
        estimate=est,
    )


def estimate_shot_time(
    source: FrameSource,
    *,
    engine: OcrEngine | None = None,
    start_s: float = 0.0,
    duration_s: float | None = None,
    detect_samples: int = 12,
    sample_fps: float = 2.0,
    hint_t: float | None = None,
    region: BBox | None = None,
    max_width: int | None = 960,
    i0: int = 0,
) -> tuple[ClockFit | None, list[ClockRegion], list[ClockReading]]:
    """End-to-end: find a clock, read it, fit it.

    Returns ``(fit, regions, readings)`` so a caller (or the CLI) can report
    what was found even when the fit fails.
    """
    engine = engine or default_engine()
    regions: list[ClockRegion] = []
    if region is None:
        span = duration_s if duration_s is not None else 60.0
        det_fps = max(detect_samples / max(span, 1e-3), 0.05)
        sample = list(
            source.iter(
                start_s=start_s,
                duration_s=duration_s,
                sample_fps=det_fps,
                max_width=max_width,
            )
        )[:detect_samples]
        regions = find_clock_regions(sample, engine, hint_t=hint_t)
        if not regions:
            return (None, [], [])
        region = regions[0].bbox

    readings = read_clock_sequence(
        source,
        region,
        engine,
        start_s=start_s,
        duration_s=duration_s,
        sample_fps=sample_fps,
        hint_t=hint_t,
        max_width=max_width,
    )
    fit = fit_clock(readings, fps=source.fps, i0=i0)
    return (fit, regions, readings)


__all__ = [
    "ClockFit",
    "ClockReading",
    "ClockRegion",
    "OcrBox",
    "OcrEngine",
    "RapidOcrEngine",
    "default_engine",
    "estimate_shot_time",
    "find_clock_regions",
    "fit_clock",
    "max_overlap_interval",
    "parse_clock_suffix",
    "parse_clock_text",
    "read_clock_image",
    "read_clock_sequence",
    "reading_to_project_seconds",
    "temporal_std",
]
