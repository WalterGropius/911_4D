"""Aligning an untimed clip to a timed one by audio.

Video is the primary carrier of the cues in :mod:`wtc4d.sync.clock_ocr` and
:mod:`wtc4d.sync.events`, but most amateur footage of the day has no visible
clock and no view of a tower.  What it usually does have is a soundtrack that
overlaps something already timed: the same TV broadcast playing on a nearby
set, the same distant rumble of a collapse, or (rarer, but decisive) the same
live anchor sentence picked up by two different recordings.  Cross-correlating
audio turns any of those into a **relative** timing constraint between two
shots, which :mod:`wtc4d.sync.fuse` chains onto the absolute cues.

Method
------
1. Decode both clips to mono at a low sample rate (8-16 kHz -- see
   :func:`wtc4d.sync.video.load_audio`; the cues here live under 4 kHz).
2. Correlate with **GCC-PHAT** (generalised cross-correlation, phase
   transform): whiten each signal's spectrum before correlating.  Plain
   cross-correlation is dominated by whichever frequency band happens to be
   loudest, which on re-encoded broadcast audio is often a re-compression
   artifact; the phase transform normalises every frequency to unit
   magnitude, so it aligns on *timing structure* (onsets, formant
   transitions) instead.  This is the standard technique for time-delay
   estimation from acoustic signals (Knapp & Carter 1976) and needs no
   training data.
3. Refine the integer-sample peak to sub-sample precision with a parabolic
   fit through its three neighbouring correlation values.
4. Search over a small grid of **playback-rate** factors (one clip run
   slightly fast or slow -- NTSC pulldown, a variable-speed transfer, a
   phone's clock drift) and keep the rate that gives the sharpest peak.
5. Turn the peak's sharpness and prominence over the rest of the correlation
   function into a confidence and a sigma, rather than reporting a bare
   offset: a flat, ambiguous correlation (two clips of unrelated crowd noise)
   must not look as certain as a sharp one (the same sentence in both).

Two derived cues reuse the same machinery:

* :func:`detect_rumble_onset` -- the collapses produced a very low frequency
  (sub-40 Hz) rumble audible for tens of seconds even far from the towers.
  Detected the same way :mod:`wtc4d.sync.events` detects a visual collapse
  onset (robust z-score + persistence), just on a frequency-band energy
  envelope instead of a pixel-change mask.
* :func:`match_landmark` -- align a clip against a short **reference**
  snippet of known absolute time (a famous piece of live audio heard on
  several recordings) using the same GCC-PHAT alignment as clip-to-clip
  sync, just with one side fixed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, Field

from wtc4d.schema.time import TimeEstimate, TimeMethod
from wtc4d.sync.types import PairwiseOffset
from wtc4d.sync.video import load_audio
from wtc4d.timeline import EVENTS_BY_ID, Event, fmt_local

DEFAULT_SAMPLE_RATE = 8000
"""8 kHz mono is enough for speech-envelope and rumble alignment and keeps
the correlation FFTs small; raise it only for a landmark that needs more
spectral detail (e.g. music)."""


# --- low-level building blocks ------------------------------------------------


def resample_linear(x: np.ndarray, rate_factor: float) -> np.ndarray:
    """Stretch/compress ``x`` by ``rate_factor`` (>1 = played back faster).

    Linear interpolation is enough here: the rate factors searched are within
    a percent or two of 1.0 (see :data:`DEFAULT_RATE_CANDIDATES`), where a
    cheap resampler and a high-quality one agree to well under one sample.
    """
    if rate_factor == 1.0 or len(x) < 2:
        return x
    n_out = max(1, int(round(len(x) / rate_factor)))
    src_idx = np.arange(n_out, dtype=np.float64) * rate_factor
    src_idx = np.clip(src_idx, 0, len(x) - 1)
    i0 = np.floor(src_idx).astype(np.int64)
    i1 = np.minimum(i0 + 1, len(x) - 1)
    frac = src_idx - i0
    return (x[i0] * (1.0 - frac) + x[i1] * frac).astype(np.float32)


def _next_pow2(n: int) -> int:
    return 1 << (int(n) - 1).bit_length()


@dataclass(frozen=True)
class Correlogram:
    """A GCC-PHAT correlation curve, with the array index of zero lag.

    ``cc[zero_index + k]`` is the correlation at ``sig`` delayed by ``k``
    samples relative to ``ref``.  Kept explicit -- rather than assumed to be
    ``len(cc) // 2`` -- because ``sig`` and ``ref`` are almost never the same
    length here: that is exactly what happens every time :func:`align_signals`
    resamples ``sig`` by a candidate rate before correlating it.
    """

    cc: np.ndarray
    zero_index: int


def gcc_phat(
    sig: np.ndarray,
    ref: np.ndarray,
    *,
    max_shift: int | None = None,
    epsilon: float = 1e-10,
) -> Correlogram:
    """Generalised cross-correlation with phase transform.

    Use :func:`best_shift` to pull the peak out of the result.
    """
    n = len(sig) + len(ref)
    nfft = _next_pow2(n)
    SIG = np.fft.rfft(sig, n=nfft)
    REF = np.fft.rfft(ref, n=nfft)
    cross = SIG * np.conj(REF)
    mag = np.abs(cross)
    mag[mag < epsilon] = epsilon
    cross /= mag
    cc = np.fft.irfft(cross, n=nfft)
    # Linear (non-circular) cross-correlation for lags -(len(ref)-1) .. len(sig)-1:
    # negative lags wrap to the end of the circular IFFT, positive lags sit at
    # the start.  Zero lag -- sig[i] lined up with ref[i] -- is therefore at
    # index len(ref) - 1 of the concatenated array, not at its midpoint.
    cc = np.concatenate((cc[-(len(ref) - 1) :], cc[: len(sig)]))
    zero_index = len(ref) - 1
    if max_shift is not None:
        lo = max(0, zero_index - max_shift)
        hi = min(len(cc), zero_index + max_shift + 1)
        cc = cc[lo:hi]
        zero_index = zero_index - lo
    return Correlogram(cc=cc, zero_index=zero_index)


def parabolic_peak(y: np.ndarray, i: int) -> tuple[float, float]:
    """Sub-sample peak location and value via a 3-point parabolic fit.

    Returns ``(offset_from_i, interpolated_value)``; ``offset_from_i`` is in
    ``[-0.5, 0.5]`` except at an array edge, where no correction is made.
    """
    if i <= 0 or i >= len(y) - 1:
        return (0.0, float(y[i]))
    y0, y1, y2 = float(y[i - 1]), float(y[i]), float(y[i + 1])
    denom = y0 - 2.0 * y1 + y2
    if abs(denom) < 1e-12:
        return (0.0, y1)
    delta = 0.5 * (y0 - y2) / denom
    delta = max(-0.5, min(0.5, delta))
    peak = y1 - 0.25 * (y0 - y2) * delta
    return (delta, peak)


@dataclass(frozen=True)
class CorrelationResult:
    """A GCC-PHAT peak, plus what it takes to judge how trustworthy it is."""

    shift_samples: float  # sub-sample shift of `sig` relative to `ref`
    peak: float
    noise_floor: float  # robust spread of the correlation away from the peak
    prominence: float  # (peak - noise_floor) / noise_floor
    second_peak_ratio: float  # height of the best *other* local max / peak


def best_shift(cg: Correlogram, *, exclude_radius: int = 3) -> CorrelationResult:
    """Pull the best offset and a trustworthiness readout out of a GCC-PHAT curve."""
    cc = cg.cc
    i = int(np.argmax(cc))
    delta, peak = parabolic_peak(cc, i)
    shift = (i - cg.zero_index) + delta

    med = float(np.median(cc))
    mad = float(np.median(np.abs(cc - med))) * 1.4826
    noise_floor = max(mad, 1e-12)
    prominence = (peak - med) / noise_floor

    masked = cc.copy()
    lo, hi = max(0, i - exclude_radius), min(len(cc), i + exclude_radius + 1)
    masked[lo:hi] = -np.inf
    second = float(np.max(masked)) if np.any(np.isfinite(masked)) else med
    second_peak_ratio = second / peak if peak > 0 else 1.0

    return CorrelationResult(
        shift_samples=shift,
        peak=float(peak),
        noise_floor=noise_floor,
        prominence=float(prominence),
        second_peak_ratio=float(max(0.0, second_peak_ratio)),
    )


def confidence_from_correlation(res: CorrelationResult) -> float:
    """Map a :class:`CorrelationResult` to a 0-1 confidence.

    Rewards a peak that stands well above the noise floor (``prominence``)
    and punishes a strong rival peak elsewhere (``second_peak_ratio``, which
    is exactly what a periodic signal -- a ticking clock, a repeated jingle --
    produces, and which a raw peak height cannot see).
    """
    prom_term = 1.0 - math.exp(-res.prominence / 8.0)
    rival_term = max(0.0, 1.0 - res.second_peak_ratio)
    return float(max(0.0, min(1.0, prom_term * rival_term)))


def sigma_from_correlation(res: CorrelationResult, sample_period_s: float) -> float:
    """1-sigma of the shift estimate, in seconds.

    A sharp, prominent peak with no rival is worth a fraction of a sample; a
    weak or ambiguous one degrades gracefully rather than reporting the
    sample period as if it were exact.  Floored at half a sample, since that
    is the resolution :func:`parabolic_peak` can actually deliver.
    """
    if res.prominence <= 0:
        return float("inf")
    quality = max(res.prominence * (1.0 - res.second_peak_ratio), 1e-3)
    sigma_samples = max(0.5, 5.0 / quality)
    return sigma_samples * sample_period_s


# --- rate + offset search -----------------------------------------------------

DEFAULT_RATE_CANDIDATES: tuple[float, ...] = tuple(
    sorted({1.0, 30000 / 30030, 30030 / 30000, 25 / 24, 24 / 25, 1001 / 1000, 1000 / 1001})
)
"""Common analogue-video rate ratios: NTSC drop-frame (30000/1001 vs 30 fps),
a 25 -> 24 fps telecine mismatch, and a bare 0.1% clock-drift allowance.  A
caller with a specific suspicion (a known VHS deck) can pass a tighter or
wider set."""


def align_signals(
    sig: np.ndarray,
    ref: np.ndarray,
    sample_rate: int,
    *,
    rate_candidates: tuple[float, ...] = DEFAULT_RATE_CANDIDATES,
    max_shift_s: float | None = None,
) -> tuple[float, float, CorrelationResult]:
    """Best (rate_factor, shift_seconds, correlation) aligning ``sig`` onto ``ref``.

    ``shift_seconds`` is measured *before* the rate correction is applied,
    i.e. it is the offset of ``sig``'s sample 0 relative to ``ref``'s sample 0
    once ``sig`` has been resampled by ``rate_factor``.  ``max_shift_s``
    restricts the search window, which both speeds up the FFT and avoids
    locking onto a spurious far-away peak when the clips are long.
    """
    max_shift = int(round(max_shift_s * sample_rate)) if max_shift_s else None
    best: tuple[float, float, CorrelationResult] | None = None
    for rate in rate_candidates:
        resampled = resample_linear(sig, rate) if rate != 1.0 else sig
        cc = gcc_phat(resampled, ref, max_shift=max_shift)
        res = best_shift(cc)
        score = res.prominence * (1.0 - res.second_peak_ratio)
        if best is None or score > best[2].prominence * (1.0 - best[2].second_peak_ratio):
            shift_s = res.shift_samples / sample_rate
            best = (rate, shift_s, res)
    assert best is not None
    return best


# --- public API ----------------------------------------------------------


def align_clips(
    path_a: str,
    path_b: str,
    *,
    a_id: str = "a",
    b_id: str = "b",
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    start_s: float = 0.0,
    duration_s: float | None = None,
    rate_candidates: tuple[float, ...] = DEFAULT_RATE_CANDIDATES,
    max_shift_s: float | None = None,
    min_confidence: float = 0.15,
) -> PairwiseOffset | None:
    """Align two media files by audio and return a :class:`PairwiseOffset`.

    ``dt`` is defined so that ``t(b) - t(a) = dt`` for the analysed windows'
    start samples once ``rate_ratio`` is accounted for; see
    :class:`~wtc4d.sync.types.PairwiseOffset`.  Returns ``None`` when the
    correlation is too ambiguous to trust (below ``min_confidence``) rather
    than reporting a number nobody should use.
    """
    a = load_audio(path_a, sample_rate=sample_rate, start_s=start_s, duration_s=duration_s)
    b = load_audio(path_b, sample_rate=sample_rate, start_s=start_s, duration_s=duration_s)
    if len(a) < sample_rate // 2 or len(b) < sample_rate // 2:
        return None  # under half a second: not enough signal to correlate

    rate, shift_s, res = align_signals(
        a, b, sample_rate, rate_candidates=rate_candidates, max_shift_s=max_shift_s
    )
    confidence = confidence_from_correlation(res)
    if confidence < min_confidence:
        return None
    sigma = sigma_from_correlation(res, 1.0 / sample_rate)

    # align_signals(sig=a, ref=b) returns shift_s such that ref's window start
    # is `shift_s` seconds after sig's: T_b0 = T_a0 + shift_s, i.e.
    # shift_s == t(b) - t(a), exactly the PairwiseOffset.dt contract -- no
    # sign flip needed. (Verified empirically in tests/test_sync_audio.py by
    # trimming one real file at two known offsets and checking the recovered
    # dt against the true difference, not just re-derived on paper.)
    return PairwiseOffset(
        a=a_id,
        b=b_id,
        dt=shift_s,
        sigma=sigma,
        method=TimeMethod.AUDIO_XCORR,
        confidence=confidence,
        rate_ratio=rate,
        evidence=(
            f"GCC-PHAT audio alignment at {sample_rate} Hz: shift {shift_s:+.4f} s, "
            f"rate factor {rate:.6f}, prominence {res.prominence:.1f}, "
            f"second-peak ratio {res.second_peak_ratio:.2f}, confidence {confidence:.2f}"
        ),
    )


# --- rumble onset (collapse signature in the audio) --------------------------


class AudioDetectorConfig(BaseModel):
    """Thresholds for :func:`detect_rumble_onset`."""

    band_hz: tuple[float, float] = Field(default=(5.0, 40.0))
    z_threshold: float = Field(default=5.0)
    persist_s: float = Field(default=3.0, description="rumble must sustain this long")
    persist_floor: float = Field(default=0.3, description="fraction of peak sustained")
    frame_s: float = Field(default=0.25)


@dataclass(frozen=True)
class RumbleCandidate:
    """A detected low-frequency onset, e.g. a collapse rumble."""

    t_rel: float
    sigma_rel: float
    confidence: float
    band_energy_db: float


def _bandpass_energy_envelope(
    x: np.ndarray, sample_rate: int, band_hz: tuple[float, float], frame_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """Short-time energy in one frequency band, via STFT (numpy only)."""
    frame = max(8, int(round(frame_s * sample_rate)))
    hop = frame // 2
    n = len(x)
    if n < frame:
        return (np.array([0.0]), np.array([0.0]))
    window = np.hanning(frame)
    freqs = np.fft.rfftfreq(frame, d=1.0 / sample_rate)
    band = (freqs >= band_hz[0]) & (freqs <= band_hz[1])

    n_frames = 1 + (n - frame) // hop
    energy = np.empty(n_frames, dtype=np.float64)
    t_centre = np.empty(n_frames, dtype=np.float64)
    for k in range(n_frames):
        s = k * hop
        seg = x[s : s + frame] * window
        spec = np.fft.rfft(seg)
        energy[k] = float(np.sum(np.abs(spec[band]) ** 2))
        t_centre[k] = (s + frame / 2.0) / sample_rate
    return (t_centre, energy)


def detect_rumble_onset(
    x: np.ndarray,
    sample_rate: int,
    cfg: AudioDetectorConfig | None = None,
) -> list[RumbleCandidate]:
    """Detect a sustained rise in low-frequency energy (a collapse rumble).

    Mirrors :func:`wtc4d.sync.events.detect_collapse_onset`'s shape: a robust
    z-score flags a candidate, and a persistence check over
    ``cfg.persist_s`` throws out anything that does not sustain -- a single
    loud transient (a door, a passing truck, a microphone bump) spikes the
    same way a rumble onset does but does not keep going.
    """
    cfg = cfg or AudioDetectorConfig()
    t_centre, energy = _bandpass_energy_envelope(x, sample_rate, cfg.band_hz, cfg.frame_s)
    if len(energy) < 8:
        return []
    energy_db = 10.0 * np.log10(energy + 1e-12)
    med = float(np.median(energy_db))
    mad = float(np.median(np.abs(energy_db - med))) * 1.4826 or 1.0
    z = (energy_db - med) / mad

    out: list[RumbleCandidate] = []
    n = len(energy_db)
    hop_s = float(t_centre[1] - t_centre[0]) if n > 1 else cfg.frame_s
    for i in range(1, n - 1):
        if z[i] < cfg.z_threshold:
            continue
        j = i
        while j < n - 1 and t_centre[j] - t_centre[i] < cfg.persist_s:
            j += 1
        window = energy_db[i:j] - med
        peak = energy_db[i] - med
        if window.size < 3 or peak <= 0 or window.mean() < cfg.persist_floor * peak:
            continue
        out.append(
            RumbleCandidate(
                t_rel=float(t_centre[i]),
                sigma_rel=hop_s / 2.0,
                confidence=float(min(1.0, z[i] / (3.0 * cfg.z_threshold))),
                band_energy_db=float(energy_db[i]),
            )
        )
    # Collapse each run of candidates into the *earliest* one per cluster (see
    # wtc4d.sync.events._suppress for the full rationale): a sustained rumble
    # clears the threshold on many consecutive frames and its z-score need
    # not decrease monotonically (frame-boundary effects on a low-frequency
    # tone ripple it), so keeping whichever frame scores highest can report a
    # time seconds after the true onset.  Clustering is by the gap between
    # *consecutive* candidates, not distance from the kept representative --
    # gating on the latter would re-report a fresh "onset" every
    # `cfg.persist_s` for any rumble that runs longer than that, which real
    # collapse rumbles routinely do (tens of seconds).  `out` is already in
    # ascending time order (built by a forward scan over frames).
    kept: list[RumbleCandidate] = []
    prev_t: float | None = None
    for c in out:
        if prev_t is None or c.t_rel - prev_t >= cfg.persist_s:
            kept.append(c)
        prev_t = c.t_rel
    return kept


RUMBLE_DETECTION_LAG_S = 3.0
"""Sound travels slower than light gets to a camera: at ~340 m/s, a
microphone several km from the towers hears a collapse several seconds after
it is seen.  This is a *distance-dependent* delay, unlike
:data:`wtc4d.sync.events.COLLAPSE_DETECTION_LAG_S`'s detector lag, so it
cannot be corrected without knowing (at least roughly) how far the recording
device was from the towers.  Left at 0 by default (near-field recordings
dominate the corpus); pass ``distance_m`` to :func:`rumble_anchor_estimate` to
apply ``distance_m / 340``.
"""

SPEED_OF_SOUND_M_S = 340.0


def rumble_anchor_estimate(
    candidate: RumbleCandidate,
    anchor_id: str,
    *,
    media_offset_s: float = 0.0,
    distance_m: float = 0.0,
    distance_sigma_m: float = 500.0,
    shot_id: str = "",
) -> TimeEstimate:
    """Absolute time of media offset 0, from a rumble matched to a collapse anchor.

    ``distance_m`` is the recording location's straight-line distance from
    the WTC complex; its travel-time is added to how late the rumble was
    heard, and its uncertainty (default 500 m -- "somewhere in Lower
    Manhattan or across the river" if unknown) is propagated into the sigma.
    """
    ev: Event | None = EVENTS_BY_ID.get(anchor_id)
    if ev is None:
        raise KeyError(f"unknown timeline anchor {anchor_id!r}")
    travel_s = distance_m / SPEED_OF_SOUND_M_S
    travel_sigma_s = distance_sigma_m / SPEED_OF_SOUND_M_S
    t_heard = ev.t + travel_s
    t_zero = t_heard - (candidate.t_rel + media_offset_s)
    sigma = math.sqrt(ev.sigma**2 + candidate.sigma_rel**2 + travel_sigma_s**2)
    return TimeEstimate(
        t=t_zero,
        sigma=sigma,
        method=TimeMethod.AUDIO_XCORR,
        evidence=(
            f"low-frequency rumble ({candidate.band_energy_db:.1f} dB) detected at media "
            f"offset {candidate.t_rel + media_offset_s:.3f} s, confidence "
            f"{candidate.confidence:.2f}, matched to {ev.id} @ {fmt_local(ev.t)} "
            f"+-{ev.sigma:g} s [{ev.source}]"
            + (f"; +{travel_s:.2f} s sound travel time ({distance_m:.0f} m)" if distance_m else "")
        ),
        derived_from=[shot_id] if shot_id else [],
    )


# --- landmark matching --------------------------------------------------------


class AudioLandmark(BaseModel):
    """A short reference audio clip of known absolute time.

    Typically a distinctive sentence from a live broadcast that several
    recordings happened to pick up (a bystander's radio, a nearby TV).  Its
    ``time`` is the absolute time of the *start* of the reference file, from
    whatever cue timed that source in the first place (usually broadcast
    metadata or clock OCR) -- this is how an untimed clip gets chained onto
    the TV archive without itself containing a clock or an anchor event.
    """

    id: str
    path: str
    time: TimeEstimate
    sample_rate: int = DEFAULT_SAMPLE_RATE
    notes: str = ""


def match_landmark(
    clip_path: str,
    landmark: AudioLandmark,
    *,
    clip_id: str = "",
    start_s: float = 0.0,
    duration_s: float | None = None,
    rate_candidates: tuple[float, ...] = DEFAULT_RATE_CANDIDATES,
    max_shift_s: float | None = None,
    min_confidence: float = 0.15,
) -> TimeEstimate | None:
    """Absolute time of ``clip_path``'s media offset 0, via a known landmark.

    Internally this is :func:`align_clips` with the landmark's own audio as
    the reference, followed by folding the landmark's own time and sigma into
    the result.
    """
    off = align_clips(
        clip_path,
        landmark.path,
        a_id=clip_id or clip_path,
        b_id=landmark.id,
        sample_rate=landmark.sample_rate,
        start_s=start_s,
        duration_s=duration_s,
        rate_candidates=rate_candidates,
        max_shift_s=max_shift_s,
        min_confidence=min_confidence,
    )
    if off is None:
        return None
    # off.dt = t(landmark window) - t(clip window).  The landmark window
    # starts `start_s` into the landmark file, whose own offset 0 is at
    # landmark.time.t, so t(landmark window) = landmark.time.t + start_s;
    # the clip window is offset the same `start_s` into the clip file, so
    # those two `start_s` terms cancel when solving for the clip's own
    # offset-0 time -- see tests/test_sync_audio.py for the worked check.
    t_zero = landmark.time.t - off.dt
    sigma = math.sqrt(landmark.time.sigma**2 + off.sigma**2)
    return TimeEstimate(
        t=t_zero,
        sigma=sigma,
        method=TimeMethod.AUDIO_XCORR,
        evidence=(
            f"audio landmark {landmark.id!r} @ {fmt_local(landmark.time.t)} "
            f"+-{landmark.time.sigma:g} s; {off.evidence}"
        ),
        derived_from=[landmark.id],
    )


__all__ = [
    "DEFAULT_RATE_CANDIDATES",
    "DEFAULT_SAMPLE_RATE",
    "RUMBLE_DETECTION_LAG_S",
    "SPEED_OF_SOUND_M_S",
    "AudioDetectorConfig",
    "AudioLandmark",
    "CorrelationResult",
    "Correlogram",
    "RumbleCandidate",
    "align_clips",
    "align_signals",
    "best_shift",
    "confidence_from_correlation",
    "detect_rumble_onset",
    "gcc_phat",
    "match_landmark",
    "parabolic_peak",
    "resample_linear",
    "rumble_anchor_estimate",
    "sigma_from_correlation",
]
