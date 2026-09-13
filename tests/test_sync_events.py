"""Event-anchor detector tests on fully synthetic frame sequences.

No video decoding here -- :class:`~wtc4d.sync.video.DecodedFrame` objects are
built directly from numpy arrays, so these run fast and need no ffmpeg.
"""

import numpy as np
import pytest

from wtc4d.sync import events as E
from wtc4d.sync.types import BBox, RoiTrack
from wtc4d.sync.video import DecodedFrame
from wtc4d.timeline import WTC1_IMPACT, WTC2_COLLAPSE


def _frame(idx, img, fps=30.0):
    return DecodedFrame(index=idx, t_rel=idx / fps, image=img)


def _flat_frame(w, h, value):
    return np.full((h, w), value, dtype=np.uint8)[:, :, None].repeat(3, axis=2).astype(np.uint8)


def _synthetic_fire_burst_frames(n=60, w=64, h=64, burst_at=30):
    """Steady grey scene, plus a fireball (orange patch) starting at `burst_at`."""
    frames = []
    for i in range(n):
        img = _flat_frame(w, h, 120)
        if i >= burst_at:
            # BGR "orange": high R, medium G, low B -> hue ~ 15-25 in OpenCV HSV
            img[20:40, 20:40] = (20, 120, 230)
        frames.append(_frame(i, img))
    return frames


def test_detect_impact_flash_on_synthetic_burst():
    frames = _synthetic_fire_burst_frames()
    roi = RoiTrack.static(BBox(x=10, y=10, w=44, h=44))
    feats = E.compute_features(iter(frames), roi)
    cands = E.detect_impact_flash(feats)
    assert len(cands) == 1
    assert cands[0].frame_idx == 30


def test_detect_impact_flash_rejects_full_frame_cut():
    """A hard cut brightens the *whole* frame, not just the ROI -- must be rejected."""
    n, w, h = 60, 64, 64
    frames = []
    for i in range(n):
        value = 120 if i < 30 else 230  # whole-frame brightness step at i=30
        img = _flat_frame(w, h, value)
        if i >= 30:
            img[20:40, 20:40] = (20, 120, 230)  # ROI also "orange" after the cut
        frames.append(_frame(i, img))
    roi = RoiTrack.static(BBox(x=10, y=10, w=44, h=44))
    feats = E.compute_features(iter(frames), roi)
    cands = E.detect_impact_flash(feats)
    assert cands == []


def _make_feats(
    n: int,
    *,
    fps: float = 30.0,
    onset: int | None = None,
    duration: int = 40,
    baseline_change: float = 0.01,
    event_change: float = 0.4,
    baseline_luma: float = 100.0,
    event_luma: float = 130.0,
    downward: bool = True,
    seed: int = 0,
) -> E.FrameFeatures:
    """Directly construct :class:`~wtc4d.sync.events.FrameFeatures`.

    This tests the *detector's* threshold/persistence/direction/luma logic
    on realistic-shaped signals (a small nonzero baseline change, as any real
    encoded video has, plus a sustained elevated event) without depending on
    the pixel-level statistics of a synthesised frame sequence to happen to
    produce the right robust z-score -- that plumbing is covered separately
    by the flash/luma/cut-rejection tests, which do use real pixel frames.
    """
    rng = np.random.default_rng(seed)
    feats = E.FrameFeatures()
    for i in range(n):
        in_event = onset is not None and onset <= i < onset + duration
        feats.index.append(i)
        feats.t_rel.append(i / fps)
        feats.roi_mean.append(event_luma if in_event else baseline_luma)
        feats.roi_fire.append(0.0)
        if in_event:
            change = max(0.0, event_change + rng.normal(0, event_change * 0.1))
        else:
            # Real encoded video never sits at an exactly constant baseline
            # (compression/sensor noise always jitters it a little); without
            # that jitter the robust z-score's MAD is exactly zero for any
            # event occupying close to half the window, capping z at
            # 1/sqrt(p(1-p)) <= 2 regardless of how large the event is.
            change = max(0.0, rng.normal(baseline_change, baseline_change * 0.4))
        feats.roi_change.append(change)
        # Baseline cy starts at whatever the event's own t=0 value would be
        # (0.0 for a downward-moving event, 1.0 for upward), so the sequence
        # is continuous across the baseline->event boundary -- an artificial
        # jump right at onset would otherwise dominate the 3-frame average
        # the detector's direction check uses, masking the very trend
        # (up vs down) this test means to exercise.
        if in_event:
            frac = (i - onset) / max(1, duration - 1)
            feats.roi_change_cy.append(frac if downward else 1.0 - frac)
        else:
            feats.roi_change_cy.append(0.0 if downward else 1.0)
        feats.outside_change.append(0.0)
    return feats


def test_detect_collapse_onset_on_synthetic_churn():
    feats = _make_feats(90, onset=30, duration=40)
    cands = E.detect_collapse_onset(feats)
    assert len(cands) == 1
    assert 29 <= cands[0].frame_idx <= 31


def test_detect_collapse_onset_rejects_single_frame_fade_to_black():
    """A brief fade-to-black is one spike, not a sustained event."""
    n, w, h = 60, 64, 64
    frames = []
    for i in range(n):
        if 30 <= i < 34:  # a short 4-frame dip to black then back
            value = 5
        else:
            value = 130
        frames.append(_frame(i, _flat_frame(w, h, value)))
    roi = RoiTrack.static(BBox(x=5, y=0, w=54, h=60))
    feats = E.compute_features(iter(frames), roi)
    cands = E.detect_collapse_onset(feats)
    assert cands == []  # luminance floor should catch this


def test_detect_collapse_onset_rejects_upward_motion():
    """Change that drifts up the frame (not down) must not read as a collapse."""
    feats = _make_feats(90, onset=30, duration=40, downward=False)
    cands = E.detect_collapse_onset(feats)
    assert cands == []


def test_detect_collapse_onset_long_sustained_event_reports_once():
    """A very long churn (much longer than persist_s) must yield ONE onset,
    not repeated re-triggers every persist_s -- see the fuse-with-events
    clustering fix documented in _suppress."""
    # Duty cycle matters here: the robust z-score is computed against the
    # *whole* window's median/MAD, so the event must stay a minority of it
    # for the "elevated" frames to actually read as elevated (a window that
    # is mostly collapse footage inverts which value looks anomalous -- not
    # a realistic way to call this detector in the first place).
    feats = _make_feats(500, onset=20, duration=150)
    cands = E.detect_collapse_onset(feats)
    assert len(cands) == 1


# --- anchor matching ---------------------------------------------------------


def test_anchor_estimate_impact_flash_no_lag_correction():
    cand = E.EventCandidate(
        kind=E.EventKind.IMPACT_FLASH, frame_idx=100, t_rel=10.0, confidence=1.0, sigma_rel=0.05
    )
    est = E.anchor_estimate(cand, "wtc1_impact")
    # t_zero + t_rel should equal the anchor time exactly (no lag, no delay)
    assert abs((est.t + cand.t_rel) - WTC1_IMPACT.t) < 1e-9
    assert est.sigma == pytest.approx((WTC1_IMPACT.sigma**2 + cand.sigma_rel**2) ** 0.5)


def test_anchor_estimate_collapse_onset_applies_lag_correction():
    cand = E.EventCandidate(
        kind=E.EventKind.COLLAPSE_ONSET, frame_idx=100, t_rel=10.0, confidence=1.0, sigma_rel=0.05
    )
    est_corrected = E.anchor_estimate(cand, "wtc2_collapse", correct_detection_lag=True)
    est_raw = E.anchor_estimate(cand, "wtc2_collapse", correct_detection_lag=False)
    # the corrected estimate should differ by exactly the lag constant: the
    # detector fires late, so correcting for that pushes the inferred shot
    # start (and hence t_zero) later too, to compensate.
    assert abs((est_corrected.t - est_raw.t) - E.COLLAPSE_DETECTION_LAG_S) < 1e-9
    assert est_corrected.sigma > est_raw.sigma  # lag sigma folded in


def test_match_to_anchors_without_prior_returns_both_candidates():
    cand = E.EventCandidate(
        kind=E.EventKind.IMPACT_FLASH, frame_idx=0, t_rel=0.0, confidence=1.0, sigma_rel=0.05
    )
    ests = E.match_to_anchors([cand])
    assert len(ests) == 2  # ambiguous between wtc1_impact and wtc2_impact


def test_match_to_anchors_with_prior_disambiguates():
    cand = E.EventCandidate(
        kind=E.EventKind.IMPACT_FLASH, frame_idx=0, t_rel=0.0, confidence=1.0, sigma_rel=0.05
    )
    from wtc4d.schema.time import TimeEstimate, TimeMethod

    prior = TimeEstimate(t=WTC2_COLLAPSE.t - 10.0, sigma=5.0, method=TimeMethod.BROADCAST_METADATA)
    ests = E.match_to_anchors([cand], prior=prior)
    # neither impact anchor is anywhere near a prior close to the *collapse*
    # time, so with a tight prior everything should be filtered out
    assert ests == []


def test_roi_track_interpolation_used_in_compute_features():
    # a moving ROI (camera tracking the tower) should still let features compute
    track = RoiTrack(boxes={0: BBox(x=0, y=0, w=10, h=10), 50: BBox(x=40, y=0, w=10, h=10)})
    frames = [_frame(i, _flat_frame(64, 64, 100 + i)) for i in range(0, 60, 10)]
    feats = E.compute_features(iter(frames), track)
    assert len(feats) == 6
