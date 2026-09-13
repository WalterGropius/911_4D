"""Audio alignment tests: synthetic signals with known offset/rate/noise, plus
a real-file round trip through the ffmpeg decode path (skipped if ffmpeg is
not on PATH)."""

import numpy as np
import pytest

from wtc4d.sync import audio as A
from wtc4d.sync.video import have_ffmpeg, write_wav


def _noise_with_onsets(rng, n, sr, onset_times):
    x = rng.normal(0, 1, n).astype(np.float32)
    for onset in onset_times:
        i = int(onset * sr)
        w = int(0.05 * sr)
        if i + w <= n:
            x[i : i + w] += 3.0 * np.hanning(w).astype(np.float32)
    return x


# --- gcc-phat core -----------------------------------------------------------


def test_gcc_phat_recovers_integer_shift_regardless_of_length_mismatch():
    rng = np.random.default_rng(1)
    ref = rng.normal(0, 1, 20000).astype(np.float32)
    for shift in (0, 137, -212, 733):
        sig = np.zeros(len(ref) + 3000, dtype=np.float32)
        base = 1000
        sig[base + shift : base + shift + len(ref)] = ref
        cg = A.gcc_phat(sig, ref, max_shift=3000)
        res = A.best_shift(cg)
        assert abs(res.shift_samples - (base + shift)) < 1e-6


def test_parabolic_peak_subsample_interpolation():
    y = np.array([0.0, 0.0, 0.4, 1.0, 0.6, 0.0, 0.0])
    delta, peak = A.parabolic_peak(y, 3)
    assert -0.5 <= delta <= 0.5
    assert peak >= y[3]


def test_confidence_and_sigma_degrade_for_ambiguous_correlation():
    sharp = A.CorrelationResult(
        shift_samples=0, peak=100.0, noise_floor=1.0, prominence=99.0, second_peak_ratio=0.0
    )
    ambiguous = A.CorrelationResult(
        shift_samples=0, peak=100.0, noise_floor=1.0, prominence=1.0, second_peak_ratio=0.9
    )
    assert A.confidence_from_correlation(sharp) > A.confidence_from_correlation(ambiguous)
    assert A.sigma_from_correlation(sharp, 1 / 8000) < A.sigma_from_correlation(ambiguous, 1 / 8000)


# --- rate + shift search ------------------------------------------------------


def test_align_signals_recovers_known_shift_and_rate():
    rng = np.random.default_rng(0)
    sr = 8000
    ref = _noise_with_onsets(rng, int(20.0 * sr), sr, [2.0, 5.3, 9.7, 14.2, 17.8])

    true_shift_s = 3.257
    true_rate = 30000 / 30030  # NTSC drop-frame ratio

    stretched = A.resample_linear(ref, true_rate)
    shift_samp = int(round(true_shift_s * sr))
    sig = np.zeros(len(stretched) + shift_samp + 2000, dtype=np.float32)
    sig[shift_samp : shift_samp + len(stretched)] = stretched
    sig += rng.normal(0, 0.3, len(sig)).astype(np.float32)
    ref_noisy = ref + rng.normal(0, 0.3, len(ref)).astype(np.float32)

    rate, shift_s, res = A.align_signals(sig, ref_noisy, sr, max_shift_s=10.0)

    expected_rate = 1.0 / true_rate
    expected_shift_s = true_rate * shift_samp / sr  # see resample_linear semantics
    assert abs(rate - expected_rate) < 1e-6
    assert abs(shift_s - expected_shift_s) < 0.001
    assert A.confidence_from_correlation(res) > 0.9


def test_align_signals_low_confidence_on_unrelated_noise():
    rng = np.random.default_rng(2)
    sr = 8000
    a = rng.normal(0, 1, int(10 * sr)).astype(np.float32)
    b = rng.normal(0, 1, int(10 * sr)).astype(np.float32)
    _, _, res = A.align_signals(a, b, sr, rate_candidates=(1.0,), max_shift_s=10.0)
    assert A.confidence_from_correlation(res) < 0.3


# --- rumble onset --------------------------------------------------------


def test_detect_rumble_onset_recovers_known_onset():
    rng = np.random.default_rng(3)
    sr = 8000
    n = int(30.0 * 8000)
    t = np.arange(n) / sr
    x = rng.normal(0, 0.05, n).astype(np.float32) + 0.05 * np.sin(2 * np.pi * 400 * t).astype(
        np.float32
    )
    onset, dur_rumble = 12.0, 8.0
    env = np.zeros(n, dtype=np.float32)
    i0, i1 = int(onset * sr), int((onset + dur_rumble) * sr)
    ramp = np.clip((np.arange(i1 - i0) / sr) / 1.5, 0, 1)
    env[i0:i1] = ramp
    rumble = (0.6 * np.sin(2 * np.pi * 15 * t) + 0.3 * np.sin(2 * np.pi * 22 * t)).astype(
        np.float32
    )
    x += env * rumble

    cands = A.detect_rumble_onset(x, sr)
    assert len(cands) == 1  # one onset reported, not a re-trigger every persist_s
    assert abs(cands[0].t_rel - onset) < 0.5


def test_detect_rumble_onset_empty_for_pure_noise():
    rng = np.random.default_rng(4)
    x = rng.normal(0, 0.05, int(10 * 8000)).astype(np.float32)
    assert A.detect_rumble_onset(x, 8000) == []


def test_rumble_anchor_estimate_applies_travel_time():
    cand = A.RumbleCandidate(t_rel=5.0, sigma_rel=0.1, confidence=1.0, band_energy_db=40.0)
    near = A.rumble_anchor_estimate(cand, "wtc2_collapse", distance_m=0.0, distance_sigma_m=1.0)
    far = A.rumble_anchor_estimate(cand, "wtc2_collapse", distance_m=3400.0, distance_sigma_m=100.0)
    # sound takes 10s to cross 3400m at 340 m/s, so the far recording is
    # inferred to have started its clock 10s later than the near one
    assert abs((far.t - near.t) - 10.0) < 1e-6
    assert far.sigma > near.sigma


# --- real-file round trip (ffmpeg decode path) -------------------------------


@pytest.mark.skipif(not have_ffmpeg(), reason="ffmpeg not on PATH")
def test_align_clips_and_match_landmark_real_files(tmp_path):
    from wtc4d.schema.time import TimeEstimate, TimeMethod

    rng = np.random.default_rng(5)
    sr = 16000
    ref = _noise_with_onsets(rng, int(15 * sr), sr, [2.0, 6.0, 10.0, 13.0])

    a_path = tmp_path / "a.wav"
    b_path = tmp_path / "b.wav"
    write_wav(a_path, ref, sr)  # a: the reference content starting at offset 0
    shift_s = 2.5
    shift_samp = int(round(shift_s * sr))
    # b: shift_s of leading silence, then the same content -- i.e. b's
    # recording started `shift_s` earlier than a's, relative to the shared
    # content, so b's own offset-0 is `shift_s` *before* a's.
    b = np.zeros(len(ref) + shift_samp, dtype=np.float32)
    b[shift_samp:] = ref
    write_wav(b_path, b, sr)

    off = A.align_clips(
        str(a_path), str(b_path), a_id="a", b_id="b", sample_rate=sr, max_shift_s=5.0
    )
    assert off is not None
    assert abs(off.dt - (-shift_s)) < 0.01  # t(b) - t(a) = -shift_s
    assert off.confidence > 0.8

    landmark = A.AudioLandmark(
        id="lm", path=str(a_path), time=TimeEstimate(t=1000.0, sigma=0.2, method=TimeMethod.MANUAL)
    )
    est = A.match_landmark(str(b_path), landmark, max_shift_s=5.0)
    assert est is not None
    assert abs(est.t - (1000.0 - shift_s)) < 0.01
