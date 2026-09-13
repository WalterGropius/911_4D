"""On-screen clock OCR tests.

Parsing, region scoring and the robust interval fit are pure-numpy/regex and
run unconditionally.  One end-to-end test renders synthetic clock-digit
images with Pillow and reads them back through the real OCR engine
(``rapidocr-onnxruntime``); it is the one place in this suite that pays the
OCR model's one-time load cost, and is skipped if the package is not
installed.
"""

import numpy as np
import pytest

from wtc4d.sync import clock_ocr as C
from wtc4d.sync.types import BBox
from wtc4d.timeline import hms

# --- parsing -------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("09:03:11", (9, 3, 11, "E")),
        ("9:58", (9, 58, None, "E")),
        ("9:02a ET", (9, 2, None, "E")),
        ("8:02a CT", (8, 2, None, "C")),
        ("7:03a MT", (7, 3, None, "M")),
        ("6:02a PT", (6, 2, None, "P")),
        ("7:02aMT", (7, 2, None, "M")),  # OCR glued the zone onto the ampm marker
        ("10:28:22 EDT", (10, 28, 22, "E")),
        ("13:00:00 UTC", (13, 0, 0, "UTC")),
        ("ET 9:02", (9, 2, None, "E")),  # zone label before the digits
        ("I0:28:22", (10, 28, 22, "E")),  # OCR 'I' -> '1' digit fixup
        ("O9:O3", (9, 3, None, "E")),  # OCR 'O' -> '0' digit fixup
    ],
)
def test_parse_clock_text_valid(text, expected):
    assert C.parse_clock_text(text, hint_t=hms(9, 0, 0)) == expected


@pytest.mark.parametrize(
    "text",
    [
        "FUTURES:DOW 10.00 NAS 1.50",  # stock ticker, not a clock
        "S&P 0.30",
        "SGP0.20",
        "hello world",
        "99:99",  # not a valid time
    ],
)
def test_parse_clock_text_rejects_non_clocks(text):
    assert C.parse_clock_text(text, hint_t=hms(9, 0, 0)) is None


def test_parse_clock_text_12h_ambiguity_resolved_by_hint():
    # "9:15" with no am/pm marker: hint near 21:15 should pick the PM branch
    assert C.parse_clock_text("9:15", hint_t=hms(21, 10, 0)) == (21, 15, None, "E")
    # hint near 09:15 should pick the AM branch
    assert C.parse_clock_text("9:15", hint_t=hms(9, 10, 0)) == (9, 15, None, "E")


def test_reading_to_project_seconds_zone_conversion():
    # 8:02 CT -> 9:02 ET (Central is 1 hour behind Eastern, so +1h to convert)
    assert C.reading_to_project_seconds(8, 2, None, "C") == hms(9, 2, 0)
    assert C.reading_to_project_seconds(13, 0, 0, "UTC") == hms(9, 0, 0)
    assert C.reading_to_project_seconds(9, 2, 0, "E") == hms(9, 2, 0)


# --- interval fit ----------------------------------------------------------


def test_max_overlap_interval_finds_consensus():
    # three intervals overlapping in [5, 6), one outlier far away
    intervals = [(0.0, 10.0), (4.0, 8.0), (5.0, 6.0), (100.0, 101.0)]
    lo, hi, members = C.max_overlap_interval(intervals)
    assert (lo, hi) == (5.0, 6.0)
    assert sorted(members) == [0, 1, 2]


def _reading(frame_idx, hour, minute, second, zone="E", res=1.0):
    return C.ClockReading(
        frame_idx=frame_idx,
        t_rel=frame_idx / 30.0,
        text=f"{hour}:{minute}",
        hour=hour,
        minute=minute,
        second=second,
        zone=zone,
        confidence=1.0,
        t_display=C.reading_to_project_seconds(hour, minute, second, zone),
        resolution_s=res,
    )


def test_fit_clock_recovers_known_linear_map():
    # ground truth: t = 1000.0 + i/30.0 (30 fps). Sample at frames 0, 300, 600,
    # 900 (i.e. every 10s), with second-resolution readings.
    fps = 30.0
    t0_true = 1000.0
    readings = []
    for i in (0, 300, 600, 900):
        t = t0_true + i / fps
        h, m, s = int(t // 3600), int((t % 3600) // 60), int(t % 60)
        readings.append(_reading(i, h, m, s))
    fit = C.fit_clock(readings, fps=fps)
    assert fit is not None
    assert fit.n_inliers == 4
    assert abs(fit.clock.t0 - t0_true) < 1.0
    assert abs(fit.clock.rate - fps) < 0.5


def test_fit_clock_rejects_outlier_reading():
    fps = 30.0
    t0_true = 2000.0
    readings = []
    for i in (0, 300, 600, 900):
        t = t0_true + i / fps
        h, m, s = int(t // 3600), int((t % 3600) // 60), int(t % 60)
        readings.append(_reading(i, h, m, s))
    # one badly wrong reading (a misread minute) far outside the consensus
    readings.append(_reading(450, 5, 5, 5))
    fit = C.fit_clock(readings, fps=fps)
    assert fit is not None
    assert fit.n_inliers == 4
    assert 450 in fit.outlier_frames
    assert abs(fit.clock.t0 - t0_true) < 1.0


def test_fit_clock_minute_only_readings_pin_transition():
    # Minute-resolution readings spanning a single minute transition should
    # narrow the band to about the sampling gap, not the full 60s bucket.
    fps = 30.0
    readings = [
        _reading(0, 9, 2, None, res=60.0),  # asserts [09:02:00, 09:03:00)
        _reading(60, 9, 2, None, res=60.0),  # frame 60 = +2s, same minute
        _reading(90, 9, 3, None, res=60.0),  # frame 90 = +3s, next minute
    ]
    fit = C.fit_clock(readings, fps=fps, i0=0)
    assert fit is not None
    assert fit.band_s < 5.0  # much tighter than the raw 60s bucket


def test_fit_clock_returns_none_with_too_few_readings():
    assert C.fit_clock([], fps=30.0) is None
    assert C.fit_clock([_reading(0, 9, 0, 0)], fps=30.0, min_inliers=2) is None


# --- temporal-variance region scoring ---------------------------------------


def test_temporal_std_detects_changing_region():
    frames = []
    base = np.full((20, 20), 128, dtype=np.uint8)
    for k in range(5):
        f = base.copy()
        f[5:10, 5:10] = 100 + k * 20  # a ticking patch
        frames.append(f)
    std = C.temporal_std(frames)
    assert std[7, 7] > std[0, 0]


# --- end-to-end OCR on a synthetic clock image ------------------------------


def test_ocr_reads_synthetic_clock_digits():
    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("rapidocr_onnxruntime")

    img = np.full((80, 260, 3), 255, dtype=np.uint8)
    cv2.putText(img, "09:03:11", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 3)

    engine = C.RapidOcrEngine()
    reading = C.read_clock_image(img, engine, hint_t=hms(9, 0, 0))
    assert reading is not None
    assert (reading.hour, reading.minute, reading.second) == (9, 3, 11)


def test_bbox_at_and_clip_used_by_read_clock_sequence():
    box = BBox(x=5, y=5, w=10, h=10)
    clipped = box.clip(8, 8)
    assert clipped.x1 <= 8 and clipped.y1 <= 8
