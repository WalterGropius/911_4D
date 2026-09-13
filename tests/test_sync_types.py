from wtc4d.schema.time import TimeEstimate, TimeMethod
from wtc4d.sync.types import BBox, LinearClock, PairwiseOffset, RoiTrack, ShotTimeRecord


def test_bbox_clip_and_slices():
    b = BBox(x=-5, y=10, w=20, h=20)
    c = b.clip(15, 15)
    assert c.x >= 0 and c.y >= 0 and c.x1 <= 15 and c.y1 <= 15
    rows, cols = c.as_slices()
    assert rows.stop <= 15 and cols.stop <= 15


def test_bbox_scaled():
    b = BBox(x=10, y=20, w=100, h=50)
    s = b.scaled(0.5, 0.5)
    assert (s.x, s.y, s.w, s.h) == (5, 10, 50, 25)


def test_roi_track_static_and_interp():
    a = BBox(x=0, y=0, w=10, h=10)
    b = BBox(x=100, y=0, w=10, h=10)
    track = RoiTrack(boxes={0: a, 100: b})
    assert track.at(0) == a
    assert track.at(100) == b
    mid = track.at(50)
    assert mid is not None and 40 <= mid.x <= 60
    # before/after the key range holds the edge value
    assert track.at(-10) == a
    assert track.at(1000) == b

    static = RoiTrack.static(a)
    assert static.at(0) == a
    assert static.at(999) == a


def test_linear_clock_time_and_sigma():
    clk = LinearClock(t0=1000.0, i0=0, rate=30.0, sigma_t0=0.5, sigma_rate=0.001)
    assert clk.time_of(30) == 1001.0
    assert clk.time_of(0) == 1000.0
    # sigma grows away from i0
    assert clk.sigma_of(0) == 0.5
    assert clk.sigma_of(3000) > clk.sigma_of(30)
    # frame_of inverts time_of
    assert abs(clk.frame_of(clk.time_of(123)) - 123) < 1e-9


def test_shot_time_record_frame_time_uses_clock_when_present():
    est = TimeEstimate(t=100.0, sigma=1.0, method=TimeMethod.ONSCREEN_CLOCK)
    clk = LinearClock(t0=100.0, i0=0, rate=30.0, sigma_t0=0.1, sigma_rate=0.0)
    rec = ShotTimeRecord(shot_id="s1", fps=30.0, start_frame=0, time=est, clock=clk)
    ft = rec.frame_time(30)
    assert ft is not None
    assert abs(ft.t - 101.0) < 1e-9

    rec_no_clock = ShotTimeRecord(shot_id="s2", fps=30.0, start_frame=0, time=est)
    ft2 = rec_no_clock.frame_time(30)
    assert ft2 is not None
    assert abs(ft2.t - 101.0) < 1e-9
    assert ft2.sigma == est.sigma  # nominal-fps extrapolation carries the shot sigma


def test_shot_time_record_frame_time_none_without_time():
    rec = ShotTimeRecord(shot_id="s3", fps=30.0)
    assert rec.frame_time(10) is None


def test_pairwise_offset_roundtrip():
    off = PairwiseOffset(a="s1", b="s2", dt=5.0, sigma=0.2, method=TimeMethod.AUDIO_XCORR)
    off2 = PairwiseOffset.model_validate_json(off.model_dump_json())
    assert off2 == off
