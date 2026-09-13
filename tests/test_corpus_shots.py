import pytest

pytest.importorskip("cv2")
pytest.importorskip("scenedetect")

from wtc4d.corpus import shots  # noqa: E402
from wtc4d.schema import TimeEstimate  # noqa: E402
from wtc4d.schema.time import TimeMethod  # noqa: E402


def test_classify_motion_static():
    assert (
        shots.classify_motion(
            {"avg_mag": 0.0, "mag_cv": 0.0, "ang_circvar": 1.0, "radial_dot": 0.0}
        )
        == "static"
    )


def test_classify_motion_pan_tilt():
    stats = {"avg_mag": 2.0, "mag_cv": 0.05, "ang_circvar": 0.01, "radial_dot": 0.01}
    assert shots.classify_motion(stats) == "pan_tilt"


def test_classify_motion_aerial_parallax():
    stats = {"avg_mag": 2.0, "mag_cv": 0.9, "ang_circvar": 0.01, "radial_dot": 0.01}
    assert shots.classify_motion(stats) == "aerial"


def test_classify_motion_handheld():
    stats = {"avg_mag": 2.0, "mag_cv": 0.8, "ang_circvar": 0.9, "radial_dot": 0.0}
    assert shots.classify_motion(stats) == "handheld"


def test_classify_motion_zoom():
    stats = {"avg_mag": 2.0, "mag_cv": 0.1, "ang_circvar": 0.9, "radial_dot": 0.8}
    assert shots.classify_motion(stats) == "zoom"


def test_detect_shots_static_video(static_video):
    result = shots.detect_shots(static_video, "src-static")
    assert len(result) == 1
    assert result[0].camera_motion == "static"
    assert result[0].source_id == "src-static"
    assert 0.0 <= result[0].quality_score <= 1.0


def test_detect_shots_pan_video(pan_video):
    result = shots.detect_shots(pan_video, "src-pan")
    assert len(result) == 1
    assert result[0].camera_motion == "pan_tilt"


def test_detect_shots_propagates_time_hint(static_video):
    hint = TimeEstimate(t=1000.0, sigma=3.0, method=TimeMethod.BROADCAST_METADATA, evidence="test")
    result = shots.detect_shots(static_video, "src-static", source_time_hint=hint)
    assert result[0].time is not None
    assert result[0].time.t == pytest.approx(1000.0)
    assert result[0].time.derived_from == ["src-static"]


def test_shot_start_end_seconds_consistent_with_fps(static_video):
    result = shots.detect_shots(static_video, "src-static")
    shot = result[0]
    assert shot.start_s == pytest.approx(shot.start_frame / shot.fps)
    assert shot.end_s == pytest.approx(shot.end_frame / shot.fps)
