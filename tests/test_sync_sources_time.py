"""EXIF / container-metadata timing tests.

EXIF fixtures are written in-memory with Pillow's own ``Image.Exif`` (no
extra dependency, no network); container-metadata fixtures use ffmpeg to mux
a tiny synthetic clip with a ``creation_time`` tag, guarded on ffmpeg being
on PATH so the suite still passes in an environment without it.
"""

import subprocess

import pytest
from PIL import Image

from wtc4d.sync import sources_time as ST
from wtc4d.sync.video import have_ffmpeg
from wtc4d.timeline import fmt_local


def _jpeg_with_exif(path, *, date_original=None, offset=None) -> None:
    img = Image.new("RGB", (8, 8), color=(200, 100, 50))
    exif = Image.Exif()
    if date_original:
        exif[36867] = date_original
    if offset:
        exif[36881] = offset
    img.save(path, format="JPEG", exif=exif)


def test_exif_datetime_reads_date_time_original(tmp_path):
    p = tmp_path / "photo.jpg"
    _jpeg_with_exif(p, date_original="2001:09:11 08:50:00")
    dt, field, raw = ST.exif_datetime(p)
    assert dt is not None
    assert (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second) == (
        2001,
        9,
        11,
        8,
        50,
        0,
    )
    assert field == "EXIF:DateTimeOriginal"


def test_exif_datetime_none_without_exif(tmp_path):
    p = tmp_path / "plain.jpg"
    Image.new("RGB", (8, 8)).save(p, format="JPEG")
    dt, field, raw = ST.exif_datetime(p)
    assert dt is None


def test_photo_time_applies_camera_offset(tmp_path):
    p = tmp_path / "photo.jpg"
    _jpeg_with_exif(p, date_original="2001:09:11 09:50:00")
    offsets = ST.CameraClockOffsets(
        by_source={"src1": ST.CameraClockOffset(offset_s=3600.0, sigma_s=5.0, evidence="1h fast")}
    )
    res = ST.photo_time(p, source_id="src1", offsets=offsets)
    assert res.ok
    # true = camera - offset: 09:50:00 - 3600s = 08:50:00
    assert fmt_local(res.estimate.t) == "08:50:00"
    assert res.estimate.sigma == 5.0


def test_photo_time_rejects_implausible_date(tmp_path):
    p = tmp_path / "photo.jpg"
    _jpeg_with_exif(p, date_original="2000:01:01 00:00:00")  # camera fresh out of the box
    res = ST.photo_time(p)
    assert not res.ok
    assert "outside the 2001-09-11/12 window" in res.rejected_reason


def test_photo_time_no_exif_gives_reason(tmp_path):
    p = tmp_path / "plain.jpg"
    Image.new("RGB", (8, 8)).save(p, format="JPEG")
    res = ST.photo_time(p)
    assert not res.ok
    assert "no EXIF" in res.rejected_reason


def test_camera_clock_offsets_lookup():
    offsets = ST.CameraClockOffsets(
        by_source={"s1": ST.CameraClockOffset(offset_s=1.0)},
        by_camera={"Canon PowerShot S100": ST.CameraClockOffset(offset_s=2.0)},
    )
    assert offsets.lookup("s1", None).offset_s == 1.0
    assert offsets.lookup(None, "Canon PowerShot S100").offset_s == 2.0
    assert offsets.lookup("unknown", "unknown") is None


@pytest.mark.skipif(not have_ffmpeg(), reason="ffmpeg not on PATH")
def test_container_creation_time_and_rejection(tmp_path):
    good = tmp_path / "good.mp4"
    bad = tmp_path / "bad.mp4"
    _make_tiny_mp4(good, creation_time="2001-09-11T09:15:00")
    _make_tiny_mp4(bad, creation_time="2024-01-01T00:00:00")  # a re-encode date

    dt, field, raw = ST.container_creation_time(good)
    assert dt is not None and dt.year == 2001

    res_good = ST.video_metadata_time(good)
    assert res_good.ok

    res_bad = ST.video_metadata_time(bad)
    assert not res_bad.ok
    assert "transcode" in res_bad.rejected_reason


@pytest.mark.skipif(not have_ffmpeg(), reason="ffmpeg not on PATH")
def test_file_time_dispatches_by_extension(tmp_path):
    p = tmp_path / "photo.jpg"
    _jpeg_with_exif(p, date_original="2001:09:11 10:00:00")
    res = ST.file_time(p)
    assert res.ok
    assert fmt_local(res.estimate.t) == "10:00:00"

    v = tmp_path / "clip.mp4"
    _make_tiny_mp4(v, creation_time="2001-09-11T11:00:00")
    res_v = ST.file_time(v)
    assert res_v.ok


def _make_tiny_mp4(path, *, creation_time: str) -> None:
    """A one-frame mp4 with a given container creation_time, via ffmpeg."""
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-nostdin",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=32x32:d=0.1",
            "-metadata",
            f"creation_time={creation_time}",
            str(path),
        ],
        check=True,
    )
