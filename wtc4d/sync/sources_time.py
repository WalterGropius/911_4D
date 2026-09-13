"""Absolute time from a media file's own metadata (photos and camcorder video).

Covers the non-broadcast half of the corpus: stills with EXIF, MiniDV/Digital8
tapes with a recording date in the DV packs, and anything whose container
carries a ``creation_time``.

The hard part is not reading the fields, it is **not believing them**:

* Consumer camera clocks in 2001 were set by hand and drifted.  A few minutes
  of error is normal, an hour (DST mis-set) is common, and a camera fresh out
  of the box says 2000-01-01.  So an EXIF time gets a large default sigma
  (:data:`EXIF_SIGMA_S`) unless a per-camera offset has been measured and
  recorded in ``data/time/camera_clock_offsets.yaml``.
* A container ``creation_time`` on anything that has been through a
  re-encode (every YouTube download, every DVD rip, most archive derivatives)
  is the *transcode* date, not the shooting date.  Every value produced here
  is therefore checked against :data:`PLAUSIBLE_WINDOW` and rejected -- with a
  stated reason -- when it does not land on the day in question.
* DV timecode is only a time of day if the camera operator set it that way.
  It is offered as :func:`dv_timecode_time` but is off by default.

Timezone: EXIF and DV times are naive local time.  They are read as **EDT**
(the project's local zone) unless the file carries an explicit UTC offset.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from wtc4d.schema.time import TimeEstimate, TimeMethod
from wtc4d.sync.video import probe
from wtc4d.timeline import EDT, fmt_local, from_datetime, hms

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "time"
CAMERA_OFFSETS_PATH = DATA_DIR / "camera_clock_offsets.yaml"

EXIF_SIGMA_S = 300.0
"""Default 1-sigma for a hand-set consumer camera clock, seconds (5 minutes)."""

CONTAINER_SIGMA_S = 120.0
"""Default 1-sigma for a container ``creation_time`` that survived the checks."""

DV_TIMECODE_SIGMA_S = 120.0
"""Default 1-sigma when a DV timecode is interpreted as time of day."""

PLAUSIBLE_WINDOW = (hms(0, 0, 0), hms(48, 0, 0))
"""Project seconds a file-metadata time must fall in to be used at all:
2001-09-11 00:00 EDT through the end of 2001-09-12."""

# EXIF tag numbers (PIL exposes the numeric ids in the Exif IFD).
_TAG_DATETIME_ORIGINAL = 36867
_TAG_DATETIME_DIGITIZED = 36868
_TAG_SUBSEC_ORIGINAL = 37521
_TAG_OFFSET_ORIGINAL = 36881
_TAG_DATETIME = 306
_EXIF_IFD = 34665


class SourceTimeResult(BaseModel):
    """What a metadata read produced, including why it produced nothing."""

    estimate: TimeEstimate | None = None
    field: str = ""
    raw: str = ""
    rejected_reason: str = ""

    @property
    def ok(self) -> bool:
        return self.estimate is not None


class CameraClockOffset(BaseModel):
    """A measured offset of one camera's clock: ``true = camera - offset_s``."""

    offset_s: float = 0.0
    sigma_s: float = Field(default=EXIF_SIGMA_S, gt=0.0)
    evidence: str = ""


class CameraClockOffsets(BaseModel):
    """``data/time/camera_clock_offsets.yaml``, keyed by source id or camera model."""

    default_sigma_s: float = Field(default=EXIF_SIGMA_S, gt=0.0)
    by_source: dict[str, CameraClockOffset] = Field(default_factory=dict)
    by_camera: dict[str, CameraClockOffset] = Field(default_factory=dict)
    notes: str = ""

    def lookup(self, source_id: str | None, camera: str | None) -> CameraClockOffset | None:
        if source_id and source_id in self.by_source:
            return self.by_source[source_id]
        if camera and camera in self.by_camera:
            return self.by_camera[camera]
        return None

    @classmethod
    def load(cls, path: str | Path | None = None) -> CameraClockOffsets:
        p = Path(path) if path else CAMERA_OFFSETS_PATH
        if not p.exists():
            return cls()
        return cls.model_validate(yaml.safe_load(p.read_text()) or {})


def _naive_local_to_project(dt: datetime) -> float:
    """A naive wall-clock datetime, read as EDT, in project seconds."""
    return from_datetime(dt.replace(tzinfo=EDT))


def _check_plausible(t: float, field: str) -> str:
    lo, hi = PLAUSIBLE_WINDOW
    if lo <= t < hi:
        return ""
    return (
        f"{field} resolves to {fmt_local(t)} (project t={t:.0f}), outside the "
        f"2001-09-11/12 window -- almost certainly a transcode or an unset clock, not a capture time"
    )


# --- EXIF --------------------------------------------------------------------


def exif_datetime(path: str | Path) -> tuple[datetime | None, str, str]:
    """Read the best EXIF capture time from a still image.

    Returns ``(datetime | None, field_name, raw_string)``.  The datetime is
    timezone-aware: EDT unless ``OffsetTimeOriginal`` says otherwise.
    """
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError('reading EXIF needs Pillow: pip install -e ".[sync]"') from exc

    with Image.open(path) as img:
        exif = img.getexif()
        if not exif:
            return (None, "", "")
        merged: dict[int, object] = dict(exif)
        try:
            merged.update(dict(exif.get_ifd(_EXIF_IFD)))
        except (KeyError, AttributeError, ValueError):
            pass

    for tag, name in (
        (_TAG_DATETIME_ORIGINAL, "EXIF:DateTimeOriginal"),
        (_TAG_DATETIME_DIGITIZED, "EXIF:DateTimeDigitized"),
        (_TAG_DATETIME, "EXIF:DateTime"),
    ):
        raw = merged.get(tag)
        if not raw:
            continue
        raw_s = str(raw).strip().rstrip("\x00")
        try:
            dt = datetime.strptime(raw_s, "%Y:%m:%d %H:%M:%S")
        except ValueError:
            continue
        subsec = merged.get(_TAG_SUBSEC_ORIGINAL)
        if subsec and str(subsec).strip().isdigit():
            frac = float(f"0.{str(subsec).strip()}")
            dt = dt + timedelta(seconds=frac)
        offset = merged.get(_TAG_OFFSET_ORIGINAL)
        tz = EDT
        if offset:
            m = re.match(r"^([+-])(\d{2}):?(\d{2})$", str(offset).strip())
            if m:
                sign = 1 if m.group(1) == "+" else -1
                delta = timedelta(hours=int(m.group(2)), minutes=int(m.group(3)))
                from datetime import timezone

                tz = timezone(sign * delta)
        return (dt.replace(tzinfo=tz), name, raw_s)
    return (None, "", "")


def photo_time(
    path: str | Path,
    *,
    source_id: str | None = None,
    camera: str | None = None,
    offsets: CameraClockOffsets | None = None,
) -> SourceTimeResult:
    """Absolute time of a still photo from its EXIF, with a defensible sigma."""
    dt, field, raw = exif_datetime(path)
    if dt is None:
        return SourceTimeResult(rejected_reason="no EXIF capture time in file")
    offsets = offsets or CameraClockOffsets.load()
    corr = offsets.lookup(source_id, camera)
    offset_s = corr.offset_s if corr else 0.0
    sigma = corr.sigma_s if corr else offsets.default_sigma_s

    t = from_datetime(dt) - offset_s
    reason = _check_plausible(t, field)
    if reason:
        return SourceTimeResult(field=field, raw=raw, rejected_reason=reason)

    note = (
        f"{corr.evidence} (camera clock offset {offset_s:+.1f} s applied)"
        if corr
        else "no measured camera clock offset; using the default sigma for a hand-set clock"
    )
    return SourceTimeResult(
        field=field,
        raw=raw,
        estimate=TimeEstimate(
            t=t,
            sigma=sigma,
            method=TimeMethod.BROADCAST_METADATA,
            evidence=f"{field}={raw} read as {dt.tzname() or 'EDT'}; {note}",
            derived_from=[source_id] if source_id else [],
        ),
    )


# --- container / DV ----------------------------------------------------------


def container_creation_time(path: str | Path) -> tuple[datetime | None, str, str]:
    """``creation_time`` (or DV ``date``) from the container, via ffprobe."""
    info = probe(path)
    for tags, scope in ((info.format_tags, "format"), (info.stream_tags, "stream")):
        for key in (
            "creation_time",
            "date",
            "date_time_original",
            "com.apple.quicktime.creationdate",
        ):
            raw = tags.get(key)
            if not raw:
                continue
            dt = _parse_iso_like(raw)
            if dt is not None:
                return (dt, f"{scope}:{key}", raw)
    return (None, "", "")


def _parse_iso_like(raw: str) -> datetime | None:
    s = raw.strip()
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y:%m:%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
            try:
                dt = datetime.strptime(s[: len(fmt) + 2], fmt)
                break
            except ValueError:
                continue
        else:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=EDT)
    return dt


def video_metadata_time(
    path: str | Path,
    *,
    source_id: str | None = None,
    sigma_s: float = CONTAINER_SIGMA_S,
) -> SourceTimeResult:
    """Absolute time of media offset 0 from a container ``creation_time``.

    Rejected unless it lands inside :data:`PLAUSIBLE_WINDOW`, because a
    re-encoded file's ``creation_time`` is the transcode date and silently
    trusting it would inject a wildly wrong anchor into the fusion.
    """
    dt, field, raw = container_creation_time(path)
    if dt is None:
        return SourceTimeResult(rejected_reason="no creation_time/date tag in container")
    t = from_datetime(dt)
    reason = _check_plausible(t, field)
    if reason:
        return SourceTimeResult(field=field, raw=raw, rejected_reason=reason)
    return SourceTimeResult(
        field=field,
        raw=raw,
        estimate=TimeEstimate(
            t=t,
            sigma=sigma_s,
            method=TimeMethod.BROADCAST_METADATA,
            evidence=(
                f"container {field}={raw}; only usable because it falls on the day in question "
                "-- confirm the file is an original capture and not a re-encode"
            ),
            derived_from=[source_id] if source_id else [],
        ),
    )


TIMECODE_RE = re.compile(r"^(\d{2}):(\d{2}):(\d{2})[:;](\d{2})$")


def dv_timecode(path: str | Path) -> tuple[str | None, str]:
    """SMPTE timecode string from the container/stream tags, if any."""
    info = probe(path)
    for tags, scope in ((info.stream_tags, "stream"), (info.format_tags, "format")):
        raw = tags.get("timecode")
        if raw and TIMECODE_RE.match(raw.strip()):
            return (raw.strip(), f"{scope}:timecode")
    return (None, "")


def dv_timecode_time(
    path: str | Path,
    *,
    source_id: str | None = None,
    fps: float = 29.97,
    sigma_s: float = DV_TIMECODE_SIGMA_S,
    assume_time_of_day: bool = False,
) -> SourceTimeResult:
    """Interpret a DV/SMPTE timecode as a time of day on 2001-09-11.

    Off unless ``assume_time_of_day`` is set: most camcorders default to a
    record-run timecode that starts at 00:00:00:00 on a fresh tape, which
    would place every shot just after midnight.  Only enable it for a tape
    whose operator is known to have used time-of-day timecode.
    """
    tc, field = dv_timecode(path)
    if tc is None:
        return SourceTimeResult(rejected_reason="no SMPTE timecode in container")
    if not assume_time_of_day:
        return SourceTimeResult(
            field=field,
            raw=tc,
            rejected_reason=(
                "timecode present but not interpreted: pass assume_time_of_day=True only for "
                "tapes known to use time-of-day timecode"
            ),
        )
    m = TIMECODE_RE.match(tc)
    assert m is not None
    h, mm, ss, ff = (int(g) for g in m.groups())
    t = hms(h, mm, ss + ff / fps)
    reason = _check_plausible(t, field)
    if reason:
        return SourceTimeResult(field=field, raw=tc, rejected_reason=reason)
    return SourceTimeResult(
        field=field,
        raw=tc,
        estimate=TimeEstimate(
            t=t,
            sigma=sigma_s,
            method=TimeMethod.BROADCAST_METADATA,
            evidence=f"DV timecode {tc} at {fps:g} fps read as time of day on 2001-09-11",
            derived_from=[source_id] if source_id else [],
        ),
    )


def file_time(
    path: str | Path,
    *,
    source_id: str | None = None,
    camera: str | None = None,
    offsets: CameraClockOffsets | None = None,
    assume_time_of_day_timecode: bool = False,
) -> SourceTimeResult:
    """Try every file-metadata cue, best first, and report what happened.

    Photos go through EXIF; anything else through the container
    ``creation_time`` and then the timecode.  On failure the result carries
    the reason each cue was rejected, which is what the CLI prints.
    """
    p = Path(path)
    reasons: list[str] = []
    if p.suffix.lower() in (".jpg", ".jpeg", ".tif", ".tiff", ".png", ".heic", ".webp"):
        res = photo_time(p, source_id=source_id, camera=camera, offsets=offsets)
        if res.ok:
            return res
        reasons.append(res.rejected_reason)
    res = video_metadata_time(p, source_id=source_id)
    if res.ok:
        return res
    reasons.append(res.rejected_reason)
    res = dv_timecode_time(p, source_id=source_id, assume_time_of_day=assume_time_of_day_timecode)
    if res.ok:
        return res
    reasons.append(res.rejected_reason)
    return SourceTimeResult(rejected_reason="; ".join(r for r in reasons if r))


__all__ = [
    "CONTAINER_SIGMA_S",
    "CameraClockOffset",
    "CameraClockOffsets",
    "DV_TIMECODE_SIGMA_S",
    "EXIF_SIGMA_S",
    "PLAUSIBLE_WINDOW",
    "SourceTimeResult",
    "container_creation_time",
    "dv_timecode",
    "dv_timecode_time",
    "exif_datetime",
    "file_time",
    "photo_time",
    "video_metadata_time",
]
