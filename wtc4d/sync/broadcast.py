"""Absolute time from TV-archive metadata.

The Internet Archive's *Understanding 9/11* television collection
(``collection:911``) is the single richest source of already-timed footage of
the morning.  Its items are half-hour (or hour, or three-hour) recordings made
off-air by the TV News Archive's capture station, and each carries an airtime
to the second.

What the metadata actually contains
-----------------------------------
Verified against live items (see ``data/time/examples/``) on 2026-09-13:

``identifier``
    ``CHANNEL_YYYYMMDD_HHMMSS_Program_Title``.  The timestamp in the
    identifier is **UTC**, and equals ``start_time``.
``start_time`` / ``stop_time``
    ``"2001-09-11 13:00:00"``, naive, **UTC**.
``start_localtime``
    The same instant in the *capture station's* local time.
``utc_offset``
    Signed ``HHMM`` of that local zone (``-400`` for EDT, ``100`` for BST).
    ``start_localtime == start_time + utc_offset`` always held on the items
    checked, which is how :meth:`ArchiveItemTiming.consistent` validates a
    parse.
``runtime``
    ``HH:MM:SS`` -- the real duration of the media, typically a few seconds
    *short* of the nominal slot (``00:29:56`` for a 30-minute slot).
    ``stop_time - start_time == runtime``, so the missing seconds are at the
    **end**: media offset 0 corresponds to ``start_time``, which is the
    assumption used here.
``frames_per_second``
    29.97 for the NTSC captures.
``imagecount``
    *Not* a frame count: it is the number of derivative thumbnails, one per
    second, so ``imagecount ~= runtime`` in seconds (CNN: 10798 vs 10797 s).
``source``
    The reception chain, e.g. ``"Antenna > FutureTel NS320 > ..."`` or
    ``"DISH Network > FutureTel NS320 > ..."``.  This matters: a channel taken
    off a DBS satellite receiver arrives seconds later than the same content
    taken off-air, because of the uplink MPEG-2 encoder and the set-top
    decoder buffer.  :attr:`ArchiveItemTiming.reception` classifies it and the
    delay table is keyed on it.
``previous_item`` / ``next_item``
    The adjacent slots of the same channel, so a long event can be followed
    across a segment boundary.

Two corrections stand between ``start_time`` and the time an event *happened*:

1. **Archive clock error** -- how well the capture station's segmentation
   matched wall-clock time.  Treated as a fixed per-collection sigma
   (:data:`ARCHIVE_CLOCK_SIGMA_S`), not a bias.
2. **Channel delay** -- the broadcast chain between the event and the antenna
   the archive recorded from: camera link, master control, network switching,
   satellite hop, cable headend, encoder buffer.  This is a genuine positive
   bias of order seconds, it differs per channel, and it is tabulated in
   ``data/time/channel_delays.yaml``.

So::

    t_event = archive_start_time + media_offset - channel_delay
    sigma   = sqrt(archive_clock_sigma^2 + channel_delay_sigma^2 + offset_sigma^2)
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from wtc4d.schema.corpus import Shot, Source
from wtc4d.schema.time import TimeEstimate, TimeMethod
from wtc4d.timeline import from_datetime

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "time"
CHANNEL_DELAYS_PATH = DATA_DIR / "channel_delays.yaml"
EXAMPLES_DIR = DATA_DIR / "examples"

ARCHIVE_CLOCK_SIGMA_S = 2.0
"""1-sigma of the TV News Archive segmentation clock, seconds.

Not a measured quantity yet -- it is a deliberately conservative allowance for
NTP drift plus MPEG encoder segmentation jitter at the capture station.  The
one cross-check we have (``wtc4d/sync/README.md``, "Validation") is consistent
with an error of a couple of seconds or less.  Lower it only with evidence.
"""

IDENTIFIER_RE = re.compile(
    r"^(?P<channel>[A-Z0-9]+)_(?P<date>\d{8})_(?P<time>\d{6})(?:_(?P<title>.*))?$"
)


class ArchiveTimingError(ValueError):
    """The metadata could not be parsed into a usable timing."""


def parse_archive_datetime(value: str) -> datetime:
    """``"2001-09-11 13:00:00"`` -> timezone-aware UTC datetime."""
    v = value.strip().replace("T", " ").rstrip("Z")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(v, fmt).replace(tzinfo=UTC)
        except ValueError:
            continue
    raise ArchiveTimingError(f"unparsable archive timestamp {value!r}")


def parse_runtime(value: str) -> float:
    """``"00:29:56"`` (or ``"29:56"``, or ``"1796.26"``) -> seconds."""
    v = value.strip()
    if ":" not in v:
        return float(v)
    parts = [float(p) for p in v.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    h, m, s = parts[-3:]
    return h * 3600.0 + m * 60.0 + s


def parse_utc_offset(value: str | int) -> int:
    """``-400`` / ``"-400"`` / ``"+0100"`` -> minutes east of UTC."""
    s = str(value).strip()
    sign = -1 if s.startswith("-") else 1
    digits = s.lstrip("+-").zfill(4)
    return sign * (int(digits[:-2]) * 60 + int(digits[-2:]))


def parse_identifier(identifier: str) -> tuple[str, datetime, str]:
    """``CHANNEL_YYYYMMDD_HHMMSS_Title`` -> (channel, UTC datetime, title)."""
    m = IDENTIFIER_RE.match(identifier.strip())
    if not m:
        raise ArchiveTimingError(f"not a TV-archive identifier: {identifier!r}")
    dt = datetime.strptime(m["date"] + m["time"], "%Y%m%d%H%M%S").replace(tzinfo=UTC)
    return m["channel"], dt, (m["title"] or "").replace("_", " ")


class ArchiveItemTiming(BaseModel):
    """Timing facts for one archive.org TV-archive item."""

    identifier: str
    channel: str
    title: str = ""
    start_t: float = Field(description="project seconds at media offset 0")
    stop_t: float | None = None
    runtime_s: float | None = None
    fps: float | None = None
    imagecount: int | None = None
    source_chain: str = ""
    utc_offset_min: int | None = None
    start_localtime: str | None = None
    previous_item: str | None = None
    next_item: str | None = None

    # --- constructors ---------------------------------------------------

    @classmethod
    def from_identifier(cls, identifier: str) -> ArchiveItemTiming:
        """Timing from the identifier alone (the timestamp there is UTC)."""
        channel, dt, title = parse_identifier(identifier)
        return cls(
            identifier=identifier,
            channel=channel,
            title=title,
            start_t=from_datetime(dt),
        )

    @classmethod
    def from_metadata(cls, md: dict) -> ArchiveItemTiming:
        """Timing from an archive.org metadata dict.

        Accepts either the item's ``metadata`` sub-dict or the full
        ``item_metadata`` structure, so both
        ``get_item(id).item_metadata`` and ``...item_metadata["metadata"]``
        work.
        """
        if "metadata" in md and isinstance(md["metadata"], dict):
            md = md["metadata"]
        identifier = str(md.get("identifier") or "")
        if not identifier:
            raise ArchiveTimingError("metadata has no identifier")
        channel, id_dt, title = parse_identifier(identifier)

        start_dt = parse_archive_datetime(str(md["start_time"])) if md.get("start_time") else id_dt
        stop_dt = parse_archive_datetime(str(md["stop_time"])) if md.get("stop_time") else None
        runtime = parse_runtime(str(md["runtime"])) if md.get("runtime") else None
        if runtime is None and stop_dt is not None:
            runtime = (stop_dt - start_dt).total_seconds()

        fps = None
        if md.get("frames_per_second"):
            try:
                fps = float(md["frames_per_second"])
            except (TypeError, ValueError):
                fps = None
        imagecount = None
        if md.get("imagecount"):
            try:
                imagecount = int(md["imagecount"])
            except (TypeError, ValueError):
                imagecount = None

        return cls(
            identifier=identifier,
            channel=channel,
            title=str(md.get("title") or title),
            start_t=from_datetime(start_dt),
            stop_t=from_datetime(stop_dt) if stop_dt else None,
            runtime_s=runtime,
            fps=fps,
            imagecount=imagecount,
            source_chain=str(md.get("source") or ""),
            utc_offset_min=parse_utc_offset(md["utc_offset"]) if md.get("utc_offset") else None,
            start_localtime=str(md["start_localtime"]) if md.get("start_localtime") else None,
            previous_item=str(md["previous_item"]) if md.get("previous_item") else None,
            next_item=str(md["next_item"]) if md.get("next_item") else None,
        )

    # --- checks ----------------------------------------------------------

    def consistent(self) -> tuple[bool, list[str]]:
        """Cross-check the redundant timing fields.

        Returns ``(ok, problems)``.  A failure means the item's metadata does
        not follow the collection's usual shape and its airtime should not be
        trusted without a look.
        """
        problems: list[str] = []
        _, id_dt, _ = parse_identifier(self.identifier)
        if abs(from_datetime(id_dt) - self.start_t) > 1.0:
            problems.append(
                f"identifier timestamp {id_dt:%Y-%m-%d %H:%M:%S}Z disagrees with start_time"
            )
        if self.start_localtime and self.utc_offset_min is not None:
            local = parse_archive_datetime(self.start_localtime)
            expected = from_datetime(local) - self.utc_offset_min * 60.0
            if abs(expected - self.start_t) > 1.0:
                problems.append("start_localtime - utc_offset does not equal start_time")
        if self.stop_t is not None and self.runtime_s is not None:
            if abs((self.stop_t - self.start_t) - self.runtime_s) > 2.0:
                problems.append("stop_time - start_time does not equal runtime")
        if self.imagecount and self.runtime_s:
            # imagecount is the one-per-second thumbnail count, not a frame count
            if abs(self.imagecount - self.runtime_s) > 5.0:
                problems.append("imagecount does not match runtime (one thumbnail per second)")
        return (not problems, problems)

    @property
    def reception(self) -> str:
        """How the capture station received the channel: antenna | satellite | cable | unknown."""
        s = self.source_chain.lower()
        if "antenna" in s:
            return "antenna"
        if "dish" in s or "directv" in s or "satellite" in s:
            return "satellite"
        if "cable" in s or "comcast" in s or "rcn" in s:
            return "cable"
        return "unknown"

    # --- timing ----------------------------------------------------------

    def covers(self, t: float) -> bool:
        end = self.stop_t if self.stop_t is not None else (self.start_t + (self.runtime_s or 0.0))
        return self.start_t <= t < end

    def offset_of(self, t: float) -> float:
        """Project seconds -> media offset in seconds (uncorrected)."""
        return t - self.start_t

    def time_at_offset(
        self,
        offset_s: float,
        *,
        delays: ChannelDelays | None = None,
        offset_sigma_s: float = 0.0,
    ) -> TimeEstimate:
        """Absolute time of a point ``offset_s`` into the media.

        The channel delay is *subtracted*: what the recording shows at a given
        airtime happened slightly earlier.
        """
        delays = delays or load_channel_delays()
        delay = delays.for_channel(self.channel, reception=self.reception)
        t = self.start_t + offset_s - delay.delay_s
        sigma = (ARCHIVE_CLOCK_SIGMA_S**2 + delay.sigma_s**2 + offset_sigma_s**2) ** 0.5
        return TimeEstimate(
            t=t,
            sigma=sigma,
            method=TimeMethod.BROADCAST_METADATA,
            evidence=(
                f"archive.org {self.identifier} start_time "
                f"{self.start_t:.0f} project-s + media offset {offset_s:.3f} s "
                f"- {self.channel} ({self.reception}) chain delay {delay.delay_s:.2f} s "
                f"(+-{delay.sigma_s:.2f}); archive clock sigma "
                f"{ARCHIVE_CLOCK_SIGMA_S:.1f} s. {delay.evidence}"
            ),
            derived_from=[f"ia:{self.identifier}"],
        )

    def frame_time(
        self,
        frame_idx: int,
        *,
        fps: float | None = None,
        delays: ChannelDelays | None = None,
    ) -> TimeEstimate:
        """Absolute time of frame ``frame_idx`` counted from media offset 0."""
        rate = fps or self.fps
        if not rate:
            raise ArchiveTimingError(f"{self.identifier}: no frame rate available")
        return self.time_at_offset(frame_idx / rate, delays=delays)


# --- channel delay table -----------------------------------------------------


class ChannelDelay(BaseModel):
    """Seconds between an event happening and it reaching the recorder."""

    delay_s: float = 0.0
    sigma_s: float = Field(default=2.0, gt=0.0)
    evidence: str = ""
    verified: bool = False


class ChannelDelays(BaseModel):
    """The whole ``data/time/channel_delays.yaml`` table."""

    default: ChannelDelay = ChannelDelay()
    reception: dict[str, ChannelDelay] = Field(
        default_factory=dict,
        description="fallback by reception path: antenna | satellite | cable",
    )
    channels: dict[str, ChannelDelay] = Field(default_factory=dict)
    notes: str = ""

    def for_channel(self, channel: str | None, *, reception: str | None = None) -> ChannelDelay:
        """Most specific entry available: channel, then reception path, then default."""
        if channel:
            hit = self.channels.get(channel.upper())
            if hit is not None:
                return hit
        if reception:
            hit = self.reception.get(reception.lower())
            if hit is not None:
                return hit
        return self.default

    @classmethod
    def load(cls, path: str | Path | None = None) -> ChannelDelays:
        p = Path(path) if path else CHANNEL_DELAYS_PATH
        if not p.exists():
            return cls()
        raw = yaml.safe_load(p.read_text()) or {}
        return cls.model_validate(raw)


@lru_cache(maxsize=8)
def _load_channel_delays_cached(path: str) -> ChannelDelays:
    return ChannelDelays.load(path)


def load_channel_delays(path: str | Path | None = None) -> ChannelDelays:
    """Load (and cache) the channel delay table."""
    return _load_channel_delays_cached(str(path or CHANNEL_DELAYS_PATH))


# --- item metadata retrieval -------------------------------------------------


def load_item_metadata(path: str | Path) -> dict:
    """Read a cached archive.org metadata JSON from disk."""
    return json.loads(Path(path).read_text())


def fetch_item_metadata(identifier: str, *, cache_dir: str | Path | None = None) -> dict:
    """Fetch item metadata from archive.org (network) with an optional cache.

    Requires the ``internetarchive`` package (in the ``sync`` extra).  Never
    called from tests.
    """
    cache_path = Path(cache_dir) / f"{identifier}.json" if cache_dir else None
    if cache_path and cache_path.exists():
        return load_item_metadata(cache_path)
    try:
        from internetarchive import get_item
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "fetching archive.org metadata needs the 'internetarchive' package: "
            'pip install -e ".[sync]"'
        ) from exc
    md = dict(get_item(identifier).item_metadata)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(md, indent=1, sort_keys=True))
    return md


def timing_for_source(
    source: Source,
    *,
    metadata: dict | None = None,
) -> ArchiveItemTiming | None:
    """Best available :class:`ArchiveItemTiming` for a catalogued source.

    Uses full metadata when given; otherwise falls back to the identifier
    embedded in ``Source.id`` (``ia-<identifier>``) or in the URL.
    """
    if metadata is not None:
        return ArchiveItemTiming.from_metadata(metadata)
    candidates = []
    if source.id.startswith("ia-"):
        candidates.append(source.id[3:])
    m = re.search(r"archive\.org/(?:details|download|metadata)/([^/?#]+)", source.url or "")
    if m:
        candidates.append(m.group(1))
    for cand in candidates:
        try:
            return ArchiveItemTiming.from_identifier(cand)
        except ArchiveTimingError:
            continue
    return None


# --- shot/source level API ---------------------------------------------------


def estimate_for_offset(
    source: Source,
    offset_s: float,
    *,
    timing: ArchiveItemTiming | None = None,
    delays: ChannelDelays | None = None,
) -> TimeEstimate | None:
    """Absolute time of a point ``offset_s`` into a source's media file.

    Prefers a TV-archive :class:`ArchiveItemTiming` (channel delay aware); if
    there is none, falls back to ``Source.time_hint``, which the corpus
    workstream fills with whatever the catalogue knew (EXIF, upload date, a
    manual note).  ``time_hint`` is documented as the time of the *start of
    the item*, so the offset is simply added and the hint's own sigma is
    carried through unchanged.
    """
    timing = timing or timing_for_source(source)
    if timing is not None:
        return timing.time_at_offset(offset_s, delays=delays)
    hint = source.time_hint
    if hint is None:
        return None
    return TimeEstimate(
        t=hint.t + offset_s,
        sigma=hint.sigma,
        method=hint.method,
        evidence=f"Source.time_hint of {source.id} + media offset {offset_s:.3f} s. {hint.evidence}",
        derived_from=[source.id, *hint.derived_from],
    )


def estimate_for_shot(
    source: Source,
    shot: Shot,
    *,
    timing: ArchiveItemTiming | None = None,
    delays: ChannelDelays | None = None,
) -> TimeEstimate | None:
    """Absolute time of ``shot.start_frame``, from source-level metadata."""
    return estimate_for_offset(source, shot.start_s, timing=timing, delays=delays)


__all__ = [
    "ARCHIVE_CLOCK_SIGMA_S",
    "ArchiveItemTiming",
    "ArchiveTimingError",
    "ChannelDelay",
    "ChannelDelays",
    "estimate_for_offset",
    "estimate_for_shot",
    "fetch_item_metadata",
    "load_channel_delays",
    "load_item_metadata",
    "parse_archive_datetime",
    "parse_identifier",
    "parse_runtime",
    "parse_utc_offset",
    "timing_for_source",
]
