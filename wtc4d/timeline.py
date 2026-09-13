"""Absolute time for 2001-09-11 and the epoch structure of the reconstruction.

Time convention
---------------
All times in this project are **seconds since local midnight, 2001-09-11
00:00:00 EDT (UTC-4)**, stored as floats.  This makes values human readable
(08:46:30 -> 31590.0) while remaining a plain scalar.  Use :func:`to_utc`
when an absolute datetime is required.

Event anchor times follow NIST NCSTAR 1 (2005) where available.  The 9/11
Commission Report gives slightly different values for the impacts (08:46:40
and 09:03:11); both sets are recorded so the choice is explicit.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

from pydantic import BaseModel

EDT = timezone(timedelta(hours=-4), name="EDT")
LOCAL_MIDNIGHT = datetime(2001, 9, 11, 0, 0, 0, tzinfo=EDT)


def hms(h: int, m: int, s: float = 0.0) -> float:
    """Local clock time -> project seconds."""
    return h * 3600.0 + m * 60.0 + float(s)


def to_utc(t: float) -> datetime:
    """Project seconds -> timezone-aware UTC datetime."""
    return (LOCAL_MIDNIGHT + timedelta(seconds=float(t))).astimezone(UTC)


def to_local(t: float) -> datetime:
    """Project seconds -> timezone-aware EDT datetime."""
    return LOCAL_MIDNIGHT + timedelta(seconds=float(t))


def from_datetime(dt: datetime) -> float:
    """Timezone-aware datetime -> project seconds."""
    if dt.tzinfo is None:
        raise ValueError("datetime must be timezone-aware")
    return (dt - LOCAL_MIDNIGHT).total_seconds()


def fmt_local(t: float) -> str:
    """08:46:30 style string for a project time (with a day suffix past midnight)."""
    dt = to_local(t)
    days = (dt.date() - LOCAL_MIDNIGHT.date()).days
    suffix = f" +{days}d" if days else ""
    return dt.strftime("%H:%M:%S") + suffix


class Event(BaseModel):
    """A time anchor with an uncertainty (1 sigma, seconds)."""

    id: str
    name: str
    t: float
    sigma: float
    source: str
    notes: str = ""


# --- Canonical anchors (NIST NCSTAR 1, Table of key times) -------------------
WTC1_IMPACT = Event(
    id="wtc1_impact",
    name="AA11 strikes WTC1 (North Tower), floors 93-99, north face",
    t=hms(8, 46, 30),
    sigma=5.0,
    source="NIST NCSTAR 1 (08:46:30). 9/11 Commission: 08:46:40.",
)
WTC2_IMPACT = Event(
    id="wtc2_impact",
    name="UA175 strikes WTC2 (South Tower), floors 77-85, south face",
    t=hms(9, 2, 59),
    sigma=5.0,
    source="NIST NCSTAR 1 (09:02:59). 9/11 Commission: 09:03:11.",
)
WTC2_COLLAPSE = Event(
    id="wtc2_collapse",
    name="WTC2 (South Tower) collapse initiation",
    t=hms(9, 58, 59),
    sigma=2.0,
    source="NIST NCSTAR 1 (09:58:59).",
    notes="Total collapse duration roughly 10-15 s to ground level; dust cloud persists for minutes.",
)
WTC1_COLLAPSE = Event(
    id="wtc1_collapse",
    name="WTC1 (North Tower) collapse initiation",
    t=hms(10, 28, 22),
    sigma=2.0,
    source="NIST NCSTAR 1 (10:28:22).",
)
WTC7_COLLAPSE = Event(
    id="wtc7_collapse",
    name="WTC7 collapse (out of core scope, optional late epoch)",
    t=hms(17, 20, 52),
    sigma=2.0,
    source="NIST NCSTAR 1A (17:20:52).",
)

EVENTS: list[Event] = [WTC1_IMPACT, WTC2_IMPACT, WTC2_COLLAPSE, WTC1_COLLAPSE, WTC7_COLLAPSE]
EVENTS_BY_ID: dict[str, Event] = {e.id: e for e in EVENTS}


class Epoch(BaseModel):
    """A time window in which the *static* scene is approximately constant.

    Static gaussian splats are trained per epoch; dynamic content (smoke,
    fire, debris) is handled separately inside each epoch.
    """

    id: str
    name: str
    t_start: float
    t_end: float
    description: str = ""

    def contains(self, t: float) -> bool:
        return self.t_start <= t < self.t_end


# Core scope: first shot of the morning through both towers down.
EPOCHS: list[Epoch] = [
    Epoch(
        id="E0",
        name="Both towers intact",
        t_start=hms(0, 0, 0),
        t_end=WTC1_IMPACT.t,
        description="Pre-impact. Sparse same-morning footage; historical photos and the "
        "geo prior supply static geometry.",
    ),
    Epoch(
        id="E1",
        name="WTC1 burning, WTC2 intact",
        t_start=WTC1_IMPACT.t,
        t_end=WTC2_IMPACT.t,
    ),
    Epoch(
        id="E2",
        name="Both towers burning",
        t_start=WTC2_IMPACT.t,
        t_end=WTC2_COLLAPSE.t,
        description="Densest footage coverage of the day.",
    ),
    Epoch(
        id="E3",
        name="WTC2 collapsed, WTC1 burning",
        t_start=WTC2_COLLAPSE.t,
        t_end=WTC1_COLLAPSE.t,
    ),
    Epoch(
        id="E4",
        name="Both collapsed, dust cloud",
        t_start=WTC1_COLLAPSE.t,
        t_end=hms(12, 0, 0),
    ),
    Epoch(
        id="E5",
        name="Rubble pile / aftermath",
        t_start=hms(12, 0, 0),
        t_end=hms(48, 0, 0),
        description="Extends into 9/12+ for aerial imagery and LiDAR of the debris field.",
    ),
]
EPOCHS_BY_ID: dict[str, Epoch] = {e.id: e for e in EPOCHS}


class DynamicWindow(BaseModel):
    """Short, densely-covered windows that are candidates for true 4D splats."""

    id: str
    name: str
    t_start: float
    t_end: float


DYNAMIC_WINDOWS: list[DynamicWindow] = [
    DynamicWindow(
        id="D_impact2", name="Second impact", t_start=WTC2_IMPACT.t - 10, t_end=WTC2_IMPACT.t + 40
    ),
    DynamicWindow(
        id="D_collapse2",
        name="WTC2 collapse",
        t_start=WTC2_COLLAPSE.t - 10,
        t_end=WTC2_COLLAPSE.t + 60,
    ),
    DynamicWindow(
        id="D_collapse1",
        name="WTC1 collapse",
        t_start=WTC1_COLLAPSE.t - 10,
        t_end=WTC1_COLLAPSE.t + 60,
    ),
]


def epoch_at(t: float) -> Epoch | None:
    for ep in EPOCHS:
        if ep.contains(t):
            return ep
    return None
