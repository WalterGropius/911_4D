"""Sun position and shadow-based timing.

Implements the NOAA solar position algorithm (the one behind NOAA's Solar
Calculator spreadsheet, Global Monitoring Laboratory), which is accurate to
better than 0.02 deg for dates near 2001 -- far below the measurement error of
any shadow read off 480i video.

Why this matters and what it can actually deliver
-------------------------------------------------
The sun traverses the sky at ~15 deg/hour = **0.25 deg/minute**.  Whatever cue
you use (shadow azimuth on the ground, sun elevation from a shadow-length
ratio), the time uncertainty is

    sigma_t  =  sigma_angle / |d(angle)/dt|

so a **30 s** estimate requires the sun direction to be known to about
**0.12 deg**, and a 1 deg measurement is worth roughly 4 minutes.  The rate
varies through the morning -- :func:`azimuth_rate_deg_per_s` reports it for the
time in question and the inversion propagates it, so the sigma written into a
:class:`~wtc4d.schema.time.TimeEstimate` is honest rather than aspirational.
Solar timing is therefore the *weak* cue of the fusion: it disambiguates
"morning vs afternoon" and catches gross errors, and only occasionally reaches
the tens of seconds.

Angle conventions
-----------------
* Azimuth: degrees clockwise from true north (N=0, E=90, S=180, W=270).
* Elevation: degrees above the astronomical horizon; ``apparent`` includes
  atmospheric refraction, which is what a camera sees.
* A **shadow** on level ground points *away* from the sun:
  ``shadow_azimuth = (sun_azimuth + 180) mod 360``.
* Magnetic declination is never used; everything is true north.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel, Field

from wtc4d.schema.time import TimeEstimate, TimeMethod
from wtc4d.timeline import hms, to_utc

# World Trade Center site, close enough for solar purposes (a 1 km position
# error moves the sun by <0.01 deg).  Matches wtc4d.world.WORLD_ORIGIN.
WTC_LAT = 40.71120
WTC_LON = -74.01320

_DEG = math.pi / 180.0


@dataclass(frozen=True)
class SunPosition:
    """Sun direction and the derived quantities used for timing."""

    t: float  # project seconds
    azimuth_deg: float
    elevation_deg: float  # true (geometric)
    apparent_elevation_deg: float  # refraction-corrected, i.e. what is seen
    declination_deg: float
    equation_of_time_min: float
    hour_angle_deg: float

    @property
    def zenith_deg(self) -> float:
        return 90.0 - self.elevation_deg

    @property
    def shadow_azimuth_deg(self) -> float:
        """Direction a shadow points on level ground, deg clockwise from north."""
        return (self.azimuth_deg + 180.0) % 360.0

    def shadow_length_ratio(self) -> float:
        """shadow length / object height on level ground (apparent elevation)."""
        e = self.apparent_elevation_deg
        if e <= 0.05:
            return float("inf")
        return 1.0 / math.tan(e * _DEG)


def julian_day(t: float) -> float:
    """Project seconds -> Julian Day (UT), including the fractional day."""
    dt = to_utc(t)
    y, m = dt.year, dt.month
    day = (
        dt.day + (dt.hour + (dt.minute + (dt.second + dt.microsecond * 1e-6) / 60.0) / 60.0) / 24.0
    )
    if m <= 2:
        y -= 1
        m += 12
    a = y // 100
    b = 2 - a + a // 4
    return math.floor(365.25 * (y + 4716)) + math.floor(30.6001 * (m + 1)) + day + b - 1524.5


def _refraction_deg(elevation_deg: float) -> float:
    """Atmospheric refraction (NOAA's piecewise fit), degrees to be *added*."""
    te = elevation_deg
    if te > 85.0:
        return 0.0
    tan_te = math.tan(te * _DEG)
    if te > 5.0:
        r = 58.1 / tan_te - 0.07 / tan_te**3 + 0.000086 / tan_te**5
    elif te > -0.575:
        r = 1735.0 + te * (-518.2 + te * (103.4 + te * (-12.79 + te * 0.711)))
    else:
        r = -20.772 / tan_te
    return r / 3600.0


def sun_position(t: float, lat: float = WTC_LAT, lon: float = WTC_LON) -> SunPosition:
    """Sun position at project time ``t`` for a geodetic location.

    ``lat``/``lon`` in degrees, east-positive longitude.
    """
    jd = julian_day(t)
    jc = (jd - 2451545.0) / 36525.0  # Julian century

    geom_mean_long = (280.46646 + jc * (36000.76983 + jc * 0.0003032)) % 360.0
    geom_mean_anom = 357.52911 + jc * (35999.05029 - 0.0001537 * jc)
    eccent = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)
    sun_eq_ctr = (
        math.sin(geom_mean_anom * _DEG) * (1.914602 - jc * (0.004817 + 0.000014 * jc))
        + math.sin(2 * geom_mean_anom * _DEG) * (0.019993 - 0.000101 * jc)
        + math.sin(3 * geom_mean_anom * _DEG) * 0.000289
    )
    true_long = geom_mean_long + sun_eq_ctr
    app_long = true_long - 0.00569 - 0.00478 * math.sin((125.04 - 1934.136 * jc) * _DEG)

    mean_obliq = (
        23.0 + (26.0 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60.0) / 60.0
    )
    obliq_corr = mean_obliq + 0.00256 * math.cos((125.04 - 1934.136 * jc) * _DEG)

    declination = math.degrees(math.asin(math.sin(obliq_corr * _DEG) * math.sin(app_long * _DEG)))

    var_y = math.tan(obliq_corr / 2.0 * _DEG) ** 2
    eq_time = 4.0 * math.degrees(
        var_y * math.sin(2.0 * geom_mean_long * _DEG)
        - 2.0 * eccent * math.sin(geom_mean_anom * _DEG)
        + 4.0
        * eccent
        * var_y
        * math.sin(geom_mean_anom * _DEG)
        * math.cos(2.0 * geom_mean_long * _DEG)
        - 0.5 * var_y * var_y * math.sin(4.0 * geom_mean_long * _DEG)
        - 1.25 * eccent * eccent * math.sin(2.0 * geom_mean_anom * _DEG)
    )  # minutes

    utc = to_utc(t)
    minutes_utc = utc.hour * 60.0 + utc.minute + (utc.second + utc.microsecond * 1e-6) / 60.0
    true_solar_time = (minutes_utc + eq_time + 4.0 * lon) % 1440.0
    hour_angle = true_solar_time / 4.0 - 180.0
    if hour_angle < -180.0:
        hour_angle += 360.0

    lat_r = lat * _DEG
    dec_r = declination * _DEG
    ha_r = hour_angle * _DEG
    cos_zen = math.sin(lat_r) * math.sin(dec_r) + math.cos(lat_r) * math.cos(dec_r) * math.cos(ha_r)
    cos_zen = max(-1.0, min(1.0, cos_zen))
    zenith = math.degrees(math.acos(cos_zen))
    elevation = 90.0 - zenith

    sin_zen = math.sin(zenith * _DEG)
    if abs(sin_zen) < 1e-9 or abs(math.cos(lat_r)) < 1e-9:
        azimuth = 180.0 if lat > 0 else 0.0
    else:
        arg = (math.sin(lat_r) * cos_zen - math.sin(dec_r)) / (math.cos(lat_r) * sin_zen)
        arg = max(-1.0, min(1.0, arg))
        acos_deg = math.degrees(math.acos(arg))
        azimuth = (acos_deg + 180.0) % 360.0 if hour_angle > 0 else (540.0 - acos_deg) % 360.0

    return SunPosition(
        t=t,
        azimuth_deg=azimuth,
        elevation_deg=elevation,
        apparent_elevation_deg=elevation + _refraction_deg(elevation),
        declination_deg=declination,
        equation_of_time_min=eq_time,
        hour_angle_deg=hour_angle,
    )


def solar_noon(day_offset_s: float = 0.0, lat: float = WTC_LAT, lon: float = WTC_LON) -> float:
    """Project seconds of solar noon (iterating once on the equation of time)."""
    t = day_offset_s + hms(12, 0, 0)
    for _ in range(3):
        eq = sun_position(t, lat, lon).equation_of_time_min
        # local standard time meridian for EDT (UTC-4) is -60 deg
        t = day_offset_s + (720.0 - 4.0 * lon - eq + (-4.0) * 60.0) * 60.0
    return t


def _hour_angle_for_zenith(zenith_deg: float, lat: float, declination_deg: float) -> float | None:
    lat_r, dec_r = lat * _DEG, declination_deg * _DEG
    arg = math.cos(zenith_deg * _DEG) / (math.cos(lat_r) * math.cos(dec_r)) - math.tan(
        lat_r
    ) * math.tan(dec_r)
    if not -1.0 <= arg <= 1.0:
        return None  # sun never reaches this zenith on this day
    return math.degrees(math.acos(arg))


def sunrise_sunset(
    day_offset_s: float = 0.0, lat: float = WTC_LAT, lon: float = WTC_LON
) -> tuple[float | None, float | None]:
    """Project seconds of sunrise and sunset (90.833 deg zenith, upper limb)."""
    noon = solar_noon(day_offset_s, lat, lon)
    dec = sun_position(noon, lat, lon).declination_deg
    ha = _hour_angle_for_zenith(90.833, lat, dec)
    if ha is None:
        return (None, None)
    half = ha * 4.0 * 60.0  # deg -> minutes -> seconds
    return (noon - half, noon + half)


def azimuth_rate_deg_per_s(t: float, lat: float = WTC_LAT, lon: float = WTC_LON) -> float:
    """d(azimuth)/dt in deg/s (central difference over +-30 s)."""
    a = sun_position(t - 30.0, lat, lon).azimuth_deg
    b = sun_position(t + 30.0, lat, lon).azimuth_deg
    d = (b - a + 540.0) % 360.0 - 180.0
    return d / 60.0


def elevation_rate_deg_per_s(t: float, lat: float = WTC_LAT, lon: float = WTC_LON) -> float:
    """d(elevation)/dt in deg/s (central difference over +-30 s)."""
    a = sun_position(t - 30.0, lat, lon).apparent_elevation_deg
    b = sun_position(t + 30.0, lat, lon).apparent_elevation_deg
    return (b - a) / 60.0


# --- annotation format -------------------------------------------------------


class ShadowAnnotation(BaseModel):
    """A hand-made shadow measurement, one JSON object per annotation.

    Two supported forms (see ``data/time/annotations/`` for examples):

    ``kind="measured_azimuth"``
        ``world_azimuth_deg`` is already the true-north bearing of the shadow
        (someone read it off a map, or a registered camera was used to
        un-project it).  ``p0``/``p1`` are then optional provenance.

    ``kind="ground_shadow_edge"``
        ``p0``/``p1`` are image points (pixels) on a shadow edge that lies on
        a horizontal ground plane, ordered *from* the casting object *towards*
        the shadow tip.  Turning those into a world bearing needs a camera
        pose, which the camreg workstream supplies; pass it to
        :func:`shadow_azimuth_from_annotation` as ``ground_point_of`` -- a
        callable mapping an image point ``(u, v)`` to a world ground point
        ``(east_m, north_m)`` in the ENU frame of :mod:`wtc4d.world` (that is
        exactly a ray/ground-plane intersection with a registered camera).
        Without it the annotation cannot be inverted and is skipped.

    ``sigma_deg`` must be a *real* estimate of the bearing error, including
    how well the ground plane and the north direction are known.  Below about
    0.12 deg the result beats 30 s; at 1 deg it is worth ~4 minutes.
    """

    shot_id: str
    frame_idx: int = 0
    kind: str = Field(default="measured_azimuth", pattern="^(measured_azimuth|ground_shadow_edge)$")
    world_azimuth_deg: float | None = None
    p0: tuple[float, float] | None = None
    p1: tuple[float, float] | None = None
    sigma_deg: float = Field(default=2.0, gt=0.0)
    lat: float = WTC_LAT
    lon: float = WTC_LON
    t_window: tuple[float, float] | None = Field(
        default=None,
        description="project-second search window; defaults to civil daylight on 2001-09-11",
    )
    notes: str = ""


GroundPointFn = Callable[[tuple[float, float]], tuple[float, float] | None]
"""Image point (u, v) in pixels -> world ground point (east_m, north_m), ENU."""


def bearing_enu(p0: tuple[float, float], p1: tuple[float, float]) -> float:
    """True-north bearing (deg) of the ENU segment ``p0 -> p1``.

    ENU is east-north-up, so the bearing clockwise from north is
    ``atan2(dEast, dNorth)``.
    """
    d_e = p1[0] - p0[0]
    d_n = p1[1] - p0[1]
    return math.degrees(math.atan2(d_e, d_n)) % 360.0


def shadow_azimuth_from_annotation(
    ann: ShadowAnnotation,
    ground_point_of: GroundPointFn | None = None,
) -> float | None:
    """True-north bearing (deg) of the shadow described by ``ann``, or None."""
    if ann.kind == "measured_azimuth":
        return None if ann.world_azimuth_deg is None else ann.world_azimuth_deg % 360.0
    if ground_point_of is None or ann.p0 is None or ann.p1 is None:
        return None
    a = ground_point_of(tuple(ann.p0))
    b = ground_point_of(tuple(ann.p1))
    if a is None or b is None:
        return None
    return bearing_enu(a, b)


# --- inversion ---------------------------------------------------------------

DAYLIGHT_WINDOW = (hms(6, 30, 0), hms(19, 15, 0))
"""Default search window on 2001-09-11: between sunrise and sunset in NYC."""


def time_from_sun_azimuth(
    sun_azimuth_deg: float,
    *,
    sigma_deg: float = 2.0,
    t_window: tuple[float, float] = DAYLIGHT_WINDOW,
    lat: float = WTC_LAT,
    lon: float = WTC_LON,
    evidence: str = "",
) -> TimeEstimate | None:
    """Invert a *sun* azimuth to a time inside ``t_window``.

    The sun's azimuth is monotonic through the day at these latitudes, so a
    bisection on ``azimuth(t) - target`` (wrapped to [-180, 180]) is enough.
    The sigma is the angular sigma divided by the local azimuth rate.

    That wrap is itself discontinuous at a difference of +-180 deg -- i.e. at
    whatever moment the true azimuth passes through ``target + 180``, which
    has nothing to do with ``target`` itself.  A plain sign-change bisection
    cannot tell that jump apart from a genuine crossing, so a bisection whose
    only "root" is really this artifact (most simply hit when ``target`` is
    exactly opposite the window's own azimuth turning point, e.g. a window
    straddling solar noon and ``target`` near due north) would otherwise
    report a confident but wrong time.  The result is verified against the
    unwrapped azimuth before being trusted.
    """
    target = sun_azimuth_deg % 360.0
    lo, hi = t_window

    def f(t: float) -> float:
        return (sun_position(t, lat, lon).azimuth_deg - target + 540.0) % 360.0 - 180.0

    f_lo, f_hi = f(lo), f(hi)
    if f_lo == 0.0:
        root = lo
    elif f_hi == 0.0:
        root = hi
    elif f_lo * f_hi > 0:
        return None  # the sun never has this azimuth inside the window
    else:
        a, b = lo, hi
        for _ in range(80):
            m = 0.5 * (a + b)
            fm = f(m)
            if f(a) * fm <= 0:
                b = m
            else:
                a = m
        root = 0.5 * (a + b)

    # Reject a spurious wrap-discontinuity "root": the true azimuth at the
    # converged time must actually equal `target`, not merely have a wrapped
    # difference near a +-180 deg boundary.
    residual = abs((sun_position(root, lat, lon).azimuth_deg - target + 180.0) % 360.0 - 180.0)
    if residual > 1e-3:
        return None

    rate = abs(azimuth_rate_deg_per_s(root, lat, lon))
    if rate < 1e-6:
        return None
    sigma = sigma_deg / rate
    return TimeEstimate(
        t=root,
        sigma=sigma,
        method=TimeMethod.SOLAR_SHADOW,
        evidence=(
            evidence
            or f"sun azimuth {target:.2f} deg +-{sigma_deg:.2f} deg at "
            f"lat {lat:.5f} lon {lon:.5f}; d(az)/dt = {rate * 60:.3f} deg/min"
        ),
    )


def time_from_shadow_azimuth(
    shadow_azimuth_deg: float,
    *,
    sigma_deg: float = 2.0,
    t_window: tuple[float, float] = DAYLIGHT_WINDOW,
    lat: float = WTC_LAT,
    lon: float = WTC_LON,
) -> TimeEstimate | None:
    """Invert a *shadow* bearing (opposite the sun) to a time."""
    est = time_from_sun_azimuth(
        (shadow_azimuth_deg + 180.0) % 360.0,
        sigma_deg=sigma_deg,
        t_window=t_window,
        lat=lat,
        lon=lon,
        evidence=(
            f"ground shadow bearing {shadow_azimuth_deg % 360.0:.2f} deg "
            f"+-{sigma_deg:.2f} deg (sun at {(shadow_azimuth_deg + 180.0) % 360.0:.2f} deg)"
        ),
    )
    return est


def time_from_sun_elevation(
    elevation_deg: float,
    *,
    sigma_deg: float = 2.0,
    t_window: tuple[float, float] = DAYLIGHT_WINDOW,
    lat: float = WTC_LAT,
    lon: float = WTC_LON,
    morning: bool = True,
) -> TimeEstimate | None:
    """Invert an apparent sun elevation (e.g. from a shadow-length ratio).

    Elevation is *not* monotonic over a full day, so the branch is selected by
    ``morning`` (before solar noon) and the search window is clipped to it.
    """
    noon = solar_noon(0.0, lat, lon)
    lo, hi = t_window
    if morning:
        hi = min(hi, noon)
    else:
        lo = max(lo, noon)
    if hi <= lo:
        return None

    def f(t: float) -> float:
        return sun_position(t, lat, lon).apparent_elevation_deg - elevation_deg

    if f(lo) * f(hi) > 0:
        return None
    a, b = lo, hi
    for _ in range(80):
        m = 0.5 * (a + b)
        if f(a) * f(m) <= 0:
            b = m
        else:
            a = m
    root = 0.5 * (a + b)
    rate = abs(elevation_rate_deg_per_s(root, lat, lon))
    if rate < 1e-6:
        return None
    return TimeEstimate(
        t=root,
        sigma=sigma_deg / rate,
        method=TimeMethod.SOLAR_SHADOW,
        evidence=(
            f"apparent sun elevation {elevation_deg:.2f} deg +-{sigma_deg:.2f} deg "
            f"({'morning' if morning else 'afternoon'} branch); "
            f"d(elev)/dt = {rate * 60:.3f} deg/min"
        ),
    )


def estimate_from_annotation(
    ann: ShadowAnnotation,
    ground_point_of: GroundPointFn | None = None,
) -> TimeEstimate | None:
    """Turn a :class:`ShadowAnnotation` into a :class:`TimeEstimate`."""
    az = shadow_azimuth_from_annotation(ann, ground_point_of)
    if az is None:
        return None
    est = time_from_shadow_azimuth(
        az,
        sigma_deg=ann.sigma_deg,
        t_window=ann.t_window or DAYLIGHT_WINDOW,
        lat=ann.lat,
        lon=ann.lon,
    )
    if est is None:
        return None
    suffix = f" [{ann.shot_id} frame {ann.frame_idx}]"
    if ann.notes:
        suffix += f" {ann.notes}"
    return est.model_copy(update={"evidence": est.evidence + suffix})


# --- weak cue: smoke plume direction ----------------------------------------


class WindCue(BaseModel):
    """Surface wind on 2001-09-11, used as a *very* weak direction cue.

    The plume from the towers ran roughly with the surface wind, so the
    bearing of the plume in an image says something about the wind, not about
    the time -- the wind barely changed during the morning.  Its real use is
    the reverse: as a **consistency check** that a shot belongs to 2001-09-11
    at all, and to reject shadow annotations that mistook a plume shadow for a
    ground shadow.

    ``direction_from_deg`` is the meteorological convention: the direction the
    wind blows *from*, true-north bearing.

    The values below are the project's working assumption and are flagged as
    such: they come from the plume geometry visible in the footage (the smoke
    is carried south-southeast over Brooklyn all morning), not yet from a
    retrieved observation.  Replace them with the NOAA NCEI Integrated Surface
    Database hourly records for KNYC / KEWR / KLGA before quoting them --
    see ``data/time/README.md`` for the exact station ids and how to load them.
    """

    direction_from_deg: float = 340.0
    sigma_deg: float = 20.0
    speed_kt: float = 10.0
    sigma_speed_kt: float = 4.0
    source: str = "working assumption from plume geometry; verify against NOAA NCEI ISD"
    verified: bool = False


DEFAULT_WIND = WindCue()


def plume_consistency(plume_bearing_deg: float, wind: WindCue = DEFAULT_WIND) -> tuple[bool, float]:
    """Is a plume bearing consistent with the day's wind?

    ``plume_bearing_deg`` is the direction the plume travels *towards*.
    Returns ``(consistent, n_sigma)`` where the tolerance is 3 sigma of the
    wind direction widened by 15 deg for plume meander.
    """
    expected_to = (wind.direction_from_deg + 180.0) % 360.0
    diff = abs((plume_bearing_deg - expected_to + 540.0) % 360.0 - 180.0)
    sigma = math.hypot(wind.sigma_deg, 15.0)
    return diff <= 3.0 * sigma, diff / sigma


def epoch_sun_table(lat: float = WTC_LAT, lon: float = WTC_LON) -> list[dict[str, float | str]]:
    """Sun position at each epoch boundary -- rendered into the README."""
    from wtc4d.timeline import EPOCHS, fmt_local

    rows: list[dict[str, float | str]] = []
    seen: set[float] = set()
    for ep in EPOCHS:
        for label, t in ((f"{ep.id} start", ep.t_start), (f"{ep.id} end", ep.t_end)):
            if t in seen:
                continue
            seen.add(t)
            sp = sun_position(t, lat, lon)
            rows.append(
                {
                    "label": label,
                    "local": fmt_local(t),
                    "t": t,
                    "azimuth_deg": round(sp.azimuth_deg, 2),
                    "elevation_deg": round(sp.apparent_elevation_deg, 2),
                    "shadow_azimuth_deg": round(sp.shadow_azimuth_deg, 2),
                    "shadow_len_ratio": round(min(sp.shadow_length_ratio(), 99.0), 3),
                    "az_rate_deg_per_min": round(azimuth_rate_deg_per_s(t, lat, lon) * 60.0, 4),
                }
            )
    return rows


__all__ = [
    "DAYLIGHT_WINDOW",
    "DEFAULT_WIND",
    "GroundPointFn",
    "ShadowAnnotation",
    "SunPosition",
    "WTC_LAT",
    "WTC_LON",
    "WindCue",
    "azimuth_rate_deg_per_s",
    "bearing_enu",
    "elevation_rate_deg_per_s",
    "epoch_sun_table",
    "estimate_from_annotation",
    "julian_day",
    "plume_consistency",
    "shadow_azimuth_from_annotation",
    "solar_noon",
    "sun_position",
    "sunrise_sunset",
    "time_from_shadow_azimuth",
    "time_from_sun_azimuth",
    "time_from_sun_elevation",
]
