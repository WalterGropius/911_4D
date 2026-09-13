"""Smoke plume and post-collapse dust cloud, as gaussian blobs.

Both are represented as clouds of gaussian blobs strung along a rising,
wind-advected centreline (the smoke plume) or expanding radially from the
collapse point (the dust cloud). Wind and collapse timings are documented
in ``params.yaml``; the spreading/decay laws themselves are simple,
clearly-labelled kinematic approximations (Gaussian growth, exponential and
diffusive decay), not a fluid simulation.

All arrays below are plain numpy, matching the fields of
``wtc4d.procedural.gaussians.GaussianCloud`` (xyz, scale, quat, rgb,
opacity), so callers can feed them straight into ``GaussianCloud._cloud``.
"""

from __future__ import annotations

import math

import numpy as np

from wtc4d import world
from wtc4d.procedural.params import DEFAULT_PARAMS, Params
from wtc4d.procedural.towers import _COLLAPSE, _IMPACT, _damage_zone

_KT_TO_MS = 0.514444
_IDENTITY_QUAT = (1.0, 0.0, 0.0, 0.0)

Blobs = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
"""(xyz (N,3), scale (N,3), quat (N,4), rgb (N,3), opacity (N,)), float32."""


def _empty() -> Blobs:
    z3 = np.zeros((0, 3), dtype=np.float32)
    return (
        z3,
        z3.copy(),
        np.zeros((0, 4), dtype=np.float32),
        z3.copy(),
        np.zeros((0,), dtype=np.float32),
    )


def wind_vector(z_m: float, params: Params = DEFAULT_PARAMS) -> np.ndarray:
    """Horizontal wind velocity (m/s, world ENU x/y) at height ``z_m`` above
    plaza level, linearly interpolated between the surface and 'aloft'
    reference values in ``params.yaml``."""
    w = params.wind
    frac = min(1.0, max(0.0, z_m / w.reference_height_m))
    from_deg = w.surface.from_deg + frac * (w.aloft.from_deg - w.surface.from_deg)
    speed_kt = w.surface.speed_kt + frac * (w.aloft.speed_kt - w.surface.speed_kt)
    # Meteorological bearing is where the wind blows FROM; the plume moves
    # in the opposite direction.
    to_rad = math.radians(from_deg + 180.0)
    vx = math.sin(to_rad)
    vy = math.cos(to_rad)
    return np.array([vx, vy, 0.0]) * speed_kt * _KT_TO_MS


def _tower_source(tower_id: str) -> np.ndarray:
    spec = world.WTC1 if tower_id == "WTC1" else world.WTC2
    dz = _damage_zone(tower_id)
    floor_h = spec.roof_height_m / spec.floors
    z = 0.5 * (dz.floor_lo + dz.floor_hi) * floor_h
    c = spec.enu_center()
    return np.array([c[0], c[1], z])


def plume_blobs(tower_id: str, t: float, params: Params = DEFAULT_PARAMS) -> Blobs:
    """One tower's smoke plume at time ``t``: emission from the impact
    floors, buoyant rise then levelling, wind advection, Gaussian spreading
    and colour fade from dark (fuel-rich) to lighter grey with age."""
    impact_t = _IMPACT[tower_id].t
    stop_emit_t = _COLLAPSE[tower_id].t  # the source vanishes once the tower falls
    if t <= impact_t:
        return _empty()

    sp = params.smoke
    source = _tower_source(tower_id)
    last_emit = min(t, stop_emit_t)
    n = int((last_emit - impact_t) // sp.emit_interval_s) + 1
    emit_times = impact_t + np.arange(n) * sp.emit_interval_s
    emit_times = emit_times[emit_times <= last_emit]
    ages = t - emit_times
    ages = ages[ages <= sp.max_age_s]
    if ages.size == 0:
        return _empty()

    z = np.minimum(source[2] + sp.rise_rate_m_s * ages, sp.level_height_m)
    xy = np.tile(source[:2], (ages.size, 1))
    # Crude integral of a height-varying wind: use the current height's
    # speed as a constant-velocity approximation over the blob's age. Good
    # enough for a kinematic sketch; not a trajectory integration.
    for i, age in enumerate(ages):
        xy[i] += wind_vector(float(z[i]), params)[:2] * age
    xyz = np.column_stack([xy, z]).astype(np.float32)

    scale_iso = (sp.base_scale_m + sp.growth_per_sqrt_s * np.sqrt(ages)).astype(np.float32)
    scale = np.tile(scale_iso[:, None], (1, 3))

    opacity = np.clip(sp.opacity0 * np.exp(-sp.decay_per_s * ages), 0.0, 1.0).astype(np.float32)

    fade = np.clip(ages / sp.color_fade_age_s, 0.0, 1.0)
    dark = np.array([0.16, 0.16, 0.17])
    light = np.array([0.72, 0.72, 0.71])
    rgb = (dark[None, :] + (light - dark)[None, :] * fade[:, None]).astype(np.float32)

    quat = np.tile(np.array(_IDENTITY_QUAT, dtype=np.float32), (ages.size, 1))
    return xyz, scale, quat, rgb, opacity


def dust_blobs(
    tower_id: str,
    t: float,
    params: Params = DEFAULT_PARAMS,
    n_radii: int = 10,
    n_angles: int = 16,
) -> Blobs:
    """The ground-hugging dust cloud expanding from one tower's collapse
    point, anisotropically biased along the north-south / east-west street
    grid ("street canyons"), thinning after ``dust.thin_start_s``."""
    collapse_t = _COLLAPSE[tower_id].t
    if t <= collapse_t:
        return _empty()

    dp = params.dust
    age = t - collapse_t
    front = min(dp.max_radius_m, dp.front_rate_m_s2_sqrt * math.sqrt(age))
    if front <= 0.5:
        return _empty()

    spec = world.WTC1 if tower_id == "WTC1" else world.WTC2
    center = spec.enu_center()

    thinning = 1.0
    if age > dp.thin_start_s:
        thinning = math.exp(-(age - dp.thin_start_s) / dp.thin_tau_s)

    radii = np.linspace(front / n_radii, front, n_radii)
    angles = np.linspace(0.0, 2.0 * math.pi, n_angles, endpoint=False)
    rr, aa = np.meshgrid(radii, angles, indexing="ij")
    # Anisotropic radial gain: peaks along the +/-X and +/-Y axes, biasing
    # the front to travel further along the (approximate) street grid.
    canyon = 1.0 + dp.canyon_gain * np.abs(np.cos(2.0 * aa))
    r_eff = rr * canyon / canyon.mean()

    x = center[0] + r_eff.ravel() * np.cos(aa.ravel())
    y = center[1] + r_eff.ravel() * np.sin(aa.ravel())
    frac_out = np.clip(rr.ravel() / front, 0.0, 1.0)
    z = np.maximum(2.0, 25.0 * (1.0 - frac_out))  # taller near the source, low at the leading edge
    xyz = np.column_stack([x, y, z]).astype(np.float32)

    cell = max(2.0, front / n_radii)
    scale = np.column_stack(
        [np.full(xyz.shape[0], cell * 0.9), np.full(xyz.shape[0], cell * 0.9), z * 0.6]
    )
    scale = scale.astype(np.float32)

    opacity = np.clip(0.5 * (1.0 - 0.6 * frac_out) * thinning, 0.0, 1.0).astype(np.float32)
    rgb = np.tile(np.array([0.55, 0.53, 0.5], dtype=np.float32), (xyz.shape[0], 1))
    quat = np.tile(np.array(_IDENTITY_QUAT, dtype=np.float32), (xyz.shape[0], 1))
    return xyz, scale, quat, rgb, opacity
