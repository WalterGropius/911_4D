"""Tower (and auxiliary building) state machine.

Every timing anchor is taken from ``wtc4d.timeline`` (NIST NCSTAR 1 / 1A).
Collapse *kinematics* -- duration, WTC2's initial tilt -- are approximate,
drawn from published frame-by-frame video analyses since NIST does not
publish a single canonical collapse-duration number; each such value is
called out in ``params.yaml``. This module is a kinematic sketch: it gives
continuous, monotonic fractions for damage/fire/collapse so that
``wtc4d.procedural.gaussians`` can sample a coherent scene at any time, not
a structural or fluid simulation.

Phases, in order: ``intact`` -> ``burning`` (post-impact, pre-collapse) ->
``collapsing`` -> ``gone``.
"""

from __future__ import annotations

from dataclasses import dataclass

from wtc4d import timeline, world
from wtc4d.procedural.params import DEFAULT_PARAMS, Params

TOWER_IDS: tuple[str, ...] = ("WTC1", "WTC2")

_IMPACT: dict[str, timeline.Event] = {"WTC1": timeline.WTC1_IMPACT, "WTC2": timeline.WTC2_IMPACT}
_COLLAPSE: dict[str, timeline.Event] = {
    "WTC1": timeline.WTC1_COLLAPSE,
    "WTC2": timeline.WTC2_COLLAPSE,
}


def _smoothstep(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


@dataclass(frozen=True)
class DamageZone:
    """The impact gash: a rectangular hole on one face, in floor/face-fraction
    coordinates (see ``wtc4d.procedural.gaussians._face_points``)."""

    face: str  # "north" | "south" | "east" | "west"
    floor_lo: int
    floor_hi: int
    center_frac: float  # 0..1 across the face width
    width_frac: float


def _damage_zone(tower_id: str) -> DamageZone:
    if tower_id == "WTC1":
        # AA11 struck the north face, floors 93-99 (NIST NCSTAR 1), roughly centred.
        return DamageZone(face="north", floor_lo=93, floor_hi=99, center_frac=0.5, width_frac=0.6)
    # UA175 struck the south face, floors 77-85 (NIST NCSTAR 1), entering
    # near the south-east corner of the tower.
    return DamageZone(face="south", floor_lo=77, floor_hi=85, center_frac=0.82, width_frac=0.55)


@dataclass(frozen=True)
class TowerState:
    id: str
    t: float
    phase: str  # "intact" | "burning" | "collapsing" | "gone"
    fire_extent: float  # 0..1, growth of the visible fire zone since impact
    collapse_progress: float  # 0..1 within the collapse animation
    tilt_deg: float  # WTC2 only: initial top-block tilt
    tilt_azimuth_deg: float  # compass bearing the top block tilts toward
    top_height_m: float  # current highest point of the remaining intact structure
    antenna_present: bool  # WTC1 only
    damage: DamageZone | None
    rubble_height_m: float  # height of the rubble mound once "gone"
    remnant_present: bool  # WTC1 only: the standing north-face facade remnant
    remnant_height_m: float


def tower_state(tower_id: str, t: float, params: Params = DEFAULT_PARAMS) -> TowerState:
    """The state of one tower (``"WTC1"`` | ``"WTC2"``) at project time ``t``."""
    spec = world.WTC1 if tower_id == "WTC1" else world.WTC2
    impact_t = _IMPACT[tower_id].t
    collapse_t = _COLLAPSE[tower_id].t

    if t < impact_t:
        return TowerState(
            id=tower_id,
            t=t,
            phase="intact",
            fire_extent=0.0,
            collapse_progress=0.0,
            tilt_deg=0.0,
            tilt_azimuth_deg=0.0,
            top_height_m=spec.roof_height_m,
            antenna_present=True,
            damage=None,
            rubble_height_m=0.0,
            remnant_present=False,
            remnant_height_m=0.0,
        )

    fire_extent = _smoothstep((t - impact_t) / params.towers.fire.spread_time_s)

    if t < collapse_t:
        return TowerState(
            id=tower_id,
            t=t,
            phase="burning",
            fire_extent=fire_extent,
            collapse_progress=0.0,
            tilt_deg=0.0,
            tilt_azimuth_deg=0.0,
            top_height_m=spec.roof_height_m,
            antenna_present=True,
            damage=_damage_zone(tower_id),
            rubble_height_m=0.0,
            remnant_present=False,
            remnant_height_m=0.0,
        )

    age = t - collapse_t
    if tower_id == "WTC1":
        tp1 = params.towers.wtc1
        duration = tp1.collapse_duration_s
        antenna_present = age < tp1.antenna_drop_lead_s
        tilt_deg = 0.0
        tilt_azimuth_deg = 0.0
    else:
        tp2 = params.towers.wtc2
        duration = tp2.collapse_duration_s
        antenna_present = False
        tilt_deg = tp2.tilt_max_deg * _smoothstep(age / tp2.tilt_phase_s)
        tilt_azimuth_deg = tp2.tilt_azimuth_deg

    progress = _smoothstep(age / duration)

    if progress >= 1.0:
        is_wtc1 = tower_id == "WTC1"
        return TowerState(
            id=tower_id,
            t=t,
            phase="gone",
            fire_extent=fire_extent,
            collapse_progress=1.0,
            tilt_deg=0.0,
            tilt_azimuth_deg=0.0,
            top_height_m=0.0,
            antenna_present=False,
            damage=_damage_zone(tower_id),
            rubble_height_m=params.towers.rubble.height_m,
            remnant_present=is_wtc1,
            remnant_height_m=params.towers.remnant.wtc1_height_m if is_wtc1 else 0.0,
        )

    top_height_m = spec.roof_height_m * (1.0 - progress) + params.towers.rubble.height_m * progress
    return TowerState(
        id=tower_id,
        t=t,
        phase="collapsing",
        fire_extent=fire_extent,
        collapse_progress=progress,
        tilt_deg=tilt_deg,
        tilt_azimuth_deg=tilt_azimuth_deg,
        top_height_m=top_height_m,
        antenna_present=antenna_present,
        damage=_damage_zone(tower_id),
        rubble_height_m=params.towers.rubble.height_m,
        remnant_present=False,
        remnant_height_m=0.0,
    )


def all_tower_states(t: float, params: Params = DEFAULT_PARAMS) -> dict[str, TowerState]:
    return {tid: tower_state(tid, t, params) for tid in TOWER_IDS}


@dataclass(frozen=True)
class AuxBuildingState:
    """A coarser state for background structures the ``geo`` workstream will
    eventually model properly: 7 WTC and 3 WTC (Marriott)."""

    id: str
    phase: str  # "intact" | "damaged" | "collapsing" | "gone"
    progress: float  # 0..1 within the current phase, where meaningful


def wtc7_state(t: float, params: Params = DEFAULT_PARAMS) -> AuxBuildingState:
    bp = params.aux_buildings.wtc7
    damage_t = _COLLAPSE["WTC1"].t + bp.damage_start_offset_s
    collapse_t = timeline.WTC7_COLLAPSE.t

    if t < damage_t:
        return AuxBuildingState(id="WTC7", phase="intact", progress=0.0)
    if t < collapse_t:
        span = max(1.0, collapse_t - damage_t)
        return AuxBuildingState(
            id="WTC7", phase="damaged", progress=_smoothstep((t - damage_t) / span)
        )
    progress = _smoothstep((t - collapse_t) / bp.collapse_duration_s)
    if progress >= 1.0:
        return AuxBuildingState(id="WTC7", phase="gone", progress=1.0)
    return AuxBuildingState(id="WTC7", phase="collapsing", progress=progress)


def three_wtc_state(t: float, params: Params = DEFAULT_PARAMS) -> AuxBuildingState:
    bp = params.aux_buildings.three_wtc
    start = _COLLAPSE["WTC2"].t + bp.crushed_start_offset_s
    if t < start:
        return AuxBuildingState(id="3WTC", phase="intact", progress=0.0)
    progress = _smoothstep((t - start) / bp.crushed_duration_s)
    phase = "gone" if progress >= 1.0 else "collapsing"
    return AuxBuildingState(id="3WTC", phase=phase, progress=progress)
