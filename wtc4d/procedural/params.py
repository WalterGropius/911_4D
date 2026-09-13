"""Typed access to ``params.yaml``: every physical/artistic number the
procedural model uses, outside the event *times* that live in
``wtc4d.timeline``. See ``params.yaml`` for sources.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel


class WindLevel(BaseModel):
    from_deg: float
    speed_kt: float


class WindParams(BaseModel):
    surface: WindLevel
    aloft: WindLevel
    reference_height_m: float


class WTC1Params(BaseModel):
    collapse_duration_s: float
    antenna_drop_lead_s: float


class WTC2Params(BaseModel):
    collapse_duration_s: float
    tilt_phase_s: float
    tilt_max_deg: float
    tilt_azimuth_deg: float


class FireParams(BaseModel):
    spread_time_s: float


class RubbleParams(BaseModel):
    height_m: float


class RemnantParams(BaseModel):
    wtc1_height_m: float


class TowerParams(BaseModel):
    wtc1: WTC1Params
    wtc2: WTC2Params
    fire: FireParams
    rubble: RubbleParams
    remnant: RemnantParams


class DustParams(BaseModel):
    front_rate_m_s2_sqrt: float
    max_radius_m: float
    canyon_gain: float
    thin_start_s: float
    thin_tau_s: float


class SmokeParams(BaseModel):
    emit_interval_s: float
    max_age_s: float
    rise_rate_m_s: float
    level_height_m: float
    base_scale_m: float
    growth_per_sqrt_s: float
    opacity0: float
    decay_per_s: float
    color_fade_age_s: float


class WTC7Params(BaseModel):
    footprint_m: float
    collapse_duration_s: float
    damage_start_offset_s: float


class ThreeWTCParams(BaseModel):
    center_lat: float
    center_lon: float
    height_m: float
    footprint_m: float
    crushed_start_offset_s: float
    crushed_duration_s: float


class AuxBuildingsParams(BaseModel):
    wtc7: WTC7Params
    three_wtc: ThreeWTCParams


class GroundParams(BaseModel):
    half_extent_m: float
    grid_step_m: float


class SkylineParams(BaseModel):
    default_footprint_m: float


class Params(BaseModel):
    wind: WindParams
    towers: TowerParams
    dust: DustParams
    smoke: SmokeParams
    aux_buildings: AuxBuildingsParams
    ground: GroundParams
    skyline: SkylineParams
    seed: int = 0


_DEFAULT_PATH = Path(__file__).with_name("params.yaml")


def load_params(path: Path | str | None = None) -> Params:
    """Load and validate a params file (defaults to the packaged ``params.yaml``)."""
    p = Path(path) if path is not None else _DEFAULT_PATH
    with open(p) as fh:
        data = yaml.safe_load(fh)
    return Params.model_validate(data)


DEFAULT_PARAMS: Params = load_params()
