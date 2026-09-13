"""``GaussianCloud``: the sampled scene representation at one project time.

``sample(t, ...)`` combines a coarse ground plane, a placeholder skyline
(from ``wtc4d.world.LANDMARKS``), the two towers (facade, impact damage,
fire), background buildings (7 WTC, 3 WTC), smoke plumes and post-collapse
dust clouds into one cloud. Positions are world ENU metres
(``wtc4d.world``). Everything here is a kinematic sketch: a physically
*plausible* prior and initialisation for ``recon``, not a measurement --
see ``README.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wtc4d import world
from wtc4d.procedural import smoke, towers
from wtc4d.procedural.params import DEFAULT_PARAMS, Params
from wtc4d.schema.geometry import LatLonAlt

SH_C0 = 0.28209479177387814
"""Real spherical-harmonic DC-term normalisation used by the standard 3DGS
PLY convention: rgb = f_dc * SH_C0 + 0.5."""

LAYER_NAMES: tuple[str, ...] = (
    "ground",
    "skyline",
    "towers",
    "damage",
    "fire",
    "smoke",
    "dust",
    "rubble",
)
LAYER_INDEX: dict[str, int] = {name: i for i, name in enumerate(LAYER_NAMES)}

_IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


@dataclass
class GaussianCloud:
    """A set of 3D gaussians. All arrays share a leading dimension N.

    xyz      (N, 3) float32  world ENU metres
    scale    (N, 3) float32  linear (not log) std-dev along local axes, metres
    quat     (N, 4) float32  unit quaternion wxyz, local -> world rotation
    rgb      (N, 3) float32  linear colour in [0, 1]
    opacity  (N,)   float32  in [0, 1]
    layer    (N,)   int32    index into LAYER_NAMES
    """

    xyz: np.ndarray
    scale: np.ndarray
    quat: np.ndarray
    rgb: np.ndarray
    opacity: np.ndarray
    layer: np.ndarray

    def __len__(self) -> int:
        return int(self.xyz.shape[0])

    @staticmethod
    def empty() -> GaussianCloud:
        return GaussianCloud(
            xyz=np.zeros((0, 3), np.float32),
            scale=np.zeros((0, 3), np.float32),
            quat=np.zeros((0, 4), np.float32),
            rgb=np.zeros((0, 3), np.float32),
            opacity=np.zeros((0,), np.float32),
            layer=np.zeros((0,), np.int32),
        )

    def concat(self, *others: GaussianCloud) -> GaussianCloud:
        clouds = [c for c in (self, *others) if len(c) > 0]
        if not clouds:
            return GaussianCloud.empty()
        return GaussianCloud(
            xyz=np.concatenate([c.xyz for c in clouds]),
            scale=np.concatenate([c.scale for c in clouds]),
            quat=np.concatenate([c.quat for c in clouds]),
            rgb=np.concatenate([c.rgb for c in clouds]),
            opacity=np.concatenate([c.opacity for c in clouds]),
            layer=np.concatenate([c.layer for c in clouds]),
        )

    def layer_mask(self, name: str) -> np.ndarray:
        return self.layer == LAYER_INDEX[name]

    # -- I/O ----------------------------------------------------------------

    def save_ply(self, path: str | Path) -> None:
        _save_ply(self, Path(path))

    def save_npz(self, path: str | Path) -> None:
        _save_npz(self, Path(path))

    @staticmethod
    def load_ply(path: str | Path) -> GaussianCloud:
        return _load_ply(Path(path))

    @staticmethod
    def load_npz(path: str | Path) -> GaussianCloud:
        return _load_npz(Path(path))


def _cloud(
    xyz: np.ndarray,
    scale: np.ndarray,
    quat: np.ndarray,
    rgb: np.ndarray,
    opacity: np.ndarray,
    layer_name: str,
) -> GaussianCloud:
    n = xyz.shape[0]
    if n == 0:
        return GaussianCloud.empty()
    return GaussianCloud(
        xyz=np.asarray(xyz, dtype=np.float32).reshape(n, 3),
        scale=np.asarray(scale, dtype=np.float32).reshape(n, 3),
        quat=np.asarray(quat, dtype=np.float32).reshape(n, 4),
        rgb=np.clip(np.asarray(rgb, dtype=np.float32).reshape(n, 3), 0.0, 1.0),
        opacity=np.clip(np.asarray(opacity, dtype=np.float32).reshape(n), 0.0, 1.0),
        layer=np.full(n, LAYER_INDEX[layer_name], dtype=np.int32),
    )


# --- ground plane and placeholder skyline -----------------------------------


def _ground(params: Params) -> GaussianCloud:
    g = params.ground
    n = int(2 * g.half_extent_m / g.grid_step_m) + 1
    xs = np.linspace(-g.half_extent_m, g.half_extent_m, n)
    xx, yy = np.meshgrid(xs, xs)
    xyz = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])
    scale = np.tile([g.grid_step_m * 0.55, g.grid_step_m * 0.55, 0.15], (xyz.shape[0], 1))
    quat = np.tile(_IDENTITY_QUAT, (xyz.shape[0], 1))
    rgb = np.tile([0.47, 0.47, 0.49], (xyz.shape[0], 1))
    opacity = np.full(xyz.shape[0], 0.6)
    return _cloud(xyz, scale, quat, rgb, opacity, "ground")


def _skyline(params: Params) -> GaussianCloud:
    """Coarse placeholder boxes from ``wtc4d.world.LANDMARKS``. The ``geo``
    workstream owns the real Lower Manhattan model; if it's importable and
    exposes a richer registry we prefer that, else we fall back to these
    boxes so the scene is never bare."""
    try:
        from wtc4d import geo  # noqa: F401  (lazy: geo workstream, may not exist yet)

        buildings = getattr(geo, "BUILDINGS_2001", None)
        if buildings:
            return _skyline_from_geo(buildings)
    except ImportError:
        pass
    return _skyline_from_landmarks(params)


def _skyline_from_geo(buildings) -> GaussianCloud:  # pragma: no cover - exercised once geo exists
    parts = []
    for b in buildings:
        parts.append(
            _box_column(np.asarray(b.center_enu, dtype=np.float64), b.footprint_m, b.height_m)
        )
    xyz, scale, quat, rgb, opacity = _concat_parts(parts)
    return _cloud(xyz, scale, quat, rgb, opacity, "skyline")


def _box_column(
    center_xy: np.ndarray, footprint_m: float, height_m: float, levels_step_m: float = 20.0
):
    n_levels = max(1, int(round(height_m / levels_step_m)))
    level_h = height_m / n_levels
    zs = np.linspace(level_h * 0.5, height_m - level_h * 0.5, n_levels)
    xyz = np.column_stack([np.full(n_levels, center_xy[0]), np.full(n_levels, center_xy[1]), zs])
    scale = np.tile([footprint_m * 0.5, footprint_m * 0.5, level_h * 0.55], (n_levels, 1))
    quat = np.tile(_IDENTITY_QUAT, (n_levels, 1))
    rgb = np.tile([0.5, 0.52, 0.55], (n_levels, 1))
    opacity = np.full(n_levels, 0.85)
    return xyz, scale, quat, rgb, opacity


def _skyline_from_landmarks(params: Params) -> GaussianCloud:
    fp = params.skyline.default_footprint_m
    parts = []
    for lm in world.LANDMARKS:
        if not lm.existed_on_2001_09_11:
            continue
        if lm.kind in ("spire", "statue", "bridge_tower"):
            continue  # thin features: a box placeholder would mislead more than help
        if lm.id.startswith("wtc1_") or lm.id.startswith("wtc2_"):
            continue  # the towers themselves are modelled separately
        parts.append(_box_column(lm.enu()[:2], fp, lm.height_m))
    if not parts:
        return GaussianCloud.empty()
    xyz, scale, quat, rgb, opacity = _concat_parts(parts)
    return _cloud(xyz, scale, quat, rgb, opacity, "skyline")


def _concat_parts(parts):
    xyz = np.concatenate([p[0] for p in parts])
    scale = np.concatenate([p[1] for p in parts])
    quat = np.concatenate([p[2] for p in parts])
    rgb = np.concatenate([p[3] for p in parts])
    opacity = np.concatenate([p[4] for p in parts])
    return xyz, scale, quat, rgb, opacity


# --- tower facades -----------------------------------------------------------

_FACE_NORMALS_LOCAL = {
    "north": np.array([0.0, 1.0]),
    "south": np.array([0.0, -1.0]),
    "east": np.array([1.0, 0.0]),
    "west": np.array([-1.0, 0.0]),
}
_FACE_TANGENTS_LOCAL = {  # direction of increasing face-fraction (0 -> 1)
    "north": np.array([1.0, 0.0]),  # west -> east
    "south": np.array([1.0, 0.0]),  # west -> east
    "east": np.array([0.0, 1.0]),  # south -> north
    "west": np.array([0.0, 1.0]),  # south -> north
}

_FACADE_RGB = np.array([0.72, 0.74, 0.78])  # approx aluminium-silver curtain wall
_FIRE_RGB = np.array([0.85, 0.35, 0.05])


def _face_points(spec, face: str, z_lo: float, z_hi: float, u_step: float, z_step: float):
    """Grid of points on one vertical face of ``spec``'s box, plus each
    point's fraction (0..1) across the face width. Handles the box's
    ``rotation_deg`` so this stays correct if ``geo`` refines it."""
    if z_hi <= z_lo:
        return np.zeros((0, 3)), np.zeros((0,))
    c = spec.enu_center()
    h = spec.footprint_m / 2.0
    r = math.radians(spec.rotation_deg)
    rot = np.array([[math.cos(r), -math.sin(r)], [math.sin(r), math.cos(r)]])
    normal = rot @ _FACE_NORMALS_LOCAL[face]
    tangent = rot @ _FACE_TANGENTS_LOCAL[face]
    face_center_xy = c[:2] + normal * h

    n_u = max(2, int(spec.footprint_m / u_step) + 1)
    n_z = max(2, int((z_hi - z_lo) / z_step) + 1)
    u = np.linspace(-h, h, n_u)
    zs = np.linspace(z_lo, z_hi, n_z)
    uu, zz = np.meshgrid(u, zs)
    frac = (uu + h) / (2.0 * h)
    xy = face_center_xy[None, None, :] + uu[..., None] * tangent[None, None, :]
    xyz = np.concatenate([xy, zz[..., None]], axis=-1).reshape(-1, 3)
    return xyz, frac.ravel()


def _tower_parts(tower_id: str, state: towers.TowerState, params: Params):
    """Returns (facade_parts, fire_parts) tuples of raw (xyz, scale, quat,
    rgb, opacity) arrays for one tower, or ``None`` once it's gone."""
    if state.phase == "gone":
        return None

    spec = world.WTC1 if tower_id == "WTC1" else world.WTC2
    floor_h = spec.roof_height_m / spec.floors
    z_top = state.top_height_m
    u_step = 6.0
    z_step = max(floor_h * 2.0, 3.0)

    facade_xyz, facade_rgb = [], []
    fire_xyz, fire_rgb = [], []

    for face in ("north", "south", "east", "west"):
        xyz, frac = _face_points(spec, face, 0.0, z_top, u_step, z_step)
        if xyz.shape[0] == 0:
            continue
        rgb = np.tile(_FACADE_RGB, (xyz.shape[0], 1))
        keep = np.ones(xyz.shape[0], dtype=bool)
        is_fire = np.zeros(xyz.shape[0], dtype=bool)

        if state.damage is not None and face == state.damage.face:
            dz = state.damage
            z_lo_hole = (dz.floor_lo - 1) * floor_h
            z_hi_hole = dz.floor_hi * floor_h
            u_lo, u_hi = dz.center_frac - dz.width_frac / 2, dz.center_frac + dz.width_frac / 2
            in_hole = (
                (xyz[:, 2] >= z_lo_hole)
                & (xyz[:, 2] <= z_hi_hole)
                & (frac >= u_lo)
                & (frac <= u_hi)
            )
            keep &= ~in_hole

            spread = 6.0 * state.fire_extent * floor_h
            margin = 0.15 * state.fire_extent
            is_fire = (
                keep
                & (xyz[:, 2] >= z_lo_hole - spread)
                & (xyz[:, 2] <= z_hi_hole + spread)
                & (frac >= u_lo - margin)
                & (frac <= u_hi + margin)
            )

        facade_xyz.append(xyz[keep & ~is_fire])
        facade_rgb.append(rgb[keep & ~is_fire])
        if np.any(is_fire):
            fire_xyz.append(xyz[is_fire])
            fire_rgb.append(np.tile(_FIRE_RGB, (int(is_fire.sum()), 1)))

    # WTC1's antenna: a thin column from the roofline to the true top height.
    if tower_id == "WTC1" and state.antenna_present and state.phase in ("intact", "burning"):
        n_ant = 6
        c = spec.enu_center()
        ant_z = np.linspace(spec.roof_height_m, spec.top_height_m, n_ant)
        ant_xyz = np.column_stack([np.full(n_ant, c[0]), np.full(n_ant, c[1]), ant_z])
        facade_xyz.append(ant_xyz)
        facade_rgb.append(np.tile(_FACADE_RGB, (n_ant, 1)))

    # WTC2's tilting top block during the initial collapse phase: a small,
    # laterally-offset cap standing in for the ~10 floors that visibly tilt
    # before the tower descends.
    if tower_id == "WTC2" and state.tilt_deg > 0.5:
        lever_arm_m = 25.0
        off = lever_arm_m * math.sin(math.radians(state.tilt_deg))
        az = math.radians(state.tilt_azimuth_deg)
        c = spec.enu_center()
        cx, cy = c[0] + off * math.sin(az), c[1] + off * math.cos(az)
        n_side = 3
        h = spec.footprint_m / 2.0
        offsets = np.linspace(-h * 0.7, h * 0.7, n_side)
        gx, gy = np.meshgrid(offsets, offsets)
        cap_xyz = np.column_stack([cx + gx.ravel(), cy + gy.ravel(), np.full(gx.size, z_top + 3.0)])
        facade_xyz.append(cap_xyz)
        facade_rgb.append(np.tile(_FACADE_RGB, (cap_xyz.shape[0], 1)))

    def _finish(xyz_list, rgb_list, scale_val, opacity_val):
        if not xyz_list or all(a.shape[0] == 0 for a in xyz_list):
            return None
        xyz = np.concatenate([a for a in xyz_list if a.shape[0]])
        rgb = np.concatenate([a for a in rgb_list if a.shape[0]])
        scale = np.tile(scale_val, (xyz.shape[0], 1))
        quat = np.tile(_IDENTITY_QUAT, (xyz.shape[0], 1))
        opacity = np.full(xyz.shape[0], opacity_val)
        return xyz, scale, quat, rgb, opacity

    facade = _finish(
        facade_xyz, facade_rgb, np.array([u_step * 0.55, u_step * 0.55, z_step * 0.55]), 0.95
    )
    fire = _finish(
        fire_xyz, fire_rgb, np.array([u_step * 0.65, u_step * 0.65, z_step * 0.65]), 0.55
    )
    return facade, fire


def _damage_debris(tower_id: str, state: towers.TowerState, rng: np.random.Generator):
    """A small jittered cluster filling the impact hole, so the gash reads
    as a wound rather than a clean rectangular cut-out."""
    if state.damage is None or state.phase == "gone":
        return None
    spec = world.WTC1 if tower_id == "WTC1" else world.WTC2
    floor_h = spec.roof_height_m / spec.floors
    dz = state.damage
    z_lo, z_hi = (dz.floor_lo - 1) * floor_h, dz.floor_hi * floor_h
    z_hi = min(z_hi, state.top_height_m)
    if z_hi <= z_lo:
        return None
    u_lo, u_hi = dz.center_frac - dz.width_frac / 2, dz.center_frac + dz.width_frac / 2

    n = 40
    h = spec.footprint_m / 2.0
    u = rng.uniform(u_lo, u_hi, n) * (2 * h) - h
    z = rng.uniform(z_lo, z_hi, n)
    c = spec.enu_center()
    r = math.radians(spec.rotation_deg)
    rot = np.array([[math.cos(r), -math.sin(r)], [math.sin(r), math.cos(r)]])
    normal = rot @ _FACE_NORMALS_LOCAL[dz.face]
    tangent = rot @ _FACE_TANGENTS_LOCAL[dz.face]
    face_center_xy = c[:2] + normal * h * 0.98  # tucked slightly inside the facade line
    xy = face_center_xy[None, :] + u[:, None] * tangent[None, :]
    xyz = np.column_stack([xy, z])

    scale = np.tile([2.5, 2.5, 2.0], (n, 1))
    quat = np.tile(_IDENTITY_QUAT, (n, 1))
    dark = np.array([0.08, 0.07, 0.07])
    orange = np.array([0.6, 0.25, 0.05])
    mix = rng.uniform(0.0, 1.0, n)[:, None]
    rgb = dark[None, :] * (1 - mix) + orange[None, :] * mix
    opacity = np.full(n, 0.8)
    return xyz, scale, quat, rgb, opacity


def _rubble_and_remnant(tower_id: str, state: towers.TowerState, rng: np.random.Generator):
    if state.phase != "gone":
        return None
    spec = world.WTC1 if tower_id == "WTC1" else world.WTC2
    c = spec.enu_center()
    h = spec.footprint_m / 2.0
    n = 60
    x = c[0] + rng.uniform(-h, h, n)
    y = c[1] + rng.uniform(-h, h, n)
    z = rng.uniform(0.5, state.rubble_height_m, n)
    xyz = np.column_stack([x, y, z])
    scale = np.tile([6.0, 6.0, 3.0], (n, 1))
    quat = np.tile(_IDENTITY_QUAT, (n, 1))
    rgb = np.tile([0.42, 0.4, 0.37], (n, 1)) + rng.uniform(-0.05, 0.05, (n, 3))
    opacity = np.full(n, 0.85)
    parts = [(xyz, scale, quat, rgb, opacity)]

    if state.remnant_present:
        # The WTC1 north-face facade remnant stood at the northwest corner.
        corner_local = np.array([-h * 0.8, h * 0.8])
        r = math.radians(spec.rotation_deg)
        rot = np.array([[math.cos(r), -math.sin(r)], [math.sin(r), math.cos(r)]])
        corner = c[:2] + rot @ corner_local
        n_r = 20
        rz = np.linspace(1.0, state.remnant_height_m, n_r)
        rxyz = np.column_stack([np.full(n_r, corner[0]), np.full(n_r, corner[1]), rz])
        rscale = np.tile([4.0, 1.5, state.remnant_height_m / n_r * 0.6], (n_r, 1))
        rquat = np.tile(_IDENTITY_QUAT, (n_r, 1))
        rrgb = np.tile(_FACADE_RGB * 0.7, (n_r, 1))
        ropacity = np.full(n_r, 0.9)
        parts.append((rxyz, rscale, rquat, rrgb, ropacity))

    xyz, scale, quat, rgb, opacity = _concat_parts(parts)
    return xyz, scale, quat, rgb, opacity


# --- auxiliary background buildings (7 WTC, 3 WTC) --------------------------


def _aux_buildings(t: float, params: Params) -> GaussianCloud:
    parts = []

    wtc7 = towers.wtc7_state(t, params)
    bp7 = params.aux_buildings.wtc7
    roof7 = world.LANDMARKS_BY_ID["wtc7_roof"]
    if wtc7.phase != "gone":
        cur_h = roof7.height_m
        if wtc7.phase == "collapsing":
            cur_h = roof7.height_m * (1.0 - wtc7.progress) + bp7.footprint_m * 0.18 * wtc7.progress
        parts.append(_box_column(roof7.enu()[:2], bp7.footprint_m, cur_h))
    else:
        parts.append(_box_column(roof7.enu()[:2], bp7.footprint_m, bp7.footprint_m * 0.18))

    tw = towers.three_wtc_state(t, params)
    bp3 = params.aux_buildings.three_wtc
    center3 = world.latlon_to_enu(LatLonAlt(lat=bp3.center_lat, lon=bp3.center_lon, alt_m=0.0))
    if tw.phase != "gone":
        cur_h = bp3.height_m
        if tw.phase == "collapsing":
            cur_h = bp3.height_m * (1.0 - tw.progress) + bp3.footprint_m * 0.15 * tw.progress
        parts.append(_box_column(center3[:2], bp3.footprint_m, cur_h))
    else:
        parts.append(_box_column(center3[:2], bp3.footprint_m, bp3.footprint_m * 0.15))

    xyz, scale, quat, rgb, opacity = _concat_parts(parts)
    return _cloud(xyz, scale, quat, rgb, opacity, "skyline")


# --- top-level sampling -------------------------------------------------------


def sample(t: float, params: Params = DEFAULT_PARAMS, seed: int = 0) -> GaussianCloud:
    """Sample the whole scene at project time ``t`` (seconds since
    2001-09-11 00:00 EDT, see ``wtc4d.timeline``) into one ``GaussianCloud``.

    Deterministic for a fixed ``(t, params, seed)``: the only randomness
    (impact-hole and rubble jitter) is drawn from a ``numpy.random.default_rng(seed)``.
    """
    rng = np.random.default_rng(seed)
    clouds = [_ground(params), _skyline(params), _aux_buildings(t, params)]

    for tower_id in towers.TOWER_IDS:
        state = towers.tower_state(tower_id, t, params)
        tower_parts = _tower_parts(tower_id, state, params)
        if tower_parts is not None:
            facade, fire = tower_parts
            if facade is not None:
                clouds.append(_cloud(*facade, "towers"))
            if fire is not None:
                clouds.append(_cloud(*fire, "fire"))
            debris = _damage_debris(tower_id, state, rng)
            if debris is not None:
                clouds.append(_cloud(*debris, "damage"))
        else:
            rubble = _rubble_and_remnant(tower_id, state, rng)
            if rubble is not None:
                clouds.append(_cloud(*rubble, "rubble"))

        clouds.append(_cloud(*smoke.plume_blobs(tower_id, t, params), "smoke"))
        clouds.append(_cloud(*smoke.dust_blobs(tower_id, t, params), "dust"))

    return GaussianCloud.empty().concat(*clouds)


# --- standard 3DGS PLY / compact NPZ export ----------------------------------

_PLY_PROPERTIES = (
    "x",
    "y",
    "z",
    "nx",
    "ny",
    "nz",
    "f_dc_0",
    "f_dc_1",
    "f_dc_2",
    "opacity",
    "scale_0",
    "scale_1",
    "scale_2",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
)


def _inverse_sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 1e-6, 1.0 - 1e-6)
    return np.log(x / (1.0 - x))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _save_ply(cloud: GaussianCloud, path: Path) -> None:
    n = len(cloud)
    dtype = [(name, "f4") for name in _PLY_PROPERTIES]
    data = np.zeros(n, dtype=dtype)
    data["x"], data["y"], data["z"] = cloud.xyz[:, 0], cloud.xyz[:, 1], cloud.xyz[:, 2]
    data["nx"] = data["ny"] = data["nz"] = 0.0
    f_dc = (cloud.rgb - 0.5) / SH_C0
    data["f_dc_0"], data["f_dc_1"], data["f_dc_2"] = f_dc[:, 0], f_dc[:, 1], f_dc[:, 2]
    data["opacity"] = _inverse_sigmoid(cloud.opacity)
    log_scale = np.log(np.clip(cloud.scale, 1e-6, None))
    data["scale_0"], data["scale_1"], data["scale_2"] = (
        log_scale[:, 0],
        log_scale[:, 1],
        log_scale[:, 2],
    )
    data["rot_0"], data["rot_1"], data["rot_2"], data["rot_3"] = (
        cloud.quat[:, 0],
        cloud.quat[:, 1],
        cloud.quat[:, 2],
        cloud.quat[:, 3],
    )

    header_lines = ["ply", "format binary_little_endian 1.0", f"element vertex {n}"]
    header_lines += [f"property float {name}" for name in _PLY_PROPERTIES]
    header_lines += ["end_header", ""]
    header = "\n".join(header_lines).encode("ascii")

    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(data.tobytes())


def _load_ply(path: Path) -> GaussianCloud:
    with open(path, "rb") as fh:
        raw = fh.read()
    header_end = raw.index(b"end_header\n") + len(b"end_header\n")
    header = raw[:header_end].decode("ascii")
    lines = header.splitlines()
    n = 0
    props: list[str] = []
    for line in lines:
        if line.startswith("element vertex"):
            n = int(line.split()[-1])
        elif line.startswith("property float"):
            props.append(line.split()[-1])
    dtype = [(name, "f4") for name in props]
    data = np.frombuffer(raw[header_end:], dtype=dtype, count=n)

    xyz = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float32)
    f_dc = np.column_stack([data["f_dc_0"], data["f_dc_1"], data["f_dc_2"]])
    rgb = np.clip(f_dc * SH_C0 + 0.5, 0.0, 1.0).astype(np.float32)
    opacity = _sigmoid(data["opacity"]).astype(np.float32)
    log_scale = np.column_stack([data["scale_0"], data["scale_1"], data["scale_2"]])
    scale = np.exp(log_scale).astype(np.float32)
    quat = np.column_stack([data["rot_0"], data["rot_1"], data["rot_2"], data["rot_3"]]).astype(
        np.float32
    )
    layer = np.zeros(n, dtype=np.int32)  # layer id is not part of the standard 3DGS PLY schema
    return GaussianCloud(xyz=xyz, scale=scale, quat=quat, rgb=rgb, opacity=opacity, layer=layer)


def _save_npz(cloud: GaussianCloud, path: Path) -> None:
    np.savez_compressed(
        path,
        xyz=cloud.xyz,
        scale=cloud.scale,
        quat=cloud.quat,
        rgb=cloud.rgb,
        opacity=cloud.opacity,
        layer=cloud.layer,
    )


def _load_npz(path: Path) -> GaussianCloud:
    with np.load(path) as data:
        return GaussianCloud(
            xyz=data["xyz"],
            scale=data["scale"],
            quat=data["quat"],
            rgb=data["rgb"],
            opacity=data["opacity"],
            layer=data["layer"],
        )
