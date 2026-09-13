"""Gaussian initialisation from the geometric priors we have.

Unlike an ordinary 3DGS capture we do **not** start from an SfM point cloud
(classic SfM fails on this footage: smoke everywhere, huge baselines, SD
compression).  Instead we start from what is actually known:

* the **city mesh** for 2001 (``wtc4d.geo.load_scene()``) -- surface-sampled,
  one colour per building;
* the **procedural model** (``wtc4d.procedural.sample(t)``) -- towers, fire,
  plume and collapse for any ``t``, already expressed as gaussians;
* optionally a **COLMAP** sparse cloud, when a sub-scene (the rubble pile,
  a single rooftop sequence) really was reconstructed photogrammetrically.

Both sibling packages are developed in parallel, so every import here is lazy
and every entry point has a fallback that only needs ``wtc4d.world``: two
boxes for the towers plus a ground plane.  That fallback is also what the CPU
tests use, which keeps them fast and dependency-free.

Everything returns :class:`wtc4d.recon.gaussians.Gaussians` in world ENU
metres with :class:`~wtc4d.recon.gaussians.Layer` ids assigned, because the
losses treat the layers differently (static geometry is supervised outside
dynamic masks, smoke only inside them).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from wtc4d import world

from .gaussians import Gaussians, Layer

__all__ = [
    "assign_layers",
    "deduplicate",
    "estimate_spacing",
    "from_colmap_points",
    "from_mesh",
    "from_ply",
    "from_procedural",
    "init_gaussians",
    "load_prior_scene",
    "prior_boxes_scene",
]

# Rough albedos for the fallback prior.  The towers were aluminium-clad
# (bright, slightly warm); the plaza and streets are dark asphalt/granite.
_TOWER_RGB = (0.62, 0.61, 0.58)
_GROUND_RGB = (0.26, 0.26, 0.27)
_BUILDING_RGB = (0.48, 0.46, 0.44)

_LAYER_KEYWORDS: tuple[tuple[str, Layer], ...] = (
    ("wtc1", Layer.TOWERS),
    ("wtc2", Layer.TOWERS),
    ("tower", Layer.TOWERS),
    ("ground", Layer.GROUND),
    ("terrain", Layer.GROUND),
    ("street", Layer.GROUND),
    ("plaza", Layer.GROUND),
    ("water", Layer.GROUND),
    ("smoke", Layer.SMOKE),
    ("plume", Layer.SMOKE),
    ("fire", Layer.SMOKE),
    ("dust", Layer.SMOKE),
    ("debris", Layer.DEBRIS),
    ("rubble", Layer.DEBRIS),
    ("sky", Layer.SKY),
)


def layer_for_name(name: str, default: Layer = Layer.BUILDINGS) -> Layer:
    """Map a mesh/geometry name to a :class:`Layer` by keyword."""
    low = (name or "").lower()
    for key, layer in _LAYER_KEYWORDS:
        if key in low:
            return layer
    return default


def color_for_name(name: str) -> tuple[float, float, float]:
    """A stable, plausible albedo for a named piece of geometry.

    Deterministic (hash of the name) so that reruns and the viewer agree, and
    desaturated so that it reads as "unknown grey building" rather than as a
    claim about the real colour.
    """
    layer = layer_for_name(name)
    if layer is Layer.TOWERS:
        return _TOWER_RGB
    if layer is Layer.GROUND:
        return _GROUND_RGB
    h = hashlib.sha1(name.encode("utf-8")).digest()
    jitter = [(b / 255.0 - 0.5) * 0.12 for b in h[:3]]
    return tuple(
        float(np.clip(c + j, 0.15, 0.85)) for c, j in zip(_BUILDING_RGB, jitter, strict=True)
    )


# --------------------------------------------------------------------- priors
def prior_boxes_scene():
    """Fallback prior: both towers as boxes plus a ground plane, as a ``trimesh.Scene``.

    Geometry names are ``WTC1``, ``WTC2``, ``ground`` so the layer assignment
    below works identically on the real city mesh.
    """
    import trimesh

    scene = trimesh.Scene()
    for tower in world.TOWERS:
        corners = tower.box_corners()
        centre = corners.mean(axis=0)
        box = trimesh.creation.box(
            extents=(tower.footprint_m, tower.footprint_m, tower.roof_height_m)
        )
        box.apply_translation(centre)
        scene.add_geometry(box, geom_name=tower.id)

    # 1.2 km square of ground at plaza level; enough to anchor the lower
    # portion of every camera's view in the toy/fallback scene.
    ground = trimesh.creation.box(extents=(1200.0, 1200.0, 1.0))
    ground.apply_translation((0.0, 0.0, -0.5))
    scene.add_geometry(ground, geom_name="ground")
    return scene


def load_prior_scene(epoch_id: str | None = None):
    """The static geometric prior as a ``trimesh.Scene``.

    Uses ``wtc4d.geo.load_scene()`` when the ``geo`` workstream is installed
    (passing ``epoch_id`` when it accepts one -- WTC2 is gone in E3), and
    falls back to :func:`prior_boxes_scene`.
    """
    try:
        from wtc4d import geo  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return prior_boxes_scene()
    loader = getattr(geo, "load_scene", None)
    if loader is None:
        return prior_boxes_scene()
    try:
        return loader(epoch_id=epoch_id) if epoch_id is not None else loader()
    except TypeError:
        try:
            return loader()
        except Exception:  # noqa: BLE001
            return prior_boxes_scene()
    except Exception:  # noqa: BLE001
        return prior_boxes_scene()


# ------------------------------------------------------------------ samplers
def from_mesh(
    scene_or_mesh=None,
    n_points: int = 100_000,
    *,
    epoch_id: str | None = None,
    use_vertex_colors: bool = True,
    scale_m: float | None = None,
    opacity: float = 0.1,
    sh_degree: int = 0,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> Gaussians:
    """Sample gaussians on the surface of the prior mesh.

    Points are distributed over the geometries proportionally to area, which
    puts most of the budget on the tower facades and the ground -- the
    surfaces the cameras actually see.  Colours come from the mesh when it has
    them (``use_vertex_colors``), otherwise from :func:`color_for_name`.
    Initial scales default to the local point spacing so the surface is just
    covered.
    """
    import trimesh

    scene = scene_or_mesh if scene_or_mesh is not None else load_prior_scene(epoch_id)
    if isinstance(scene, trimesh.Trimesh):
        geometries = {"mesh": scene}
    elif isinstance(scene, trimesh.Scene):
        geometries = {str(k): v for k, v in scene.geometry.items()}
    else:
        raise TypeError(f"expected a trimesh Scene or Trimesh, got {type(scene)!r}")

    meshes = {k: v for k, v in geometries.items() if isinstance(v, trimesh.Trimesh) and v.area > 0}
    if not meshes:
        raise ValueError("prior scene has no triangle meshes with area")

    total_area = sum(m.area for m in meshes.values())
    rng = np.random.default_rng(seed)
    xyz_parts, rgb_parts, layer_parts = [], [], []
    for name, mesh in meshes.items():
        count = max(16, int(round(n_points * mesh.area / total_area)))
        pts, face_idx = trimesh.sample.sample_surface(mesh, count, seed=int(rng.integers(1 << 30)))
        xyz_parts.append(np.asarray(pts, dtype=np.float64))
        rgb_parts.append(_sample_colors(mesh, face_idx, name, len(pts), use_vertex_colors))
        layer_parts.append(np.full(len(pts), int(layer_for_name(name)), dtype=np.int64))

    xyz = np.concatenate(xyz_parts)
    rgb = np.concatenate(rgb_parts)
    layers = np.concatenate(layer_parts)
    scales = estimate_spacing(xyz) if scale_m is None else np.full(len(xyz), float(scale_m))
    return Gaussians.from_points(
        xyz, rgb, scale_m=scales, opacity=opacity, layer=layers, sh_degree=sh_degree, device=device
    )


def _sample_colors(mesh, face_idx, name: str, count: int, use_vertex_colors: bool) -> np.ndarray:
    if use_vertex_colors:
        visual = getattr(mesh, "visual", None)
        try:
            if visual is not None and getattr(visual, "kind", None) == "face":
                cols = np.asarray(visual.face_colors, dtype=np.float64)[face_idx][:, :3] / 255.0
                return cols
            if visual is not None and getattr(visual, "kind", None) == "vertex":
                faces = mesh.faces[face_idx]
                vc = np.asarray(visual.vertex_colors, dtype=np.float64)[:, :3] / 255.0
                return vc[faces].mean(axis=1)
        except Exception:  # noqa: BLE001 - colour extraction is best-effort
            pass
    return np.tile(np.asarray(color_for_name(name), dtype=np.float64), (count, 1))


def from_ply(
    path: str | Path, *, layer: Layer | None = None, device: str | torch.device = "cpu"
) -> Gaussians:
    """Load a 3DGS ``.ply`` (e.g. the procedural export) as an init.

    ``layer`` overrides the layer of every gaussian, which is what you want
    when loading a plume-only or debris-only export that predates the
    ``layer`` field.
    """
    g = Gaussians.load_ply(path, device=device)
    if layer is not None:
        g.layer = torch.full((g.n,), int(layer), dtype=torch.int64, device=g.means.device)
    return g


def from_procedural(
    t: float, *, device: str | torch.device = "cpu", ply_fallback: str | Path | None = None
) -> Gaussians:
    """Gaussians from ``wtc4d.procedural.sample(t)``, or from a ``.ply`` fallback.

    The procedural workstream returns its own ``GaussianCloud``; we accept
    anything exposing ``means``/``scales``/``opacities`` style attributes or a
    ``to_ply`` / ``as_dict`` method, and raise otherwise, because guessing
    wrong would silently corrupt the world frame.
    """
    try:
        from wtc4d import procedural  # noqa: PLC0415

        cloud = procedural.sample(t)
    except Exception as exc:  # noqa: BLE001
        if ply_fallback is not None and Path(ply_fallback).exists():
            return from_ply(ply_fallback, device=device)
        raise RuntimeError("wtc4d.procedural is unavailable and no ply_fallback was given") from exc
    return gaussians_from_cloud(cloud, device=device)


def gaussians_from_cloud(cloud, *, device: str | torch.device = "cpu") -> Gaussians:
    """Adapt a foreign gaussian cloud object to :class:`Gaussians`.

    Accepts our own type, a mapping, or any object with ``means`` plus either
    ``scales``/``log_scales`` and ``opacities``/``logit_opacities``.  Colours
    may be ``rgb``/``colors`` (0..1) or ``sh_dc``.
    """
    if isinstance(cloud, Gaussians):
        return cloud.to(device=device)
    get = cloud.get if isinstance(cloud, dict) else lambda k, d=None: getattr(cloud, k, d)

    means = np.asarray(get("means", get("xyz")), dtype=np.float64)
    if means is None:
        raise TypeError("gaussian cloud has no means/xyz")
    n = len(means)

    scales = get("scales")
    log_scales = get("log_scales")
    if log_scales is not None:
        scales = np.exp(np.asarray(log_scales, dtype=np.float64))
    scales = np.broadcast_to(
        np.asarray(scales if scales is not None else 1.0, dtype=np.float64), (n, 3)
    )

    rgb = get("rgb", get("colors"))
    if rgb is None:
        sh_dc = get("sh_dc")
        rgb = None if sh_dc is None else 0.28209479177387814 * np.asarray(sh_dc) + 0.5

    opac = get("opacities", get("opacity"))
    logit = get("logit_opacities")
    if logit is None:
        o = np.clip(
            np.broadcast_to(np.asarray(opac if opac is not None else 0.5, dtype=np.float64), (n,)),
            1e-4,
            1 - 1e-4,
        )
        logit = np.log(o / (1.0 - o))

    quats = get("quats", get("rotations"))
    layers = get("layer", get("layers"))

    g = Gaussians.from_points(means, rgb, scale_m=1.0, device=device)
    g.log_scales = torch.as_tensor(
        np.log(np.clip(scales, 1e-4, None)), dtype=g.means.dtype, device=g.device
    )
    g.logit_opacities = torch.as_tensor(
        np.asarray(logit, dtype=np.float64), dtype=g.means.dtype, device=g.device
    )
    if quats is not None:
        g.quats = torch.as_tensor(
            np.asarray(quats, dtype=np.float64), dtype=g.means.dtype, device=g.device
        )
    if layers is not None:
        g.layer = torch.as_tensor(np.asarray(layers, dtype=np.int64), device=g.device)
    return g


def from_colmap_points(
    model_or_path,
    *,
    scale_m: float | None = None,
    opacity: float = 0.1,
    layer: Layer = Layer.OTHER,
    sh_degree: int = 0,
    device: str | torch.device = "cpu",
) -> Gaussians:
    """Gaussians at a COLMAP sparse cloud (``points3D.txt``), coloured by it."""
    from .colmap import ColmapModel, read_colmap_model

    model = (
        model_or_path
        if isinstance(model_or_path, ColmapModel)
        else read_colmap_model(model_or_path)
    )
    xyz, rgb = model.points_array()
    if len(xyz) == 0:
        raise ValueError("COLMAP model has no 3D points")
    scales = estimate_spacing(xyz) if scale_m is None else np.full(len(xyz), float(scale_m))
    return Gaussians.from_points(
        xyz, rgb, scale_m=scales, opacity=opacity, layer=layer, sh_degree=sh_degree, device=device
    )


# ------------------------------------------------------------------ utilities
def estimate_spacing(
    xyz, k: int = 3, max_ref: int = 20_000, chunk: int = 2048, seed: int = 0
) -> np.ndarray:
    """Mean distance to the ``k`` nearest neighbours, per point (metres).

    Used as the initial isotropic scale: a gaussian roughly the size of the
    gap to its neighbours covers the surface without ballooning.  For large
    clouds the neighbour search uses a random reference subset (``max_ref``),
    which is accurate enough for an initialisation and keeps this O(N).
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    n = len(xyz)
    if n <= 1:
        return np.full(max(n, 0), 1.0)
    rng = np.random.default_rng(seed)
    ref = xyz if n <= max_ref else xyz[rng.choice(n, max_ref, replace=False)]
    ref_t = torch.as_tensor(ref)
    k_eff = min(k + 1, len(ref))
    out = np.empty(n, dtype=np.float64)
    for start in range(0, n, chunk):
        block = torch.as_tensor(xyz[start : start + chunk])
        d = torch.cdist(block, ref_t)
        vals, _ = torch.topk(d, k_eff, dim=1, largest=False)
        out[start : start + block.shape[0]] = vals[:, 1:].mean(dim=1).numpy()
    scale = float(len(ref)) / float(n)
    if scale < 1.0:  # subsampled: neighbours are further apart than they really are
        out *= scale ** (1.0 / 3.0)
    return np.clip(out, 1e-3, None)


def deduplicate(g: Gaussians, radius: float = 0.5) -> Gaussians:
    """Keep at most one gaussian per ``radius``-sized voxel, per layer.

    Merging mesh samples with a procedural cloud (or two epochs' inits)
    otherwise stacks near-coincident gaussians, which wastes budget and makes
    the opacity gradient fight itself.  The most opaque gaussian in each voxel
    wins, so deliberately dense structures survive.
    """
    if g.n == 0 or radius <= 0:
        return g
    keys = torch.floor(g.means.detach() / radius).to(torch.int64)
    layer = g.layer if g.layer is not None else torch.zeros(g.n, dtype=torch.int64, device=g.device)
    stacked = torch.cat([keys, layer.reshape(-1, 1)], dim=1).cpu().numpy()
    _, first_idx, inverse = np.unique(stacked, axis=0, return_index=True, return_inverse=True)

    op = g.logit_opacities.detach().cpu().numpy()
    inverse = inverse.reshape(-1)
    best = np.asarray(first_idx, dtype=np.int64).copy()
    order = np.argsort(-op, kind="stable")
    seen = np.zeros(len(first_idx), dtype=bool)
    for i in order:
        b = inverse[i]
        if not seen[b]:
            seen[b] = True
            best[b] = i
    keep = torch.as_tensor(np.sort(best), device=g.means.device)
    return g.select(keep)


def assign_layers(g: Gaussians, *, ground_z: float = 2.0, tower_margin_m: float = 8.0) -> Gaussians:
    """(Re)assign layer ids from geometry alone, for inits that lack them.

    Anything inside a tower footprint (plus ``tower_margin_m``) and above
    ``ground_z`` becomes :attr:`Layer.TOWERS`; anything below ``ground_z``
    becomes :attr:`Layer.GROUND`; the rest is :attr:`Layer.BUILDINGS`.
    Smoke/debris layers are never inferred here -- those come from the
    procedural model or from a dynamic-mask-driven split.
    """
    means = g.means.detach().cpu().numpy()
    layer = np.full(len(means), int(Layer.BUILDINGS), dtype=np.int64)
    layer[means[:, 2] < ground_z] = int(Layer.GROUND)
    for tower in world.TOWERS:
        c = tower.enu_center()
        half = tower.footprint_m / 2.0 + tower_margin_m
        inside = (
            (np.abs(means[:, 0] - c[0]) <= half)
            & (np.abs(means[:, 1] - c[1]) <= half)
            & (means[:, 2] >= ground_z)
            & (means[:, 2] <= c[2] + tower.top_height_m + tower_margin_m)
        )
        layer[inside] = int(Layer.TOWERS)
    g.layer = torch.as_tensor(layer, device=g.means.device)
    return g


def init_gaussians(
    *,
    mesh_points: int = 100_000,
    epoch_id: str | None = None,
    procedural_t: float | None = None,
    procedural_ply: str | Path | None = None,
    colmap_path: str | Path | None = None,
    dedup_radius: float = 0.5,
    sh_degree: int = 0,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> Gaussians:
    """Build a training init by merging every prior that is available.

    Sources are tried in order (mesh, procedural/ply, COLMAP); any that fail
    are skipped with the others still used, because in Phase 1 the sibling
    packages land at different times.  The merged cloud is deduplicated so
    overlapping priors do not double-count.
    """
    parts: list[Gaussians] = []
    if mesh_points > 0:
        parts.append(
            from_mesh(
                None, mesh_points, epoch_id=epoch_id, sh_degree=sh_degree, seed=seed, device=device
            )
        )
    if procedural_t is not None or procedural_ply is not None:
        try:
            if procedural_t is not None:
                parts.append(
                    from_procedural(procedural_t, device=device, ply_fallback=procedural_ply)
                )
            else:
                parts.append(from_ply(procedural_ply, device=device))
        except Exception:  # noqa: BLE001 - optional prior
            pass
    if colmap_path is not None:
        try:
            parts.append(from_colmap_points(colmap_path, sh_degree=sh_degree, device=device))
        except Exception:  # noqa: BLE001 - optional prior
            pass
    if not parts:
        raise ValueError("no initialisation source produced gaussians")
    merged = Gaussians.cat([p.with_sh_degree(sh_degree) for p in parts])
    return deduplicate(merged, dedup_radius)
