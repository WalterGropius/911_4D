"""Small mesh constructors used to build the 2001 scene.

All functions return :class:`trimesh.Trimesh` in world ENU metres.  Polygon
inputs are ``(N, 2)`` arrays of ``(east, north)`` and are closed automatically.
"""

from __future__ import annotations

import math

import numpy as np


def _trimesh():
    try:
        import trimesh
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError("trimesh is required: pip install 'wtc4d[geo]'") from exc
    return trimesh


def signed_area(poly: np.ndarray) -> float:
    """Shoelace area of an open ring; positive when counter-clockwise."""
    p = np.asarray(poly, dtype=np.float64)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def clean_ring(poly) -> np.ndarray:
    """Drop a duplicated closing vertex and consecutive duplicates."""
    p = np.asarray(poly, dtype=np.float64)[:, :2]
    if len(p) > 1 and np.allclose(p[0], p[-1]):
        p = p[:-1]
    keep = [0]
    for i in range(1, len(p)):
        if not np.allclose(p[i], p[keep[-1]], atol=1e-9):
            keep.append(i)
    return p[keep]


def extrude(poly, z0: float, z1: float, *, holes=None):
    """Watertight prism between ``z0`` and ``z1`` over a simple polygon."""
    trimesh = _trimesh()
    from shapely.geometry import Polygon

    ring = clean_ring(poly)
    if len(ring) < 3 or z1 <= z0:
        return None
    shell = Polygon(ring, holes=[clean_ring(h) for h in holes] if holes else None)
    if not shell.is_valid:
        shell = shell.buffer(0)
    if shell.is_empty or shell.area <= 0:
        return None
    if shell.geom_type == "MultiPolygon":
        parts = [extrude(np.asarray(g.exterior.coords), z0, z1) for g in shell.geoms]
        parts = [p for p in parts if p is not None]
        return trimesh.util.concatenate(parts) if parts else None
    mesh = trimesh.creation.extrude_polygon(shell, height=float(z1 - z0))
    mesh.apply_translation([0.0, 0.0, float(z0)])
    return mesh


def box_from_center(center_en, side_m: float, rot_deg: float, z0: float, z1: float):
    """Square prism, ``rot_deg`` counter-clockwise from east-aligned faces."""
    h = side_m / 2.0
    r = math.radians(rot_deg)
    rot = np.array([[math.cos(r), -math.sin(r)], [math.sin(r), math.cos(r)]])
    sq = np.array([[-h, -h], [h, -h], [h, h], [-h, h]]) @ rot.T
    return extrude(sq + np.asarray(center_en, dtype=np.float64), z0, z1)


def cylinder(center_en, radius_m: float, z0: float, z1: float, sections: int = 16):
    """Vertical cylinder (used for the WTC 1 rooftop mast)."""
    trimesh = _trimesh()
    m = trimesh.creation.cylinder(radius=float(radius_m), height=float(z1 - z0), sections=sections)
    m.apply_translation([float(center_en[0]), float(center_en[1]), float((z0 + z1) / 2.0)])
    return m


def pyramid_cap(poly, z0: float, z1: float, apex_en=None):
    """Pyramid from a base ring up to a single apex."""
    trimesh = _trimesh()
    ring = clean_ring(poly)
    if len(ring) < 3 or z1 <= z0:
        return None
    apex = np.asarray(apex_en if apex_en is not None else ring.mean(axis=0), dtype=np.float64)
    verts = np.vstack([np.column_stack([ring, np.full(len(ring), z0)]), [apex[0], apex[1], z1]])
    n = len(ring)
    faces = [[i, (i + 1) % n, n] for i in range(n)]
    # base cap (fan) so the cap alone is closed
    faces += [[0, i + 1, i] for i in range(1, n - 1)]
    m = trimesh.Trimesh(vertices=verts, faces=np.asarray(faces), process=True)
    m.fix_normals()
    return m


def stepped_cap(poly, z0: float, z1: float, steps: int = 4, inset_frac: float = 0.62):
    """Stepped-pyramid cap (the 1 WFC / 4 WFC roof idiom)."""
    trimesh = _trimesh()
    ring = clean_ring(poly)
    c = ring.mean(axis=0)
    parts = []
    for i in range(steps):
        f = 1.0 - (1.0 - inset_frac) * i / max(steps - 1, 1)
        za = z0 + (z1 - z0) * i / steps
        zb = z0 + (z1 - z0) * (i + 1) / steps
        parts.append(extrude(c + (ring - c) * f, za, zb))
    parts = [p for p in parts if p is not None]
    return trimesh.util.concatenate(parts) if parts else None


def dome_cap(poly, z0: float, z1: float, rings: int = 6, sections: int = 20):
    """Half-ellipsoid cap over a polygon's bounding circle (the 2 WFC dome)."""
    trimesh = _trimesh()
    ring = clean_ring(poly)
    c = ring.mean(axis=0)
    r = float(np.max(np.linalg.norm(ring - c, axis=1)))
    h = float(z1 - z0)
    if r <= 0 or h <= 0:
        return None
    verts = []
    for i in range(rings):
        phi = (math.pi / 2.0) * i / rings
        rr, zz = r * math.cos(phi), z0 + h * math.sin(phi)
        for j in range(sections):
            th = 2 * math.pi * j / sections
            verts.append([c[0] + rr * math.cos(th), c[1] + rr * math.sin(th), zz])
    top = len(verts)
    verts.append([c[0], c[1], z1])
    faces = []
    for i in range(rings - 1):
        for j in range(sections):
            a = i * sections + j
            b = i * sections + (j + 1) % sections
            faces += [[a, b, b + sections], [a, b + sections, a + sections]]
    for j in range(sections):
        a = (rings - 1) * sections + j
        b = (rings - 1) * sections + (j + 1) % sections
        faces.append([a, b, top])
    m = trimesh.Trimesh(vertices=np.asarray(verts), faces=np.asarray(faces), process=True)
    m.fix_normals()
    return m


def barrel_vault(poly, z0: float, z1: float, sections: int = 14):
    """Half-cylinder vault over the long axis of a rectangle (Winter Garden)."""
    trimesh = _trimesh()
    ring = clean_ring(poly)
    c = ring.mean(axis=0)
    d = ring - c
    # principal axis of the footprint
    u, _, _ = np.linalg.svd(d.T @ d)
    axis = u[:, 0]
    perp = np.array([-axis[1], axis[0]])
    half_len = float(np.max(np.abs(d @ axis)))
    half_w = float(np.max(np.abs(d @ perp)))
    h = float(z1 - z0)
    if half_len <= 0 or half_w <= 0 or h <= 0:
        return None
    verts, faces = [], []
    for j in range(sections + 1):
        th = math.pi * j / sections
        w, zz = half_w * math.cos(th), z0 + h * math.sin(th)
        for s in (-1.0, 1.0):
            p = c + axis * (half_len * s) + perp * w
            verts.append([p[0], p[1], zz])
    for j in range(sections):
        a, b = 2 * j, 2 * j + 1
        cc, dd = 2 * (j + 1), 2 * (j + 1) + 1
        faces += [[a, b, dd], [a, dd, cc]]
    # end walls
    faces += [[2 * j, 2 * (j + 1), 0] for j in range(1, sections)]
    faces += [[2 * j + 1, 1, 2 * (j + 1) + 1] for j in range(1, sections)]
    m = trimesh.Trimesh(vertices=np.asarray(verts), faces=np.asarray(faces), process=True)
    m.fix_normals()
    return m


ROOF_CAPS = {
    "flat": None,
    "pyramid": pyramid_cap,
    "stepped": stepped_cap,
    "dome": dome_cap,
    "barrel": barrel_vault,
}
