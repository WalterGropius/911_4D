"""A tiny numpy software renderer of the 2001 skyline, for tests and validation.

This is deliberately *not* a graphics engine.  It exists so that the camreg
pipeline can be tested end to end against ground truth without any optional
dependency, without a GPU and without committing a single image to the repo:
generate a frame from a known camera, degrade it to something that looks like
2001 SD footage, register it blind, compare with the camera we started from.

Geometry comes from the same prior the real pipeline uses --
:data:`wtc4d.world.TOWERS` and :data:`wtc4d.world.LANDMARKS` -- so a test that
passes here proves the *conventions* and the *solvers* agree, not that the
world model is right.

The renderer is a per-triangle z-buffer rasteriser with perspective-correct
interpolation, flat shading from a plausible sun direction for the morning of
2001-09-11, and a deterministic 3D value-noise texture so that feature
trackers have something to lock onto.  Triangles crossing the near plane are
dropped rather than clipped, which is fine for the distant views the project
actually cares about and visibly wrong for a camera inside the geometry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from wtc4d.camreg.conventions import invert_rigid
from wtc4d.schema.camera import CameraIntrinsics
from wtc4d.world import LANDMARKS, TOWERS, latlon_to_enu

__all__ = ["RenderResult", "SyntheticScene", "degrade", "default_scene"]

# Sun direction for ~09:00 EDT on 2001-09-11 in New York: azimuth ~110 deg
# (east-south-east), elevation ~38 deg.  Used only for shading, but it is the
# same geometry the sync workstream uses for solar-shadow timing, so keep the
# numbers here honest rather than "nice".
SUN_AZIMUTH_DEG = 110.0
SUN_ELEVATION_DEG = 38.0


def _sun_vector() -> np.ndarray:
    az, el = math.radians(SUN_AZIMUTH_DEG), math.radians(SUN_ELEVATION_DEG)
    return np.array([math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), math.sin(el)])


@dataclass
class RenderResult:
    """Output of :meth:`SyntheticScene.render`."""

    image: np.ndarray
    """(H, W) float32 in [0, 1], grayscale."""
    depth: np.ndarray
    """(H, W) float32 metres along the optical axis; ``inf`` for sky."""
    face_id: np.ndarray
    """(H, W) int32 index into ``SyntheticScene.faces``; -1 for sky."""
    object_id: np.ndarray
    """(H, W) int32 index into ``SyntheticScene.object_names``; -1 for sky."""

    @property
    def sky(self) -> np.ndarray:
        return ~np.isfinite(self.depth)

    def to_uint8(self) -> np.ndarray:
        return np.clip(self.image * 255.0, 0, 255).astype(np.uint8)


# --- value noise --------------------------------------------------------------
def _hash01(ix: np.ndarray, iy: np.ndarray, iz: np.ndarray) -> np.ndarray:
    """Deterministic pseudo-random value in [0, 1) for an integer lattice point."""
    h = ix * 374761393 + iy * 668265263 + iz * 2147483647
    h = (h ^ (h >> np.int64(13))) * np.int64(1274126177)
    h = h ^ (h >> np.int64(16))
    return (h & np.int64(0xFFFFFF)).astype(np.float64) / float(0x1000000)


def value_noise(points: np.ndarray, period_m: float) -> np.ndarray:
    """Smooth 3D value noise in [0, 1) sampled at world points (N, 3)."""
    p = np.asarray(points, dtype=np.float64) / max(period_m, 1e-6)
    i = np.floor(p).astype(np.int64)
    f = p - i
    w = f * f * (3.0 - 2.0 * f)  # smoothstep
    out = np.zeros(len(p))
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                c = _hash01(i[:, 0] + dx, i[:, 1] + dy, i[:, 2] + dz)
                wx = w[:, 0] if dx else 1.0 - w[:, 0]
                wy = w[:, 1] if dy else 1.0 - w[:, 1]
                wz = w[:, 2] if dz else 1.0 - w[:, 2]
                out += c * wx * wy * wz
    return out


# --- scene --------------------------------------------------------------------
@dataclass
class SyntheticScene:
    vertices: np.ndarray  # (V, 3) world ENU metres
    faces: np.ndarray  # (F, 3) vertex indices
    face_albedo: np.ndarray  # (F,) base reflectance in [0, 1]
    face_object: np.ndarray  # (F,) index into object_names
    object_names: list[str]

    # --- construction -----------------------------------------------------
    @staticmethod
    def _box(centre_xy, half_x: float, half_y: float, z0: float, z1: float, rot_deg: float = 0.0):
        r = math.radians(rot_deg)
        rot = np.array([[math.cos(r), -math.sin(r)], [math.sin(r), math.cos(r)]])
        sq = np.array([[-half_x, -half_y], [half_x, -half_y], [half_x, half_y], [-half_x, half_y]])
        sq = sq @ rot.T + np.asarray(centre_xy, dtype=np.float64).reshape(2)
        verts = np.vstack(
            [np.column_stack([sq, np.full(4, z0)]), np.column_stack([sq, np.full(4, z1)])]
        )
        faces = np.array(
            [
                [0, 1, 5],
                [0, 5, 4],  # south-ish wall
                [1, 2, 6],
                [1, 6, 5],  # east-ish wall
                [2, 3, 7],
                [2, 7, 6],
                [3, 0, 4],
                [3, 4, 7],
                [4, 5, 6],
                [4, 6, 7],  # roof
            ],
            dtype=np.int64,
        )
        return verts, faces

    @staticmethod
    def _ground_grid(half_extent: float = 8000.0, n: int = 16, z: float = 0.0):
        """A flat grid of quads at height ``z``.

        The ground is tiled rather than a single huge quad because the
        rasteriser drops any triangle that crosses the near plane: one giant
        quad under the camera would always be dropped, and the horizon with it.
        """
        xs = np.linspace(-half_extent, half_extent, n + 1)
        gx, gy = np.meshgrid(xs, xs, indexing="ij")
        verts = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, z)])
        faces = []
        stride = n + 1
        for i in range(n):
            for j in range(n):
                a = i * stride + j
                faces += [[a, a + 1, a + stride + 1], [a, a + stride + 1, a + stride]]
        return verts, np.array(faces, dtype=np.int64)

    @classmethod
    def from_boxes(cls, boxes: list[tuple[str, np.ndarray, np.ndarray, float]]) -> SyntheticScene:
        verts_all, faces_all, albedo_all, obj_all, names = [], [], [], [], []
        offset = 0
        for name, v, f, albedo in boxes:
            names.append(name)
            verts_all.append(v)
            faces_all.append(f + offset)
            albedo_all.append(np.full(len(f), albedo))
            obj_all.append(np.full(len(f), len(names) - 1, dtype=np.int64))
            offset += len(v)
        return cls(
            vertices=np.vstack(verts_all),
            faces=np.vstack(faces_all),
            face_albedo=np.concatenate(albedo_all),
            face_object=np.concatenate(obj_all),
            object_names=names,
        )

    # --- rendering --------------------------------------------------------
    def render(
        self,
        c2w: np.ndarray,
        intrinsics: CameraIntrinsics,
        texture: bool = True,
        znear: float = 1.0,
    ) -> RenderResult:
        """Rasterise the scene from a camera-to-world pose (OpenCV axes)."""
        w, h = int(intrinsics.width), int(intrinsics.height)
        w2c = invert_rigid(c2w)
        cam_all = self.vertices @ w2c[:3, :3].T + w2c[:3, 3]

        depth = np.full((h, w), np.inf, dtype=np.float64)
        face_id = np.full((h, w), -1, dtype=np.int32)
        wpos = np.zeros((h, w, 3), dtype=np.float64)

        for fi, tri in enumerate(self.faces):
            cam = cam_all[tri]
            world = self.vertices[tri]
            if cam[:, 2].max() <= znear:
                continue
            if cam[:, 2].min() < znear:
                cam, world = _clip_near(cam, world, znear)
                if len(cam) < 3:
                    continue
            for j in range(1, len(cam) - 1):  # fan triangulation of the clipped polygon
                idx = [0, j, j + 1]
                _raster_tri(cam[idx], world[idx], fi, intrinsics, depth, face_id, wpos, w, h)

        # shading
        img = np.zeros((h, w), dtype=np.float64)
        sky = face_id < 0
        vv = np.linspace(0.0, 1.0, h)[:, None]
        img[:] = 0.92 - 0.22 * (1.0 - vv)  # bright horizon, slightly darker zenith
        lit = ~sky
        if lit.any():
            sun = _sun_vector()
            normals = self._face_normals()
            n = normals[face_id[lit]]
            lambert = np.clip(n @ sun, 0.0, 1.0)
            albedo = self.face_albedo[face_id[lit]]
            shade = albedo * (0.35 + 0.65 * lambert)
            if texture:
                p = wpos[lit]
                t = 0.55 * value_noise(p, 3.0) + 0.45 * value_noise(p, 17.0)
                shade = shade * (0.80 + 0.40 * t)
            img[lit] = np.clip(shade, 0.0, 1.0)
        obj = np.where(face_id >= 0, self.face_object[np.clip(face_id, 0, None)], -1)
        return RenderResult(
            image=img.astype(np.float32),
            depth=depth.astype(np.float32),
            face_id=face_id,
            object_id=obj.astype(np.int32),
        )

    def _face_normals(self) -> np.ndarray:
        v = self.vertices[self.faces]
        n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
        norm = np.linalg.norm(n, axis=1, keepdims=True)
        return n / np.where(norm > 1e-12, norm, 1.0)


def _clip_near(cam: np.ndarray, world: np.ndarray, znear: float):
    """Sutherland-Hodgman clip of one triangle against ``z >= znear``.

    Returns the clipped polygon in both camera and world coordinates (the same
    interpolation parameter is applied to each), so shading and texturing stay
    correct across the clip.
    """
    out_cam: list[np.ndarray] = []
    out_world: list[np.ndarray] = []
    n = len(cam)
    for i in range(n):
        a_c, b_c = cam[i], cam[(i + 1) % n]
        a_w, b_w = world[i], world[(i + 1) % n]
        a_in, b_in = a_c[2] >= znear, b_c[2] >= znear
        if a_in:
            out_cam.append(a_c)
            out_world.append(a_w)
        if a_in != b_in:
            t = (znear - a_c[2]) / (b_c[2] - a_c[2])
            out_cam.append(a_c + t * (b_c - a_c))
            out_world.append(a_w + t * (b_w - a_w))
    if not out_cam:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return np.array(out_cam), np.array(out_world)


def _raster_tri(cam, world, fi, intrinsics, depth, face_id, wpos, w, h) -> None:
    """Z-buffer one camera-space triangle with perspective-correct interpolation."""
    zs = cam[:, 2]
    u = intrinsics.fx * cam[:, 0] / zs + intrinsics.cx
    v = intrinsics.fy * cam[:, 1] / zs + intrinsics.cy
    p = np.column_stack([u, v])
    x0 = max(int(math.floor(p[:, 0].min())), 0)
    x1 = min(int(math.ceil(p[:, 0].max())) + 1, w)
    y0 = max(int(math.floor(p[:, 1].min())), 0)
    y1 = min(int(math.ceil(p[:, 1].max())) + 1, h)
    if x1 <= x0 or y1 <= y0:
        return
    area = (p[1, 0] - p[0, 0]) * (p[2, 1] - p[0, 1]) - (p[2, 0] - p[0, 0]) * (p[1, 1] - p[0, 1])
    if abs(area) < 1e-9:
        return
    ys, xs = np.mgrid[y0:y1, x0:x1]
    xf, yf = xs.astype(np.float64), ys.astype(np.float64)
    w0 = ((p[1, 0] - xf) * (p[2, 1] - yf) - (p[2, 0] - xf) * (p[1, 1] - yf)) / area
    w1 = ((p[2, 0] - xf) * (p[0, 1] - yf) - (p[0, 0] - xf) * (p[2, 1] - yf)) / area
    w2 = 1.0 - w0 - w1
    inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
    if not inside.any():
        return
    inv_z = w0 / zs[0] + w1 / zs[1] + w2 / zs[2]
    with np.errstate(divide="ignore", invalid="ignore"):
        tri_depth = np.where(inv_z > 0, 1.0 / inv_z, np.inf)
    sub = depth[y0:y1, x0:x1]
    hit = inside & (tri_depth < sub)
    if not hit.any():
        return
    sub[hit] = tri_depth[hit]
    face_id[y0:y1, x0:x1][hit] = fi
    bw = np.stack([w0[hit] / zs[0], w1[hit] / zs[1], w2[hit] / zs[2]], axis=-1)
    wpos[y0:y1, x0:x1][hit] = (bw * tri_depth[hit][:, None]) @ world


def default_scene(include_ground: bool = True) -> SyntheticScene:
    """Both towers, the nearby landmark buildings and a ground slab.

    Landmark *points* (spire tips, roof corners) keep their registry positions,
    so a pose recovered from this scene can be compared directly against a pose
    recovered from clicked landmarks in a real frame.
    """
    boxes: list[tuple[str, np.ndarray, np.ndarray, float]] = []
    for tower in TOWERS:
        c = tower.enu_center()
        v, f = SyntheticScene._box(
            c[:2],
            tower.footprint_m / 2,
            tower.footprint_m / 2,
            c[2],
            c[2] + tower.roof_height_m,
            tower.rotation_deg,
        )
        boxes.append((tower.id, v, f, 0.58))

    # a squat box under each surviving building-top landmark, so the skyline
    # silhouette has the right shape.  Footprints are invented (the geo
    # workstream owns the real ones); only the tops are registry values.
    footprints = {"spire": 12.0, "bridge_tower": 14.0, "statue": 10.0}
    for lm in LANDMARKS:
        if not lm.existed_on_2001_09_11 or lm.id.startswith(("wtc1", "wtc2")):
            continue
        p = latlon_to_enu(lm.point)
        half = footprints.get(lm.kind, 35.0)
        v, f = SyntheticScene._box(p[:2], half, half, 0.0, max(p[2], 5.0))
        boxes.append((lm.id, v, f, 0.50))

    if include_ground:
        v, f = SyntheticScene._ground_grid()
        boxes.append(("ground", v, f, 0.30))
    return SyntheticScene.from_boxes(boxes)


# --- degradation ---------------------------------------------------------------
def degrade(
    image: np.ndarray,
    scale: float = 1.0,
    blur_sigma_px: float = 1.2,
    noise_sigma: float = 0.02,
    jpeg_quality: int | None = 45,
    interlace_comb: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Make a clean render look like a 2001 broadcast frame.

    ``scale`` < 1 downsamples then upsamples back (resolution loss),
    ``blur_sigma_px`` is lens/deinterlace softness, ``noise_sigma`` is sensor
    and tape noise, ``jpeg_quality`` re-encodes if OpenCV is available, and
    ``interlace_comb`` (0..1) darkens/brightens alternate scan lines to mimic
    a naive deinterlace.  Returns float32 in [0, 1].
    """
    rng = np.random.default_rng(seed)
    img = np.clip(np.asarray(image, dtype=np.float64), 0.0, 1.0)
    h, w = img.shape[:2]
    if blur_sigma_px > 0:
        img = _gaussian_blur(img, blur_sigma_px)
    if scale < 1.0:
        sh, sw = max(2, int(h * scale)), max(2, int(w * scale))
        img = _resize(img, sw, sh)
        img = _resize(img, w, h)
    if interlace_comb > 0:
        img[0::2] = np.clip(img[0::2] * (1.0 + interlace_comb * 0.1), 0, 1)
        img[1::2] = np.clip(img[1::2] * (1.0 - interlace_comb * 0.1), 0, 1)
    if noise_sigma > 0:
        img = img + rng.normal(0.0, noise_sigma, img.shape)
    img = np.clip(img, 0.0, 1.0)
    if jpeg_quality is not None:
        try:
            import cv2

            u8 = (img * 255).astype(np.uint8)
            ok, buf = cv2.imencode(".jpg", u8, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
            if ok:
                img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255.0
        except Exception:  # noqa: BLE001 -- degradation is cosmetic; never fail on it
            pass
    return img.astype(np.float32)


def _gaussian_kernel(sigma: float) -> np.ndarray:
    radius = max(1, int(3.0 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-(x**2) / (2.0 * sigma**2))
    return k / k.sum()


def _gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    k = _gaussian_kernel(sigma)
    pad = len(k) // 2
    out = np.pad(img, ((0, 0), (pad, pad)), mode="edge")
    out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="valid"), 1, out)
    out = np.pad(out, ((pad, pad), (0, 0)), mode="edge")
    return np.apply_along_axis(lambda m: np.convolve(m, k, mode="valid"), 0, out)


def _resize(img: np.ndarray, width: int, height: int) -> np.ndarray:
    """Bilinear resize without SciPy/OpenCV."""
    h, w = img.shape[:2]
    ys = (np.arange(height) + 0.5) * h / height - 0.5
    xs = (np.arange(width) + 0.5) * w / width - 0.5
    y0 = np.clip(np.floor(ys).astype(int), 0, h - 1)
    x0 = np.clip(np.floor(xs).astype(int), 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    wy = np.clip(ys - y0, 0, 1)[:, None]
    wx = np.clip(xs - x0, 0, 1)[None, :]
    top = img[y0][:, x0] * (1 - wx) + img[y0][:, x1] * wx
    bot = img[y1][:, x0] * (1 - wx) + img[y1][:, x1] * wx
    return top * (1 - wy) + bot * wy
