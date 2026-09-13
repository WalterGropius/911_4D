"""A tiny CPU preview renderer: numpy painter's-algorithm splatting of
gaussians as soft, alpha-blended ellipses, viewed from a handful of
canonical vantage points. This is for quick visual sanity-checking of the
procedural model, not a differentiable rasterizer -- see ``recon`` for that.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from wtc4d.procedural.gaussians import GaussianCloud

SKY_RGB = (0.75, 0.82, 0.90)


@dataclass(frozen=True)
class Camera:
    eye: np.ndarray  # world ENU metres
    target: np.ndarray
    up: np.ndarray
    fov_deg: float
    width: int
    height: int

    def _basis(self) -> np.ndarray:
        """Rows are the camera's right/up/forward axes in world space."""
        fwd = self.target - self.eye
        fwd = fwd / np.linalg.norm(fwd)
        right = np.cross(fwd, self.up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, fwd)
        return np.stack([right, up, fwd], axis=0)

    def project(self, xyz_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World points -> (pixel xy, depth). Points behind the camera get
        depth <= 0 and should be discarded by the caller."""
        basis = self._basis()
        rel = xyz_world - self.eye[None, :]
        cam = rel @ basis.T  # (N, 3): x=right, y=up, z=forward
        depth = cam[:, 2]
        f = 0.5 * self.width / math.tan(math.radians(self.fov_deg) / 2.0)
        safe_depth = np.clip(depth, 1e-3, None)
        px = self.width / 2.0 + f * cam[:, 0] / safe_depth
        py = self.height / 2.0 - f * cam[:, 1] / safe_depth
        return np.column_stack([px, py]), depth


VIEWPOINTS: dict[str, Callable[[], Camera]] = {}


def _viewpoint(name: str, eye, target=(0.0, 0.0, 220.0), fov_deg=45.0, width=640, height=360):
    VIEWPOINTS[name] = lambda: Camera(
        eye=np.array(eye, dtype=np.float64),
        target=np.array(target, dtype=np.float64),
        up=np.array([0.0, 0.0, 1.0]),
        fov_deg=fov_deg,
        width=width,
        height=height,
    )


# Roughly: Jersey City waterfront (west, across the Hudson), the Brooklyn
# Promenade (south-east, across the East River), and a news helicopter
# orbiting north-east of the complex. Distances/altitudes are approximate,
# chosen to frame the towers, not surveyed vantage points.
_viewpoint("jersey_city", eye=(-1600.0, 60.0, 110.0), fov_deg=40.0)
_viewpoint("brooklyn_promenade", eye=(1250.0, -1450.0, 120.0), fov_deg=38.0)
_viewpoint("helicopter", eye=(650.0, 550.0, 700.0), target=(0.0, 0.0, 200.0), fov_deg=55.0)


def render(
    cloud: GaussianCloud, cam: Camera, sky_rgb: tuple[float, float, float] = SKY_RGB
) -> np.ndarray:
    """Soft-splat ``cloud`` from ``cam`` into an (H, W, 3) float image in [0, 1]."""
    w, h = cam.width, cam.height
    color = np.tile(np.array(sky_rgb, dtype=np.float32), (h, w, 1))
    alpha = np.zeros((h, w), dtype=np.float32)
    if len(cloud) == 0:
        return color

    px, depth = cam.project(cloud.xyz)
    front = depth > 0.1
    idx = np.nonzero(front)[0]
    idx = idx[np.argsort(-depth[idx])]  # far to near

    f = 0.5 * cam.width / math.tan(math.radians(cam.fov_deg) / 2.0)
    r_world = cloud.scale[:, :2].max(axis=1)
    r_screen = np.clip(f * r_world / np.clip(depth, 1e-3, None), 0.6, float(max(w, h)))

    for i in idx:
        cx, cy = px[i]
        rad = float(r_screen[i])
        if cx < -rad or cx > w + rad or cy < -rad or cy > h + rad:
            continue
        x0, x1 = int(max(0, cx - rad)), int(min(w, cx + rad + 1))
        y0, y1 = int(max(0, cy - rad)), int(min(h, cy + rad + 1))
        if x1 <= x0 or y1 <= y0:
            continue
        ys, xs = np.mgrid[y0:y1, x0:x1]
        d2 = (xs - cx) ** 2 + (ys - cy) ** 2
        sigma = max(rad * 0.5, 0.5)
        mask = float(cloud.opacity[i]) * np.exp(-d2 / (2.0 * sigma * sigma))
        patch = color[y0:y1, x0:x1]
        patch[:] = cloud.rgb[i][None, None, :] * mask[..., None] + patch * (1.0 - mask[..., None])
        alpha[y0:y1, x0:x1] = mask + alpha[y0:y1, x0:x1] * (1.0 - mask)

    return np.clip(color, 0.0, 1.0)


def save_png(path: str | Path, image_f: np.ndarray) -> None:
    arr = (np.clip(image_f, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)
