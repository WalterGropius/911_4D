"""Camera convention conversions.

Project convention (see :mod:`wtc4d.world`)
-------------------------------------------
* World: local ENU tangent frame at ``wtc4d.world.WORLD_ORIGIN``, **metres**,
  +X east, +Y north, +Z up.
* Camera axes: **OpenCV** -- +X right, +Y down, +Z forward (into the scene).
* Poses are stored **camera-to-world** (``c2w``), 4x4 row-major, as
  ``wtc4d.schema.camera.CameraPose.c2w``.

Foreign conventions handled here
--------------------------------
* **COLMAP**: same camera axes (OpenCV) but stores **world-to-camera**
  (``w2c``) as a unit quaternion ``(qw, qx, qy, qz)`` plus translation
  ``t``, such that ``x_cam = R(q) @ x_world + t``.
* **nerfstudio / NeRF "blender" style**: camera-to-world but **OpenGL** axes
  -- +X right, +Y **up**, +Z **backward**.  Conversion from OpenCV is a flip
  of the Y and Z columns of the rotation, i.e. ``c2w_gl = c2w_cv @ diag(1, -1, -1, 1)``.
  The flip is an involution, so the same matrix converts back.

Everything here is pure numpy and has no torch dependency, so the
conversions can be unit-tested without the heavy optional stack.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "OPENCV_TO_OPENGL",
    "c2w_to_colmap",
    "c2w_to_opengl",
    "colmap_to_c2w",
    "invert_rigid",
    "look_at_c2w",
    "opengl_to_c2w",
    "quat_to_rotmat",
    "rotmat_to_quat",
]

#: Right-multiplied onto an OpenCV c2w matrix to obtain the OpenGL one (and back).
OPENCV_TO_OPENGL = np.diag([1.0, -1.0, -1.0, 1.0])


def quat_to_rotmat(q) -> np.ndarray:
    """Unit quaternion ``(w, x, y, z)`` -> 3x3 rotation matrix.

    Matches COLMAP's ``qvec2rotmat`` (and the Hamilton convention used by
    the 3DGS ``.ply`` ``rot_0..rot_3`` fields).
    """
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = np.linalg.norm(q)
    if n == 0.0:
        raise ValueError("zero quaternion")
    w, x, y, z = q / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def rotmat_to_quat(r) -> np.ndarray:
    """3x3 rotation matrix -> unit quaternion ``(w, x, y, z)`` with ``w >= 0``.

    Shepperd's method (numerically stable for all rotations).  The sign is
    canonicalised so that round trips through :func:`quat_to_rotmat` are
    bit-comparable across implementations.
    """
    m = np.asarray(r, dtype=np.float64).reshape(3, 3)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    if q[0] < 0.0:
        q = -q
    return q / np.linalg.norm(q)


def invert_rigid(m) -> np.ndarray:
    """Invert a 4x4 rigid transform (rotation + translation) analytically."""
    m = np.asarray(m, dtype=np.float64).reshape(4, 4)
    r = m[:3, :3]
    t = m[:3, 3]
    out = np.eye(4)
    out[:3, :3] = r.T
    out[:3, 3] = -r.T @ t
    return out


def c2w_to_colmap(c2w) -> tuple[np.ndarray, np.ndarray]:
    """OpenCV camera-to-world 4x4 -> COLMAP ``(qvec, tvec)`` of the w2c pose.

    Returns ``(q, t)`` with ``q = (qw, qx, qy, qz)`` and ``x_cam = R(q) @ x_world + t``.
    """
    w2c = invert_rigid(c2w)
    return rotmat_to_quat(w2c[:3, :3]), w2c[:3, 3].copy()


def colmap_to_c2w(qvec, tvec) -> np.ndarray:
    """COLMAP ``(qvec, tvec)`` (world-to-camera) -> OpenCV camera-to-world 4x4."""
    w2c = np.eye(4)
    w2c[:3, :3] = quat_to_rotmat(qvec)
    w2c[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return invert_rigid(w2c)


def c2w_to_opengl(c2w) -> np.ndarray:
    """OpenCV c2w -> OpenGL c2w (nerfstudio ``transform_matrix``)."""
    return np.asarray(c2w, dtype=np.float64).reshape(4, 4) @ OPENCV_TO_OPENGL


def opengl_to_c2w(m) -> np.ndarray:
    """OpenGL c2w (nerfstudio ``transform_matrix``) -> OpenCV c2w."""
    return np.asarray(m, dtype=np.float64).reshape(4, 4) @ OPENCV_TO_OPENGL


def look_at_c2w(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Build an OpenCV camera-to-world matrix looking from ``eye`` at ``target``.

    ``up`` defaults to world +Z (ENU up).  Handy for synthetic tests and for
    placing viewer cameras; not used by registration.
    """
    eye = np.asarray(eye, dtype=np.float64).reshape(3)
    target = np.asarray(target, dtype=np.float64).reshape(3)
    up = np.asarray(up, dtype=np.float64).reshape(3)

    forward = target - eye
    n = np.linalg.norm(forward)
    if n < 1e-12:
        raise ValueError("eye and target coincide")
    forward = forward / n

    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-9:  # looking straight along `up`
        up = np.array([0.0, 1.0, 0.0]) if abs(forward[2]) > 0.9 else np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)  # OpenCV +Y points down

    c2w = np.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = down
    c2w[:3, 2] = forward
    c2w[:3, 3] = eye
    return c2w
