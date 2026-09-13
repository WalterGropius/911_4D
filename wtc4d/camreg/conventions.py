"""Coordinate and camera conventions for ``wtc4d.camreg``.

Everything here is plain numpy so that the conventions can be tested and used
without OpenCV, SciPy or any optional dependency.

World frame
-----------
Local ENU tangent frame in **metres** at :data:`wtc4d.world.WORLD_ORIGIN`:
+X east, +Y north, +Z up.

Camera frame
------------
OpenCV / COLMAP: +X right, +Y down, +Z forward (into the scene).  A pose is
stored **camera-to-world** (``c2w``), a 4x4 row-major matrix whose upper-left
3x3 block ``R_cw`` has the camera axes as *columns* expressed in world
coordinates, and whose translation column is the camera centre ``C`` in world
coordinates::

    X_world = R_cw @ X_cam + C
    X_cam   = R_cw.T @ (X_world - C)

OpenCV's ``solvePnP`` returns ``(rvec, tvec)`` of the **world-to-camera**
transform, i.e. ``X_cam = R(rvec) @ X_world + tvec``, so
``R_cw = R(rvec).T`` and ``C = -R(rvec).T @ tvec``.  Use :func:`c2w_from_rt`
and :func:`rt_from_c2w` rather than doing this by hand.

Orientation parameters
----------------------
For search grids and priors it is convenient to describe an orientation by
compass **azimuth** (degrees clockwise from north, so 0 = looking north,
90 = looking east), **elevation** (degrees, positive looking up) and **roll**
(degrees, positive rolls the image clockwise, i.e. rotates the camera
counter-clockwise about its optical axis).  See :func:`pose_from_look` /
:func:`look_from_pose`.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "WORLD_UP",
    "camera_center",
    "c2w_from_rt",
    "c2w_from_w2c",
    "distort",
    "invert_rigid",
    "look_from_pose",
    "opencv_to_opengl",
    "opengl_to_opencv",
    "pose_from_look",
    "project_points",
    "rodrigues",
    "rodrigues_inverse",
    "rt_from_c2w",
    "undistort_normalized",
    "w2c_from_c2w",
]

WORLD_UP = np.array([0.0, 0.0, 1.0])
"""World +Z (up) in the ENU frame."""


# --- rotations ---------------------------------------------------------------
def rodrigues(rvec) -> np.ndarray:
    """Axis-angle (3,) -> rotation matrix (3, 3).  Same as ``cv2.Rodrigues``."""
    r = np.asarray(rvec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(r))
    if theta < 1e-12:
        # second-order expansion is not needed; the first-order term dominates
        k = np.array([[0.0, -r[2], r[1]], [r[2], 0.0, -r[0]], [-r[1], r[0], 0.0]])
        return np.eye(3) + k + 0.5 * k @ k
    axis = r / theta
    k = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return np.eye(3) + math.sin(theta) * k + (1.0 - math.cos(theta)) * (k @ k)


def rodrigues_inverse(R) -> np.ndarray:  # noqa: N803
    """Rotation matrix (3, 3) -> axis-angle (3,), the inverse of :func:`rodrigues`."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)  # noqa: N806
    cos_theta = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    theta = math.acos(cos_theta)
    if theta < 1e-8:
        return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) * 0.5
    if theta > math.pi - 1e-6:
        # near 180 deg: recover the axis from the symmetric part (R + I)/2 = a a^T
        a2 = np.clip(np.diag((R + np.eye(3)) / 2.0), 0.0, None)
        axis = np.sqrt(a2)
        i = int(np.argmax(axis))
        if axis[i] > 0:
            axis = axis * np.sign((R + np.eye(3))[i] / axis[i] if axis[i] else 1.0)
            axis[i] = abs(axis[i])
        axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
        return axis * theta
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return axis * (theta / (2.0 * math.sin(theta)))


# --- rigid transforms --------------------------------------------------------
def invert_rigid(m) -> np.ndarray:
    """Invert a 4x4 rigid transform (exact, via the transpose of its rotation)."""
    m = np.asarray(m, dtype=np.float64).reshape(4, 4)
    out = np.eye(4)
    out[:3, :3] = m[:3, :3].T
    out[:3, 3] = -m[:3, :3].T @ m[:3, 3]
    return out


def w2c_from_c2w(c2w) -> np.ndarray:
    """camera-to-world 4x4 -> world-to-camera 4x4."""
    return invert_rigid(c2w)


def c2w_from_w2c(w2c) -> np.ndarray:
    """world-to-camera 4x4 -> camera-to-world 4x4."""
    return invert_rigid(w2c)


def c2w_from_rt(rvec, tvec) -> np.ndarray:
    """OpenCV ``solvePnP`` output (world-to-camera rvec/tvec) -> ``c2w`` 4x4."""
    R = rodrigues(rvec)  # noqa: N806
    t = np.asarray(tvec, dtype=np.float64).reshape(3)
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def rt_from_c2w(c2w) -> tuple[np.ndarray, np.ndarray]:
    """``c2w`` 4x4 -> OpenCV ``(rvec, tvec)`` of the world-to-camera transform."""
    c2w = np.asarray(c2w, dtype=np.float64).reshape(4, 4)
    R = c2w[:3, :3].T  # noqa: N806
    t = -R @ c2w[:3, 3]
    return rodrigues_inverse(R), t


def camera_center(c2w) -> np.ndarray:
    """World-frame position of the camera centre (metres)."""
    return np.asarray(c2w, dtype=np.float64).reshape(4, 4)[:3, 3].copy()


# --- OpenCV <-> OpenGL --------------------------------------------------------
_CV_GL = np.diag([1.0, -1.0, -1.0, 1.0])
"""Flip of the Y and Z camera axes: OpenCV (+Y down, +Z forward) <-> OpenGL
(+Y up, +Z backward).  It is its own inverse."""


def opencv_to_opengl(c2w) -> np.ndarray:
    """``c2w`` with OpenCV camera axes -> ``c2w`` with OpenGL camera axes."""
    return np.asarray(c2w, dtype=np.float64).reshape(4, 4) @ _CV_GL


def opengl_to_opencv(c2w) -> np.ndarray:
    """``c2w`` with OpenGL camera axes -> ``c2w`` with OpenCV camera axes."""
    return np.asarray(c2w, dtype=np.float64).reshape(4, 4) @ _CV_GL


# --- look-at parameterisation -------------------------------------------------
def forward_from_azel(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """Unit viewing direction in ENU for a compass azimuth and elevation."""
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    return np.array(
        [math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), math.sin(el)],
        dtype=np.float64,
    )


def pose_from_look(
    position, azimuth_deg: float, elevation_deg: float, roll_deg: float = 0.0
) -> np.ndarray:
    """Build a ``c2w`` from a camera centre and compass azimuth/elevation/roll.

    ``position`` is the camera centre in world ENU metres.
    """
    c = np.asarray(position, dtype=np.float64).reshape(3)
    fwd = forward_from_azel(azimuth_deg, elevation_deg)
    right = np.cross(fwd, WORLD_UP)
    n = float(np.linalg.norm(right))
    if n < 1e-9:  # looking straight up/down: fall back to the azimuth plane
        az = math.radians(azimuth_deg)
        right = np.array([math.cos(az), -math.sin(az), 0.0])
        n = 1.0
    right = right / n
    down = np.cross(fwd, right)
    if roll_deg:
        cr, sr = math.cos(math.radians(roll_deg)), math.sin(math.radians(roll_deg))
        right, down = cr * right + sr * down, -sr * right + cr * down
    out = np.eye(4)
    out[:3, 0] = right
    out[:3, 1] = down
    out[:3, 2] = fwd
    out[:3, 3] = c
    return out


def look_at(position, target, roll_deg: float = 0.0) -> np.ndarray:
    """``c2w`` for a camera at ``position`` looking at the world point ``target``."""
    p = np.asarray(position, dtype=np.float64).reshape(3)
    d = np.asarray(target, dtype=np.float64).reshape(3) - p
    horiz = float(math.hypot(d[0], d[1]))
    az = math.degrees(math.atan2(d[0], d[1]))
    el = math.degrees(math.atan2(d[2], horiz))
    return pose_from_look(p, az, el, roll_deg)


def look_from_pose(c2w) -> tuple[float, float, float]:
    """``c2w`` -> ``(azimuth_deg, elevation_deg, roll_deg)``.  Inverse of
    :func:`pose_from_look` (azimuth wrapped to [-180, 180))."""
    c2w = np.asarray(c2w, dtype=np.float64).reshape(4, 4)
    right, fwd = c2w[:3, 0], c2w[:3, 2]
    az = math.degrees(math.atan2(fwd[0], fwd[1]))
    el = math.degrees(math.asin(float(np.clip(fwd[2], -1.0, 1.0))))
    ref_right = np.cross(fwd, WORLD_UP)
    n = float(np.linalg.norm(ref_right))
    if n < 1e-9:
        azr = math.radians(az)
        ref_right = np.array([math.cos(azr), -math.sin(azr), 0.0])
        n = 1.0
    ref_right = ref_right / n
    ref_down = np.cross(fwd, ref_right)
    roll = math.degrees(math.atan2(float(right @ ref_down), float(right @ ref_right)))
    return (az + 180.0) % 360.0 - 180.0, el, roll


# --- projection ---------------------------------------------------------------
def distort(xn, dist) -> np.ndarray:
    """Apply the OpenCV radial/tangential model to normalised coordinates.

    ``xn`` is (N, 2) of ``(x/z, y/z)``; ``dist`` is ``[k1, k2, p1, p2, k3]``
    (any prefix of it, missing terms are zero).
    """
    xn = np.asarray(xn, dtype=np.float64).reshape(-1, 2)
    d = np.zeros(5)
    if dist is not None:
        dd = np.asarray(dist, dtype=np.float64).reshape(-1)
        d[: min(5, dd.size)] = dd[:5]
    k1, k2, p1, p2, k3 = d
    x, y = xn[:, 0], xn[:, 1]
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return np.column_stack([xd, yd])


def undistort_normalized(xd, dist, iters: int = 8) -> np.ndarray:
    """Invert :func:`distort` by fixed-point iteration (N, 2) -> (N, 2)."""
    xd = np.asarray(xd, dtype=np.float64).reshape(-1, 2)
    xn = xd.copy()
    for _ in range(iters):
        err = distort(xn, dist) - xd
        xn = xn - err
    return xn


def project_points(points_world, c2w, intrinsics, dist=None) -> tuple[np.ndarray, np.ndarray]:
    """Project world points into a camera.

    Parameters
    ----------
    points_world : (N, 3) array, world ENU metres.
    c2w : 4x4 camera-to-world matrix.
    intrinsics : :class:`wtc4d.schema.camera.CameraIntrinsics` or a 3x3 K matrix.
    dist : distortion coefficients; defaults to ``intrinsics.dist`` when the
        intrinsics object carries them.

    Returns
    -------
    uv : (N, 2) pixel coordinates (may fall outside the image).
    valid : (N,) bool, True where the point is in front of the camera.
    """
    pts = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    if hasattr(intrinsics, "K"):
        k = np.asarray(intrinsics.K(), dtype=np.float64)
        if dist is None:
            dist = list(getattr(intrinsics, "dist", []) or [])
    else:
        k = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
    w2c = invert_rigid(c2w)
    cam = pts @ w2c[:3, :3].T + w2c[:3, 3]
    z = cam[:, 2]
    valid = z > 1e-6
    zs = np.where(valid, z, 1.0)
    xn = np.column_stack([cam[:, 0] / zs, cam[:, 1] / zs])
    if dist is not None and len(dist):
        xn = distort(xn, dist)
    u = k[0, 0] * xn[:, 0] + k[0, 1] * xn[:, 1] + k[0, 2]
    v = k[1, 1] * xn[:, 1] + k[1, 2]
    return np.column_stack([u, v]), valid
