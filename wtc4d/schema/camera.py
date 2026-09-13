from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from wtc4d.schema.geometry import LatLonAlt


class CameraIntrinsics(BaseModel):
    """OpenCV pinhole (+ optional radial/tangential distortion), pixels."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    model: str = "OPENCV"  # PINHOLE | OPENCV | OPENCV_FISHEYE (COLMAP naming)
    dist: list[float] = Field(default_factory=list)  # k1 k2 p1 p2 [k3 ...]

    def K(self):  # noqa: N802
        import numpy as np

        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])

    @property
    def hfov_deg(self) -> float:
        import math

        return math.degrees(2 * math.atan(self.width / (2 * self.fx)))


class CameraPose(BaseModel):
    """Camera-to-world transform of one frame, in the world ENU frame (metres).

    ``c2w`` is a 4x4 row-major matrix flattened to 16 floats.  Camera axes
    follow OpenCV (+X right, +Y down, +Z forward)."""

    shot_id: str
    frame_idx: int
    c2w: list[float] = Field(min_length=16, max_length=16)
    intrinsics: CameraIntrinsics
    method: str = "unknown"  # landmark_pnp | sfm | tracking | manual | ...
    position_sigma_m: float | None = None
    rotation_sigma_deg: float | None = None
    reproj_rmse_px: float | None = None
    n_landmarks: int | None = None
    notes: str = ""

    @field_validator("c2w")
    @classmethod
    def _check_affine(cls, v: list[float]) -> list[float]:
        if [round(x, 6) for x in v[12:16]] != [0.0, 0.0, 0.0, 1.0]:
            raise ValueError("c2w last row must be [0,0,0,1]")
        return v

    def matrix(self):
        import numpy as np

        return np.array(self.c2w, dtype=np.float64).reshape(4, 4)

    @classmethod
    def from_matrix(cls, m, **kw) -> CameraPose:
        import numpy as np

        return cls(c2w=[float(x) for x in np.asarray(m, dtype=np.float64).reshape(16)], **kw)

    def position(self):
        return self.matrix()[:3, 3]


class CameraPrior(BaseModel):
    """A known vantage point (e.g. a named news helicopter, a rooftop camera,
    the Brooklyn Promenade) used as a prior for registration."""

    id: str
    name: str
    location: LatLonAlt
    position_sigma_m: float = 50.0
    moving: bool = False  # helicopters, boats, cars
    notes: str = ""
