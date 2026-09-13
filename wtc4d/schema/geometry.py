from __future__ import annotations

from pydantic import BaseModel, Field


class LatLonAlt(BaseModel):
    """WGS84 geodetic position. Altitude is metres above the ellipsoid
    (treated as ~MSL for this project; the geo workstream fixes the datum)."""

    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    alt_m: float = 0.0
