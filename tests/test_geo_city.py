"""Tests for the 2001 filtering logic in wtc4d.geo.city. Synthetic rows only,
no network."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("shapely")

from wtc4d.geo.city import (
    REFERENCE_YEAR,
    _nyc_rows_to_records,
    _osm_elements_to_records,
    _polygon_area,
    _ring_to_enu,
    simplify_ring,
)
from wtc4d.geo.sources import BBox


def _square_lonlat(lon0: float, lat0: float, side_deg: float = 0.001) -> list[list[float]]:
    ring = [
        [lon0, lat0],
        [lon0 + side_deg, lat0],
        [lon0 + side_deg, lat0 + side_deg],
        [lon0, lat0 + side_deg],
        [lon0, lat0],
    ]
    return ring


def _nyc_row(
    lon0, lat0, *, year, height_roof_ft=300.0, ground_elevation_ft=10.0, bin_="1", doitt=None
):
    return {
        "the_geom": {"type": "Polygon", "coordinates": [_square_lonlat(lon0, lat0)]},
        "bin": bin_,
        "doitt_id": doitt if doitt is not None else bin_,
        "base_bbl": "1000000001",
        "construction_year": str(year) if year is not None else "0",
        "height_roof": str(height_roof_ft),
        "ground_elevation": str(ground_elevation_ft),
        "shape_area": "1000",
    }


def test_drops_post_2001_construction():
    rows = [_nyc_row(-74.01, 40.71, year=2010, bin_="post2001")]
    recs = _nyc_rows_to_records(rows, historic=False)
    assert recs == []


def test_keeps_pre_2001_construction():
    rows = [_nyc_row(-74.01, 40.71, year=1975, bin_="pre2001")]
    recs = _nyc_rows_to_records(rows, historic=False)
    assert len(recs) == 1
    assert recs[0]["year"] == 1975
    assert recs[0]["height_m"] == pytest.approx(300.0 * 0.3048, abs=0.01)


def test_keeps_unknown_year_construction():
    """construction_year == 0 (unknown) must not be dropped."""
    rows = [_nyc_row(-74.01, 40.71, year=None, bin_="unknown")]
    recs = _nyc_rows_to_records(rows, historic=False)
    assert len(recs) == 1
    assert recs[0]["year"] is None


def test_historic_readds_only_buildings_standing_on_2001_09_11():
    still_there = _nyc_row(-74.01, 40.71, year=1974, bin_="deutsche_bank")
    still_there["demolition_year"] = "2011"  # demolished well after 2001
    gone_before_2001 = _nyc_row(-74.02, 40.72, year=1915, bin_="old_demo")
    gone_before_2001["demolition_year"] = "1998"
    built_after_2001 = _nyc_row(-74.03, 40.73, year=2003, bin_="future_demo")
    built_after_2001["demolition_year"] = "2009"

    recs = _nyc_rows_to_records([still_there, gone_before_2001, built_after_2001], historic=True)
    ids = {r["id"] for r in recs}
    assert "nyc_h_deutsche_bank" in ids
    assert "nyc_h_old_demo" not in ids
    assert "nyc_h_future_demo" not in ids


def test_osm_start_date_filters_future_buildings():
    data = {
        "elements": [
            {
                "type": "way",
                "id": 1,
                "tags": {"building": "yes", "name": "Old Pier Building"},
                "geometry": [{"lat": p[1], "lon": p[0]} for p in _square_lonlat(-74.03, 40.73)],
            },
            {
                "type": "way",
                "id": 2,
                "tags": {
                    "building": "yes",
                    "name": "New Tower",
                    "start_date": "2015",
                    "height": "150",
                },
                "geometry": [{"lat": p[1], "lon": p[0]} for p in _square_lonlat(-74.05, 40.75)],
            },
        ]
    }
    box = BBox(40.70, -74.10, 40.80, -74.00)
    recs = _osm_elements_to_records(data, box=box)
    names = {r["name"] for r in recs}
    assert "Old Pier Building" in names
    assert "New Tower" not in names


def test_polygon_area_and_ring_conversion():
    ring_ll = _square_lonlat(-74.01, 40.71, side_deg=0.001)
    ring_enu = _ring_to_enu(ring_ll)
    assert ring_enu.shape[0] == 4  # closing point dropped
    area = _polygon_area(ring_enu)
    assert area > 1000.0  # ~0.001 deg ~= 85-111 m per side depending on axis


def test_simplify_ring_preserves_small_polygons():
    ring = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    out = simplify_ring(ring, tol_m=0.5)
    assert len(out) >= 3
    assert _polygon_area(out) == pytest.approx(100.0, rel=0.05)


def test_reference_year_constant():
    assert REFERENCE_YEAR == 2001
