import numpy as np

from wtc4d import world
from wtc4d.schema.geometry import LatLonAlt


def test_enu_roundtrip():
    p = LatLonAlt(lat=40.74844, lon=-73.98565, alt_m=443.0)
    enu = world.latlon_to_enu(p)
    back = world.enu_to_latlon(enu)
    assert abs(back.lat - p.lat) < 1e-8
    assert abs(back.lon - p.lon) < 1e-8
    assert abs(back.alt_m - p.alt_m) < 1e-3


def test_towers_are_close_and_tall():
    c1 = world.WTC1.enu_center()
    c2 = world.WTC2.enu_center()
    d = np.linalg.norm(c1[:2] - c2[:2])
    # centres of the two footprints were ~ 150-200 m apart
    assert 100 < d < 250
    corners = world.WTC1.box_corners()
    assert corners.shape == (8, 3)
    assert abs(corners[4:, 2].mean() - corners[:4, 2].mean() - world.WTC1.roof_height_m) < 1e-6


def test_empire_state_is_north_east():
    enu = world.LANDMARKS_BY_ID["empire_state_spire"].enu()
    assert enu[0] > 1000 and enu[1] > 3000  # east and well north
    assert 4000 < np.linalg.norm(enu[:2]) < 6000
