"""Solar model tests: known sunrise/noon/sunset for NYC on 2001-09-11, and
that the shadow/elevation inversions round-trip through the forward model."""

import math

from wtc4d.sync import solar as S
from wtc4d.timeline import WTC1_COLLAPSE, WTC1_IMPACT, WTC2_COLLAPSE, WTC2_IMPACT, hms


def test_sunrise_noon_sunset_known_values():
    # Published NYC 2001-09-11 civil almanac values (NOAA solar calculator /
    # USNO for this date and latitude) are approximately sunrise 06:33 EDT,
    # solar noon ~12:53 EDT, sunset ~19:11-19:12 EDT. Allow a couple of
    # minutes of slack since published almanac values round to the minute.
    sunrise, sunset = S.sunrise_sunset()
    noon = S.solar_noon()
    assert abs(sunrise - hms(6, 33, 0)) < 120
    assert abs(noon - hms(12, 52, 35)) < 120
    assert abs(sunset - hms(19, 12, 0)) < 120
    assert sunrise < noon < sunset


def test_solar_noon_is_where_azimuth_crosses_180():
    noon = S.solar_noon()
    az_before = S.sun_position(noon - 60).azimuth_deg
    az_after = S.sun_position(noon + 60).azimuth_deg
    assert az_before < 180.0 < az_after


def test_known_anchor_sun_positions_are_physically_sane():
    # Morning of 2001-09-11 in NYC: sun in the east-southeast, climbing.
    for ev in (WTC1_IMPACT, WTC2_IMPACT, WTC2_COLLAPSE, WTC1_COLLAPSE):
        pos = S.sun_position(ev.t)
        assert 90.0 < pos.azimuth_deg < 140.0  # ESE-ish
        assert 15.0 < pos.apparent_elevation_deg < 45.0
    # elevation and azimuth both increase monotonically across the morning
    elevations = [
        S.sun_position(ev.t).apparent_elevation_deg
        for ev in (WTC1_IMPACT, WTC2_IMPACT, WTC2_COLLAPSE, WTC1_COLLAPSE)
    ]
    assert elevations == sorted(elevations)


def test_shadow_azimuth_time_inversion_round_trips():
    t_true = hms(9, 30, 0)
    sun = S.sun_position(t_true)
    est = S.time_from_shadow_azimuth(sun.shadow_azimuth_deg, sigma_deg=0.01)
    assert est is not None
    assert abs(est.t - t_true) < 5.0  # a tight angle sigma buys tight time sigma
    assert est.sigma < 10.0


def test_sun_azimuth_time_inversion_round_trips():
    t_true = hms(10, 15, 0)
    sun = S.sun_position(t_true)
    est = S.time_from_sun_azimuth(sun.azimuth_deg, sigma_deg=0.01)
    assert est is not None
    assert abs(est.t - t_true) < 5.0


def test_elevation_inversion_morning_branch():
    t_true = hms(9, 0, 0)
    sun = S.sun_position(t_true)
    est = S.time_from_sun_elevation(sun.apparent_elevation_deg, sigma_deg=0.05, morning=True)
    assert est is not None
    assert abs(est.t - t_true) < 60.0


def test_sigma_scales_with_angle_uncertainty_and_azimuth_rate():
    t_true = hms(9, 0, 0)
    sun = S.sun_position(t_true)
    tight = S.time_from_shadow_azimuth(sun.shadow_azimuth_deg, sigma_deg=0.1)
    loose = S.time_from_shadow_azimuth(sun.shadow_azimuth_deg, sigma_deg=2.0)
    assert tight is not None and loose is not None
    assert loose.sigma > tight.sigma
    # matches the documented sigma_t = sigma_angle / |d(az)/dt| formula
    rate = abs(S.azimuth_rate_deg_per_s(t_true))
    assert math.isclose(tight.sigma, 0.1 / rate, rel_tol=1e-6)


def test_no_solution_outside_window_returns_none():
    # An azimuth the sun never reaches during the (default daylight) window
    # -- due-north bearing near local noon in the northern hemisphere at this
    # latitude -- must not silently return a wrong time.
    est = S.time_from_sun_azimuth(0.0, t_window=(hms(11, 0, 0), hms(13, 0, 0)))
    assert est is None


def test_bearing_enu_matches_compass_convention():
    # due north
    assert math.isclose(S.bearing_enu((0.0, 0.0), (0.0, 10.0)), 0.0, abs_tol=1e-9)
    # due east
    assert math.isclose(S.bearing_enu((0.0, 0.0), (10.0, 0.0)), 90.0, abs_tol=1e-9)
    # due south
    assert math.isclose(S.bearing_enu((0.0, 0.0), (0.0, -10.0)), 180.0, abs_tol=1e-9)


def test_shadow_annotation_measured_azimuth():
    ann = S.ShadowAnnotation(shot_id="s1", world_azimuth_deg=286.0, sigma_deg=1.0)
    est = S.estimate_from_annotation(ann)
    assert est is not None
    # 286 deg shadow azimuth on the morning of 9/11 corresponds to close to
    # the WTC1 impact time (see epoch_sun_table / test above: shadow az at
    # 08:46:30 is ~286.0 deg).
    assert abs(est.t - WTC1_IMPACT.t) < 120.0


def test_shadow_annotation_ground_edge_uses_camera_callback():
    ann = S.ShadowAnnotation(
        shot_id="s1", kind="ground_shadow_edge", p0=(0.0, 0.0), p1=(1.0, 1.0), sigma_deg=1.0
    )

    def ground_point_of(p):
        # trivial "camera": image (u, v) IS the ENU ground point (east, north)
        return p

    est = S.estimate_from_annotation(ann, ground_point_of)
    assert est is not None
    # segment (0,0)->(1,1) points NE (bearing 45 deg); shadow az near 286 deg
    # is much closer to the WTC1-impact time than to noon, so this is mostly
    # a "it produces *something* sane, not a crash" check
    assert 0.0 <= est.t < hms(24, 0, 0)


def test_shadow_annotation_without_callback_returns_none():
    ann = S.ShadowAnnotation(shot_id="s1", kind="ground_shadow_edge", p0=(0.0, 0.0), p1=(1.0, 1.0))
    assert S.estimate_from_annotation(ann) is None


def test_wind_plume_consistency():
    consistent, n_sigma = S.plume_consistency(160.0)  # matches default NW wind blowing SE
    assert consistent
    inconsistent, n_sigma2 = S.plume_consistency(0.0)  # plume blowing due north: not the wind
    assert not inconsistent
    assert n_sigma2 > n_sigma


def test_epoch_sun_table_has_all_boundaries():
    rows = S.epoch_sun_table()
    labels = {r["label"] for r in rows}
    assert "E0 start" in labels
    assert "E5 end" in labels
    for r in rows:
        assert 0.0 <= r["azimuth_deg"] < 360.0
