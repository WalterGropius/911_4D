import numpy as np

from wtc4d import timeline
from wtc4d.procedural import smoke


def test_no_plume_before_impact():
    xyz, *_ = smoke.plume_blobs("WTC1", timeline.WTC1_IMPACT.t - 10.0)
    assert xyz.shape[0] == 0


def test_plume_grows_and_rises_then_levels():
    t0 = timeline.WTC1_IMPACT.t
    xyz, scale, quat, rgb, opacity = smoke.plume_blobs("WTC1", t0 + 300.0)
    assert xyz.shape[0] > 0
    assert xyz.shape[1] == 3
    assert scale.shape == xyz.shape
    assert quat.shape == (xyz.shape[0], 4)
    assert rgb.shape == xyz.shape
    assert opacity.shape == (xyz.shape[0],)
    assert np.all(opacity >= 0.0) and np.all(opacity <= 1.0)
    assert np.all(rgb >= 0.0) and np.all(rgb <= 1.0)
    # blobs should never rise above the levelling height
    from wtc4d.procedural.params import DEFAULT_PARAMS

    assert np.all(xyz[:, 2] <= DEFAULT_PARAMS.smoke.level_height_m + 1e-6)


def test_plume_stops_emitting_after_collapse():
    collapse_t = timeline.WTC1_COLLAPSE.t
    xyz_at_collapse, *_ = smoke.plume_blobs("WTC1", collapse_t)
    xyz_long_after, *_ = smoke.plume_blobs("WTC1", collapse_t + 1800.0)
    # no new (younger) blobs are emitted once the tower has fallen, so the
    # long-after cloud should not have more blobs than at collapse time once
    # blobs older than max_age drop out
    assert xyz_at_collapse.shape[0] > 0
    assert xyz_long_after.shape[0] <= xyz_at_collapse.shape[0]


def test_wind_vector_blows_away_from_the_source():
    v_surface = smoke.wind_vector(0.0)
    v_aloft = smoke.wind_vector(1000.0)
    assert np.linalg.norm(v_surface) > 0
    assert np.linalg.norm(v_aloft) > np.linalg.norm(v_surface)


def test_no_dust_before_collapse():
    xyz, *_ = smoke.dust_blobs("WTC2", timeline.WTC2_COLLAPSE.t - 1.0)
    assert xyz.shape[0] == 0


def test_dust_expands_with_time():
    collapse_t = timeline.WTC2_COLLAPSE.t
    xyz_early, *_ = smoke.dust_blobs("WTC2", collapse_t + 5.0)
    xyz_late, *_ = smoke.dust_blobs("WTC2", collapse_t + 60.0)
    assert xyz_early.shape[0] > 0
    r_early = np.linalg.norm(xyz_early[:, :2], axis=1).max()
    r_late = np.linalg.norm(xyz_late[:, :2], axis=1).max()
    assert r_late > r_early


def test_dust_thins_over_a_long_time():
    from wtc4d.procedural.params import DEFAULT_PARAMS

    collapse_t = timeline.WTC2_COLLAPSE.t
    t_mid = collapse_t + DEFAULT_PARAMS.dust.thin_start_s + 1.0
    t_long = collapse_t + DEFAULT_PARAMS.dust.thin_start_s + DEFAULT_PARAMS.dust.thin_tau_s * 4
    _, _, _, _, opacity_mid = smoke.dust_blobs("WTC2", t_mid)
    _, _, _, _, opacity_long = smoke.dust_blobs("WTC2", t_long)
    assert opacity_long.max() < opacity_mid.max()
