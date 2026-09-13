from wtc4d.corpus.harvest import documentary, noaa_usgs


def test_noaa_harvest_default():
    result = noaa_usgs.harvest()
    assert len(result) == 1
    assert result[0].archive == "noaa"


def test_usgs_harvest_variant():
    result = noaa_usgs.harvest("usgs_landsat7_wtc")
    assert {s.archive for s in result} <= {"usgs", "other"}
    assert len(result) == 2


def test_commercial_satellite_variant():
    result = noaa_usgs.harvest("commercial_satellite_ikonos_spot")
    assert len(result) == 2
    assert all(s.license == "proprietary" for s in result)


def test_all_noaa_usgs_sources_have_unique_ids():
    all_srcs = (
        noaa_usgs.harvest("noaa_wtc_aerial_lidar")
        + noaa_usgs.harvest("usgs_landsat7_wtc")
        + noaa_usgs.harvest("commercial_satellite_ikonos_spot")
    )
    ids = [s.id for s in all_srcs]
    assert len(ids) == len(set(ids))


def test_documentary_harvest():
    result = documentary.harvest()
    assert len(result) == 4
    ids = [s.id for s in result]
    assert len(ids) == len(set(ids))
    assert all(s.title for s in result)
