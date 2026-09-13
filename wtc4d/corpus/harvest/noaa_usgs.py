"""Aerial and remote-sensing sources: NOAA, USGS/NASA, and commercial satellite.

No scriptable API was found for any of these (see ``data/manifests/registry.yaml``
for what was tried and why): NOAA's current emergency-imagery viewer
(``storms.ngs.noaa.gov``) serves post-2005 disasters and 403s on the legacy
2001 mission; USGS EarthExplorer needs an account to pull the actual Landsat
scene; the commercial IKONOS/SPOT imagery is proprietary. So this module is
a small set of hand-curated, individually-verified ``Source`` pointer
records (real URLs, checked 2026-09-13) rather than an API client -- it still
plugs into ``corpus harvest`` like the others, it just has no "refresh" to
run.
"""

from __future__ import annotations

from wtc4d.schema import Source, SourceKind, TimeEstimate
from wtc4d.schema.time import TimeMethod
from wtc4d.timeline import hms

NOAA_AERIAL_LIDAR = Source(
    id="noaa-wtc-ground-zero-2001",
    kind=SourceKind.AERIAL,
    url="https://www.ngs.noaa.gov/RSD/special/sept11/pobarticle.html",
    archive="noaa",
    title="NOAA post-9/11 aerial orthophotography and LiDAR, Ground Zero",
    creator="NOAA National Geodetic Survey (NGS) + EarthData International",
    license="public-domain",
    time_hint=TimeEstimate(
        t=hms(12, 0, 0),  # first NGS-documented flight, 2001-09-23; midday, hour-level guess
        sigma=21600.0,  # +/- 6h: exact flight time not published
        method=TimeMethod.MANUAL,
        evidence="NGS RSD article: RC30 photo mission flown 2001-09-23 at ~3,300 ft; "
        "NGS field support began 2001-09-15; EarthData LiDAR/thermal missions ran "
        "2001-09-15 through 2001-10-23. No published flight timestamp finer than the date.",
    ),
    location_hint_text="Ground Zero / Lower Manhattan, aerial",
    tags=["aerial", "lidar", "orthophoto", "e5_rubble_pile"],
    notes=(
        "Sensors: Leica/LH Systems RC30 film camera (photo), Optech ALTM "
        "(airborne LiDAR), ILRIS-3D (ground-based terrestrial LiDAR). Live "
        "viewer at storms.ngs.noaa.gov covers post-2005 disasters only and "
        "its JSON endpoints 403 for this mission; contacting NOAA NGS/NCEI "
        "directly is the next step to get the actual orthophoto/LAS files."
    ),
)

USGS_LANDSAT7_ETM = Source(
    id="usgs-landsat7-etm-wtc-20010912",
    kind=SourceKind.AERIAL,
    url="https://eros.usgs.gov/media-gallery/image-of-the-week/landsat-records-aftermath-of-historic-world-trade-center-attack",
    archive="usgs",
    title="Landsat 7 ETM+ true-color image, WTC smoke plume",
    creator="USGS EROS / NASA Landsat 7",
    license="public-domain",
    time_hint=TimeEstimate(
        t=hms(11, 30, 0),
        sigma=600.0,
        method=TimeMethod.MANUAL,
        evidence="USGS EROS 'Image of the Week' article: Landsat 7 ETM+ imaged the plume "
        "on 2001-09-12 at roughly 11:30 EDT.",
    ),
    location_hint_text="Lower Manhattan, satellite nadir",
    tags=["satellite", "landsat7", "etm+", "smoke_plume", "e5_rubble_pile"],
    notes=(
        "Approx. WRS-2 path 013 / row 032 (nominal NYC scene; not confirmed "
        "against the specific granule). Downloading the Level-1 product "
        "needs a free USGS EarthExplorer/M2M account, not available here."
    ),
)

NASA_MISR_PLUME = Source(
    id="nasa-misr-terra-wtc-plume-20010912",
    kind=SourceKind.DOCUMENT,
    url="https://www.earthdata.nasa.gov/news/feature-articles/following-world-trade-center-plume",
    archive="other",
    title="MISR/Terra stereo plume-height analysis of the WTC smoke plume",
    creator="NASA JPL (MISR team)",
    license="public-domain",
    time_hint=TimeEstimate(
        t=hms(12, 0, 0),
        sigma=1800.0,
        method=TimeMethod.MANUAL,
        evidence="NASA/JPL: MISR stereo imagery acquired 'about noon' 2001-09-12; "
        "plume height derived as 1.25-1.5 km at four sites.",
    ),
    tags=["satellite", "misr", "terra", "smoke_plume", "plume_height_analysis"],
    notes=(
        "Not raw imagery -- a derived plume-height analysis combining MISR "
        "stereo pairs with ground photographs. Useful as a constraint on "
        "the procedural smoke-plume model, not for camera registration."
    ),
)

COMMERCIAL_SATELLITE = [
    Source(
        id="ikonos-manhattan-20010912",
        kind=SourceKind.AERIAL,
        url="https://landinfo.com/september-11-2001-world-trade-center-recovery-efforts/ikonos-1m-satellite-image-of-lower-manhattan-september-12-2001/",
        archive="other",
        title="IKONOS 1m satellite image of Lower Manhattan",
        creator="Space Imaging (now Maxar/DigitalGlobe)",
        license="proprietary",
        time_hint=TimeEstimate(
            t=hms(11, 43, 0),
            sigma=120.0,
            method=TimeMethod.MANUAL,
            evidence="LAND INFO / press coverage: IKONOS pass at ~11:43 EDT, 2001-09-12.",
        ),
        tags=["satellite", "ikonos", "commercial", "e5_rubble_pile"],
        notes="Proprietary/licensed; pointer only, not for direct ingestion.",
    ),
    Source(
        id="ikonos-manhattan-20010915",
        kind=SourceKind.AERIAL,
        url="https://landinfo.com/september-11-2001-world-trade-center-recovery-efforts/ikonos-1m-satellite-imagery-side-by-side-wtc-comparison-september-15-2001/",
        archive="other",
        title="IKONOS 1m satellite image of Lower Manhattan (post-smoke-clearing)",
        creator="Space Imaging (now Maxar/DigitalGlobe)",
        license="proprietary",
        time_hint=TimeEstimate(
            t=hms(11, 54, 0) + 4 * 86400,
            sigma=120.0,
            method=TimeMethod.MANUAL,
            evidence="LAND INFO / press coverage: IKONOS pass at ~11:54 EDT, 2001-09-15.",
        ),
        tags=["satellite", "ikonos", "commercial", "e5_rubble_pile"],
        notes="Proprietary/licensed; pointer only, not for direct ingestion.",
    ),
]


def harvest(collection: str = "noaa_wtc_aerial_lidar") -> list[Source]:
    """Return the hand-curated pointer records for one registry collection id."""
    if collection == "usgs_landsat7_wtc":
        return [USGS_LANDSAT7_ETM, NASA_MISR_PLUME]
    if collection == "commercial_satellite_ikonos_spot":
        return list(COMMERCIAL_SATELLITE)
    return [NOAA_AERIAL_LIDAR]
