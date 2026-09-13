# `wtc4d.geo` — the static scene prior (Lower Manhattan, 2001-09-11)

This package builds the 2001 scene that camera registration (`camreg`) and
splat initialisation (`recon`) render against: a georeferenced LOD1 city
model, a parametric mesh of the World Trade Center complex, a 50+ point
landmark registry, coarse terrain, and (documented, not committed) the
aftermath debris pile.

Everything is expressed in the project world frame: a local ENU tangent
frame in metres, origin `wtc4d.world.WORLD_ORIGIN`, `Z` = metres above
MSL/NAVD88. See `wtc4d/world.py` for the frame and `wtc4d/schema/` for the
shared pydantic models.

## Quick start

```bash
pip install -e ".[geo,dev]"
wtc4d geo info                       # summarise the committed scene
wtc4d geo preview --from 40.706,-74.010,300 --look-at 0,0,300 --out preview.png
python -m pytest tests/test_geo_*.py
```

```python
from wtc4d.geo import load_scene, load_landmarks, render_view

scene = load_scene()  # trimesh.Scene, world ENU metres
lms = load_landmarks()  # merged wtc4d.world.LANDMARKS + data/geo/landmarks.json
```

## What's committed vs. what needs the network

| file | committed | needs network to rebuild |
|---|---|---|
| `data/geo/wtc_complex.yaml` | yes | no (hand-authored spec) |
| `data/geo/corrections_2001.yaml` | yes | no (hand-authored) |
| `data/geo/landmarks.json` | yes | yes (`wtc4d geo landmarks`) |
| `data/geo/terrain_grid.json` | yes | yes (`wtc4d geo terrain`) |
| `data/geo/buildings_2001.geojson` | yes | yes (`wtc4d geo build`) |
| `data/geo/lower_manhattan_2001.glb` | yes, if < 15 MB | yes (`wtc4d geo build`) |
| aftermath debris-pile raster | no (see below) | yes (`wtc4d geo aftermath`) |

`load_scene()`, `load_landmarks()` and `render_view()` only ever read the
committed files above — no test or import in this package touches the
network. `wtc4d/geo/build.py`, `city.py`, `terrain.py` and `aftermath.py`
contain the fetch/build logic, driven by the `wtc4d geo` CLI.

Downloads are cached outside the repo, in `$WTC4D_GEO_CACHE`
(default `~/.cache/wtc4d/geo`); `wtc4d geo cache` prints the paths in use.

## The World Trade Center complex

`data/geo/wtc_complex.yaml` specifies every structure in the **WTC site
frame**: a 2D frame aligned with the superblock, `u` at azimuth 119.118°
(toward Church St), `v` at azimuth 29.118° (toward Vesey St), origin at
`WORLD_ORIGIN`. `wtc4d.geo.wtc.site_uv_to_enu()` converts to world ENU.

**Grid azimuth.** Derived from the two September 11 Memorial reflecting
pools (OpenStreetMap ways `697722178` / `697722181`), which the National
9/11 Memorial states are built on the original tower footprints. Their eight
sides fit a square to within 0.2 m and agree on bearing to 0.19°, giving
`29.118°` as the WTC/Manhattan grid azimuth and `rotation_deg = -29.118` for
`wtc4d.world.TowerSpec` (a square has 90° symmetry, so `+60.882` is
equivalent).

**Tower centres.** The centroids of the two memorial pools. Cross-checked
against NYC DoITT planimetric footprints in the same block: 31 matched
buildings agree with OSM to a mean offset of 0.8 m E / 0.0 m N with 2.5–3.5 m
scatter, which sets `center_sigma_m`.

**Tower footprint / heights.** 207 ft 2 in (63.14 m) square — NIST NCSTAR
1-1 §2.2. Roof heights 1,368 ft / 1,362 ft above the plaza, WTC1 antenna tip
1,727 ft — NIST NCSTAR 1 and the Wikipedia WTC complex summary (which agree
to the foot). Impact floors (93–99, 77–85) — NIST NCSTAR 1.

**Plaza datum (`PLAZA_ELEVATION_M = 8.23`).** The Port Authority "WTC datum"
is MSL + 300.0 ft; Austin J. Tobin Plaza sat at WTC-datum elevation 327 ft,
i.e. 27 ft = **8.23 m above MSL**. Cross-checked two ways: (1) Church Street
frontage at WTC-datum 321 ft (21 ft above MSL) matches NYC DoITT
`ground_elevation` = 21 ft on that block; (2) a USGS 3DEP sample at the
current memorial plaza gives 4.34 m, matching the 2011 memorial's published
WTC-datum elevation of 313 ft (3.96 m above MSL) to within 0.4 m. Residual
uncertainty ±1.0 m.

**7 WTC (original, 1987).** Irregular trapezoid, north face ≈100 m (329 ft),
south face ≈75 m (247 ft), depth ≈44 m (144 ft), 610 ft / 47 storeys — NIST
NCSTAR 1A §1.1. Placed against OSM centrelines for Vesey Street and West
Broadway (the original building "bordered West Broadway on the east"); this
placement is geometric reconstruction, not survey, hence `sigma_m: 8.0`.

**3/4/5/6 WTC.** Heights from the Wikipedia WTC-complex summary (Marriott
250 ft/22 fl; 4 & 5 WTC 120 ft/9 fl; 6 WTC 110 ft/8 fl). The real footprints
were L/U-shaped; here they are simplified to axis-aligned rectangles filling
the corresponding corner of the superblock (`sigma_m: 10–12`, `approx: true`)
— adequate as occluders and scale references, not as exact footprints.

**World Financial Center 1–4, Verizon, 90 West, Millenium Hilton, One
Liberty Plaza, Deutsche Bank / 130 Liberty St, St Nicholas.** Pulled in
through the general city pipeline (below) with named overrides in
`corrections_2001.yaml`; roof shapes approximated in `wtc4d/geo/meshes.py`
(`pyramid_cap`, `dome_cap`, `stepped_cap`, `barrel_vault` for the Winter
Garden) and selected per-building in future scene-assembly code — currently
all secondary buildings extrude flat-topped; the roof-shape helpers are
available for `camreg`/`recon` code that wants a closer silhouette on those
specific landmarks.

## The city model (`wtc4d/geo/city.py`)

1. Fetch NYC DoITT **BUILDING** (`5zhs-2jue`): footprint, `construction_year`,
   `height_roof` (ft above local ground), `ground_elevation` (ft, NAVD88).
2. Fetch NYC DoITT **BUILDING_HISTORIC** (`ipkp-snf6`) and re-add every record
   demolished after 2001 (and, where the year is known, built by 2001) — this
   restores 130 Liberty St / Deutsche Bank, Fiterman Hall and similar.
   `construction_year == 0` in this dataset means "unknown", which for this
   footprint is overwhelmingly pre-2001 stock, so unknown-year records are
   **kept**, not dropped.
3. Fetch OpenStreetMap footprints for two waterfront strips NYC planimetrics
   don't cover — Jersey City/Hoboken and Brooklyn Heights/DUMBO (see
   `WATERFRONT_BOXES` in `city.py`) — using `height` / `building:height` /
   `building:levels` (× 3.2 m/level) tags; OSM's own `start_date` filters out
   post-2001 construction there.
4. Drop everything with `construction_year > 2001`, apply
   `data/geo/corrections_2001.yaml` (explicit `drop_bins` for post-2001
   structures the dataset dates wrong or leaves unknown — new 7 WTC, the new
   towers on the WTC site, 200 West St — plus attribute `overrides` and
   hand-added `add` entries for structures no dataset has, e.g. St Nicholas
   Church), and layer in the WTC complex from `wtc.py`.
5. Simplify footprints (Douglas-Peucker, default 0.5 m) and extrude from
   terrain to roof.

Run: `wtc4d geo build --out data/geo/`. Produces `buildings_2001.geojson`
(WGS84 footprints + attributes: id, name, height, base/roof elevation, year,
source, sigma, approx-flag) and, unless `--no-glb`, `lower_manhattan_2001.glb`
(only committed if under 15 MB — otherwise it's written to the build cache
and this README's job is to tell you it exists there).

As built for this PR (2.6 km radius, 0.5 m simplification): **12,703
buildings** — 10,715 current DoITT + 537 re-added historic + 1,441 OSM
waterfront + the WTC complex and manual additions — `buildings_2001.geojson`
≈5.9 MB, `lower_manhattan_2001.glb` ≈11.2 MB (`data/geo/` total < 20 MB).

### Known gaps

- Secondary WTC buildings (3–6 WTC) are rectangular stand-ins, not their real
  L/U footprints.
- `construction_year == 0` (unknown) is treated as pre-2001; for a
  fast-changing block this could occasionally be wrong in either direction —
  `corrections_2001.yaml` is the escape hatch when a specific BIN is found to
  be mis-dated.
- OSM coverage stops at the two hand-picked waterfront boxes; Staten Island,
  Governors Island and Midtown are out of scope for this radius.
- Non-tower roof shapes (WFC pyramids/dome, Winter Garden vault, WTC7's
  trapezoid massing) are extruded flat-topped in the assembled scene today;
  the shape helpers in `meshes.py` exist but are not yet wired to specific
  building ids in `scene.py` — a good first extension.

## Landmark registry (`wtc4d/geo/landmarks.py`, `data/geo/landmarks.json`)

`wtc4d geo landmarks` builds:

- **WTC complex** — every footprint corner and the roof-line of all 7
  numbered buildings, plus the WTC1 mast tip (from `wtc.py`, so it moves in
  lockstep with any future update to `wtc_complex.yaml`).
- **Auto-derived DoITT roofs** — the tallest pre-2001 buildings within 7 km,
  named by PLUTO address where available (`roof_<address>`) and by BIN
  otherwise, positioned at the footprint centroid, altitude = DoITT
  `ground_elevation + height_roof`.
- **Manually curated** — landmarks no NYC dataset can supply: Statue of
  Liberty torch, Brooklyn/Manhattan Bridge tower tops, Empire State/Chrysler/
  Citigroup/MetLife spires and roofs, Jersey City waterfront towers, the
  Colgate Clock, St Nicholas Church (destroyed 2001-09-11) — each with an
  inline source. Includes `goldman_30_hudson_jc` and `one_wtc_spire` flagged
  `existed_on_2001_09_11=False` as an explicit negative control.

As built: **91 landmarks**, 89 of them present on 2001-09-11. Every entry has
`sigma_m` and `source`; `point.alt_m` is metres above MSL (the same datum as
world `Z`), so `Landmark.enu()` is directly usable as a PnP object point.
`load_landmarks()` merges this file over the small bootstrap list in
`wtc4d.world.LANDMARKS` (JSON wins on `id`), so the shared module stays valid
without duplicating the registry.

## Terrain (`wtc4d/geo/terrain.py`)

Water is `0.0 m`. Land elevation is a coarse 200 m grid sampled from the
USGS 3DEP 1 m DEM (`wtc4d geo terrain`, committed as
`data/geo/terrain_grid.json`, 27×27 cells / 4.8 KB) with per-zone constants
as a fallback when the grid file is absent (documented in `ZONES`: WTC site
~6 m, Lower Manhattan ~5.5 m, Jersey City ~3 m, Brooklyn ~6 m, each with a
2–4 m sigma). The WTC superblock itself is excluded from the DEM sample and
instead uses `wtc.PLAZA_ELEVATION_M`, because 3DEP reflects the *present-day*
memorial plaza (~4 m lower than the original Austin J. Tobin Plaza), not the
2001 surface.

## Aftermath (E5) surface

Not committed — the source rasters are large and the exact archive layout
changes over time. `wtc4d/geo/aftermath.py` documents the fetch procedure in
full (NOAA/USGS Ground Zero lidar, Sept–Oct 2001, several dated flights; the
NOAA 23 Sept 2001 orthophoto) and provides `rasterise_pile()` /
`write_pile()` to turn a downloaded point cloud into a small (few hundred KB)
2 m height-field JSON that *is* committed once produced, plus `load_pile()`
to read it back. No epoch has been rasterised in this PR; `recon`/`viewer`
code that wants the pile before then should treat
`wtc4d.geo.aftermath.available_pile_dates()` returning `[]` as "not yet
built", not an error.

## Rendering (`wtc4d/geo/render.py`)

`render_view(c2w, intrinsics, scene=None, backend="auto")` returns
`depth` (m, `inf` = miss), `instance_id` (index into `instance_names`, `-1` =
miss) and `normals` (world-frame unit normals) for a camera in the world
frame (`c2w` camera-to-world, OpenCV axes). Three backends:

- `raster` (default fallback): a pure-numpy triangle rasteriser with a
  perspective-correct z-buffer. No system libraries, no GPU — this is what CI
  and the test suite use.
- `pyrender`: OpenGL offscreen (EGL/OSMesa) via the optional `pyrender`
  package, when a context can actually be created (`backend="auto"` tries
  this first, falls back to `raster` on any failure).
- `raycast`: `trimesh` ray casting (needs the `rtree` package, in the `geo`
  dependency group), one ray per pixel — slow, used as an independent
  cross-check for the rasteriser in tests, not for bulk rendering.

## Accuracy budget (summary)

| quantity | value | 1σ | source |
|---|---|---|---|
| WTC1/2 footprint side | 63.14 m | 0.1 m | NIST NCSTAR 1-1 |
| WTC1/2 centres | see `world.py` | 2.5 m | memorial pool centroids vs. DoITT |
| Site grid azimuth | 29.118° | 0.2° | memorial pool side bearings |
| Plaza datum | 8.23 m MSL | 1.0 m | PA WTC datum, cross-checked 2 ways |
| WTC1/2 roof/antenna height | 417.0 / 415.1 / 526.3 m | 1.0 m | NIST NCSTAR 1 |
| WTC7 (1987) footprint | trapezoid, 100×75×44 m | 8 m (placement) | NIST NCSTAR 1A |
| DoITT building roofs | ground+height_roof | ~2 m | NYC DoITT `BUILDING` |
| OSM waterfront roofs | tag or level×3.2 m | 4–8 m | OSM `height`/`levels` |
| Terrain (grid cells) | USGS 3DEP sample | 1–2 m | USGS 3DEP EPQS |
| Terrain (zone fallback) | per-zone constant | 2–4 m | DoITT / 3DEP medians |
| Manual landmarks | see `data/geo/landmarks.json` | 3–10 m | per-entry `source` |

## Regenerating everything

```bash
export WTC4D_GEO_CACHE=~/.cache/wtc4d/geo   # or anywhere outside the repo
wtc4d geo landmarks --out data/geo/landmarks.json
wtc4d geo terrain    --out data/geo/terrain_grid.json
wtc4d geo build      --out data/geo/
```

Each step is independent and idempotent; `--refresh` on `build`/`landmarks`
ignores the download cache. All three need outbound HTTPS to NYC Open Data,
OpenStreetMap/Overpass and USGS; none is required by `load_scene()`,
`load_landmarks()`, `render_view()` or the test suite.
