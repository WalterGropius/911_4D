# wtc4d.procedural

A parametric, time-dependent model of the WTC complex on 2001-09-11: given
any project time `t` (seconds since 00:00 EDT, see `wtc4d.timeline`), it
samples a coherent 3D gaussian scene -- towers, impact damage, fire, smoke,
post-collapse dust, rubble, and a placeholder skyline/ground.

**This is a kinematic sketch, not a simulation.** Every event *time* (impact,
collapse initiation) comes straight from `wtc4d.timeline`, which is sourced
to NIST NCSTAR 1 / 1A. Everything about *how* those events unfold in between
-- collapse duration, WTC2's tilt angle, plume turbulence, dust-front speed
-- is either a widely-cited approximate figure from published video analyses
or an explicit artistic default chosen for visual coherence. `params.yaml`
labels every number one way or the other; nothing here should be read as a
structural or fluid-dynamics result.

## Why this exists

1. **A complete timeline from minute one.** Real footage only covers dense
   windows (the impacts, the collapses); the procedural layer gives the
   viewer *something* coherent to look at at every `t`, including the long
   stretches where no synchronised multi-view footage exists yet.
2. **Init + prior for `recon`.** The learned gaussian splats can be
   initialised from (and regularised toward) the procedural cloud instead of
   from scratch or from SfM point clouds that don't exist for this footage.
3. **A physical sketch for investigators.** Every parameter is named, sourced
   where possible, and adjustable -- so a claim like "the plume should have
   drifted this way by this time" can be checked and tuned against footage,
   not taken on faith.

## Model

### Towers (`towers.py`)

`tower_state(tower_id, t) -> TowerState` is a small state machine per tower
(`"WTC1"` / `"WTC2"`), phases `intact -> burning -> collapsing -> gone`:

- **Impact** (`WTC1_IMPACT` / `WTC2_IMPACT` from `wtc4d.timeline`): opens a
  `DamageZone` -- a rectangular gash on the correct face and floor range
  (WTC1: north face, floors 93-99, centred; WTC2: south face, floors 77-85,
  offset toward the south-east corner, matching where UA175 struck).
- **Burning**: `fire_extent` grows continuously (smoothstep) from the impact
  time, controlling how far the fire glow spreads along the facade around
  the gash. It keeps growing smoothly through the collapse (there is no
  hard reset at collapse initiation) so nothing snaps discontinuously.
- **Collapsing**: `collapse_progress` ramps 0->1 over `collapse_duration_s`
  (per-tower, `params.yaml`). WTC2 additionally tilts toward the
  east/south-east (`tilt_deg`, `tilt_azimuth_deg`) over an initial
  `tilt_phase_s` before the top-down descent dominates, matching the
  widely-filmed initial lean. WTC1 has no tilt; instead its antenna
  (`antenna_present`) drops in the first `antenna_drop_lead_s` of the
  collapse, slightly before the roofline itself visibly falls.
- **Gone**: a rubble mound (`rubble_height_m`, "a few storeys", artistic) at
  the footprint. WTC1 additionally leaves a standing north-face facade
  remnant (`remnant_present`, `remnant_height_m`) at the northwest corner,
  matching the widely-photographed structure that stood for weeks.

`wtc7_state(t)` and `three_wtc_state(t)` give a coarser
intact/damaged/collapsing/gone state for two background buildings the `geo`
workstream will eventually model properly: 7 WTC (damaged by WTC1's debris,
collapses at 17:20:52 per NCSTAR 1A) and 3 WTC / the Marriott (crushed
during the tower collapses).

### Smoke and dust (`smoke.py`)

- `plume_blobs(tower_id, t)`: emits a blob every `emit_interval_s` from
  impact until the tower falls. Each blob rises at a constant buoyant rate
  until it levels off at `level_height_m`, drifts horizontally at the wind
  speed/direction for *its current height* (a constant-velocity
  approximation per blob, not a trajectory integral -- good enough for a
  sketch), grows with `sqrt(age)` (turbulent spreading), fades in opacity
  (dilution) and colour (dark, fuel-rich -> lighter grey).
- `wind_vector(z)`: linear interpolation between a documented surface wind
  (LGA/EWR METAR, ~NW at 10-15 kt) and a stronger, more-westerly "aloft"
  value (no sounding on hand; within the commonly described range).
- `dust_blobs(tower_id, t)`: after each collapse, a ground-hugging cloud
  whose front radius grows as `rate * sqrt(age)` (diffusive), biased to
  travel further along the +/-X and +/-Y axes to loosely mimic channeling
  down the Manhattan street grid ("street canyons"), thinning after
  `thin_start_s`.

### Gaussians (`gaussians.py`)

`sample(t, params=DEFAULT_PARAMS, seed=0) -> GaussianCloud` combines:

| layer | source |
|---|---|
| `ground` | a coarse flat grid |
| `skyline` | placeholder boxes from `wtc4d.world.LANDMARKS` heights (or `wtc4d.geo.BUILDINGS_2001` if that package exists -- imported lazily, so this package works standalone today and picks up the real model later without a code change) |
| `towers` | facade grid over the two towers' `TowerSpec.box_corners()`, minus the impact hole, following `top_height_m` as they collapse |
| `damage` | a small jittered debris cluster filling the impact gash |
| `fire` | an emissive-tinted subset of the facade near the gash, sized by `fire_extent` |
| `smoke` | `smoke.plume_blobs` for both towers |
| `dust` | `smoke.dust_blobs` for both towers |
| `rubble` | the post-collapse mound (+ WTC1's remnant) |

`GaussianCloud` is plain numpy (`xyz`, `scale`, `quat`, `rgb`, `opacity`,
`layer`) so it's cheap to slice, filter (`cloud.layer_mask("smoke")`) or feed
into a renderer. The only randomness (impact-hole and rubble jitter) is
drawn from `numpy.random.default_rng(seed)`, so `sample(t, seed=...)` is
deterministic.

Export:
- `GaussianCloud.save_ply` / `.load_ply`: the standard 3DGS PLY convention
  (`x y z nx ny nz f_dc_0..2 opacity scale_0..2 rot_0..3`, DC-term colour,
  log-scale, logit-opacity). `layer` is not part of that schema and is
  dropped on PLY export.
- `GaussianCloud.save_npz` / `.load_npz`: a compact, exact round trip
  including `layer`, for internal / pipeline use.

### Preview renderer (`render.py`)

A tiny CPU-only "painter's algorithm": projects gaussians with a pinhole
camera, sorts back-to-front, and alpha-composites each as a soft
(Gaussian-falloff) ellipse. Three canonical viewpoints are defined in
`VIEWPOINTS`: `jersey_city`, `brooklyn_promenade`, `helicopter`. This is for
quick visual sanity checks, not a differentiable rasterizer (see `recon`).

## CLI

```bash
# one frame
wtc4d procedural export --t 09:30:00 --out out.ply

# a time series + a SceneManifest (wtc4d.schema.scene) pointing at each frame
wtc4d procedural export --sequence 08:40:00 10:35:00 --step 30 --out-dir out/

# a quick PNG preview from a canonical viewpoint
wtc4d procedural preview --t 09:30:00 --out preview.png --viewpoint jersey_city
```

`--format npz` switches the export format; `--params path/to/params.yaml`
overrides the packaged defaults; `--seed` controls the jitter RNG.

See `docs/img/procedural_*.png` for sample previews and
`data/scenes/procedural_manifest.example.json` for a sample manifest.

## Parameters and sources

Every physical/artistic number lives in `params.yaml`, with the source (or
"artistic") noted next to it: METAR-derived wind, NIST NCSTAR 1/1A event
times (via `wtc4d.timeline`, not duplicated here), published collapse
durations and tilt estimates, and clearly-labelled visual defaults (plume
turbulence constants, rubble/remnant heights, debris colours).

## Using this from `recon`

- **Initialisation**: seed a training run's gaussians from
  `gaussians.sample(t, seed=...)` at the epoch's representative time instead
  of random initialisation or an SfM point cloud that doesn't exist for most
  of this footage.
- **Prior / regulariser**: for epochs or regions with weak camera coverage,
  pull learned gaussians softly toward the procedural cloud's positions,
  colours and opacities (e.g. an L2 term in early training iterations,
  annealed out as real observations dominate).
- **Layer masks**: `cloud.layer_mask("smoke")` / `"dust"` etc. let `recon`
  treat dynamic layers differently from static ones (e.g. looser priors,
  different densification behaviour) without re-deriving the classification.

## Limitations

- Tower/skyline geometry uses `wtc4d.world`'s approximate footprints and
  landmark heights (marked `approx` there); this package does not attempt to
  improve on those, and picks up `wtc4d.geo`'s real Lower Manhattan model
  automatically once that package exists (see `_skyline` in `gaussians.py`).
- Collapse kinematics, plume turbulence and dust-front speed are simple
  closed-form approximations, not a structural or CFD simulation -- do not
  use this for anything requiring engineering accuracy.
- 3 WTC's location/height and 7 WTC's footprint are approximate placeholders
  pending survey data from `geo`.
- The wind model is a two-level (surface/aloft) linear interpolation, not an
  actual sounding profile.
