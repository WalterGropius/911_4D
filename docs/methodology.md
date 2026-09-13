# Methodology (for investigators)

This document explains how a claim in the 911_4D reconstruction — "this
camera was here, at this time, and the tower looked like this" — is built
and how to check it. It is written for someone who wants to *use* the
reconstruction as evidence, not just view it, so it is precise about
conventions, units, and where uncertainty lives. For narrative/prose
context on sources, see `docs/sources_overview.md`; for what the project
will and won't render, see `docs/ethics.md`.

## Pipeline, end to end

```
 corpus  ──► sources.jsonl, shots.jsonl, frames on volume
    │
    ├──► sync    ──► TimeEstimate per shot / frame (absolute time + sigma)
    │
    ├──► camreg  ──► CameraPose per frame (uses geo prior + landmarks)
    │                    ▲
 geo ───────────────────┘  static Lower-Manhattan-2001 model (+ rubble LiDAR for E5)
    │
    ├──► procedural ──► parametric towers/smoke/collapse -> gaussians for ANY time (baseline + init)
    │
    └──► recon    ──► static splat per epoch (E0..E5), then 4D splats for dynamic windows
                          │
                          └──► SceneManifest ──► web viewer (timeline scrubber, camera frusta, provenance)
```

(This is the same diagram as `PLAN.md` §2 — it is the project's single
data-flow contract, reproduced here with the investigator's question in
mind: *what supports what*.)

1. **`corpus`** (`wtc4d/corpus`, `wtc4d.schema.corpus`) finds and catalogues
   footage: one `Source` per retrievable item (a video, a photo, a
   dataset), split into `Shot`s (contiguous single-camera segments) and
   individual `Frame`s. Nothing here is geometric or temporal yet — this
   is inventory.
2. **`sync`** (`wtc4d/sync`) assigns each shot/frame a `TimeEstimate`:
   an absolute time in **project seconds** (see below) with a 1-sigma
   uncertainty and a `TimeMethod` (how it was derived).
3. **`geo`** (`wtc4d/geo`, plus `wtc4d/world.py`) is not a pipeline stage
   so much as a shared prior: a georeferenced static model of Lower
   Manhattan as it stood on the morning of 2001-09-11 — tower geometry,
   a landmark registry, and (for epoch E5) a rubble-pile surface from
   post-collapse LiDAR. Every other stage measures *against* this model.
4. **`camreg`** (`wtc4d/camreg`) registers each frame's camera: intrinsics
   (`CameraIntrinsics`) and a camera-to-world pose (`CameraPose`), found by
   matching visible landmarks from the `geo` registry against the frame
   (PnP), not by classical structure-from-motion between frames (see
   "Why not just feature-match between videos?" in `docs/faq.md` for why).
5. **`procedural`** (`wtc4d/procedural`) is a parametric, physics-informed
   model of the towers, fire, smoke plume, and collapse kinematics that
   can be evaluated at *any* project time `t`, producing a gaussian
   initialization/baseline. It exists to (a) fill time and viewpoints the
   footage doesn't densely cover, and (b) initialize `recon`'s optimizer
   with a physically plausible starting point instead of noise.
6. **`recon`** (`wtc4d/recon`) trains actual gaussian splats: a static
   splat per epoch where the scene is approximately constant, and true 4D
   (time-varying) splats for the densely-filmed `DynamicWindow`s (the two
   impacts and two collapses), using the registered cameras from `camreg`
   and the `geo`/`procedural` output as structural prior and
   initialization.
7. The **`SceneManifest`** (`wtc4d.schema.scene`) ties it together for the
   **viewer** (`web/`): assets (`SplatAsset`, tagged by epoch/kind/layer),
   registered camera viewpoints (`CameraRef`, each pointing back at a
   `shot_id`/`frame_idx`), and timeline events (`TimelineEvent`, mirroring
   `wtc4d.timeline.EVENTS`).

## Conventions (exact, not approximate)

These are the units and frames every workstream's output must use. They
are documented in code at `wtc4d/world.py` and `wtc4d/timeline.py`
docstrings; this section restates them for a reader who wants to consume
the data, not write to it.

- **Space:** a local **East-North-Up (ENU)** tangent frame, in **metres**,
  origin `wtc4d.world.WORLD_ORIGIN` (`lat=40.71120, lon=-74.01320`,
  roughly the midpoint between the two towers at plaza level). +X = east,
  +Y = north, +Z = up. Every spatial quantity in the project — camera
  poses, gaussians, meshes, landmarks — lives in this frame. Convert with
  `wtc4d.world.latlon_to_enu` / `enu_to_latlon`.
- **Camera convention:** OpenCV/COLMAP pinhole — camera +X right, +Y down,
  +Z forward. `CameraPose.c2w` is **camera-to-world**, a 4x4 row-major
  matrix flattened to 16 floats (`wtc4d.schema.camera.CameraPose`); a
  world-space point is `c2w @ [x_cam, y_cam, z_cam, 1]`. The last row is
  enforced to be `[0,0,0,1]` by a `field_validator`. Intrinsics
  (`CameraIntrinsics`) follow the OpenCV pinhole (+ optional radial/
  tangential distortion) model with `model` naming COLMAP-style
  (`PINHOLE | OPENCV | OPENCV_FISHEYE`).
- **Time:** every time value in the project is **seconds since local
  midnight, 2001-09-11 00:00:00 EDT (UTC-4)**, a plain float
  (`wtc4d.timeline`, "project seconds"). `hms(8, 46, 30)` gives 31590.0;
  `fmt_local(t)` gives it back as `"08:46:30"`. Use `to_utc`/`to_local`
  when an absolute timezone-aware datetime is needed. This convention
  exists so times are both human-readable at a glance and a plain sortable
  scalar — don't reinvent a second time representation in a new
  workstream.
- **Event anchors:** `wtc4d.timeline.EVENTS` gives the canonical anchor
  times (impacts, collapses) as `Event` objects: `t`, `sigma` (seconds,
  1-sigma), and `source`. These follow **NIST NCSTAR 1 (2005)** where
  available; where the 9/11 Commission Report gives a different value
  (impacts: 08:46:40 / 09:03:11 vs. NIST's 08:46:30 / 09:02:59), *both*
  are recorded in `Event.source` rather than silently picking one — this
  is the project's general policy for disagreement between credible
  sources (see `docs/ethics.md`, "Contested claims").
- **Epochs (`E0`-`E5`):** `wtc4d.timeline.EPOCHS` partitions the day into
  windows where the *static* scene is approximately constant (both towers
  intact / WTC1 burning / both burning / WTC2 down / both down / rubble).
  Static splats are trained per epoch; `epoch_at(t)` maps a time to its
  epoch.
- **Dynamic windows:** `wtc4d.timeline.DYNAMIC_WINDOWS` marks the three
  short, densely-covered windows (second impact, WTC2 collapse, WTC1
  collapse) that are candidates for true 4D (time-varying) splats rather
  than a single static epoch splat.

## Uncertainty: representation and propagation

Every measurement in this project carries an explicit error bar. There is
no field anywhere in `wtc4d.schema` that represents a number without a
place to say how sure we are of it.

- **Time:** `wtc4d.schema.time.TimeEstimate.sigma` (seconds, 1-sigma),
  alongside `method` (`TimeMethod`: `broadcast_metadata`, `onscreen_clock`,
  `event_anchor`, `audio_xcorr`, `visual_xcorr`, `solar_shadow`, `manual`,
  `unknown` — ordered roughly by typical accuracy) and `evidence` (a
  human-readable justification). Critically, `derived_from` lists the ids
  of other estimates a given estimate depends on — e.g. a shot timed by
  audio cross-correlation against an already-timed reference clip should
  list that reference's id, so its sigma is not independent of the
  reference's. When combining estimates, the sigma of a dependent estimate
  is never smaller than what its `derived_from` chain supports.
- **Pose:** `wtc4d.schema.camera.CameraPose` carries `position_sigma_m`,
  `rotation_sigma_deg`, `reproj_rmse_px` (reprojection error against the
  landmarks used), and `n_landmarks` (how many landmarks constrained the
  solve). A pose from 3 marginal landmarks and a pose from 12 well-spread
  ones can have the same point estimate and very different confidence —
  always check `n_landmarks` and `reproj_rmse_px`, not just whether a pose
  exists.
- **Geometry:** `wtc4d.world.Landmark.approx` flags whether a landmark's
  coordinates are a placeholder (most of the initial registry) or a
  verified survey value; `existed_on_2001_09_11` flags landmarks that must
  never appear in pre-collapse renders (e.g. `goldman_30_hudson_jc`, built
  2004). `TowerSpec` fields have similar approx/verify notes in
  `wtc4d/world.py` docstrings and comments — the geo workstream is
  responsible for replacing placeholders with surveyed values (NIST
  structural drawings, NYC DoITT planimetrics, memorial pool footprints).
- **Manifest-level:** `wtc4d.schema.scene.TimelineEvent.sigma` and
  `SplatAsset` metadata carry enough of this through to the viewer that an
  investigator can see, at the point of consumption, how solid a given
  displayed element is — not just at the pipeline stage that produced it.
- **Propagation rule of thumb:** uncertainty only shrinks with independent
  corroboration (a second, unrelated method agreeing), never by
  averaging correlated estimates as if they were independent. If a time or
  pose is derived from another estimate (`derived_from`, or a pose
  bootstrapped from a nearby frame's pose), its uncertainty is at least as
  large as what it depends on, plus the new method's own error.

## Tracing a viewer claim back to source frames

Every element the viewer can show is designed to be traceable:

1. A rendered splat (`SplatAsset`) carries `epoch_id` and/or a time range
   (`t_start`/`t_end`) and a `kind`/`layer` (static, dynamic, procedural;
   scene, smoke, debris, towers, ground). A `kind="procedural"` asset is
   the kinematic sketch (see below) — trace it to `wtc4d.procedural`'s
   model and its NIST-derived structural parameters, not to specific
   footage.
2. A registered camera (`CameraRef` in `wtc4d.schema.scene`) links a
   viewer-visible frustum back to a `shot_id` and `frame_idx`, which in
   turn resolve (via `corpus`'s `sources.jsonl`/`shots.jsonl`) to a
   `Source` — its `url`, `archive`, `creator`, and `license`. The viewer's
   "jump to source frame" feature (see `PLAN.md` §4, workstream 8) is
   exactly this link surfaced in the UI, alongside the frame's `CameraPose`
   and its `TimeEstimate`.
3. For a `kind="dynamic"`/`kind="static"` splat asset trained by `recon`,
   the frames and cameras that trained it are the same registered
   `CameraRef`s active in its time range — i.e. "which frames support this
   surface" is answerable as "which registered cameras had this region in
   view during this asset's `t_start`/`t_end`," pending `recon` publishing
   a more direct per-gaussian provenance link (planned for Phase 3, see
   `PLAN.md` §6).
4. A defensible claim always cites, at minimum: the `Source.id`(s)
   involved, the `TimeEstimate` (value + sigma + method) for the relevant
   moment, and — if a spatial claim is being made — the `CameraPose`
   (value + sigma + `n_landmarks`) for the relevant camera. A number
   without these is not yet a claim the project can back.

## Registering your own footage

The `camreg` workstream owns the registration workflow; what follows is
the level of detail in `PLAN.md` — expect the concrete tool/CLI to live at
`wtc4d camreg ...` (see `wtc4d/cli.py`, which mounts each workstream's
`app: typer.Typer` automatically). The workflow, at a high level:

1. **Ingest** your footage as a `Source` (`corpus`'s job, or a manual
   entry if you're contributing a single clip) — this records `url`,
   `archive`, `creator`, `license`, and other provenance fields even
   before any geometry is known.
2. **Split into shots** and pick the frame(s) you want registered.
3. **Annotate landmarks:** for each frame, identify 4+ visible points that
   match entries in `wtc4d.world.LANDMARKS` (or a locally-added landmark —
   the `geo` workstream owns extending this registry). More, well-spread
   landmarks give a better-conditioned solve; a handful of landmarks
   clustered in one corner of the frame will produce a pose with a large
   `rotation_sigma_deg` even if `reproj_rmse_px` looks fine.
4. **Solve pose via PnP:** given 2D landmark pixel positions + their known
   3D `LatLonAlt` (converted to world ENU via `wtc4d.world.latlon_to_enu`)
   + assumed/estimated intrinsics, `camreg` solves for the camera-to-world
   pose (`CameraPose.c2w`) using a Perspective-n-Point solver, then
   reports `reproj_rmse_px` and (from the solve's covariance/residuals)
   `position_sigma_m`/`rotation_sigma_deg`.
5. **Render-and-match refinement:** the pose is refined by rendering the
   known `geo` prior from the estimated pose and comparing against the
   frame, rather than continuing to rely on hand-picked points alone —
   this is what lets registration work on wide, weak-baseline shots where
   classical feature matching between videos fails.
6. **Track across the shot:** once one frame in a shot is registered,
   subsequent frames are tracked (not re-solved from scratch each frame),
   with dynamic-region masks (smoke, fire, moving debris, people) excluded
   from the tracking signal.
7. **Attach a `CameraPrior`** if the vantage point is a known, named one
   (a specific news helicopter, a rooftop camera, the Brooklyn Promenade)
   — this gives future registrations of the same vantage a strong starting
   prior (`CameraPrior.position_sigma_m`, `moving` for helicopters/boats).

If you are proposing to contribute footage you personally filmed, start
with `docs/faq.md` ("How do I submit footage I filmed?") and
`docs/ethics.md` for how footage containing people is handled.

## Known limitations and failure modes

- **SD source material.** Most 2001 footage is 480i, VHS/DV-sourced,
  heavily re-encoded from broadcast. Expect weak features, motion blur,
  and interlacing artifacts; deinterlacing/super-resolution are for
  *viewing* only and are never used as a substitute for the original frame
  when solving pose or time.
- **Weak baselines.** Most cameras are far from the towers (New Jersey,
  Brooklyn, Midtown, helicopters), giving tiny parallax on the towers
  themselves — this is why registration uses a known 3D prior + PnP on
  landmarks rather than classical structure-from-motion, and why far-away
  shots will generally have larger `position_sigma_m` along the
  camera-to-tower axis than across it.
- **Smoke, fire, and dust.** These dominate large regions of frame for
  much of the day and change every frame; they break cross-video feature
  matching and are explicitly masked out of the static registration and
  reconstruction signal, then modeled separately as time-tagged gaussians
  (procedural plume model, or `recon`'s dynamic layer).
- **Timing ambiguity.** Not every shot has a strong `TimeMethod`; a shot
  timed only by `visual_xcorr` against another estimate's plume shape, or
  by `solar_shadow`, will have a larger, and possibly correlated
  (`derived_from`-chained), sigma than one with an on-screen broadcast
  clock. Always check `TimeEstimate.method` and `sigma`, not just presence
  of a time.
- **The procedural layer is a kinematic sketch, not evidence.** `wtc4d.procedural`'s
  output (`kind="procedural"` in `SplatAsset`) is a parametric,
  physics-informed model used to fill time/viewpoint gaps and initialize
  `recon`'s optimizer — it is not derived from footage of the specific
  moment it depicts, and must be visually and provenance-wise
  distinguished from evidence-based (`kind="static"`/`"dynamic"`) splats
  everywhere it appears: in the viewer UI, in dataset releases, and in any
  claim made from it. A defensible investigative claim never rests on the
  procedural layer alone.
- **Camera priors and moving cameras.** Helicopter and boat-mounted cameras
  (`CameraPrior.moving=True`) have inherently larger position uncertainty
  and no fixed prior to anchor repeat registrations; each frame is closer
  to an independent solve.

## Checklist for a defensible claim

- [ ] Identify the specific `Source.id`(s) the claim rests on, and check
      their `license`/`archive`/`creator` are recorded.
- [ ] Identify the `TimeEstimate` for the relevant moment: value, `sigma`,
      `method`, and whether it's independent or `derived_from` another
      estimate whose own error it inherits.
- [ ] If spatial: identify the `CameraPose` involved, its `position_sigma_m`/
      `rotation_sigma_deg`, `reproj_rmse_px`, and `n_landmarks`.
- [ ] Confirm whether the visual you're pointing at is evidence-based
      (`kind="static"`/`"dynamic"`, trained from registered cameras) or
      procedural (`kind="procedural"`, a kinematic sketch) — never present
      the latter as if it were the former.
- [ ] If two credible sources disagree on the underlying fact (as NIST and
      the 9/11 Commission do for the impact times), say so and cite both,
      rather than silently picking one.
- [ ] State uncertainty in the same sentence as the claim, not as a
      footnote — "at 09:58:59 ± 2s" not "at 09:58:59" with the sigma
      buried elsewhere.
