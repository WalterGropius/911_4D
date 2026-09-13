# camreg — camera registration

Estimates intrinsics and world-frame pose for 2001-09-11 footage frames,
against the known geometry of the World Trade Center and Lower Manhattan
skyline, and propagates a registered frame's pose through the rest of its
shot. Owns `wtc4d/camreg/` and `data/cameras/`.

## Why registration, not SfM

Classic structure-from-motion needs parallax and stable features across
frames. Almost every camera that morning was far from the towers (helicopters
excepted), which gives tiny parallax on the one subject everyone was filming;
footage is SD, heavily compressed, and often re-encoded from broadcast; and
smoke covers a growing fraction of the frame as the morning goes on. SfM fails
on this footage in ways that are hard to detect after the fact.

The alternative used here: we already know where the towers were, to a few
metres, and we know a lot of other tall buildings' positions too
(`wtc4d.world.LANDMARKS`, extended by `wtc4d.geo` once merged). So instead of
triangulating between frames, every frame is registered against that known 3D
prior — either by a human clicking a handful of identifiable points
(`annotate.py` → `pnp.py`), or automatically by matching the frame's skyline
silhouette against a render of the prior (`render_match.py`). A registered
keyframe's pose is then propagated through the rest of its shot by 2D
tracking + per-frame re-solving (`track.py`).

## Conventions

Everything here uses the project-wide conventions
(`wtc4d.world`, `wtc4d.schema.camera`): world = local ENU tangent frame in
metres at `WORLD_ORIGIN`; camera = OpenCV axes (+X right, +Y down, +Z
forward); poses stored camera-to-world (`c2w`) as a 4×4 row-major matrix.
`wtc4d/camreg/conventions.py` is the single place that converts between this
and everything else camreg needs: OpenCV's world-to-camera `(rvec, tvec)`,
azimuth/elevation/roll for building and describing poses, and OpenCV↔OpenGL
axis flips. **Every other module imports these helpers rather than
reimplementing rotation math** — if you find yourself writing `Rodrigues`
again, use `wtc4d.camreg.conventions` instead.

Compass **azimuth** is degrees clockwise from north (0 = looking north, 90 =
east); **elevation** is degrees above the horizon; **roll** is degrees
clockwise in the image. `pose_from_look` / `look_from_pose` /
`look_at` convert between this and `c2w`.

## Modules

| module | responsibility |
|---|---|
| `conventions.py` | rotation/pose math, OpenCV↔OpenGL, projection, distortion |
| `priors.py` + `data/cameras/priors.yaml` | known vantage points (`CameraPriorRecord`) |
| `landmarks.py` | merges tower-corner geometry, `wtc4d.world.LANDMARKS`, and `wtc4d.geo` (when available) into one registry usable for PnP |
| `annotations.py` | the 2D-observation file format (`data/cameras/annotations/*.json`) |
| `annotate.py` | matplotlib / optional FastAPI click UI that writes that format |
| `pnp.py` | pose + focal length from 2D-3D correspondences, with covariance |
| `render_match.py` | annotation-free registration: render the prior, match its silhouette to the frame |
| `masks.py` | dynamic-content (smoke/fire/dust/graphics) masks, swappable for a learned segmenter later |
| `track.py` | lift keyframe features to 3D, LK-track through the shot, windowed bundle adjustment |
| `synthetic.py` | a tiny numpy renderer of the skyline + SD-style degradation, used for tests and validation |
| `validate.py` | the synthetic end-to-end scenarios behind the accuracy table below |
| `cli.py` | `wtc4d camreg priors\|pnp\|annotate\|auto\|track\|report\|synthetic-report` |

## The math, briefly

**PnP + focal (`pnp.py`).** Given landmark pixel clicks with per-click
`sigma_px`, we do *not* assume a known focal length (unlike COLMAP-style SfM,
there is no EXIF and no calibration target). Instead: sweep a geometric grid
of horizontal fields of view (6°–110° by default — a broadcast long lens to a
handheld camcorder at full wide), get an initial pose per hypothesis from
`cv2.solvePnPRansac`/`solveP3P`, then refine the best few with
Levenberg-Marquardt (`scipy.optimize.least_squares`) over
`[rvec, tvec, log f, (k1)]`, weighting each residual by `1/sigma_px` and
adding a **soft prior** on the camera centre from the vantage point's
`position_sigma_m`. The soft prior is what makes 3–4 point cases solvable and
keeps the weakly observed *range* to the subject from running away.
Covariance comes from `(JᵀJ)⁻¹` at the solution, rescaled by the reduced
chi-square (floored at 1) so an over-optimistic set of click sigmas cannot
produce an over-confident pose.

**The focal/range ambiguity is real and reported, not hidden.** For a
distant subject, doubling the focal length and doubling the distance produce
almost the same image. `PnPResult.focal_depth_correlation` reports
`|corr(log f, range)|` from the covariance; above ~0.95 the *range* comes from
the prior, not from the image, and `position_sigma_m` will (correctly) be
large. The fix is never a better solver — it's a landmark at a different
depth (a bridge tower, a near rooftop) or an independently known focal
length.

**Render-and-match (`render_match.py`).** Coarse grid search over
azimuth/elevation/roll/focal around the prior, scored by an *oriented*
chamfer distance on Canny/Sobel edges plus a silhouette term (the model's
projected roofline vs. the image's sky/building boundary — see
`wtc4d.camreg.masks.skyline_profile`). The silhouette term is asymmetric: a
model point in the image's open sky is *contradicted* (charged in full), a
model point matching the image's silhouette is scored by pixel offset, and
image structure the (deliberately incomplete) model doesn't know about costs
nothing. The best few hypotheses are refined with Powell's method, then model
samples are snapped to nearby image edges to produce real 2D–3D
correspondences for the same `pnp.py` solver.

**Tracking (`track.py`).** A keyframe's 2D features are back-projected to 3D
by ray-casting against the tower boxes (or `wtc4d.geo`'s mesh) using the
keyframe's own pose, then followed with pyramidal Lucas-Kanade + a
forward-backward consistency check, re-masked by `masks.py` every frame. Each
frame with enough surviving tracks gets its own PnP solve — this bounds pose
error against reprojection rather than letting 2D drift accumulate silently.
A small windowed bundle adjustment (`bundle_adjust`) then couples a handful of
keyframes: a smoothness prior (second difference of position and rotation)
for handheld/pan-tilt shots, or a fixed-position prior for a tripod/rooftop
camera. **Bug fixed during development:** the fixed-position prior must pin
the actual camera centre `C = -Rᵀt`, not the raw `w2c` translation `t` — the
two are only the same thing when rotation is held fixed too, and for a
panning tripod shot they diverge badly (see `tests/test_camreg_track.py`).

## Synthetic validation

`wtc4d/camreg/synthetic.py` renders the two towers, a handful of nearby
landmark buildings, and the ground as flat-shaded, noise-textured boxes with a
proper z-buffer and near-plane clipping, then degrades the render (blur,
resample, sensor noise, JPEG re-encoding) to look like SD footage. This is
enough to test the whole pipeline against exact ground truth without a GPU,
without model weights, and without committing a single image.

`wtc4d camreg synthetic-report` (or `wtc4d.camreg.validate.run_synthetic_validation`)
reproduces this table — five vantage points from `priors.yaml`, a rendered
frame degraded as above, landmarks clicked at their true projected pixel
±1.5 px noise (simulating a careful annotator), solved with `pnp.solve_pose`:

| prior | hfov° | landmarks | pos err (m) | reported σ (m) | rot err (°) | reproj RMSE (px) | planarity | focal/range corr |
|---|---|---|---|---|---|---|---|---|
| brooklyn_heights_promenade | 35 | 29 | 30.9 | 73.7 | 0.25 | 1.84 | 0.36 | 0.985 |
| jc_exchange_place | 28 | 29 | 42.6 | 51.5 | 0.31 | 1.76 | 0.29 | 0.990 |
| liberty_state_park | 20 | 27 | 104.5 | 194.3 | 0.17 | 2.08 | 0.63 | 0.999 |
| hoboken_waterfront | 15 | 26 | 64.1 | 158.1 | 0.15 | 2.27 | 0.40 | 0.996 |
| west_street_vesey | 70 | 14 | 2.6 | 5.8 | 0.19 | 2.12 | 0.36 | 0.811 |

Takeaways:

* **Rotation is always good** (< 0.35°) — the skyline is angularly rigid and
  the click noise is small relative to the baseline of landmarks in frame.
* **Position accuracy tracks the focal/range correlation, exactly as
  predicted.** The three distant, narrow-lens waterfront views (Brooklyn,
  Jersey City, Liberty State Park, Hoboken) have correlation ≥ 0.98 and
  position error of tens to ~100 m — consistent with, and safely inside, the
  *reported* sigma every time. `west_street_vesey`, a near, wide (70°) street
  view where the towers subtend most of the frame, has much weaker
  focal/range coupling (0.81) and recovers position to **2.6 m**.
* **The reported uncertainty is the thing to trust, not a fixed accuracy
  number.** `tests/test_camreg_pnp.py` asserts position error stays within a
  generous multiple of the pose's own `position_sigma_m`, not within a fixed
  metre count — that is the honest claim this module makes about itself.

Two real-frame stand-ins are committed as a worked example of the full
annotation → PnP path: `data/cameras/annotations/demo-brooklyn-promenade_000000.json`
and `demo-jc-exchange-place_000000.json`, with the resulting poses in
`data/cameras/poses.jsonl`. **These are synthetic**, generated from
`wtc4d.camreg.synthetic` rather than clicked on real archival footage — this
development session had no vetted, licensed frame to annotate and does not
download/cache media into the repo. Each annotation's `notes` field says so
explicitly. Replace them with a real annotation (`wtc4d camreg annotate`) on
an actual Brooklyn Promenade / Exchange Place frame as soon as one is
available in the corpus; the pipeline they exercise is otherwise identical.

## `render_match.py`: what works and what doesn't (read before trusting it)

Automatic registration is the harder, less finished half of this module —
report success and failure honestly rather than overselling it:

* **The correspondence + PnP stage is reliable** once given a roughly correct
  starting pose: `edge_correspondences` + `pnp.solve_pose` on a hand-verified
  hypothesis behaves exactly like the annotation path above.
* **The blind coarse search is not reliably convergent.** Matching a sparse
  two-tower-box silhouette (the fallback model, before `wtc4d.geo`'s real
  building mesh lands) against a real chamfer field is genuinely
  multi-modal: a narrow, tightly-cropped field of view can score as well as
  or better than the true wide one, because explaining a small part of the
  frame very precisely can beat explaining most of it approximately. The
  scoring in `_score`/`_skyline_cost` is written to reward *coverage* (every
  image column and every model sample count against a fixed denominator, so
  a wide correct hypothesis isn't penalised for "showing more to get wrong"),
  and `min_skyline_columns` rejects degenerate slivers outright, but this
  does not eliminate the ambiguity — validated by hand against the synthetic
  scenarios above, the coarse+refine stages alone find the right basin only
  some of the time.
* **What to do about it today:** treat `auto_register`'s output as a
  *candidate*, not a pose — inspect `RegistrationResult.chamfer_px` and
  `column_coverage`, try several priors and compare, or seed
  `centre_azimuth_deg`/`centre_elevation_deg` from a rough manual estimate.
  For production registrations, the recommended path is still
  `annotate.py` → `pnp.py`, with `track.py` propagating each annotated
  keyframe through its shot.
* **What should fix it:** the real Lower Manhattan building mesh from
  `wtc4d.geo` (a full skyline of dozens of buildings at their true heights,
  not two boxes) should sharpen the chamfer field enough to close most of
  this gap, because a full skyline's silhouette is far more distinctive than
  two towers at a coarse grid step. Re-run `synthetic-report` once `wtc4d.geo`
  is merged and update this section.
* **Other known failure modes:** smoke covering part of the silhouette (pass
  a `masks.dynamic_mask`); very long lenses, where azimuth and position trade
  off (use `fix_position=True` once position is known from another frame of
  the same shot); night or extreme close-ups with no skyline in frame
  (`column_coverage` will be near zero — check it); a wrong prior (the search
  is local to it, so a mislabelled vantage point produces a confident wrong
  answer — try several and compare).

## Workflow

```
wtc4d camreg priors                          # list known vantage points
wtc4d camreg annotate FRAME.jpg --shot S --frame 0 --camera-prior jc_exchange_place
wtc4d camreg pnp data/cameras/annotations/S_000000.json --out data/cameras/poses.jsonl
wtc4d camreg auto FRAME.jpg --camera-prior jc_exchange_place   # best-effort, see above
wtc4d camreg track FRAMES_DIR --keyframe-annotation ... --keyframe-index 0 --out poses.jsonl
wtc4d camreg report data/cameras/poses.jsonl --out docs/img/camreg_coverage.png
wtc4d camreg synthetic-report                # reproduces the accuracy table above
```

`data/cameras/priors.yaml` documents 27 vantage points researched for this
PR: the three network news helicopters and NYPD Aviation, the Naudet
brothers' street and firehouse positions, Brooklyn Heights Promenade and
Fulton Ferry/DUMBO, Jersey City (Exchange Place, Newport), Hoboken (waterfront
and Castle Point), Liberty State Park, Weehawken and Bayonne, Manhattan
ground positions near the complex (West St/Vesey, Church St, Fulton St,
Battery Park), the Woolworth Building, Empire State Building observatory, a
placeholder Midtown rooftop-camera entry, both East River bridges, FDR Drive,
the Staten Island Ferry, and Governors Island. Every record carries a
`confidence` level and `notes` on which well-known shots are associated with
it — treat `low`/`medium` confidence entries and specific shot attributions
as leads for the `corpus` workstream to verify, not as established fact.

## Tests

`tests/test_camreg_conventions.py`, `test_camreg_pnp.py`, `test_camreg_masks.py`,
`test_camreg_track.py` — round trips (Rodrigues, `c2w`↔`w2c`, OpenCV↔OpenGL,
distortion), synthetic PnP recovery against both ground truth and reported
uncertainty (the table above, machine-checked), mask sanity against the
renderer's ground truth, and a synthetic moving-camera tracking + bundle
adjustment scenario. All run on CPU with no network access in a few seconds
total (`pytest.importorskip("cv2")` guards the tracker tests, which need
`opencv-python-headless`'s LK implementation).

## Next steps (flagged for scale-up, not blocking this PR)

* **Learned matching at scale.** `render_match.match_learned` is a stub
  interface for LightGlue+SuperPoint or kornia's LoFTR between a real frame
  and a textured render of the refined hypothesis — torch-based, so it
  belongs in the GPU image (`infra/`), not the CPU `camreg` extra.
* **Re-run `synthetic-report` once `wtc4d.geo` lands** and update the
  render-and-match section above; `landmarks.py` and `render_match.py`
  already prefer `wtc4d.geo.load_landmarks()`/`load_scene()` when importable
  and fall back to `wtc4d.world` otherwise, so no code change should be
  needed, only re-validation.
* **A learned dynamic-content segmenter.** `masks.MaskSegmenter` is a
  `Protocol` so a fine-tuned model can be dropped in without touching
  `track.py`; today's heuristics (colour/saturation/texture/temporal) are
  intentionally simple and CPU-only.
* **Replace the two synthetic demonstration annotations** with real clicked
  frames once the corpus workstream has a licensed, vetted candidate for the
  classic Brooklyn Promenade / Exchange Place shots.
