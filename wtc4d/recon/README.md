# `wtc4d.recon` — gaussian splat reconstruction

Owns: `wtc4d/recon/`, `data/scenes/`, `tests/test_recon_*.py`, the `recon`
extra in `pyproject.toml`, `docs/recon_4d_design.md`. See `PLAN.md` /
`CONTRIBUTING.md` for the project-wide rules; this file is the workstream's
own design notes and operating instructions.

## Why this pipeline looks the way it does

Ordinary 3D Gaussian Splatting assumes SfM-quality poses, roughly consistent
image quality, and a mostly-static scene. None of that holds here:

- **Cameras are pre-registered, not solved.** `camreg` gives fixed poses in
  the shared ENU world frame via landmark PnP against the geo prior — there
  is no bundle adjustment step to fall back on if a pose is wrong. Training
  therefore treats poses as fixed by default, with an optional *small,
  bounded* refinement (`PoseRefineConfig`, capped in metres/degrees) rather
  than free optimisation.
- **A strong geometric prior already exists** (the city mesh, the procedural
  towers/plume/collapse model), so initialisation does not need — and for
  this footage, cannot get — a classical SfM point cloud. `init.py` builds
  gaussians directly from the mesh surface and the procedural cloud.
- **Broadcasts disagree wildly on colour.** A per-image (or per-shot)
  appearance transform (`AppearanceModel` in `train_static.py`) absorbs
  exposure/white-balance differences so the *geometry* isn't forced to
  compromise between a warm VHS transfer and a cool network feed.
  Held-out/unseen shots fall back to the mean transform (documented on
  `AppearanceModel.forward`) — there is no way to recover an unseen
  broadcast's grade from its image alone.
- **Most pixels are not usable static evidence.** Smoke, fire, dust and
  station bugs are masked; masked pixels never supervise the static layers.
  `LossConfig.mask_mode = "supervise_smoke"` optionally routes them to a
  separate, time-tagged smoke/debris gaussian set instead of discarding them.
- **Two true-4D windows exist inside the epochs** (second impact, both
  collapses). Those get the prototype in `dynamic.py`; see
  `docs/recon_4d_design.md` for the full survey and the reasoning behind the
  canonical-set + deformation-field + temporal-opacity design.

## Module map

| module | what |
|---|---|
| `conventions.py` | quaternion/rotation/COLMAP/OpenGL conversions, pure numpy |
| `colmap.py` | COLMAP text model (`cameras.txt`/`images.txt`/`points3D.txt`) import/export |
| `nerfstudio.py` | nerfstudio `transforms.json` import/export |
| `data.py` | `ReconDataset` (poses + frames + masks + times -> `FrameSample`), re-exports the two interop modules |
| `gaussians.py` | `Gaussians` parameter container, `Layer` enum, `.ply`/`.npz` I/O |
| `init.py` | gaussian initialisation from mesh / procedural cloud / COLMAP points; merge, dedup, layer assignment |
| `cpu_raster.py` | pure-torch EWA-splatting reference rasteriser |
| `backends.py` | `cpu` / `gsplat` backend selection behind one `render()` call |
| `train_static.py` | the static per-epoch training loop (appearance, masks, pose refinement, densification, anchor/depth priors) |
| `dynamic.py` | the 4D prototype (canonical + deformation field + temporal opacity) |
| `evaluate.py` | PSNR/SSIM(/LPIPS), residual images, per-gaussian coverage |
| `job.py` | `run(spec) -> dict` entrypoint for the infra runner; `write_scene_manifest` |
| `synthetic.py` | CPU toy scenes (box towers, a "collapse") used by the tests |
| `cli.py` | `wtc4d recon ...` typer app |
| `jobs/*.json` | example job specs (two need real data + GPU, two are dependency-free CPU smoke tests) |

## Running on CPU (today, no data needed)

```bash
pip install -e ".[recon,dev]"
wtc4d recon toy static      # synthetic box towers, 6 cameras, reference rasteriser
wtc4d recon toy dynamic     # synthetic "collapse" + drifting blob, 10 timesteps
wtc4d recon run-job wtc4d/recon/jobs/toy_static_cpu.json
python -m pytest tests/test_recon_*.py
```

These do not touch any real footage or the Modal volume; they exist so the
training *logic* (losses, densification, the deformation field, format
conversions) is exercised without a GPU. `wtc4d recon info` reports which
rasteriser backend is active — `cpu` here, always.

## Running on GPU (via `infra`)

This session has no GPU and cannot reach Modal directly (gRPC is blocked from
this environment — see `infra/README.md`); GPU runs go through the
`infra`-provided `run_job(spec)` / GitHub Actions dispatch, calling this
package's `wtc4d.recon.job:run` as the entrypoint. What's needed in the image:

```bash
pip install -e ".[recon]" gsplat   # gsplat needs CUDA + nvcc at install time
```

`backends.gsplat_available()` guards every use of `gsplat` behind a try/except
plus a CUDA check, so the same code runs correctly on a CPU box (falls back to
`cpu_raster`) and picks up `gsplat` automatically wherever CUDA is present —
no code change between environments, only the installed package.

Example specs (`wtc4d/recon/jobs/`):

- `train_e2_static.json` — epoch E2 (both towers burning) static splat. Needs
  `data/cameras/poses.jsonl`, a frame store, dynamic masks and (ideally)
  `sync` time estimates on the Modal volume. **Not runnable yet** — no real
  poses/frames exist. Expect a few hours at ~2M gaussians / 30k iterations on
  one A10G-class GPU; this is a rough estimate pending a real run since no
  epoch has been trained yet.
- `train_wtc2_collapse_4d.json` — the WTC2 collapse 4D window, initialising
  the canonical set from `E2_static.ply` (so E2 must be trained first).
  **Also blocked** on `camreg` poses and, critically, `sync` per-frame times
  (frames without a `TimeEstimate` are dropped by `train_dynamic`).
- `toy_static_cpu.json`, `toy_dynamic_cpu.json` — dependency-free CPU smoke
  tests. **Run these first** whenever a new GPU image is stood up, to confirm
  the entrypoint and dependencies work before spending GPU time on real data.

## Interoperability

`data.py` / `colmap.py` / `nerfstudio.py` convert exactly between:

- **ours**: camera-to-world, OpenCV axes, world ENU metres (`CameraPose.c2w`);
- **COLMAP**: world-to-camera quaternion + translation, OpenCV axes;
- **nerfstudio/OpenGL**: camera-to-world, +Y up / +Z back (`transform_matrix`
  in `transforms.json`).

`wtc4d recon export-colmap` / `export-transforms` / `import-colmap` wrap
these for the CLI. Round-trip unit tests are in
`tests/test_recon_conventions.py`. Exported `transforms.json` sets
`applied_transform` to the identity and a matching reader **refuses** to load
a file whose `applied_transform` is not the identity — that means nerfstudio's
own auto-orientation ran and the poses are no longer in the wtc4d world frame;
re-export with `--orientation-method none --center-method none` if you hit
this.

## Interfaces to sibling workstreams (not merged yet — coded against with fallbacks)

- `wtc4d.geo.load_scene() -> trimesh.Scene` — `init.load_prior_scene()` falls
  back to `init.prior_boxes_scene()` (two boxes + a ground plane from
  `wtc4d.world.TOWERS`) when `geo` is not installed or raises.
- `wtc4d.procedural.sample(t) -> GaussianCloud` — `init.from_procedural()`
  adapts any object exposing `means`/`scales`/`opacities`-style attributes
  (see `init.gaussians_from_cloud`); falls back to a `.ply` path
  (`procedural_ply=`) or is skipped entirely if neither is available.
- `wtc4d.camreg` → `data/cameras/poses.jsonl` (`CameraPose` per line) and a
  masks directory — `data.ReconDataset.from_paths` reads exactly this layout.
  **Not produced yet**; nothing in `recon` can train on real footage until it
  exists.
- `wtc4d.sync` → per-frame `TimeEstimate`s — `data.load_time_estimates`
  tolerates a few plausible JSONL shapes (flat or `{"estimate": {...}}`
  nested; `shot_id`/`shot`, `frame_idx`/`frame`). **Not produced yet**; the 4D
  windows cannot train without it (frames lacking a time estimate are simply
  dropped, per `dynamic.train_dynamic`'s docstring).

Every import of these is lazy (inside a function, in a `try/except`), so
`wtc4d.recon` imports cleanly regardless of what else is installed, and this
package will pick up the real implementations automatically the moment those
PRs merge — no code change needed here, only better inputs.

## What's verified on CPU vs. what awaits GPU

**Verified on CPU (this PR, `tests/test_recon_*.py`, ~a couple of minutes total):**
convention round trips (COLMAP, nerfstudio, both directions), the reference
rasteriser's projection/occlusion/culling and its gradients against finite
differences, `.ply`/`.npz` round trips including the extra `layer`/temporal
fields, mesh-based initialisation/dedup/layer assignment, PSNR/SSIM/coverage
metrics, the static training loop actually improving PSNR on a synthetic
scene (including the appearance model measurably absorbing exposure jitter,
masked pixels provably not affecting the static PSNR, the anchor loss
measurably reducing drift, densification growing the gaussian count, and pose
refinement staying within its configured bounds), and the dynamic prototype
improving PSNR on a synthetic "collapse" (including the learned temporal
window localising in time, `freeze_canonical_iters` actually freezing, and
the rigidity regulariser reducing deformation magnitude).

**Awaits GPU + real data:** everything above operates on a few hundred
gaussians at 20-50 px resolution because the reference rasteriser is
`O(N × pixels)`; real epochs need `gsplat` and real poses/frames/times. Also
untested for lack of real data: whether the anchor/depth-prior weights
suggested in `train_e2_static.json` are remotely correct (they are
placeholders extrapolated from the 3DGS literature, not tuned against this
footage), how well pose refinement behaves against `camreg`'s actual sigma
distribution, and whether the deformation field's capacity/regularisation
balance holds up against `sync`'s real timing sigma (see design doc §5).

## Open questions for later phases

See `docs/recon_4d_design.md` §5 for the 4D-specific risks. Pipeline-level
ones: what anchor/depth-prior weight actually keeps facades from drifting
without preventing gsplat's own densification from fixing prior geometry
errors (needs a real epoch to tune); whether one `SplatAsset` per epoch is
the right viewer granularity or whether per-building assets would let the
viewer toggle individual structures; how coverage (`evaluate.coverage`)
should be surfaced in the manifest for the viewer's uncertainty visualisation
(`SceneManifest` has no field for it yet — an additive schema change, to be
proposed once the viewer workstream has an opinion on the representation).
