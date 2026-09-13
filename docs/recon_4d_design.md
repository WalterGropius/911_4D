# True-4D reconstruction: design survey and decision

Owner: `recon` workstream. Companion to `wtc4d/recon/dynamic.py`, which
implements the prototype described below. This document explains the choice,
not just the code.

## 1. The problem, precisely

Three windows need genuine 4D (not per-epoch-static-plus-a-mask) treatment:
second impact (`D_impact2`, ~50 s), WTC2 collapse (`D_collapse2`, ~70 s), WTC1
collapse (`D_collapse1`, ~70 s) — see `wtc4d.timeline.DYNAMIC_WINDOWS`. Inside
each window:

- **10-30 synchronised SD cameras**, most 1-6 km away, several handheld or on
  a moving platform (helicopter, boat).
- **Timing uncertainty of ~0.5-1 s per camera** even after `wtc4d.sync`'s best
  effort (audio cross-correlation, on-screen clocks, event anchors). Cameras
  are *not* frame-synchronised to each other.
- **No ground truth 3D at any instant.** Nothing to compare a reconstructed
  timestep against except the 2D footage itself.
- **Huge dynamic range and topology change.** A tower turns into a debris
  cloud in ~12 seconds: this is not small non-rigid deformation of a fixed
  topology, it is closer to a fluid/granular process with a coherent onset.
- **Broadcast heterogeneity** compounds across time too — a single shot's own
  exposure can change (auto-iris reacting to the fireball).

Any method has to degrade gracefully when a camera's time estimate is off by
a second, because that is the normal case here, not an edge case.

## 2. Candidates considered

### 2.1 Per-timestep independent gaussian sets
Reconstruct a static splat at each of, say, 20 discrete times, independently.

- *Pro:* simplest possible extension of the static pipeline; no new model.
- *Con:* with 10-30 cameras total and a scene this large, each independent
  timestep is desperately underconstrained — it is a single-epoch
  reconstruction problem with an order of magnitude less data than a real
  epoch gets. No temporal coherence is enforced, so the geometry can flicker
  or invent structure frame to frame, and small timing errors show up as
  outright geometry changes rather than as smooth registration error.
- **Verdict: rejected as the sole method** for this data density. Might be
  usable as a per-shot *complement* around the canonical field (see 2.5) for
  content the field cannot represent at all (an explosion's first frame).

### 2.2 4D Gaussian Splatting (Wu et al., 2024) — deformation field
One canonical set of gaussians in a rest frame, plus a neural field
`F(x, t) -> (Δmean, Δscale, Δrotation, Δopacity)` (a HexPlane / K-Planes-style
factored feature grid feeding a small MLP in the original paper; we use a
plain sinusoidal-encoded MLP, see §3).

- *Pro:* parameter count does not grow with the number of timesteps; motion
  is continuous in `t` by construction, so it is inherently robust to a
  camera's time being off by a fraction of a second — nearby times predict
  nearly the same deformation. Naturally supports rendering at *any* `t`,
  including between observed times, which is exactly what a scrubber-driven
  viewer needs. Degrades to the static solution when the field predicts zero
  (easy to initialise that way, see §3.2).
- *Con:* a single coherent field is a poor fit for genuinely discontinuous
  events (a piece of facade detaching and falling separately from the frame
  it was part of) unless given enough capacity and training signal to
  represent branching motion, which is exactly the low-data-per-timestep
  regime we are in.

### 2.3 Spacetime Gaussians (Li et al., 2024) — temporal opacity + polynomial motion
Each gaussian carries a temporal centre/width (so it exists — is opaque — only
near one moment) and a low-order polynomial motion trajectory, trained
per-segment.

- *Pro:* the temporal-opacity idea is exactly right for content that is born
  and dies within the window — a dust puff, a falling panel, the initial
  fireball — which is a large fraction of what a collapse actually looks
  like. Polynomial motion is compact and stable to fit with little data per
  gaussian.
- *Con:* as the *sole* representation it under-fits large coherent structural
  motion (an entire face of the building descending as a unit) unless the
  polynomial degree and gaussian count both grow, at which point it starts to
  resemble the per-timestep approach's data hunger.

### 2.4 Physically-grounded / procedural-driven deformation
Let the `procedural` workstream's parametric collapse model (mass distribution
falling under gravity with a drag/crush term) supply the coarse motion field,
and have gaussians learn only the *residual* against it.

- *Pro:* enormous inductive bias for exactly the hardest part (large rigid or
  quasi-rigid motion with almost no cross-view evidence to constrain it from
  photometric loss alone) — physically implausible reconstructions become
  hard to reach because the residual has to fight the strong prior to get
  there. Directly reuses a sibling workstream's output.
- *Con:* the procedural model's own fidelity for the true kinematics of a
  progressive collapse is itself uncertain (it is a stylised model, not a
  structural simulation); leaning on it too hard risks the reconstruction
  reproducing the procedural model's assumptions rather than what the
  footage shows. Also couples `recon` to `procedural`'s API and schedule.

### 2.5 Hybrid (chosen)
Combine 2.2 and 2.3: a **canonical gaussian set + deformation field** for
coherent structural motion, where every gaussian *additionally* may carry a
**temporal opacity window** (`t_center`, `t_log_scale` on `Gaussians`, §3) for
content that is fundamentally transient. The procedural model (2.4) is used
as it already is in the static pipeline — as an **initialisation** source
(`init.from_procedural`) and, optionally, a coarse target for a future
motion-prior loss — rather than as a hard architectural commitment; that
keeps the option in 2.4 open without gating on it.

## 3. What is implemented (Phase 1 prototype)

`wtc4d.recon.dynamic`:

- `DynamicGaussians`: canonical parameters (means, log-scales, quaternions,
  SH, logit-opacities — the exact same fields as static `Gaussians`) plus one
  `DeformationField`.
- `DeformationField`: `PositionEncoding` (sinusoidal, normalised by scene
  radius) + `TimeEncoding` (sinusoidal) feed an MLP whose four heads predict
  `Δmean` (bounded by `tanh` to `max_translation_m`, default 400 m — a tower's
  worth of fall, chosen so a bad early gradient cannot fling gaussians across
  Lower Manhattan), `Δlog_scale` (bounded), `Δquat` (additive, renormalised
  on use), `Δlogit_opacity`. **Every head is zero-initialised**, so at
  construction `model.at(t) == canonical` for all `t`: training starts from
  the static solution, which is the correct prior — most of the frame *is*
  static even during a collapse (the surrounding skyline, the untouched
  tower, the ground).
- Optional **learned temporal window** (`learn_time_window=True`) makes
  `t_center`/`t_log_scale` trainable per-gaussian, giving the Spacetime-style
  birth/death behaviour from §2.3 on top of the field.
- `render(t, c2w, intrinsics, w, h)` matches the static backend signature
  exactly (`wtc4d.recon.backends.render`), so evaluation, the viewer export
  path, and the CPU/gsplat backend split are all shared with the static
  pipeline.
- A **temporal-smoothness regulariser** (`time_smooth_weight`, a
  finite-difference second-derivative penalty on `Δmean` in `t`) is the
  concrete answer to the ~0.5-1 s timing sigma: it makes the field's own
  induced uncertainty band wider than one bad time estimate, so one
  mistimed camera cannot inject a discontinuity.
- A **rigidity regulariser** (`rigidity_weight`, penalises `‖Δmean‖²`) biases
  toward "nothing moves unless the photometric loss demands it," which
  matters when 10-30 cameras cannot triangulate most of the volume at any
  single instant.
- `freeze_canonical_iters` freezes the canonical gaussians (means, scales,
  quats, colour, opacity — everything except the field) for the first N
  steps, so the field is not trying to co-adapt with an unstable canonical
  set from the first iteration; in production the canonical set comes
  pre-trained from the preceding static epoch anyway (see the
  `train_wtc2_collapse_4d.json` example spec, which initialises from
  `E2_static.ply`).

## 4. Validation strategy

- **CPU toy** (`wtc4d.recon.synthetic.toy_dynamic_scene`): a box scene where
  material above 200 m "falls" and spreads over `frac ∈ [0, 1]`, plus a
  drifting dust blob, rendered from 6 cameras × 10 timesteps with the
  reference rasteriser. `tests/test_recon_dynamic.py` asserts PSNR improves
  by >3 dB over training, that the learned time window actually localises
  (some gaussian's sigma shrinks below the window span), that
  `freeze_canonical_iters` really freezes, and that the rigidity term reduces
  mean squared deformation relative to an unregularised run.
- **On GPU** (not yet run — needs registered cameras + `sync` times): the plan
  is (1) initialise the canonical set from the preceding static epoch's
  trained `.ply` (E2 for both collapses, E1 for the second impact) rather
  than from the mesh prior directly — the static epoch already resolved the
  intact-building geometry; (2) train with `freeze_canonical_iters` ~2000 so
  the field has to explain the motion, not reshape a converged canonical set
  to cheat around the loss; (3) hold out whole shots (not just frames) for
  evaluation, since consecutive frames of one broadcast are highly
  correlated and would otherwise overstate generalisation; (4) inspect
  per-timestep exports (`export_timestep`) visually before trusting any
  numeric metric — with this little data, a model can hit a low loss while
  producing an implausible interpolation between two well-observed instants.

## 5. Risks and open questions

- **Underdetermination.** With 10-30 cameras for the *entire* collapse (not
  per instant), most of the volume at most instants is unconstrained by
  photometric evidence. The rigidity + temporal-smoothness regularisers make
  the failure mode "doesn't move enough" rather than "moves in an unphysical
  way," which is the safer failure for a forensic tool, but it means the
  reconstruction should be presented with visible uncertainty (§ coverage in
  `evaluate.coverage`), not as a confident claim.
- **Topology change is not really modeled.** A deformation field is
  fundamentally a diffeomorphism of the canonical set; a tower does not stay
  diffeomorphic to a rubble pile. In practice this means the field will
  represent the *early* seconds of a collapse (the point cloud "falling
  apart" is still a deformation of sorts) far better than the terminal state
  (settled rubble + dust), where the per-epoch static E5 splat is the more
  honest representation and the 4D window should be scoped to stop once the
  dust cloud dominates every camera.
- **Timing sigma vs. field capacity.** There is a real tension: more MLP
  capacity fits sharper motion but also fits noise from mistimed cameras as
  spurious high-frequency motion. The `time_smooth_weight` knob trades this
  off; it needs tuning against real data, which requires `sync`'s actual
  sigma distribution, not just its target.
- **Procedural coupling (2.4) is deferred, not abandoned.** Once
  `wtc4d.procedural` exposes `sample(t)` for the collapse windows
  specifically (not just epoch baselines), the natural next step is an
  additional loss term pulling the canonical-set-plus-deformation state
  toward the procedural model's coarse geometry at the same `t`, weighted low
  enough to be a prior rather than a constraint.
- **Evaluation without ground truth.** PSNR/SSIM against held-out *frames* of
  cameras that also contributed training frames from the same shot is a weak
  test (temporal correlation). Held-out *shots* are the right test and are
  supported (`val_dataset` in `train_dynamic`), but with only 10-30 shots per
  window, holding several out for validation is a real cost in reconstruction
  quality — a trade-off to make deliberately per window, not by default.
