# Onboarding

Welcome. This page gets you from a clean checkout to a productive first
contribution. For *what* the project is trying to build and why, read
`README.md` and `PLAN.md` first if you haven't. For the rules that keep
nine parallel workstreams mergeable without coordination, read
`CONTRIBUTING.md` — this page assumes it.

## Setup

```bash
pip install -e ".[dev]"      # add your workstream's extra, e.g. ".[dev,geo]"
wtc4d info                   # world origin, epochs, which workstreams are installed
python -m pytest
```

`wtc4d info` (`wtc4d/cli.py`) is a useful smoke test: it prints the ENU
world origin, the epoch table, and — for each workstream in
`wtc4d.cli.WORKSTREAMS` — whether its package imports cleanly. A workstream
showing `missing` just means its optional dependency group isn't
installed yet (expected if you haven't `pip install -e ".[<pkg>]"`), not
that something is broken.

CI (`.github/workflows/ci.yml`) installs `.[all,dev]` and runs `ruff check
.` and `pytest` on every push/PR to `main`. Run both locally before
opening a PR:

```bash
ruff check . && ruff format . && python -m pytest
```

## Repo layout

```
wtc4d/schema      shared data models (the contract between workstreams)
wtc4d/world.py    world frame (ENU, metres), tower geometry, landmarks
wtc4d/timeline.py event anchors (NIST times) and epochs
wtc4d/cli.py      root CLI; mounts wtc4d.<pkg>.cli:app automatically
wtc4d/<pkg>       one package per workstream (corpus, sync, geo, camreg, procedural, recon)
infra/            Modal compute + GitHub Actions dispatch
web/              viewer (Three.js)
data/             small derived data only — never raw footage (see data/README.md)
docs/             this directory: ethics, licensing, methodology, onboarding
tests/            cross-cutting tests; per-package tests may also live inside wtc4d/<pkg>
```

`wtc4d/schema` is the shared contract every workstream reads and writes:
`corpus.py` (`Source`, `Shot`, `Frame`), `time.py` (`TimeEstimate`,
`TimeMethod`), `camera.py` (`CameraIntrinsics`, `CameraPose`,
`CameraPrior`), `geometry.py` (`LatLonAlt`), `scene.py` (`SceneManifest`,
`SplatAsset`, `CameraRef`, `TimelineEvent`). Read these four files before
writing code that produces or consumes their models — this doc doesn't
restate their fields; see `docs/methodology.md` for how they fit together
conceptually, and the source for the exact contract.

## Workstream ownership map

From `CONTRIBUTING.md` — each workstream owns exactly one directory and
edits nothing outside it (except its own `pyproject.toml` dependency
group, its own new files under `data/<its dir>/`, and — for `docs`
specifically — the files listed below):

| workstream | branch | owns |
|---|---|---|
| corpus | `claude/ws-corpus` | `wtc4d/corpus/`, `data/manifests/` |
| sync | `claude/ws-sync` | `wtc4d/sync/`, `data/time/` |
| geo | `claude/ws-geo` | `wtc4d/geo/`, `data/geo/` |
| camreg | `claude/ws-camreg` | `wtc4d/camreg/`, `data/cameras/` |
| procedural | `claude/ws-procedural` | `wtc4d/procedural/` |
| recon | `claude/ws-recon` | `wtc4d/recon/`, `data/scenes/` |
| infra | `claude/ws-infra` | `infra/`, `.github/workflows/modal-*.yml` |
| viewer | `claude/ws-web` (or similar) | `web/`, `.github/workflows/web.yml` |
| docs | `claude/ws-docs` | `docs/ethics.md`, `docs/licensing.md`, `docs/methodology.md`, `docs/onboarding.md` (plus, by convention, other new `docs/*.md` files, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `.github/ISSUE_TEMPLATE/*`, `.github/PULL_REQUEST_TEMPLATE.md`) |
| orchestrator | — | `README.md`, `PLAN.md`, `CONTRIBUTING.md`, `wtc4d/schema/`, `wtc4d/world.py`, `wtc4d/timeline.py`, `wtc4d/cli.py`, `.github/workflows/ci.yml` |

Need a change to a shared module (`wtc4d/schema`, `world.py`,
`timeline.py`)? Make it additive (new optional field, new constant), call
it out explicitly in your PR description, and never rename or remove an
existing field — other branches are editing in parallel against the
current shape.

## How to pick up a task

1. Read `PLAN.md` §4 (phase 1 workstreams table) to find your workstream's
   row and deliverable, and §1-3 for the problem context (why 2001 footage
   is hard, the epoch/dynamic-window structure) that shapes the design.
2. Branch from `main` as `claude/ws-<name>` (or continue your existing
   branch).
3. Check whether your deliverable depends on another workstream's schema
   additions landing first (e.g. `recon` depends on `camreg`'s
   `CameraPose` shape, which already exists in `wtc4d.schema.camera`) —
   most phase-1 deliverables are designed to be buildable against the
   *current* schema without waiting on another in-flight PR.
4. Put your CLI at `wtc4d/<pkg>/cli.py` exposing `app: typer.Typer`; it's
   mounted automatically (`wtc4d/cli.py:_mount_workstreams`).
5. Add dependencies only to your own `[project.optional-dependencies]`
   group in `pyproject.toml`. Everything must `pip install` on Ubuntu CPU
   in CI; GPU-only or non-pip tools belong in `infra/` container images,
   guarded with `pytest.importorskip` in tests that need them.
6. Write tests under `tests/test_<pkg>_*.py` or inside your package; keep
   total test time low and never download large files in tests.
7. Before opening a PR: `ruff check . && ruff format . && python -m
   pytest`. When CI is green and you've re-read your own diff, squash-merge
   your own PR (`CONTRIBUTING.md`, "Mechanics").

## Glossary

- **Epoch** — a time window in which the *static* scene is approximately
  constant (`wtc4d.timeline.Epoch`, `E0`-`E5`: intact → WTC1 burning →
  both burning → WTC2 down → both down → rubble). Static splats are
  trained per epoch.
- **Dynamic window** — a short, densely multi-view-covered window (second
  impact, WTC2 collapse, WTC1 collapse; `wtc4d.timeline.DynamicWindow`)
  that's a candidate for a true time-varying (4D) splat rather than one
  static-per-epoch splat.
- **ENU** — East-North-Up: the local tangent-plane coordinate frame
  (metres, origin `wtc4d.world.WORLD_ORIGIN`) everything spatial in this
  project lives in. See `wtc4d.world.latlon_to_enu`.
- **c2w** — camera-to-world: the 4x4 transform (`CameraPose.c2w`, OpenCV
  camera axes: +X right, +Y down, +Z forward) that maps a point in camera
  space to the world ENU frame.
- **Project seconds** — this project's time unit: seconds since local
  midnight 2001-09-11 00:00:00 EDT, a plain float (`wtc4d.timeline`,
  `hms()`/`fmt_local()`). 08:46:30 is `31590.0`.
- **Landmark** — a fixed, identifiable 3D point (a roofline corner, a
  spire tip, a bridge tower) used to register cameras against the `geo`
  prior by PnP (`wtc4d.world.Landmark`, the `LANDMARKS` registry).
- **PnP** — Perspective-n-Point: solving for a camera's pose given a set
  of 2D pixel observations of points whose 3D positions are already known
  (here, landmarks) — the core of how `camreg` registers a frame without
  needing feature matches between different videos.
- **Gaussian splat / 4DGS** — a scene representation as a set of
  translucent 3D Gaussians (position, covariance, color, opacity)
  optimized to reproduce a set of posed camera views; **4DGS** extends
  this with time-varying parameters for dynamic content. Trained by
  `wtc4d.recon`.
- **Provenance** — the traceable link from any rendered element back to
  the specific `Source`/`Shot`/`Frame`, `TimeEstimate`, and `CameraPose`
  that support it. See `docs/methodology.md`, "Tracing a viewer claim
  back to source frames."
