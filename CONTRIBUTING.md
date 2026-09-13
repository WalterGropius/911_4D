# Contributing / working in parallel

This repo is developed by several independent sessions at once.  These rules
keep branches mergeable without coordination.

## Ownership

Each workstream owns exactly one directory and edits nothing outside it,
except its own dependency group in `pyproject.toml` and new files under
`data/<its dir>/`, `docs/` (own files only) and `.github/workflows/` (own
new file only).

| workstream | owns |
|---|---|
| corpus | `wtc4d/corpus/`, `data/manifests/` |
| sync | `wtc4d/sync/`, `data/time/` |
| geo | `wtc4d/geo/`, `data/geo/` |
| camreg | `wtc4d/camreg/`, `data/cameras/` |
| procedural | `wtc4d/procedural/` |
| recon | `wtc4d/recon/`, `data/scenes/` |
| infra | `infra/`, `.github/workflows/modal-*.yml` |
| viewer | `web/`, `.github/workflows/web.yml` |
| docs | `docs/ethics.md`, `docs/licensing.md`, `docs/methodology.md`, `docs/onboarding.md` |
| orchestrator | `README.md`, `PLAN.md`, `CONTRIBUTING.md`, `wtc4d/schema/`, `wtc4d/world.py`, `wtc4d/timeline.py`, `wtc4d/cli.py`, `.github/workflows/ci.yml` |

Need a change in a shared module (`wtc4d/schema`, `world`, `timeline`)?
Make it **additive** (new optional field, new constant) in your PR and call it
out in the PR description.  Never rename or remove.

## Mechanics

- Branch from `main`, work on your assigned `claude/ws-<name>` branch.
- Put your CLI at `wtc4d/<pkg>/cli.py` exposing `app: typer.Typer`; the root
  CLI mounts it automatically as `wtc4d <pkg> ...`.
- Add dependencies only to your `[project.optional-dependencies]` group.
  Everything must `pip install` on Ubuntu CPU.  Heavy/GPU-only tools belong
  in `infra/` container images; guard imports and `pytest.importorskip`.
- Tests: `tests/test_<pkg>_*.py` or inside your package.  CI runs
  `ruff check .` and `pytest` with `.[all,dev]`; keep the total test time
  under a few minutes and never download large files in tests.
- Never commit media or large binaries (see `.gitignore`).  Keep the repo
  under ~50 MB; anything bigger goes to the Modal volume or a release.
- Before opening the PR: `ruff check . && ruff format . && python -m pytest`.
- Merge policy: when CI is green and you have re-read your own diff, **squash
  merge your own PR**.  If `main` moved, merge `main` into your branch first
  and re-run checks.  Do not touch other workstreams' PRs.

## Style

Python 3.11+, type hints, pydantic models for anything serialised, numpy for
geometry.  Docstrings state units and coordinate conventions (metres, ENU,
OpenCV camera axes, project seconds).  Every hard-coded number about the
event or the buildings carries a source in a comment.
