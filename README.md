# 911_4D

A 4D reconstruction — three dimensions plus time — of the World Trade Center
attacks of September 11, 2001, from the morning skyline with both towers
standing to the rubble pile, built from every piece of public footage we can
find and register.  The project is an homage and a forensic tool: every
surface in the reconstruction is traceable to the footage that supports it.

Status: **phase 1 scaffolding**.  See [`PLAN.md`](PLAN.md) for the roadmap and
[`CONTRIBUTING.md`](CONTRIBUTING.md) for how the parallel workstreams fit
together.

## Approach in one paragraph

Gather and catalogue footage (`corpus`), assign every shot an absolute time
with an uncertainty (`sync`), build a georeferenced model of Lower Manhattan
as it stood that morning (`geo`), register each camera against it
(`camreg`), and train gaussian splats per epoch and, for the densely filmed
seconds of the impacts and collapses, true 4D splats (`recon`).  A procedural
model of the towers, smoke and collapse (`procedural`) fills the timeline
between reconstructed windows and initialises the splats.  A web viewer
(`web/`) lets you scrub time, fly the scene, and jump into any registered
camera to see its source frame aligned with the reconstruction.

## Quick start

```bash
pip install -e ".[dev]"
wtc4d info
python -m pytest
```

## Layout

```
wtc4d/schema      shared data models (the contract between workstreams)
wtc4d/world.py    world frame (ENU, metres), tower geometry, landmarks
wtc4d/timeline.py event anchors (NIST times) and epochs
wtc4d/<pkg>       one package per workstream (see PLAN.md)
infra/            Modal compute + GitHub Actions dispatch
web/              viewer
data/             small derived data only (no media)
docs/             ethics, licensing, methodology
```

## Compute

GPU work runs on Modal, dispatched from GitHub Actions (cloud coding sessions
cannot reach Modal's gRPC API directly).  Repository secrets `MODAL_TOKEN_ID`
and `MODAL_TOKEN_SECRET` are required; see `infra/`.

## License

Code: Apache-2.0.  Footage is catalogued, not redistributed; see
`docs/licensing.md`.
