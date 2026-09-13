## Summary

<!-- One or two sentences: what changed and why. -->

## Ownership check

- [ ] This PR only touches files my workstream owns (see `CONTRIBUTING.md`'s
      ownership table), or any shared-module change (`wtc4d/schema`,
      `wtc4d/world.py`, `wtc4d/timeline.py`) is strictly additive and
      called out below.
- [ ] If this changes `pyproject.toml`, it only adds to my own
      `[project.optional-dependencies]` group.

<!-- If you touched a shared module, describe the additive change here: -->

## Tests

- [ ] `ruff check . && ruff format . && python -m pytest` pass locally.
- [ ] New/changed behavior has test coverage under `tests/test_<pkg>_*.py`
      or inside the package.
- [ ] No test downloads large files or depends on network access.

## Media check

- [ ] No footage, frames, or other large binaries are committed (see
      `.gitignore` and `data/README.md` — only small derived data belongs
      in `data/`).
- [ ] Anything user-facing that surfaces a source frame/thumbnail has been
      checked against `docs/ethics.md` (identifiable people, falling/
      jumping, human remains are never rendered).
