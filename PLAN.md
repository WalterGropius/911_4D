# 911_4D — Master Plan

**Goal.** A navigable 4D (3D + time) reconstruction of the World Trade Center
attacks on 2001-09-11, from the moment both towers stand (before 08:46) to
the rubble pile (after 10:28), built from *all* publicly available footage,
with every element traceable to its source frames.  Open source, later a tool
for investigators (forensic video analysis, debunking) — so provenance and
uncertainty are first-class, not afterthoughts.

## 1. Why this is hard (and how we get around it)

| Problem | Consequence | Approach |
|---|---|---|
| 2001 footage is SD (480i, VHS/DV), heavily compressed, often re-encoded from TV | Weak features, motion blur, interlacing | Deinterlace + super-resolution only for viewing; register on stable structures (building edges, skyline), not fine texture |
| Almost all cameras are far away (NJ, Brooklyn, helicopters, Midtown) | Weak baselines, tiny parallax on the towers themselves | Use a **known 3D prior** (georeferenced 2001 city model) and register cameras by PnP on landmarks instead of classic SfM |
| Smoke/fire/dust dominate and change every second | Cross-video feature matching fails; classic SfM breaks | Mask dynamic regions; register on static content; treat smoke as separate time-tagged gaussians |
| Footage is unsynchronised, clocks unknown | 4D is impossible without absolute time | Multi-cue **temporal alignment**: TV archive airtimes, on-screen clocks, event anchors, audio cross-correlation, solar shadows |
| Copyright and sensitivity | Cannot redistribute raw footage; must be respectful | Catalogue + pointers, derived works only; ethics guidelines in `docs/` |

## 2. Architecture (data flow)

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

Shared contract: `wtc4d/schema` (models), `wtc4d/world.py` (ENU frame, towers,
landmarks), `wtc4d/timeline.py` (events, epochs).  See `CONTRIBUTING.md`.

Compute: Modal (GPU) via GitHub Actions dispatch — see `infra/README.md`.

## 3. Epochs and dynamic windows

| Epoch | Window (EDT) | Static state | Footage density |
|---|---|---|---|
| E0 | < 08:46:30 | both intact | sparse (use historical photos + prior) |
| E1 | 08:46:30 – 09:02:59 | WTC1 burning | medium (Naudet, first helicopters ~08:52) |
| E2 | 09:02:59 – 09:58:59 | both burning | very dense |
| E3 | 09:58:59 – 10:28:22 | WTC2 gone, WTC1 burning | dense |
| E4 | 10:28:22 – 12:00 | both gone, dust | dense |
| E5 | 12:00 → 9/12+ | rubble | aerial (NOAA 9/23 orthophoto, LiDAR), ground photos |

True 4D windows (dense multi-view, seconds long): second impact, WTC2
collapse, WTC1 collapse.  Elsewhere time is carried by the procedural layer
plus per-epoch static splats and time-tagged smoke gaussians.

## 4. Phase 1 workstreams (parallel, independent branches)

| # | Workstream | Package / dir | Model | Deliverable |
|---|---|---|---|---|
| 1 | corpus | `wtc4d/corpus` | Sonnet | Source registry (archive.org TV archive, NIST FOIA releases, Wikimedia, Flickr, YouTube, NOAA/USGS), harvest -> `sources.jsonl`, download tooling (yt-dlp / `ia`), pHash dedup, shot detection, quality scoring |
| 2 | infra | `infra/` | Sonnet | Modal app(s) + volumes, GH Actions `workflow_dispatch` runner, job spec format, smoke-test GPU job, docs |
| 3 | geo | `wtc4d/geo` | Opus | Lower Manhattan as of 2001-09-11: NYC 3D building model filtered to 2001, WTC complex parametric meshes, verified landmark registry (50+), glTF export, rubble surface from NOAA LiDAR |
| 4 | sync | `wtc4d/sync` | Opus | Absolute-time estimation: TV metadata, clock OCR, event-anchor detection, audio xcorr, solar shadow timing; fusion -> `TimeEstimate` |
| 5 | camreg | `wtc4d/camreg` | Opus | Landmark PnP + focal, render-and-match against prior, per-frame tracking with smoke masks, camera priors registry; synthetic-render tests |
| 6 | procedural | `wtc4d/procedural` | Sonnet | Parametric 4D baseline: tower state machine, plume model (wind), collapse animation, gaussian export for any `t`, `SceneManifest` sample |
| 7 | recon | `wtc4d/recon` | Opus | gsplat-based static training with fixed registered cameras + mesh init + dynamic masks; CPU reference rasterizer for tests; 4D design + prototype |
| 8 | viewer | `web/` | Sonnet | Three.js splat viewer: timeline scrubber with event markers, epoch switching, camera frusta -> source frame overlay, map inset |
| 9 | docs | `docs/` | Sonnet | ethics, licensing/fair-use policy, investigator methodology, contributor onboarding |

## 5. Phase 2 (after phase 1 merges)

- Run corpus harvest + download at scale on Modal; build the frame store.
- Register the top ~200 shots; publish `poses.jsonl` and a coverage map.
- Train E2 static splat (both burning) — first real proof of concept.
- Time-align the WTC2 collapse window across all cameras; train a 4D splat.
- Viewer release with procedural baseline + E2 splat + collapse 4D chunk.

## 6. Phase 3

- Full timeline 4D (all windows), smoke plume gaussians per minute.
- Investigator features: per-gaussian provenance (which frames support it),
  uncertainty visualisation, "align my footage" upload flow.
- Public dataset release (poses, times, manifests) on Hugging Face.

## 7. Guardrails

- Never commit media; catalogue and pointer only.  Derived works only.
- Respect victims: no emphasis on individuals; content warnings in viewer.
- Every number has a source and a sigma.

## 8. Phase 1 execution log

Spawned 2026-09-13 by the orchestrator session. Each session works on its own
branch, opens a PR to `main`, and squash-merges it when CI is green.

| workstream | model | branch | session |
|---|---|---|---|
| corpus | Sonnet | `claude/ws-corpus` | `session_019TBteADJeDdrjh7DpmfBHp` |
| infra | Sonnet | `claude/ws-infra` | `session_01Vf1sSowYy5zCfBpgeaBr9C` |
| geo | Opus | `claude/ws-geo` | `session_01H5YB5HiyiMTLSknGbCT1kv` |
| sync | Opus | `claude/ws-sync` | `session_01PMJV3ZDjoTvWc6yyL77K2X` |
| camreg | Opus | `claude/ws-camreg` | `session_01B3SCJAEYLV9k53wFwvQxQx` |
| procedural | Sonnet | `claude/ws-procedural` | `session_01CcWbrXmyRtktqrHnw8XBFK` |
| recon | Opus | `claude/ws-recon` | `session_017iVMdP2SYy1uvHQeqDYASq` |
| viewer | Sonnet | `claude/ws-viewer` | `session_01SUxZAfi1ekGqiBPb37R84P` |
| docs | Sonnet | `claude/ws-docs` | `session_01Y6fwaEFdsHBBaBx9VUHkue` |

Blockers outside the sessions' control:

- Modal is unreachable from cloud sessions (gRPC through the egress proxy is
  not supported). GPU jobs run through GitHub Actions; the repository owner
  must add the secrets `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`
  (Settings -> Secrets and variables -> Actions).
- Public hosting of large derived data (splats, frame stores) needs a
  Hugging Face token or a bucket; not required for phase 1.
