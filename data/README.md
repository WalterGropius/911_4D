# data/

Small, versionable derived data only.  **Never commit footage, frames, point
clouds or splats.**  Those live on the Modal volume `wtc4d-data` (see
`infra/`) and, for public release, on a Hugging Face dataset repo or GitHub
Releases.

| path | contents | owner |
|---|---|---|
| `data/manifests/` | `sources.jsonl`, `shots.jsonl` (see `wtc4d.schema.corpus`) | corpus |
| `data/time/` | `time_estimates.jsonl` keyed by shot/frame | sync |
| `data/geo/` | landmark registry, small city model (<20 MB) | geo |
| `data/cameras/` | `poses.jsonl` (`wtc4d.schema.camera.CameraPose`), `priors.yaml` | camreg |
| `data/scenes/` | viewer manifests (`wtc4d.schema.scene.SceneManifest`) | recon / procedural |

Layout on the compute volume (`/data` inside Modal containers):

```
/data/raw/<source_id>/...          original downloads (immutable)
/data/frames/<shot_id>/%06d.jpg    extracted frames
/data/work/<job>/...               intermediate outputs
/data/splats/<epoch_or_window>/    trained splats
```
