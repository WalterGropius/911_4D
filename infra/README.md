# infra/

Compute lives on [Modal](https://modal.com).  This directory is owned by the
`infra` workstream: Modal apps (`infra/modal/`), the GitHub Actions dispatch
workflows that run/deploy them, the `infra/dispatch.py` HTTP-only client,
and this runbook.

Important constraint: Claude Code cloud sessions cannot reach Modal's gRPC
API directly (the egress proxy does not support gRPC — attempts fail with
"Could not connect to the Modal server").  Modal jobs are therefore launched
from GitHub Actions (`workflow_dispatch`), which does have full network
access, using the repository secrets `MODAL_TOKEN_ID` and
`MODAL_TOKEN_SECRET`.  A contributor with those tokens configured locally
can also just run `modal run infra/modal/jobs.py ...` directly.

## 1. One-time setup: add the Modal secrets

1. Create a Modal account/workspace, then create an API token: Modal
   dashboard -> Settings -> API Tokens -> "New token".  This gives you a
   `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` pair.
2. In the GitHub repo: **Settings -> Secrets and variables -> Actions ->
   New repository secret**.  Add both:
   - `MODAL_TOKEN_ID`
   - `MODAL_TOKEN_SECRET`
3. That's it — both workflows below check for these and fail with a clear
   message if they're missing, instead of a confusing Modal auth error.

## 2. First deploy

`modal-deploy.yml` runs automatically on every push to `main` that touches
`infra/**` or `wtc4d/**`, and deploys `infra/modal/jobs.py` (which pulls in
`infra/modal/common.py`'s app/images/volume as a side effect of import).
This registers the `run_job_cpu`, `run_job_gpu` and `list_volume` functions
as long-lived Modal Functions so they can be invoked without redeploying
each time, and provisions the `wtc4d-data` volume on first run
(`Volume.from_name(..., create_if_missing=True)`).

You can also trigger it manually: Actions tab -> "modal-deploy" ->
"Run workflow", or `workflow_dispatch` via the API/MCP tool (see below).

The optional HTTPS gateway (`infra/modal/gateway.py`, section 6) is **not**
part of this automatic deploy — it needs a Secret created first, so it's a
manual one-time `modal deploy infra/modal/gateway.py` when you actually want
it.

## 3. Volume layout

The volume `wtc4d-data` is mounted at `/data` in every job container:

```
/data/raw/<source_id>/...          original downloads (immutable)
/data/frames/<shot_id>/%06d.jpg     extracted frames
/data/work/<job_id>/log.txt         per-job log
/data/work/<job_id>/result.json     per-job result (also returned directly)
/data/splats/<epoch_or_window>/     trained splats
```

See `data/README.md` for the small, versioned repo-side counterpart
(`data/manifests/`, `data/geo/`, etc.) — that's separate from this volume
and lives in git.

## 4. Job spec format

Every job is one JSON object, validated against `infra.modal.jobs.JobSpec`:

```json
{
  "kind": "smoke_test_gpu",     // smoke_test_gpu | download_sources | extract_frames | shell | run_colmap
  "job_id": "optional-explicit-id",
  "gpu": "none",                // none | T4 | L4 | A10G | A100 | H100
  "timeout_s": 900,
  "params": { }                  // kind-specific, see infra/modal/jobs.py docstrings
}
```

Results always land at `/data/work/<job_id>/result.json` and are returned
directly by whichever dispatch method you used.  Every result has at least
`{"ok": bool, "job_id": ..., "kind": ...}`.

### Job kinds

- **`smoke_test_gpu`** — runs `nvidia-smi`, checks `torch.cuda.is_available()`,
  does a small CUDA matmul, writes a marker file to the volume.  `ok` is
  `false` if no usable GPU was found (this is the point of the test).
- **`download_sources`** — `params: {"items": [{"id": ..., "url": ...}]}`.
  Uses `wtc4d.corpus`'s downloader if that package is importable (once the
  `corpus` workstream lands it), otherwise falls back to `yt-dlp` (YouTube
  etc.) or `ia download` (archive.org) directly, writing to
  `/data/raw/<id>/`.
- **`extract_frames`** — `params: {"shot_id", "source_path", "fps", "crop",
  "deinterlace"}`.  Runs ffmpeg (deinterlace + fps + optional crop) to
  `/data/frames/<shot_id>/%06d.jpg`.
- **`shell`** — `params: {"entrypoint": "some.module:function", ...rest}`.
  Imports `some.module`, calls `function(rest)` (i.e. everything in
  `params` except `entrypoint`/`gpu`) inside the gpu or cpu image, and
  returns whatever it returns (or `{"ok": true}` if it returns `None`).
  **This is how every other workstream plugs compute in without touching
  `infra/`** — e.g. `recon` calls `wtc4d.recon.job:run` with
  `{"entrypoint": "wtc4d.recon.job:run", "scene_id": "E2", "gpu": true}`.
- **`run_colmap`** — placeholder; use `shell` with a `recon`-owned
  entrypoint instead until `recon` lands a real COLMAP pipeline.

Set `"gpu": true` inside `params` for a `shell` job that needs the GPU
image/a GPU attached; otherwise it runs on the (cheaper, faster-starting)
CPU image.

## 5. Submitting a job, three ways

### A. GitHub Actions UI

Actions tab -> "modal-job" -> "Run workflow".  Fill in `job_spec` (JSON),
`gpu`, `timeout_min`, `ref`.  Logs and `result.json` are attached as a
workflow artifact named `modal-job-<run_id>`, and the result JSON is also
printed in the run's step summary.

### B. `infra/dispatch.py` (stdlib + `requests` only — works from anywhere with plain HTTPS and a GitHub token)

```bash
export GH_TOKEN=...   # or GITHUB_TOKEN; needs `actions: write`
python -m infra.dispatch \
  --spec '{"kind": "smoke_test_gpu"}' \
  --gpu A10G --timeout-min 15 --ref main
```

Prints progress to stderr, then the final result JSON to stdout, and exits
non-zero if the run didn't conclude with `success`.  If the token can't
dispatch workflows (403 — needs `actions: write`), it prints the equivalent
`gh workflow run ...` command and the GitHub MCP `actions_run_trigger` call
so you can do it through a channel that does have write access.

### C. GitHub MCP tool (from a Claude Code session)

```
actions_run_trigger(owner="WalterGropius", repo="911_4D",
                     workflow_id="modal-job.yml", ref="main",
                     inputs={"job_spec": "{\"kind\": \"smoke_test_gpu\"}",
                             "gpu": "none", "timeout_min": "15", "ref": "main"})
```

Then poll with `actions_list` / `get_job_logs`, or just watch the Actions
UI.

### Reading logs

- Quick: the workflow's step summary and the `modal_output.log` /
  `result.json` artifact (method A/B above already fetch this for you).
- Full per-job log on the volume: `/data/work/<job_id>/log.txt` — fetch it
  with `modal volume get wtc4d-data work/<job_id>/log.txt .` (needs a
  working local Modal connection), or add a tiny `shell` job that reads and
  returns it.
- `infra.modal.jobs.list_volume(path)` (a deployed Modal function) lists any
  directory on the volume, e.g. to enumerate `work/` or `raw/`.

## 6. Optional: HTTPS gateway

`infra/modal/gateway.py` exposes `POST /submit` and `GET /status` as a
`@modal.fastapi_endpoint`, so once deployed, any HTTPS-capable client
(including a cloud session that cannot reach Modal's gRPC API) can submit
and poll jobs directly, without going through GitHub Actions at all. It is
**not** deployed by `modal-deploy.yml`; deploy it manually once you want it:

```bash
modal secret create wtc4d-gateway GATEWAY_TOKEN=$(openssl rand -hex 32)
modal deploy infra/modal/gateway.py
```

Then:

```bash
curl -X POST https://<workspace>--wtc4d-submit.modal.run \
     -H "Authorization: Bearer <GATEWAY_TOKEN>" \
     -H "Content-Type: application/json" \
     -d '{"kind": "smoke_test_gpu"}'
# {"job_id": "...", "call_id": "..."}

curl "https://<workspace>--wtc4d-status.modal.run?call_id=<call_id>" \
     -H "Authorization: Bearer <GATEWAY_TOKEN>"
# {"status": "running"} | {"status": "done", "result": {...}}
```

## 7. Cost notes (approximate, check the Modal pricing page for current rates)

GPUs are billed per second while the container is running, not per job
submitted — an idle `gpu="none"` job (downloads, ffmpeg) costs only CPU
time. Rough relative cost, cheapest to most expensive: **T4** (light
inference/smoke tests) < **L4** < **A10G** (good default for reconstruction
work: gsplat training, LightGlue matching) < **A100** < **H100** (only for
large/batched training that actually saturates it). Default to `none` or
`T4` for anything that isn't GPU-bound, and to `A10G` for real recon jobs
unless you've measured you need more. Always set a realistic `timeout_min`
— a stuck job on an A100 is the expensive failure mode, not a slow one.

## 8. How other workstreams plug in

You do not need to edit anything under `infra/`. Submit a `kind: "shell"`
job whose `params.entrypoint` points at a function in your own package,
e.g. `wtc4d.recon.job:run`, `wtc4d.camreg.job:run`. The function receives
everything in `params` (minus `entrypoint`/`gpu`) as a single dict argument
and can do whatever it needs — including reading/writing the shared
`/data` volume, since that's mounted at the same path in both the `shell`
job's container and every other job's. Set `"gpu": true` in `params` if
your function needs CUDA/a GPU attached (it then runs inside `gpu_image`,
which already has torch/gsplat/pycolmap/kornia/LightGlue/trimesh/pyrender —
see `infra/modal/common.py` for exact pins); omit it to run on the cheaper
CPU `base_image` (ffmpeg, yt-dlp, `ia`, opencv, numpy, pydantic).

## Verification status

After merging, this workstream triggered `modal-job.yml` on `main` with
`{"kind": "smoke_test_gpu"}` via the GitHub MCP `actions_run_trigger` tool
to confirm the whole path end-to-end. Since `MODAL_TOKEN_ID` /
`MODAL_TOKEN_SECRET` are not yet configured on the repo (this session
cannot add them — only a repo admin can, per section 1 above), the run is
expected to fail at the "Check Modal secrets" step with the explicit
`::error::` message described there, rather than a confusing Modal auth
failure. **The orchestrator should ask the repo owner to add the two
secrets**; once that's done, re-running the same workflow (or triggering it
again) should reach the real `modal run` step and produce a
`{"ok": true, "cuda_available": true, ...}` result for a GPU'd run.
