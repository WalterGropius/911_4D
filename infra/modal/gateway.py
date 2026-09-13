"""Optional HTTPS job-submission gateway.

GitHub Actions dispatch (``modal-job.yml``) is the primary, always-available
way to run jobs (see ``infra/README.md``). This module is a convenience for
later: once deployed, any HTTPS-capable client (including a Claude Code
cloud session, which cannot reach Modal's gRPC API directly) can submit a
job with a plain ``curl``/``requests`` call instead of going through GitHub
Actions.

It is NOT deployed automatically — ``modal-deploy.yml`` only deploys
``jobs.py``. Deploy it manually, once, after creating the bearer-token
secret it checks requests against:

    modal secret create wtc4d-gateway GATEWAY_TOKEN=$(openssl rand -hex 32)
    modal deploy infra/modal/gateway.py

Then:

    curl -X POST https://<workspace>--wtc4d-submit.modal.run \\
         -H "Authorization: Bearer <GATEWAY_TOKEN>" \\
         -H "Content-Type: application/json" \\
         -d '{"kind": "smoke_test_gpu"}'
    # -> {"job_id": "...", "call_id": "..."}

    curl "https://<workspace>--wtc4d-status.modal.run?call_id=<call_id>" \\
         -H "Authorization: Bearer <GATEWAY_TOKEN>"
    # -> {"status": "running"} | {"status": "done", "result": {...}}
"""

from __future__ import annotations

import modal
from fastapi import Header

from infra.modal.common import app, volume
from infra.modal.jobs import JobSpec, _dispatch_for

gateway_secret = modal.Secret.from_name("wtc4d-gateway")

# Small, separate image: fastapi is only needed by the gateway endpoints,
# not by every job container.
gateway_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("fastapi[standard]==0.115.0", "pydantic==2.9.2")
    .add_local_python_source("wtc4d")
    .add_local_python_source("infra")
)


def _check_auth(authorization: str | None) -> None:
    import os

    from fastapi import HTTPException

    expected = os.environ.get("GATEWAY_TOKEN")
    if not expected:
        raise HTTPException(status_code=500, detail="GATEWAY_TOKEN not configured server-side")
    if authorization != f"Bearer {expected}":
        raise HTTPException(status_code=401, detail="missing or invalid bearer token")


@app.function(image=gateway_image, secrets=[gateway_secret], volumes={"/data": volume})
@modal.fastapi_endpoint(method="POST", label="wtc4d-submit")
def submit(spec: dict, authorization: str | None = Header(default=None)) -> dict:
    """Validate + spawn a job asynchronously; returns immediately with a
    ``call_id`` to poll via ``status``."""
    _check_auth(authorization)
    job_spec = JobSpec.model_validate(spec)
    fn = _dispatch_for(job_spec)
    call = fn.spawn(spec)
    return {"job_id": job_spec.resolved_job_id(), "call_id": call.object_id}


@app.function(image=gateway_image, secrets=[gateway_secret])
@modal.fastapi_endpoint(method="GET", label="wtc4d-status")
def status(call_id: str, authorization: str | None = Header(default=None)) -> dict:
    """Poll a previously-submitted job. Non-blocking: returns immediately
    with ``running`` if the result isn't ready yet."""
    _check_auth(authorization)
    call = modal.FunctionCall.from_id(call_id)
    try:
        result = call.get(timeout=0)
    except TimeoutError:
        return {"status": "running", "call_id": call_id}
    return {"status": "done", "call_id": call_id, "result": result}
