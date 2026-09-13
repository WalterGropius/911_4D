"""Trigger ``modal-job.yml`` on GitHub Actions and wait for the result.

This is the escape hatch for anywhere that cannot reach Modal's gRPC API
directly (e.g. a Claude Code cloud session behind an HTTP-only egress
proxy) but does have plain HTTPS access to the GitHub REST API. It has no
dependency on ``modal`` or ``wtc4d`` itself — only the standard library and
``requests`` — so it can be copied/run anywhere a job needs triggering from.

Usage::

    python -m infra.dispatch --spec spec.json --gpu A10G --timeout-min 30
    python -m infra.dispatch --spec '{"kind": "smoke_test_gpu"}' --gpu none

Auth: reads a token from ``GH_TOKEN`` or ``GITHUB_TOKEN``. It needs the
``actions:write`` permission (a fine-grained PAT, or ``contents:
read``+``actions: write`` for a GitHub App / Actions token) to dispatch and
read the run; a read-only token gets a 403 on the dispatch call, in which
case this prints the equivalent ``gh`` command and the GitHub MCP tool call
so the caller can do it through a channel that does have write access.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

API_ROOT = "https://api.github.com"
DEFAULT_WORKFLOW = "modal-job.yml"


class DispatchError(RuntimeError):
    pass


@dataclass
class GhClient:
    repo: str  # "owner/name"
    token: str

    @property
    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def _request(self, method: str, path: str, **kwargs) -> requests.Response:
        url = f"{API_ROOT}/repos/{self.repo}{path}"
        resp = requests.request(method, url, headers=self.headers, timeout=30, **kwargs)
        return resp


def _repo_from_git_remote() -> str | None:
    try:
        out = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    # git@github.com:owner/repo.git  or  https://github.com/owner/repo.git
    out = out.removesuffix(".git")
    if out.startswith("git@github.com:"):
        return out.split("git@github.com:", 1)[1]
    if "github.com/" in out:
        return out.split("github.com/", 1)[1]
    return None


def resolve_repo(explicit: str | None) -> str:
    repo = explicit or os.environ.get("GITHUB_REPOSITORY") or _repo_from_git_remote()
    if not repo:
        raise DispatchError("could not determine the repo (owner/name); pass --repo explicitly")
    return repo


def resolve_token() -> str:
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        raise DispatchError(
            "no GitHub token found: set GH_TOKEN or GITHUB_TOKEN in the environment"
        )
    return token


def _fallback_message(repo: str, workflow: str, ref: str, inputs: dict[str, Any]) -> str:
    gh_inputs = " ".join(f"-f {k}={json.dumps(str(v))}" for k, v in inputs.items())
    return (
        "GitHub API call failed (likely a token without 'actions: write').\n"
        "Equivalent gh CLI command:\n"
        f"  gh workflow run {workflow} --repo {repo} --ref {ref} {gh_inputs}\n"
        "Equivalent GitHub MCP tool call:\n"
        f'  actions_run_trigger(owner="{repo.split("/")[0]}", repo="{repo.split("/")[1]}", '
        f'workflow_id="{workflow}", ref="{ref}", inputs={json.dumps(inputs)})'
    )


def dispatch_workflow(client: GhClient, workflow: str, ref: str, inputs: dict[str, Any]) -> float:
    """POST the workflow_dispatch event. Returns the dispatch timestamp
    (epoch seconds) used to find the resulting run."""
    dispatched_at = time.time()
    resp = client._request(
        "POST",
        f"/actions/workflows/{workflow}/dispatches",
        json={"ref": ref, "inputs": inputs},
    )
    if resp.status_code == 403:
        raise DispatchError(
            "403 Forbidden dispatching the workflow.\n"
            + _fallback_message(client.repo, workflow, ref, inputs)
        )
    if resp.status_code >= 400:
        raise DispatchError(f"dispatch failed: {resp.status_code} {resp.text}")
    return dispatched_at


def find_run(
    client: GhClient,
    workflow: str,
    ref: str,
    after: float,
    *,
    poll_interval_s: float = 3.0,
    max_wait_s: float = 60.0,
) -> dict:
    """Poll the workflow's run list until a run created at/after
    ``after`` shows up (GitHub does not return the run id from the
    dispatch call itself)."""
    deadline = time.time() + max_wait_s
    while time.time() < deadline:
        resp = client._request(
            "GET",
            f"/actions/workflows/{workflow}/runs",
            params={"event": "workflow_dispatch", "branch": ref, "per_page": 5},
        )
        resp.raise_for_status()
        for run in resp.json().get("workflow_runs", []):
            created_dt = datetime.strptime(run["created_at"], "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=UTC
            )
            if created_dt.timestamp() >= after - 5:
                return run
        time.sleep(poll_interval_s)
    raise DispatchError("timed out waiting for the dispatched run to appear")


def poll_run(
    client: GhClient, run_id: int, *, poll_interval_s: float = 10.0, max_wait_s: float = 3600.0
) -> dict:
    deadline = time.time() + max_wait_s
    while time.time() < deadline:
        resp = client._request("GET", f"/actions/runs/{run_id}")
        resp.raise_for_status()
        run = resp.json()
        if run["status"] == "completed":
            return run
        time.sleep(poll_interval_s)
    raise DispatchError(f"run {run_id} did not complete within {max_wait_s}s")


def download_artifacts(client: GhClient, run_id: int, dest_dir: Path) -> list[Path]:
    resp = client._request("GET", f"/actions/runs/{run_id}/artifacts")
    resp.raise_for_status()
    dest_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for artifact in resp.json().get("artifacts", []):
        dl = client._request(
            "GET", f"/actions/artifacts/{artifact['id']}/zip", allow_redirects=True
        )
        dl.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(dl.content)) as zf:
            out = dest_dir / artifact["name"]
            zf.extractall(out)
            paths.append(out)
    return paths


def run(
    *,
    spec_json: str,
    gpu: str,
    timeout_min: int,
    ref: str,
    repo: str | None,
    workflow: str,
    out_dir: Path,
) -> dict:
    client = GhClient(repo=resolve_repo(repo), token=resolve_token())
    inputs = {"job_spec": spec_json, "gpu": gpu, "timeout_min": str(timeout_min), "ref": ref}

    dispatched_at = dispatch_workflow(client, workflow, ref, inputs)
    print(f"dispatched {workflow} on {client.repo}@{ref}", file=sys.stderr)

    run_info = find_run(client, workflow, ref, dispatched_at)
    print(f"run: {run_info['html_url']}", file=sys.stderr)

    completed = poll_run(client, run_info["id"])
    print(f"conclusion: {completed['conclusion']}", file=sys.stderr)

    artifact_paths = download_artifacts(client, run_info["id"], out_dir)
    result: dict[str, Any] = {
        "run_url": completed["html_url"],
        "conclusion": completed["conclusion"],
        "artifacts": [str(p) for p in artifact_paths],
    }
    for p in artifact_paths:
        result_json = p / "result.json"
        if result_json.exists():
            result["result"] = json.loads(result_json.read_text())
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, help="job spec JSON, or a path to a JSON file")
    parser.add_argument(
        "--gpu", default="none", choices=["none", "T4", "L4", "A10G", "A100", "H100"]
    )
    parser.add_argument("--timeout-min", type=int, default=15)
    parser.add_argument("--ref", default="main")
    parser.add_argument(
        "--repo", default=None, help="owner/name; inferred from git remote if omitted"
    )
    parser.add_argument("--workflow", default=DEFAULT_WORKFLOW)
    parser.add_argument("--out-dir", default="./modal-job-artifacts")
    args = parser.parse_args(argv)

    spec_arg = args.spec
    if os.path.isfile(spec_arg):
        spec_json = Path(spec_arg).read_text()
    else:
        spec_json = spec_arg
    try:
        json.loads(spec_json)  # validate it parses before making any network call
    except json.JSONDecodeError as exc:
        print(f"error: --spec is not valid JSON: {exc}", file=sys.stderr)
        return 1

    try:
        result = run(
            spec_json=spec_json,
            gpu=args.gpu,
            timeout_min=args.timeout_min,
            ref=args.ref,
            repo=args.repo,
            workflow=args.workflow,
            out_dir=Path(args.out_dir),
        )
    except DispatchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2))
    return 0 if result.get("conclusion") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
