"""Tests for infra.dispatch: payload building, run discovery, error paths.

All HTTP is mocked (via a fake ``requests.Response``/``requests.request``);
these tests must never make a real network call.
"""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass, field

import pytest

from infra import dispatch


@dataclass
class FakeResponse:
    status_code: int
    _json: object = None
    text: str = ""
    content: bytes = b""

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise dispatch.requests.HTTPError(f"status {self.status_code}")


@dataclass
class FakeSession:
    """Records every call made through GhClient._request and replays a
    scripted queue of responses."""

    responses: list = field(default_factory=list)
    calls: list = field(default_factory=list)

    def request(self, method, url, headers=None, timeout=None, **kwargs):
        self.calls.append({"method": method, "url": url, "kwargs": kwargs})
        # Keep returning the last scripted response once the queue is down
        # to one, so callers that poll in a tight loop (with time.sleep
        # mocked out) don't run out of canned responses.
        if len(self.responses) > 1:
            return self.responses.pop(0)
        return self.responses[0]


def test_resolve_token_prefers_gh_token(monkeypatch):
    monkeypatch.setenv("GH_TOKEN", "tok-gh")
    monkeypatch.setenv("GITHUB_TOKEN", "tok-github")
    assert dispatch.resolve_token() == "tok-gh"


def test_resolve_token_falls_back_to_github_token(monkeypatch):
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.setenv("GITHUB_TOKEN", "tok-github")
    assert dispatch.resolve_token() == "tok-github"


def test_resolve_token_missing_raises(monkeypatch):
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    with pytest.raises(dispatch.DispatchError):
        dispatch.resolve_token()


def test_resolve_repo_explicit_wins(monkeypatch):
    monkeypatch.setenv("GITHUB_REPOSITORY", "env/repo")
    assert dispatch.resolve_repo("explicit/repo") == "explicit/repo"


def test_resolve_repo_from_env(monkeypatch):
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/name")
    assert dispatch.resolve_repo(None) == "owner/name"


def test_resolve_repo_raises_when_undeterminable(monkeypatch):
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    monkeypatch.setattr(dispatch, "_repo_from_git_remote", lambda: None)
    with pytest.raises(dispatch.DispatchError):
        dispatch.resolve_repo(None)


def test_dispatch_workflow_sends_expected_payload(monkeypatch):
    fake = FakeSession(responses=[FakeResponse(status_code=204)])
    monkeypatch.setattr(dispatch.requests, "request", fake.request)

    client = dispatch.GhClient(repo="owner/repo", token="tok")
    inputs = {"job_spec": '{"kind": "smoke_test_gpu"}', "gpu": "A10G", "timeout_min": "30"}
    dispatch.dispatch_workflow(client, "modal-job.yml", "main", inputs)

    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call["method"] == "POST"
    assert call["url"].endswith("/repos/owner/repo/actions/workflows/modal-job.yml/dispatches")
    assert call["kwargs"]["json"] == {"ref": "main", "inputs": inputs}


def test_dispatch_workflow_403_gives_actionable_message(monkeypatch):
    fake = FakeSession(responses=[FakeResponse(status_code=403, text="forbidden")])
    monkeypatch.setattr(dispatch.requests, "request", fake.request)

    client = dispatch.GhClient(repo="owner/repo", token="tok")
    inputs = {"job_spec": "{}", "gpu": "none", "timeout_min": "15"}

    with pytest.raises(dispatch.DispatchError) as excinfo:
        dispatch.dispatch_workflow(client, "modal-job.yml", "main", inputs)

    msg = str(excinfo.value)
    assert "gh workflow run modal-job.yml" in msg
    assert "actions_run_trigger" in msg
    assert "owner" in msg and "repo" in msg


def test_dispatch_workflow_other_error_raises(monkeypatch):
    fake = FakeSession(responses=[FakeResponse(status_code=500, text="boom")])
    monkeypatch.setattr(dispatch.requests, "request", fake.request)
    client = dispatch.GhClient(repo="owner/repo", token="tok")
    with pytest.raises(dispatch.DispatchError, match="500"):
        dispatch.dispatch_workflow(client, "modal-job.yml", "main", {})


def test_find_run_matches_recent_run(monkeypatch):
    import time as time_mod

    now = time_mod.time()
    fake = FakeResponse(
        status_code=200,
        _json={
            "workflow_runs": [
                {
                    "id": 42,
                    "created_at": time_mod.strftime("%Y-%m-%dT%H:%M:%SZ", time_mod.gmtime(now)),
                    "html_url": "https://github.com/owner/repo/actions/runs/42",
                }
            ]
        },
    )
    session = FakeSession(responses=[fake])
    monkeypatch.setattr(dispatch.requests, "request", session.request)

    client = dispatch.GhClient(repo="owner/repo", token="tok")
    run = dispatch.find_run(client, "modal-job.yml", "main", after=now - 2, max_wait_s=5)
    assert run["id"] == 42


def test_find_run_times_out_when_nothing_matches(monkeypatch):
    import time as time_mod

    old_run = FakeResponse(
        status_code=200,
        _json={"workflow_runs": [{"id": 1, "created_at": "2000-01-01T00:00:00Z"}]},
    )
    # Repeated identical response until deadline.
    session = FakeSession(responses=[old_run] * 5)
    monkeypatch.setattr(dispatch.requests, "request", session.request)
    monkeypatch.setattr(dispatch.time, "sleep", lambda s: None)

    client = dispatch.GhClient(repo="owner/repo", token="tok")
    with pytest.raises(dispatch.DispatchError, match="timed out"):
        dispatch.find_run(client, "modal-job.yml", "main", after=time_mod.time(), max_wait_s=0.01)


def test_poll_run_waits_for_completed(monkeypatch):
    running = FakeResponse(status_code=200, _json={"status": "in_progress"})
    done = FakeResponse(status_code=200, _json={"status": "completed", "conclusion": "success"})
    session = FakeSession(responses=[running, done])
    monkeypatch.setattr(dispatch.requests, "request", session.request)
    monkeypatch.setattr(dispatch.time, "sleep", lambda s: None)

    client = dispatch.GhClient(repo="owner/repo", token="tok")
    run = dispatch.poll_run(client, 42, poll_interval_s=0)
    assert run["conclusion"] == "success"
    assert len(session.calls) == 2


def _zip_bytes(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()


def test_download_artifacts_extracts_result_json(monkeypatch, tmp_path):
    list_resp = FakeResponse(
        status_code=200,
        _json={"artifacts": [{"id": 7, "name": "modal-job-1"}]},
    )
    zip_resp = FakeResponse(
        status_code=200,
        content=_zip_bytes({"result.json": json.dumps({"ok": True}).encode()}),
    )
    session = FakeSession(responses=[list_resp, zip_resp])
    monkeypatch.setattr(dispatch.requests, "request", session.request)

    client = dispatch.GhClient(repo="owner/repo", token="tok")
    paths = dispatch.download_artifacts(client, 42, tmp_path / "out")

    assert len(paths) == 1
    result = json.loads((paths[0] / "result.json").read_text())
    assert result == {"ok": True}


def test_main_rejects_bad_spec_json(capsys):
    rc = dispatch.main(["--spec", "{not json", "--repo", "owner/repo"])
    assert rc == 1
    assert "not valid JSON" in capsys.readouterr().err
