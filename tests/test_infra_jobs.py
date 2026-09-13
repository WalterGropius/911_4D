"""Tests for infra.modal.jobs: spec validation, dispatch, handler logic.

Nothing here touches the real Modal API or the real ``/data`` volume:
``WTC4D_DATA_ROOT`` redirects job I/O into a tmp dir, and subprocess calls
(ffmpeg/yt-dlp/ia/nvidia-smi) are monkeypatched or simply absent (which the
handlers must degrade gracefully from, since CI runners don't have them
either).
"""

from __future__ import annotations

import json
import sys
import types

import pytest
from pydantic import ValidationError


@pytest.fixture(autouse=True)
def _no_modal_token(monkeypatch):
    monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)


@pytest.fixture()
def data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("WTC4D_DATA_ROOT", str(tmp_path))
    return tmp_path


def test_import_does_not_touch_modal_network(monkeypatch):
    # Re-import fresh with no Modal credentials at all in the environment;
    # constructing App/Volume/Image objects must stay purely local.
    for mod in list(sys.modules):
        if mod == "infra.modal.jobs" or mod == "infra.modal.common":
            del sys.modules[mod]
    import infra.modal.jobs as jobs  # noqa: PLC0415

    assert jobs.app is not None
    assert jobs.volume is not None


def test_job_spec_valid_kinds():
    from infra.modal.jobs import JobSpec

    for kind in ("smoke_test_gpu", "download_sources", "extract_frames", "shell", "run_colmap"):
        spec = JobSpec(kind=kind)
        assert spec.kind == kind
        assert spec.timeout_s > 0


def test_job_spec_rejects_unknown_kind():
    from infra.modal.jobs import JobSpec

    with pytest.raises(ValidationError):
        JobSpec(kind="not_a_real_kind")


def test_job_spec_rejects_bad_timeout():
    from infra.modal.jobs import JobSpec

    with pytest.raises(ValidationError):
        JobSpec(kind="shell", timeout_s=0)


def test_resolved_job_id_defaults_and_is_stable():
    from infra.modal.jobs import JobSpec

    spec = JobSpec(kind="smoke_test_gpu", job_id="my-job")
    assert spec.resolved_job_id() == "my-job"

    auto = JobSpec(kind="smoke_test_gpu")
    job_id = auto.resolved_job_id()
    assert job_id.startswith("smoke_test_gpu-")


@pytest.mark.parametrize(
    ("kind", "params", "expected"),
    [
        ("smoke_test_gpu", {}, True),
        ("run_colmap", {}, True),
        ("download_sources", {}, False),
        ("extract_frames", {}, False),
        ("shell", {}, False),
        ("shell", {"gpu": True}, True),
    ],
)
def test_needs_gpu(kind, params, expected):
    from infra.modal.jobs import JobSpec

    assert JobSpec(kind=kind, params=params).needs_gpu() is expected


def test_dispatch_for_picks_cpu_and_gpu_functions(monkeypatch):
    # Function.info/.tag are deprecated (removed in modal 1.6), so identify
    # the chosen function by monkeypatching with_options instead of
    # inspecting it after the fact.
    from infra.modal.jobs import JobSpec, _dispatch_for, run_job_cpu, run_job_gpu

    cpu_calls: list[dict] = []
    gpu_calls: list[dict] = []
    monkeypatch.setattr(
        run_job_cpu, "with_options", lambda **kw: cpu_calls.append(kw) or run_job_cpu
    )
    monkeypatch.setattr(
        run_job_gpu, "with_options", lambda **kw: gpu_calls.append(kw) or run_job_gpu
    )

    _dispatch_for(JobSpec(kind="download_sources", params={"items": []}))
    assert cpu_calls and not gpu_calls

    cpu_calls.clear()
    gpu_calls.clear()
    _dispatch_for(JobSpec(kind="smoke_test_gpu"))
    assert gpu_calls and not cpu_calls
    assert gpu_calls[0]["gpu"] == "A10G"  # DEFAULT_GPU

    cpu_calls.clear()
    gpu_calls.clear()
    _dispatch_for(JobSpec(kind="download_sources", params={"items": []}, gpu="A100"))
    assert gpu_calls and not cpu_calls
    assert gpu_calls[0]["gpu"] == "A100"


def test_job_spec_rejects_bad_gpu_choice():
    from infra.modal.jobs import JobSpec

    with pytest.raises(ValidationError):
        JobSpec(kind="smoke_test_gpu", gpu="RTX3090")


def test_run_job_writes_log_and_result(data_root):
    from infra.modal.jobs import run_job

    result = run_job({"kind": "smoke_test_gpu", "job_id": "t-smoke"})

    work_dir = data_root / "work" / "t-smoke"
    assert (work_dir / "log.txt").exists()
    assert (work_dir / "result.json").exists()
    on_disk = json.loads((work_dir / "result.json").read_text())
    assert on_disk["job_id"] == "t-smoke"
    assert on_disk == result


def test_smoke_test_gpu_degrades_without_a_gpu(data_root):
    from infra.modal.jobs import run_job

    result = run_job({"kind": "smoke_test_gpu", "job_id": "t-smoke2"})
    assert result["kind"] == "smoke_test_gpu"
    # Whether `torch` itself is *installed* now depends on which workstream
    # extras are present (e.g. `recon` needs a CPU-installable torch for its
    # reference rasteriser) -- that is no longer something this environment
    # guarantees either way, so it is not asserted here. What the smoke test
    # actually exists to check is that it correctly reports "no usable GPU"
    # when there is none, which holds regardless of torch's presence: with
    # torch installed, `cuda_available` comes from `torch.cuda.is_available()`
    # (false without a GPU); without torch, the handler's `except ImportError`
    # path reports the same `False`.
    assert result["cuda_available"] is False
    assert result["ok"] is False  # a real smoke test SHOULD fail without a GPU


def test_extract_frames_requires_params(data_root):
    from infra.modal.jobs import run_job

    result = run_job({"kind": "extract_frames", "job_id": "t-extract-bad"})
    assert result["ok"] is False
    assert "shot_id" in result["error"]


def test_extract_frames_builds_expected_ffmpeg_command(data_root, monkeypatch):
    from infra.modal import jobs as J

    captured = {}

    def fake_run_subprocess(cmd, *, log, cwd=None):
        captured["cmd"] = cmd
        return {"returncode": 0, "stdout": "", "stderr": ""}

    monkeypatch.setattr(J, "_run_subprocess", fake_run_subprocess)

    result = J.run_job(
        {
            "kind": "extract_frames",
            "job_id": "t-extract",
            "params": {
                "shot_id": "shot001",
                "source_path": "/data/raw/src/video.mp4",
                "fps": 10,
                "crop": "640:480:0:0",
                "deinterlace": True,
            },
        }
    )

    assert result["ok"] is True
    cmd = captured["cmd"]
    assert cmd[0] == "ffmpeg"
    assert "/data/raw/src/video.mp4" in cmd
    vf = cmd[cmd.index("-vf") + 1]
    assert "yadif" in vf
    assert "fps=10" in vf
    assert "crop=640:480:0:0" in vf
    assert str(data_root / "frames" / "shot001" / "%06d.jpg") in cmd


def test_download_sources_requires_items(data_root):
    from infra.modal.jobs import run_job

    result = run_job({"kind": "download_sources", "job_id": "t-dl-bad"})
    assert result["ok"] is False


def test_download_sources_falls_back_to_yt_dlp_and_ia(data_root, monkeypatch):
    from infra.modal import jobs as J

    commands = []

    def fake_run_subprocess(cmd, *, log, cwd=None):
        commands.append(cmd)
        return {"returncode": 0, "stdout": "", "stderr": ""}

    monkeypatch.setattr(J, "_run_subprocess", fake_run_subprocess)

    result = J.run_job(
        {
            "kind": "download_sources",
            "job_id": "t-dl",
            "params": {
                "items": [
                    {"id": "yt-a", "url": "https://youtube.com/watch?v=abc"},
                    {"id": "ia-b", "url": "https://archive.org/details/some-video"},
                ]
            },
        }
    )

    assert result["ok"] is True
    assert commands[0][0] == "yt-dlp"
    assert commands[1][0] == "ia"
    assert commands[1][1] == "download"


def test_download_sources_uses_wtc4d_corpus_when_importable(data_root, monkeypatch):
    from infra.modal import jobs as J

    calls = []
    fake_corpus = types.ModuleType("wtc4d.corpus")
    fake_corpus.download = lambda url, dest: calls.append((url, dest))
    monkeypatch.setitem(sys.modules, "wtc4d.corpus", fake_corpus)

    result = J.run_job(
        {
            "kind": "download_sources",
            "job_id": "t-dl-corpus",
            "params": {"items": [{"id": "src-1", "url": "https://example.org/a.mp4"}]},
        }
    )

    assert result["ok"] is True
    assert result["items"][0]["method"] == "wtc4d.corpus"
    assert len(calls) == 1


def test_shell_kind_calls_entrypoint_with_sub_spec(data_root, monkeypatch):
    from infra.modal import jobs as J

    fake_mod = types.ModuleType("fake_recon_job")

    def fake_run(spec):
        assert spec == {"scene_id": "E2"}
        return {"trained": True}

    fake_mod.run = fake_run
    monkeypatch.setitem(sys.modules, "fake_recon_job", fake_mod)

    result = J.run_job(
        {
            "kind": "shell",
            "job_id": "t-shell",
            "params": {"entrypoint": "fake_recon_job:run", "scene_id": "E2"},
        }
    )

    assert result["ok"] is True
    assert result["trained"] is True


def test_shell_kind_requires_entrypoint(data_root):
    from infra.modal.jobs import run_job

    result = run_job({"kind": "shell", "job_id": "t-shell-bad", "params": {}})
    assert result["ok"] is False
    assert "entrypoint" in result["error"]


def test_run_colmap_placeholder(data_root):
    from infra.modal.jobs import run_job

    result = run_job({"kind": "run_colmap", "job_id": "t-colmap"})
    assert result["ok"] is False
    assert result["status"] == "not_implemented"


def test_list_dir(data_root):
    from infra.modal.jobs import list_dir

    (data_root / "work").mkdir()
    (data_root / "raw").mkdir()
    assert list_dir() == ["raw", "work"]
    assert list_dir("does-not-exist") == []
