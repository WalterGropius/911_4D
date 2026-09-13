"""Validate infra's GitHub Actions workflow YAML files.

Structural checks always run (pure ``yaml.safe_load``, no network, no extra
binaries). ``yamllint`` and, if present on PATH, ``actionlint`` add a
second, best-effort layer of validation.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

WORKFLOWS_DIR = Path(__file__).resolve().parents[1] / ".github" / "workflows"
MODAL_JOB = WORKFLOWS_DIR / "modal-job.yml"
MODAL_DEPLOY = WORKFLOWS_DIR / "modal-deploy.yml"


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def test_workflow_files_exist():
    assert MODAL_JOB.is_file()
    assert MODAL_DEPLOY.is_file()


def test_modal_job_parses_and_has_expected_inputs():
    doc = _load(MODAL_JOB)
    # PyYAML parses the bare `on:` key as boolean True; workflows use it as
    # a literal keyword, so check both spellings.
    on = doc.get("on", doc.get(True))
    inputs = on["workflow_dispatch"]["inputs"]

    assert set(inputs) == {"job_spec", "gpu", "timeout_min", "ref"}
    assert inputs["job_spec"]["required"] is True
    assert inputs["gpu"]["type"] == "choice"
    assert set(inputs["gpu"]["options"]) == {"none", "T4", "L4", "A10G", "A100", "H100"}
    assert inputs["ref"]["default"] == "main"


def test_modal_job_uses_secrets_and_uploads_artifacts():
    text = MODAL_JOB.read_text()
    assert "secrets.MODAL_TOKEN_ID" in text
    assert "secrets.MODAL_TOKEN_SECRET" in text
    assert "actions/upload-artifact@v4" in text
    assert "modal run infra/modal/jobs.py" in text


def test_modal_deploy_triggers_on_infra_and_wtc4d_changes():
    doc = _load(MODAL_DEPLOY)
    on = doc.get("on", doc.get(True))
    push = on["push"]
    assert push["branches"] == ["main"]
    assert set(push["paths"]) == {"infra/**", "wtc4d/**"}


def test_modal_deploy_guards_on_missing_secrets():
    text = MODAL_DEPLOY.read_text()
    assert "skip=true" in text
    assert "modal deploy infra/modal/jobs.py" in text


@pytest.mark.parametrize("path", [MODAL_JOB, MODAL_DEPLOY])
def test_yamllint_reports_no_errors(path):
    from yamllint import linter
    from yamllint.config import YamlLintConfig

    # Relaxed config: we only care about YAML-correctness errors here
    # (duplicate keys, bad indentation, truthy/document issues), not house
    # style like line length or comment spacing.
    config = YamlLintConfig(
        "extends: relaxed\n"
        "rules:\n"
        "  line-length: disable\n"
        "  truthy: disable\n"
        "  comments: disable\n"
        "  comments-indentation: disable\n"
    )
    problems = list(linter.run(path.read_text(), config))
    errors = [p for p in problems if p.level == "error"]
    assert errors == [], f"{path.name}: {errors}"


@pytest.mark.parametrize("path", [MODAL_JOB, MODAL_DEPLOY])
def test_actionlint_if_available(path):
    binary = shutil.which("actionlint")
    if not binary:
        pytest.skip("actionlint not installed; yamllint + structural checks still ran")
    proc = subprocess.run([binary, str(path)], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stdout + proc.stderr
