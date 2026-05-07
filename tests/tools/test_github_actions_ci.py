"""Tests for GitHub Actions CI wiring."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def test_workflows_define_shared_hf_cache_env() -> None:
    for workflow in ("nightly.yml", "trtmc-ci.yml"):
        text = (REPO_ROOT / ".github" / "workflows" / workflow).read_text()
        assert "TRTMC_STORAGE_ROOT:" in text
        assert "HF_HOME:" in text
        assert "HF_HUB_CACHE:" in text
        assert "HUGGINGFACE_HUB_CACHE:" in text
        assert "HF_MODULES_CACHE:" in text
        assert "/workspace/users/yifeif/tensorrt-model-connect/hf-cache" in text


def test_github_stage_wrapper_mounts_and_exports_hf_cache_env() -> None:
    text = (REPO_ROOT / ".github" / "scripts" / "run-gha-stage.sh").read_text()
    for name in (
        "TRTMC_STORAGE_ROOT",
        "HF_HOME",
        "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "HF_MODULES_CACHE",
    ):
        assert f'mkdir_if_set "${{{name}:-}}"' in text
        assert f"-e {name}" in text


def test_shared_setup_action_creates_hf_cache_dirs() -> None:
    text = (REPO_ROOT / ".github" / "actions" / "setup-trtmc" / "action.yml").read_text()
    assert '"${HF_HOME:-}"' in text
    assert '"${HF_HUB_CACHE:-}"' in text
    assert '"${HUGGINGFACE_HUB_CACHE:-}"' in text
    assert '"${HF_MODULES_CACHE:-}"' in text
