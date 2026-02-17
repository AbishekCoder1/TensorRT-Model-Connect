"""E2E: Inspect bundles and verify header fields."""

from __future__ import annotations

import subprocess
import pytest


@pytest.mark.e2e
def test_inspect_produces_output(model_entry, trtf_binary, ld_library_path):
    """trtf inspect <bundle> should produce valid output."""
    env = {"LD_LIBRARY_PATH": ld_library_path}
    result = subprocess.run(
        [str(trtf_binary), "inspect", model_entry["bundle_path"]],
        capture_output=True, text=True, timeout=30, env=env)
    assert result.returncode == 0, f"inspect failed: {result.stderr}"
    assert len(result.stdout.strip()) > 0, "inspect produced no output"


@pytest.mark.e2e
def test_inspect_shows_runtime_strategy(model_entry, trtf_binary, ld_library_path):
    """Inspect output should mention the runtime strategy."""
    env = {"LD_LIBRARY_PATH": ld_library_path}
    result = subprocess.run(
        [str(trtf_binary), "inspect", model_entry["bundle_path"]],
        capture_output=True, text=True, timeout=30, env=env)
    assert result.returncode == 0
    expected = model_entry.get("runtime_strategy", "decoder_kv_cache")
    assert expected in result.stdout, (
        f"Expected runtime_strategy '{expected}' in inspect output")
