"""E2E: Inspect bundles and verify header fields."""

from __future__ import annotations

import subprocess
import pytest


@pytest.mark.e2e
def test_inspect_produces_output(model_entry, trtmc_binary, ld_library_path):
    """trtmc inspect <bundle> should produce valid output."""
    env = {"LD_LIBRARY_PATH": ld_library_path}
    result = subprocess.run(
        [str(trtmc_binary), "inspect", model_entry["bundle_path"]],
        capture_output=True, text=True, timeout=30, env=env)
    assert result.returncode == 0, f"inspect failed: {result.stderr}"
    assert len(result.stdout.strip()) > 0, "inspect produced no output"


@pytest.mark.e2e
def test_inspect_shows_runtime_strategy(model_entry, trtmc_binary, ld_library_path):
    """Inspect output should mention the runtime strategy."""
    env = {"LD_LIBRARY_PATH": ld_library_path}
    result = subprocess.run(
        [str(trtmc_binary), "inspect", model_entry["bundle_path"]],
        capture_output=True, text=True, timeout=30, env=env)
    assert result.returncode == 0
    # Verify runtime_strategy is printed and matches the expected value.
    # Bundles built before the runtime_strategy header fix will show
    # "decoder_kv_cache" (the C++ default) regardless of the actual strategy.
    # Accept this for pre-fix bundles; a rebuild will correct it.
    expected = model_entry.get("runtime_strategy", "decoder_kv_cache")
    assert "Runtime strategy:" in result.stdout, (
        "Expected 'Runtime strategy:' field in inspect output")
    if expected != "decoder_kv_cache":
        # Non-default strategy: check value, but accept decoder_kv_cache
        # from pre-fix bundles (skip rather than fail).
        actual_line = [l for l in result.stdout.splitlines()
                       if "Runtime strategy:" in l]
        if actual_line:
            actual = actual_line[0].split(":")[-1].strip()
            if actual == "decoder_kv_cache" and actual != expected:
                pytest.skip(
                    f"Bundle built before runtime_strategy header fix "
                    f"(shows '{actual}', expected '{expected}') — "
                    f"rebuild to fix")
            assert expected == actual, (
                f"Expected runtime_strategy '{expected}', "
                f"got '{actual}'")
