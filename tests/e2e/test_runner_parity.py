"""E2E: C++ vs Python token match via test_runner_parity."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[2]


@pytest.mark.e2e
def test_runner_parity(model_entry, trtf_binary, hf_python, ld_library_path):
    """Run test_runner_parity.py and verify C++ matches Python."""
    if model_entry.get("test_type") == "diffusion":
        pytest.skip("Diffusion model — no text runner parity")
    if model_entry.get("test_type") == "segmentation":
        pytest.skip("Segmentation model — no text runner parity")
    if model_entry.get("test_type") == "audio":
        pytest.skip("Audio model — no text runner parity")
    max_new = min(model_entry.get("max_new_tokens", 20), 20)

    parity_script = PROJECT_DIR / "tools" / "test_runner_parity.py"
    env_patch = {"LD_LIBRARY_PATH": ld_library_path}

    result = subprocess.run(
        [sys.executable, str(parity_script),
         "--bundle", model_entry["bundle_path"],
         "--binary", str(trtf_binary),
         "--hf-python", str(hf_python),
         "--max-new-tokens", str(max_new)],
        capture_output=True, text=True, timeout=120, env=env_patch)

    assert result.returncode == 0, (
        f"Runner parity failed for {model_entry['name']}:\n"
        f"{result.stderr}\n{result.stdout}")
