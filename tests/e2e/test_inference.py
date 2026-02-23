"""E2E: C++ binary produces non-empty inference output."""

from __future__ import annotations

import subprocess
import pytest


@pytest.mark.e2e
def test_inference_produces_text(model_entry, trtf_binary, hf_python, ld_library_path):
    """trtf run <bundle> should generate non-empty text."""
    if model_entry.get("test_type") == "diffusion":
        pytest.skip("Diffusion model — no text inference")
    if model_entry.get("test_type") == "segmentation":
        pytest.skip("Segmentation model — use test_segmentation_pipeline")
    if model_entry.get("test_type") == "audio":
        pytest.skip("Audio model — use test_audio_pipeline")
    prompt = model_entry.get("prompt", "Hello")
    max_new = model_entry.get("max_new_tokens", 10)

    env = {"LD_LIBRARY_PATH": ld_library_path}
    result = subprocess.run(
        [str(trtf_binary), "run", model_entry["bundle_path"],
         "--prompt", prompt,
         "--max-new-tokens", str(max_new),
         "--hf-python", str(hf_python)],
        capture_output=True, text=True, timeout=120, env=env)

    assert result.returncode == 0, f"Inference failed: {result.stderr}"
    output = result.stdout.strip()
    assert len(output) > 0, "Inference produced no output"


@pytest.mark.e2e
def test_inference_deterministic(model_entry, trtf_binary, hf_python, ld_library_path):
    """Two runs with the same prompt should produce identical output."""
    if model_entry.get("test_type") == "diffusion":
        pytest.skip("Diffusion model — no text inference")
    if model_entry.get("test_type") == "segmentation":
        pytest.skip("Segmentation model — use test_segmentation_pipeline")
    if model_entry.get("test_type") == "audio":
        pytest.skip("Audio model — use test_audio_pipeline")
    prompt = model_entry.get("prompt", "Hello")
    max_new = min(model_entry.get("max_new_tokens", 10), 5)

    env = {"LD_LIBRARY_PATH": ld_library_path}
    cmd = [str(trtf_binary), "run", model_entry["bundle_path"],
           "--prompt", prompt,
           "--max-new-tokens", str(max_new),
           "--hf-python", str(hf_python)]

    r1 = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env)
    r2 = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env)

    assert r1.returncode == 0 and r2.returncode == 0
    assert r1.stdout.strip() == r2.stdout.strip(), "Non-deterministic output"
