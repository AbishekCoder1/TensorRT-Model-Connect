"""E2E: Full pipeline — build bundle, run inference (C++ & Python), compare to HF reference.

Exercises the complete flow: trtf_build.build() -> C++ runtime -> Python TrtRunner -> HF
transformers, then saves a results JSON alongside the bundle for traceability.

All GPU-heavy work (TRT inference, HF model loading) runs in subprocesses to avoid OOM
when multiple engines share a single GPU.

Usage:
    # Full pipeline with fresh build (GPU required):
    pytest tests/e2e/test_full_pipeline.py -v \
      --engine-dir /mnt/storage/trt-transformers/engines --rebuild-engines

    # Use cached bundles (default):
    pytest tests/e2e/test_full_pipeline.py -v \
      --engine-dir /mnt/storage/trt-transformers/engines

    # With explicit binary and python paths:
    pytest tests/e2e/test_full_pipeline.py -v \
      --engine-dir /mnt/storage/trt-transformers/engines \
      --trtf-binary ./build/trtf \
      --hf-python .venv/bin/python
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

PROJECT_DIR = Path(__file__).resolve().parents[2]
TOOLS_DIR = PROJECT_DIR / "tools"

# Ensure tools/ is importable for compare_logits, run_hf, etc.
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_cpp_inference(binary, bundle_path, prompt, max_new_tokens,
                       hf_python, ld_library_path, image_path=None):
    """Run C++ trtf binary inference (subprocess — no in-process GPU usage).

    Returns:
        dict with keys: text, time_s, returncode, stderr
    """
    cmd = [
        str(binary), "run", str(bundle_path),
        "--prompt", prompt,
        "--max-new-tokens", str(max_new_tokens),
        "--hf-python", str(hf_python),
    ]
    if image_path:
        cmd.extend(["--image", str(image_path)])

    env = {"LD_LIBRARY_PATH": ld_library_path}

    t0 = time.monotonic()
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=300, env=env)
    elapsed = time.monotonic() - t0

    return {
        "text": result.stdout.strip(),
        "time_s": elapsed,
        "returncode": result.returncode,
        "stderr": result.stderr,
    }


def _run_diff_logits_subprocess(hf_id, atol, max_cache_length, max_new_tokens,
                                trust_remote_code=False):
    """Run diff_logits.py as a subprocess (HF model + TRT engine in separate process).

    Returns:
        dict with keys: passed, max_diff, returncode, output
    """
    diff_logits = TOOLS_DIR / "diff_logits.py"
    cmd = [
        sys.executable, str(diff_logits),
        "--model", hf_id,
        "--atol", str(atol),
        "--max-cache-length", str(max_cache_length),
        "--max-new-tokens", str(max_new_tokens),
        "--battery",
    ]
    if trust_remote_code:
        cmd.append("--trust-remote-code")

    t0 = time.monotonic()
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=600)
    elapsed = time.monotonic() - t0

    # Parse max_abs_logit_diff from output
    max_diff = None
    for line in result.stdout.splitlines():
        if "max_abs_logit_diff" in line:
            try:
                max_diff = float(line.split(":")[-1].strip())
            except ValueError:
                pass

    return {
        "passed": result.returncode == 0,
        "max_diff": max_diff,
        "returncode": result.returncode,
        "output": result.stdout + result.stderr,
        "time_s": elapsed,
    }


def _run_diff_vl_subprocess(bundle_path, image_path, hf_id, binary, hf_python,
                            ld_library_path, atol=0.1, max_new_tokens=20):
    """Run diff_vl.py as a subprocess (all GPU work isolated).

    Returns:
        dict with keys: passed, returncode, output, time_s
    """
    diff_vl = TOOLS_DIR / "diff_vl.py"
    cmd = [
        sys.executable, str(diff_vl),
        "--bundle", str(bundle_path),
        "--image", str(image_path),
        "--model", hf_id,
        "--atol", str(atol),
        "--max-new-tokens", str(max_new_tokens),
        "--binary", str(binary),
    ]
    if hf_python:
        cmd.extend(["--hf-python", str(hf_python)])

    env = {"LD_LIBRARY_PATH": ld_library_path}

    t0 = time.monotonic()
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=600, env=env)
    elapsed = time.monotonic() - t0

    return {
        "passed": result.returncode == 0,
        "returncode": result.returncode,
        "output": result.stdout + result.stderr,
        "time_s": elapsed,
    }


def _get_gpu_name():
    """Get GPU name via nvidia-smi, or return 'unknown'."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            return r.stdout.strip().split("\n")[0]
    except Exception:
        pass
    return "unknown"


def _get_trt_version():
    """Get TensorRT version string, or 'unknown'."""
    try:
        import tensorrt as trt
        return trt.__version__
    except Exception:
        return "unknown"


def _save_results(bundle_path, results_dict):
    """Save results JSON next to the bundle file.

    Writes <bundle_stem>.results.json in the same directory as the bundle.
    Handles numpy types for JSON serialization.
    """
    bundle_p = Path(bundle_path)
    results_path = bundle_p.with_suffix(".results.json")

    class _NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.bool_):
                return bool(obj)
            return super().default(obj)

    with open(results_path, "w") as f:
        json.dump(results_dict, f, indent=2, cls=_NumpyEncoder)

    return str(results_path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.e2e
def test_full_pipeline(built_bundle, trtf_binary, hf_python, ld_library_path):
    """Full pipeline: build -> C++ inference -> diff_logits (HF vs Python runner)."""
    entry = built_bundle["entry"]
    bundle_path = built_bundle["path"]

    # Skip VL models (handled by test_full_pipeline_vlm)
    if entry.get("runtime_strategy") == "vision_language":
        pytest.skip("VL model — use test_full_pipeline_vlm")

    prompt = entry.get("prompt", "The capital of France is")
    max_new_tokens = entry.get("max_new_tokens", 30)
    atol = entry.get("logit_atol", 1e-3)
    max_cache_length = entry.get("max_cache_length", 256)
    trust_remote_code = entry.get("trust_remote_code", False)
    hf_id = entry["hf_id"]

    # Step 1: C++ binary inference (subprocess)
    cpp_result = _run_cpp_inference(
        trtf_binary, bundle_path, prompt, max_new_tokens,
        hf_python, ld_library_path)
    assert cpp_result["returncode"] == 0, (
        f"C++ inference failed (rc={cpp_result['returncode']}):\n"
        f"{cpp_result['stderr']}")
    assert len(cpp_result["text"]) > 0, "C++ inference produced no output"

    # Step 2: diff_logits — HF vs Python TrtRunner (subprocess, memory-isolated)
    diff_result = _run_diff_logits_subprocess(
        hf_id, atol, max_cache_length, max_new_tokens,
        trust_remote_code=trust_remote_code)

    # Step 3: Build and save results
    bundle_size = os.path.getsize(bundle_path)
    results = {
        "model": {
            "name": entry["name"],
            "hf_id": hf_id,
            "family": entry.get("family", "unknown"),
            "runtime_strategy": entry.get("runtime_strategy", "decoder_kv_cache"),
        },
        "build": {
            "was_cached": built_bundle["was_cached"],
            "build_time_s": built_bundle["build_time_s"],
            "engine_size_bytes": bundle_size,
            "max_cache_length": max_cache_length,
            "trt_version": _get_trt_version(),
            "gpu_name": _get_gpu_name(),
        },
        "inference": {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "generated_text_cpp": cpp_result["text"],
        },
        "comparison": {
            "logit_atol": atol,
            "logit_parity_pass": diff_result["passed"],
            "max_logit_diff": diff_result["max_diff"],
        },
        "timing": {
            "build_time_s": built_bundle["build_time_s"],
            "cpp_inference_time_s": cpp_result["time_s"],
            "diff_logits_time_s": diff_result["time_s"],
        },
        "status": "PASS" if diff_result["passed"] else "FAIL",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    results_path = _save_results(bundle_path, results)

    # Assertions
    assert diff_result["passed"], (
        f"diff_logits FAILED for {entry['name']}:\n"
        f"{diff_result['output'][-2000:]}")

    print(f"\n[full_pipeline] {entry['name']}: PASS "
          f"(max_diff={diff_result['max_diff']}, cpp_text={len(cpp_result['text'])} chars)")
    print(f"  Results saved: {results_path}")


@pytest.mark.e2e
def test_full_pipeline_vlm(built_bundle, trtf_binary, hf_python, ld_library_path,
                           engine_dir):
    """Full VL pipeline: build -> C++ inference -> diff_vl (HF vs Python runner)."""
    entry = built_bundle["entry"]
    bundle_path = built_bundle["path"]

    if entry.get("runtime_strategy") != "vision_language":
        pytest.skip(f"{entry['name']} is not a VL model")

    test_image = entry.get("test_image")
    if not test_image:
        pytest.skip(f"No test_image configured for {entry['name']}")

    # Resolve image path (relative to engine_dir)
    image_path = Path(test_image)
    if not image_path.is_absolute():
        image_path = engine_dir / test_image
    if not image_path.is_file():
        pytest.skip(f"Test image not found: {image_path}")

    prompt = entry.get("prompt", "Describe this image.")
    max_new_tokens = entry.get("max_new_tokens", 30)
    atol = entry.get("logit_atol", 1e-3)
    hf_id = entry["hf_id"]

    # Step 1: C++ binary with image (subprocess)
    cpp_result = _run_cpp_inference(
        trtf_binary, bundle_path, prompt, max_new_tokens,
        hf_python, ld_library_path, image_path=str(image_path))
    assert cpp_result["returncode"] == 0, (
        f"C++ VL inference failed (rc={cpp_result['returncode']}):\n"
        f"{cpp_result['stderr']}")
    assert len(cpp_result["text"]) > 0, "C++ VL inference produced no output"

    # Step 2: diff_vl — HF vs Python VLTrtRunner (subprocess, memory-isolated)
    diff_result = _run_diff_vl_subprocess(
        bundle_path, str(image_path), hf_id, trtf_binary, hf_python,
        ld_library_path, atol=atol, max_new_tokens=max_new_tokens)

    # Step 3: Build and save results
    bundle_size = os.path.getsize(bundle_path)
    results = {
        "model": {
            "name": entry["name"],
            "hf_id": hf_id,
            "family": entry.get("family", "unknown"),
            "runtime_strategy": "vision_language",
        },
        "build": {
            "was_cached": built_bundle["was_cached"],
            "build_time_s": built_bundle["build_time_s"],
            "engine_size_bytes": bundle_size,
            "max_cache_length": entry.get("max_cache_length", 256),
            "trt_version": _get_trt_version(),
            "gpu_name": _get_gpu_name(),
        },
        "inference": {
            "prompt": prompt,
            "test_image": str(image_path),
            "max_new_tokens": max_new_tokens,
            "generated_text_cpp": cpp_result["text"],
        },
        "comparison": {
            "diff_vl_pass": diff_result["passed"],
        },
        "timing": {
            "build_time_s": built_bundle["build_time_s"],
            "cpp_inference_time_s": cpp_result["time_s"],
            "diff_vl_time_s": diff_result["time_s"],
        },
        "status": "PASS" if diff_result["passed"] else "FAIL",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    results_path = _save_results(bundle_path, results)

    # Assertions
    assert diff_result["passed"], (
        f"diff_vl FAILED for {entry['name']}:\n"
        f"{diff_result['output'][-2000:]}")

    print(f"\n[full_pipeline_vlm] {entry['name']}: PASS")
    print(f"  C++ output: {cpp_result['text'][:200]}")
    print(f"  Results saved: {results_path}")
