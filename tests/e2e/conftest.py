"""Fixtures for E2E tests — engine directory, binary, model parametrization."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

E2E_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = E2E_DIR / "engines.json"
PROJECT_DIR = E2E_DIR.parents[1]


def _load_manifest():
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def _engine_dir():
    env = os.environ.get("TRTF_ENGINE_DIR")
    if env:
        return Path(env)
    manifest = _load_manifest()
    return Path(manifest.get("engine_dir", "/mnt/storage/trt-transformers/engines"))


def _models():
    manifest = _load_manifest()
    return manifest.get("models", [])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def engine_dir():
    """Resolved engine directory. Skips entire session if missing."""
    d = _engine_dir()
    if not d.is_dir():
        pytest.skip(f"Engine directory not found: {d}")
    return d


@pytest.fixture(scope="session")
def trtf_binary():
    """Path to the C++ trtf binary."""
    env = os.environ.get("TRTF_BINARY")
    binary = Path(env) if env else PROJECT_DIR / "build" / "trtf"
    if not binary.is_file():
        pytest.skip(f"trtf binary not found: {binary}")
    return binary


@pytest.fixture(scope="session")
def hf_python():
    """Python interpreter with HuggingFace tokenizers."""
    env = os.environ.get("TRTF_HF_PYTHON")
    if env:
        return Path(env)
    venv_python = PROJECT_DIR / ".venv" / "bin" / "python"
    if venv_python.is_file():
        return venv_python
    return Path(sys.executable)


@pytest.fixture(scope="session")
def ld_library_path():
    """LD_LIBRARY_PATH with TRT libs."""
    try:
        result = subprocess.run(
            [sys.executable, "-c",
             "import importlib.util; s=importlib.util.find_spec('tensorrt_libs'); "
             "print(s.submodule_search_locations[0])"],
            capture_output=True, text=True, timeout=10)
        trt_lib_dir = result.stdout.strip()
    except Exception:
        trt_lib_dir = ""
    base = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [p for p in [trt_lib_dir, "/usr/local/cuda/lib64", base] if p]
    return ":".join(parts)


# ---------------------------------------------------------------------------
# Model parametrization
# ---------------------------------------------------------------------------

def _model_ids():
    return [m["name"] for m in _models()]


def _model_by_name(name):
    for m in _models():
        if m["name"] == name:
            return m
    return None


@pytest.fixture(params=_model_ids() or ["__no_models__"])
def model_entry(request, engine_dir):
    """Parametrized fixture yielding one model entry at a time."""
    name = request.param
    if name == "__no_models__":
        pytest.skip("No models in engines.json")
    entry = _model_by_name(name)
    bundle_path = engine_dir / entry["bundle"]
    if not bundle_path.is_file():
        pytest.skip(f"Bundle not found: {bundle_path}")
    entry["bundle_path"] = str(bundle_path)
    return entry
