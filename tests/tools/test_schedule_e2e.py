"""Tests for the E2E parallel scheduler."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


_SCHEDULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "schedule_e2e.py"
_SPEC = importlib.util.spec_from_file_location("schedule_e2e", _SCHEDULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
schedule_e2e = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(schedule_e2e)


def _write_manifest(manifest_dir: Path, name: str, **fields: object) -> None:
    manifest = {
        "name": name,
        "hf_id": f"org/{name}",
        "runtime_strategy": "decoder_kv_cache",
        **fields,
    }
    (manifest_dir / f"{name}.json").write_text(json.dumps(manifest))


def _test_id(name: str) -> str:
    return f"tests/test_e2e.py::test_e2e[{name}]"


def test_diffusion_family_strategies_are_large() -> None:
    assert schedule_e2e.classify_size({"runtime_strategy": "diffusion_flux"}) == "large"
    assert schedule_e2e.classify_size({"runtime_strategy": "diffusion_ltx"}) == "large"
    assert schedule_e2e.classify_size({"runtime_strategy": "diffusion_pixart_torchtrt"}) == "large"


def test_exclusive_gpu_resource_reserves_gpu(tmp_path: Path) -> None:
    _write_manifest(
        tmp_path,
        "flux-2-dev-fp8-l0",
        runtime_strategy="diffusion_flux",
        e2e_parallel_resource="exclusive_gpu",
    )
    _write_manifest(
        tmp_path,
        "flux-schnell-l0",
        runtime_strategy="diffusion_flux",
        e2e_parallel_resource="exclusive_gpu",
    )
    _write_manifest(
        tmp_path,
        "flux-2-dev-l0",
        runtime_strategy="diffusion_flux",
        e2e_parallel_resource="exclusive_gpu",
    )
    _write_manifest(tmp_path, "small-a")
    _write_manifest(tmp_path, "small-b")
    _write_manifest(tmp_path, "large-a", hf_id="org/model-9B")

    assignments = schedule_e2e.schedule(
        [
            _test_id("flux-2-dev-fp8-l0"),
            _test_id("flux-schnell-l0"),
            _test_id("flux-2-dev-l0"),
            _test_id("small-a"),
            _test_id("small-b"),
            _test_id("large-a"),
        ],
        tmp_path,
        num_gpus=4,
        workers_per_gpu=2,
    )

    assert assignments["0"] == [[_test_id("flux-2-dev-fp8-l0")]]
    assert assignments["1"] == [[_test_id("flux-schnell-l0")]]
    assert assignments["2"] == [[_test_id("flux-2-dev-l0")]]
    assert assignments["3"]
    shared_tests = [test for worker in assignments["3"] for test in worker]
    assert sorted(shared_tests) == sorted([
        _test_id("small-a"),
        _test_id("small-b"),
        _test_id("large-a"),
    ])
