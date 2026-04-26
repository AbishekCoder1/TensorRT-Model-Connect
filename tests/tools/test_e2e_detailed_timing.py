"""Tests for E2E detailed timing normalization."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TRTF_BUILD_ROOT = REPO_ROOT / "trtf_build"
if str(TRTF_BUILD_ROOT) not in sys.path:
    sys.path.insert(0, str(TRTF_BUILD_ROOT))

from tests.e2e_harness.orchestrator import _build_detailed_timing  # noqa: E402
from trtf_build.engine_builder import (  # noqa: E402
    _compile_time_excluding_component_weight_load,
    _untracked_compile_time,
)


def test_detailed_timing_uses_actual_phase_measurements_without_overlap():
    details = _build_detailed_timing(
        {
            "bundle_build_s": 30.0,
            "trt_generate_s": 3.0,
            "trt_compile_s": 99.0,
            "ref_generate_s": 2.0,
            "contract_generate_s": 0.3,
            "compare_generate_s": 0.2,
            "preflight_s": 0.1,
        },
        {
            "timing": {
                "total_s": 25.0,
                "phases": {
                    "weights_loading_s": 1.5,
                    "trt_compile_s": 20.0,
                    "trt_compile_main_engine_s": 20.0,
                    "bundle_write_s": 0.4,
                },
            },
        },
    )

    assert details["weights_loading_s"] == 1.5
    assert details["trt_compile_s"] == 20.0
    assert details["trt_compile_main_engine_s"] == 20.0
    assert details["bundle_write_s"] == 0.4
    assert details["inference_s"] == 3.0
    assert details["reference_s"] == 2.0
    assert details["comparison_s"] == 0.5
    assert details["preflight_s"] == 0.1
    assert "build_total_s" not in details
    assert "bundle_total_s" not in details
    assert "build_overhead_s" not in details


def test_diffusion_compile_time_excludes_component_weight_loading():
    timing = {
        "phases": {
            "weights_loading_s": 13.0,
            "weights_loading_qwen3_encoder_s": 8.0,
            "weights_loading_z_image_dit_s": 5.0,
        },
    }

    compile_s = _compile_time_excluding_component_weight_load(
        components_elapsed=100.0,
        weights_before_components=1.0,
        build_timing=timing,
    )

    assert compile_s == 88.0


def test_diffusion_compile_time_adds_only_untracked_compile_residual():
    timing = {
        "phases": {
            "trt_compile_s": 80.0,
            "trt_compile_qwen3_encoder_s": 30.0,
            "trt_compile_z_image_dit_s": 50.0,
        },
    }

    residual = _untracked_compile_time(
        measured_compile_elapsed=88.0,
        compile_before_components=0.0,
        build_timing=timing,
    )

    assert residual == 8.0


def test_component_weight_timings_are_preserved_in_detailed_timing():
    details = _build_detailed_timing(
        {},
        {
            "timing": {
                "phases": {
                    "weights_loading_s": 13.0,
                    "weights_loading_qwen3_encoder_s": 8.0,
                    "weights_loading_z_image_dit_s": 5.0,
                    "trt_compile_s": 88.0,
                    "trt_compile_qwen3_encoder_s": 30.0,
                    "trt_compile_z_image_dit_s": 50.0,
                },
            },
        },
    )

    assert details["weights_loading_s"] == 13.0
    assert details["weights_loading_qwen3_encoder_s"] == 8.0
    assert details["weights_loading_z_image_dit_s"] == 5.0
    assert details["trt_compile_s"] == 88.0
    assert details["trt_compile_qwen3_encoder_s"] == 30.0
    assert details["trt_compile_z_image_dit_s"] == 50.0
