"""Tests for E2E orchestrator repro command generation.

Trace: ARCH-E2E-001, UD-E2E-REPRO
Intent: Validate that E2E orchestrator generates correct reproduction commands for each task strategy
Preconditions: E2ECase and RunContext are constructed with known strategy and input parameters
Postconditions: Generated repro commands contain correct binary subcommand, flags, and input paths
"""

from __future__ import annotations

from tests.e2e_harness.contracts import E2ECase, RunContext
from tests.e2e_harness.orchestrator import _build_repro_commands


def _make_ctx(tmp_path) -> RunContext:
    return RunContext(
        case=E2ECase(
            name="case-a",
            hf_id="dummy/model",
            family="dummy",
            runtime_strategy="decoder_kv_cache",
            bundle="case-a.trtfb",
            stages=[],
        ),
        artifacts_dir=str(tmp_path),
        binary_path="./build/trtf",
        hf_python="/usr/bin/python3",
        engine_dir="/tmp/engines",
    )


def test_repro_commands_use_segment_sam_for_prompted_segmentation(tmp_path) -> None:
    case = E2ECase(
        name="sam-case",
        hf_id="facebook/sam-vit-base",
        family="sam",
        runtime_strategy="prompted_segmentation",
        task_strategy="prompted_segmentation",
        bundle="sam-vit-base.trtfb",
        inputs={
            "test_image": "data/test_img.jpeg",
            "point_x": 0.5,
            "point_y": 0.25,
        },
        stages=[],
    )
    repro = _build_repro_commands(
        case,
        _make_ctx(tmp_path),
        "/tmp/engines/sam-vit-base.trtfb",
        {},
    )

    cmd = repro["trt_inference"]
    assert " segment-sam " in f" {cmd} "
    assert "--output /tmp/trtf_masks" in cmd
    assert "--point-x 0.5" in cmd
    assert "--point-y 0.25" in cmd



def test_repro_commands_use_generate_video_for_diffusion(tmp_path) -> None:
    case = E2ECase(
        name="flux-case",
        hf_id="black-forest-labs/FLUX.2-dev",
        family="flux",
        runtime_strategy="diffusion",
        task_strategy="diffusion_media_generation",
        bundle="flux-2-dev.trtfb",
        inputs={
            "test_prompt": "A photo of a cat sitting on a windowsill at sunset",
            "num_inference_steps": 28,
        },
        stages=[],
    )
    repro = _build_repro_commands(
        case,
        _make_ctx(tmp_path),
        "/tmp/engines/flux-2-dev.trtfb",
        {},
    )

    cmd = repro["trt_inference"]
    assert " generate-video " in f" {cmd} "
    assert "--output /tmp/trtf_frames" in cmd
    assert "--num-steps 28" in cmd
