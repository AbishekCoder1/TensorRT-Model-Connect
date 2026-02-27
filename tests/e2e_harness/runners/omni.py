"""Omni-multimodal and composite pipeline strategy runners.

OmniMultimodalRunner handles multi-branch models (thinker, vision, audio,
talker, code2wav) with stage-by-stage execution.

CompositePipelineRunner handles generic composite pipelines following
the stage graph from the manifest.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from typing import Any

from .. import save_full_stderr
from ..contracts import E2ECase, RunContext, StageOutput, StageSpec
from ..registry import register_runner

logger = logging.getLogger(__name__)


class OmniMultimodalRunner:
    """Execute TRT omni-multimodal inference via the C++ binary.

    Multi-branch model with separate stages: thinker text decoding,
    vision encoding, audio encoding, talker decoding, and code2wav.
    """

    @property
    def strategy_name(self) -> str:
        return "omni_multimodal"

    def run_stage(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        bundle_path = os.path.join(ctx.engine_dir, case.bundle)
        stage_name = stage.name

        # Map stage names to CLI subcommands/flags
        cmd = [ctx.binary_path, "run", bundle_path]
        cmd.extend(["--stage", stage_name])

        prompt = case.inputs.get("prompt", "")
        if prompt:
            cmd.extend(["--prompt", prompt])

        image = case.inputs.get("image")
        if image and stage_name in ("vision_encode", "end_to_end"):
            cmd.extend(["--image", image])

        audio = case.inputs.get("audio")
        if audio and stage_name in ("audio_encode", "end_to_end"):
            cmd.extend(["--audio", audio])

        max_new_tokens = case.inputs.get("max_new_tokens", 30)
        if stage_name in ("thinker_decode", "talker_decode", "end_to_end"):
            cmd.extend(["--max-new-tokens", str(max_new_tokens)])

        if ctx.hf_python:
            cmd.extend(["--hf-python", ctx.hf_python])

        env = dict(os.environ)
        if ctx.ld_library_path:
            env["LD_LIBRARY_PATH"] = ctx.ld_library_path

        logger.info("Running omni stage %s: %s", stage_name, " ".join(cmd))
        t0 = time.monotonic()
        result = subprocess.run(
            cmd, capture_output=True, text=True, env=env, timeout=600,
        )
        elapsed = time.monotonic() - t0

        if result.returncode != 0:
            truncated, log_path = save_full_stderr(
                result.stderr, ctx.artifacts_dir or "",
                f"omni_{stage_name}", case.name)
            msg = (f"Omni stage {stage_name} failed (rc={result.returncode}): "
                   f"{truncated}")
            if log_path:
                msg += f" (full stderr: {log_path})"
            raise RuntimeError(msg)

        data = _parse_stage_output(result.stdout.strip(), stage_name)

        return StageOutput(
            stage_name=stage_name,
            data=data,
            text=data.get("text"),
            timing_s=elapsed,
            metadata={
                "command": cmd,
                "returncode": result.returncode,
            },
        )


class CompositePipelineRunner:
    """Execute a generic composite pipeline following stage graph from manifest.

    Each stage is run sequentially via the C++ binary with ``--stage`` flag.
    Intermediate outputs from prior stages are passed forward.
    """

    @property
    def strategy_name(self) -> str:
        return "composite_pipeline"

    def run_stage(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        bundle_path = os.path.join(ctx.engine_dir, case.bundle)
        stage_name = stage.name

        cmd = [ctx.binary_path, "run", bundle_path]
        cmd.extend(["--stage", stage_name])

        prompt = case.inputs.get("prompt", "")
        if prompt:
            cmd.extend(["--prompt", prompt])

        image = case.inputs.get("image")
        if image:
            cmd.extend(["--image", image])

        max_new_tokens = case.inputs.get("max_new_tokens", 30)
        cmd.extend(["--max-new-tokens", str(max_new_tokens)])

        if ctx.hf_python:
            cmd.extend(["--hf-python", ctx.hf_python])

        env = dict(os.environ)
        if ctx.ld_library_path:
            env["LD_LIBRARY_PATH"] = ctx.ld_library_path

        logger.info("Running composite stage %s: %s", stage_name, " ".join(cmd))
        t0 = time.monotonic()
        result = subprocess.run(
            cmd, capture_output=True, text=True, env=env, timeout=600,
        )
        elapsed = time.monotonic() - t0

        if result.returncode != 0:
            truncated, log_path = save_full_stderr(
                result.stderr, ctx.artifacts_dir or "",
                f"composite_{stage_name}", case.name)
            msg = (f"Composite stage {stage_name} failed (rc={result.returncode}): "
                   f"{truncated}")
            if log_path:
                msg += f" (full stderr: {log_path})"
            raise RuntimeError(msg)

        data = _parse_stage_output(result.stdout.strip(), stage_name)

        return StageOutput(
            stage_name=stage_name,
            data=data,
            text=data.get("text"),
            timing_s=elapsed,
            metadata={
                "command": cmd,
                "returncode": result.returncode,
            },
        )


def _parse_stage_output(stdout: str, stage_name: str) -> dict[str, Any]:
    """Parse stage output from C++ binary stdout."""
    # Try JSON
    try:
        data = json.loads(stdout)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, ValueError):
        pass

    # For text-producing stages, treat stdout as generated text
    if stage_name in ("thinker_decode", "talker_decode", "end_to_end"):
        return {"text": stdout}

    # For encoding stages, try to parse as embedding
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            values = [float(x) for x in line.split()]
            if values:
                return {"embedding": values}
        except ValueError:
            continue

    return {"raw_output": stdout}


# Primary plugin for auto-discovery
plugin = OmniMultimodalRunner()

# Explicitly register the composite pipeline runner
register_runner(CompositePipelineRunner())
