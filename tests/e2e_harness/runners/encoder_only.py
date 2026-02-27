"""Encoder-only NLP strategy runner — TRT inference for encoder-only models.

Runs the C++ binary for encoder-only forward pass (e.g. BERT) and
captures hidden states / CLS embedding.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time

from ..contracts import E2ECase, RunContext, StageOutput, StageSpec

logger = logging.getLogger(__name__)


class EncoderOnlyRunner:
    """Execute TRT encoder-only inference via the C++ binary."""

    @property
    def strategy_name(self) -> str:
        return "encoder_only_nlp"

    def run_stage(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        bundle_path = os.path.join(ctx.engine_dir, case.bundle)
        prompt = case.inputs.get("prompt", "")

        # encoder-only models use 'run' with the bundle; output is hidden states
        cmd = [
            ctx.binary_path, "run", bundle_path,
            "--prompt", prompt,
            "--max-new-tokens", "1",
        ]

        if ctx.hf_python:
            cmd.extend(["--hf-python", ctx.hf_python])

        env = dict(os.environ)
        if ctx.ld_library_path:
            env["LD_LIBRARY_PATH"] = ctx.ld_library_path

        logger.info("Running encoder-only: %s", " ".join(cmd))
        t0 = time.monotonic()
        result = subprocess.run(
            cmd, capture_output=True, text=True, env=env, timeout=300,
        )
        elapsed = time.monotonic() - t0

        if result.returncode != 0:
            raise RuntimeError(
                f"Encoder-only inference failed (rc={result.returncode}): "
                f"{result.stderr[-2000:]}"
            )

        data = _parse_encoder_output(result.stdout.strip())

        return StageOutput(
            stage_name=stage.name,
            data=data,
            timing_s=elapsed,
            metadata={
                "command": cmd,
                "returncode": result.returncode,
            },
        )


def _parse_encoder_output(stdout: str) -> dict:
    """Parse encoder-only output (hidden states / CLS embedding)."""
    # Try JSON
    try:
        data = json.loads(stdout)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, ValueError):
        pass

    # Try to extract CLS embedding from last non-empty line
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            values = [float(x) for x in line.split()]
            if values:
                return {"cls_embedding": values}
        except ValueError:
            continue

    # Fall back: store raw output for downstream comparator to handle
    return {"raw_output": stdout}


plugin = EncoderOnlyRunner()
