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

from .. import save_full_stderr
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

        # encoder-only models use 'encode' to get hidden states / CLS embedding
        cmd = [
            ctx.binary_path, "encode", bundle_path,
            "--prompt", prompt,
        ]

        if ctx.hf_python:
            cmd.extend(["--hf-python", ctx.hf_python])

        env = dict(os.environ)
        if ctx.ld_library_path:
            env["LD_LIBRARY_PATH"] = ctx.ld_library_path

        logger.info("Running encoder-only: %s", " ".join(cmd))
        t0 = time.monotonic()
        result = subprocess.run(
            cmd, capture_output=True, text=True, env=env, timeout=600,
        )
        elapsed = time.monotonic() - t0

        if result.returncode != 0:
            truncated, log_path = save_full_stderr(
                result.stderr, ctx.artifacts_dir or "",
                "encoder_only", case.name)
            msg = (f"Encoder-only inference failed (rc={result.returncode}): "
                   f"{truncated}")
            if log_path:
                msg += f" (full stderr: {log_path})"
            raise RuntimeError(msg)

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
    """Parse encoder-only output (hidden states / CLS embedding).

    trtf encode outputs:
        Hidden states shape: [512, 768]
        [CLS] embedding (first 8 dims): -0.0522 0.0800 ...
    """
    # Try JSON first
    try:
        data = json.loads(stdout)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, ValueError):
        pass

    # Parse "trtf encode" text format
    cls_embedding = []
    for line in stdout.splitlines():
        line = line.strip()
        # "[CLS] embedding (first N dims): 0.1 0.2 0.3 ..."
        if line.startswith("[CLS] embedding"):
            colon_idx = line.find(":")
            if colon_idx >= 0:
                values_str = line[colon_idx + 1:].replace("...", "").strip()
                try:
                    cls_embedding = [float(x) for x in values_str.split() if x]
                except ValueError:
                    pass

    if cls_embedding:
        return {"cls_embedding": cls_embedding}

    # Fall back: try last line as whitespace-separated floats
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

    return {"raw_output": stdout}


plugin = EncoderOnlyRunner()
