"""Embedding strategy runner — TRT inference for embedding models.

Runs the C++ binary with ``trtf embed`` to produce L2-normalized vectors.
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


class EmbeddingRunner:
    """Execute TRT embedding inference via the C++ binary."""

    @property
    def strategy_name(self) -> str:
        return "embedding"

    def run_stage(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        bundle_path = os.path.join(ctx.engine_dir, case.bundle)
        prompt = case.inputs.get("prompt", "")

        cmd = [ctx.binary_path, "embed", bundle_path, "--prompt", prompt]

        if ctx.hf_python:
            cmd.extend(["--hf-python", ctx.hf_python])

        env = dict(os.environ)
        if ctx.ld_library_path:
            env["LD_LIBRARY_PATH"] = ctx.ld_library_path

        logger.info("Running embedding: %s", " ".join(cmd))
        t0 = time.monotonic()
        result = subprocess.run(
            cmd, capture_output=True, text=True, env=env, timeout=600,
        )
        elapsed = time.monotonic() - t0

        if result.returncode != 0:
            truncated, log_path = save_full_stderr(
                result.stderr, ctx.artifacts_dir or "",
                "embedding", case.name)
            msg = (f"Embedding inference failed (rc={result.returncode}): "
                   f"{truncated}")
            if log_path:
                msg += f" (full stderr: {log_path})"
            raise RuntimeError(msg)

        # Parse embedding vector from stdout (JSON array or whitespace-separated floats)
        stdout = result.stdout.strip()
        embedding = _parse_embedding(stdout)

        return StageOutput(
            stage_name=stage.name,
            data={"embedding": embedding},
            timing_s=elapsed,
            metadata={
                "command": cmd,
                "returncode": result.returncode,
            },
        )


def _parse_embedding(stdout: str) -> list[float]:
    """Parse embedding vector from C++ binary output."""
    # Try JSON array first
    try:
        data = json.loads(stdout)
        if isinstance(data, list):
            return [float(x) for x in data]
        if isinstance(data, dict) and "embedding" in data:
            return [float(x) for x in data["embedding"]]
    except (json.JSONDecodeError, ValueError):
        pass

    # Try whitespace-separated floats (last line may be the embedding)
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            values = [float(x) for x in line.split()]
            if len(values) > 1:
                return values
        except ValueError:
            continue

    raise ValueError(f"Could not parse embedding from output: {stdout[:500]}")


plugin = EmbeddingRunner()
