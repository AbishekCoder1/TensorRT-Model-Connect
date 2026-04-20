"""Reranking strategy runner — TRT inference for reranking models.

Runs the C++ binary with ``trtf rerank`` to produce relevance scores.
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


class RerankingRunner:
    """Execute TRT reranking inference via the C++ binary."""

    @property
    def strategy_name(self) -> str:
        return "reranking"

    def run_stage(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        bundle_path = os.path.join(ctx.engine_dir, case.bundle)
        query = case.inputs.get("prompt", "")
        passages = case.inputs.get("passages", [])

        cmd = [ctx.binary_path, "rerank", bundle_path, "--query", query]
        if passages:
            cmd.extend(["--passages"] + list(passages))

        runtime_cli_python = ctx.runtime_cli_hf_python()
        if runtime_cli_python:
            cmd.extend(["--hf-python", runtime_cli_python])

        env = dict(os.environ)
        if ctx.ld_library_path:
            env["LD_LIBRARY_PATH"] = ctx.ld_library_path

        logger.info("Running reranking: %s", " ".join(cmd))
        t0 = time.monotonic()
        result = subprocess.run(
            cmd, capture_output=True, text=True, env=env, timeout=600,
        )
        elapsed = time.monotonic() - t0

        if result.returncode != 0:
            truncated, log_path = save_full_stderr(
                result.stderr, ctx.artifacts_dir or "",
                "reranking", case.name)
            msg = (f"Reranking inference failed (rc={result.returncode}): "
                   f"{truncated}")
            if log_path:
                msg += f" (full stderr: {log_path})"
            raise RuntimeError(msg)

        scores = _parse_scores(result.stdout.strip())

        return StageOutput(
            stage_name=stage.name,
            data={"scores": scores},
            timing_s=elapsed,
            metadata={
                "command": cmd,
                "returncode": result.returncode,
            },
        )


def _parse_scores(stdout: str) -> list[float]:
    """Parse relevance scores from C++ binary output."""
    # Try JSON first
    try:
        data = json.loads(stdout)
        if isinstance(data, list):
            return [float(x) for x in data]
        if isinstance(data, dict) and "scores" in data:
            return [float(x) for x in data["scores"]]
    except (json.JSONDecodeError, ValueError):
        pass

    # Try one-score-per-line or whitespace-separated
    scores: list[float] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            scores.append(float(line))
        except ValueError:
            # Try whitespace-separated on this line
            try:
                scores.extend(float(x) for x in line.split())
            except ValueError:
                continue

    if scores:
        return scores

    raise ValueError(f"Could not parse scores from output: {stdout[:500]}")


plugin = RerankingRunner()
