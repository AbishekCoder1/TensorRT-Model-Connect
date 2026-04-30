"""Reranking strategy runner — TRT inference for reranking models.

Runs the C++ binary with ``trtf rerank`` to produce a relevance score for a
single (prompt, document) pair. The binary prints ``Relevance score: <float>``
on stdout; this runner parses that single score and wraps it as a one-element
``scores`` list so the reranking comparator contract is satisfied.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import time

from .. import save_full_stderr
from ..contracts import E2ECase, RunContext, StageOutput, StageSpec

logger = logging.getLogger(__name__)

_SCORE_RE = re.compile(r"Relevance score:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")


class RerankingRunner:
    """Execute TRT reranking inference via the C++ binary."""

    @property
    def strategy_name(self) -> str:
        return "reranking"

    def run_stage(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        bundle_path = os.path.join(ctx.engine_dir, case.bundle)
        prompt = case.inputs.get("prompt", "")
        document = case.inputs.get("document", "")

        if not prompt or not document:
            raise ValueError(
                "Reranking requires both 'prompt' and 'document' in "
                f"manifest inputs (case={case.name!r})"
            )

        cmd = [
            ctx.binary_path, "rerank", bundle_path,
            "--prompt", prompt,
            "--document", document,
        ]

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

        score = _parse_score(result.stdout)

        return StageOutput(
            stage_name=stage.name,
            data={"scores": [score]},
            timing_s=elapsed,
            metadata={
                "command": cmd,
                "returncode": result.returncode,
                "stdout": result.stdout or "",
                "stderr": result.stderr or "",
            },
        )


def _parse_score(stdout: str) -> float:
    """Parse the single relevance score from ``trtf rerank`` stdout."""
    match = _SCORE_RE.search(stdout)
    if match:
        return float(match.group(1))

    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            return float(line)
        except ValueError:
            continue

    raise ValueError(f"Could not parse relevance score from output: {stdout[:500]}")


plugin = RerankingRunner()
