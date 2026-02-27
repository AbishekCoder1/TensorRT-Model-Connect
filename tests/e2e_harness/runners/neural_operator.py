"""Neural operator strategy runner — TRT inference for neural operator models.

Runs the C++ binary with field inputs and captures output field arrays.
Neural operators (e.g. FourCastNet, SFNO) map physical fields to fields.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time

from ..contracts import E2ECase, RunContext, StageOutput, StageSpec

logger = logging.getLogger(__name__)


class NeuralOperatorRunner:
    """Execute TRT neural operator inference via the C++ binary."""

    @property
    def strategy_name(self) -> str:
        return "neural_operator"

    def run_stage(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        bundle_path = os.path.join(ctx.engine_dir, case.bundle)

        cmd = [ctx.binary_path, "run", bundle_path]

        # Neural operators take field inputs (e.g. --input-field <path>)
        input_field = case.inputs.get("input_field")
        if input_field:
            cmd.extend(["--input-field", input_field])

        input_fields = case.inputs.get("input_fields", [])
        for field_path in input_fields:
            cmd.extend(["--input-field", field_path])

        # Output path for field arrays
        output_path = case.inputs.get("output_field")
        if output_path:
            cmd.extend(["--output-field", output_path])

        if ctx.hf_python:
            cmd.extend(["--hf-python", ctx.hf_python])

        env = dict(os.environ)
        if ctx.ld_library_path:
            env["LD_LIBRARY_PATH"] = ctx.ld_library_path

        logger.info("Running neural operator: %s", " ".join(cmd))
        t0 = time.monotonic()
        result = subprocess.run(
            cmd, capture_output=True, text=True, env=env, timeout=600,
        )
        elapsed = time.monotonic() - t0

        if result.returncode != 0:
            raise RuntimeError(
                f"Neural operator inference failed (rc={result.returncode}): "
                f"{result.stderr[-2000:]}"
            )

        data = _parse_field_output(result.stdout.strip(), output_path)

        return StageOutput(
            stage_name=stage.name,
            data=data,
            timing_s=elapsed,
            metadata={
                "command": cmd,
                "returncode": result.returncode,
            },
        )


def _parse_field_output(stdout: str, output_path: str | None) -> dict:
    """Parse neural operator output fields."""
    data: dict = {}

    # If an output file was written, record its path
    if output_path and os.path.isfile(output_path):
        data["output_field_path"] = output_path

    # Try JSON from stdout
    try:
        parsed = json.loads(stdout)
        if isinstance(parsed, dict):
            data.update(parsed)
            return data
    except (json.JSONDecodeError, ValueError):
        pass

    data["raw_output"] = stdout
    return data


plugin = NeuralOperatorRunner()
