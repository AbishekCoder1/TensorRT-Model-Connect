"""Neural operator strategy runner — TRT inference for neural operator models.

Runs the C++ binary with field inputs and captures output field arrays.
Neural operators (e.g. FourCastNet, SFNO) map physical fields to fields.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time

from .. import save_full_stderr
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
        if stage.name != "full_inference":
            return StageOutput(
                stage_name=stage.name,
                metadata={"error": f"Unknown stage: {stage.name}"},
            )

        bundle_path = _resolve_bundle_path(case, ctx)
        solve_args, input_mode, input_error = _build_solve_input_args(case.inputs)
        if solve_args is None:
            return StageOutput(
                stage_name=stage.name,
                metadata={"error": input_error, "skipped": True},
            )

        cmd = [ctx.binary_path, "solve", bundle_path, *solve_args]
        output_path = case.inputs.get("output_field")

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
            truncated, log_path = save_full_stderr(
                result.stderr, ctx.artifacts_dir or "",
                "neural_operator", case.name)
            msg = (f"Neural operator inference failed (rc={result.returncode}): "
                   f"{truncated}")
            if log_path:
                msg += f" (full stderr: {log_path})"
            raise RuntimeError(msg)

        data = _parse_field_output(result.stdout.strip(), output_path)

        return StageOutput(
            stage_name=stage.name,
            data=data,
            timing_s=elapsed,
            metadata={
                "command": cmd,
                "returncode": result.returncode,
                "input_mode": input_mode,
            },
        )


def _resolve_bundle_path(case: E2ECase, ctx: RunContext) -> str:
    bundle = case.bundle or f"{case.name}.trtfb"
    if os.path.isabs(bundle):
        return bundle
    return os.path.join(ctx.engine_dir, bundle)


def _normalize_numeric_values(raw: object) -> str:
    if raw is None:
        return ""
    if isinstance(raw, (list, tuple)):
        return ",".join(str(v) for v in raw)
    if isinstance(raw, (int, float)):
        return str(raw)
    if isinstance(raw, str):
        value = raw.strip()
        if value and os.path.isfile(value):
            try:
                with open(value, encoding="utf-8") as f:
                    payload = f.read().strip()
            except OSError:
                return value
            if not payload:
                return ""
            try:
                parsed = json.loads(payload)
                if isinstance(parsed, list):
                    return _normalize_numeric_values(parsed)
            except json.JSONDecodeError:
                pass
            return payload
        return value
    return str(raw).strip()


def _build_solve_input_args(inputs: dict) -> tuple[list[str] | None, str, str]:
    branch_input = inputs.get("branch_input")
    trunk_input = inputs.get("trunk_input")
    if branch_input is not None or trunk_input is not None:
        if branch_input is None or trunk_input is None:
            return None, "branch_trunk", (
                "Neural operator requires both branch_input and trunk_input "
                "when using DeepONet mode"
            )
        branch_csv = _normalize_numeric_values(branch_input)
        trunk_csv = _normalize_numeric_values(trunk_input)
        if not branch_csv or not trunk_csv:
            return None, "branch_trunk", (
                "Neural operator branch/trunk inputs must be non-empty"
            )
        return [
            "--branch-input", branch_csv,
            "--trunk-input", trunk_csv,
        ], "branch_trunk", ""

    field_input = inputs.get("field_input")
    if field_input is None:
        field_input = inputs.get("input_field")
    if field_input is None:
        input_fields = inputs.get("input_fields")
        if isinstance(input_fields, list) and input_fields:
            field_input = input_fields[0]
    field_csv = _normalize_numeric_values(field_input)
    if not field_csv:
        return None, "field", (
            "Neural operator requires field_input (or legacy input_field/input_fields)"
        )
    return ["--field-input", field_csv], "field", ""


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

    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue

        match = re.match(r"^Output \[(\d+)\]:\s*(.*)$", line)
        if match:
            values = match.group(2).strip()
            if values:
                data["output_dim"] = int(match.group(1))
                data["output_field"] = [
                    float(x) for x in values.split() if x.strip()
                ]
            continue

        match = re.match(r"^Output shape:\s*\[(\d+),\s*(\d+),\s*(\d+)\]$", line)
        if match:
            data["output_shape"] = [
                int(match.group(1)),
                int(match.group(2)),
                int(match.group(3)),
            ]
            continue

        match = re.match(r"^First \d+ values:\s*(.*)$", line)
        if match:
            preview_str = match.group(1).replace("...", "").strip()
            if preview_str:
                data["output_field_preview"] = [
                    float(x) for x in preview_str.split() if x.strip()
                ]

    data["raw_output"] = stdout
    return data


plugin = NeuralOperatorRunner()
