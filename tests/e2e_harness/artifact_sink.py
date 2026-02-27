"""Artifact sink — persist E2E outputs, commands, and results.

Implements the ArtifactSink protocol from contracts.py. Creates a
structured directory layout for each model case with all test artifacts.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .contracts import (
    ArtifactSink as ArtifactSinkProtocol,
    CompareResult,
    E2ECase,
    E2EResult,
    StageOutput,
)

logger = logging.getLogger(__name__)


def _collect_env_fingerprint() -> dict[str, Any]:
    """Collect environment fingerprint: GPU, CUDA, TRT, Python versions."""
    fp: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "hostname": platform.node(),
    }

    # CUDA version
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if "release" in line.lower():
                    fp["cuda_version"] = line.strip()
                    break
    except Exception:
        fp["cuda_version"] = "unknown"

    # GPU info
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,compute_cap",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            gpu_lines = result.stdout.strip().splitlines()
            if gpu_lines:
                parts = gpu_lines[0].split(", ")
                fp["gpu_name"] = parts[0] if len(parts) > 0 else "unknown"
                fp["driver_version"] = parts[1] if len(parts) > 1 else "unknown"
                fp["compute_capability"] = parts[2] if len(parts) > 2 else "unknown"
                fp["gpu_count"] = len(gpu_lines)
    except Exception:
        fp["gpu_name"] = "unknown"

    # TensorRT version
    try:
        result = subprocess.run(
            [sys.executable, "-c",
             "import tensorrt; print(tensorrt.__version__)"],
            capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            fp["tensorrt_version"] = result.stdout.strip()
    except Exception:
        fp["tensorrt_version"] = "unknown"

    # Torch version
    try:
        result = subprocess.run(
            [sys.executable, "-c",
             "import torch; print(torch.__version__)"],
            capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            fp["torch_version"] = result.stdout.strip()
    except Exception:
        fp["torch_version"] = "unknown"

    # Transformers version
    try:
        result = subprocess.run(
            [sys.executable, "-c",
             "import transformers; print(transformers.__version__)"],
            capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            fp["transformers_version"] = result.stdout.strip()
    except Exception:
        fp["transformers_version"] = "unknown"

    return fp


class FileArtifactSink:
    """File-based artifact sink that persists all E2E test outputs.

    Directory layout:
        <artifacts_dir>/<model_name>/
            result.json         — final E2E result
            case.json           — input case snapshot
            env_fingerprint.json — environment info
            commands.json       — all subprocess commands executed
            stages/
                <stage_name>/
                    trt_output.json
                    ref_output.json
                    compare.json
            logs/
                <label>.stdout
                <label>.stderr
    """

    def __init__(self, artifacts_dir: str | Path, case: E2ECase) -> None:
        self._base = Path(artifacts_dir) / case.name
        self._base.mkdir(parents=True, exist_ok=True)
        (self._base / "stages").mkdir(exist_ok=True)
        (self._base / "logs").mkdir(exist_ok=True)

        self._commands: list[dict[str, Any]] = []
        self._case = case
        self._env_fp: dict[str, Any] | None = None

        # Write case snapshot
        self._write_json("case.json", _case_to_dict(case))

    @property
    def base_dir(self) -> Path:
        """Root directory for this model's artifacts."""
        return self._base

    def ensure_env_fingerprint(self) -> dict[str, Any]:
        """Collect and cache environment fingerprint."""
        if self._env_fp is None:
            self._env_fp = _collect_env_fingerprint()
            self._write_json("env_fingerprint.json", self._env_fp)
        return self._env_fp

    def log_command(
        self,
        command: list[str],
        rc: int,
        stdout: str,
        stderr: str,
        label: str = "",
    ) -> None:
        """Log a subprocess command execution."""
        entry = {
            "command": command,
            "returncode": rc,
            "stdout_len": len(stdout),
            "stderr_len": len(stderr),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._commands.append(entry)

        # Write stdout/stderr to log files
        if label:
            safe_label = label.replace("/", "_").replace(" ", "_")
        else:
            safe_label = f"cmd_{len(self._commands)}"

        if stdout:
            log_path = self._base / "logs" / f"{safe_label}.stdout"
            log_path.write_text(stdout, encoding="utf-8")

        if stderr:
            log_path = self._base / "logs" / f"{safe_label}.stderr"
            log_path.write_text(stderr, encoding="utf-8")

        # Update commands.json
        self._write_json("commands.json", self._commands)

    def write_stage_output(
        self,
        stage_name: str,
        output: StageOutput,
        prefix: str = "trt",
    ) -> None:
        """Persist a stage output (TRT or reference)."""
        stage_dir = self._base / "stages" / stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)

        data: dict[str, Any] = {
            "stage_name": output.stage_name,
            "timing_s": output.timing_s,
            "metadata": output.metadata,
        }

        if output.text is not None:
            data["text"] = output.text

        # Serialize data dict (skip numpy/large binary)
        serializable_data: dict[str, Any] = {}
        for k, v in output.data.items():
            try:
                json.dumps(v)
                serializable_data[k] = v
            except (TypeError, ValueError):
                serializable_data[k] = f"<non-serializable: {type(v).__name__}>"

        data["data"] = serializable_data

        self._write_json(
            f"stages/{stage_name}/{prefix}_output.json",
            data,
        )

    def write_compare(
        self,
        stage_name: str,
        result: CompareResult,
    ) -> None:
        """Persist comparison result for a stage."""
        stage_dir = self._base / "stages" / stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "stage_name": result.stage_name,
            "passed": result.passed,
            "metrics": result.metrics,
            "per_metric_pass": result.per_metric_pass,
            "gate_details": result.gate_details,
            "message": result.message,
        }
        self._write_json(f"stages/{stage_name}/compare.json", data)

    def finalize(self, result: E2EResult) -> str:
        """Write final result.json and return path."""
        from .result_schema import serialize_result
        self._write_json("result.json", serialize_result(result))
        return str(self._base / "result.json")

    def _write_json(self, rel_path: str, data: Any) -> None:
        """Write a JSON file relative to base dir."""
        path = self._base / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)


def _case_to_dict(case: E2ECase) -> dict[str, Any]:
    """Serialize an E2ECase to a plain dict for JSON storage."""
    return {
        "name": case.name,
        "hf_id": case.hf_id,
        "family": case.family,
        "runtime_strategy": case.runtime_strategy,
        "task_strategy": case.task_strategy,
        "reference_backend": case.reference_backend,
        "oracle_level": case.oracle_level,
        "bundle": case.bundle,
        "inputs": case.inputs,
        "preflight": [
            {"kind": p.kind, "args": p.args, "gating": p.gating}
            for p in case.preflight
        ],
        "stages": [
            {
                "name": s.name,
                "required": s.required,
                "runner_override": s.runner_override,
                "comparator_override": s.comparator_override,
            }
            for s in case.stages
        ],
        "comparison_profile": case.comparison_profile,
        "threshold_overrides": case.threshold_overrides,
        "determinism": case.determinism,
        "metadata": case.metadata,
    }
