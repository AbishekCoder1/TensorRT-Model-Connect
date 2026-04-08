"""Torch-TRT diffusion strategy runner.

Runs torch-trt compiled diffusion engines (T5, DiT, VAE) for comparison
against HuggingFace diffusers.  Unlike the raw TRT diffusion runner, torch-trt
engines include all preprocessing internally (patch embedding, caption
projection, timestep embedding), so per-component testing loads engines
directly from the bundle via TensorRT Python API.

Stages:
  t5_encode  — Run TRT T5 encoder, save embeddings .npy
  dit_step   — Run TRT DiT single forward, save noise prediction .npy
  vae_decode — Run TRT VAE decoder, save decoded image .npy
  end_to_end — Run C++ binary (generate-video), save frame PNGs

All GPU work runs in subprocesses for memory isolation.
Fixed seed (42) everywhere for reproducibility.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

from .. import save_full_stderr, _case_artifact_dir
from ..contracts import E2ECase, RunContext, StageOutput, StageSpec

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parents[3]


def _find_trt_lib_dir() -> str:
    """Find TRT library directory from the Python tensorrt_libs package."""
    try:
        import importlib.util
        spec = importlib.util.find_spec("tensorrt_libs")
        if spec and spec.submodule_search_locations:
            return spec.submodule_search_locations[0]
    except ImportError:
        pass
    return ""


def _build_ld_library_path(ctx: RunContext) -> str:
    if ctx.ld_library_path:
        return ctx.ld_library_path
    trt_lib = _find_trt_lib_dir()
    parts = []
    if trt_lib:
        parts.append(trt_lib)
    parts.append("/usr/local/cuda/lib64")
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    if existing:
        parts.append(existing)
    return ":".join(parts)


def _resolve_bundle_path(case: E2ECase, ctx: RunContext) -> str:
    bundle_name = case.bundle or case.inputs.get("bundle", "")
    if not bundle_name:
        bundle_name = f"{case.name}.trtfb"
    if os.path.isabs(bundle_name):
        return bundle_name
    return os.path.join(ctx.engine_dir, bundle_name)


class TorchTrtDiffusionRunner:
    """Strategy runner for torch-trt compiled diffusion pipelines."""

    @property
    def strategy_name(self) -> str:
        return "torchtrt_diffusion"

    def run_stage(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        dispatch = {
            "t5_encode": self._run_t5_encode,
            "dit_step": self._run_dit_step,
            "vae_decode": self._run_vae_decode,
            "end_to_end": self._run_end_to_end,
        }
        handler = dispatch.get(stage.name)
        if handler is None:
            return StageOutput(
                stage_name=stage.name,
                data={"error": f"Unknown torchtrt_diffusion stage: {stage.name}"},
                metadata={"status": "unsupported_stage"},
            )
        return handler(case, stage, ctx)

    # ── Per-component stages ──────────────────────────────────────────

    def _run_t5_encode(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """Run torch-trt T5 encoder and save embeddings."""
        bundle_path = _resolve_bundle_path(case, ctx)
        prompt = case.inputs.get("prompt", case.inputs.get("test_prompt",
                                 "A photo of a cat sitting on a windowsill"))
        python = ctx.hf_python or sys.executable

        artifacts_dir = ctx.artifacts_dir or tempfile.gettempdir()
        model_dir = _case_artifact_dir(artifacts_dir, case.name)
        npy_path = os.path.join(model_dir, "trt_t5_output.npy")

        script_code = textwrap.dedent(f"""\
import sys, os
sys.path.insert(0, {str(PROJECT_DIR / 'ttrt_build')!r})
import numpy as np
import torch
import tensorrt as trt

# -- Load engine from bundle --
from ttrt_build.bundle_reader import read_bundle_section
data = read_bundle_section({bundle_path!r}, "text_encoder_0_plan")
logger = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(logger)
engine = runtime.deserialize_cuda_engine(data)

# -- Tokenize with HF --
from transformers import AutoTokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(
        {case.hf_id!r}, subfolder="tokenizer")
except (ValueError, OSError):
    tokenizer = AutoTokenizer.from_pretrained({case.hf_id!r})
tokens = tokenizer({prompt!r}, return_tensors="pt",
                    padding="max_length", max_length=120, truncation=True)
input_ids = tokens["input_ids"].int()  # [1, 120] int32

# -- Run engine --
ctx = engine.create_execution_context()
stream = torch.cuda.Stream()
buffers = {{}}
output_names = []
dtype_map = {{
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.INT32: torch.int32,
}}
for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    shape = list(engine.get_tensor_shape(name))
    dt = dtype_map.get(engine.get_tensor_dtype(name), torch.float32)
    mode = engine.get_tensor_mode(name)
    if mode == trt.TensorIOMode.INPUT:
        buf = input_ids.cuda().contiguous() if name == "input_ids" else \\
              torch.zeros(shape, dtype=dt, device="cuda")
    else:
        buf = torch.empty(shape, dtype=dt, device="cuda")
        output_names.append(name)
    buffers[name] = buf
    ctx.set_tensor_address(name, buf.data_ptr())
ctx.execute_async_v3(stream.cuda_stream)
stream.synchronize()

trt_out = buffers[output_names[0]].cpu().float().numpy()
np.save({npy_path!r}, trt_out)
print("output_path=" + {npy_path!r})
print("shape=" + str(list(trt_out.shape)))
print("mean=" + format(float(trt_out.mean()), ".6f"))
print("std=" + format(float(trt_out.std()), ".6f"))
""")
        t0 = time.monotonic()
        result = subprocess.run(
            [python, "-c", script_code],
            capture_output=True, text=True, timeout=600,
            env={**os.environ, "LD_LIBRARY_PATH": _build_ld_library_path(ctx)},
        )
        elapsed = time.monotonic() - t0

        stderr_truncated, stderr_log = save_full_stderr(
            result.stderr or "", ctx.artifacts_dir or "",
            "torchtrt_t5_encode", case.name)
        data: dict = {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": stderr_truncated,
        }
        if stderr_log:
            data["stderr_log"] = stderr_log
        if os.path.exists(npy_path):
            data["output_path"] = npy_path

        return StageOutput(
            stage_name=stage.name,
            data=data,
            text=result.stdout,
            timing_s=elapsed,
            metadata={"command": "torchtrt_t5_encode"},
        )

    def _run_dit_step(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """Run torch-trt DiT single forward pass with deterministic inputs."""
        bundle_path = _resolve_bundle_path(case, ctx)
        python = ctx.hf_python or sys.executable

        artifacts_dir = ctx.artifacts_dir or tempfile.gettempdir()
        model_dir = _case_artifact_dir(artifacts_dir, case.name)
        npy_path = os.path.join(model_dir, "trt_dit_output.npy")

        h_lat = case.inputs.get("image_height", 1024) // 8
        w_lat = case.inputs.get("image_width", 1024) // 8

        script_code = textwrap.dedent(f"""\
import sys
sys.path.insert(0, {str(PROJECT_DIR / 'ttrt_build')!r})
import numpy as np
import torch
import tensorrt as trt

from ttrt_build.bundle_reader import read_bundle_section
data = read_bundle_section({bundle_path!r}, "denoiser_plan")
logger = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(logger)
engine = runtime.deserialize_cuda_engine(data)

# Deterministic inputs (seed=42)
torch.manual_seed(42)
h_lat, w_lat = {h_lat}, {w_lat}
sample = torch.randn(1, 4, h_lat, w_lat, dtype=torch.float16)
text = torch.randn(1, 120, 4096, dtype=torch.float16)
timestep = torch.tensor([500.0], dtype=torch.float16)

# Run engine
ctx = engine.create_execution_context()
stream = torch.cuda.Stream()
buffers = {{}}
output_names = []
dtype_map = {{
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.INT32: torch.int32,
}}
input_map = {{
    "sample": sample, "encoder_hidden_states": text, "timestep": timestep,
}}
for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    shape = list(engine.get_tensor_shape(name))
    dt = dtype_map.get(engine.get_tensor_dtype(name), torch.float32)
    mode = engine.get_tensor_mode(name)
    if mode == trt.TensorIOMode.INPUT:
        tensor = input_map.get(name, torch.zeros(shape, dtype=dt))
        buf = tensor.to(dt).cuda().contiguous()
    else:
        buf = torch.empty(shape, dtype=dt, device="cuda")
        output_names.append(name)
    buffers[name] = buf
    ctx.set_tensor_address(name, buf.data_ptr())
ctx.execute_async_v3(stream.cuda_stream)
stream.synchronize()

trt_out = buffers[output_names[0]].cpu().float().numpy()
np.save({npy_path!r}, trt_out)
print("output_path=" + {npy_path!r})
print("shape=" + str(list(trt_out.shape)))
print("mean=" + format(float(trt_out.mean()), ".6f"))
print("std=" + format(float(trt_out.std()), ".6f"))
""")
        t0 = time.monotonic()
        result = subprocess.run(
            [python, "-c", script_code],
            capture_output=True, text=True, timeout=600,
            env={**os.environ, "LD_LIBRARY_PATH": _build_ld_library_path(ctx)},
        )
        elapsed = time.monotonic() - t0

        stderr_truncated, stderr_log = save_full_stderr(
            result.stderr or "", ctx.artifacts_dir or "",
            "torchtrt_dit_step", case.name)
        data: dict = {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": stderr_truncated,
        }
        if stderr_log:
            data["stderr_log"] = stderr_log
        if os.path.exists(npy_path):
            data["output_path"] = npy_path

        return StageOutput(
            stage_name=stage.name,
            data=data,
            text=result.stdout,
            timing_s=elapsed,
            metadata={"command": "torchtrt_dit_step"},
        )

    def _run_vae_decode(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """Run torch-trt VAE decoder with deterministic latent input."""
        bundle_path = _resolve_bundle_path(case, ctx)
        python = ctx.hf_python or sys.executable

        artifacts_dir = ctx.artifacts_dir or tempfile.gettempdir()
        model_dir = _case_artifact_dir(artifacts_dir, case.name)
        npy_path = os.path.join(model_dir, "trt_vae_output.npy")

        h_lat = case.inputs.get("image_height", 1024) // 8
        w_lat = case.inputs.get("image_width", 1024) // 8

        script_code = textwrap.dedent(f"""\
import sys
sys.path.insert(0, {str(PROJECT_DIR / 'ttrt_build')!r})
import numpy as np
import torch
import tensorrt as trt

from ttrt_build.bundle_reader import read_bundle_section
data = read_bundle_section({bundle_path!r}, "vae_decoder_plan")
logger = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(logger)
engine = runtime.deserialize_cuda_engine(data)

# Deterministic latent (seed=42)
torch.manual_seed(42)
h_lat, w_lat = {h_lat}, {w_lat}
latent = torch.randn(1, 4, h_lat, w_lat, dtype=torch.float16)

# Run engine
ctx = engine.create_execution_context()
stream = torch.cuda.Stream()
buffers = {{}}
output_names = []
dtype_map = {{
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.INT32: torch.int32,
}}
for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    shape = list(engine.get_tensor_shape(name))
    dt = dtype_map.get(engine.get_tensor_dtype(name), torch.float32)
    mode = engine.get_tensor_mode(name)
    if mode == trt.TensorIOMode.INPUT:
        buf = latent.to(dt).cuda().contiguous() if name == "latent" else \\
              torch.zeros(shape, dtype=dt, device="cuda")
    else:
        buf = torch.empty(shape, dtype=dt, device="cuda")
        output_names.append(name)
    buffers[name] = buf
    ctx.set_tensor_address(name, buf.data_ptr())
ctx.execute_async_v3(stream.cuda_stream)
stream.synchronize()

trt_out = buffers[output_names[0]].cpu().float().numpy()
np.save({npy_path!r}, trt_out)
print("output_path=" + {npy_path!r})
print("shape=" + str(list(trt_out.shape)))
print("mean=" + format(float(trt_out.mean()), ".6f"))
print("std=" + format(float(trt_out.std()), ".6f"))
""")
        t0 = time.monotonic()
        result = subprocess.run(
            [python, "-c", script_code],
            capture_output=True, text=True, timeout=600,
            env={**os.environ, "LD_LIBRARY_PATH": _build_ld_library_path(ctx)},
        )
        elapsed = time.monotonic() - t0

        stderr_truncated, stderr_log = save_full_stderr(
            result.stderr or "", ctx.artifacts_dir or "",
            "torchtrt_vae_decode", case.name)
        data: dict = {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": stderr_truncated,
        }
        if stderr_log:
            data["stderr_log"] = stderr_log
        if os.path.exists(npy_path):
            data["output_path"] = npy_path

        return StageOutput(
            stage_name=stage.name,
            data=data,
            text=result.stdout,
            timing_s=elapsed,
            metadata={"command": "torchtrt_vae_decode"},
        )

    # ── End-to-end via C++ binary ─────────────────────────────────────

    def _run_end_to_end(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """Run full generation via C++ binary (trtf generate-video)."""
        bundle_path = _resolve_bundle_path(case, ctx)
        binary = ctx.binary_path
        prompt = case.inputs.get("prompt", case.inputs.get("test_prompt",
                                 "A photo of a cat sitting on a windowsill at sunset"))
        num_steps = case.inputs.get("num_inference_steps", 20)
        ld_path = _build_ld_library_path(ctx)

        with tempfile.TemporaryDirectory(prefix="torchtrt_frames_") as frame_dir:
            # C++ pipeline defaults to seed=42 when not specified
            cmd = [
                binary, "generate-video", bundle_path,
                "--prompt", prompt,
                "--output", frame_dir,
                "--num-steps", str(num_steps),
            ]
            if ctx.hf_python:
                cmd.extend(["--hf-python", ctx.hf_python])

            env = {**os.environ, "LD_LIBRARY_PATH": ld_path}

            t0 = time.monotonic()
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=3600, env=env)
            elapsed = time.monotonic() - t0

            frame_files = sorted(Path(frame_dir).glob("frame_*.png"))
            num_frames = len(frame_files)

            frame_stats = {}
            if num_frames > 0:
                frame_stats = self._compute_frame_stats(frame_dir)

            # Persist frames to artifacts_dir
            artifact_frames_dir = None
            frame_paths = [str(f) for f in frame_files]
            if ctx.artifacts_dir and num_frames > 0:
                artifact_frames_dir = os.path.join(
                    _case_artifact_dir(ctx.artifacts_dir, case.name), "frames")
                os.makedirs(artifact_frames_dir, exist_ok=True)
                for fp in frame_files:
                    shutil.copy2(str(fp), artifact_frames_dir)
                frame_paths = [
                    os.path.join(artifact_frames_dir, fp.name)
                    for fp in frame_files
                ]

            stderr_truncated, stderr_log = save_full_stderr(
                result.stderr or "", ctx.artifacts_dir or "",
                "torchtrt_end_to_end", case.name)
            e2e_data: dict = {
                "returncode": result.returncode,
                "num_frames": num_frames,
                "frame_stats": frame_stats,
                "frames_dir": artifact_frames_dir or frame_dir,
                "frame_paths": frame_paths,
                "stdout": result.stdout,
                "stderr": stderr_truncated,
            }
            if stderr_log:
                e2e_data["stderr_log"] = stderr_log

            return StageOutput(
                stage_name=stage.name,
                data=e2e_data,
                text=result.stdout,
                timing_s=elapsed,
                metadata={"command": cmd},
            )

    @staticmethod
    def _compute_frame_stats(frame_dir: str) -> dict:
        try:
            import numpy as np
            from PIL import Image
        except ImportError:
            return {"error": "PIL or numpy not available"}

        frames = sorted(Path(frame_dir).glob("frame_*.png"))
        if not frames:
            return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

        all_pixels = []
        for fp in frames:
            img = Image.open(fp).convert("RGB")
            arr = np.array(img, dtype=np.float32) / 255.0
            all_pixels.append(arr.flatten())

        combined = np.concatenate(all_pixels)
        return {
            "count": len(frames),
            "mean": float(np.mean(combined)),
            "std": float(np.std(combined)),
            "min": float(np.min(combined)),
            "max": float(np.max(combined)),
        }


plugin = TorchTrtDiffusionRunner()
