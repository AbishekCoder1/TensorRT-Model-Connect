"""Diffusion media generation strategy runner.

Executes TRT diffusion inference for Wan-style text-to-video pipelines.
Supports per-stage execution (t5_encode, dit_step, vae_decode, end_to_end)
and full pipeline via the C++ binary.

Crossover stages allow mixing TRT and HF components to isolate regressions:
- crossover_ref_t5_trt_dit: HF T5 text encoding + TRT DiT denoising
- crossover_trt_t5_ref_dit: TRT T5 text encoding + HF DiT denoising

All GPU work runs in subprocesses for memory isolation.
"""

from __future__ import annotations

import json
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
TOOLS_DIR = PROJECT_DIR / "tools"


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
    """Build LD_LIBRARY_PATH from context or auto-detect."""
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
    """Resolve the full path to the .trtfb bundle."""
    bundle_name = case.bundle or case.inputs.get("bundle", "")
    if not bundle_name:
        bundle_name = f"{case.name}.trtfb"
    if os.path.isabs(bundle_name):
        return bundle_name
    return os.path.join(ctx.engine_dir, bundle_name)


class DiffusionMediaRunner:
    """TRT strategy runner for diffusion media generation pipelines."""

    @property
    def strategy_name(self) -> str:
        return "diffusion_media_generation"

    def run_stage(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """Execute one diffusion stage and return its output."""
        dispatch = {
            "t5_encode": self._run_t5_encode,
            "dit_step": self._run_dit_step,
            "vae_decode": self._run_vae_decode,
            "end_to_end": self._run_end_to_end,
            "end_to_end_video": self._run_end_to_end,
            "debug_pipeline": self._run_debug_pipeline,
            "generate": self._run_end_to_end,
            "frame_quality": self._run_frame_quality,
            "crossover_ref_t5_trt_dit": self._run_crossover_ref_t5_trt_dit,
            "crossover_trt_t5_ref_dit": self._run_crossover_trt_t5_ref_dit,
        }
        handler = dispatch.get(stage.name)
        if handler is None:
            return StageOutput(
                stage_name=stage.name,
                data={"error": f"Unknown diffusion stage: {stage.name}"},
                metadata={"status": "unsupported_stage"},
            )
        return handler(case, stage, ctx)

    def _run_debug_pipeline(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """Run debug_diffusion_pipeline.py for full 9-step TRT-vs-HF comparison."""
        bundle_path = _resolve_bundle_path(case, ctx)
        script = TOOLS_DIR / "debug_diffusion_pipeline.py"
        model_id = case.hf_id
        num_steps = case.inputs.get("num_inference_steps", 10)

        cmd = [
            sys.executable, str(script),
            "--bundle", bundle_path,
            "--model-id", model_id,
            "--num-steps", str(num_steps),
        ]

        t0 = time.monotonic()
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=3600)
        elapsed = time.monotonic() - t0

        stderr_truncated, stderr_log = save_full_stderr(
            result.stderr or "", ctx.artifacts_dir or "",
            "debug_pipeline", case.name)
        dbg_data: dict = {
            "passed": result.returncode == 0,
            "returncode": result.returncode,
            "output": result.stdout,
            "stderr": stderr_truncated,
        }
        if stderr_log:
            dbg_data["stderr_log"] = stderr_log

        return StageOutput(
            stage_name=stage.name,
            data=dbg_data,
            text=result.stdout,
            timing_s=elapsed,
            metadata={"command": cmd},
        )

    def _run_t5_encode(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """Run T5 text encoding stage via debug_diffusion_pipeline subprocess."""
        bundle_path = _resolve_bundle_path(case, ctx)

        # Run as a subprocess that loads the TRT T5 engine and encodes text
        prompt_text = case.inputs.get("test_prompt", "A cat sitting on a beach")
        script_code = textwrap.dedent(f"""\
            import sys
            sys.path.insert(0, {str(TOOLS_DIR)!r})
            import numpy as np
            from diffusion_helpers import load_bundle_config
            from trtf_build.diffusion_runner import DiffusionRunner

            runner = DiffusionRunner({bundle_path!r})
            prompt_text = {prompt_text!r}

            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained({case.hf_id!r})
            tokens = tokenizer(prompt_text, return_tensors="np", padding="max_length",
                               max_length=512, truncation=True)
            input_ids = tokens["input_ids"].astype(np.int32)
            text_output = runner.encode_text(input_ids)
            import os as _os
            _arts = {str(Path(_case_artifact_dir(ctx.artifacts_dir, case.name)) if ctx.artifacts_dir else Path('/tmp/claude'))!r}
            _os.makedirs(_arts, exist_ok=True)
            _npy_path = _os.path.join(_arts, "trt_t5_output.npy")
            np.save(_npy_path, text_output)
            print("output_path=" + _npy_path)
            print("shape=" + str(list(text_output.shape)))
            print("mean=" + format(float(text_output.mean()), ".6f"))
            print("std=" + format(float(text_output.std()), ".6f"))
        """)
        python = ctx.hf_python or sys.executable
        t0 = time.monotonic()
        result = subprocess.run(
            [python, "-c", script_code],
            capture_output=True, text=True, timeout=600,
            env={**os.environ, "LD_LIBRARY_PATH": _build_ld_library_path(ctx)},
        )
        elapsed = time.monotonic() - t0

        stderr_truncated, stderr_log = save_full_stderr(
            result.stderr or "", ctx.artifacts_dir or "",
            "t5_encode", case.name)
        data: dict = {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": stderr_truncated,
        }
        if stderr_log:
            data["stderr_log"] = stderr_log
        # Parse output_path from stdout (printed by the subprocess)
        for line in (result.stdout or "").splitlines():
            if line.startswith("output_path="):
                npy_path = line.split("=", 1)[1].strip()
                if os.path.exists(npy_path):
                    data["output_path"] = npy_path
                break

        return StageOutput(
            stage_name=stage.name,
            data=data,
            text=result.stdout,
            timing_s=elapsed,
            metadata={"command": "t5_encode_subprocess"},
        )

    def _run_dit_step(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """Run a single DiT forward pass via debug_diffusion_pipeline."""
        # Delegate to debug_pipeline which includes dit_step comparison
        return self._run_debug_pipeline(case, stage, ctx)

    def _run_vae_decode(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """Run VAE decoder stage."""
        # VAE decoding is part of the full pipeline; run end_to_end
        return self._run_end_to_end(case, stage, ctx)

    def _run_end_to_end(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """Run full generation via C++ binary (trtf generate-video)."""
        bundle_path = _resolve_bundle_path(case, ctx)
        binary = ctx.binary_path
        prompt = case.inputs.get("test_prompt", "A cat sitting on a beach")
        num_steps = case.inputs.get("num_inference_steps", 30)
        ld_path = _build_ld_library_path(ctx)

        with tempfile.TemporaryDirectory(prefix="trtf_frames_") as frame_dir:
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

            # Count frames
            frame_files = sorted(Path(frame_dir).glob("frame_*.png"))
            num_frames = len(frame_files)

            # Compute frame statistics if frames exist
            frame_stats = {}
            if num_frames > 0:
                frame_stats = self._compute_frame_stats(frame_dir)

            # Copy frame paths for artifact persistence
            frame_paths = [str(f) for f in frame_files]

            # Persist frames to artifacts_dir before tempdir cleanup
            artifact_frames_dir = None
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
                "end_to_end", case.name)
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

    def _run_frame_quality(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """Generate frames and compute pixel statistics for quality checks."""
        return self._run_end_to_end(case, stage, ctx)

    def _run_crossover_ref_t5_trt_dit(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """Crossover: HF T5 text encoding -> TRT DiT denoising.

        Isolates DiT engine quality by feeding HF-produced text embeddings
        into the TRT denoiser. If this passes but end_to_end fails, the
        regression is in the T5 engine.
        """
        bundle_path = _resolve_bundle_path(case, ctx)
        model_id = case.hf_id
        prompt = case.inputs.get("test_prompt", "A cat sitting on a beach")
        num_steps = case.inputs.get("num_inference_steps", 10)
        python = ctx.hf_python or sys.executable

        script_code = f"""
import sys, json, time
sys.path.insert(0, {str(TOOLS_DIR)!r})
import numpy as np
import torch

# Step 1: HF T5 encoding
from diffusers import WanPipeline
pipe = WanPipeline.from_pretrained(
    {model_id!r}, torch_dtype=torch.float32, low_cpu_mem_usage=True)
tokens = pipe.tokenizer(
    {prompt!r}, return_tensors="pt", padding="max_length",
    max_length=512, truncation=True)
with torch.no_grad():
    hf_t5_out = pipe.text_encoder(tokens.input_ids)[0].numpy()

# Step 2: TRT DiT with HF text embeddings
from trtf_build.diffusion_runner import DiffusionRunner
from diffusion_helpers import load_bundle_config, project_text as project_text_np
from diffusion_helpers import compute_timestep_embedding as compute_timestep_embedding_np
from diffusion_helpers import load_pp_weights

runner = DiffusionRunner({bundle_path!r})
pp = load_pp_weights({bundle_path!r})
cfg = load_bundle_config({bundle_path!r})

text_proj = project_text_np(hf_t5_out, pp)
z_dim = cfg.get("z_dim", 16)
pt, ph, pw = cfg.get("patch_size", [1, 2, 2])
vh, vw, vf = cfg.get("video_height", 480), cfg.get("video_width", 832), cfg.get("video_num_frames", 17)
sft, sfs = cfg.get("scale_factor_temporal", 4), cfg.get("scale_factor_spatial", 8)
t_lat = (vf - 1) // sft + 1
h_lat, w_lat = vh // sfs, vw // sfs
nt, nh, nw = t_lat // pt, h_lat // ph, w_lat // pw

rng = np.random.default_rng(42)
latents = rng.standard_normal((1, z_dim, t_lat, h_lat, w_lat)).astype(np.float32)
rope_cos, rope_sin = runner._compute_3d_rope(nt, nh, nw, 128)
temb_6d, time_embed = compute_timestep_embedding_np(999.0, pp)
patches = runner._patchify(latents, [pt, ph, pw])
hidden = patches @ pp["patch_embedding.weight"] + pp["patch_embedding.bias"]

dit_out = runner._run_engine("denoiser", {{
    "hidden_states": hidden,
    "timestep_embedding": temb_6d.reshape(1, -1),
    "time_embed": time_embed.reshape(1, -1),
    "encoder_hidden_states": text_proj,
    "rotary_cos": rope_cos, "rotary_sin": rope_sin,
}})["output"]

result = {{
    "dit_output_shape": list(dit_out.shape),
    "dit_output_mean": float(dit_out.mean()),
    "dit_output_std": float(dit_out.std()),
    "dit_output_range": [float(dit_out.min()), float(dit_out.max())],
}}
print(json.dumps(result))
np.save("/tmp/crossover_ref_t5_trt_dit.npy", dit_out)
"""
        t0 = time.monotonic()
        result = subprocess.run(
            [python, "-c", script_code],
            capture_output=True, text=True, timeout=3600,
            env={**os.environ, "LD_LIBRARY_PATH": _build_ld_library_path(ctx)},
        )
        elapsed = time.monotonic() - t0

        stderr_truncated, stderr_log = save_full_stderr(
            result.stderr or "", ctx.artifacts_dir or "",
            "crossover_ref_t5_trt_dit", case.name)
        data: dict = {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": stderr_truncated,
            "crossover_type": "ref_t5_trt_dit",
        }
        if stderr_log:
            data["stderr_log"] = stderr_log
        try:
            import json as json_mod
            parsed = json_mod.loads(result.stdout.strip())
            data.update(parsed)
        except Exception:
            pass

        npy_path = "/tmp/crossover_ref_t5_trt_dit.npy"
        if os.path.exists(npy_path):
            data["output_path"] = npy_path

        return StageOutput(
            stage_name=stage.name,
            data=data,
            timing_s=elapsed,
            metadata={"command": "crossover_ref_t5_trt_dit"},
        )

    def _run_crossover_trt_t5_ref_dit(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """Crossover: TRT T5 text encoding -> HF DiT denoising.

        Isolates T5 engine quality by feeding TRT-produced text embeddings
        into the HF denoiser. If this passes but end_to_end fails, the
        regression is in the DiT engine.
        """
        bundle_path = _resolve_bundle_path(case, ctx)
        model_id = case.hf_id
        prompt = case.inputs.get("test_prompt", "A cat sitting on a beach")
        python = ctx.hf_python or sys.executable

        script_code = f"""
import sys, json
sys.path.insert(0, {str(TOOLS_DIR)!r})
import numpy as np
import torch

# Step 1: TRT T5 encoding
from trtf_build.diffusion_runner import DiffusionRunner
from diffusion_helpers import load_bundle_config

runner = DiffusionRunner({bundle_path!r})
cfg = load_bundle_config({bundle_path!r})

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained({model_id!r})
tokens = tokenizer(
    {prompt!r}, return_tensors="np", padding="max_length",
    max_length=512, truncation=True)
input_ids = tokens["input_ids"].astype(np.int32)
trt_t5_out = runner.encode_text(input_ids)

# Step 2: HF DiT with TRT text embeddings
from diffusers import WanPipeline
pipe = WanPipeline.from_pretrained(
    {model_id!r}, torch_dtype=torch.float32, low_cpu_mem_usage=True)

z_dim = cfg.get("z_dim", 16)
vh, vw, vf = cfg.get("video_height", 480), cfg.get("video_width", 832), cfg.get("video_num_frames", 17)
sft, sfs = cfg.get("scale_factor_temporal", 4), cfg.get("scale_factor_spatial", 8)
t_lat = (vf - 1) // sft + 1
h_lat, w_lat = vh // sfs, vw // sfs

torch.manual_seed(42)
test_latent = torch.randn(1, z_dim, t_lat, h_lat, w_lat)
text_torch = torch.from_numpy(trt_t5_out.copy())
timestep = torch.tensor([999.0])

with torch.no_grad():
    hf_out = pipe.transformer(
        hidden_states=test_latent,
        timestep=timestep,
        encoder_hidden_states=text_torch,
    ).sample.numpy()

result = {{
    "dit_output_shape": list(hf_out.shape),
    "dit_output_mean": float(hf_out.mean()),
    "dit_output_std": float(hf_out.std()),
    "dit_output_range": [float(hf_out.min()), float(hf_out.max())],
}}
print(json.dumps(result))
np.save("/tmp/crossover_trt_t5_ref_dit.npy", hf_out)
"""
        t0 = time.monotonic()
        result = subprocess.run(
            [python, "-c", script_code],
            capture_output=True, text=True, timeout=3600,
            env={**os.environ, "LD_LIBRARY_PATH": _build_ld_library_path(ctx)},
        )
        elapsed = time.monotonic() - t0

        stderr_truncated, stderr_log = save_full_stderr(
            result.stderr or "", ctx.artifacts_dir or "",
            "crossover_trt_t5_ref_dit", case.name)
        data: dict = {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": stderr_truncated,
            "crossover_type": "trt_t5_ref_dit",
        }
        if stderr_log:
            data["stderr_log"] = stderr_log
        try:
            import json as json_mod
            parsed = json_mod.loads(result.stdout.strip())
            data.update(parsed)
        except Exception:
            pass

        npy_path = "/tmp/crossover_trt_t5_ref_dit.npy"
        if os.path.exists(npy_path):
            data["output_path"] = npy_path

        return StageOutput(
            stage_name=stage.name,
            data=data,
            timing_s=elapsed,
            metadata={"command": "crossover_trt_t5_ref_dit"},
        )

    @staticmethod
    def _compute_frame_stats(frame_dir: str) -> dict:
        """Load PNG frames and return aggregate pixel statistics."""
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


plugin = DiffusionMediaRunner()
