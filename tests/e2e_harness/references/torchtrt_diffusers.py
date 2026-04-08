"""HF Diffusers reference backend for torch-trt diffusion comparison.

Runs individual HuggingFace diffusers components (T5, DiT, VAE) as
subprocesses, producing per-component reference outputs for comparison
against torch-trt engines.

Uses the same deterministic inputs (seed=42) as the torch-trt runner
to ensure fair per-component comparison.

For end_to_end, delegates to the full PixArt diffusers pipeline.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

from .. import _case_artifact_dir
from ..contracts import E2ECase, RunContext, StageOutput, StageSpec

logger = logging.getLogger(__name__)


def _ref_subprocess_env() -> dict:
    """Build env for HF reference subprocesses (fix GB300 cuBLAS)."""
    env = os.environ.copy()
    sys_cublas = "/usr/local/cuda/lib64/libcublas.so.13"
    sys_cublaslt = "/usr/local/cuda/lib64/libcublasLt.so.13"
    if os.path.exists(sys_cublas) and os.path.exists(sys_cublaslt):
        existing = env.get("LD_PRELOAD", "")
        preload = f"{sys_cublas}:{sys_cublaslt}"
        env["LD_PRELOAD"] = f"{preload}:{existing}" if existing else preload
    return env


class TorchTrtDiffusersReference:
    """Reference backend for per-component torch-trt diffusion comparison."""

    @property
    def backend_name(self) -> str:
        return "torchtrt_diffusers"

    def run_stage(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        dispatch = {
            "t5_encode": self._run_t5_encode,
            "dit_step": self._run_dit_step,
            "vae_decode": self._run_vae_decode,
            "end_to_end": self._run_full_pipeline,
        }
        handler = dispatch.get(stage.name)
        if handler is None:
            return StageOutput(
                stage_name=stage.name,
                data={"error": f"Unknown stage: {stage.name}"},
            )
        return handler(case, stage, ctx)

    def _run_t5_encode(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """Run HF T5 encoder and save reference embeddings."""
        model_id = case.hf_id
        prompt = case.inputs.get("prompt", case.inputs.get("test_prompt",
                                 "A photo of a cat sitting on a windowsill"))
        python = ctx.hf_python or sys.executable

        artifacts_dir = ctx.artifacts_dir or tempfile.gettempdir()
        model_dir = _case_artifact_dir(artifacts_dir, case.name)
        npy_path = os.path.join(model_dir, "hf_t5_output.npy")

        script = textwrap.dedent(f"""\
import torch, numpy as np
from transformers import T5EncoderModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained({model_id!r}, subfolder="tokenizer")
tokens = tokenizer({prompt!r}, return_tensors="pt",
                    padding="max_length", max_length=120, truncation=True)

model = T5EncoderModel.from_pretrained(
    {model_id!r}, subfolder="text_encoder", torch_dtype=torch.float16)
model.eval()
with torch.no_grad():
    out = model(input_ids=tokens["input_ids"]).last_hidden_state
out_fp32 = out.float().numpy()

np.save({npy_path!r}, out_fp32)
print("output_path=" + {npy_path!r})
print("shape=" + str(list(out_fp32.shape)))
print("mean=" + format(float(out_fp32.mean()), ".6f"))
""")
        t0 = time.monotonic()
        result = subprocess.run(
            [python, "-c", script],
            capture_output=True, text=True, timeout=600,
            env=_ref_subprocess_env())
        elapsed = time.monotonic() - t0

        data: dict = {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        if os.path.exists(npy_path):
            data["output_path"] = npy_path

        return StageOutput(
            stage_name=stage.name,
            data=data,
            text=result.stdout,
            timing_s=elapsed,
            metadata={"backend": "torchtrt_diffusers"},
        )

    def _run_dit_step(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """Run HF DiT single forward with same deterministic inputs as TRT runner."""
        model_id = case.hf_id
        python = ctx.hf_python or sys.executable

        artifacts_dir = ctx.artifacts_dir or tempfile.gettempdir()
        model_dir = _case_artifact_dir(artifacts_dir, case.name)
        npy_path = os.path.join(model_dir, "hf_dit_output.npy")

        h_lat = case.inputs.get("image_height", 1024) // 8
        w_lat = case.inputs.get("image_width", 1024) // 8

        script = textwrap.dedent(f"""\
import torch, numpy as np
from diffusers import PixArtTransformer2DModel

# Same deterministic inputs as TRT runner (seed=42)
torch.manual_seed(42)
h_lat, w_lat = {h_lat}, {w_lat}
sample = torch.randn(1, 4, h_lat, w_lat, dtype=torch.float16)
text = torch.randn(1, 120, 4096, dtype=torch.float16)
timestep = torch.tensor([500.0], dtype=torch.float16)

model = PixArtTransformer2DModel.from_pretrained(
    {model_id!r}, subfolder="transformer", torch_dtype=torch.float16)
model.eval().cuda()
with torch.no_grad():
    out = model(
        hidden_states=sample.cuda(),
        encoder_hidden_states=text.cuda(),
        timestep=timestep.cuda(),
    ).sample.cpu()
out_fp32 = out.float().numpy()

del model
torch.cuda.empty_cache()

np.save({npy_path!r}, out_fp32)
print("output_path=" + {npy_path!r})
print("shape=" + str(list(out_fp32.shape)))
print("mean=" + format(float(out_fp32.mean()), ".6f"))
print("std=" + format(float(out_fp32.std()), ".6f"))
""")
        t0 = time.monotonic()
        result = subprocess.run(
            [python, "-c", script],
            capture_output=True, text=True, timeout=600,
            env=_ref_subprocess_env())
        elapsed = time.monotonic() - t0

        data: dict = {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        if os.path.exists(npy_path):
            data["output_path"] = npy_path

        return StageOutput(
            stage_name=stage.name,
            data=data,
            text=result.stdout,
            timing_s=elapsed,
            metadata={"backend": "torchtrt_diffusers"},
        )

    def _run_vae_decode(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """Run HF VAE decoder with same deterministic input as TRT runner."""
        model_id = case.hf_id
        python = ctx.hf_python or sys.executable

        artifacts_dir = ctx.artifacts_dir or tempfile.gettempdir()
        model_dir = _case_artifact_dir(artifacts_dir, case.name)
        npy_path = os.path.join(model_dir, "hf_vae_output.npy")

        h_lat = case.inputs.get("image_height", 1024) // 8
        w_lat = case.inputs.get("image_width", 1024) // 8

        script = textwrap.dedent(f"""\
import torch, numpy as np
from diffusers import AutoencoderKL

# Same deterministic input as TRT runner (seed=42)
torch.manual_seed(42)
h_lat, w_lat = {h_lat}, {w_lat}
latent = torch.randn(1, 4, h_lat, w_lat, dtype=torch.float16)

model = AutoencoderKL.from_pretrained(
    {model_id!r}, subfolder="vae", torch_dtype=torch.float16)
model.eval().cuda()
with torch.no_grad():
    # Apply same scaling as VAEDecoderWrapper
    scaled = latent.cuda() / 0.13025
    out = model.decode(scaled).sample.cpu()
out_fp32 = out.float().numpy()

del model
torch.cuda.empty_cache()

np.save({npy_path!r}, out_fp32)
print("output_path=" + {npy_path!r})
print("shape=" + str(list(out_fp32.shape)))
print("mean=" + format(float(out_fp32.mean()), ".6f"))
""")
        t0 = time.monotonic()
        result = subprocess.run(
            [python, "-c", script],
            capture_output=True, text=True, timeout=600,
            env=_ref_subprocess_env())
        elapsed = time.monotonic() - t0

        data: dict = {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        if os.path.exists(npy_path):
            data["output_path"] = npy_path

        return StageOutput(
            stage_name=stage.name,
            data=data,
            text=result.stdout,
            timing_s=elapsed,
            metadata={"backend": "torchtrt_diffusers"},
        )

    def _run_full_pipeline(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """Run full HF PixArt pipeline to generate reference frames."""
        model_id = case.hf_id
        prompt = case.inputs.get("prompt", case.inputs.get("test_prompt",
                                 "A photo of a cat sitting on a windowsill at sunset"))
        num_steps = case.inputs.get("num_inference_steps", 20)
        python = ctx.hf_python or sys.executable

        artifacts_dir = ctx.artifacts_dir or tempfile.gettempdir()
        model_dir = _case_artifact_dir(artifacts_dir, case.name)
        frames_dir = os.path.join(model_dir, "hf_frames")
        os.makedirs(frames_dir, exist_ok=True)

        script = textwrap.dedent(f"""\
import torch, numpy as np, os
from PIL import Image
from diffusers import PixArtSigmaPipeline

pipe = PixArtSigmaPipeline.from_pretrained({model_id!r}, torch_dtype=torch.float32)
pipe.to("cuda")
output = pipe(
    prompt={prompt!r},
    num_inference_steps={num_steps},
    height=1024, width=1024,
    generator=torch.Generator("cuda").manual_seed(42),
)
frames = output.images
for i, frame in enumerate(frames):
    frame.save(os.path.join({frames_dir!r}, f"frame_{{i:04d}}.png"))
print(f"Generated {{len(frames)}} frames")
""")
        t0 = time.monotonic()
        result = subprocess.run(
            [python, "-c", script],
            capture_output=True, text=True, timeout=3600,
            env=_ref_subprocess_env())
        elapsed = time.monotonic() - t0

        frame_files = sorted(Path(frames_dir).glob("frame_*.png"))

        if result.stderr:
            stderr_path = os.path.join(model_dir,
                                       "hf_torchtrt_diffusion_stderr.log")
            try:
                with open(stderr_path, "w") as f:
                    f.write(result.stderr)
            except OSError:
                pass
            if result.returncode != 0:
                logger.error("HF PixArt pipeline failed (rc=%d): %s",
                             result.returncode, result.stderr[-500:])

        data: dict = {
            "returncode": result.returncode,
            "num_frames": len(frame_files),
            "frames_dir": frames_dir,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

        if frame_files:
            try:
                import numpy as np
                from PIL import Image

                all_pixels = []
                for fp in frame_files:
                    img = Image.open(fp).convert("RGB")
                    arr = np.array(img, dtype=np.float32) / 255.0
                    all_pixels.append(arr.flatten())
                combined = np.concatenate(all_pixels)
                data["frame_stats"] = {
                    "count": len(frame_files),
                    "mean": float(np.mean(combined)),
                    "std": float(np.std(combined)),
                    "min": float(np.min(combined)),
                    "max": float(np.max(combined)),
                }
            except Exception as e:
                data["frame_stats_error"] = str(e)

        return StageOutput(
            stage_name=stage.name,
            data=data,
            text=result.stdout,
            timing_s=elapsed,
            metadata={"backend": "torchtrt_diffusers"},
        )


plugin = TorchTrtDiffusersReference()
