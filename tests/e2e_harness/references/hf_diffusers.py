"""HF Diffusers reference backend.

Runs HuggingFace diffusers pipeline as a subprocess for GPU isolation,
producing per-stage reference outputs for comparison against TRT.

Supports Wan-style text-to-video pipelines with per-stage extraction
(T5 encoding, single DiT step, full denoising loop, VAE decode).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .. import _case_artifact_dir
from ..contracts import E2ECase, RunContext, StageOutput, StageSpec

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parents[3]


class HfDiffusersReference:
    """Reference backend using HuggingFace diffusers pipelines."""

    @property
    def backend_name(self) -> str:
        return "hf_diffusers"

    def run_stage(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        dispatch = {
            "t5_encode": self._run_t5_encode,
            "dit_step": self._run_dit_step,
            "end_to_end": self._run_full_pipeline,
            "end_to_end_video": self._run_full_pipeline,
            "generate": self._run_full_pipeline,
            "debug_pipeline": self._run_debug_pipeline,
            "vae_decode": self._run_full_pipeline,
            "frame_quality": self._run_full_pipeline,
            "crossover_ref_t5_trt_dit": self._run_crossover_noop,
            "crossover_trt_t5_ref_dit": self._run_crossover_noop,
        }
        handler = dispatch.get(stage.name)
        if handler is None:
            return StageOutput(
                stage_name=stage.name,
                data={"error": f"Unknown diffusers reference stage: {stage.name}"},
            )
        return handler(case, stage, ctx)

    def _run_t5_encode(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """Run HF T5 encoding and save output for comparison.

        Supports Wan (WanPipeline), Flux (FluxPipeline), and generic
        diffusers pipelines. Falls back gracefully if the pipeline class
        is not available.
        """
        model_id = case.hf_id
        prompt = case.inputs.get("test_prompt", "A cat sitting on a beach")
        python = ctx.hf_python or sys.executable

        # Save to artifacts_dir so the file persists for comparator access
        artifacts_dir = ctx.artifacts_dir or tempfile.gettempdir()
        model_dir = _case_artifact_dir(artifacts_dir, case.name) if ctx.artifacts_dir else artifacts_dir
        os.makedirs(model_dir, exist_ok=True)
        output_path = os.path.join(model_dir, "hf_t5_output.npy")

        script = f"""
import torch, numpy as np, sys

model_id = {model_id!r}
prompt = {prompt!r}
output_path = {output_path!r}

# Try pipeline classes in order: Wan, Flux, generic DiffusionPipeline
pipe = None
for cls_name in ["WanPipeline", "FluxPipeline", "DiffusionPipeline"]:
    try:
        import diffusers
        cls = getattr(diffusers, cls_name, None)
        if cls is None:
            continue
        pipe = cls.from_pretrained(
            model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True)
        print(f"Loaded {{cls_name}}", file=sys.stderr)
        break
    except Exception as e:
        print(f"{{cls_name}} failed: {{e}}", file=sys.stderr)
        continue

if pipe is None:
    print("ERROR: no diffusers pipeline could load this model", file=sys.stderr)
    sys.exit(1)

# Extract text encoder and tokenizer (different pipelines use different names)
text_encoder = getattr(pipe, "text_encoder", None)
if text_encoder is None:
    text_encoder = getattr(pipe, "text_encoder_2", None)

tokenizer = getattr(pipe, "tokenizer", None)
if tokenizer is None:
    tokenizer = getattr(pipe, "tokenizer_2", None)

if text_encoder is None or tokenizer is None:
    print("ERROR: pipeline has no text_encoder/tokenizer", file=sys.stderr)
    sys.exit(1)

tokens = tokenizer(prompt, return_tensors="pt", padding="max_length",
                    max_length=512, truncation=True)
with torch.no_grad():
    t5_out = text_encoder(tokens.input_ids)[0]

np.save(output_path, t5_out.numpy())
print(f"shape={{list(t5_out.shape)}}")
print(f"mean={{float(t5_out.mean()):.6f}}")
"""
        t0 = time.monotonic()
        result = subprocess.run(
            [python, "-c", script],
            capture_output=True, text=True, timeout=600)
        elapsed = time.monotonic() - t0

        data: dict = {
            "returncode": result.returncode,
            "stdout": result.stdout,
        }
        if os.path.exists(output_path):
            data["output_path"] = output_path

        return StageOutput(
            stage_name=stage.name,
            data=data,
            text=result.stdout,
            timing_s=elapsed,
            metadata={"backend": "hf_diffusers"},
        )

    def _run_dit_step(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """Run HF DiT forward pass for comparison."""
        # This is implicitly compared in the debug_pipeline flow
        return self._run_debug_pipeline(case, stage, ctx)

    def _run_debug_pipeline(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """The debug_pipeline script runs BOTH TRT and HF internally.

        For the reference side, we report the HF results from the same script.
        The diffusion comparator handles the joint output.
        """
        # Return a marker output indicating that debug_pipeline handles
        # both sides. The comparator will use the TRT runner's output
        # which contains both TRT and HF comparison results.
        return StageOutput(
            stage_name=stage.name,
            data={
                "backend": "hf_diffusers",
                "note": "debug_pipeline runs HF internally; comparison embedded in TRT output",
            },
            timing_s=0.0,
            metadata={"backend": "hf_diffusers"},
        )

    def _run_crossover_noop(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """Crossover stages are self-contained in the runner (mix TRT + HF).

        The runner subprocess handles both the HF and TRT components itself,
        so the reference side returns a no-op marker.
        """
        return StageOutput(
            stage_name=stage.name,
            data={
                "backend": "hf_diffusers",
                "note": "Crossover stage is self-contained in runner; "
                        "reference is embedded in the runner subprocess",
            },
            timing_s=0.0,
            metadata={"backend": "hf_diffusers"},
        )

    def _run_full_pipeline(
        self, case: E2ECase, stage: StageSpec, ctx: RunContext
    ) -> StageOutput:
        """Run full HF diffusers pipeline to generate reference frames."""
        model_id = case.hf_id
        prompt = case.inputs.get("test_prompt", "A cat sitting on a beach")
        num_steps = case.inputs.get("num_inference_steps", 30)
        python = ctx.hf_python or sys.executable

        # Save frames to artifacts_dir so they persist for comparator access
        artifacts_dir = ctx.artifacts_dir or tempfile.gettempdir()
        model_dir = _case_artifact_dir(artifacts_dir, case.name) if ctx.artifacts_dir else artifacts_dir
        frames_dir = os.path.join(model_dir, "hf_frames")
        os.makedirs(frames_dir, exist_ok=True)

        script = f"""
import torch
import numpy as np
from diffusers import WanPipeline
from PIL import Image
import os

pipe = WanPipeline.from_pretrained(
    {model_id!r}, torch_dtype=torch.float16)
pipe.to("cuda")

output = pipe(
    prompt={prompt!r},
    num_inference_steps={num_steps},
    num_frames=17,
    guidance_scale=5.0,
    generator=torch.Generator("cuda").manual_seed(42),
)

frames = output.frames[0]
for i, frame in enumerate(frames):
    if isinstance(frame, Image.Image):
        frame.save(os.path.join({frames_dir!r}, f"frame_{{i:04d}}.png"))
    else:
        img = Image.fromarray(np.uint8(frame * 255) if frame.max() <= 1.0 else np.uint8(frame))
        img.save(os.path.join({frames_dir!r}, f"frame_{{i:04d}}.png"))

print(f"Generated {{len(frames)}} frames")
"""
        t0 = time.monotonic()
        result = subprocess.run(
            [python, "-c", script],
            capture_output=True, text=True, timeout=3600)
        elapsed = time.monotonic() - t0

        frame_files = sorted(Path(frames_dir).glob("frame_*.png"))

        data: dict = {
            "returncode": result.returncode,
            "num_frames": len(frame_files),
            "frames_dir": frames_dir,
            "stdout": result.stdout,
        }

        # Compute frame statistics for reference
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
            metadata={"backend": "hf_diffusers"},
        )


plugin = HfDiffusersReference()
