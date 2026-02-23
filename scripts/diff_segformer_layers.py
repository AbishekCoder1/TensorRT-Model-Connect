#!/usr/bin/env python3
"""Per-layer TRT-vs-HF comparison for SegFormer encoder.

Builds a debug TRT engine with per-stage/per-block outputs, runs
HF SegformerForSemanticSegmentation with hooks to capture the same
intermediates, and compares at each checkpoint.

Also tests input sensitivity: runs both zeros and a real image through
both backends and reports cosine similarity at each stage to pinpoint
where the signal dies.

Usage:
    python3 scripts/diff_segformer_layers.py \
      --model nvidia/segformer-b0-finetuned-ade-512-512 \
      --image tests/e2e/data/test_img.jpeg
"""
import argparse
import sys
from pathlib import Path

import numpy as np


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a_f, b_f = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
    denom = np.linalg.norm(a_f) * np.linalg.norm(b_f)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a_f, b_f) / denom)


def build_debug_engine(model_id_or_path: str, verbose: bool):
    """Build SegFormer TRT engine with debug outputs marked."""
    from trtf_build.engine_builder import _resolve_model
    from trtf_build.config import ModelConfig
    from trtf_build.families import find_plugin

    model_dir = _resolve_model(model_id_or_path)
    config = ModelConfig.from_dir(model_dir)
    plugin = find_plugin(config.model_type)
    if plugin is None:
        raise ValueError(f"No plugin for model_type={config.model_type!r}")

    print(f"[segformer-diff] Model: {config.model_type}", file=sys.stderr)
    weights = plugin.load_weights(model_dir, config)

    print(f"[segformer-diff] Building debug TRT engine ...", file=sys.stderr)
    engine_plan = plugin.build_engine(
        config, weights, 0, verbose=verbose, debug_layer_outputs=True)
    print(f"[segformer-diff] Engine built ({len(engine_plan) / 1e6:.1f} MB)",
          file=sys.stderr)

    return engine_plan, config, model_dir


def run_trt(engine_plan: bytes, pixel_values: np.ndarray) -> dict[str, np.ndarray]:
    """Run TRT engine and return all outputs (including debug)."""
    import tensorrt as trt
    try:
        from cuda.bindings import runtime as cudart
    except ImportError:
        from cuda import cudart  # type: ignore[no-redef]

    def _check(status):
        ok = getattr(cudart, 'cudaError_t', type(None))
        success = ok.cudaSuccess if hasattr(ok, 'cudaSuccess') else 0
        if status != success:
            raise RuntimeError(f"CUDA error: {status}")

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_plan)
    context = engine.create_execution_context()

    err, stream = cudart.cudaStreamCreate()
    _check(err)

    # Discover all IO tensors
    inputs, outputs = {}, {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = tuple(engine.get_tensor_shape(name))
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            inputs[name] = shape
        else:
            outputs[name] = shape

    # Allocate device buffers
    device_bufs = {}
    for name, shape in {**inputs, **outputs}.items():
        nbytes = int(np.prod(shape)) * 4  # float32
        err, ptr = cudart.cudaMalloc(nbytes)
        _check(err)
        device_bufs[name] = (ptr, shape, nbytes)

    # Copy input
    pv = np.ascontiguousarray(pixel_values, dtype=np.float32)
    ptr, shape, nbytes = device_bufs["pixel_values"]
    err, = cudart.cudaMemcpyAsync(
        ptr, pv.ctypes.data, nbytes,
        cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
    _check(err)

    # Set tensor addresses
    for name, (ptr, shape, nbytes) in device_bufs.items():
        context.set_tensor_address(name, ptr)

    # Execute
    context.execute_async_v3(stream)

    # Copy outputs back
    results = {}
    for name in outputs:
        ptr, shape, nbytes = device_bufs[name]
        host_buf = np.empty(shape, dtype=np.float32)
        err, = cudart.cudaMemcpyAsync(
            host_buf.ctypes.data, ptr, nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
        _check(err)
        results[name] = host_buf

    err, = cudart.cudaStreamSynchronize(stream)
    _check(err)

    # Cleanup
    for ptr, _, _ in device_bufs.values():
        cudart.cudaFree(ptr)
    cudart.cudaStreamDestroy(stream)

    return results


def run_hf_encoder_with_hooks(model_dir: str, pixel_values):
    """Run HF SegFormer encoder and capture per-stage/per-block outputs.

    Returns:
        patch_embeds: list of [1, C, H, W] numpy arrays per stage
        block_outputs: dict of (stage, block) -> [1, C, H, W] numpy
        stage_outputs: list of [1, C, H, W] numpy arrays per stage
    """
    import torch
    from transformers import SegformerForSemanticSegmentation

    model = SegformerForSemanticSegmentation.from_pretrained(
        model_dir, torch_dtype=torch.float32).eval()

    pv = torch.from_numpy(pixel_values) if isinstance(pixel_values, np.ndarray) \
        else pixel_values
    if pv.dim() == 3:
        pv = pv.unsqueeze(0)

    patch_embeds = []
    block_outputs = {}

    with torch.no_grad():
        encoder = model.segformer.encoder

        # Step through each stage manually to capture intermediates
        hidden_states_all = []
        all_hidden_states = ()

        # Patch embeddings and blocks
        for idx in range(len(encoder.patch_embeddings)):
            patch_embed = encoder.patch_embeddings[idx]
            blocks = encoder.block[idx]
            layer_norm = encoder.layer_norm[idx]

            if idx == 0:
                x_in = pv
            else:
                x_in = hidden_states_all[-1]

            # Patch embedding
            x_pe, height, width = patch_embed(x_in)
            # x_pe is [B, seq_len, hidden] — reshape to NCHW for comparison
            B = x_pe.shape[0]
            C = x_pe.shape[2]
            pe_nchw = x_pe.reshape(B, height, width, C).permute(0, 3, 1, 2)
            patch_embeds.append(pe_nchw.numpy())

            # Transformer blocks
            hidden = x_pe
            for blk_idx, blk in enumerate(blocks):
                layer_out = blk(hidden, height, width)
                # layer_out can be a tuple (hidden_states,) in some HF versions
                if isinstance(layer_out, tuple):
                    hidden = layer_out[0]
                else:
                    hidden = layer_out

                blk_nchw = hidden.reshape(B, height, width, C).permute(0, 3, 1, 2)
                block_outputs[(idx, blk_idx)] = blk_nchw.numpy()

            # Layer norm at end of stage
            hidden = layer_norm(hidden)

            # Reshape to NCHW: [B, H*W, C] -> [B, C, H, W]
            stage_out = hidden.reshape(B, height, width, C).permute(0, 3, 1, 2)
            hidden_states_all.append(stage_out)

        stage_outputs = [s.numpy() for s in hidden_states_all]

    return patch_embeds, block_outputs, stage_outputs


def preprocess_image(model_dir: str, image_path: str) -> np.ndarray:
    """Load and preprocess image using HF SegformerImageProcessor."""
    from transformers import SegformerImageProcessor
    from PIL import Image

    proc = SegformerImageProcessor.from_pretrained(model_dir)
    img = Image.open(image_path).convert("RGB")
    inputs = proc(images=img, return_tensors="np")
    return inputs["pixel_values"]  # [1, 3, H, W]


def print_comparison_table(rows: list[dict]):
    """Print a formatted comparison table."""
    hdr = (f"{'Checkpoint':<30} {'Shape':>18} {'MaxDiff':>10} "
           f"{'MeanDiff':>10} {'Cosine':>8} {'TRT_std':>10} "
           f"{'HF_std':>10} {'Status':>8}")
    print(f"\n{'=' * len(hdr)}")
    print(hdr)
    print(f"{'-' * len(hdr)}")

    for r in rows:
        status = "OK" if r.get("ok", True) else "FAIL"
        cos_str = f"{r['cosine']:.5f}" if r.get('cosine') is not None else "---"
        print(f"{r['name']:<30} {r['shape']:>18} "
              f"{r['max_diff']:>10.6f} {r['mean_diff']:>10.6f} "
              f"{cos_str:>8} {r['trt_std']:>10.4f} "
              f"{r['hf_std']:>10.4f} {status:>8}")


def compare_arrays(name: str, trt_arr: np.ndarray, hf_arr: np.ndarray,
                   atol: float) -> dict:
    """Compare two arrays and return a row dict."""
    diff = np.abs(trt_arr.astype(np.float64) - hf_arr.astype(np.float64))
    md = float(diff.max())
    return {
        "name": name,
        "shape": str(trt_arr.shape),
        "max_diff": md,
        "mean_diff": float(diff.mean()),
        "cosine": _cosine(trt_arr, hf_arr),
        "trt_std": float(trt_arr.std()),
        "hf_std": float(hf_arr.std()),
        "ok": md <= atol,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Per-layer TRT-vs-HF comparison for SegFormer")
    parser.add_argument("--model", required=True,
                        help="HF repo ID or local model directory")
    parser.add_argument("--image", required=True,
                        help="Path to test image")
    parser.add_argument("--atol", type=float, default=0.01,
                        help="Absolute tolerance for comparison")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Build debug engine
    engine_plan, config, model_dir = build_debug_engine(args.model, args.verbose)

    raw = config.raw
    num_encoder_blocks = raw.get("depths", [2, 2, 2, 2])

    # Preprocess image
    print(f"[segformer-diff] Preprocessing image: {args.image}", file=sys.stderr)
    pv_real = preprocess_image(model_dir, args.image)
    pv_zero = np.zeros_like(pv_real)

    # ---- Part 1: TRT vs HF comparison (real image) ----
    print(f"\n[segformer-diff] Running TRT (real image) ...", file=sys.stderr)
    trt_real = run_trt(engine_plan, pv_real)

    print(f"[segformer-diff] Running HF (real image) ...", file=sys.stderr)
    hf_pe, hf_blocks, hf_stages = run_hf_encoder_with_hooks(model_dir, pv_real)

    print(f"\n## Part 1: TRT vs HF (real image)")
    rows = []
    for stage_idx in range(4):
        # Patch embed
        key = f"debug_stage{stage_idx}_patch_embed"
        if key in trt_real:
            rows.append(compare_arrays(
                f"stage{stage_idx}.patch_embed",
                trt_real[key], hf_pe[stage_idx], args.atol))

        # Per-block
        for blk_idx in range(num_encoder_blocks[stage_idx]):
            key = f"debug_stage{stage_idx}_block{blk_idx}"
            if key in trt_real and (stage_idx, blk_idx) in hf_blocks:
                rows.append(compare_arrays(
                    f"stage{stage_idx}.block{blk_idx}",
                    trt_real[key], hf_blocks[(stage_idx, blk_idx)], args.atol))

        # Stage output
        key = f"debug_stage{stage_idx}"
        if key in trt_real:
            rows.append(compare_arrays(
                f"stage{stage_idx} (final)",
                trt_real[key], hf_stages[stage_idx], args.atol))

    # Logits
    if "logits" in trt_real:
        import torch
        from transformers import SegformerForSemanticSegmentation
        hf_model = SegformerForSemanticSegmentation.from_pretrained(
            model_dir, torch_dtype=torch.float32).eval()
        with torch.no_grad():
            hf_logits = hf_model(
                torch.from_numpy(pv_real)).logits.numpy()
        rows.append(compare_arrays("logits", trt_real["logits"], hf_logits, args.atol))
        del hf_model

    print_comparison_table(rows)

    # ---- Part 2: Input sensitivity (zeros vs real) ----
    print(f"\n[segformer-diff] Running TRT (zeros) ...", file=sys.stderr)
    trt_zero = run_trt(engine_plan, pv_zero)

    print(f"[segformer-diff] Running HF (zeros) ...", file=sys.stderr)
    hf_pe_z, hf_blocks_z, hf_stages_z = run_hf_encoder_with_hooks(model_dir, pv_zero)

    print(f"\n## Part 2: Input sensitivity (cosine between zeros and real)")
    hdr = (f"{'Checkpoint':<30} {'TRT cos(0,real)':>16} "
           f"{'HF cos(0,real)':>16} {'Diagnosis':>20}")
    print(f"\n{'=' * len(hdr)}")
    print(hdr)
    print(f"{'-' * len(hdr)}")

    for stage_idx in range(4):
        for blk_idx in range(num_encoder_blocks[stage_idx]):
            key = f"debug_stage{stage_idx}_block{blk_idx}"
            if key in trt_real and key in trt_zero:
                trt_cos = _cosine(trt_real[key], trt_zero[key])
                hf_key = (stage_idx, blk_idx)
                if hf_key in hf_blocks and hf_key in hf_blocks_z:
                    hf_cos = _cosine(hf_blocks[hf_key], hf_blocks_z[hf_key])
                else:
                    hf_cos = float('nan')

                diag = ""
                if trt_cos > 0.99 and hf_cos < 0.95:
                    diag = "** SIGNAL DEAD **"
                elif trt_cos > 0.99:
                    diag = "(both insensitive)"

                print(f"{'stage' + str(stage_idx) + '.block' + str(blk_idx):<30}"
                      f" {trt_cos:>16.5f} {hf_cos:>16.5f} {diag:>20}")

        key = f"debug_stage{stage_idx}"
        if key in trt_real and key in trt_zero:
            trt_cos = _cosine(trt_real[key], trt_zero[key])
            hf_cos = _cosine(hf_stages[stage_idx], hf_stages_z[stage_idx])

            diag = ""
            if trt_cos > 0.99 and hf_cos < 0.95:
                diag = "** SIGNAL DEAD **"
            elif trt_cos > 0.99:
                diag = "(both insensitive)"

            print(f"{'stage' + str(stage_idx) + ' (final)':<30}"
                  f" {trt_cos:>16.5f} {hf_cos:>16.5f} {diag:>20}")

    if "logits" in trt_real and "logits" in trt_zero:
        trt_cos = _cosine(trt_real["logits"], trt_zero["logits"])
        print(f"\n  TRT logits cos(zeros, real): {trt_cos:.5f}")

    # Summary
    all_ok = all(r.get("ok", True) for r in rows)
    print(f"\n{'PASS' if all_ok else 'FAIL'} (atol={args.atol})")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
