#!/usr/bin/env python3
"""Generate gold tensor fixtures for TRT graph ops using NumPy.

Each op gets a .safetensors file in tests/gold/ with named tensors:
  input, output, and any parameters needed.

Usage:
    python3 scripts/generate_op_gold_tensors.py [--output-dir tests/gold]
"""
import argparse
import math
import os
import struct
import json

import numpy as np


def write_safetensors(path: str, tensors: dict):
    """Write a dict of name -> numpy array to safetensors format."""
    header = {}
    offset = 0
    ordered_names = list(tensors.keys())

    for name in ordered_names:
        arr = tensors[name].astype(np.float32)
        byte_size = arr.nbytes
        header[name] = {
            "dtype": "F32",
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + byte_size],
        }
        offset += byte_size

    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_len = len(header_json)

    with open(path, "wb") as f:
        f.write(struct.pack("<Q", header_len))
        f.write(header_json)
        for name in ordered_names:
            f.write(tensors[name].astype(np.float32).tobytes())


def generate_rms_norm(output_dir: str):
    """Generate gold tensors for RMS norm."""
    np.random.seed(42)
    hidden_size = 16
    eps = 1e-5

    x = np.random.randn(1, hidden_size).astype(np.float32)
    gamma = np.random.randn(hidden_size).astype(np.float32) * 0.5 + 1.0

    # RMS norm: x * gamma / sqrt(mean(x^2) + eps)
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    output = (x / rms) * gamma

    write_safetensors(os.path.join(output_dir, "rms_norm.safetensors"), {
        "input": x,
        "gamma": gamma,
        "output": output.astype(np.float32),
        "eps": np.array([eps], dtype=np.float32),
    })
    print(f"  rms_norm: input={x.shape}, gamma={gamma.shape}, output={output.shape}")


def generate_matmul_rhs_constant(output_dir: str):
    """Generate gold tensors for matmul with constant RHS."""
    np.random.seed(43)
    m, k, n = 1, 16, 32

    x = np.random.randn(m, k).astype(np.float32)
    w = np.random.randn(k, n).astype(np.float32) * 0.1
    output = x @ w

    write_safetensors(os.path.join(output_dir, "matmul_rhs_constant.safetensors"), {
        "input": x,
        "weight": w,
        "output": output.astype(np.float32),
    })
    print(f"  matmul_rhs_constant: input={x.shape}, weight={w.shape}, output={output.shape}")


def generate_swiglu(output_dir: str):
    """Generate gold tensors for SwiGLU activation."""
    np.random.seed(44)
    size = 32

    gate = np.random.randn(1, size).astype(np.float32)
    up = np.random.randn(1, size).astype(np.float32)

    # SwiGLU: SiLU(gate) * up = gate * sigmoid(gate) * up
    sigmoid_gate = 1.0 / (1.0 + np.exp(-gate))
    swish = gate * sigmoid_gate
    output = swish * up

    write_safetensors(os.path.join(output_dir, "swiglu.safetensors"), {
        "gate": gate,
        "up": up,
        "output": output.astype(np.float32),
    })
    print(f"  swiglu: gate={gate.shape}, up={up.shape}, output={output.shape}")


def generate_rope(output_dir: str):
    """Generate gold tensors for rotary position embedding."""
    np.random.seed(45)
    hidden_size = 16
    num_heads = 2
    head_dim = hidden_size // num_heads
    seq_pos = 3
    theta = 10000.0

    x = np.random.randn(1, hidden_size).astype(np.float32)

    # Compute RoPE using the same formula as trtf's make_rope_table + rotate_half_matrix.
    # trtf does: output = x * cos_table[pos] + (x @ rotate_half_matrix) * sin_table[pos]
    #
    # rotate_half_matrix for row-vector multiply (x @ M):
    #   M[base+i, base+half+i] = 1.0   → rotate_half(x)[base+half+i] = x[base+i]
    #   M[base+half+i, base+i] = -1.0  → rotate_half(x)[base+i] = -x[base+half+i]
    #
    # So: rotate_half(x)[dim] = -x[dim+half] for dim < half, x[dim-half] for dim >= half
    # And: output[dim] = x[dim]*cos + rotate_half(x)[dim]*sin
    output = np.zeros_like(x)
    for h in range(num_heads):
        base = h * head_dim
        half = head_dim // 2
        for dim in range(head_dim):
            freq_idx = dim % half
            exponent = (2.0 * freq_idx) / head_dim
            inv_freq = theta ** (-exponent)
            angle = seq_pos * inv_freq
            cos_val = np.cos(angle)
            sin_val = np.sin(angle)

            if dim < half:
                rot_val = -x[0, base + half + dim]
            else:
                rot_val = x[0, base + dim - half]

            output[0, base + dim] = x[0, base + dim] * cos_val + rot_val * sin_val

    # Store metadata as F32 for SafetensorReader compatibility
    write_safetensors(os.path.join(output_dir, "rope.safetensors"), {
        "input": x,
        "output": output.astype(np.float32),
        "position": np.array([seq_pos], dtype=np.float32),
        "theta": np.array([theta], dtype=np.float32),
        "num_heads": np.array([num_heads], dtype=np.float32),
    })
    print(f"  rope: input={x.shape}, position={seq_pos}, output={output.shape}")


def generate_rms_norm_per_head(output_dir: str):
    """Generate gold tensors for per-head RMS norm."""
    np.random.seed(46)
    num_heads = 2
    head_dim = 8
    hidden_size = num_heads * head_dim
    eps = 1e-5

    x = np.random.randn(1, hidden_size).astype(np.float32)
    gamma = np.random.randn(hidden_size).astype(np.float32) * 0.5 + 1.0

    output = np.zeros_like(x)
    for h in range(num_heads):
        start = h * head_dim
        end = start + head_dim
        head_x = x[0, start:end]
        head_gamma = gamma[start:end]
        rms = np.sqrt(np.mean(head_x ** 2) + eps)
        output[0, start:end] = (head_x / rms) * head_gamma

    # Store metadata as F32 for SafetensorReader compatibility
    write_safetensors(os.path.join(output_dir, "rms_norm_per_head.safetensors"), {
        "input": x,
        "gamma": gamma,
        "output": output.astype(np.float32),
        "eps": np.array([eps], dtype=np.float32),
        "num_heads": np.array([num_heads], dtype=np.float32),
        "head_dim": np.array([head_dim], dtype=np.float32),
    })
    print(f"  rms_norm_per_head: input={x.shape}, gamma={gamma.shape}, output={output.shape}")


def generate_bias_sum(output_dir: str):
    """Generate gold tensors for bias addition."""
    np.random.seed(47)
    width = 16

    x = np.random.randn(1, width).astype(np.float32)
    bias = np.random.randn(width).astype(np.float32) * 0.1

    output = x + bias.reshape(1, width)

    write_safetensors(os.path.join(output_dir, "bias_sum.safetensors"), {
        "input": x,
        "bias": bias,
        "output": output.astype(np.float32),
    })
    print(f"  bias_sum: input={x.shape}, bias={bias.shape}, output={output.shape}")


def main():
    parser = argparse.ArgumentParser(description="Generate gold tensor fixtures for TRT ops")
    parser.add_argument("--output-dir", default="tests/gold", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Generating gold tensors in {args.output_dir}/")

    generate_rms_norm(args.output_dir)
    generate_matmul_rhs_constant(args.output_dir)
    generate_swiglu(args.output_dir)
    generate_rope(args.output_dir)
    generate_rms_norm_per_head(args.output_dir)
    generate_bias_sum(args.output_dir)

    print(f"\nDone. {6} gold tensor files written to {args.output_dir}/")


if __name__ == "__main__":
    main()
