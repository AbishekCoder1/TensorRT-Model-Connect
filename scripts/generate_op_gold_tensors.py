#!/usr/bin/env python3
"""Generate gold tensor fixtures for TRT graph ops using PyTorch.

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

    # Compute RoPE frequencies
    inv_freq = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    freqs = seq_pos * inv_freq
    cos_vals = np.cos(freqs)
    sin_vals = np.sin(freqs)

    # Apply RoPE per head
    output = np.zeros_like(x)
    for h in range(num_heads):
        start = h * head_dim
        for i in range(head_dim // 2):
            x1 = x[0, start + i]
            x2 = x[0, start + head_dim // 2 + i]
            output[0, start + i] = x1 * cos_vals[i] - x2 * sin_vals[i]
            output[0, start + head_dim // 2 + i] = x2 * cos_vals[i] + x1 * sin_vals[i]

    write_safetensors(os.path.join(output_dir, "rope.safetensors"), {
        "input": x,
        "output": output.astype(np.float32),
        "position": np.array([seq_pos], dtype=np.int32),
        "theta": np.array([theta], dtype=np.float32),
        "num_heads": np.array([num_heads], dtype=np.int32),
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

    write_safetensors(os.path.join(output_dir, "rms_norm_per_head.safetensors"), {
        "input": x,
        "gamma": gamma,
        "output": output.astype(np.float32),
        "eps": np.array([eps], dtype=np.float32),
        "num_heads": np.array([num_heads], dtype=np.int32),
        "head_dim": np.array([head_dim], dtype=np.int32),
    })
    print(f"  rms_norm_per_head: input={x.shape}, gamma={gamma.shape}, output={output.shape}")


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

    print(f"\nDone. {5} gold tensor files written to {args.output_dir}/")


if __name__ == "__main__":
    main()
