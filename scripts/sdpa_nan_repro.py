"""Reproducer: SDPA with attn_mask in cross-attention (different Q/KV seq lengths).

Tests multiple cases to identify which SDPA + mask configurations produce NaN
after torch_tensorrt compilation.

The NaN was observed in PixArt-Sigma DiT cross-attention where:
  Q = image patches [1, heads, 4096, head_dim]
  K/V = text tokens [1, heads, 120, head_dim]
  mask = [1, 1, 1, 120] additive bias (0 or -10000)

Versions tested:
  torch 2.10.0+cu128, torch_tensorrt 2.10.0+cu130, tensorrt 10.14.1.48, RTX 4090
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionSDPA(nn.Module):
    """Cross-attention using SDPA with attn_mask (diffusion-style)."""

    def __init__(self, dim=64, kv_dim=128, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(kv_dim, dim, bias=False)
        self.to_v = nn.Linear(kv_dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)

    def forward(self, x, context, attn_mask):
        B, Q, _ = x.shape
        KV = context.shape[1]
        q = self.to_q(x).view(B, Q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(context).view(B, KV, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(context).view(B, KV, self.num_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(B, Q, -1)
        return self.to_out(out)


class SelfAttentionSDPA(nn.Module):
    """Self-attention using SDPA with attn_mask."""

    def __init__(self, dim=64, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)

    def forward(self, x, attn_mask):
        B, S, _ = x.shape
        q = self.to_q(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(B, S, -1)
        return self.to_out(out)


def test_trt_compilation(name, model_cls, model_kwargs, input_fn, mask_fn):
    """Test a model with SDPA + mask through torch_tensorrt compilation."""
    import torch_tensorrt
    import tensorrt as trt

    print(f"\n{'='*60}")
    print(f"Test: {name}")
    print(f"{'='*60}")

    model = model_cls(**model_kwargs).cuda().half().eval()

    # Eager mode
    inputs_cuda = input_fn("cuda")
    mask_cuda = mask_fn("cuda")
    with torch.no_grad():
        eager_out = model(*inputs_cuda, mask_cuda)
    print(f"  Eager  — NaN: {eager_out.isnan().any().item()}, "
          f"mean: {eager_out.mean().item():.6f}")

    # TRT compilation
    cpu_model = model_cls(**model_kwargs).half().eval()
    cpu_model.load_state_dict(model.cpu().state_dict())

    inputs_cpu = input_fn("cpu")
    mask_cpu = mask_fn("cpu")

    try:
        exported = torch.export.export(
            cpu_model, (*inputs_cpu, mask_cpu), strict=False)

        engine_bytes = (
            torch_tensorrt.dynamo
            .convert_exported_program_to_serialized_trt_engine(
                exported,
                inputs=[*input_fn("cuda"), mask_fn("cuda")],
                use_explicit_typing=True,
                min_block_size=1,
            ))
    except Exception as e:
        print(f"  TRT    — COMPILATION FAILED: {e}")
        return

    # Deserialize and run
    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    ctx = engine.create_execution_context()
    stream = torch.cuda.Stream()

    test_inputs = input_fn("cuda")
    test_mask = mask_fn("cuda")
    out = torch.empty_like(eager_out.cuda())

    input_names = [engine.get_tensor_name(i)
                   for i in range(engine.num_io_tensors)
                   if engine.get_tensor_mode(engine.get_tensor_name(i))
                   == trt.TensorIOMode.INPUT]
    output_names = [engine.get_tensor_name(i)
                    for i in range(engine.num_io_tensors)
                    if engine.get_tensor_mode(engine.get_tensor_name(i))
                    == trt.TensorIOMode.OUTPUT]

    all_inputs = [*test_inputs, test_mask]
    for name_i, tensor in zip(input_names, all_inputs):
        ctx.set_input_shape(name_i, tuple(tensor.shape))
        ctx.set_tensor_address(name_i, tensor.data_ptr())
    for name_o in output_names:
        ctx.set_tensor_address(name_o, out.data_ptr())

    ctx.execute_async_v3(stream.cuda_stream)
    stream.synchronize()

    has_nan = out.isnan().any().item()
    if has_nan:
        non_nan = out[~out.isnan()]
        mean_str = f"{non_nan.mean().item():.6f}" if len(non_nan) > 0 else "ALL NaN"
        pct = out.isnan().float().mean().item() * 100
        print(f"  TRT    — NaN: True ({pct:.1f}% of values), "
              f"non-NaN mean: {mean_str}")
        print("  *** BUG: SDPA + attn_mask produces NaN after TRT compilation ***")
    else:
        # Check accuracy vs eager
        eager_cuda = eager_out.cuda()
        cos = F.cosine_similarity(out.flatten().unsqueeze(0),
                                   eager_cuda.flatten().unsqueeze(0)).item()
        print(f"  TRT    — NaN: False, mean: {out.mean().item():.6f}, "
              f"cosine vs eager: {cos:.6f}")


# === Test 1: Self-attention with all-zeros mask ===
test_trt_compilation(
    "Self-attention (seq=16) + zeros mask [1,1,16,16]",
    SelfAttentionSDPA, {"dim": 64, "num_heads": 4},
    input_fn=lambda dev: (torch.randn(1, 16, 64, device=dev, dtype=torch.float16),),
    mask_fn=lambda dev: torch.zeros(1, 1, 16, 16, device=dev, dtype=torch.float16),
)

# === Test 2: Self-attention with non-zero mask ===
test_trt_compilation(
    "Self-attention (seq=16) + padding mask (last 8 masked)",
    SelfAttentionSDPA, {"dim": 64, "num_heads": 4},
    input_fn=lambda dev: (torch.randn(1, 16, 64, device=dev, dtype=torch.float16),),
    mask_fn=lambda dev: torch.cat([
        torch.zeros(1, 1, 16, 8, device=dev, dtype=torch.float16),
        torch.full((1, 1, 16, 8), -10000.0, device=dev, dtype=torch.float16),
    ], dim=-1),
)

# === Test 3: Cross-attention with all-zeros mask ===
test_trt_compilation(
    "Cross-attention (Q=64, KV=16) + zeros mask",
    CrossAttentionSDPA, {"dim": 64, "kv_dim": 128, "num_heads": 4},
    input_fn=lambda dev: (
        torch.randn(1, 64, 64, device=dev, dtype=torch.float16),
        torch.randn(1, 16, 128, device=dev, dtype=torch.float16),
    ),
    mask_fn=lambda dev: torch.zeros(1, 1, 64, 16, device=dev, dtype=torch.float16),
)

# === Test 4: Cross-attention with padding mask ===
test_trt_compilation(
    "Cross-attention (Q=64, KV=16) + padding mask (last 8 KV masked)",
    CrossAttentionSDPA, {"dim": 64, "kv_dim": 128, "num_heads": 4},
    input_fn=lambda dev: (
        torch.randn(1, 64, 64, device=dev, dtype=torch.float16),
        torch.randn(1, 16, 128, device=dev, dtype=torch.float16),
    ),
    mask_fn=lambda dev: torch.cat([
        torch.zeros(1, 1, 64, 8, device=dev, dtype=torch.float16),
        torch.full((1, 1, 64, 8), -10000.0, device=dev, dtype=torch.float16),
    ], dim=-1),
)

# === Test 5: Larger cross-attention (PixArt-like scale) ===
test_trt_compilation(
    "Cross-attention (Q=256, KV=120) + padding mask (PixArt-like)",
    CrossAttentionSDPA, {"dim": 256, "kv_dim": 256, "num_heads": 8},
    input_fn=lambda dev: (
        torch.randn(1, 256, 256, device=dev, dtype=torch.float16),
        torch.randn(1, 120, 256, device=dev, dtype=torch.float16),
    ),
    mask_fn=lambda dev: torch.cat([
        torch.zeros(1, 1, 256, 60, device=dev, dtype=torch.float16),
        torch.full((1, 1, 256, 60), -10000.0, device=dev, dtype=torch.float16),
    ], dim=-1),
)

# === Test 6: 3D mask (broadcast over heads) — our workaround shape ===
test_trt_compilation(
    "Cross-attention (Q=64, KV=16) + 3D mask [1,1,16] broadcast",
    CrossAttentionSDPA, {"dim": 64, "kv_dim": 128, "num_heads": 4},
    input_fn=lambda dev: (
        torch.randn(1, 64, 64, device=dev, dtype=torch.float16),
        torch.randn(1, 16, 128, device=dev, dtype=torch.float16),
    ),
    mask_fn=lambda dev: torch.cat([
        torch.zeros(1, 1, 8, device=dev, dtype=torch.float16),
        torch.full((1, 1, 8), -10000.0, device=dev, dtype=torch.float16),
    ], dim=-1).unsqueeze(2),  # [1, 1, 1, 16] — broadcasts over Q dim
)

print("\n" + "="*60)
print("Done.")
print("="*60)
