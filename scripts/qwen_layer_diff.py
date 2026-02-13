#!/usr/bin/env python3
"""Layer/op diff framework for Qwen-style decoder math against HF ground truth.

This script runs token-by-token decode on a local HF Qwen model and compares:
- HF layer output (ground truth)
- per-head attention with current token included (expected faithful math)
- per-head attention with current token excluded
- flattened-head attention with current token excluded

It also captures HF eager-attention internals (q/k/v, scores, probs, context)
and emits op-level diffs for the selected variant.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
import transformers.utils.import_utils as hf_import_utils
from transformers import AutoModelForCausalLM, AutoTokenizer

# Force-disable torchvision dependency for this NLP-only diff workflow.
hf_import_utils._torchvision_available = False
from transformers.models.qwen3 import modeling_qwen3  # noqa: E402


@dataclass
class DiffRecord:
    step: int
    token_id: int
    layer: int
    variant: str
    max_abs: float
    mean_abs: float


@dataclass
class OpDiffRecord:
    step: int
    token_id: int
    layer: int
    variant: str
    op: str
    max_abs: float
    mean_abs: float


@dataclass
class AttentionCapture:
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    scores: torch.Tensor
    probs: torch.Tensor
    context: torch.Tensor


class AttentionCaptureState:
    def __init__(self) -> None:
        self.step = 0
        self.records: dict[tuple[int, int], AttentionCapture] = {}


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    denom = torch.rsqrt(torch.mean(x * x) + eps)
    return x * denom * weight


def per_head_rms_norm(x: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    # x: [num_heads, head_dim], gamma: [head_dim] or [num_heads, head_dim]
    if gamma.dim() == 1:
        gamma = gamma.unsqueeze(0).expand(x.shape[0], -1)
    denom = torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    return x * denom * gamma


def build_rope_cos_sin(
    position: int,
    head_dim: int,
    rope_theta: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if head_dim <= 0:
        raise ValueError("head_dim must be > 0")
    half = head_dim // 2
    if half <= 0:
        raise ValueError("head_dim must be >= 2 for RoPE")

    freq_idx = torch.arange(half, device=device, dtype=dtype)
    inv_freq = torch.pow(torch.tensor(rope_theta, device=device, dtype=dtype), -(2.0 * freq_idx) / float(head_dim))
    freqs = torch.tensor(float(position), device=device, dtype=dtype) * inv_freq

    # HF Qwen3 uses emb = cat(freqs, freqs), then cos/sin over emb.
    emb = torch.cat([freqs, freqs], dim=0)
    cos = torch.cos(emb)
    sin = torch.sin(emb)

    if emb.shape[0] < head_dim:
        cos = torch.cat([cos, torch.ones(head_dim - emb.shape[0], device=device, dtype=dtype)], dim=0)
        sin = torch.cat([sin, torch.zeros(head_dim - emb.shape[0], device=device, dtype=dtype)], dim=0)

    return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    # HF Qwen3 rotate_half: concat(-x[..., d/2:], x[..., :d/2]).
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos + rotate_half(x) * sin


def linear(x: torch.Tensor, weight_out_in: torch.Tensor) -> torch.Tensor:
    # x: [in], weight: [out, in]
    return torch.matmul(x, weight_out_in.transpose(0, 1))


def expand_kv_to_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    # x: [num_kv_heads, seq, head_dim]
    num_kv_heads = x.shape[0]
    if num_kv_heads <= 0:
        raise ValueError("num_kv_heads must be > 0")
    if num_heads % num_kv_heads != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")
    repeat = num_heads // num_kv_heads
    return x.repeat_interleave(repeat, dim=0)


def to_legacy_cache(cache: Any) -> Any:
    if cache is None:
        return None
    if hasattr(cache, "to_legacy_cache"):
        return cache.to_legacy_cache()
    return cache


def snapshot_legacy_cache(cache: Any) -> Any:
    legacy = to_legacy_cache(cache)
    if legacy is None:
        return None

    frozen = []
    for layer in legacy:
        if len(layer) < 2:
            raise RuntimeError("Unexpected cache layer tuple length")
        frozen.append((layer[0].detach().clone(), layer[1].detach().clone()))
    return tuple(frozen)


def cache_tensors_for_layer(
    past_legacy: Any,
    layer_idx: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if past_legacy is None:
        return (
            torch.empty((0, 0, 0), device=device, dtype=dtype),
            torch.empty((0, 0, 0), device=device, dtype=dtype),
        )

    key = past_legacy[layer_idx][0]
    value = past_legacy[layer_idx][1]

    # Expected HF shape: [batch, num_kv_heads, seq, head_dim]
    if key.dim() == 4:
        key = key[0]
        value = value[0]
    elif key.dim() != 3:
        raise RuntimeError(f"Unsupported cache key rank: {key.dim()}")

    # Return [num_kv_heads, seq, head_dim]
    return key.to(device=device, dtype=dtype), value.to(device=device, dtype=dtype)


def layer_forward_variant(
    x_in: torch.Tensor,
    past_k: torch.Tensor,
    past_v: torch.Tensor,
    layer: Any,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rope_theta: float,
    position: int,
    variant: str,
    collect_debug: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
    eps = float(getattr(layer.input_layernorm, "variance_epsilon", 1.0e-6))

    # Weights are stored as [out, in]
    w_q = layer.self_attn.q_proj.weight.to(dtype=x_in.dtype, device=x_in.device)
    w_k = layer.self_attn.k_proj.weight.to(dtype=x_in.dtype, device=x_in.device)
    w_v = layer.self_attn.v_proj.weight.to(dtype=x_in.dtype, device=x_in.device)
    w_o = layer.self_attn.o_proj.weight.to(dtype=x_in.dtype, device=x_in.device)

    w_gate = layer.mlp.gate_proj.weight.to(dtype=x_in.dtype, device=x_in.device)
    w_up = layer.mlp.up_proj.weight.to(dtype=x_in.dtype, device=x_in.device)
    w_down = layer.mlp.down_proj.weight.to(dtype=x_in.dtype, device=x_in.device)

    gamma_in = layer.input_layernorm.weight.to(dtype=x_in.dtype, device=x_in.device)
    gamma_post = layer.post_attention_layernorm.weight.to(dtype=x_in.dtype, device=x_in.device)

    q_norm_weight = None
    if hasattr(layer.self_attn, "q_norm") and hasattr(layer.self_attn.q_norm, "weight"):
        q_norm_weight = layer.self_attn.q_norm.weight.to(dtype=x_in.dtype, device=x_in.device)
    k_norm_weight = None
    if hasattr(layer.self_attn, "k_norm") and hasattr(layer.self_attn.k_norm, "weight"):
        k_norm_weight = layer.self_attn.k_norm.weight.to(dtype=x_in.dtype, device=x_in.device)

    x_norm = rms_norm(x_in, gamma_in, eps)

    q = linear(x_norm, w_q).reshape(num_heads, head_dim)
    k = linear(x_norm, w_k).reshape(num_kv_heads, head_dim)
    v = linear(x_norm, w_v).reshape(num_kv_heads, head_dim)

    if q_norm_weight is not None:
        q = per_head_rms_norm(q, q_norm_weight, eps)
    if k_norm_weight is not None:
        k = per_head_rms_norm(k, k_norm_weight, eps)

    cos, sin = build_rope_cos_sin(position, head_dim, rope_theta, q.device, q.dtype)
    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)

    past_len = int(past_k.shape[1]) if past_k.numel() > 0 else 0
    scale = 1.0 / math.sqrt(float(head_dim))

    debug: dict[str, torch.Tensor] | None = None

    if variant == "per_head_with_current":
        if past_len > 0:
            all_k = torch.cat([past_k, k.unsqueeze(1)], dim=1)
            all_v = torch.cat([past_v, v.unsqueeze(1)], dim=1)
        else:
            all_k = k.unsqueeze(1)
            all_v = v.unsqueeze(1)

        all_k_exp = expand_kv_to_heads(all_k, num_heads)
        all_v_exp = expand_kv_to_heads(all_v, num_heads)

        scores = torch.einsum("hd,hsd->hs", q, all_k_exp) * scale
        probs = torch.softmax(scores, dim=-1)
        context_heads = torch.einsum("hs,hsd->hd", probs, all_v_exp)
        context = context_heads.reshape(-1)

        if collect_debug:
            debug = {
                "q": q,
                "k": all_k_exp,
                "v": all_v_exp,
                "scores": scores,
                "probs": probs,
                "context": context_heads,
            }

    elif variant == "per_head_no_current":
        if past_len <= 0:
            context_heads = torch.zeros((num_heads, head_dim), device=x_in.device, dtype=x_in.dtype)
        else:
            all_k_exp = expand_kv_to_heads(past_k, num_heads)
            all_v_exp = expand_kv_to_heads(past_v, num_heads)
            scores = torch.einsum("hd,hsd->hs", q, all_k_exp) * scale
            probs = torch.softmax(scores, dim=-1)
            context_heads = torch.einsum("hs,hsd->hd", probs, all_v_exp)
        context = context_heads.reshape(-1)

    elif variant == "flat_no_current":
        if past_len <= 0:
            context = torch.zeros(num_heads * head_dim, device=x_in.device, dtype=x_in.dtype)
        else:
            all_k_exp = expand_kv_to_heads(past_k, num_heads)
            all_v_exp = expand_kv_to_heads(past_v, num_heads)
            q_flat = q.reshape(-1)
            k_flat = all_k_exp.permute(1, 0, 2).reshape(past_len, -1)
            v_flat = all_v_exp.permute(1, 0, 2).reshape(past_len, -1)
            scores = torch.matmul(k_flat, q_flat) * scale
            probs = torch.softmax(scores, dim=-1)
            context = torch.matmul(probs, v_flat)

    else:
        raise ValueError(f"Unknown variant: {variant}")

    attn_out = linear(context, w_o)
    residual1 = x_in + attn_out

    x_post = rms_norm(residual1, gamma_post, eps)
    gate = linear(x_post, w_gate)
    up = linear(x_post, w_up)
    down = linear(torch.sigmoid(gate) * gate * up, w_down)

    return residual1 + down, debug


def build_eager_capture_wrapper(
    state: AttentionCaptureState,
):
    def wrapped_eager_attention_forward(
        module: Any,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        scaling: float,
        dropout: float = 0.0,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del kwargs

        key_states = modeling_qwen3.repeat_kv(key, module.num_key_value_groups)
        value_states = modeling_qwen3.repeat_kv(value, module.num_key_value_groups)

        scores = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            scores = scores + causal_mask

        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        probs = F.dropout(probs, p=dropout, training=module.training)

        context = torch.matmul(probs, value_states)
        attn_output = context.transpose(1, 2).contiguous()

        state.records[(state.step, int(module.layer_idx))] = AttentionCapture(
            query=query.detach().to(device="cpu", dtype=torch.float32).clone(),
            key=key_states.detach().to(device="cpu", dtype=torch.float32).clone(),
            value=value_states.detach().to(device="cpu", dtype=torch.float32).clone(),
            scores=scores.detach().to(device="cpu", dtype=torch.float32).clone(),
            probs=probs.detach().to(device="cpu", dtype=torch.float32).clone(),
            context=attn_output.detach().to(device="cpu", dtype=torch.float32).clone(),
        )

        return attn_output, probs

    return wrapped_eager_attention_forward


def diff_metrics(pred: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    pred_f = pred.detach().to(device="cpu", dtype=torch.float32)
    ref_f = ref.detach().to(device="cpu", dtype=torch.float32)
    if pred_f.shape != ref_f.shape:
        raise RuntimeError(f"Shape mismatch while diffing tensors: {tuple(pred_f.shape)} vs {tuple(ref_f.shape)}")
    diff = torch.abs(pred_f - ref_f)
    return float(torch.max(diff).item()), float(torch.mean(diff).item())


def main() -> int:
    parser = argparse.ArgumentParser(description="Qwen layer/op diff vs HF ground truth")
    parser.add_argument("--model-dir", required=True, help="Local HF model dir (for example models/hf/Qwen__Qwen3-0.6B)")
    parser.add_argument("--prompt", default="Hello", help="Prompt to trace")
    parser.add_argument("--max-layers", type=int, default=4, help="Number of layers to diff (from 0 upward)")
    parser.add_argument("--max-steps", type=int, default=8, help="Number of prompt tokens to trace")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Execution device for diff script")
    parser.add_argument("--attn-implementation", default="eager", choices=["eager", "sdpa"], help="HF attention implementation used as ground truth")
    parser.add_argument("--op-variant", default="per_head_with_current", choices=["per_head_with_current", "per_head_no_current", "flat_no_current"], help="Variant used for op-level attention diff")
    parser.add_argument("--json-out", default="", help="Optional path to save JSON diff records")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
    ).to(device)
    model.eval()

    cfg = model.config
    num_heads = int(getattr(cfg, "num_attention_heads"))
    num_kv_heads = int(getattr(cfg, "num_key_value_heads", num_heads))
    head_dim = int(getattr(cfg, "head_dim", cfg.hidden_size // num_heads))
    rope_theta = float(getattr(cfg, "rope_theta", 10000.0))

    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(device)
    max_steps = min(args.max_steps, int(input_ids.shape[1]))
    max_layers = min(args.max_layers, len(model.model.layers))

    variants = [
        "per_head_with_current",
        "per_head_no_current",
        "flat_no_current",
    ]

    capture_state: AttentionCaptureState | None = None
    original_eager = None
    if args.attn_implementation == "eager":
        capture_state = AttentionCaptureState()
        original_eager = modeling_qwen3.eager_attention_forward
        modeling_qwen3.eager_attention_forward = build_eager_capture_wrapper(capture_state)
    else:
        print("op-level capture is only enabled for eager attention; continuing with layer-level diff only")

    records: list[DiffRecord] = []
    op_records: list[OpDiffRecord] = []

    past = None
    try:
        for step in range(max_steps):
            token_id = int(input_ids[0, step].item())
            token_input = input_ids[:, step : step + 1]
            past_legacy = snapshot_legacy_cache(past)

            if capture_state is not None:
                capture_state.step = step

            with torch.no_grad():
                out = model(
                    input_ids=token_input,
                    past_key_values=past,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True,
                )

            hidden_states = out.hidden_states
            if hidden_states is None:
                raise RuntimeError("Model did not return hidden_states")

            print(f"step={step} token_id={token_id}")

            for layer_idx in range(max_layers):
                layer = model.model.layers[layer_idx]
                x_in = hidden_states[layer_idx][0, 0, :].to(device=device, dtype=dtype)
                hf_out = hidden_states[layer_idx + 1][0, 0, :].to(device=device, dtype=dtype)

                past_k, past_v = cache_tensors_for_layer(past_legacy, layer_idx, device=device, dtype=dtype)
                position = int(past_k.shape[1]) if past_k.numel() > 0 else 0

                layer_msgs = []
                op_debug: dict[str, torch.Tensor] | None = None

                for variant in variants:
                    collect_debug = capture_state is not None and variant == args.op_variant
                    pred, debug = layer_forward_variant(
                        x_in=x_in,
                        past_k=past_k,
                        past_v=past_v,
                        layer=layer,
                        num_heads=num_heads,
                        num_kv_heads=num_kv_heads,
                        head_dim=head_dim,
                        rope_theta=rope_theta,
                        position=position,
                        variant=variant,
                        collect_debug=collect_debug,
                    )
                    if collect_debug:
                        op_debug = debug

                    max_abs, mean_abs = diff_metrics(pred, hf_out)
                    records.append(
                        DiffRecord(
                            step=step,
                            token_id=token_id,
                            layer=layer_idx,
                            variant=variant,
                            max_abs=max_abs,
                            mean_abs=mean_abs,
                        )
                    )
                    layer_msgs.append(f"{variant}:max={max_abs:.6f},mean={mean_abs:.6f}")

                print(f"  layer={layer_idx} " + " | ".join(layer_msgs))

                if capture_state is None:
                    continue

                hf_cap = capture_state.records.get((step, layer_idx))
                if hf_cap is None:
                    print("    ops: missing HF capture")
                    continue
                if op_debug is None:
                    print("    ops: missing variant debug tensors")
                    continue

                hf_q = hf_cap.query[0, :, 0, :]
                hf_k = hf_cap.key[0]
                hf_v = hf_cap.value[0]
                hf_scores = hf_cap.scores[0, :, 0, :]
                hf_probs = hf_cap.probs[0, :, 0, :]
                hf_context = hf_cap.context[0, 0, :, :]

                op_tensors = {
                    "q": (op_debug["q"], hf_q),
                    "k": (op_debug["k"], hf_k),
                    "v": (op_debug["v"], hf_v),
                    "scores": (op_debug["scores"], hf_scores),
                    "probs": (op_debug["probs"], hf_probs),
                    "context": (op_debug["context"], hf_context),
                }

                op_msgs = []
                for op_name, (pred_op, hf_op) in op_tensors.items():
                    op_max, op_mean = diff_metrics(pred_op, hf_op)
                    op_records.append(
                        OpDiffRecord(
                            step=step,
                            token_id=token_id,
                            layer=layer_idx,
                            variant=args.op_variant,
                            op=op_name,
                            max_abs=op_max,
                            mean_abs=op_mean,
                        )
                    )
                    op_msgs.append(f"{op_name}:max={op_max:.6f},mean={op_mean:.6f}")
                print("    ops " + " | ".join(op_msgs))

            past = out.past_key_values
    finally:
        if original_eager is not None:
            modeling_qwen3.eager_attention_forward = original_eager

    # Aggregate summary
    print("\nSummary")
    summary: dict[str, dict[str, float | int]] = {}
    for variant in variants:
        vr = [r for r in records if r.variant == variant]
        if not vr:
            continue
        summary[variant] = {
            "max_abs": max(r.max_abs for r in vr),
            "mean_abs": sum(r.mean_abs for r in vr) / float(len(vr)),
            "records": len(vr),
        }
        print(
            f"  {variant}: max_abs={summary[variant]['max_abs']:.6f}, "
            f"mean_abs={summary[variant]['mean_abs']:.6f}, n={summary[variant]['records']}"
        )

    op_summary: dict[str, dict[str, float | int]] = {}
    if op_records:
        print("\nOp Summary")
        keys = sorted({(r.variant, r.op) for r in op_records})
        for variant, op in keys:
            rows = [r for r in op_records if r.variant == variant and r.op == op]
            op_key = f"{variant}:{op}"
            op_summary[op_key] = {
                "max_abs": max(r.max_abs for r in rows),
                "mean_abs": sum(r.mean_abs for r in rows) / float(len(rows)),
                "records": len(rows),
            }
            print(
                f"  {op_key}: max_abs={op_summary[op_key]['max_abs']:.6f}, "
                f"mean_abs={op_summary[op_key]['mean_abs']:.6f}, n={op_summary[op_key]['records']}"
            )

    if args.json_out:
        payload = {
            "model_dir": args.model_dir,
            "prompt": args.prompt,
            "max_layers": max_layers,
            "max_steps": max_steps,
            "summary": summary,
            "records": [r.__dict__ for r in records],
            "op_summary": op_summary,
            "op_records": [r.__dict__ for r in op_records],
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"wrote JSON: {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
