#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import Cache


REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = REPO_ROOT / "artifacts" / "triattention" / "upstream"
sys.path.insert(0, str(UPSTREAM_ROOT))

from triattention.methods.triattention import apply_triattention_patch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-jsonl", required=True)
    parser.add_argument("--native-trace-jsonl", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--mode", choices=["dense", "tri"], default="dense")
    parser.add_argument("--stats-path")
    parser.add_argument("--kv-budget", type=int, default=0)
    parser.add_argument("--divide-length", type=int, default=128)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument(
        "--dtype", choices=["bfloat16", "float16"], default="bfloat16"
    )
    parser.add_argument(
        "--attn-implementation",
        choices=["eager", "sdpa", "flash_attention_2"],
        default="eager",
    )
    parser.add_argument("--start-position", type=int, default=0)
    parser.add_argument("--end-position", type=int, default=2**31 - 1)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--hidden-dump-dir")
    parser.add_argument("--warmup-chunk-size", type=int, default=0)
    parser.add_argument(
        "--count-prompt-tokens",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def load_sample_prompt(path: Path) -> str:
    return json.loads(path.read_text().splitlines()[0])["prompt"]


def load_native_trace(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows.sort(key=lambda row: int(row["position_before"]))
    return rows


def get_cache_len(past_key_values: Any) -> int:
    if past_key_values is None:
        return 0
    if isinstance(past_key_values, Cache):
        return int(past_key_values.get_seq_length())
    if isinstance(past_key_values, (tuple, list)) and past_key_values:
        return int(past_key_values[0][0].shape[2])
    return 0


def get_comp_state(model: torch.nn.Module) -> dict[str, Any]:
    comp = getattr(model, "_triattention_compressor", None)
    if comp is None:
        return {}
    return {
        "absolute_position": int(comp.absolute_position),
        "cache_positions_len": int(len(comp.cache_positions)),
        "prefix_length": int(comp.prefix_length),
    }


def dump_hidden_states(hidden_states: tuple[torch.Tensor, ...], out_path: Path) -> None:
    payload = {
        "layers": [
            layer[:, -1, :].detach().to(dtype=torch.float16, device="cpu").contiguous()
            for layer in hidden_states
        ]
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)


def main() -> None:
    args = parse_args()
    sample_path = Path(args.sample_jsonl)
    trace_path = Path(args.native_trace_jsonl)
    output_path = Path(args.output_jsonl)
    hidden_dump_dir = Path(args.hidden_dump_dir) if args.hidden_dump_dir else None

    prompt = load_sample_prompt(sample_path)
    native_rows = load_native_trace(trace_path)

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="auto",
        use_cache=True,
        attn_implementation=args.attn_implementation,
    )
    model.eval()

    if args.mode == "tri":
        if not args.stats_path or args.kv_budget <= 0:
            raise ValueError("--stats-path and --kv-budget are required in tri mode")
        apply_triattention_patch(
            model,
            stats_path=Path(args.stats_path),
            model_path=Path(args.model_path),
            kv_budget=int(args.kv_budget),
            offset_max_length=65536,
            score_aggregation="mean",
            pruning_seed=0,
            metadata_expectations={},
            normalize_scores=True,
            count_prompt_tokens=bool(args.count_prompt_tokens),
            allow_prefill_compression=False,
            divide_length=int(args.divide_length),
            use_slack_trigger=True,
            per_head_pruning=True,
            per_layer_perhead_pruning=False,
            layer_perhead_aggregation="max",
            disable_mlr=False,
            disable_trig=False,
        )

    encoded = tokenizer(
        [prompt], padding="longest", return_tensors="pt", add_special_tokens=True
    ).to("cuda")

    with torch.no_grad():
        out = model(
            **encoded,
            use_cache=True,
            return_dict=True,
            output_hidden_states=False,
        )

    past_key_values = out.past_key_values
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_token_count = int(encoded["input_ids"].shape[1])

    # Native TRT step traces start at position 0 and include the prompt-side
    # prefill steps. When we already prefetched the prompt through HF here, we
    # must skip those prompt rows or we will double-count the prompt and drift
    # all later RoPE/cache positions by exactly the prompt length.
    trace_includes_prompt = bool(native_rows) and int(native_rows[0]["position_before"]) == 0
    replay_source_rows = native_rows
    if trace_includes_prompt:
        replay_source_rows = [
            row for row in native_rows if int(row["position_before"]) >= prompt_token_count
        ]

    warmup_rows = [
        row for row in replay_source_rows if int(row["position_before"]) < args.start_position
    ]
    replay_rows = [
        row
        for row in replay_source_rows
        if args.start_position <= int(row["position_before"]) <= args.end_position
    ]

    if args.warmup_chunk_size > 0 and warmup_rows:
        chunk_size = max(1, int(args.warmup_chunk_size))
        for start in range(0, len(warmup_rows), chunk_size):
            chunk = warmup_rows[start : start + chunk_size]
            chunk_ids = torch.tensor(
                [[int(row["token_id"]) for row in chunk]],
                device="cuda",
                dtype=torch.long,
            )
            with torch.no_grad():
                out = model(
                    input_ids=chunk_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                    output_hidden_states=False,
                )
            past_key_values = out.past_key_values

    with output_path.open("w") as fout:
        for row in replay_rows:
            position_before = int(row["position_before"])
            token_id = int(row["token_id"])
            want_hidden = hidden_dump_dir is not None
            rows_before = get_cache_len(past_key_values)
            comp_before = get_comp_state(model)

            with torch.no_grad():
                out = model(
                    input_ids=torch.tensor([[token_id]], device="cuda", dtype=torch.long),
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                    output_hidden_states=want_hidden,
                )

            past_key_values = out.past_key_values
            rows_after = get_cache_len(past_key_values)
            logits = out.logits[0, -1].detach().to(dtype=torch.float32, device="cpu")
            top_k = min(args.top_k, int(logits.shape[0]))
            top_vals, top_ids = torch.topk(logits, k=top_k)
            comp_after = get_comp_state(model)

            record = {
                "position_before": position_before,
                "token_id": token_id,
                "rows_before": rows_before,
                "rows_after": rows_after,
                "argmax_token": int(top_ids[0]),
                "argmax_logit": float(top_vals[0]),
                "top_ids": [int(x) for x in top_ids.tolist()],
                "top_logits": [float(x) for x in top_vals.tolist()],
            }
            if comp_before or comp_after:
                record["comp_before"] = comp_before
                record["comp_after"] = comp_after
            fout.write(json.dumps(record) + "\n")
            fout.flush()

            if want_hidden and out.hidden_states is not None:
                dump_hidden_states(
                    out.hidden_states,
                    hidden_dump_dir / f"pos_{position_before}.pt",
                )


if __name__ == "__main__":
    main()
