#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList


REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = REPO_ROOT / "artifacts" / "triattention" / "upstream"
sys.path.insert(0, str(UPSTREAM_ROOT))

from triattention.methods.triattention import apply_triattention_patch


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


class BoxedAnswerStopper(StoppingCriteria):
    def __init__(self, tokenizer, prompt_length: int, check_every: int = 4) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.check_every = max(1, check_every)
        self.last_checked_len = 0

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        generated = input_ids[0, self.prompt_length :]
        gen_len = int(generated.shape[0])
        if gen_len <= 0 or gen_len == self.last_checked_len or gen_len % self.check_every != 0:
            return False
        self.last_checked_len = gen_len
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        marker = "\\boxed{"
        start = text.rfind(marker)
        if start == -1:
            return False
        return "}" in text[start + len(marker) :]


def extract_boxed_answer(text: str) -> str:
    marker = "boxed"
    idx = text.rfind(marker)
    if idx == -1:
        return ""
    tail = text[idx + len(marker) :]
    if not tail:
        return ""
    if tail[0] != "{":
        return tail.split("$")[0].strip()
    depth = 1
    out = []
    for ch in tail[1:]:
        if ch == "{":
            depth += 1
            out.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
            out.append(ch)
        else:
            out.append(ch)
    return "".join(out).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-jsonl", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--stats-path", required=True)
    parser.add_argument("--kv-budget", type=int, required=True)
    parser.add_argument("--divide-length", type=int, required=True)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--attn-implementation", choices=["eager", "sdpa", "flash_attention_2"], default="eager")
    parser.add_argument(
        "--count-prompt-tokens",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample = json.loads(Path(args.sample_jsonl).read_text().splitlines()[0])
    prompt = sample["prompt"]
    gold = sample["answer"]

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

    encoded = tokenizer([prompt], padding="longest", return_tensors="pt", add_special_tokens=True).to("cuda")
    prompt_len = int(encoded["attention_mask"].sum().item())

    set_seed(args.seed)
    stopper = BoxedAnswerStopper(tokenizer, prompt_len)
    do_sample = args.temperature > 0.0
    with torch.no_grad():
        out = model.generate(
            **encoded,
            max_new_tokens=args.max_new_tokens,
            do_sample=do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            stopping_criteria=StoppingCriteriaList([stopper]),
            use_cache=True,
        )

    generated = out[0, prompt_len:]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    pred = extract_boxed_answer(text)
    result = {
        "sample_id": sample.get("sample_id"),
        "gold_answer": gold,
        "pred_answer": pred,
        "generated_tokens": int(generated.shape[0]),
        "stopped_on_boxed": stopper.last_checked_len > 0 and stopper.last_checked_len == int(generated.shape[0]),
        "text": text,
    }
    print(json.dumps(result, indent=2))
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
