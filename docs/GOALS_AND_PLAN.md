# Goals and Plan

## Long-term goal
Provide a C++ library that mirrors mainstream `transformers` APIs and model behaviors using TensorRT-first execution.

## Guiding approach
- API-first graph construction with TensorRT C++ API.
- ONNX parser is optional fallback, not default path.
- Missing ops are covered by plugins with naive but correct kernels first.

## Scope by phases

## Phase M0: runnable E2E vertical slice (completed)
- Implement `text-generation` pipeline API shape.
- Build backend abstraction (`trt`, `cpu-reference`).
- Make first E2E demo runnable and testable.
- Produce realistic project plan and coverage analysis.

## Phase M1: first real decoder-only model family
- Add weight loader (e.g., safetensors + metadata).
- Implement GPT/LLaMA-style decoder block with TensorRT API.
- Add KV cache management and batched decode loop.
- Add correctness tests against reference outputs for short prompts.

Current M1 increment delivered:
- TensorRT decoder-step graph with attention + MLP block shape.
- Host-side iterative decode loop with fixed-size KV-style cache updates.
- Force/fallback behavior preserved from M0 scaffolding.
- Tensor checkpoint loader (`weights.txt`) and model-driven TRT weight wiring (checkpoint tensors preferred over transition-derived logits).
- Local Hugging Face checkpoint execution path (`hf-transformers` backend) for parity validation against Python transformers.

## Phase M2: mainstream expansion
- Seq2seq generation path (T5/BART-like).
- Encoder-only families (BERT-like).
- Vision transformer path (ViT/CLIP image encoder).
- Unified tokenizer/model registry and model config normalization.

## Phase M3: productionization
- Quantization variants (FP16/BF16/INT8/FP8 where available).
- Plugin optimization and kernel fusion.
- Scheduler, batching, and streaming token APIs.
- CI matrix with GPU integration tests.

## Deliverables tracked continuously
- Architecture decisions and tradeoffs in `docs/WORKLOG.md`.
- Model coverage difficulty updates in `docs/TRANSFORMERS_COVERAGE_ANALYSIS.md`.
- Test intent and gaps in `docs/TEST_PLAN.md`.
