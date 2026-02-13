# Test Plan and Intentions

## Principles
- Keep initial tests deterministic and cheap.
- Validate API behavior first, then backend-specific numerical correctness.
- Track each test's intention explicitly.

## Current tests

## `test_tokenizer`
Intention:
- Verify text tokenization and detokenization are stable for the supported toy vocabulary.
- Guard against regressions in punctuation handling.

Behavior checked:
- Encode/decode round-trip for simple generation prompt text.
- Unknown token fallback path remains deterministic.

## `test_pipeline`
Intention:
- Ensure first E2E text-generation pipeline path is functional.
- Ensure fallback backend selection works when TRT backend is unavailable.

Behavior checked:
- Pipeline produces one generated output object.
- Generated text contains deterministic completion phrase.
- Selected backend is not empty and currently expected to be `cpu-reference` in this environment.

## `test_model_loader`
Intention:
- Ensure built-in model metadata and checkpoint tensors are loaded from model assets.
- Guard against accidental tensor shape regressions in `weights.txt`.
- Validate native `.safetensors` checkpoint loading path.

Behavior checked:
- Built-in model reports `has_checkpoint=true`.
- All required tensor sizes match expected `(vocab, hidden, mlp)` dimensions.
- Synthetic safetensors checkpoint with direct tensor names loads through `LoadDecoderModel(...)` and produces valid checkpoint tensors.

## `test_trt_smoke`
Intention:
- Ensure the TensorRT backend can be selected in force mode and run decoder-step execution end-to-end.
- Ensure `force_trt=true` fails fast when TensorRT is unavailable.
- Ensure TensorRT path executes with model-loaded checkpoint tensors (not code-generated transition logits) when checkpoint is present.

Behavior checked:
- Runtime TRT available: backend selection is `trt` and output contains expected deterministic phrase.
- Runtime TRT unavailable: pipeline construction with `force_trt=true` throws.

## `test_hf_family_registry`
Intention:
- Validate family-registry extension path where model contributors provide only family matching + model definition loading.
- Guard priority ordering and metadata parsing behavior.

Behavior checked:
- HF metadata (`model_type`, `architectures`) is parsed and visible to matcher callbacks.
- Higher-priority family registration wins.
- Resolved kind becomes `kDecoderDefinition` and runs through shared runtime path.

## `test_qwen_family`
Intention:
- Validate built-in Qwen-style family route for shared TRT/CPU runtime infrastructure.
- Ensure no-regression fallback to raw HF path when no normalized decoder definition is present.

Behavior checked:
- Qwen HF directory with `trtf_decoder/` resolves to `kDecoderDefinition`.
- Pipeline executes via shared backend path (`cpu-reference` in test config).
- Qwen HF directory without `trtf_decoder/` resolves to `kHuggingFaceLocal`.
- Built-in model alias `QWEN3` resolves to `kDecoderDefinition` and supports `loadModel(...).generate(...)`.
- Built-in `QWEN3` alias loads bundled checkpoint tensors (`weights.txt`).

## HF parity script (`scripts/compare_hf_pipeline_vs_transformers.py`)
Intention:
- Validate that `hf-transformers` backend output in `trtf_text_generation` matches direct `transformers` generation for the same model/prompt/config.

Behavior checked:
- Exact generated text match.
- Token-level accuracy (using the same HF tokenizer) equals `1.0`.

## Next test additions
- Golden-output comparison against a tiny fixed reference model.
- Plugin correctness tests for any newly introduced missing-op plugin.
- Negative tests for malformed checkpoint tensor files (missing tensor, shape mismatch, invalid op syntax).
