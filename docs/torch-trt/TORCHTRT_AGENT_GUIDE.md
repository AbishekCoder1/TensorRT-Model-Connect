You are working in tensorrt-model-connect. Add one model end-to-end to the
  Torch-TRT pipeline so it is truly working and CI-ready, not just compiling.

  The Torch-TRT pipeline converts HuggingFace models into `.trtfb` bundles
  via `torch.export` + `torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine`,
  and runs them through the same C++ runtime as the raw TRT pipeline (DeviceKvCache + TRT C API).
  No LibTorch dependency.

  Pipeline flow:
  ```
  HF model (safetensors + config.json)
    -> AutoModelForCausalLM.from_pretrained()
    -> StatelessCacheWrapper (explicit cache I/O, scatter-patched StaticCache)
    -> torch.export.export(strict=False)
    -> convert_exported_program_to_serialized_trt_engine(use_explicit_typing=True)
    -> Raw TRT engine plan
    -> .trtfb bundle [engine + tokenizer + config, runtime_strategy=decoder_kv_cache]
    -> C++ runtime (DeviceKvCache + TRT C API, no LibTorch)
  ```

  Hard requirements:
  1) Run everything inside your own container/workspace only.
  2) Do not ask for confirmation for routine commands; execute directly.
  3) Keep changes minimal and scoped to this model.
  4) Do not relax thresholds unless repeated evidence proves it is necessary.
  5) Validate functional quality (text continuation makes sense), not only
     metric pass.
  6) Log all work in docs/torch-trt/TORCHTRT_WORKLOG.md (what was done, what failed,
     what still needs work).

  Inputs you must set before starting:
  - MODEL_NAME: <e2e case name, e.g. qwen3-0.6b>
  - HF_ID: <huggingface model id, e.g. Qwen/Qwen3-0.6B>
  - FAMILY: <family plugin name, e.g. qwen>
  - BUNDLE_NAME: <model>.trtfb
  - PROMPT: <test prompt, e.g. "The capital of France is">
  - TRUST_REMOTE_CODE: <true|false>
  - MAX_CACHE_LENGTH: <KV cache length, default 256>

  Key files:

  | File | Purpose |
  |------|---------|
  | `tensorrt_model_connect/tensorrt_model_connect/engine_defs/torch_trt/families/<family>.py` | Family plugin (you create this) |
  | `tensorrt_model_connect/tensorrt_model_connect/engine_defs/torch_trt/families/base.py` | Plugin protocol definition |
  | `tensorrt_model_connect/tensorrt_model_connect/engine_defs/torch_trt/families/qwen.py` | Reference plugin (Qwen family — decoder strategy) |
  | `tensorrt_model_connect/tensorrt_model_connect/engine_defs/torch_trt/families/bert.py` | BERT plugin (encoder_only strategy) |
  | `tensorrt_model_connect/tensorrt_model_connect/engine_defs/torch_trt/strategies/__init__.py` | Strategy registry: `get_strategy(name)` |
  | `tensorrt_model_connect/tensorrt_model_connect/engine_defs/torch_trt/strategies/base.py` | `BuildStrategy` Protocol |
  | `tensorrt_model_connect/tensorrt_model_connect/engine_defs/torch_trt/strategies/decoder.py` | `DecoderBuildStrategy` + `StatelessCacheWrapper` |
  | `tensorrt_model_connect/tensorrt_model_connect/engine_defs/torch_trt/strategies/encoder_only.py` | `EncoderOnlyBuildStrategy` + `EncoderOnlyWrapper` |
  | `tensorrt_model_connect/tensorrt_model_connect/engine_defs/torch_trt/cache_config.py` | KV cache tensors + export args (raw TRT format) |
  | `tensorrt_model_connect/tensorrt_model_connect/engine_defs/torch_trt/compiler.py` | Build orchestrator (strategy dispatch + export + TRT convert) |
  | `tensorrt_model_connect/tensorrt_model_connect/engine_defs/torch_trt/config.py` | Model config parser (uses tensorrt_model_connect.config.ModelConfig) |
  | `tensorrt_model_connect/tensorrt_model_connect/engine_defs/torch_trt/bundle_writer.py` | Bundle packaging (.trtfb) |
  | `src/runtime/core/kv_cache.cpp` | C++ auto-detects tensor naming from engine |
  | `scripts/new_torchtrt_family.py` | Auto-scaffold a plugin from HF repo |
  | `scripts/validate_torchtrt_family.sh` | One-command validation gate |
  | `tools/diff_logits.py` | Logit comparison (TRT vs HF eager) |
  | `tests/engine_defs/torch_trt/` | Unit tests |

  Execution plan (must follow in order):

  A) Environment precheck/build
  - python3 -c "import tensorrt, torch, transformers, torch_tensorrt; print('ok')"
  - pip install --no-deps -e tensorrt_model_connect/
  # Note: torch-trt backend is now part of tensorrt_model_connect (tensorrt_model_connect/engine_defs/torch_trt/)
  - Auto-detect TRT include dir:
    TRT_INC_DIR=$(find /usr/include -maxdepth 2 -name NvInferRuntime.h \
      -printf '%h' -quit 2>/dev/null || echo "/usr/include")
  - cmake -S . -B build -G Ninja \
    -DTRTMC_TRT_INCLUDE_DIR="$TRT_INC_DIR" \
    -DTRTMC_TRT_LIBRARY="${TRT_LIB_DIR:-/opt/venv/lib/python3.12/site-packages/
    tensorrt_libs}/libnvinfer.so" \
    -DTRTMC_CUDA_INCLUDE_DIR=/usr/local/cuda/include \
    -DTRTMC_CUDART_LIBRARY=/usr/local/cuda/lib64/libcudart.so
  - cmake --build build -j

  B) Implement model support
  - If existing family plugin matches the model's model_type:
    - No plugin changes needed. Skip to C.
  - If new family required:
    - Scaffold: python3 scripts/new_family.py \
        --model-type MODEL_TYPE --hf-repo HF_ID --family-name FAMILY
    - Review generated tensorrt_model_connect/tensorrt_model_connect/engine_defs/torch_trt/families/<family>.py
    - Plugin has 3 methods:
      - matches(model_type) -> bool: return True for matching model_type strings
      - load_model(model_dir, config, max_cache_length) -> nn.Module:
        load HF model with AutoModelForCausalLM.from_pretrained()
      - get_export_args(model, config, max_cache_length, precision) -> tuple:
        call make_export_args(config, max_cache_length)
        Returns tuple: (token_id, position_id, attention_mask, *cache_kv)
    - Plugin is auto-discovered (no __init__.py edits needed)

  C) Build bundle
  - trtmc-build build --backend torchtrt HF_ID -o /tmp/BUNDLE_NAME --max-cache-length MAX_CACHE_LENGTH [--precision fp16|bf16|fp32] --verbose
  - Verify bundle: trtmc-build inspect /tmp/BUNDLE_NAME
  - Expected output: JSON with sections (engine_plan, tokenizer.json, config.json, etc.)
  - Bundle uses TRTFB magic and runtime_strategy=decoder_kv_cache (build_backend=torch_trt)

  D) Diff validation (TRT vs HF eager)
  - python3 tools/diff_logits.py \
      --model HF_ID --atol 1e-2 --battery --verbose \
      [--trust-remote-code if needed]
  - Check metrics:
    - top1_match_rate >= 80% (argmax agreement)
    - mean_cosine_sim > 0.99 (logit distribution similarity)
  - If model needs trust_remote_code, add --trust-remote-code flag.

  E) C++ inference validation
  - ./build/trtmc run /tmp/BUNDLE_NAME \
      --prompt "PROMPT" --max-new-tokens 20 \
      --hf-python /opt/venv/bin/python
  - Verify output text makes sense (not garbage/repetition).
  - Same CLI as raw TRT pipeline — auto-detects IO naming from engine tensors.

  F) Full validation gate
  - ./scripts/validate_family.sh HF_ID
  - This runs: bundle build + diff_logits battery + C++ inference.
  - All must PASS.

  G) Determinism/repro sanity
  - Re-run C++ inference at least twice.
  - Confirm stable output (same text each time — model is deterministic
    with argmax decoding).
  - If output varies, investigate cache reset logic.

  H) CI-equivalent local gates
  Run all of these:
  - python -m pytest tests/engine_defs/torch_trt/ -v
  - ctest --test-dir build --output-on-failure
  - python -m pytest tests/builder/ -v --ignore=tests/builder/test_cli.py
  - python -m pytest tests/tools/ -v

  I) Output/report format (final answer must be strict)
  Provide:

  1. Summary
  - What was added/fixed for MODEL_NAME.

  2. Files changed
  - Exact file list with one-line reason each.

  3. Commands run
  - Exact commands, in execution order.

  4. Validation results
  - diff_logits metrics (top1_match_rate, cosine_sim, max_diff).
  - C++ inference output text.
  - Determinism findings from reruns.

  5. CI readiness
  - Which CI-equivalent gates passed.
  - Any remaining risk or known limitation.

  6. Artifacts
  - Absolute paths to bundle, logs, or other outputs.

  7. Worklog
  - Append entry to docs/torch-trt/TORCHTRT_WORKLOG.md with: what was done, what
    failed, decisions made, what still needs work.

  8. Commit
  - Create one clean commit with a clear message.

  Architecture notes (for debugging):

  - **Strategy dispatch**: `compiler.py` reads `plugin.runtime_strategy`
    (defaults to "decoder") and calls `get_strategy(name)` to select a
    `BuildStrategy`. The strategy handles model wrapping, export arg
    construction, and pre-export setup. Available strategies:
    - `decoder` → `DecoderBuildStrategy` (StatelessCacheWrapper, KV cache I/O)
    - `encoder_only` → `EncoderOnlyBuildStrategy` (EncoderOnlyWrapper, no cache)
  - StatelessCacheWrapper wraps the HF model with explicit KV cache I/O
    matching the raw TRT format: int32 IDs, float32 mask/cache with
    GQA-expanded heads.
  - patch_static_cache_scatter() patches StaticLayer.update to use
    functional torch.scatter instead of in-place index_copy_ for a clean
    TRT graph (no in-place mutations). Thread-safe via threading.Lock.
  - torch.export.export(strict=False) is required because the wrapper
    creates a StaticCache internally which has dynamic control flow.
  - convert_exported_program_to_serialized_trt_engine(use_explicit_typing=True)
    produces a raw TRT engine plan. use_explicit_typing=True is required
    to preserve original dtypes (without it, fp16 precision conversion
    produces all-zero logits).
  - The C++ runtime auto-detects tensor naming from engine I/O. Torch-TRT
    engines use cache_kv_N / outputN naming; standard TRT uses cache_k_N /
    present_k_N. Both use the same decoder_kv_cache strategy.
  - Cache uses num_attention_heads * head_dim (GQA-expanded), not
    num_key_value_heads * head_dim.
  - Token-by-token prefill: model is exported at seq_len=1, so both
    prefill and decode process one token at a time.
  - No LibTorch dependency at runtime — pure TRT C API.
  - **GPU cleanup**: build_bundle() uses try/finally with gc.collect() +
    torch.cuda.empty_cache() to prevent OOM in multi-bundle builds.
  - **Adding a new strategy**: Create `strategies/<name>.py` implementing
    `BuildStrategy`, register in `strategies/__init__.py._init_registry()`,
    set `runtime_strategy = "<name>"` on the family plugin.

  Known warnings (harmless, suppressed in compiler.py):
  - "TRTLLM_PLUGIN_PATH is not set" — we don't use TRT-LLM plugins
  - "transformers version X not tested with nvidia-modelopt" — version
    check not updated for transformers 5.x
  - "The logger passed into createInferBuilder differs" — upstream
    torch_tensorrt bug, TRT correctly uses first logger
  - "tensorrt.plugin module is experimental" — TRT C++ library, cannot
    suppress from Python

  Stop only when all required steps are complete and the model is CI-ready.
