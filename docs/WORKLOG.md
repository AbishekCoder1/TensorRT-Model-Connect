# Worklog

## 2026-02-09
- Created new repository scaffold at `/home/yifeif/repos/trt-transformers-cpp`.
- Chosen strategy: API-first TensorRT implementation, avoid default dependence on ONNX parser.
- Verified host/system TensorRT artifacts and API surface (including newer attention/rotary/KV APIs in local TRT build headers).
- Noted environment mismatch: early sandbox checks failed CUDA setup even though host GPU is available.
- Implemented M0 codebase with CPU-reference backend for deterministic runnable E2E while maintaining TensorRT backend scaffolding.
- Added persistent planning docs, model coverage analysis, and test plan with intentions.
- Added Docker assets to run with host GPU where full TensorRT path can be validated.
- Ran M0 native validation:
  - Build succeeded
  - `ctest` passed (`test_tokenizer`, `test_pipeline`)
  - E2E example produced expected text-generation output with backend `cpu-reference`.
- Improved CMake dependency detection:
  - Fixed TensorRT path search behavior.
  - Added CUDA dev-header/runtime checks before enabling TRT backend compilation.
- Validated Docker flow:
  - Fixed CUDA base image tag to a valid one (`nvidia/cuda:12.6.3-devel-ubuntu22.04`).
  - Built `trtf-dev` image successfully.
  - Verified GPU visibility with in-container `nvidia-smi`.
- Verified in-container configure/build/test/example path with mounted TRT artifacts.
- Added a minimal real TensorRT execution path:
  - `trt` backend now builds tiny constant-output TensorRT graphs and executes them during generation.
  - Added `force_trt` pipeline mode to fail fast instead of falling back.
  - Added runtime CUDA-availability gating so default mode still falls back cleanly when TRT is compiled but not runnable.
  - Added `test_trt_smoke` to validate forced TRT behavior (or expected failure when TRT is unavailable).
- Started M1 decoder path implementation:
  - Replaced constant-output TRT path with a true decoder-step graph using TensorRT API (`embedding -> attention -> MLP -> logits`).
  - Added host-side iterative decode loop with fixed-size KV-style cache updates and prefill handling.
  - Kept deterministic toy-token behavior so current text-generation tests remain stable.
- Validated real TRT execution in GPU container:
  - Updated container runtime library path setup so TensorRT builder/runtime resources load correctly.
  - Built with `/opt/trt/Debug/lib/libnvinfer.so` and verified `./build-gpu-debug/trtf_text_generation --force-trt` runs with `backend=trt`.
- Completed model-driven M1 wiring:
  - Added `DecoderModel` loader (`config.json`, `vocab.txt`, `transitions.txt`) and built-in model assets at `models/tiny-cake-v1`.
  - Switched tokenizer construction to vocab-from-model and converted CPU/TRT backends to consume model transitions/default token.
  - Threaded model cache length into TRT decoder engine/input validation instead of fixed compile-time cache length.
  - Updated CLI/tests/docs to use explicit built-in model id `trtf/tiny-cake-v1` (or a local model directory path).
- Replaced code-generated transition logits in TRT path with loaded checkpoint tensors:
  - Added `weights.txt` checkpoint parser in model loader (tensor blocks + simple ops).
  - Added `weights_file` support in `config.json` and built-in `models/tiny-cake-v1/weights.txt`.
  - Updated TRT backend to consume checkpoint tensors when available, with compatibility fallback for transition-only model dirs.
  - Added `test_model_loader` to validate checkpoint presence and tensor shapes.
  - Revalidated host tests and GPU-container force-TRT E2E with checkpoint-backed generation.
- Added raw Hugging Face checkpoint run path and parity validation:
  - Downloaded `hf-internal-testing/tiny-random-gpt2` with `model.safetensors` into `models/hf/`.
  - Added `hf-transformers` backend in pipeline for model dirs containing `config.json` + `model.safetensors`.
  - Added Python runner script `scripts/hf_generate.py` and parity script `scripts/compare_hf_pipeline_vs_transformers.py`.
  - Installed Python deps in GPU container (`torch`, `transformers`, `safetensors`, etc.) and validated exact output match against direct transformers generation.
- Decoupled pipeline orchestration from model/backend specifics:
  - Added model resolver seam (`include/trtf/model_resolver.h`, `src/model/model_resolver.cpp`).
  - Added runtime assembly seam (`include/trtf/runtime_factory.h`, `src/runtime/runtime_factory.cpp`).
  - Simplified `Pipeline::CreateTextGeneration` to orchestrate only through resolver + runtime factory.
  - Added custom extension hooks (`RegisterTextGenerationModelResolver`, `RegisterTextGenerationRuntimeAssembler`) for out-of-tree model support.
  - Added seam-level tests (`tests/test_model_resolver.cpp`, `tests/test_runtime_factory.cpp`) and custom extension E2E test (`tests/test_extension_registry.cpp`) to keep extension points stable.

## 2026-02-12
- Clarified direction: "distributed model implementation" means distributed developer ownership for new model families, not distributed runtime inference.
- Recorded target architecture for TRT backend onboarding so model-specific contributors implement only model definition, while TRT and generation internals stay shared.

Planned architecture (authoritative for follow-on agents):
- Add a strict model-definition contract for decoder-only families:
  - family matcher over Hugging Face metadata (`config.json`-derived).
  - family-specific loader that returns normalized `DecoderModel` definition/tensor mapping data.
  - no family-owned TRT builder code and no family-owned decode loop code.
- Keep runtime internals centralized:
  - shared TRT build path (graph construction, engine build/cache lifecycle).
  - shared generation runtime (prefill/autoregressive loop, KV cache management, stop rules/sampling policy).
- Keep user-facing pipeline API stable:
  - `Pipeline::CreateTextGeneration(model_id, ...)` remains orchestration-only.
  - `load("QWEN3")` style IDs should resolve through family registry to normalized model definition, then run through shared TRT runtime.
- Backward compatibility / migration:
  - existing resolver/runtime extension seams remain available.
  - family-definition registry path is the preferred onboarding route for new model families.
- Initial scope limit:
  - TRT backend only for compiled inference path.
  - existing `hf-transformers` backend can remain as fallback for raw local HF dirs with no family definition.

Implementation progress started:
- Added HF family registration API to focus on model-definition loaders:
  - `HfModelFamilyRegistration` now uses `matcher + model_definition_loader`.
- Implemented `src/model/hf_family_registry.cpp`:
  - local HF-dir detection (`config.json` + `model.safetensors`).
  - metadata extraction (`model_type`, `architectures`).
  - priority-based family resolution to `ResolvedModelKind::kDecoderDefinition`.
- Wired resolver path:
  - `ResolveTextGenerationModel(...)` now consults HF family registry before default raw-HF fallback.
- Added test coverage:
  - `tests/test_hf_family_registry.cpp` validates:
    - metadata parsing into matcher input,
    - registration priority behavior,
    - resolution to decoder definition,
    - execution through shared pipeline/runtime path (`cpu-reference` in test).
- Build integration:
  - Added `src/model/hf_family_registry.cpp` and `test_hf_family_registry` target to `CMakeLists.txt`.
- Added first built-in real family path (`qwen-style`) through shared infrastructure:
  - `RegisterBuiltinHfModelFamilies()` now registers `qwen-decoder-definition`.
  - Match rule: HF local model with `model_type` prefix `qwen`/`qwq` and normalized definition files under `<model_dir>/trtf_decoder/`.
  - Loader rule: family loader calls shared `LoadDecoderModel(...)` on `trtf_decoder/` and returns `kDecoderDefinition`.
  - Fallback preserved: Qwen HF dirs without `trtf_decoder/` continue through raw `kHuggingFaceLocal` path.
- Added test coverage for built-in Qwen family:
  - `tests/test_qwen_family.cpp` validates match + load + shared runtime execution and no-regression fallback behavior.
- Added end-to-end built-in QWEN3 alias path:
  - `ResolveHfModelViaFamilyRegistry(...)` now maps model id aliases (`QWEN3`) to bundled HF-style assets at `models/hf/qwen3`.
  - Added bundled QWEN3 model assets with normalized decoder-definition files under `models/hf/qwen3/trtf_decoder`.
  - Added ergonomic API wrapper path `trtf::loadModel(...).generate(...)` on top of `Pipeline` for direct model-style usage.
- Hardened QWEN3 bundled model assets:
  - Added `models/hf/qwen3/trtf_decoder/weights.txt` and wired `weights_file` in decoder config so TRT path uses checkpoint tensors.
- Added direct model-style demo executable:
  - `examples/load_model.cpp` (`trtf_load_model`) demonstrates `auto model = trtf::loadModel(\"QWEN3\", ...); std::string out = model.generate(\"Hello\");`.
- Validated QWEN3 E2E generation:
  - Host non-GPU path: `./build/trtf_text_generation QWEN3 "Hello"` returns `backend=cpu-reference` with output `hello from qwen3.`.
  - GPU container TRT path: built in `build-gpu-qwen3` and ran `./build-gpu-qwen3/trtf_text_generation --force-trt QWEN3 "Hello"` with result `backend=trt` and output `hello from qwen3.`.
- Started upstream Qwen3 safetensors bridge implementation:
  - Added native C++ safetensors reader (`src/model/safetensors_loader.cpp`) with F32/F16/BF16 decode support.
  - Extended model loader to accept `.safetensors` in `weights_file` and to auto-build placeholder vocab/transitions when checkpoint-backed models omit `vocab.txt`/`transitions.txt`.
  - Added initial Qwen bridge mapping (`architecture_family=qwen3`) from upstream tensor keys (`model.layers.0.*`, `model.embed_tokens.weight`, `lm_head.weight`) into shared decoder checkpoint tensors.
  - Added prep script `scripts/prepare_qwen3_trtf_decoder.py` to scaffold `trtf_decoder/` for local upstream Qwen3 model dirs.
  - Added safetensors coverage in `tests/test_model_loader.cpp` by generating a synthetic safetensors checkpoint and loading it through `LoadDecoderModel`.
- Extended Qwen family resolver for direct upstream root loading:
  - Qwen family now accepts either `trtf_decoder/` assets or direct HF root `config.json + model.safetensors`.
  - Model loader auto-detects root `model.safetensors` when `weights_file` is not explicitly set.
  - `tests/test_qwen_family.cpp` now validates qwen-root safetensors bridge resolution (`kDecoderDefinition` + checkpoint loaded).

Next implementation steps:
- Add built-in family definitions for real TRT-target families (starting with one concrete family).
- Move/expand normalized tensor mapping so families can load from safetensors into shared TRT tensors without family-owned TRT code.
- Refactor TRT backend internals into clearer shared subsystems (definition -> weights -> engine -> runtime loop) for easier multi-family reuse.

Further implementation completed (Qwen full-stack iteration):
- Upgraded Qwen safetensors mapping from layer-0 bridge to full multi-layer loader path:
  - `src/model/model_loader.cpp` now loads all `model.layers.{i}` Qwen tensors (`input/post norms`, `q/k/v/o`, `gate/up/down`) plus `model.norm.weight`.
  - Populates `DecoderCheckpoint.has_qwen_layers`, `qwen_layers`, and `final_norm`.
  - Preserves layer-0 compatibility tensors for legacy paths.
- Reworked TRT backend implementation for shared Qwen-family execution:
  - Added new backend implementation file `src/runtime/trt_backend_qwen.cpp`.
  - CMake now builds this file instead of legacy `src/runtime/trt_backend.cpp`.
  - Added shared multi-layer decode-step graph path with per-layer cache I/O tensors:
    - RMSNorm (pre-attn and post-attn),
    - scaled attention with KV cache,
    - SwiGLU MLP,
    - final RMSNorm and lm_head projection.
  - Added RoPE application in TRT graph with `position_id` input and precomputed sin/cos tables.
  - Generalized runtime loop and enqueue bindings to support N-layer cache inputs/outputs while keeping legacy one-layer tiny-cake path working.
- Upgraded Qwen family synthetic test fixtures:
  - `tests/test_qwen_family.cpp` now writes a 2-layer upstream-style Qwen safetensors checkpoint including all required keys.
  - Added assertions for `has_qwen_layers`, expected layer count, and `final_norm`.
- Updated onboarding script/docs:
  - `scripts/prepare_qwen3_trtf_decoder.py` mapping mode updated to `qwen3-full-stack-v2`.
  - `README.md` updated to describe full Qwen layer-stack safetensors mapping and shared TRT runtime path.

Validation run results after full-stack iteration:
- Host:
  - `cmake --build build -j` passed.
  - `ctest --test-dir build --output-on-failure` passed (`9/9`).
  - `./build/trtf_load_model QWEN3 Hello` => `backend=cpu-reference`, output `hello from qwen3.`.
  - `./build/trtf_load_model --force-trt QWEN3 Hello` fails on host as expected when TRT runtime is unavailable in host session.
- GPU container (`trtf-dev`):
  - `cmake --build build-container -j` passed.
  - `ctest --test-dir build-container --output-on-failure` passed (`9/9`).
  - `./build-container/trtf_load_model --force-trt QWEN3 Hello` => `backend=trt`, output `hello from qwen3.`.

Refactor + validation follow-up (model-definition ownership + MMLU):
- Refactored TRT model-definition ownership out of runtime and into `src/model`:
  - Added `src/model/trt_model_definition.h` and `src/model/trt_model_definition.cpp` with `BuildTrtDecoderWeights(...)`.
  - Added Qwen-family-specific definition module `src/model/qwen3_trt_model_definition.h` + `src/model/qwen3_trt_model_definition.cpp`.
  - `src/runtime/trt_backend_qwen.cpp` now consumes normalized model definitions from `src/model` and no longer owns checkpoint-to-runtime mapping logic.
  - `CMakeLists.txt` updated to compile the new model-definition translation units and add private `src/` include path.
- Added MMLU evaluator:
  - `scripts/eval_mmlu.py` supports:
    - `--backend transformers` (reference quality path for upstream checkpoints),
    - `--backend trtf` (direct binary-driven validation path).
  - Reports `accuracy`, `answered`, and enforces `--min-accuracy`.
- Qwen3 MMLU validation run:
  - Model: `Qwen/Qwen3-0.6B`
  - Dataset: `cais/mmlu`, `subject=all`, `split=test`, sampled `64` questions
  - Result: `accuracy=0.3906` (`25/64`), `status=PASS` against `min_accuracy=0.35`
  - Date: 2026-02-13

## 2026-02-13 (continued: upstream Qwen3 parity iteration)
- Implemented direct sharded safetensors loading in decoder-definition path:
  - `src/model/model_loader.cpp` now detects and loads `model.safetensors.index.json`.
  - Added weight-map routing across shard files through a shared tensor source abstraction.
  - Removed previous hard failure for sharded checkpoints in `LoadDecoderModel(...)`.
- Extended HF/Qwen model recognition for sharded roots:
  - `src/model/hf_family_registry.cpp` and `src/model/model_resolver.cpp` now treat either
    `model.safetensors` or `model.safetensors.index.json` as a valid HF local model root.
- Upgraded Qwen checkpoint mapping with upstream q/k norm tensors:
  - `DecoderLayerCheckpoint` now carries `q_norm` / `k_norm`.
  - Loader maps `model.layers.{i}.self_attn.{q_norm,k_norm}.weight` and expands them to hidden-size layout.
  - TRT model definition path validates and forwards these tensors.
- Upgraded shared TRT Qwen runtime math and decode bookkeeping:
  - Added per-head RMSNorm helper and applied q/k norm before RoPE in each Qwen layer.
  - Fixed multi-layer cache length advancement bug (cache length now advances once per token, not per layer/tensor write).
- Added tokenizer parity bridge for decoder-definition TRT path:
  - Added `CreateHfPythonTokenizer(...)` in `src/tokenizer/hf_python_tokenizer.cpp`.
  - Added helper script `scripts/hf_tokenizer.py`.
  - `BuildRuntimeForTextGeneration(...)` now prefers HF tokenizer when model metadata indicates HF tokenizer assets.
- Added token-id and cache-config metadata handling:
  - `DecoderArchitectureConfig` now includes `bos_token_id`, `eos_token_id`, `pad_token_id`.
  - Loader parses token ids from config (int or first array element).
  - Added optional env override `TRTF_MAX_CACHE_LENGTH`.
  - Added default practical cap for Qwen root checkpoints when `max_cache_length` is not explicitly provided.
- Test coverage updates:
  - `tests/test_model_loader.cpp` now validates sharded safetensors load path.
  - `tests/test_qwen_family.cpp` fixture now includes `q_norm` / `k_norm` tensors.
- Build/test validation:
  - `cmake --build build -j` passed.
  - `ctest --test-dir build --output-on-failure` passed (`9/9`).
  - `./build/trtf_load_model QWEN3 Hello` still works (`backend=cpu-reference`, output `hello from qwen3.`).

Remaining gap note:
- Built-in `QWEN3` alias remains bundled demo assets for plumbing checks.
- Real upstream production parity is improved but not yet complete end-to-end (notably beyond host CPU fallback this turn and with remaining TRT math/runtime fidelity work for large-scale upstream checkpoints).
- GPU container validation after this iteration:
  - Ran full container configure/build/test + force-TRT smoke in `trtf-dev`.
  - Result: build succeeded, `ctest` passed (`9/9`), and `./build-container/trtf_load_model --force-trt QWEN3 Hello` returned `backend=trt` with `hello from qwen3.`.

## 2026-02-13 (continued: real upstream Qwen3 TRT parity root-cause fix)

Authoritative plan update (what was needed to reach real upstream Qwen3 TRT parity):
1. Prove whether mismatch is model-definition mapping or TRT runtime math.
2. Add direct diff instrumentation (HF vs TRT logits) for first-step next-token decisions.
3. Fix deterministic correctness blockers in shared infrastructure before model-specific changes.
4. Re-run real upstream Qwen3 E2E generation in TRT container and confirm token-level parity.
5. Run a post-fix MMLU sanity pass on TRT backend; then scale to larger eval runs.

What was implemented this iteration:
- Root-cause 1 (major): HF tokenizer bridge output contamination.
  - Problem: `src/tokenizer/hf_python_tokenizer.cpp` merged stderr/stdout (`2>&1`), and Transformers startup warnings were captured with tokenizer results.
  - Impact: `encode()` could return empty token IDs, causing TRT generation to start from BOS instead of the prompt; decoded output also contained warning lines.
  - Fix:
    - Added output sanitization helpers in `src/tokenizer/hf_python_tokenizer.cpp` to strip known warning lines and blank lines.
    - Hardened `parse_int_list(...)` to parse numeric tokens robustly instead of failing on first non-numeric token.
    - Applied sanitized output path to `encode`, `decode`, `id_for_token`, and `token_for_id`.

- Root-cause 2 (numerical stability/lifetime): short-lived TRT constant buffers in Qwen path.
  - Problem: per-layer ephemeral constants (`eps`, attention scale) were created from local vectors inside helper/layer functions in `src/runtime/trt_backend_qwen.cpp`.
  - Risk: pointer lifetime could end before network build completion.
  - Fix:
    - Refactored Qwen TRT graph path to create shared `eps` and attention-scale constant tensors in `create_decoder_step_engine_qwen(...)` and pass them through helper calls.
    - Updated `add_rms_norm(...)`, `add_rms_norm_per_head(...)`, and `add_qwen_layer_block(...)` signatures/callers to consume shared tensors.

- Added targeted diff instrumentation for TRT logits:
  - `TRTF_DEBUG_LOGITS_TOPK=<k>` (env) in `src/runtime/trt_backend_qwen.cpp` prints per-step top-k `token_id:logit` for direct HF-vs-TRT comparison.
  - Kept `TRTF_DEBUG_MASK` env-gated mask dump hook for attention-mask verification.

Validation outcomes (real upstream assets, TRT backend):
- Before tokenizer-sanitization fix:
  - TRT top logits for prompt `Hello` were wrong (`14582:12.2307 ...`) and output was `Question`.
- After fixes:
  - TRT top-5 for prompt `Hello`:
    - `21806:8.13904`, `14582:8.07674`, `15846:7.63189`, `477:7.57898`, `1957:7.34415`
  - HF top-5 reference (same prompt/model) matched numerically at same ranking.
  - `./build-container-qwen2/trtf_load_model --force-trt QWEN3 Hello` now yields:
    - `backend=trt`
    - `Hello Answer`
- Longer generation sanity check:
  - Prompt: `The capital of France is`
  - Output example: `The capital of France is Paris. The capital of Italy is Rome. ...`

MMLU TRT sanity check (post-fix):
- Command used backend `trtf` with real `QWEN3` and forced TRT in container.
- Sampled run: `num-samples=4` (small sanity pass due per-sample startup cost in current evaluator path).
- Result:
  - `answered=4`, `correct=2`, `accuracy=0.5000`, `status=PASS`.

Current status:
- Real upstream Qwen3 TRT E2E generation is functioning with corrected tokenizer + runtime math path.
- Built-in `QWEN3` alias now effectively exercises real upstream path when local real assets are present.

Next steps for future agents:
1. Add regression tests for tokenizer warning-contamination behavior in HF tokenizer bridge.
2. Keep `TRTF_DEBUG_LOGITS_TOPK` as gated debug tooling, and consider removing/limiting `TRTF_DEBUG_MASK` once no longer needed.
3. Improve `scripts/eval_mmlu.py` TRT mode to avoid one-process-per-question startup (persistent runner), then run a larger official sample (for example 64).
4. Re-record full MMLU TRT metric after persistent-eval optimization.

## 2026-02-13 (Phase 1 cleanup kickoff: model/runtime boundary + TRT reuse)

Comprehensive cleanup plan (authoritative):
1. Baseline audit and dependency map (files, build graph, tests, docs, generated artifacts).
2. Refactor TRT runtime into shared core + TRT utility modules; eliminate dead legacy backend file.
3. Move model-family specific TRT graph code behind model-owned builders under `src/model`.
4. Split monolithic model loader into focused loaders/parsers and deduplicate resolver/registry helpers.
5. Harden test layout: unit tests + deterministic HF layer/op diff gate + optional MMLU benchmark gate.
6. Clean repo artifacts/docs/ignore rules and align README with production-parity Qwen3 path.
7. Run full validation matrix (host + container TRT + Qwen3 real-model checks) and update worklog.

Repo scrub findings captured for cleanup decisions:
- `src/runtime/trt_backend.cpp` was dead (not compiled by `CMakeLists.txt`) and duplicated TRT backend symbols.
- `src/runtime/trt_backend_qwen.cpp` remained monolithic (builder utils + graph construction + decode runtime).
- `src/model/model_loader.cpp` was multi-responsibility and needs extraction in later phases.
- Qwen HF diff tooling existed (`scripts/qwen_layer_diff.py`) but was not yet a ctest gate.

Phase 1 implementation completed in this iteration:
- Removed dead legacy TRT file:
  - deleted `src/runtime/trt_backend.cpp`.
- Started shared TRT utility extraction:
  - added `src/utils/trt/engine_cache.h`.
  - added `src/utils/trt/engine_cache.cpp`.
  - added utility source to build target in `CMakeLists.txt`.
- Implemented engine-plan reuse to avoid repeated TRT rebuilds across process invocations:
  - `src/runtime/trt_backend_qwen.cpp` now uses cache key + load/store hooks in
    `finalize_decoder_step_engine(...)`.
  - First run builds serialized engine plan and persists it.
  - Subsequent runs deserialize cached plan directly when cache key matches model/runtime definition.

Engine cache behavior notes:
- Cache key includes model-definition scalars and tensor contents (including Qwen per-layer tensors), plus runtime flags.
- Cache location defaults to:
  - `$TRTF_TRT_ENGINE_CACHE_DIR` when set, else
  - `$HOME/.cache/trtf/trt_engine_plans`, else
  - `/tmp/trtf/trt_engine_plans`.
- Cache can be disabled with `TRTF_DISABLE_ENGINE_CACHE=1`.

Validation policy for this branch (requested by user):
- After every code change set, run all unit tests and E2E tests (host + TRT container path).
- Log exact commands/results in worklog for future agents.

## 2026-02-13 (continued: TRT logger passthrough + E2E stdout diagnostics)

Implemented to support direct stdout-based debugging:
- Added TRT logger passthrough in `src/runtime/trt_backend_qwen.cpp`:
  - `TRTF_TRT_LOG_STDERR=1` enables forwarding TensorRT `ILogger` lines to stderr/stdout stream.
  - `TRTF_TRT_LOG_MIN_SEVERITY=<INTERNAL_ERROR|ERROR|WARNING|INFO|VERBOSE>` controls verbosity (default: `INFO`).
- Added new reproducible E2E diagnostics script:
  - `scripts/test_qwen3_trt_e2e.sh`
  - Runs configure/build/ctest, then two forced-TRT `QWEN3` runs with timing and log capture.
  - Writes logs to `/tmp/trtf_qwen3_trt_e2e.log` by default.

Validation run summary:
- Host:
  - `cmake --build build -j` passed.
  - `ctest --test-dir build --output-on-failure` passed (`9/9`).
  - `./build/trtf_load_model trtf/tiny-cake-v1 Hello` passed (`backend=cpu-reference`).
- Container via `scripts/test_qwen3_trt_e2e.sh`:
  - configure/build/ctest passed (`9/9`).
  - Run 1 (`--force-trt QWEN3 Hello`) output:
    - `backend=trt`
    - `Hello Answer`
    - timing ~ `1m51s`
    - TRT logger includes explicit build markers such as `Engine generation completed ...`.
  - Run 2 (`--force-trt QWEN3 Hello`) output:
    - `backend=trt`
    - `Hello Answer`
    - timing ~ `4m01s`
    - TRT logger includes `Loaded engine size ...` without repeating full build marker sequence in sampled tail.

Notes:
- This script/logging path now makes startup behavior inspectable entirely from stdout/stderr, per request.
- TRT compilation warnings from TensorRT headers remain visible during build and are expected in this environment.
