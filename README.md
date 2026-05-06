# TensorRT-Model-Connect

Python builds TensorRT engines from HuggingFace models. C++ runs them.

Two pipelines are available:

| Pipeline | Builder | Bundle | Runtime | How it works |
|----------|---------|--------|---------|--------------|
| **Raw TRT** | `trtmc-build` (Python) | `.trtfb` | C++ (TensorRT C API) | Builds a TRT engine from scratch using the TensorRT network API with hand-written graph ops. Fastest inference, most control. |
| **Torch-TRT** | `trtmc-build --torch-trt` (Python) | `.trtfb` | C++ (TensorRT C API) | Uses `torch.export` + `torch_tensorrt` to compile an HF model into a raw TRT engine. No manual graph construction needed — just a family plugin that loads the model. |

Both pipelines produce `.trtfb` bundles that run on the same C++ runtime with the same `./build/trtmc` CLI. The bundle extension is retained for compatibility with existing engine artifacts. No LibTorch dependency.

The live C++ runtime has one composition path: `trtmc_c.cpp -> strategy builder -> PipelineServices -> PipelineRouter -> ports/adapters -> TRT executors`. Core services operate on in-memory request/result DTOs; file and artifact IO stays at the router and adapter edge.

## Quick start

Prerequisites: Docker + NVIDIA Container Toolkit.

```bash
# 1. Build and launch the dev container
./scripts/docker_build_gb300.sh
./scripts/docker_run_gb300.sh

# 2. Inside the container: install packages + build C++ runtime
./scripts/setup_container.sh
```

### Raw TRT pipeline (Qwen3-0.6B)

```bash
# Build a bundle
trtmc-build build Qwen/Qwen3-0.6B -o /tmp/qwen3.trtfb

# Run inference
./build/trtmc run /tmp/qwen3.trtfb \
  --prompt "The capital of France is" \
  --max-new-tokens 20 \
  --hf-python /opt/venv/bin/python
```

### Torch-TRT pipeline (Qwen3-0.6B)

```bash
# Build a bundle
trtmc-build build --torch-trt Qwen/Qwen3-0.6B -o /tmp/qwen3.trtfb --max-cache-length 256

# Run inference (same CLI — auto-detects bundle type)
./build/trtmc run /tmp/qwen3.trtfb \
  --prompt "The capital of France is" \
  --max-new-tokens 20 \
  --hf-python /opt/venv/bin/python
# Output: " Paris. The capital of Italy is Rome. The capital of Spain is Madrid. The capital of China"
```

---

## Raw TRT pipeline

The raw TRT pipeline builds engines from scratch using the TensorRT network API with hand-written graph ops. It supports all modalities (text, vision, audio, diffusion, segmentation, etc.) and is the primary pipeline for production use.

## Python API

```python
import tensorrt_model_connect

# From a HuggingFace repo ID (auto-downloads):
tensorrt_model_connect.build("Qwen/Qwen3-0.6B", "qwen3.trtfb")

# From a local directory:
tensorrt_model_connect.build("models/hf/Qwen__Qwen3-0.6B", "qwen3.trtfb")

# With options:
tensorrt_model_connect.build("Qwen/Qwen3-0.6B", "qwen3.trtfb",
                  max_cache_length=512, verbose=True)
```

Install the package:
```bash
pip install --no-deps -e tensorrt_model_connect/
```

All commands in this repo are expected to run inside the dev container image.

## CLI

### Python builder (`trtmc-build`)

```bash
# Build from HF repo ID (auto-downloads)
trtmc-build build Qwen/Qwen3-0.6B -o qwen3.trtfb

# Build from local directory
trtmc-build build models/hf/Qwen__Qwen3-0.6B -o qwen3.trtfb

# Options
trtmc-build build Qwen/Qwen3-0.6B -o qwen3.trtfb --max-cache-length 512 --verbose

# Inspect a bundle
trtmc-build inspect qwen3.trtfb

# Version info
trtmc-build version
```

### C++ runtime (`trtmc`)

```bash
# Text generation (and VLM generation with --image when supported)
./build/trtmc run qwen3.trtfb --prompt "Hello" --max-new-tokens 50 \
  --hf-python /opt/venv/bin/python

# Encoder-only hidden states
./build/trtmc encode qwen3.trtfb --prompt "Hello" --hf-python /opt/venv/bin/python

# Segmentation / prompted segmentation / detection
./build/trtmc segment segformer.trtfb --image input.png --output mask.png
./build/trtmc segment-sam sam.trtfb --image input.png --output sam_masks/
./build/trtmc detect yolox.trtfb --image input.png --output detections.json --threshold 0.5

# Embedding / reranking
./build/trtmc embed bert.trtfb --prompt "Hello"
./build/trtmc rerank personaplex.trtfb --prompt "query" --document "candidate passage"

# Audio + speech
./build/trtmc transcribe whisper.trtfb --audio sample.wav
./build/trtmc generate-audio bark.trtfb --prompt "A calm narration" --output out.wav
./build/trtmc speak qwen3_omni.trtfb --audio-in in.wav --audio-out out.wav

# Neural operators (DeepONet / FNO)
./build/trtmc solve deeponet.trtfb --branch-input "0.1,0.2" --trunk-input "0.5,0.5"
./build/trtmc solve fno.trtfb --field-input "0.1,0.2,0.3,0.4"

# Diffusion video
./build/trtmc generate-video wan_t2v.trtfb --prompt "A cat riding a bike" --output frames/

# Inspect bundle metadata
./build/trtmc inspect qwen3.trtfb

# Version info
./build/trtmc version
```

## C API

The public C++ API is a single header (`include/trtmc/pipeline.h`):

```cpp
#include <trtmc/pipeline.h>

// Create from .trtfb bundle
TrtmcPipelineOptions opts = {.max_new_tokens = 50, .hf_python = "/path/to/python"};
auto* p = trtmc_create_pipeline_ex("model.trtfb", &opts);

p->generate("Hello", 50);    // text generation
p->generate("Describe image", "/tmp/input.png", 50);  // if supports_vision()
p->segment("/tmp/input.png", "/tmp/mask.png");        // if supports_segmentation()
p->detect("/tmp/input.png", "/tmp/detections.json");  // if supports_detection()
p->transcribe("/tmp/audio.wav");                      // if supports_transcription()
p->generate_audio("Hello", "/tmp/out.wav");           // if supports_audio()
int embed_dim = 0;
p->embed("Hello", &embed_dim);                        // if supports_embedding()
p->rerank("query", "document");                       // if supports_reranking()
p->model_id();               // returns const char*
p->backend_name();           // returns const char*
delete p;

// Utilities
trtmc_last_error();           // thread-local error message
trtmc_version();
trtmc_has_trt();              // 1 if compiled with TRT support
```

## Torch-TRT pipeline

The Torch-TRT pipeline compiles HF models via `torch.export` and `torch_tensorrt`, producing `.trtfb` bundles with raw TRT engines. It requires no hand-written graph ops — just a family plugin that loads the model. Bundles run on the same C++ runtime as the raw TRT pipeline (no LibTorch dependency).

### How it works

1. **Load** — Downloads the HF model and wraps it with `StatelessCacheWrapper`, which adapts the model's I/O to match the raw TRT format: int32 token/position IDs, float32 attention mask, float32 GQA-expanded cache tensors.
2. **Export** — `torch.export.export(strict=False)` traces the wrapped model into an `ExportedProgram`.
3. **Compile** — `torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine(use_explicit_typing=True)` converts the exported graph into a raw TRT engine plan.
4. **Bundle** — The engine plan, tokenizer, and config are packaged into a `.trtfb` bundle with `runtime_strategy=torchtrt_decoder`.
5. **Run** — The C++ runtime loads the bundle, deserializes the TRT engine, and runs autoregressive inference via `DeviceKvCache` — the same backend used by the raw TRT pipeline.

### Python API

```python
import tensorrt_model_connect

# From a HuggingFace repo ID (auto-downloads):
tensorrt_model_connect.build("Qwen/Qwen3-0.6B", "qwen3.trtfb")

# With options:
tensorrt_model_connect.build("Qwen/Qwen3-0.6B", "qwen3.trtfb",
                  max_cache_length=512, verbose=True)
```

Install the package:
```bash
pip install --no-deps -e tensorrt_model_connect/
```

### CLI (`trtmc-build --torch-trt`)

```bash
# Build from HF repo ID (default: fp16)
trtmc-build build --torch-trt Qwen/Qwen3-0.6B -o qwen3.trtfb --max-cache-length 256

# Build with fp32 precision
trtmc-build build --torch-trt Qwen/Qwen3-0.6B -o qwen3_fp32.trtfb --precision fp32

# Build a smaller model
trtmc-build build --torch-trt Qwen/Qwen2.5-0.5B -o qwen2.5-0.5b.trtfb --max-cache-length 256

# Inspect a bundle
trtmc-build inspect qwen3.trtfb

# Version info
trtmc-build version
```

### Validation tools

**Accuracy validation** — compares Torch-TRT engine logits against HF eager reference:

```bash
# Battery test (multiple prompts, detailed metrics)
python3 tools/diff_torchtrt.py \
  --model Qwen/Qwen2.5-0.5B --atol 1e-2 --battery --verbose

# Single prompt
python3 tools/diff_torchtrt.py \
  --model Qwen/Qwen3-0.6B --prompt "The capital of France is" --atol 1e-2

# Key metrics to check:
#   top1_match_rate >= 80%   (argmax agreement between TRT and HF)
#   mean_cosine_sim > 0.99   (logit distribution similarity)
```

**Performance comparison** — measures inference speed across backends:

```bash
# Torch-TRT only (build + benchmark, default fp16)
python3 tools/perf_compare_torchtrt.py \
  --model Qwen/Qwen2.5-0.5B --max-new-tokens 50 --json results.json

# Force fp32 precision (all backends except Raw TRT which is always fp32)
python3 tools/perf_compare_torchtrt.py \
  --model Qwen/Qwen2.5-0.5B --precision fp32 --max-new-tokens 50

# All 4 backends: Torch-TRT vs Raw TRT vs torch.compile vs HF eager
python3 tools/perf_compare_all.py \
  --model Qwen/Qwen2.5-0.5B \
  --prompt "The largest ocean on Earth is" \
  --max-new-tokens 50 --json results.json

# Compare fp16 vs fp32 for Torch-TRT (run twice, save JSON, compare)
python3 tools/perf_compare_torchtrt.py \
  --model Qwen/Qwen3-0.6B --precision fp16 --json fp16.json
python3 tools/perf_compare_torchtrt.py \
  --model Qwen/Qwen3-0.6B --precision fp32 --json fp32.json

# With pre-built bundles (skips engine build, faster iteration)
python3 tools/perf_compare_all.py \
  --model Qwen/Qwen3-0.6B \
  --torchtrt-bundle /tmp/qwen3_torchtrt.trtfb \
  --rawtrt-bundle /tmp/qwen3_raw.trtfb \
  --prompt "The capital of France is" \
  --max-new-tokens 50

# Skip specific backends
python3 tools/perf_compare_all.py \
  --model Qwen/Qwen3-0.6B --skip-compile --skip-rawtrt

# Output includes: tokens/sec, per-token latency, total latency, per-backend dtype
```

The `--precision` flag controls Torch-TRT, HF eager, and torch.compile. Raw TRT always uses fp32 (hardcoded in graph ops). The report header shows the dtype used by each backend.

## Supported models

Model support is plugin-driven and auto-discovered from `tensorrt_model_connect/tensorrt_model_connect/families/` at build time. Any HF model whose `config.json` `model_type` matches a plugin is supported without C++ registration changes.

Canonical source of truth in your checkout:

```bash
ls tensorrt_model_connect/tensorrt_model_connect/families/*.py \
  | sed 's|.*/||; s|\.py$||' \
  | rg -v '^(__init__|base)$' \
  | sort
```

As of **February 26, 2026**, this repository contains 44 family modules:

`bark`, `bert`, `bloom`, `codegen`, `deeponet`, `deepseek_ocr`, `deepseek_v2`, `eagle_vlm`, `falcon`, `flux`, `fno`, `gemma`, `gpt2`, `gpt_neo`, `gpt_neox`, `granite`, `internlm`, `internvl`, `llama`, `mamba`, `mistral`, `mixtral`, `nemotron`, `nemotron_h`, `olmo`, `opt`, `personaplex`, `phi`, `phi4_multimodal`, `phi_moe`, `qwen`, `qwen3_omni`, `qwen_moe`, `qwen_vl`, `rwkv`, `sam`, `segformer`, `stablelm`, `starcoder2`, `wan_t2v`, `whisper`, `xglm`, `yolox`, `z_image`.

These are composed at runtime by the strategy builders selected from `src/cabi/api/trtmc_c.cpp`, including:
- `decoder_kv_cache`, `decoder_moe`
- `ssm_recurrent`, `rwkv_recurrent`, `hybrid_mamba_attention`
- `vision_language`, `segmentation`, `prompted_segmentation`, `object_detection`
- `speech_to_text`, `text_to_audio`, `speech_to_speech`
- `encoder_only`, `embedding`, `reranking`, `neural_operator`
- `omni_multimodal`, `diffusion`

Store bundles on persistent storage with `-o /workspace/users/yifeif/tensorrt-model-connect/engines/model.trtfb`.

## Adding a model family

```bash
# 1. Scaffold a plugin from a HuggingFace model
python3 scripts/new_family.py \
  --model-type phi3 --hf-repo microsoft/Phi-3-mini-4k-instruct --family-name phi

# 2. Review and customize the generated plugin
#    (edit tensorrt_model_connect/tensorrt_model_connect/families/phi.py)

# 3. One-command validation (build + diff_logits + diff_layers + runner parity)
./scripts/validate_family.sh microsoft/Phi-3-mini-4k-instruct
```

Plugins are auto-discovered — no registration code needed. No C++ changes required.

See [Adding a Model Family](docs/wiki/Adding-a-Model-Family.md) for the full guide, or `tensorrt_model_connect/tensorrt_model_connect/families/qwen.py` for an example.

## Environment variables

| Variable | Description |
|----------|-------------|
| `TRTMC_TRT_LOG_STDERR=1` | Enable TRT logger output to stderr |
| `TRTMC_TRT_LOG_MIN_SEVERITY` | Minimum TRT log severity |

## Testing

The test suite has six layers, from fast unit tests to full E2E GPU validation.
See [Testing and Validation](docs/wiki/Testing-and-Validation.md) for the
comprehensive manual (every file, every abstraction layer, every pytest marker).
Traceability policy is documented in
[Traceability Matrix](docs/wiki/Traceability-Matrix.md): every test must
state intent, preconditions, and postconditions, and map to architecture/design IDs.

### Quick reference: run everything

```bash
# === Tier 1: Unit tests (no GPU, ~10 min) ===

# Python builder (50 modules, ~940 tests — config, weights, plugins, bundles)
/opt/venv/bin/python -m pytest tests/builder/ -v --ignore=tests/builder/test_cli.py

# Tools self-tests (11 modules, ~160 tests — diff framework, comparison utilities)
/opt/venv/bin/python -m pytest tests/tools/ -v

# C++ runtime (19 executables, 20 tests — bundle format, tokenizers, CUDA, KV cache)
ctest --test-dir build --output-on-failure

# C++ cyclomatic complexity report + gate (lizard-based CCM)
python tools/check_cyclomatic_complexity.py src
python tools/check_cyclomatic_complexity.py src --max-ccn 10

# CI parity check (same policy as check-cyclomatic-complexity job)
CCM_MAX_CCN=10 python tools/check_cyclomatic_complexity.py src --max-ccn ${CCM_MAX_CCN}

# Coverage gates (strict 100% thresholds)
tools/coverage/python_coverage.sh -v --ignore=tests/builder/test_cli.py
tools/coverage/cpp_coverage.sh
tools/coverage/run_coverage_all.sh

# === Tier 2: Graph-op GPU tests (~2 min, needs TRT) ===

/opt/venv/bin/python -m pytest tests/builder/test_graph_ops.py \
  tests/builder/test_graph_ops_extended.py \
  tests/builder/test_graph_blocks.py -v -m trt

# === Tier 3: E2E single-model smoke test (~5 min, needs GPU) ===

/opt/venv/bin/python -m pytest tests/test_e2e.py::test_e2e[qwen3-0.6b] -v \
  --engine-dir /workspace/users/yifeif/tensorrt-model-connect/engines \
  --trtmc-binary ./build/trtmc --hf-python /opt/venv/bin/python \
  --rebuild-engines

# === Tier 4: Full E2E suite (50 models, ~2-3 hours, needs GPU) ===

/opt/venv/bin/python -m pytest tests/test_e2e.py -v \
  --engine-dir /workspace/users/yifeif/tensorrt-model-connect/engines \
  --trtmc-binary ./build/trtmc --hf-python /opt/venv/bin/python \
  --rebuild-engines --e2e-artifacts-dir /tmp/e2e_artifacts

# === Tier 5: Performance + accuracy comparison (manual, per model) ===

# Accuracy validation: Torch-TRT logits vs HF eager
python3 tools/diff_torchtrt.py \
  --model Qwen/Qwen2.5-0.5B --atol 1e-2 --battery --verbose

# All 4 backends: Torch-TRT vs Raw TRT vs torch.compile vs HF eager
python3 tools/perf_compare_all.py \
  --model Qwen/Qwen3-0.6B \
  --prompt "The capital of France is" \
  --max-new-tokens 50 --json results.json

# With pre-built bundles (skips engine build)
python3 tools/perf_compare_all.py \
  --model Qwen/Qwen3-0.6B \
  --torchtrt-bundle /tmp/qwen3_torchtrt.trtfb \
  --rawtrt-bundle /tmp/qwen3_raw.trtfb \
  --prompt "The capital of France is" \
  --max-new-tokens 50

# Skip specific backends
python3 tools/perf_compare_all.py \
  --model Qwen/Qwen3-0.6B --skip-compile --skip-rawtrt

# Force fp32 precision (compare accuracy vs speed tradeoff)
python3 tools/perf_compare_all.py \
  --model Qwen/Qwen3-0.6B --precision fp32

# Individual pipeline comparisons
python3 tools/perf_compare.py \
  --model Qwen/Qwen3-0.6B --max-new-tokens 20 --json raw_trt.json

python3 tools/perf_compare_torchtrt.py \
  --model Qwen/Qwen3-0.6B --precision fp16 --max-new-tokens 20 --json torch_trt.json
```

### What to run when

| Change type | What to run |
|-------------|------------|
| Python builder logic | Tier 1 + Tier 2 |
| Family plugin | Tier 1, Tier 2, Tier 3 (specific model) |
| C++ runtime | Tier 1 (ctest), Tier 3 |
| Graph ops | Tier 1, Tier 2 |
| KV cache / position logic | Tier 1, Tier 3, Tier 4 |
| New model (existing family) | Add JSON manifest to `tests/e2e/models/`, run Tier 3 |
| New model family | Tier 1, Tier 2, `validate_family.sh`, add manifest, Tier 4 |
| Performance-sensitive change | Tier 5 (`perf_compare_all.py`, `diff_torchtrt.py`) |

### Filtering with pytest markers

```bash
# Only unit tests (no GPU)
pytest tests/builder/ -m unit -v

# Only GPU/TRT tests
pytest tests/builder/ -m gpu -v

# E2E by modality
pytest tests/test_e2e.py --e2e-task-strategy text_generation_causal -v ...
pytest tests/test_e2e.py --e2e-task-strategy diffusion_media_generation -v ...
pytest tests/test_e2e.py --e2e-task-strategy vision_language_generation -v ...
```

### Coverage

```bash
/opt/venv/bin/python -m pytest tests/builder/ tests/tools/ -v \
  --ignore=tests/builder/test_cli.py --cov --cov-report=term-missing
```

## Documentation

| Page | Description |
|------|-------------|
| [Architecture Overview](docs/wiki/Architecture-Overview.md) | Two-stage pipeline, bundle format |
| [Pipeline Deep Dive](docs/wiki/Pipeline-Deep-Dive.md) | Full call chain, data structures |
| [TRT Internals](docs/wiki/TRT-Internals.md) | Decoder layer anatomy, graph ops |
| [HF vs TRT Comparison](docs/wiki/HF-vs-TRT-Comparison.md) | Side-by-side comparison |
| [Testing and Validation](docs/wiki/Testing-and-Validation.md) | Complete test manual: 6 layers, every file, every marker |
| [Coverage in GitLab](docs/coverage/gitlab_coverage_report.md) | Cobertura setup, strict gates, local/CI commands |
| [Traceability Matrix](docs/wiki/Traceability-Matrix.md) | Bi-directional architecture -> unit design -> test mapping and maintenance process |
| [Adding a Model Family](docs/wiki/Adding-a-Model-Family.md) | Step-by-step guide |
| [Static Design](docs/wiki/Static-Design.md) | Class-level UML diagrams and descriptions |
| [Source Layout](docs/wiki/Source-Layout.md) | File-by-file guide |
| [Extensibility Assessment](docs/wiki/Architecture-Extensibility-Assessment.md) | MoE, Mamba/SSM, MLA support status |
| [Torch-TRT Agent Guide](docs/torch-trt/TORCHTRT_AGENT_GUIDE.md) | Torch-TRT pipeline architecture and development guide |
| [Torch-TRT Work Log](docs/torch-trt/TORCHTRT_WORKLOG.md) | Torch-TRT pipeline development history |
| [Development Log](docs/WORKLOG.md) | Chronological history (raw TRT pipeline) |
| [CLAUDE.md](CLAUDE.md) | Full build/test/container runbook |
