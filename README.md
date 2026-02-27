# trt-transformers-cpp

Python builds TensorRT engines from HuggingFace models. C++ runs them.

The system is split into two stages:
- **`trtf_build`** (Python) — downloads an HF model, builds a TRT engine, and packages it into a `.trtfb` bundle.
- **`trtf`** (C++) — loads a `.trtfb` bundle and runs task-specific GPU inference (text, vision, audio, diffusion, neural operators, etc.).

## Quick start

Prerequisites: Docker + NVIDIA Container Toolkit.

```bash
# 1. Build and launch the dev container
./scripts/docker_build.sh
./scripts/docker_run.sh

# 2. Inside the container: install deps + build C++ runtime
./scripts/setup_container.sh
source .venv/bin/activate

# 3. Build a bundle from a HuggingFace model (auto-downloads)
#    Bundles are stored on persistent storage so they survive container restarts.
trtf-build build Qwen/Qwen3-0.6B -o /mnt/storage/trt-transformers/engines/qwen3.trtfb

# 4. Run inference
export LD_LIBRARY_PATH="$(python3 -c 'import importlib.util; s=importlib.util.find_spec(\"tensorrt_libs\"); print(s.submodule_search_locations[0])'):/usr/local/cuda/lib64"
./build/trtf run /mnt/storage/trt-transformers/engines/qwen3.trtfb \
  --prompt "The capital of France is" \
  --max-new-tokens 20 \
  --hf-python .venv/bin/python
```

## Python API

```python
import trtf_build

# From a HuggingFace repo ID (auto-downloads):
trtf_build.build("Qwen/Qwen3-0.6B", "qwen3.trtfb")

# From a local directory:
trtf_build.build("models/hf/Qwen__Qwen3-0.6B", "qwen3.trtfb")

# With options:
trtf_build.build("Qwen/Qwen3-0.6B", "qwen3.trtfb",
                  max_cache_length=512, verbose=True)
```

Install the package:
```bash
pip install --no-deps -e trtf_build/
```

For non-Docker environments, also install the matching TensorRT wheel (`tensorrt_cu12` or `tensorrt`) and ensure TensorRT/CUDA libraries are on `LD_LIBRARY_PATH` before running `./build/trtf`.

## CLI

### Python builder (`trtf-build`)

```bash
# Build from HF repo ID (auto-downloads)
trtf-build build Qwen/Qwen3-0.6B -o qwen3.trtfb

# Build from local directory
trtf-build build models/hf/Qwen__Qwen3-0.6B -o qwen3.trtfb

# Options
trtf-build build Qwen/Qwen3-0.6B -o qwen3.trtfb --max-cache-length 512 --verbose

# Inspect a bundle
trtf-build inspect qwen3.trtfb

# Version info
trtf-build version
```

### C++ runtime (`trtf`)

```bash
# Text generation (and VLM generation with --image when supported)
./build/trtf run qwen3.trtfb --prompt "Hello" --max-new-tokens 50 \
  --hf-python .venv/bin/python

# Encoder-only hidden states
./build/trtf encode qwen3.trtfb --prompt "Hello" --hf-python .venv/bin/python

# Segmentation / prompted segmentation / detection
./build/trtf segment segformer.trtfb --image input.png --output mask.png
./build/trtf segment-sam sam.trtfb --image input.png --output sam_masks/
./build/trtf detect yolox.trtfb --image input.png --output detections.json --threshold 0.5

# Embedding / reranking
./build/trtf embed bert.trtfb --prompt "Hello"
./build/trtf rerank personaplex.trtfb --prompt "query" --document "candidate passage"

# Audio + speech
./build/trtf transcribe whisper.trtfb --audio sample.wav
./build/trtf generate-audio bark.trtfb --prompt "A calm narration" --output out.wav
./build/trtf speak qwen3_omni.trtfb --audio-in in.wav --audio-out out.wav

# Neural operators (DeepONet / FNO)
./build/trtf solve deeponet.trtfb --branch-input "0.1,0.2" --trunk-input "0.5,0.5"
./build/trtf solve fno.trtfb --field-input "0.1,0.2,0.3,0.4"

# Diffusion video
./build/trtf generate-video wan_t2v.trtfb --prompt "A cat riding a bike" --output frames/

# Inspect bundle metadata
./build/trtf inspect qwen3.trtfb

# Version info
./build/trtf version
```

## C API

The public C++ API is a single header (`include/trtf/pipeline.h`):

```cpp
#include <trtf/pipeline.h>

// Create from .trtfb bundle
TrtfPipelineOptions opts = {.max_new_tokens = 50, .hf_python = "/path/to/python"};
auto* p = trtf_create_pipeline_ex("model.trtfb", &opts);

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
trtf_last_error();           // thread-local error message
trtf_version();
trtf_has_trt();              // 1 if compiled with TRT support
```

## Supported models

Model support is plugin-driven and auto-discovered from `trtf_build/trtf_build/families/` at build time. Any HF model whose `config.json` `model_type` matches a plugin is supported without C++ registration changes.

Canonical source of truth in your checkout:

```bash
ls trtf_build/trtf_build/families/*.py \
  | sed 's|.*/||; s|\.py$||' \
  | rg -v '^(__init__|base)$' \
  | sort
```

As of **February 26, 2026**, this repository contains 44 family modules:

`bark`, `bert`, `bloom`, `codegen`, `deeponet`, `deepseek_ocr`, `deepseek_v2`, `eagle_vlm`, `falcon`, `flux`, `fno`, `gemma`, `gpt2`, `gpt_neo`, `gpt_neox`, `granite`, `internlm`, `internvl`, `llama`, `mamba`, `mistral`, `mixtral`, `nemotron`, `nemotron_h`, `olmo`, `opt`, `personaplex`, `phi`, `phi4_multimodal`, `phi_moe`, `qwen`, `qwen3_omni`, `qwen_moe`, `qwen_vl`, `rwkv`, `sam`, `segformer`, `stablelm`, `starcoder2`, `wan_t2v`, `whisper`, `xglm`, `yolox`, `z_image`.

These map to runtime strategies in `src/cabi/trtf_c.cpp`, including:
- `decoder_kv_cache`, `decoder_moe`
- `ssm_recurrent`, `rwkv_recurrent`, `hybrid_mamba_attention`
- `vision_language`, `segmentation`, `prompted_segmentation`, `object_detection`
- `speech_to_text`, `text_to_audio`, `speech_to_speech`
- `encoder_only`, `embedding`, `reranking`, `neural_operator`
- `omni_multimodal`, `diffusion`

Store bundles on persistent storage with `-o /mnt/storage/trt-transformers/engines/model.trtfb`.

## Adding a model family

```bash
# 1. Scaffold a plugin from a HuggingFace model
python3 scripts/new_family.py \
  --model-type phi3 --hf-repo microsoft/Phi-3-mini-4k-instruct --family-name phi

# 2. Review and customize the generated plugin
#    (edit trtf_build/trtf_build/families/phi.py)

# 3. One-command validation (build + diff_logits + diff_layers + runner parity)
./scripts/validate_family.sh microsoft/Phi-3-mini-4k-instruct
```

Plugins are auto-discovered — no registration code needed. No C++ changes required.

See [Adding a Model Family](docs/wiki/Adding-a-Model-Family.md) for the full guide, or `trtf_build/trtf_build/families/qwen.py` for an example.

## Environment variables

| Variable | Description |
|----------|-------------|
| `TRTF_TRT_LOG_STDERR=1` | Enable TRT logger output to stderr |
| `TRTF_TRT_LOG_MIN_SEVERITY` | Minimum TRT log severity |

## Testing

The test suite has six layers, from fast unit tests to full E2E GPU validation.
See [Testing and Validation](docs/wiki/Testing-and-Validation.md) for the
comprehensive manual (every file, every abstraction layer, every pytest marker).

### Quick reference: run everything

```bash
# === Tier 1: Unit tests (no GPU, ~10 min) ===

# Python builder (50 modules, ~940 tests — config, weights, plugins, bundles)
.venv/bin/python -m pytest tests/builder/ -v --ignore=tests/builder/test_cli.py

# Tools self-tests (11 modules, ~160 tests — diff framework, comparison utilities)
.venv/bin/python -m pytest tests/tools/ -v

# C++ runtime (19 executables, 20 tests — bundle format, tokenizers, CUDA, KV cache)
ctest --test-dir build --output-on-failure

# === Tier 2: Graph-op GPU tests (~2 min, needs TRT) ===

.venv/bin/python -m pytest tests/builder/test_graph_ops.py \
  tests/builder/test_graph_ops_extended.py \
  tests/builder/test_graph_blocks.py -v -m trt

# === Tier 3: E2E single-model smoke test (~5 min, needs GPU) ===

.venv/bin/python -m pytest tests/test_e2e.py::test_e2e[qwen3-0.6b] -v \
  --engine-dir /mnt/storage/trt-transformers/engines \
  --trtf-binary ./build/trtf --hf-python .venv/bin/python \
  --rebuild-engines

# === Tier 4: Full E2E suite (50 models, ~2-3 hours, needs GPU) ===

.venv/bin/python -m pytest tests/test_e2e.py -v \
  --engine-dir /mnt/storage/trt-transformers/engines \
  --trtf-binary ./build/trtf --hf-python .venv/bin/python \
  --rebuild-engines --e2e-artifacts-dir /tmp/e2e_artifacts

# === Tier 5: Performance regression (manual, per model) ===

python3 tools/perf_compare.py \
  --model Qwen/Qwen3-0.6B \
  --bundle /mnt/storage/trt-transformers/engines/qwen3-0.6b.trtfb \
  --prompt "The capital of France is" --max-new-tokens 20 --json results.json
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
.venv/bin/python -m pytest tests/builder/ tests/tools/ -v \
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
| [Adding a Model Family](docs/wiki/Adding-a-Model-Family.md) | Step-by-step guide |
| [Static Design](docs/wiki/Static-Design.md) | Class-level UML diagrams and descriptions |
| [Source Layout](docs/wiki/Source-Layout.md) | File-by-file guide |
| [Extensibility Assessment](docs/wiki/Architecture-Extensibility-Assessment.md) | MoE, Mamba/SSM, MLA support status |
| [Development Log](docs/WORKLOG.md) | Chronological history |
| [CLAUDE.md](CLAUDE.md) | Full build/test/container runbook |
