# trt-transformers-cpp

Python builds TensorRT engines from HuggingFace models. C++ runs them.

The system is split into two stages:
- **`trtf_build`** (Python) — downloads an HF model, builds a TRT engine, and packages it into a `.trtfb` bundle.
- **`trtf`** (C++) — loads a `.trtfb` bundle and runs autoregressive inference on GPU.

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
# Run inference from a bundle
./build/trtf run qwen3.trtfb --prompt "Hello" --max-new-tokens 50 \
  --hf-python .venv/bin/python

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

p->generate("Hello", 50);   // returns const char*
p->model_id();               // returns const char*
p->backend_name();           // returns const char*
delete p;

// Utilities
trtf_last_error();           // thread-local error message
trtf_version();
trtf_has_trt();              // 1 if compiled with TRT support
```

## Supported models

13 model families covering dense decoders, MoE, and multiple architecture variants.

### Standard decoder (RMSNorm + RoPE + SwiGLU)

| Family | Example Model | `model_type` | Build Command |
|--------|--------------|-------------|---------------|
| **Qwen** | Qwen/Qwen3-0.6B | `qwen`, `qwen2`, `qwen3` | `trtf-build build Qwen/Qwen3-0.6B -o qwen3.trtfb` |
| **LLaMA** | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | `llama` | `trtf-build build TinyLlama/TinyLlama-1.1B-Chat-v1.0 -o tinyllama.trtfb` |
| **Mistral** | mistralai/Mistral-7B-v0.1 | `mistral` | `trtf-build build mistralai/Mistral-7B-v0.1 -o mistral.trtfb` |
| **Gemma** | google/gemma-2-2b | `gemma`, `gemma2` | `trtf-build build google/gemma-2-2b -o gemma2.trtfb` |
| **Phi-3** | microsoft/Phi-3-mini-4k-instruct | `phi`, `phi3` | `trtf-build build microsoft/Phi-3-mini-4k-instruct -o phi3.trtfb` |
| **Granite** | ibm-granite/granite-3.1-2b-base | `granite` | `trtf-build build ibm-granite/granite-3.1-2b-base -o granite.trtfb` |
| **InternLM** | internlm/internlm2-math-plus-1_8b | `internlm2` | `trtf-build build internlm/internlm2-math-plus-1_8b -o internlm2.trtfb` |

### Extended decoder (LayerNorm, GELU, learned positions)

| Family | Example Model | `model_type` | Build Command |
|--------|--------------|-------------|---------------|
| **StarCoder2** | bigcode/starcoder2-3b | `starcoder2` | `trtf-build build bigcode/starcoder2-3b -o starcoder2.trtfb` |
| **GPT-2** | openai-community/gpt2 | `gpt2` | `trtf-build build openai-community/gpt2 -o gpt2.trtfb` |
| **OPT** | facebook/opt-125m | `opt` | `trtf-build build facebook/opt-125m -o opt.trtfb` |
| **Falcon** | tiiuae/Falcon3-1B-Base | `falcon` | `trtf-build build tiiuae/Falcon3-1B-Base -o falcon.trtfb` |
| **StableLM** | stabilityai/stablelm-2-1_6b | `stablelm` | `trtf-build build stabilityai/stablelm-2-1_6b -o stablelm.trtfb` |

### Mixture of Experts (MoE)

| Family | Example Model | `model_type` | Build Command |
|--------|--------------|-------------|---------------|
| **Phi-MoE** | microsoft/Phi-tiny-MoE-instruct | `phimoe` | `trtf-build build microsoft/Phi-tiny-MoE-instruct -o phi-moe.trtfb` |

Any HF model whose `model_type` matches a supported family works automatically. Store bundles on persistent storage with `-o /mnt/storage/trt-transformers/engines/model.trtfb`.

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

## Documentation

| Page | Description |
|------|-------------|
| [Architecture Overview](docs/wiki/Architecture-Overview.md) | Two-stage pipeline, bundle format |
| [Pipeline Deep Dive](docs/wiki/Pipeline-Deep-Dive.md) | Full call chain, data structures |
| [TRT Internals](docs/wiki/TRT-Internals.md) | Decoder layer anatomy, graph ops |
| [HF vs TRT Comparison](docs/wiki/HF-vs-TRT-Comparison.md) | Side-by-side comparison |
| [Adding a Model Family](docs/wiki/Adding-a-Model-Family.md) | Step-by-step guide |
| [Source Layout](docs/wiki/Source-Layout.md) | File-by-file guide |
| [Development Log](docs/WORKLOG.md) | Chronological history |
| [CLAUDE.md](CLAUDE.md) | Full build/test/container runbook |
