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

15 model families covering dense decoders, MoE, SSM, vision-language, and multiple architecture variants. Any HF model whose `config.json` `model_type` matches a supported family works automatically.

### Standard decoder (RMSNorm + RoPE + SwiGLU)

| Family | Matched `model_type` | Compatible Models | Build Command |
|--------|---------------------|-------------------|---------------|
| **Qwen** | `qwen`, `qwen2`, `qwen3`, `qwq` | Qwen3 (0.6B–72B), Qwen2.5, QwQ, DeepSeek-R1-Distill-Qwen, DeepSeek-Coder-V2-Lite | `trtf-build build Qwen/Qwen3-0.6B -o qwen3.trtfb` |
| **LLaMA** | `llama` | LLaMA 2/3, TinyLlama, CodeLlama, Vicuna, Yi, Yi-Coder, DeepSeek-Coder-1.3B | `trtf-build build TinyLlama/TinyLlama-1.1B-Chat-v1.0 -o tinyllama.trtfb` |
| **Mistral** | `mistral` | Mistral 7B, Mistral Nemo | `trtf-build build mistralai/Mistral-7B-v0.1 -o mistral.trtfb` |
| **Gemma** | `gemma`, `gemma2` | Gemma, Gemma 2 (2B–27B) | `trtf-build build google/gemma-2-2b -o gemma2.trtfb` |
| **Phi** | `phi`, `phi3` (not `phimoe`) | Phi-3, Phi-3.5, Phi-4 | `trtf-build build microsoft/Phi-3-mini-4k-instruct -o phi3.trtfb` |
| **Granite** | `granite` | Granite 3.1 (2B–8B) | `trtf-build build ibm-granite/granite-3.1-2b-base -o granite.trtfb` |
| **InternLM** | `internlm`, `internlm2` | InternLM2, InternLM2.5, InternLM2-Math | `trtf-build build internlm/internlm2-math-plus-1_8b -o internlm2.trtfb` |

### Extended decoder (LayerNorm, GELU, learned positions)

| Family | Matched `model_type` | Compatible Models | Build Command |
|--------|---------------------|-------------------|---------------|
| **StarCoder2** | `starcoder2` | StarCoder2 (3B–15B) | `trtf-build build bigcode/starcoder2-3b -o starcoder2.trtfb` |
| **GPT-2** | `gpt2` | GPT-2 (125M–1.5B), DistilGPT-2 | `trtf-build build openai-community/gpt2 -o gpt2.trtfb` |
| **OPT** | `opt` | OPT (125M–66B) | `trtf-build build facebook/opt-125m -o opt.trtfb` |
| **Falcon** | `falcon` | Falcon 3 (1B–10B), Falcon (7B–180B) | `trtf-build build tiiuae/Falcon3-1B-Base -o falcon.trtfb` |
| **StableLM** | `stablelm` | StableLM 2 (1.6B–12B) | `trtf-build build stabilityai/stablelm-2-1_6b -o stablelm.trtfb` |

### Mixture of Experts (MoE)

| Family | Matched `model_type` | Compatible Models | Build Command |
|--------|---------------------|-------------------|---------------|
| **Phi-MoE** | `phimoe` | Phi-tiny-MoE (3.8B/1.1B active) | `trtf-build build microsoft/Phi-tiny-MoE-instruct -o phi-moe.trtfb` |

### State Space Models (SSM)

| Family | Matched `model_type` | Compatible Models | Build Command |
|--------|---------------------|-------------------|---------------|
| **Mamba** | `mamba` | Mamba (130M–2.8B) | `trtf-build build state-spaces/mamba-130m-hf -o mamba-130m.trtfb` |

### Vision-Language (text decoder only)

| Family | Matched `model_type` | Compatible Models | Build Command |
|--------|---------------------|-------------------|---------------|
| **Qwen-VL** | `qwen2_vl`, `qwen2_5_vl` | Qwen2.5-VL (text decoder) | `trtf-build build Qwen/Qwen2.5-VL-3B -o qwen-vl.trtfb` |

### Notable compatible models (via model_type matching)

These popular models work out of the box because they use a supported `model_type`:

```bash
# DeepSeek R1 distilled variants (model_type=qwen2 → Qwen plugin)
trtf-build build deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B -o deepseek-r1-1.5b.trtfb
trtf-build build deepseek-ai/DeepSeek-R1-Distill-Qwen-7B -o deepseek-r1-7b.trtfb

# Yi models (model_type=llama → LLaMA plugin)
trtf-build build 01-ai/Yi-1.5-6B -o yi-1.5-6b.trtfb

# CodeLlama (model_type=llama → LLaMA plugin)
trtf-build build codellama/CodeLlama-7b-hf -o codellama-7b.trtfb

# Vicuna (model_type=llama → LLaMA plugin)
trtf-build build lmsys/vicuna-7b-v1.5 -o vicuna-7b.trtfb

# Gemma 2 (model_type=gemma2 → Gemma plugin)
trtf-build build google/gemma-2-9b -o gemma2-9b.trtfb

# Phi-4 (model_type=phi3 → Phi plugin)
trtf-build build microsoft/phi-4 -o phi4.trtfb
```

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

## Documentation

| Page | Description |
|------|-------------|
| [Architecture Overview](docs/wiki/Architecture-Overview.md) | Two-stage pipeline, bundle format |
| [Pipeline Deep Dive](docs/wiki/Pipeline-Deep-Dive.md) | Full call chain, data structures |
| [TRT Internals](docs/wiki/TRT-Internals.md) | Decoder layer anatomy, graph ops |
| [HF vs TRT Comparison](docs/wiki/HF-vs-TRT-Comparison.md) | Side-by-side comparison |
| [Adding a Model Family](docs/wiki/Adding-a-Model-Family.md) | Step-by-step guide |
| [Static Design](docs/wiki/Static-Design.md) | Class-level UML diagrams and descriptions |
| [Source Layout](docs/wiki/Source-Layout.md) | File-by-file guide |
| [Extensibility Assessment](docs/wiki/Architecture-Extensibility-Assessment.md) | MoE, Mamba/SSM, MLA support status |
| [Development Log](docs/WORKLOG.md) | Chronological history |
| [CLAUDE.md](CLAUDE.md) | Full build/test/container runbook |
