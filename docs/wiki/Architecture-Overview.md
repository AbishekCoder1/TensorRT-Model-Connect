# Architecture Overview

## Python Builds, C++ Runs

The system is split into two phases across two languages:

| Phase | Language | Tool | Input | Output |
|-------|----------|------|-------|--------|
| **Build** | Python | `trtf-build` / `trtf_build.build()` | HF repo ID or local directory | `.trtfb` bundle |
| **Run** | C++ | `trtf` / C ABI | `.trtfb` bundle | Generated text |

**Build phase** (Python `trtf_build/` package — accepts HF repo IDs or local directories):
1. Resolve model: auto-download from HuggingFace Hub if needed
2. Parse `config.json` to identify model family and architecture
3. Load safetensors weights via the `safetensors` Python library
4. Map HF tensor keys to canonical format (per-family checkpoint mapper)
5. Build TRT `INetworkDefinition` via the TensorRT Python API
6. Compile to `ICudaEngine` (GPU kernel compilation)
7. Package engine plan + tokenizer files into a `.trtfb` bundle

**Run phase** (C++ runtime, ~13 source files):
1. Load `.trtfb` bundle, deserialize TRT engine plan
2. Create tokenizer (HfPythonTokenizer or VocabTokenizer)
3. Run autoregressive generation loop on GPU with KV-cache management
4. Return generated text via C ABI

---

## Build Phase: Python CLI + API

### CLI (`trtf-build`)

```
trtf-build build Qwen/Qwen3-0.6B -o output.trtfb [--max-cache-length N]
trtf-build build <local-model-dir> -o output.trtfb [--max-cache-length N]
trtf-build inspect <bundle.trtfb>
trtf-build version
```

### Python API

```python
import trtf_build
trtf_build.build("Qwen/Qwen3-0.6B", "qwen3.trtfb")            # auto-downloads
trtf_build.build("models/hf/Qwen__Qwen3-0.6B", "qwen3.trtfb")  # local directory
```

The build phase uses the `trtf_build/` Python package, which provides:

### Family Plugin System

Each model family is a single Python file in `trtf_build/trtf_build/families/`:

```
trtf_build/trtf_build/families/
  base.py       # FamilyPlugin protocol
  qwen.py       # Qwen, Qwen2, Qwen3, QWQ
  llama.py      # LLaMA, TinyLlama
  mistral.py    # Mistral, TinyMistral
  gemma.py      # Gemma (adds +1.0 to RMSNorm gamma, scales embedding)
```

A family plugin provides:
- **Matcher**: Checks `model_type` from `config.json` to claim the model
- **Checkpoint mapper**: Translates HF safetensors tensor keys to canonical format (transpose, GQA KV expansion, etc.)
- **Graph builder**: Constructs the TRT network definition (reusing shared ops for RMSNorm, RoPE, GQA attention, SwiGLU, etc.)

### What Python replaced in C++

The following C++ code (~3500 lines) was removed and replaced by Python equivalents:

| Former C++ Component | Python Replacement |
|---|---|
| `SafetensorReader`, `TensorSource` | `safetensors` Python library |
| `ICheckpointMapper`, `StandardCheckpointMapper` | Per-family Python checkpoint mappers |
| `LoadDecoderModel`, `model_loader.cpp` | Python config.json parsing + weight loading |
| `StandardDecoderGraphBuilder`, `trt_graph_ops` | Python TRT network construction via `tensorrt` API |
| `TrtDecoderDefinition`, `BuildTrtDecoderWeights` | Direct Python weight-to-TRT conversion |
| `IModelRuntime`, `RegisterModelRuntime` | Python family plugin dispatch |
| `ResolveTextGenerationModel`, `ResolveHfModelViaFamilyRegistry` | Python model resolution |
| `RegisterHfModelFamily`, `RegisterBuiltinHfModelFamilies` | Python family registry |
| Engine cache system | Bundle-based caching (build once, run many times) |
| `FastPathModelConfig` | Not needed (Python builds, C++ only loads bundles) |
| `tensor_math.h/cpp` | NumPy operations in Python |

---

## Run Phase: C++ Runtime

### C ABI Entry Point (`trtf_c.cpp`)

Users access the library through `extern "C"` factory functions `trtf_create_pipeline()` and `trtf_create_pipeline_ex()` which return an `IPipeline*`. These functions accept `.trtfb` bundle paths. The factory loads the pre-compiled engine plan, creates a tokenizer, and returns a ready-to-use pipeline.

`trtf_create_pipeline_ex()` accepts a `TrtfPipelineOptions` struct:

```cpp
struct TrtfPipelineOptions {
    int max_new_tokens;           // 0 = use model default (20)
    const char* hf_python;        // nullptr = auto-detect
};
```

### Bundle Loading Flow

```
trtf_create_pipeline("model.trtfb", TRTF_FORCE_TRT)
  |
  +-- Detect bundle magic bytes
  +-- ReadBundleFile() -> engine plan bytes + tokenizer files
  +-- TRT deserializeCudaEngine(plan)
  +-- Extract tokenizer files to temp dir
  +-- Create HfPythonTokenizer or VocabTokenizer
  +-- Create TrtBackendFastPath (engine + generation loop)
  +-- Return PipelineImpl
```

### The Two Backends

Both implement `IGenerationBackend`:

```cpp
class IGenerationBackend {
    virtual bool is_available() const = 0;
    virtual const char* name() const = 0;
    virtual std::vector<int32_t> generate(const std::vector<int32_t>& input_ids,
                                           const GenerationConfig& config) = 0;
};
```

**TRT Backend** (`trt_backend_shared.cpp`):
- **Name**: `"trt"`
- **How it works**: Deserializes a pre-compiled TRT engine from a `.trtfb` bundle, then runs an autoregressive generation loop on GPU with KV-cache management, CUDA memory allocation, and greedy argmax sampling.

**HF Python Backend** (`hf_python_backend.cpp`):
- **Name**: `"hf-transformers"`
- **How it works**: Spawns a Python subprocess with HF transformers for maximum compatibility fallback.

---

## CLI Split

### Python CLI: `trtf-build`

```bash
trtf-build build Qwen/Qwen3-0.6B -o output.trtfb [--max-cache-length N]  # HF repo ID
trtf-build build <local-model-dir> -o output.trtfb [--max-cache-length N]  # local dir
trtf-build inspect <bundle.trtfb>
trtf-build version
```

### C++ CLI: `trtf`

```bash
trtf run <bundle.trtfb> --prompt "text" [--max-new-tokens N] [--hf-python PATH]
trtf inspect <bundle.trtfb>
trtf version
```

---

## Key Design Decisions

### Why split Python/C++?

The TensorRT C++ API for network construction is verbose and error-prone. The Python API is more ergonomic for graph building, checkpoint loading (via the `safetensors` library and NumPy), and iterating on new model families. Meanwhile, C++ excels at the runtime: engine deserialization, CUDA memory management, and the tight autoregressive loop. The bundle format bridges the two cleanly.

### Why no ONNX?

ONNX export introduces an intermediate representation that limits control over the TensorRT graph. By building `INetworkDefinition` directly via the TensorRT Python API, we get:
- Exact control over layer fusion and memory layout
- No ONNX parser dependency
- Reusable op primitives shared across all families
- Easier debugging of the graph structure

### Why bundles?

`.trtfb` bundles are self-contained artifacts: compiled TRT engine plan + tokenizer config. Benefits:
- **Build once, run anywhere** (same GPU architecture): no Python needed at runtime
- **Instant startup**: engine deserialization (~5s) vs. full build (~60-300s)
- **Deployment simplicity**: single file, no model directory needed
- **Clean separation**: Python handles the complex build, C++ handles the fast runtime
