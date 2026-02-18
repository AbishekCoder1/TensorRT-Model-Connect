# Plan: Support All NVIDIA HuggingFace Models That Fit on 24GB

## Context

The project aims to be a "speed of light" HuggingFace model deployment tool via TRT — not just for LLMs, but for all model types. This plan catalogs every relevant NVIDIA model (text gen, VL, embedding, reranking, OCR/parsing, ASR, TTS), determines which fit on 24GB, and provides a phased implementation roadmap expanding the architecture from decoder-only text generation to a general-purpose TRT inference platform.

**GPU: RTX 3090 Ti (24GB VRAM). Models up to ~4B fit comfortably; ~1B models fit easily.**

---

## Complete NVIDIA Model Inventory

### A. Text Generation — Already Supported (existing plugins)

| # | Model | HF Repo | Params | Fits? | Plugin |
|---|-------|---------|--------|-------|--------|
| 1 | Nemotron-Nano-4B | `nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1` | 4B | **Yes** | llama |
| 2 | Minitron-4B-Depth | `nvidia/Llama-3.1-Minitron-4B-Depth-Base` | 4B | **Yes** | llama |
| 3 | Minitron-4B-Width | `nvidia/Llama-3.1-Minitron-4B-Width-Base` | 4B | **Yes** | llama |
| 4 | Riva-Translate-4B | `nvidia/Riva-Translate-4B-Instruct-v1.1` | 4B | **Yes** | mistral |
| 5-13 | 8B-253B models | Various | 8B-253B | No | Already work, bigger GPU needed |

### B. Text Generation — New Nemotron-4 Plugin

| # | Model | HF Repo | Params | Fits? |
|---|-------|---------|--------|-------|
| 14 | **Nemotron-Mini-4B** | `nvidia/Nemotron-Mini-4B-Instruct` | 4B | **Yes** |
| 15 | **Nemotron-4-Mini-Hindi-4B** | `nvidia/Nemotron-4-Mini-Hindi-4B-Base` | 4B | **Yes** |
| 16-17 | 8B-340B variants | Various | 8B-340B | No |

### C. Vision-Language

| # | Model | HF Repo | Params | Fits? | Notes |
|---|-------|---------|--------|-------|-------|
| 18 | **Cosmos-Reason2-2B** | `nvidia/Cosmos-Reason2-2B` | 2.4B | **Yes** | Qwen3-VL base, existing qwen_vl plugin likely works |
| 19-26 | 7B+ VL models | Various | 7B-72B | No | Need bigger GPU |

### D. Hybrid Architectures (new C++ backend)

| # | Model | HF Repo | Params | Fits? | Architecture |
|---|-------|---------|--------|-------|-------------|
| 27 | **Hymba-1.5B** | `nvidia/Hymba-1.5B-Base` | 1.5B | **Yes** | Parallel attn+SSM, meta-tokens, KV sharing |
| 28 | **Nemotron-Flash-1B** | `nvidia/Nemotron-Flash-1B` | 1B | **Yes** | Evolved hybrid (attn+Mamba+linear-attn) |
| 29 | **Nemotron-Flash-3B** | `nvidia/Nemotron-Flash-3B` | 3B | **Yes** | Same evolved hybrid |
| 30 | **Nemotron-Flash-3B-Instruct** | `nvidia/Nemotron-Flash-3B-Instruct` | 3B | **Yes** | Same |
| 31-32 | 9B-30B hybrids | Various | 9B-30B | No | Same backend, bigger GPU |

### E. Embedding / Reranking (NEW CAPABILITY)

| # | Model | HF Repo | Params | Fits? | Architecture | Notes |
|---|-------|---------|--------|-------|-------------|-------|
| 33 | **llama-nemotron-embed-1b** | `nvidia/llama-nemotron-embed-1b-v2` | 1B | **Yes** | LLaMA 3.2 1B + avg pooling | Text embedding, 2048-dim |
| 34 | **llama-nemotron-rerank-1b** | `nvidia/llama-nemotron-rerank-1b-v2` | 1B | **Yes** | LLaMA 3.2 1B + classification head | Cross-encoder reranker |
| 35 | llama-nemotron-embed-vl-1b | `nvidia/llama-nemotron-embed-vl-1b-v2` | 1.7B | **Yes** | LLaMA + SigLip2 VL embedding | Needs VL + Eagle arch |
| 36 | llama-embed-nemotron-8b | `nvidia/llama-embed-nemotron-8b` | 8B | No | Too large |

### F. Encoder-Decoder / OCR (NEW CAPABILITY)

| # | Model | HF Repo | Params | Fits? | Architecture | Notes |
|---|-------|---------|--------|-------|-------------|-------|
| 37 | **Nemotron-Parse-v1.1** | `nvidia/NVIDIA-Nemotron-Parse-v1.1` | <1B | **Yes** | ViT-H encoder + mBart decoder (10 blocks) | Document parsing → Markdown |
| 38 | **nemotron-ocr-v1** | `nvidia/nemotron-ocr-v1` | 52M | **Yes** | RegNetY detector + Transformer recognizer + relational | 3-model OCR pipeline |

### G. Audio / Speech (OUT OF SCOPE — fundamentally different C++ runtime)

| # | Model | HF Repo | Params | Notes |
|---|-------|---------|--------|-------|
| 39 | parakeet-streaming-0.6b | `nvidia/multitalker-parakeet-streaming-0.6b-v1` | 600M | NeMo RNNT ASR — requires audio I/O, spectrogram preprocessing, NeMo checkpoint conversion |
| 40 | magpie-tts-357m | `nvidia/magpie_tts_multilingual_357m` | 357M | NeMo TTS — requires codec generation, waveform synthesis |
| 41 | audio-flamingo-3 | `nvidia/audio-flamingo-3-hf` | ~8B | Too large + audio encoder |

### H. Not Compatible / Out of Scope

| # | Model | Why |
|---|-------|-----|
| Eagle2/Eagle3 models | Excluded per user request |
| DeepSeek-V3.x-NVFP4 (394B) | Quantized format, too large |
| Qwen3-235B-A22B-Eagle3 | MoE, too large |
| gpt-oss-120b-Eagle3-* | Speculative decode infra |
| Qwen3-8B-DMS-8x | Speculative decode variant |
| KVzap-* | KV compression utilities, not standalone models |
| All molecular/protein/weather/medical/robotics models | Non-ML-inference models |

---

## Models That Fit on 24GB — Complete Summary

| # | Model | Category | Work Required | Effort |
|---|-------|----------|--------------|--------|
| 1-4 | 4 LLaMA/Mistral NVIDIA fine-tunes | Text Gen | Validation only | ~2h |
| 14-15 | Nemotron-Mini-4B + Hindi-4B | Text Gen | New activation + plugin | ~1 day |
| 18 | Cosmos-Reason2-2B | VL | Validate qwen_vl for Qwen3-VL | ~1-2 days |
| 27-30 | Hymba-1.5B + Nemotron-Flash-1B/3B | Hybrid | New C++ hybrid backend | ~3 weeks |
| 33-34 | Embed-1b + Rerank-1b | Embedding | New C++ embedding runtime | ~1-2 weeks |
| 37-38 | Nemotron-Parse + OCR | Enc-Dec | New C++ enc-dec runtime | ~2-3 weeks |

**Total: 14 models across 5 phases. ~2-3 months of work.**

---

## Phase 1: Nemotron-4 Plugin + LLaMA/Mistral Validation (~1 day)

### 1.1: Add `relu2`/`squared_relu` activation

**File:** `trtf_build/trtf_build/graph_ops.py` (lines 305-322)

New `elif` in `add_activation()`:
```python
elif activation_type in ("relu2", "squared_relu"):
    relu = network.add_activation(inp, trt.ActivationType.RELU)
    sq = network.add_elementwise(relu.get_output(0), relu.get_output(0), trt.ElementWiseOperation.PROD)
    return sq.get_output(0)
```

### 1.2: Add `norm_eps` to config

**File:** `trtf_build/trtf_build/config.py` (line 70-74) — add `d.get("norm_eps")` to epsilon chain.

### 1.3: Create Nemotron family plugin

**New file:** `trtf_build/trtf_build/families/nemotron.py` (~120 lines)

Architecture: LayerNorm1P (+1 gamma) + 2-projection MLP (`gelu_fc`, `relu2`) + GQA + partial RoPE.
Combines patterns from: **Gemma** (+1 norms), **OPT** (LayerNorm+bias, gelu_fc), **StableLM** (GQA, partial_rotary).

### 1.4: E2E entries + validation

Add 6 models to `engines.json`: nemotron-mini-4b, nemotron-hindi-4b, nemotron-nano-4b, minitron-4b-depth, minitron-4b-width, riva-translate-4b.

### 1.5: Unit test + docs

Add `relu2`/`squared_relu` to `tests/builder/test_graph_ops.py`. Update wiki docs + WORKLOG.

### Files changed
- `trtf_build/trtf_build/graph_ops.py` — add activation
- `trtf_build/trtf_build/config.py` — add norm_eps
- `trtf_build/trtf_build/standard_decoder_builder.py` — update docstring
- `trtf_build/trtf_build/families/nemotron.py` — **new**
- `tests/builder/test_graph_ops.py` — add parametrize
- `tests/e2e/engines.json` — add 6 entries
- `docs/wiki/Home.md`, `Architecture-Extensibility-Assessment.md`, `WORKLOG.md`

---

## Phase 2: Cosmos-Reason2-2B VL Validation (~1-2 days)

Cosmos-Reason2-2B uses `model_type: qwen3_vl`. The existing `qwen_vl.py` matches `"qwen" in mt and "vl" in mt`. Attempt build with existing plugin; fix any Qwen3-VL vs 2.5-VL vision encoder differences.

### Files potentially changed
- `trtf_build/trtf_build/families/qwen_vl.py` — possible minor updates
- `trtf_build/trtf_build/qwen_vl_vision_builder.py` — if Qwen3-VL vision differs
- `tests/e2e/engines.json` — add 1 VL entry

---

## Phase 3: Embedding / Reranking Runtime (~1-2 weeks)

**This is a new runtime capability.** Both models use LLaMA 3.2 1B as their backbone but instead of autoregressive generation, they do a single forward pass and pool the hidden states.

### 3.1: New runtime strategy: `embedding`

The key difference from text generation:
- **No autoregressive loop** — single forward pass over all input tokens
- **Mean pooling** of last hidden states (masked by attention mask)
- **L2 normalization** of output
- **Return embedding vector** instead of generated text

### 3.2: New C++ files

| File | Purpose |
|------|---------|
| `src/runtime/trt/embedding_backend.h/cpp` | `EmbeddingBackend` — single forward pass, pool hidden states, normalize |

The TRT engine is the same decoder engine (built with `build_standard_decoder_engine` but configured to output hidden states instead of just logits). The `EmbeddingBackend`:
1. Tokenizes input
2. Runs single TRT inference (all tokens at once, not autoregressive)
3. Mean-pools the hidden state outputs
4. L2-normalizes the result
5. Returns float vector

### 3.3: New C ABI functions

```c
// In include/trtf/trtf.h
TrtfEmbedder* trtf_create_embedder(const char* bundle_path, const TrtfEmbedderOptions* options);
int trtf_embed(TrtfEmbedder* embedder, const char* text, float* output, int output_dim);
void trtf_destroy_embedder(TrtfEmbedder* embedder);

// For reranker
TrtfReranker* trtf_create_reranker(const char* bundle_path, const TrtfRerankerOptions* options);
float trtf_rerank(TrtfReranker* reranker, const char* query, const char* document);
void trtf_destroy_reranker(TrtfReranker* reranker);
```

### 3.4: Python builder changes

The embedding/reranking models are LLaMA 3.2 1B — the `llama.py` plugin already works for weight loading. The difference is in the engine build: we need the engine to output hidden states (not just logits) and handle batch input (not single-token step).

New `runtime_strategy` values: `"embedding"`, `"reranking"`

### 3.5: CLI extensions

```bash
trtf embed <bundle.trtfb> --text "query text" [--dim 2048]
trtf rerank <bundle.trtfb> --query "question" --document "passage"
```

### 3.6: E2E entries

Add `llama-nemotron-embed-1b` and `llama-nemotron-rerank-1b` to `engines.json` with `runtime_strategy: "embedding"` / `"reranking"`.

### Files changed
- `src/runtime/trt/embedding_backend.h/cpp` — **new**
- `src/cabi/trtf_c.cpp` — add embed/rerank dispatch
- `include/trtf/trtf.h` — new C ABI functions
- `src/cabi/fast_path_config.h/cpp` — parse new strategies
- `CMakeLists.txt` — add sources
- `trtf_build/trtf_build/families/llama_embed.py` — **new** (or extend llama.py with embedding mode)
- `trtf_build/trtf_build/standard_decoder_builder.py` — add option to output hidden states
- `trtf_build/trtf_build/debug_runner.py` — add `EmbeddingTrtRunner`, `RerankerTrtRunner`
- `tools/diff_embeddings.py` — **new** diff tool
- `tools/diff_reranker.py` — **new** diff tool
- `tests/e2e/test_full_pipeline_embed.py` — **new** E2E test
- `tests/e2e/test_embed_parity.py` — **new** parity test
- CLI source — add `embed` and `rerank` subcommands

---

## Phase 4: Hybrid Backend — Hymba + Nemotron-Flash (~3 weeks)

### 4.1: New C++ files

| File | Purpose |
|------|---------|
| `src/runtime/trt/hybrid_step_state.h/cpp` | `HybridStepState` — KV cache + SSM recurrent state per layer |
| `src/runtime/trt/hybrid_decode_runtime.h/cpp` | `run_hybrid_step()` — binds both cache types |
| `src/runtime/trt/hybrid_backend.h/cpp` | `HybridBackendFastPath` — autoregressive loop |

### 4.2: Hymba specifics
- 32 layers: 3 full attention + 29 sliding window
- Parallel attention + Mamba SSM heads per layer (outputs summed)
- 128 learnable meta-tokens prepended to prompt
- Cross-layer KV sharing (adjacent layer pairs)
- New `runtime_strategy: "hybrid_attention_ssm"`

### 4.3: Nemotron-Flash specifics
- Per-layer operator type discovered by evolutionary search
- Layers can be: attention / Mamba / linear-attention / MLP-only
- Same hybrid C++ backend, different per-layer configuration
- May need Mamba-2 SSD ops if different from Mamba-1

### 4.4: New Python files
- `trtf_build/trtf_build/families/hymba.py` — parallel attn+SSM graph per layer
- `trtf_build/trtf_build/families/nemotron_flash.py` — evolved hybrid layers
- `trtf_build/trtf_build/debug_runner.py` — add `HybridTrtRunner`

### 4.5: Modified C++ files
- `src/cabi/fast_path_config.h/cpp` — hybrid strategy parsing
- `src/cabi/trtf_c.cpp` — hybrid dispatch
- `CMakeLists.txt` — add sources

### 4.6: Diff tool extensions
- `tools/diff_logits.py` — add `HybridTrtRunner` dispatch for hybrid model_types
- `tools/diff_layers.py` — add hybrid debug engine support (parallel attn+SSM hidden states)
- `trtf_build/trtf_build/debug_runner.py` — add `HybridTrtRunner` class
- `tests/e2e/test_full_pipeline_hybrid.py` — **new** E2E test for hybrid models

### 4.7: E2E entries
Add hymba-1.5b, nemotron-flash-1b, nemotron-flash-3b, nemotron-flash-3b-instruct.

---

## Phase 5: Encoder-Decoder Runtime — Nemotron-Parse + OCR (~2-3 weeks)

**This is a new architectural capability.** Encoder-decoder models have:
- An encoder that processes input once (image/text → features)
- A decoder that generates output autoregressively, with cross-attention to encoder output

### 5.1: Nemotron-Parse-v1.1 (<1B)

Architecture: **ViT-H (C-RADIO) encoder** → 1D conv adapter (13,184→3,201 tokens) → **mBart decoder** (10 blocks)

Input: Document image → Output: Markdown/LaTeX text

This is similar to the existing VL pipeline but with an encoder-decoder rather than decoder-only:
- Vision encoder runs once → features
- Adapter compresses features
- mBart decoder generates text autoregressively, attending to encoder features via cross-attention

### 5.2: nemotron-ocr-v1 (52M)

Architecture: 3-stage pipeline:
1. **RegNetY-8GF detector** (45M) — text region localization
2. **Transformer recognizer** (5M) — text transcription per region
3. **Relational model** (2M) — layout analysis + reading order

This is a multi-model pipeline, each stage gets its own TRT engine in the bundle.

### 5.3: New C++ files

| File | Purpose |
|------|---------|
| `src/runtime/trt/encoder_decoder_backend.h/cpp` | `EncoderDecoderBackend` — encoder runs once, decoder autoregressively with cross-attn |
| `src/runtime/trt/ocr_backend.h/cpp` | `OCRBackend` — 3-stage detection → recognition → layout pipeline |

### 5.4: New runtime strategies
- `"encoder_decoder"` for Nemotron-Parse
- `"ocr_pipeline"` for nemotron-ocr-v1

### 5.5: New Python builders
- `trtf_build/trtf_build/families/nemotron_parse.py` — ViT-H + mBart builder
- `trtf_build/trtf_build/families/nemotron_ocr.py` — 3-model pipeline builder
- Need new graph ops for encoder-decoder cross-attention (mBart uses standard multi-head cross-attn)

### 5.6: Bundle format extension
Bundle needs to hold multiple TRT engines:
- For Parse: vision_encoder_plan + decoder_plan
- For OCR: detector_plan + recognizer_plan + relational_plan
(Bundle format already supports multiple sections — used by VL pipeline)

### 5.7: Diff tools
- `tools/diff_enc_dec.py` — **new**: encoder feature parity + decoder generation parity + text metrics
- `tools/diff_ocr.py` — **new**: per-stage detection/recognition/layout comparison
- `trtf_build/trtf_build/debug_runner.py` — add `EncoderDecoderTrtRunner`, `OCRTrtRunner`
- `tests/e2e/test_full_pipeline_enc_dec.py` — **new** E2E test

### 5.8: CLI extensions
```bash
trtf parse <bundle.trtfb> --image document.pdf
trtf ocr <bundle.trtfb> --image document.jpg
```

---

---

## Comprehensive Validation & Diff Testing Plan

### Existing diff tools (reference)
- `tools/diff_logits.py` — per-step logit comparison: TRT Python runner vs HF `AutoModelForCausalLM`. Supports `--battery` (4 diverse prompts), `--atol`. Uses `TrtRunner` or `MambaTrtRunner` from `debug_runner.py`.
- `tools/diff_layers.py` — per-layer hidden state comparison via `debug_layer_outputs=True` engine builds.
- `tools/diff_vl.py` — VL pipeline: vision feature cosine similarity, text generation comparison, C++ binary parity.
- `tools/test_runner_parity.py` — Python debug runner vs C++ binary token-by-token comparison.

### Phase 1 Validation: Nemotron-4 (standard decoder — existing tools work)

**Unit tests (no GPU):**
```bash
pytest tests/builder/ -v  # Includes new relu2/squared_relu parametrization
```

**Per-model diff validation (in container, ~30 min each):**
```bash
# Nemotron-Mini-4B (new plugin, new activation)
python3 tools/diff_logits.py --model nvidia/Nemotron-Mini-4B-Instruct --atol 1e-3 --battery --max-cache-length 256
python3 tools/diff_layers.py --model nvidia/Nemotron-Mini-4B-Instruct --atol 0.05
python3 tools/test_runner_parity.py --bundle /mnt/storage/trt-transformers/engines/nemotron-mini-4b.trtfb \
    --binary ./build/trtf --hf-python .venv/bin/python --max-new-tokens 20

# Or use the all-in-one validation gate:
./scripts/validate_family.sh nvidia/Nemotron-Mini-4B-Instruct --max-cache-length 256 --bundle-dir /mnt/storage/trt-transformers/engines
./scripts/validate_family.sh nvidia/Nemotron-4-Mini-Hindi-4B-Base --max-cache-length 256 --bundle-dir /mnt/storage/trt-transformers/engines

# Group A models (existing llama/mistral plugins)
./scripts/validate_family.sh nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1 --max-cache-length 256 --bundle-dir /mnt/storage/trt-transformers/engines
./scripts/validate_family.sh nvidia/Llama-3.1-Minitron-4B-Depth-Base --max-cache-length 256 --bundle-dir /mnt/storage/trt-transformers/engines
./scripts/validate_family.sh nvidia/Llama-3.1-Minitron-4B-Width-Base --max-cache-length 256 --bundle-dir /mnt/storage/trt-transformers/engines
./scripts/validate_family.sh nvidia/Riva-Translate-4B-Instruct-v1.1 --max-cache-length 256 --bundle-dir /mnt/storage/trt-transformers/engines
```

**What's validated:** Per-step logit diff (atol < 1e-3), per-layer hidden state diff (atol < 0.05), token-exact C++ vs Python parity, 4 diverse prompt types.

### Phase 2 Validation: Cosmos-Reason2-2B VL

**Extend `diff_vl.py`** to handle Qwen3-VL models:
- Current `diff_vl.py` imports `Qwen2_5_VLForConditionalGeneration` hardcoded — need to add Qwen3-VL support via `AutoModelForImageTextToText` (already partially done in `diff_logits.py`).

**Validation steps:**
```bash
# 1. Build bundle
trtf-build build nvidia/Cosmos-Reason2-2B -o /mnt/storage/trt-transformers/engines/cosmos-reason2-2b.trtfb --max-cache-length 256

# 2. Vision encoder feature comparison (TRT vision engine vs HF)
python3 tools/diff_vl.py --bundle /mnt/storage/trt-transformers/engines/cosmos-reason2-2b.trtfb \
    --image /mnt/storage/trt-transformers/test_image.jpg \
    --model nvidia/Cosmos-Reason2-2B --atol 0.1
# Expected: cosine_similarity ≥ 0.999 for vision features

# 3. Vision-only smoke test (no HF model needed)
python3 tools/diff_vl.py --bundle /mnt/storage/trt-transformers/engines/cosmos-reason2-2b.trtfb \
    --image /mnt/storage/trt-transformers/test_image.jpg --vision-only

# 4. Full VL generation + C++ parity
python3 tools/diff_vl.py --bundle /mnt/storage/trt-transformers/engines/cosmos-reason2-2b.trtfb \
    --image /mnt/storage/trt-transformers/test_image.jpg \
    --model nvidia/Cosmos-Reason2-2B \
    --binary ./build/trtf --hf-python .venv/bin/python
```

**What's validated:** Vision feature cosine similarity, VL text generation match, C++ binary parity.

### Phase 3 Validation: Embedding & Reranking

**New diff tools needed** — these models don't generate text, so diff_logits.py doesn't apply.

#### New tool: `tools/diff_embeddings.py`

Compares TRT embedding output vs HuggingFace reference for text embedding models.

```bash
python3 tools/diff_embeddings.py \
    --model nvidia/llama-nemotron-embed-1b-v2 \
    --bundle /mnt/storage/trt-transformers/engines/embed-1b.trtfb \
    --atol 1e-3 --battery
```

**Battery test queries:**
```python
EMBEDDING_BATTERY = [
    ("short_query", "query: What is machine learning?"),
    ("long_query", "query: Explain the difference between supervised and unsupervised learning in detail."),
    ("passage", "passage: Machine learning is a subset of artificial intelligence that enables systems to learn from data."),
    ("code", "passage: def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)"),
]
```

**Comparison metrics:**
1. **Cosine similarity** per query: TRT embed vs HF embed (must be ≥ 0.9999)
2. **Max absolute diff** per dimension (must be ≤ atol)
3. **L2 distance** between normalized embeddings (must be ≤ 0.01)
4. **Rank preservation**: Given N queries + N documents, TRT and HF must produce same ranking

**Implementation pattern:**
```python
def run_hf_embedding(model_id, texts):
    from transformers import AutoModel, AutoTokenizer
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    inputs = tokenizer(texts, padding=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    # Average pool with attention mask
    embeddings = average_pool(outputs.last_hidden_state, inputs["attention_mask"])
    return F.normalize(embeddings, dim=-1).numpy()

def run_trt_embedding(bundle_path, texts):
    from trtf_build.debug_runner import EmbeddingTrtRunner  # NEW
    runner = EmbeddingTrtRunner(bundle_path)
    return runner.embed(texts)  # Returns (N, dim) numpy array
```

#### New tool: `tools/diff_reranker.py`

Compares TRT reranking scores vs HuggingFace reference.

```bash
python3 tools/diff_reranker.py \
    --model nvidia/llama-nemotron-rerank-1b-v2 \
    --bundle /mnt/storage/trt-transformers/engines/rerank-1b.trtfb \
    --atol 0.1
```

**Test pairs:**
```python
RERANK_BATTERY = [
    {"query": "How much protein should a female eat?",
     "docs": [
         "As a general guideline, the CDC recommends 46g of protein per day for women.",
         "The weather in Paris is usually mild in spring.",
         "Python is a popular programming language.",
     ]},
    {"query": "What is the capital of France?",
     "docs": [
         "Paris is the capital and largest city of France.",
         "Berlin is the capital of Germany.",
     ]},
]
```

**Comparison metrics:**
1. **Score diff** per pair: |TRT_score - HF_score| (must be ≤ atol)
2. **Rank ordering agreement**: Kendall's tau correlation (must be ≥ 0.99)
3. **Top-1 agreement**: Same document ranked first

#### New Python debug runner classes

**File:** `trtf_build/trtf_build/debug_runner.py`

```python
class EmbeddingTrtRunner:
    """Run embedding inference: single forward pass + mean pooling + L2 norm."""
    def embed(self, texts: list[str]) -> np.ndarray:
        # Tokenize all texts
        # Single forward pass (no autoregressive loop)
        # Mean pool hidden states with attention mask
        # L2 normalize
        return embeddings  # (N, dim)

class RerankerTrtRunner:
    """Run reranking: forward pass + mean pool + classification head."""
    def rerank(self, query: str, documents: list[str]) -> list[float]:
        # Format pairs: "question:{query}\n\npassage:{doc}"
        # Forward pass each pair
        # Mean pool + classification logit
        return scores  # list of float logits
```

#### Runner parity for embedding/reranking

```bash
# Embedding: TRT Python runner vs C++ binary
python3 tools/test_embed_parity.py \
    --bundle /mnt/storage/trt-transformers/engines/embed-1b.trtfb \
    --binary ./build/trtf --hf-python .venv/bin/python \
    --text "query: test sentence"

# Reranking: TRT Python runner vs C++ binary
python3 tools/test_rerank_parity.py \
    --bundle /mnt/storage/trt-transformers/engines/rerank-1b.trtfb \
    --binary ./build/trtf --hf-python .venv/bin/python \
    --query "What is ML?" --document "Machine learning is..."
```

### Phase 4 Validation: Hybrid Models (Hymba, Nemotron-Flash)

**Extend `diff_logits.py`** to dispatch to `HybridTrtRunner`:

```python
# In diff_logits.py:run_trt(), add:
elif config.model_type.lower() == "hymba" or is_hybrid_model(config):
    from trtf_build.debug_runner import HybridTrtRunner
    runner = HybridTrtRunner(
        engine_plan=engine_plan,
        max_cache_length=max_cache_length,
        num_layers=config.num_hidden_layers,
    )
```

The HF side already works via `trust_remote_code=True` — HF loads the hybrid model and runs it natively.

**Validation steps:**
```bash
# 1. Build bundle
trtf-build build nvidia/Hymba-1.5B-Base -o /mnt/storage/trt-transformers/engines/hymba-1.5b.trtfb \
    --max-cache-length 256 --verbose

# 2. Per-step logit comparison (TRT HybridTrtRunner vs HF)
python3 tools/diff_logits.py --model nvidia/Hymba-1.5B-Base \
    --atol 1e-3 --battery --max-cache-length 256 --trust-remote-code

# 3. Per-layer hidden states (attention path + SSM path outputs)
python3 tools/diff_layers.py --model nvidia/Hymba-1.5B-Base \
    --atol 0.05 --trust-remote-code

# 4. C++ binary parity
python3 tools/test_runner_parity.py \
    --bundle /mnt/storage/trt-transformers/engines/hymba-1.5b.trtfb \
    --binary ./build/trtf --hf-python .venv/bin/python --max-new-tokens 20

# 5. Repeat for Nemotron-Flash variants
python3 tools/diff_logits.py --model nvidia/Nemotron-Flash-1B \
    --atol 1e-3 --battery --trust-remote-code
```

**Special validation concerns for hybrid:**
- Mamba SSM state drift: compare after 50+ tokens (cumulative state errors)
- Meta-token handling (Hymba): verify 128 prepended tokens match HF behavior
- Cross-layer KV sharing: verify shared cache produces same results as independent caches

### Phase 5 Validation: Encoder-Decoder (Parse + OCR)

**New tool: `tools/diff_enc_dec.py`**

For encoder-decoder models, the comparison is:
1. **Encoder output parity** — TRT vision encoder vs HF vision encoder
2. **Decoder generation parity** — TRT autoregressive decoder vs HF decoder (both conditioned on same encoder output)

```bash
# Nemotron-Parse validation
python3 tools/diff_enc_dec.py \
    --model nvidia/NVIDIA-Nemotron-Parse-v1.1 \
    --bundle /mnt/storage/trt-transformers/engines/nemotron-parse.trtfb \
    --image test_document.jpg --atol 0.1
```

**Comparison metrics:**
1. **Encoder features:** cosine similarity (≥ 0.999)
2. **Decoder logits:** per-step max absolute diff (≤ atol), same as diff_logits
3. **Generated text:** exact match or BLEU score (≥ 0.95 for structured output)
4. **For Parse:** structural accuracy — does the generated Markdown have correct headings, tables, equations?

**New tool: `tools/diff_ocr.py`**

Multi-stage pipeline validation for nemotron-ocr-v1:

```bash
python3 tools/diff_ocr.py \
    --model nvidia/nemotron-ocr-v1 \
    --bundle /mnt/storage/trt-transformers/engines/nemotron-ocr.trtfb \
    --image test_document.jpg
```

**Per-stage comparison:**
1. **Detection:** Bounding box IoU (≥ 0.95 per region), precision/recall
2. **Recognition:** Character Error Rate per region (CER ≤ 1%)
3. **Layout:** Reading order accuracy (Kendall's tau ≥ 0.99)
4. **Full pipeline:** Final Markdown output text diff

### E2E Test Suite Structure

```
tests/e2e/
  engines.json                    # All models (19 existing + 14 NVIDIA)
  conftest.py                     # Fixtures for all model types
  test_full_pipeline.py           # Build + infer + diff (text gen + VL)
  test_full_pipeline_embed.py     # NEW: Build + embed + diff (embedding/reranking)
  test_full_pipeline_hybrid.py    # NEW: Build + infer + diff (hybrid models)
  test_full_pipeline_enc_dec.py   # NEW: Build + infer + diff (encoder-decoder)
  test_inference.py               # C++ binary smoke test
  test_logit_parity.py            # Logit diff for decoders
  test_runner_parity.py           # Python vs C++ parity
  test_bundle_inspect.py          # Bundle metadata
  test_vl_pipeline.py             # VL-specific tests
  test_embed_parity.py            # NEW: Embedding parity
  test_rerank_parity.py           # NEW: Reranking parity
```

### `engines.json` Schema Extension

New fields for non-decoder models:

```json
{
    "name": "llama-nemotron-embed-1b",
    "hf_id": "nvidia/llama-nemotron-embed-1b-v2",
    "bundle": "llama-nemotron-embed-1b.trtfb",
    "family": "llama_embed",
    "runtime_strategy": "embedding",
    "embed_dim": 2048,
    "test_queries": ["query: What is machine learning?"],
    "test_documents": ["passage: Machine learning is a subset of AI."],
    "embed_cosine_threshold": 0.9999,
    "trust_remote_code": true
}
```

For OCR:
```json
{
    "name": "nemotron-ocr",
    "hf_id": "nvidia/nemotron-ocr-v1",
    "bundle": "nemotron-ocr.trtfb",
    "family": "nemotron_ocr",
    "runtime_strategy": "ocr_pipeline",
    "test_image": "/mnt/storage/trt-transformers/test_document.jpg",
    "detection_iou_threshold": 0.95,
    "recognition_cer_threshold": 0.01,
    "trust_remote_code": true
}
```

---

## E2E Regression: Final Model Count

Current: 19 → After all phases: **~33 models** (+14 NVIDIA)

| Phase | New E2E Entries |
|-------|----------------|
| 1 | nemotron-mini-4b, nemotron-hindi-4b, nemotron-nano-4b, minitron-4b-depth, minitron-4b-width, riva-translate-4b |
| 2 | cosmos-reason2-2b |
| 3 | llama-nemotron-embed-1b, llama-nemotron-rerank-1b |
| 4 | hymba-1.5b, nemotron-flash-1b, nemotron-flash-3b, nemotron-flash-3b-instruct |
| 5 | nemotron-parse, nemotron-ocr |

---

## Effort Summary

| Phase | Scope | Effort | Key C++ Changes |
|-------|-------|--------|----------------|
| **1** | Nemotron-4 + validation | **~1 day** | None |
| **2** | Cosmos-Reason2-2B VL | **~1-2 days** | None |
| **3** | Embedding + Reranking | **~1-2 weeks** | New `EmbeddingBackend`, C ABI extensions |
| **4** | Hybrid (Hymba + Flash) | **~3 weeks** | New `HybridBackend`, `HybridStepState` |
| **5** | Encoder-Decoder (Parse + OCR) | **~2-3 weeks** | New `EncoderDecoderBackend`, `OCRBackend` |
| **Total** | 14 NVIDIA models | **~2-3 months** | 4 new C++ backends |
