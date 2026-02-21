# Scalable Parallel Model Family Implementation Plan

## Goal

Run N Claude Code subagents in parallel, each in its own git worktree + Docker container pinned to a dedicated B300 GPU, to implement and validate N new model families simultaneously. Target: 8 parallel slots (1 per GPU), scalable to 16 for small models by co-locating 2 on one GPU.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│  Orchestrator (this session)                                         │
│  - Creates worktrees, launches containers, dispatches subagents      │
│  - Collects results, merges branches                                 │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Slot 0 (GPU 0)          Slot 1 (GPU 1)         ...   Slot 7 (GPU 7)│
│  ┌──────────────┐       ┌──────────────┐              ┌────────────┐│
│  │ worktree-0   │       │ worktree-1   │              │ worktree-7 ││
│  │ branch:      │       │ branch:      │              │ branch:    ││
│  │  family/yi   │       │  family/dbrx │              │  family/X  ││
│  │              │       │              │              │            ││
│  │ container:   │       │ container:   │              │ container: ││
│  │  trtf-slot-0 │       │  trtf-slot-1 │              │  trtf-slot7││
│  │  --gpus '"0"'│       │  --gpus '"1"'│              │  --gpus "7"││
│  │              │       │              │              │            ││
│  │ subagent:    │       │ subagent:    │              │ subagent:  ││
│  │  scaffold →  │       │  scaffold →  │              │  scaffold →││
│  │  customize → │       │  customize → │              │  customize→││
│  │  validate →  │       │  validate →  │              │  validate →││
│  │  commit      │       │  commit      │              │  commit    ││
│  └──────────────┘       └──────────────┘              └────────────┘│
│                                                                      │
│  (After all slots complete)                                          │
│  - Review each branch's diff                                         │
│  - Merge passing families into master                                │
│  - Clean up worktrees and containers                                 │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Isolation Guarantees

| Resource | Isolation | How |
|----------|-----------|-----|
| Source code | Per-worktree branch | `git worktree add -b family/<name>` |
| Build artifacts | Per-worktree `build/` | gitignored, CMake is relocatable |
| Python venv | Per-worktree `.venv/` | Editable install points to that worktree |
| Engine bundles | Per-slot dir | `/mnt/storage/.../engines-slot-N/` |
| GPU | Pinned per container | `docker run --gpus "device=N"` |
| HF model cache | Shared (read-safe) | `~/.cache/huggingface` — immutable after download |

**Why this is safe:**
- CMake uses `PROJECT_SOURCE_DIR`/`PROJECT_BINARY_DIR` — fully relocatable
- `TRTF_SOURCE_DIR` compile-time define auto-points to each worktree
- `families/__init__.py` auto-discovers plugins via `pkgutil.iter_modules()` — zero shared-file edits needed
- Each family adds only new files (`families/<name>.py` + `tests/e2e/models/<name>.json`) — merge conflicts impossible

---

## Phase 1: Infrastructure — Scripts to Create

### 1a. `scripts/parallel_family_setup.sh` — Per-slot setup

Creates one isolated workspace for a single model family:

```bash
#!/usr/bin/env bash
# Usage: ./scripts/parallel_family_setup.sh <slot_id> <family_name> <gpu_id>
#
# Creates:
#   .claude/worktrees/family-<family_name>/   (git worktree)
#   Container: trtf-slot-<slot_id>            (pinned to GPU <gpu_id>)
#   Separate .venv inside the worktree
#   Separate build/ inside the worktree
#   Separate engine dir: /mnt/storage/trt-transformers/engines-slot-<slot_id>/

set -euo pipefail

SLOT_ID=$1; FAMILY=$2; GPU_ID=$3
REPO_ROOT=$(git rev-parse --show-toplevel)
WORKTREE_DIR="$REPO_ROOT/.claude/worktrees/family-${FAMILY}"
ENGINE_DIR="/mnt/storage/trt-transformers/engines-slot-${SLOT_ID}"
CONTAINER="trtf-slot-${SLOT_ID}"
STORAGE_ROOT="/mnt/storage/trt-transformers"
HF_CACHE="${STORAGE_ROOT}/model-weights"

# 1. Create git worktree on a new branch
git worktree add -b "family/${FAMILY}" "$WORKTREE_DIR" HEAD

# 2. Create per-slot engine directory
mkdir -p "$ENGINE_DIR" "$HF_CACHE"

# 3. Launch container pinned to one GPU, mounting the worktree
docker run -d \
  --gpus "\"device=${GPU_ID}\"" \
  -v "$WORKTREE_DIR":/workspace/trt-transformers-cpp \
  -v "${STORAGE_ROOT}:${STORAGE_ROOT}" \
  -v "$ENGINE_DIR":/mnt/storage/trt-transformers/engines \
  -v "${HF_CACHE}":/root/.cache/huggingface/hub \
  -w /workspace/trt-transformers-cpp \
  --name "$CONTAINER" \
  trtf-dev-gb300 sleep infinity

# 4. Run setup inside container (creates .venv, builds C++)
docker exec "$CONTAINER" bash ./scripts/setup_gb300.sh

echo "Slot $SLOT_ID ready: family=$FAMILY gpu=$GPU_ID container=$CONTAINER"
```

### 1b. `scripts/parallel_family_teardown.sh` — Per-slot cleanup

```bash
#!/usr/bin/env bash
# Usage: ./scripts/parallel_family_teardown.sh <slot_id> <family_name>
set -euo pipefail

SLOT_ID=$1; FAMILY=$2
CONTAINER="trtf-slot-${SLOT_ID}"
WORKTREE_DIR="$(git rev-parse --show-toplevel)/.claude/worktrees/family-${FAMILY}"

docker stop "$CONTAINER" 2>/dev/null && docker rm "$CONTAINER" 2>/dev/null
echo "Container $CONTAINER stopped."

# Worktree kept for review; prune after merge:
# git worktree remove "$WORKTREE_DIR"
echo "Worktree at $WORKTREE_DIR preserved for review."
echo "To remove: git worktree remove $WORKTREE_DIR"
```

### 1c. `scripts/parallel_dispatch.sh` — Orchestrator

Reads a JSON manifest of families to implement and dispatches all slots:

```bash
#!/usr/bin/env bash
# Usage: ./scripts/parallel_dispatch.sh <families_manifest.json>
#
# Phases:
#   1. Pre-download HF models (sequential, avoids rate limits)
#   2. Setup all slots in parallel (worktree + container + venv + build)
#   3. Print slot status
set -euo pipefail

MANIFEST="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Phase 1: Pre-download HF models ==="
for entry in $(jq -c '.[]' "$MANIFEST"); do
  HF_REPO=$(echo "$entry" | jq -r '.hf_repo')
  TEST_MODEL=$(echo "$entry" | jq -r '.test_model')
  echo "  Downloading $HF_REPO ..."
  python3 -c "from huggingface_hub import snapshot_download; snapshot_download('$HF_REPO')" || true
  if [ "$TEST_MODEL" != "$HF_REPO" ]; then
    echo "  Downloading $TEST_MODEL ..."
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download('$TEST_MODEL')" || true
  fi
done

echo ""
echo "=== Phase 2: Setup slots in parallel ==="
PIDS=()
for entry in $(jq -c '.[]' "$MANIFEST"); do
  SLOT=$(echo "$entry" | jq -r '.slot')
  GPU=$(echo "$entry" | jq -r '.gpu')
  FAMILY=$(echo "$entry" | jq -r '.family')

  echo "  Starting slot $SLOT: family=$FAMILY gpu=$GPU"
  "$SCRIPT_DIR/parallel_family_setup.sh" "$SLOT" "$FAMILY" "$GPU" &
  PIDS+=($!)
done

echo "  Waiting for all slots..."
FAILED=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    FAILED=$((FAILED + 1))
  fi
done

echo ""
echo "=== Phase 3: Status ==="
for entry in $(jq -c '.[]' "$MANIFEST"); do
  SLOT=$(echo "$entry" | jq -r '.slot')
  FAMILY=$(echo "$entry" | jq -r '.family')
  CONTAINER="trtf-slot-${SLOT}"
  STATUS=$(docker inspect -f '{{.State.Status}}' "$CONTAINER" 2>/dev/null || echo "not found")
  echo "  Slot $SLOT ($FAMILY): container=$STATUS"
done

if [ "$FAILED" -gt 0 ]; then
  echo "WARNING: $FAILED slot(s) failed setup."
  exit 1
fi
echo "All slots ready. Launch subagents now."
```

---

## Phase 2: Subagent Design — What Each Agent Does

Each Claude Code subagent (launched via `Task` tool with `isolation: "worktree"`) runs autonomously in its worktree and executes this pipeline:

### Step-by-step per-agent workflow

1. **Scaffold plugin** — Run `scripts/new_family.py` to generate the initial plugin file
2. **Study the HF model** — Read config.json, identify weight naming, attention pattern, norm type, activation function, tied embeddings, etc.
3. **Customize `load_weights()`** — Map HF weight keys to trtf's expected naming (study existing similar plugins for patterns)
4. **Customize `build_engine()`** — If non-standard arch (LayerNorm instead of RMSNorm, different activation, etc.), customize the graph builder
5. **Add E2E test manifest** — Create `tests/e2e/models/<family>.json`
6. **Validate (Tier 1)** — Run unit tests to ensure no import errors
7. **Validate (Tier 2)** — Run `validate_family.sh <test_model>` inside the container (builds bundle, diff_logits, diff_layers, runner parity)
8. **Iterate on failures** — Fix weight mapping / graph construction errors based on diff output
9. **Commit** — Create a clean commit on the `family/<name>` branch

### Subagent prompt template

```
You are implementing a new model family plugin for the trt-transformers-cpp project.

Family: {family_name}
Model type (HF config.json): {model_type}
Reference HF repo: {hf_repo}
Small test model: {test_model}

Container: trtf-slot-{slot_id}
GPU: {gpu_id}

Your worktree is at: .claude/worktrees/family-{family_name}/
Your branch is: family/{family_name}

Instructions:
1. Run: docker exec trtf-slot-{slot_id} python3 scripts/new_family.py \
     --model-type {model_type} --hf-repo {hf_repo} --family-name {family_name}
2. Study the generated plugin and the HF model's config.json + weight names
   (use: docker exec trtf-slot-{slot_id} python3 -c "from huggingface_hub import hf_hub_download; ...")
3. Study similar existing plugins for patterns:
   - qwen.py: standard decoder (simplest)
   - phi.py: fused QKV/gate_up weight splitting
   - bloom.py: non-standard LayerNorm + ALiBi
   - gemma.py: +1.0 RMSNorm gamma + embedding scaling
4. Customize the plugin's load_weights() and build_engine() as needed
5. Create tests/e2e/models/{test_name}.json manifest:
   {
     "name": "{family}-{size}",
     "hf_id": "{test_model}",
     "bundle": "{family}-{size}.trtfb",
     "max_cache_length": 256,
     "prompt": "The capital of France is",
     "expected_prefix": "<expected output>",
     "atol": 1e-3
   }
6. Validate:
   docker exec trtf-slot-{slot_id} bash -c \
     'source .venv/bin/activate && ./scripts/validate_family.sh {test_model}'
7. Fix any failures (iterate steps 3-6 until all 4 validation steps pass)
8. Commit changes on this branch with message: "feat: add {family_name} model family"
```

---

## Phase 3: Scaling Strategy — 8 to 16 Slots

### Option A: 1 GPU = 1 slot (8 slots total) — Default

Best for models > 1B parameters where TRT engine building + validation consumes most of a GPU's memory.

### Option B: 1 GPU = 2 slots (16 slots total) — Small models

For models < 500M params (GPT-2 125M, OPT-125M, XGLM-564M, Pythia-70M, etc.), two containers can share one GPU:

```bash
# Slot 0 and Slot 8 both on GPU 0, but with MPS or serialized access:
docker run ... --gpus "\"device=0\"" --name trtf-slot-0 ...
docker run ... --gpus "\"device=0\"" --name trtf-slot-8 ...
```

TRT engine building is mostly sequential (GPU compute + host memory), so two small-model builds on the same GPU will time-share effectively. The key constraint is GPU memory — two <500M models fit comfortably in a single B300's memory.

### Option C: Adaptive scheduling — Production-grade

A JSON dispatch manifest can specify `gpu_share: true` for small models:

```json
[
  {"slot": 0, "gpu": 0, "family": "yi",       "test_model": "01-ai/Yi-1.5-500M"},
  {"slot": 1, "gpu": 1, "family": "chatglm",  "test_model": "THUDM/chatglm-6b"},
  {"slot": 8, "gpu": 0, "family": "tiny_llama","test_model": "TinyLlama/TinyLlama-1.1B-step-50K-105b", "gpu_share": true},
  ...
]
```

Setup script checks `gpu_share` and sets CUDA_MPS or just lets them time-share.

---

## Phase 4: Orchestrator — Claude Code Integration

From the main Claude Code session, the orchestrator works like this:

```python
# 1. Pre-download all HF models (shared cache, avoids races)
for model in all_test_models:
    docker exec trtf-slot-0 bash -c "source .venv/bin/activate && \
      python3 -c \"from huggingface_hub import snapshot_download; snapshot_download('{model}')\""

# 2. Launch 8 subagents in parallel (single message, 8 Task calls)
#    Each with isolation: "worktree"
Task(subagent_type="Bash", isolation="worktree", prompt="implement family/yi ...")
Task(subagent_type="Bash", isolation="worktree", prompt="implement family/chatglm ...")
...  # 8 total, all in one message -> parallel execution

# 3. Collect results
#    Each agent returns: pass/fail, commit hash, validation log

# 4. Merge passing branches
for family in passing_families:
    git merge family/{family} --no-ff -m "feat: add {family} model family"
```

---

## Phase 5: Candidate Model Families (Not Yet Implemented)

Based on the existing 25 families in `trtf_build/trtf_build/families/`, here are candidates for new families (ordered by implementation difficulty):

### Tier A — Standard decoder, drop-in (expect `new_family.py` scaffold to mostly work)

| # | Family | model_type | Test model | Size | Notes |
|---|--------|-----------|------------|------|-------|
| 1 | **Yi** | `yi` | `01-ai/Yi-1.5-500M` | 500M | LLaMA-like, likely trivial |
| 2 | **Baichuan** | `baichuan` | `baichuan-inc/Baichuan2-7B-Base` | 7B | LLaMA-like with alibi or RoPE |
| 3 | **ChatGLM** | `chatglm` | `THUDM/chatglm3-6b` | 6B | GLM attention variant |
| 4 | **Cohere Command** | `cohere` | `CohereForAI/c4ai-command-r-v01` | 35B | Needs large GPU |
| 5 | **DeepSeek** | `deepseek` | `deepseek-ai/deepseek-llm-7b-base` | 7B | Standard or MoE |

### Tier B — Non-standard, needs customization

| # | Family | model_type | Test model | Notes |
|---|--------|-----------|------------|-------|
| 6 | **RWKV** | `rwkv` | `RWKV/rwkv-4-169m-pile` | Recurrent, needs new C++ backend like Mamba |
| 7 | **Persimmon** | `persimmon` | `adept/persimmon-8b-base` | Non-standard norm |
| 8 | **MPT** | `mpt` | `mosaicml/mpt-7b` | ALiBi position encoding |

### Recommended 8-slot dispatch for first run

Use Tier A models + easy Tier B. Use smallest test models to maximize success rate:

```json
[
  {"slot":0, "gpu":0, "family":"yi",        "model_type":"yi",        "hf_repo":"01-ai/Yi-6B",                       "test_model":"01-ai/Yi-1.5-500M"},
  {"slot":1, "gpu":1, "family":"baichuan",   "model_type":"baichuan",  "hf_repo":"baichuan-inc/Baichuan2-7B-Base",    "test_model":"baichuan-inc/Baichuan2-7B-Base"},
  {"slot":2, "gpu":2, "family":"chatglm",    "model_type":"chatglm",   "hf_repo":"THUDM/chatglm3-6b",                "test_model":"THUDM/chatglm3-6b"},
  {"slot":3, "gpu":3, "family":"deepseek",   "model_type":"deepseek",  "hf_repo":"deepseek-ai/deepseek-llm-7b-base", "test_model":"deepseek-ai/deepseek-llm-7b-base"},
  {"slot":4, "gpu":4, "family":"mpt",        "model_type":"mpt",       "hf_repo":"mosaicml/mpt-7b",                  "test_model":"mosaicml/mpt-7b"},
  {"slot":5, "gpu":5, "family":"persimmon",  "model_type":"persimmon", "hf_repo":"adept/persimmon-8b-base",           "test_model":"adept/persimmon-8b-base"},
  {"slot":6, "gpu":6, "family":"cohere",     "model_type":"cohere",    "hf_repo":"CohereForAI/c4ai-command-r-v01",   "test_model":"CohereForAI/c4ai-command-r-v01"},
  {"slot":7, "gpu":7, "family":"rwkv",       "model_type":"rwkv",      "hf_repo":"RWKV/rwkv-4-169m-pile",            "test_model":"RWKV/rwkv-4-169m-pile"}
]
```

---

## Phase 6: Merge & Regression

After all agents complete:

1. **Review each branch**: `git log --oneline master..family/<name>` + `git diff master..family/<name>`
2. **Run Tier 1 unit tests** on merged result (fast, no GPU)
3. **Run Tier 4 full E2E** with all models (including new families)
4. **Clean up**: `git worktree prune`, remove per-slot engine dirs, stop containers

---

## Files to Create

| File | Purpose |
|------|---------|
| `scripts/parallel_family_setup.sh` | Per-slot: worktree + container + venv + build |
| `scripts/parallel_family_teardown.sh` | Per-slot: stop container, optionally remove worktree |
| `scripts/parallel_dispatch.sh` | Orchestrator: reads JSON, sets up all slots in parallel |
| `scripts/families_manifest.json` | Dispatch manifest: which families on which GPUs |

---

## Key Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| HF Hub rate limiting on 8 concurrent downloads | Pre-download all models sequentially first |
| Worktree `.venv` creation is 4.3 GB x 8 = 34 GB disk | Use `--system-site-packages` or shared base venv + per-worktree editable install overlay |
| Two agents edit `families/__init__.py` | No edits needed — auto-discovery means just drop a `.py` file |
| Container OOM on large models | Use small test models for initial validation; large models later in sequential Tier 4 |
| Subagent gets stuck in iteration loop | Set `max_turns` on Task tool; alert orchestrator on timeout |
| Merge conflicts between family branches | Each family only adds new files (`families/<name>.py` + `tests/e2e/models/<name>.json`), so merge conflicts are impossible unless two touch shared code |

---

## Existing Families (Already Implemented — 25 total)

For reference, these families already exist in `trtf_build/trtf_build/families/`:

bloom, codegen, falcon, gemma, gpt2, gpt_neo, gpt_neox, granite, internlm,
llama, mamba, mistral, mixtral, nemotron, olmo, opt, phi, phi_moe, qwen,
qwen_vl, stablelm, starcoder2, xglm

---

## Summary

1. **Create 3 shell scripts** for setup/teardown/dispatch
2. **Create 1 JSON manifest** listing families -> GPUs
3. **Pre-download models** to shared HF cache
4. **Launch 8 parallel subagents** via Claude Code Task tool, each in its own worktree
5. **Each agent**: scaffold -> customize -> validate -> commit
6. **Orchestrator**: collect results, merge passing branches, run regression
7. **Scale to 16** by co-locating small models (< 500M params) on shared GPUs
