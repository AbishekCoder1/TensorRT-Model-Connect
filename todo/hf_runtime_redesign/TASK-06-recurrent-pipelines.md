# TASK-06: MambaPipeline + RwkvPipeline + HybridPipeline

## Status: blocked (needs TASK-01, TASK-02, TASK-04)
## Phase: 2 (Core pipelines)
## Risk: low — Mamba/RWKV backends are small, clear patterns

## Goal

Implement pipelines for recurrent models. These are structurally identical to
TextGenerationPipeline but use RecurrentState instead of KvCache.
Hybrid uses both (attention layers get KvCache, Mamba layers get RecurrentState).

## Classes

```cpp
class MambaPipeline final : public IPipeline {
    std::unique_ptr<TrtModule> decoder_;
    std::unique_ptr<RecurrentState> state_;  // conv + ssm per layer
    std::unique_ptr<ITokenizer> tokenizer_;
    // generate() = same prefill→decode loop as TextGeneration
};

class RwkvPipeline final : public IPipeline {
    std::unique_ptr<TrtModule> decoder_;
    std::unique_ptr<RecurrentState> state_;  // 5 state vectors per layer
    std::unique_ptr<ITokenizer> tokenizer_;
    // generate() = same loop
};

class HybridPipeline final : public IPipeline {
    std::unique_ptr<TrtModule> decoder_;
    std::unique_ptr<KvCache> kv_cache_;        // for attention layers
    std::unique_ptr<RecurrentState> ssm_state_; // for mamba layers
    std::unique_ptr<ITokenizer> tokenizer_;
    // Both bound to same TrtModule; advance both after each step
};
```

## What it replaces

- `MambaBackendFastPath` (mamba_backend.cpp — 131 lines)
- `RwkvBackendFastPath` (rwkv_backend.cpp — 149 lines)
- `HybridBackend` (hybrid_backend.cpp — 25KB)
- `MambaStepState` / `RwkvStepState` → replaced by RecurrentState
- `run_mamba_step()` / `run_rwkv_step()` → replaced by TrtModule.forward()
- ~200 lines of duplicated prefill/decode helpers across all three

## Files to create

- `src/runtime/pipelines/mamba_pipeline.h` + `.cpp`
- `src/runtime/pipelines/rwkv_pipeline.h` + `.cpp`
- `src/runtime/pipelines/hybrid_pipeline.h` + `.cpp`
- `tests/cpp/test_mamba_pipeline.cpp`
- `tests/cpp/test_rwkv_pipeline.cpp`

## Dependencies

TASK-01 (TrtModule), TASK-02 (RecurrentState, KvCache), TASK-04 (Factory)
