#pragma once

// Lightweight header for model family registrations.
// Provides the full IModelRuntime class definition (needed for unique_ptr)
// without pulling in TRT/CUDA headers — all TRT types are forward-declared.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtf {

#if TRTF_HAS_TRT

// Forward declarations — complete definitions live in trt_engine_lifecycle.h,
// trt_model_definition.h, trt_common.h, and step_state.h respectively.
class DecoderStepEngine;
struct TrtDecoderDefinition;
class TrtLogger;
class IStepState;

class IModelRuntime {
public:
    virtual ~IModelRuntime() = default;

    virtual std::unique_ptr<DecoderStepEngine> build_engine(
        const TrtDecoderDefinition& weights, TrtLogger& logger) = 0;

    virtual std::unique_ptr<IStepState> create_state(
        const DecoderStepEngine& engine) = 0;

    virtual bool run_step(const DecoderStepEngine& engine,
        IStepState& state, int32_t token_id,
        std::vector<float>& out_logits, std::string& error) = 0;
};

// Registry
void RegisterModelRuntime(const std::string& family, std::unique_ptr<IModelRuntime> runtime);
IModelRuntime* FindModelRuntime(const std::string& family);

// Standard decoder graph (Pre-RMSNorm + GQA + RoPE + SwiGLU) + standard KV-cache state/step.
std::unique_ptr<IModelRuntime> CreateStandardDecoderRuntime();

#endif // TRTF_HAS_TRT

} // namespace trtf
