#pragma once

// Internal interface for old-style generation backends.
// This header replaces the deleted public include/trtf/backend.h and
// include/trtf/generation.h.  It is used only inside src/runtime/trt/
// and keeps the existing backend implementations compiling while the
// public API migrates to IPipeline (include/trtf/pipeline.h).

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtf {

/// Lightweight config passed to old-style backend generate() methods.
struct GenerationConfig {
    std::size_t max_new_tokens{20};
    bool do_sample{false};
    float temperature{1.0F};
};

/// Abstract base for TRT generation backends (decoder, mamba, VL, etc.).
class IGenerationBackend {
public:
    virtual ~IGenerationBackend() = default;

    /// Whether the backend is ready to run inference.
    virtual bool is_available() const = 0;

    /// Human-readable backend name.
    virtual const char* name() const = 0;

    /// Autoregressive generation: input token IDs -> output token IDs.
    virtual std::vector<int32_t> generate(
        const std::vector<int32_t>& input_ids,
        const GenerationConfig& config) = 0;

    /// Optional: VL generation with pre-computed image features.
    virtual std::vector<int32_t> generate_vl(
        const std::vector<int32_t>& /*input_ids*/,
        const float* /*image_features*/, int32_t /*num_features*/, int32_t /*feature_dim*/,
        const std::vector<int32_t>& /*image_positions*/,
        const GenerationConfig& /*config*/)
    {
        return {};
    }

    /// Whether this backend supports vision input.
    virtual bool supports_vision() const { return false; }
};

} // namespace trtf
