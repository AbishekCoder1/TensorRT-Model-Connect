#pragma once

// RecurrentPipeline: Mamba, RWKV, and Hybrid models.
// Uses IStateManager to abstract recurrent vs KV cache state.

#include "trtf/pipeline.h"
#include "trtf/tokenizer.h"
#include "trtf/runtime/trt_module.h"
#include "trtf/runtime/kv_cache.h"
#include "trtf/runtime/recurrent_state.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#if TRTF_HAS_TRT

namespace trtf {

struct IStateManager {
    virtual ~IStateManager() = default;
    virtual void reset() = 0;
    virtual void bind_to(TrtModule& module) = 0;
    virtual void advance() = 0;
    virtual void build_mask(std::vector<float>& mask) const = 0;
    virtual int32_t position() const = 0;
    virtual bool has_mask() const = 0;
};

class RecurrentStateManager final : public IStateManager {
public:
    explicit RecurrentStateManager(std::unique_ptr<RecurrentState> state)
        : state_(std::move(state)) {}
    void reset() override { state_->reset(); }
    void bind_to(TrtModule& module) override { state_->bind_to(module); }
    void advance() override { state_->advance(); position_++; }
    void build_mask(std::vector<float>&) const override {}
    int32_t position() const override { return position_; }
    bool has_mask() const override { return false; }
private:
    std::unique_ptr<RecurrentState> state_;
    int32_t position_{0};
};

class HybridStateManager final : public IStateManager {
public:
    HybridStateManager(std::unique_ptr<KvCache> kv, std::unique_ptr<RecurrentState> ssm)
        : kv_(std::move(kv)), ssm_(std::move(ssm)) {}
    void reset() override { kv_->reset(); ssm_->reset(); }
    void bind_to(TrtModule& module) override { kv_->bind_to(module); ssm_->bind_to(module); }
    void advance() override { kv_->advance(); ssm_->advance(); }
    void build_mask(std::vector<float>& mask) const override { kv_->build_attention_mask(mask); }
    int32_t position() const override { return kv_->position(); }
    bool has_mask() const override { return true; }
private:
    std::unique_ptr<KvCache> kv_;
    std::unique_ptr<RecurrentState> ssm_;
};

struct RecurrentGenConfig {
    int32_t vocab_size{0};
    int32_t id_bos{0};
    int32_t id_eos{0};
    bool has_position_input{false};
};

class RecurrentPipeline final : public IPipeline {
public:
    RecurrentPipeline(
        std::unique_ptr<TrtModule> decoder,
        std::unique_ptr<IStateManager> state,
        RecurrentGenConfig config,
        cudaStream_t stream,
        const char* name,
        std::shared_ptr<ITokenizer> tokenizer = nullptr,
        std::string model_id_str = "");

    TextResult generate(const std::string& prompt, const GenerateConfig& cfg = {}) override;

    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return name_; }

    // Token-ID-based generation (for unit tests and internal callers).
    struct GenerationResult {
        std::vector<int32_t> token_ids;
    };
    GenerationResult generate_ids(const std::vector<int32_t>& input_ids,
                                  const GenerateConfig& cfg);

    static int32_t argmax(const std::vector<float>& logits);

private:
    std::unique_ptr<TrtModule> decoder_;
    std::unique_ptr<IStateManager> state_;
    RecurrentGenConfig config_;
    cudaStream_t stream_;
    const char* name_;
    std::shared_ptr<ITokenizer> tokenizer_;
    std::string model_id_;

    std::vector<int32_t> generate_from_ids(
        const std::vector<int32_t>& input_ids,
        int32_t max_new_tokens,
        int32_t eos_token_id);

    void run_step(int32_t token_id, std::vector<float>& logits);
};

} // namespace trtf

#endif // TRTF_HAS_TRT
