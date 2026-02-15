#include "runtime/trt/model_runtime.h"
#include "runtime/trt/kv_cache_step_state.h"
#include "runtime/trt/trt_decode_runtime.h"
#include "model/standard_decoder_graph_builder.h"

#include <mutex>
#include <unordered_map>
#include <utility>

namespace trtf {

#if TRTF_HAS_TRT

namespace {

std::mutex& runtimes_mutex()
{
    static std::mutex m;
    return m;
}

std::unordered_map<std::string, std::unique_ptr<IModelRuntime>>& runtimes()
{
    static std::unordered_map<std::string, std::unique_ptr<IModelRuntime>> registry;
    return registry;
}

// Anonymous implementation — not visible outside this TU.
// Custom engine builder + standard KV-cache state/step.
class KvCacheRuntimeImpl final : public IModelRuntime {
public:
    explicit KvCacheRuntimeImpl(EngineFactory engine_factory)
        : mEngineFactory(std::move(engine_factory))
    {
    }

    std::unique_ptr<DecoderStepEngine> build_engine(
        const TrtDecoderDefinition& weights, TrtLogger& logger) override
    {
        return mEngineFactory(weights, logger);
    }

    std::unique_ptr<IStepState> create_state(
        const DecoderStepEngine& engine) override
    {
        return std::make_unique<KvCacheStepState>(engine);
    }

    bool run_step(const DecoderStepEngine& engine,
        IStepState& state, int32_t token_id,
        std::vector<float>& out_logits, std::string& error) override
    {
        auto& kv_state = static_cast<KvCacheStepState&>(state);

        int32_t position_id{};
        std::vector<float> mask;
        kv_state.prepare_step(position_id, mask);

        std::vector<std::vector<float>> present_k;
        std::vector<std::vector<float>> present_v;

        if (!run_decoder_step(engine, token_id, position_id,
                kv_state.cache_k_by_layer(), kv_state.cache_v_by_layer(), mask,
                out_logits, present_k, present_v, error))
        {
            return false;
        }

        kv_state.update_after_step(present_k, present_v);
        return true;
    }

private:
    EngineFactory mEngineFactory;
};

} // namespace

void RegisterModelRuntime(const std::string& family, std::unique_ptr<IModelRuntime> runtime)
{
    std::lock_guard<std::mutex> lock(runtimes_mutex());
    runtimes()[family] = std::move(runtime);
}

IModelRuntime* FindModelRuntime(const std::string& family)
{
    std::lock_guard<std::mutex> lock(runtimes_mutex());
    const auto it = runtimes().find(family);
    if (it == runtimes().end())
    {
        return nullptr;
    }
    return it->second.get();
}

std::unique_ptr<IModelRuntime> CreateKvCacheRuntime(EngineFactory engine_factory)
{
    return std::make_unique<KvCacheRuntimeImpl>(std::move(engine_factory));
}

std::unique_ptr<IModelRuntime> CreateStandardDecoderRuntime()
{
    return CreateKvCacheRuntime(
        [](const TrtDecoderDefinition& weights, TrtLogger& logger) {
            StandardDecoderGraphBuilder builder;
            return builder.build_decoder_step_engine(weights, logger);
        });
}

#endif // TRTF_HAS_TRT

} // namespace trtf
