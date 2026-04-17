// RwkvPlugin: handles "rwkv_recurrent" strategy.
// RWKV models with 5 recurrent state vectors per layer.

#include "runtime/pipelines/recurrent_pipeline.h"
#include "runtime/plugins/shared/plugin_helpers.h"
#include "trtf/runtime/pipeline_registry.h"

#if TRTF_HAS_TRT

namespace trtf {

class RwkvPlugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        load_ffi_kernels_from_bundle(ctx.bundle);
        auto loaded =
            load_trt_module_from_plan(find_section(ctx.bundle, "engine_plan"), "engine_plan");
        auto tokenizer = create_tokenizer_from_bundle(ctx.bundle);

        cudaStream_t stream = loaded.stream->get();

        std::vector<RecurrentState::TensorSpec> specs = {
            {"attn_state", {ctx.config.hidden_size}, "present_attn"},
            {"ff_state", {ctx.config.hidden_size}, "present_ff"},
            {"num_state", {ctx.config.hidden_size}, "present_num"},
            {"den_state", {ctx.config.hidden_size}, "present_den"},
            {"max_state", {ctx.config.hidden_size}, "present_max"}};

        auto state = std::make_unique<RecurrentState>(ctx.config.num_layers, specs, stream);
        auto rgc = make_recurrent_gen_config(ctx.config);

        return std::make_unique<RecurrentPipeline>(std::move(loaded.module), std::move(state), rgc,
                                                   stream, "RwkvPipeline", std::move(tokenizer),
                                                   ctx.bundle.info.model_id);
    }
};

volatile int kForceLink_RwkvPlugin = 0;

} // namespace trtf

static trtf::RwkvPlugin g_RwkvPlugin_instance;
static trtf::PluginRegistrar g_RwkvPlugin_reg("rwkv_recurrent", &g_RwkvPlugin_instance);

#endif // TRTF_HAS_TRT
