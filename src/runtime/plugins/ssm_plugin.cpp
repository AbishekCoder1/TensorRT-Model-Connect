// SsmPlugin: handles "ssm_recurrent" strategy.
// Mamba/SSM models with conv_state + ssm_state recurrent state.

#include "trtf/runtime/pipeline_registry.h"
#include "runtime/plugins/shared/plugin_helpers.h"
#include "runtime/pipelines/recurrent_pipeline.h"
#include "utils/json_helpers.h"

#if TRTF_HAS_TRT

namespace trtf {

class SsmPlugin final : public IPipelinePlugin {
public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        auto loaded = load_trt_module_from_plan(find_section(ctx.bundle, "engine_plan"), "engine_plan");
        auto tokenizer = create_tokenizer_from_bundle(ctx.bundle);

        cudaStream_t stream = loaded.stream->get();

        int32_t d_inner = extract_json_int(ctx.config_json, "intermediate_size", 0);
        if (d_inner == 0)
            d_inner = extract_json_int(ctx.config_json, "d_inner", ctx.config.hidden_size * 2);
        int32_t state_size = extract_json_int(ctx.config_json, "state_size", 16);
        int32_t conv_kernel = extract_json_int(ctx.config_json, "conv_kernel", 4);

        std::vector<RecurrentState::TensorSpec> specs = {
            {"conv_state", {d_inner * conv_kernel}, "present_conv"},
            {"ssm_state", {state_size * d_inner}, "present_ssm"}};

        auto state = std::make_unique<RecurrentState>(ctx.config.num_layers, specs, stream);
        auto rgc = make_recurrent_gen_config(ctx.config);

        return std::make_unique<RecurrentPipeline>(
            std::move(loaded.module), std::move(state),
            rgc, stream, "MambaPipeline", std::move(tokenizer), ctx.bundle.info.model_id);
    }
};

volatile int kForceLink_SsmPlugin = 0;

} // namespace trtf

static trtf::SsmPlugin g_SsmPlugin_instance;
static trtf::PluginRegistrar g_SsmPlugin_reg("ssm_recurrent", &g_SsmPlugin_instance);

#endif // TRTF_HAS_TRT
