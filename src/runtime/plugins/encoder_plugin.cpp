// EncoderPlugin: handles "encoder_only", "embedding", "reranking", and
// "neural_operator" strategies. Single-pass encoder models (BERT, Eagle, etc.).

#include "runtime/pipelines/encoder_pipeline.h"
#include "runtime/plugins/shared/plugin_helpers.h"
#include "trtf/runtime/pipeline_registry.h"

#if TRTF_HAS_TRT

namespace trtf {

class EncoderPlugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        load_ffi_kernels_from_bundle(ctx.bundle);
        auto loaded =
            load_trt_module_from_plan(find_section(ctx.bundle, "engine_plan"), "engine_plan");
        auto tokenizer = create_tokenizer_from_bundle(ctx.bundle);

        return std::make_unique<EncoderPipeline>(std::move(loaded.module),
                                                 ctx.config.runtime_strategy, std::move(tokenizer),
                                                 ctx.bundle.info.model_id);
    }
};

volatile int kForceLink_EncoderPlugin = 0;

} // namespace trtf

static trtf::EncoderPlugin g_EncoderPlugin_instance;
static trtf::PluginRegistrar g_EncoderPlugin_reg1("encoder_only", &g_EncoderPlugin_instance);
static trtf::PluginRegistrar g_EncoderPlugin_reg2("embedding", &g_EncoderPlugin_instance);
static trtf::PluginRegistrar g_EncoderPlugin_reg3("reranking", &g_EncoderPlugin_instance);
static trtf::PluginRegistrar g_EncoderPlugin_reg4("neural_operator", &g_EncoderPlugin_instance);

#endif // TRTF_HAS_TRT
