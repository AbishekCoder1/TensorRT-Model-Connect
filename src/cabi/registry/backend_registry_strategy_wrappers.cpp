#include "cabi/registry/backend_registry_strategy_wrappers.h"

#include "cabi/factories/factory_decls.h"
#include "cabi/factories/factories_encoder.h"
#include "cabi/factories/factories_text.h"
#include "cabi/factories/factories_vision.h"

#include <utility>

namespace trtf {
namespace cabi {

#if TRTF_HAS_TRT
namespace {

bool has_core_registry_dispatch_context(const BackendRegistryDispatchContext* ctx)
{
    if (ctx == nullptr)
    {
        return false;
    }

    const void* required_ptrs[] = {
        ctx->trt_engine,
        ctx->exec_ctx,
        ctx->fp_cfg,
        ctx->model_id,
        ctx->bundle_path,
    };
    for (const void* ptr : required_ptrs)
    {
        if (ptr == nullptr)
        {
            return false;
        }
    }

    if (!*ctx->trt_engine)
    {
        return false;
    }
    if (!*ctx->exec_ctx)
    {
        return false;
    }

    return true;
}

bool has_registry_dispatch_context(
    const BackendRegistryDispatchContext* ctx,
    bool require_sections,
    bool require_runtime,
    bool require_hf_python)
{
    if (!has_core_registry_dispatch_context(ctx))
    {
        return false;
    }

    if (require_sections)
    {
        if (ctx->sections == nullptr)
        {
            return false;
        }
    }

    if (require_runtime)
    {
        if (ctx->runtime_ptr == nullptr)
        {
            return false;
        }
        if (!*ctx->runtime_ptr)
        {
            return false;
        }
    }

    if (require_hf_python)
    {
        if (ctx->hf_python == nullptr)
        {
            return false;
        }
    }

    return true;
}

} // namespace

trtf::IPipeline* create_segmentation_pipeline_via_registry(void* opaque_context)
{
    auto* ctx = static_cast<BackendRegistryDispatchContext*>(opaque_context);
    if (!has_registry_dispatch_context(
            ctx, /*require_sections=*/false,
            /*require_runtime=*/false,
            /*require_hf_python=*/false))
    {
        return nullptr;
    }

    return trtf::cabi::create_segmentation_pipeline(
        std::move(*ctx->trt_engine), std::move(*ctx->exec_ctx), *ctx->fp_cfg,
        *ctx->model_id, *ctx->bundle_path);
}

trtf::IPipeline* create_mamba_pipeline_via_registry(void* opaque_context)
{
    auto* ctx = static_cast<BackendRegistryDispatchContext*>(opaque_context);
    if (!has_registry_dispatch_context(
            ctx, /*require_sections=*/true,
            /*require_runtime=*/false,
            /*require_hf_python=*/true))
    {
        return nullptr;
    }

    return trtf::cabi::create_mamba_pipeline(
        std::move(*ctx->trt_engine), std::move(*ctx->exec_ctx), *ctx->fp_cfg,
        *ctx->sections, *ctx->model_id, *ctx->hf_python, *ctx->bundle_path);
}

trtf::IPipeline* create_rwkv_pipeline_via_registry(void* opaque_context)
{
    auto* ctx = static_cast<BackendRegistryDispatchContext*>(opaque_context);
    if (!has_registry_dispatch_context(
            ctx, /*require_sections=*/true,
            /*require_runtime=*/false,
            /*require_hf_python=*/true))
    {
        return nullptr;
    }

    return trtf::cabi::create_rwkv_pipeline(
        std::move(*ctx->trt_engine), std::move(*ctx->exec_ctx), *ctx->fp_cfg,
        *ctx->sections, *ctx->model_id, *ctx->hf_python, *ctx->bundle_path);
}

trtf::IPipeline* create_decoder_pipeline_via_registry(void* opaque_context)
{
    auto* ctx = static_cast<BackendRegistryDispatchContext*>(opaque_context);
    if (!has_registry_dispatch_context(
            ctx, /*require_sections=*/true,
            /*require_runtime=*/false,
            /*require_hf_python=*/true))
    {
        return nullptr;
    }

    return trtf::cabi::create_decoder_pipeline(
        std::move(*ctx->trt_engine), std::move(*ctx->exec_ctx), *ctx->fp_cfg,
        *ctx->sections, *ctx->model_id, *ctx->hf_python, *ctx->bundle_path);
}

trtf::IPipeline* create_whisper_pipeline_via_registry(void* opaque_context)
{
    auto* ctx = static_cast<BackendRegistryDispatchContext*>(opaque_context);
    if (!has_registry_dispatch_context(
            ctx, /*require_sections=*/true,
            /*require_runtime=*/true,
            /*require_hf_python=*/true))
    {
        return nullptr;
    }

    return trtf::cabi::detail::create_whisper_pipeline(
        std::move(*ctx->trt_engine), std::move(*ctx->exec_ctx), *ctx->fp_cfg,
        *ctx->sections, *ctx->runtime_ptr, *ctx->model_id, *ctx->hf_python, *ctx->bundle_path);
}

trtf::IPipeline* create_vl_pipeline_via_registry(void* opaque_context)
{
    auto* ctx = static_cast<BackendRegistryDispatchContext*>(opaque_context);
    if (!has_registry_dispatch_context(
            ctx, /*require_sections=*/true,
            /*require_runtime=*/true,
            /*require_hf_python=*/true))
    {
        return nullptr;
    }

    return trtf::cabi::detail::create_vl_pipeline(
        std::move(*ctx->trt_engine), std::move(*ctx->exec_ctx), *ctx->fp_cfg,
        *ctx->sections, *ctx->runtime_ptr, *ctx->model_id, *ctx->hf_python, *ctx->bundle_path);
}

trtf::IPipeline* create_encoder_pipeline_via_registry(void* opaque_context)
{
    auto* ctx = static_cast<BackendRegistryDispatchContext*>(opaque_context);
    if (!has_registry_dispatch_context(
            ctx, /*require_sections=*/true,
            /*require_runtime=*/false,
            /*require_hf_python=*/true))
    {
        return nullptr;
    }

    return trtf::cabi::create_encoder_pipeline(
        std::move(*ctx->trt_engine), std::move(*ctx->exec_ctx), *ctx->fp_cfg,
        *ctx->sections, *ctx->model_id, *ctx->hf_python, *ctx->bundle_path);
}

trtf::IPipeline* create_embedding_pipeline_via_registry(void* opaque_context)
{
    auto* ctx = static_cast<BackendRegistryDispatchContext*>(opaque_context);
    if (!has_registry_dispatch_context(
            ctx, /*require_sections=*/true,
            /*require_runtime=*/true,
            /*require_hf_python=*/true))
    {
        return nullptr;
    }

    return trtf::cabi::create_embedding_pipeline(
        std::move(*ctx->trt_engine), std::move(*ctx->exec_ctx), *ctx->fp_cfg,
        *ctx->sections, *ctx->runtime_ptr, *ctx->model_id, *ctx->hf_python, *ctx->bundle_path);
}

trtf::IPipeline* create_reranking_pipeline_via_registry(void* opaque_context)
{
    auto* ctx = static_cast<BackendRegistryDispatchContext*>(opaque_context);
    if (!has_registry_dispatch_context(
            ctx, /*require_sections=*/true,
            /*require_runtime=*/false,
            /*require_hf_python=*/true))
    {
        return nullptr;
    }

    return trtf::cabi::create_reranking_pipeline(
        std::move(*ctx->trt_engine), std::move(*ctx->exec_ctx), *ctx->fp_cfg,
        *ctx->sections, *ctx->model_id, *ctx->hf_python, *ctx->bundle_path);
}

trtf::IPipeline* create_text_to_audio_pipeline_via_registry(void* opaque_context)
{
    auto* ctx = static_cast<BackendRegistryDispatchContext*>(opaque_context);
    if (!has_registry_dispatch_context(
            ctx, /*require_sections=*/true,
            /*require_runtime=*/true,
            /*require_hf_python=*/true))
    {
        return nullptr;
    }

    if (ctx->fp_cfg->is_magpie_tts)
    {
        return trtf::cabi::detail::create_magpie_tts_pipeline(
            std::move(*ctx->trt_engine), std::move(*ctx->exec_ctx), *ctx->fp_cfg,
            *ctx->sections, *ctx->runtime_ptr, *ctx->model_id, *ctx->hf_python, *ctx->bundle_path);
    }

    return trtf::cabi::detail::create_bark_pipeline(
        std::move(*ctx->trt_engine), std::move(*ctx->exec_ctx), *ctx->fp_cfg,
        *ctx->sections, *ctx->runtime_ptr, *ctx->model_id, *ctx->hf_python, *ctx->bundle_path);
}

trtf::IPipeline* create_hybrid_pipeline_via_registry(void* opaque_context)
{
    auto* ctx = static_cast<BackendRegistryDispatchContext*>(opaque_context);
    if (!has_registry_dispatch_context(
            ctx, /*require_sections=*/true,
            /*require_runtime=*/false,
            /*require_hf_python=*/true))
    {
        return nullptr;
    }

    return trtf::cabi::create_hybrid_pipeline(
        std::move(*ctx->trt_engine), std::move(*ctx->exec_ctx), *ctx->fp_cfg,
        *ctx->sections, *ctx->model_id, *ctx->hf_python, *ctx->bundle_path);
}

trtf::IPipeline* create_detection_pipeline_via_registry(void* opaque_context)
{
    auto* ctx = static_cast<BackendRegistryDispatchContext*>(opaque_context);
    if (!has_registry_dispatch_context(
            ctx, /*require_sections=*/false,
            /*require_runtime=*/false,
            /*require_hf_python=*/false))
    {
        return nullptr;
    }

    return trtf::cabi::create_detection_pipeline(
        std::move(*ctx->trt_engine), std::move(*ctx->exec_ctx), *ctx->fp_cfg,
        *ctx->model_id, *ctx->bundle_path);
}

trtf::IPipeline* create_sam_pipeline_via_registry(void* opaque_context)
{
    auto* ctx = static_cast<BackendRegistryDispatchContext*>(opaque_context);
    if (!has_registry_dispatch_context(
            ctx, /*require_sections=*/true,
            /*require_runtime=*/true,
            /*require_hf_python=*/false))
    {
        return nullptr;
    }

    return trtf::cabi::create_sam_pipeline(
        std::move(*ctx->trt_engine), std::move(*ctx->exec_ctx), *ctx->fp_cfg,
        *ctx->sections, *ctx->runtime_ptr, *ctx->model_id, *ctx->bundle_path);
}

trtf::IPipeline* create_neural_operator_pipeline_via_registry(void* opaque_context)
{
    auto* ctx = static_cast<BackendRegistryDispatchContext*>(opaque_context);
    if (!has_registry_dispatch_context(
            ctx, /*require_sections=*/false,
            /*require_runtime=*/false,
            /*require_hf_python=*/false))
    {
        return nullptr;
    }

    return trtf::cabi::create_neural_operator_pipeline(
        std::move(*ctx->trt_engine), std::move(*ctx->exec_ctx), *ctx->fp_cfg,
        *ctx->model_id, *ctx->bundle_path);
}

trtf::IPipeline* create_omni_pipeline_via_registry(void* opaque_context)
{
    auto* ctx = static_cast<BackendRegistryDispatchContext*>(opaque_context);
    if (!has_registry_dispatch_context(
            ctx, /*require_sections=*/true,
            /*require_runtime=*/true,
            /*require_hf_python=*/true))
    {
        return nullptr;
    }

    return trtf::cabi::detail::create_omni_pipeline(
        std::move(*ctx->trt_engine), std::move(*ctx->exec_ctx), *ctx->fp_cfg,
        *ctx->sections, *ctx->runtime_ptr, *ctx->model_id, *ctx->hf_python, *ctx->bundle_path);
}

trtf::IPipeline* create_speech_pipeline_via_registry(void* opaque_context)
{
    auto* ctx = static_cast<BackendRegistryDispatchContext*>(opaque_context);
    if (!has_registry_dispatch_context(
            ctx, /*require_sections=*/true,
            /*require_runtime=*/true,
            /*require_hf_python=*/true))
    {
        return nullptr;
    }

    return trtf::cabi::detail::create_speech_pipeline(
        std::move(*ctx->trt_engine), std::move(*ctx->exec_ctx), *ctx->fp_cfg,
        *ctx->sections, *ctx->runtime_ptr, *ctx->model_id, *ctx->hf_python, *ctx->bundle_path);
}

#endif // TRTF_HAS_TRT

} // namespace cabi
} // namespace trtf
