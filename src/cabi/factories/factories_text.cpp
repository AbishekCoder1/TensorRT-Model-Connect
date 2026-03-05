#include "cabi/factories/factories_text.h"

#include "runtime/trt/recurrent/hybrid_backend.h"
#include "runtime/trt/recurrent/mamba_backend.h"
#include "runtime/trt/recurrent/rwkv_backend.h"
#include "runtime/trt/core/trt_backend_shared.h"

#include <iostream>
#include <stdexcept>
#include <utility>

namespace trtf {
namespace cabi {

#if TRTF_HAS_TRT

trtf::IPipeline* create_mamba_pipeline(
    trtf::TrtUniquePtr<nvinfer1::ICudaEngine> trt_engine,
    trtf::TrtUniquePtr<nvinfer1::IExecutionContext> exec_ctx,
    const trtf::FastPathModelConfig& fp_cfg,
    const trtf::BundleSections& sections,
    const std::string& model_id,
    const std::string& hf_python,
    const std::string& bundle_path)
{
    auto mamba_engine = std::make_unique<trtf::MambaStepEngine>();
    mamba_engine->engine = std::move(trt_engine);
    mamba_engine->context = std::move(exec_ctx);
    mamba_engine->vocab_size = fp_cfg.vocab_size;
    mamba_engine->hidden_size = fp_cfg.hidden_size;
    mamba_engine->d_inner = fp_cfg.d_inner;
    mamba_engine->state_size = fp_cfg.state_size;
    mamba_engine->conv_kernel = fp_cfg.conv_kernel;
    mamba_engine->num_layers = fp_cfg.num_layers;
    mamba_engine->id_bos = fp_cfg.id_bos;
    mamba_engine->id_eos = fp_cfg.id_eos;

    for (int32_t i = 0; i < fp_cfg.num_layers; ++i)
    {
        mamba_engine->conv_state_input_names.push_back(trtf::layer_tensor_name("conv_state", i));
        mamba_engine->ssm_state_input_names.push_back(trtf::layer_tensor_name("ssm_state", i));
        mamba_engine->present_conv_output_names.push_back(trtf::layer_tensor_name("present_conv", i));
        mamba_engine->present_ssm_output_names.push_back(trtf::layer_tensor_name("present_ssm", i));
    }

    if (!trtf::has_all_required_mamba_tensors(*mamba_engine))
    {
        throw std::runtime_error("Bundle engine missing required Mamba tensors: " + bundle_path);
    }

    auto tok = trtf::extract_tokenizer_from_bundle(sections, hf_python);
    auto backend = trtf::CreateMambaBackendFromEngine(std::move(mamba_engine));
    if (!backend || !backend->is_available())
    {
        throw std::runtime_error("Failed to create Mamba TRT backend from bundle engine");
    }

    std::cerr << "[trtf] Runtime ready (backend=trt_mamba, strategy=ssm_recurrent)" << std::endl;
    return detail::create_text_pipeline_impl(
        model_id, std::move(tok), std::move(backend), "trt_mamba");
}

trtf::IPipeline* create_rwkv_pipeline(
    trtf::TrtUniquePtr<nvinfer1::ICudaEngine> trt_engine,
    trtf::TrtUniquePtr<nvinfer1::IExecutionContext> exec_ctx,
    const trtf::FastPathModelConfig& fp_cfg,
    const trtf::BundleSections& sections,
    const std::string& model_id,
    const std::string& hf_python,
    const std::string& bundle_path)
{
    auto rwkv_engine = std::make_unique<trtf::RwkvStepEngine>();
    rwkv_engine->engine = std::move(trt_engine);
    rwkv_engine->context = std::move(exec_ctx);
    rwkv_engine->vocab_size = fp_cfg.vocab_size;
    rwkv_engine->hidden_size = fp_cfg.hidden_size;
    rwkv_engine->num_layers = fp_cfg.num_layers;
    rwkv_engine->id_bos = fp_cfg.id_bos;
    rwkv_engine->id_eos = fp_cfg.id_eos;

    for (int32_t i = 0; i < fp_cfg.num_layers; ++i)
    {
        rwkv_engine->attn_state_input_names.push_back(trtf::layer_tensor_name("attn_state", i));
        rwkv_engine->ff_state_input_names.push_back(trtf::layer_tensor_name("ff_state", i));
        rwkv_engine->num_state_input_names.push_back(trtf::layer_tensor_name("num_state", i));
        rwkv_engine->den_state_input_names.push_back(trtf::layer_tensor_name("den_state", i));
        rwkv_engine->max_state_input_names.push_back(trtf::layer_tensor_name("max_state", i));
        rwkv_engine->present_attn_output_names.push_back(trtf::layer_tensor_name("present_attn", i));
        rwkv_engine->present_ff_output_names.push_back(trtf::layer_tensor_name("present_ff", i));
        rwkv_engine->present_num_output_names.push_back(trtf::layer_tensor_name("present_num", i));
        rwkv_engine->present_den_output_names.push_back(trtf::layer_tensor_name("present_den", i));
        rwkv_engine->present_max_output_names.push_back(trtf::layer_tensor_name("present_max", i));
    }

    if (!trtf::has_all_required_rwkv_tensors(*rwkv_engine))
    {
        throw std::runtime_error("Bundle engine missing required RWKV tensors: " + bundle_path);
    }

    auto tok = trtf::extract_tokenizer_from_bundle(sections, hf_python);
    auto backend = trtf::CreateRwkvBackendFromEngine(std::move(rwkv_engine));
    if (!backend || !backend->is_available())
    {
        throw std::runtime_error("Failed to create RWKV TRT backend from bundle engine");
    }

    std::cerr << "[trtf] Runtime ready (backend=trt_rwkv, strategy=rwkv_recurrent)" << std::endl;
    return detail::create_text_pipeline_impl(
        model_id, std::move(tok), std::move(backend), "trt_rwkv");
}

trtf::IPipeline* create_hybrid_pipeline(
    trtf::TrtUniquePtr<nvinfer1::ICudaEngine> trt_engine,
    trtf::TrtUniquePtr<nvinfer1::IExecutionContext> exec_ctx,
    const trtf::FastPathModelConfig& fp_cfg,
    const trtf::BundleSections& sections,
    const std::string& model_id,
    const std::string& hf_python,
    const std::string& bundle_path)
{
    auto hybrid_engine = std::make_unique<trtf::HybridStepEngine>();
    hybrid_engine->engine = std::move(trt_engine);
    hybrid_engine->context = std::move(exec_ctx);
    hybrid_engine->vocab_size = fp_cfg.vocab_size;
    hybrid_engine->hidden_size = fp_cfg.hidden_size;
    hybrid_engine->attention_size = fp_cfg.attention_size;
    hybrid_engine->max_cache_length = fp_cfg.max_cache_length;
    hybrid_engine->d_inner = fp_cfg.d_inner;
    hybrid_engine->d_state = fp_cfg.mamba_d_state;
    hybrid_engine->d_conv = fp_cfg.mamba_d_conv;
    hybrid_engine->nheads = fp_cfg.mamba_nheads;
    hybrid_engine->head_dim = fp_cfg.mamba_head_dim;
    hybrid_engine->conv_dim = fp_cfg.conv_dim;
    hybrid_engine->num_mamba_layers = fp_cfg.num_mamba_layers;
    hybrid_engine->num_attention_layers = fp_cfg.num_attention_layers;
    hybrid_engine->layer_types = fp_cfg.layer_types;
    hybrid_engine->id_bos = fp_cfg.id_bos;
    hybrid_engine->id_eos = fp_cfg.id_eos;

    for (int32_t i = 0; i < fp_cfg.num_mamba_layers; ++i)
    {
        hybrid_engine->conv_state_input_names.push_back(trtf::layer_tensor_name("conv_state", i));
        hybrid_engine->ssm_state_input_names.push_back(trtf::layer_tensor_name("ssm_state", i));
        hybrid_engine->present_conv_output_names.push_back(trtf::layer_tensor_name("present_conv", i));
        hybrid_engine->present_ssm_output_names.push_back(trtf::layer_tensor_name("present_ssm", i));
    }

    for (int32_t i = 0; i < fp_cfg.num_attention_layers; ++i)
    {
        hybrid_engine->cache_k_input_names.push_back(trtf::layer_tensor_name("cache_k", i));
        hybrid_engine->cache_v_input_names.push_back(trtf::layer_tensor_name("cache_v", i));
        hybrid_engine->present_k_output_names.push_back(trtf::layer_tensor_name("present_k", i));
        hybrid_engine->present_v_output_names.push_back(trtf::layer_tensor_name("present_v", i));
    }

    if (!trtf::has_all_required_hybrid_tensors(*hybrid_engine))
    {
        throw std::runtime_error("Bundle engine missing required hybrid tensors: " + bundle_path);
    }

    auto tok = trtf::extract_tokenizer_from_bundle(sections, hf_python);
    auto backend = trtf::CreateHybridBackendFromEngine(std::move(hybrid_engine));
    if (!backend || !backend->is_available())
    {
        throw std::runtime_error("Failed to create hybrid TRT backend from bundle engine");
    }

    std::cerr << "[trtf] Runtime ready (backend=trt_hybrid, strategy=hybrid_mamba_attention)" << std::endl;
    return detail::create_text_pipeline_impl(
        model_id, std::move(tok), std::move(backend), "trt_hybrid");
}

trtf::IPipeline* create_decoder_pipeline(
    trtf::TrtUniquePtr<nvinfer1::ICudaEngine> trt_engine,
    trtf::TrtUniquePtr<nvinfer1::IExecutionContext> exec_ctx,
    const trtf::FastPathModelConfig& fp_cfg,
    const trtf::BundleSections& sections,
    const std::string& model_id,
    const std::string& hf_python,
    const std::string& bundle_path)
{
    auto engine_struct = trtf::make_decoder_engine(std::move(trt_engine), std::move(exec_ctx), fp_cfg);
    if (!trtf::has_all_required_tensors(*engine_struct))
    {
        throw std::runtime_error("Bundle engine missing required tensors: " + bundle_path);
    }

    // Use tokenizer_add_special_tokens from bundle config if present,
    // otherwise default to true to match HF's tokenizer.encode() default.
    // This ensures the C++ binary tokenizes prompts the same way as the
    // HF reference and debug runner.
    const bool add_special = fp_cfg.tokenizer_add_special_tokens_present
        ? fp_cfg.tokenizer_add_special_tokens
        : true;
    auto tok = trtf::extract_tokenizer_from_bundle(sections, hf_python, add_special);

    const auto& strategy = fp_cfg.runtime_strategy;
    if (strategy != "decoder_kv_cache" && strategy != "decoder_moe")
    {
        throw std::runtime_error("Unsupported runtime_strategy: " + strategy
            + " (supported: decoder_kv_cache, decoder_moe, ssm_recurrent, rwkv_recurrent, speech_to_text, vision_language)");
    }

    auto backend = trtf::CreateTrtBackendFromEngine(std::move(engine_struct));
    if (!backend || !backend->is_available())
    {
        throw std::runtime_error("Failed to create TRT backend from bundle engine");
    }

    std::cerr << "[trtf] Runtime ready (backend=trt, strategy=" << strategy << ")" << std::endl;
    return detail::create_text_pipeline_impl(
        model_id, std::move(tok), std::move(backend), "trt");
}

#endif // TRTF_HAS_TRT

} // namespace cabi
} // namespace trtf
