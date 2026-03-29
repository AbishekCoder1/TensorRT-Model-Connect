#pragma once
#include "runtime/plugins/shared/plugin_helpers.h"
#include "runtime/domains/audio/audio_configs.h"
#include "utils/json_helpers.h"

#if TRTF_HAS_TRT

namespace trtf {

std::unique_ptr<KvCache> make_coarse_kv_cache(
    const std::string& json, const BaseConfig& base, cudaStream_t stream);

MagpieTTSConfig build_magpie_config(const std::string& json, const BaseConfig& base);

SpeechConfig build_speech_config_from_bundle(
    const BundleFile& bundle,
    const std::string& json,
    const BaseConfig& base,
    const std::string& hf_python);

void infer_speech_vocab_sizes(SpeechConfig& sc, const std::string& json, const BaseConfig& base);

BaseConfig make_depth_engine_config(const std::string& json, const BaseConfig& base);

std::vector<std::unique_ptr<TrtModule>> load_depth_engines(
    const BundleFile& bundle,
    std::shared_ptr<CudaStream> shared_stream);

void allocate_cross_kv_buffers(
    int32_t num_layers, std::size_t buf_size,
    std::vector<CudaBuffer>& cross_k, std::vector<CudaBuffer>& cross_v);

std::shared_ptr<ITokenizer> make_ipa_tok(const BundleFile& bundle);

int32_t compute_kv_dim_kv_heads(const BaseConfig& base, int32_t default_dim);

int32_t safe_embed_dim(const std::vector<float>& data, int32_t divisor);

} // namespace trtf

#endif // TRTF_HAS_TRT
