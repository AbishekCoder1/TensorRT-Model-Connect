#include "runtime/plugins/shared/plugin_helpers.h"

#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace trtf {

#if TRTF_HAS_TRT

// ─── Tokenizer helpers ───

bool detect_add_special_tokens(const BundleFile& bundle)
{
    auto* config_data = find_section(bundle, "config.json");
    if (!config_data) return true;
    std::string cfg_text(config_data->begin(), config_data->end());
    auto pos = cfg_text.find("\"tokenizer_add_special_tokens\"");
    if (pos == std::string::npos) return true;
    auto val_pos = cfg_text.find(':', pos);
    if (val_pos == std::string::npos) return true;
    auto rest = cfg_text.substr(val_pos + 1, 20);
    return rest.find("false") == std::string::npos;
}

bool is_bpe_tokenizer_json(const BundleFile& bundle)
{
    auto* tok_data = find_section(bundle, "tokenizer.json");
    if (!tok_data || tok_data->empty())
        return false;
    // Quick string search — avoid full JSON parse just for type detection
    std::string_view json(tok_data->data(), tok_data->size());
    return json.find("\"type\":\"BPE\"") != std::string_view::npos
        || json.find("\"type\": \"BPE\"") != std::string_view::npos;
}

std::shared_ptr<ITokenizer> try_create_native_bpe(
    const BundleFile& bundle, bool add_special, bool throw_on_failure)
{
    auto* tok_data = find_section(bundle, "tokenizer.json");
    if (!tok_data || tok_data->empty())
        return nullptr;
    try {
        auto tok = CreateBpeTokenizer(
            tok_data->data(),
            tok_data->size(),
            add_special);
        if (tok) {
            std::cerr << "[trtf] Using native BPE tokenizer" << std::endl;
        }
        return tok;
    } catch (const std::exception& e) {
        // "Not a BPE tokenizer" → non-BPE model (WordPiece, Unigram), allow fallback
        std::string msg = e.what();
        bool is_non_bpe = msg.find("Not a BPE") != std::string::npos;

        if (throw_on_failure || (!is_non_bpe && is_bpe_tokenizer_json(bundle))) {
            // BPE model but native failed → error, no silent fallback
            throw std::runtime_error(
                std::string("Native BPE tokenizer failed for BPE model: ")
                + e.what());
        }
        std::cerr << "[trtf] Native BPE unavailable (" << e.what()
                  << "), falling back to HF Python" << std::endl;
    }
    return nullptr;
}

std::shared_ptr<ITokenizer> try_create_native_tokenizer(
    const BundleFile& bundle, bool add_special_tokens)
{
    auto* tok_data = find_section(bundle, "tokenizer.json");
    if (!tok_data || tok_data->empty())
        return nullptr;

    const char* data = tok_data->data();
    std::size_t size = tok_data->size();

    // Try BPE
    try {
        auto tok = CreateBpeTokenizer(data, size, add_special_tokens);
        if (tok) {
            std::cerr << "[trtf] Using native BPE tokenizer" << std::endl;
            return tok;
        }
    } catch (...) {}

    // Try WordPiece
    try {
        auto tok = CreateWordPieceTokenizer(data, size, add_special_tokens);
        if (tok) {
            std::cerr << "[trtf] Using native WordPiece tokenizer" << std::endl;
            return tok;
        }
    } catch (...) {}

    // Try Unigram (SentencePiece)
    try {
        auto tok = CreateUnigramTokenizer(data, size, add_special_tokens);
        if (tok) {
            std::cerr << "[trtf] Using native Unigram tokenizer" << std::endl;
            return tok;
        }
    } catch (...) {}

    return nullptr;
}

std::shared_ptr<ITokenizer> create_tokenizer_from_bundle(
    const BundleFile& bundle)
{
    bool add_special = detect_add_special_tokens(bundle);
    return try_create_native_tokenizer(bundle, add_special);
}

// ─── TRT module loading ───

LoadedModule load_trt_module_from_plan(
    const std::vector<char>* plan, const char* label,
    std::shared_ptr<CudaStream> shared_stream)
{
    if (!plan || plan->empty())
        throw std::runtime_error(std::string("Bundle missing ") + label);
    auto trt_runtime = create_trt_runtime();
    if (!trt_runtime)
        throw std::runtime_error(std::string("Failed to create TRT runtime for ") + label);
    auto engine = TrtUniquePtr<nvinfer1::ICudaEngine>(
        trt_runtime->deserializeCudaEngine(plan->data(), plan->size()));
    if (!engine)
        throw std::runtime_error(std::string("Failed to deserialize ") + label);
    auto stream = shared_stream ? shared_stream : std::make_shared<CudaStream>();
    if (!stream->ok())
        throw std::runtime_error("Failed to create CUDA stream");
    LoadedModule result;
    result.stream = stream;
    result.module = std::make_unique<TrtModule>(engine.get(), stream->get());
    if (!result.module->ok())
        throw std::runtime_error(std::string("Failed to create TrtModule for ") + label);
    nvinfer1::ICudaEngine* raw_engine = engine.release();
    result.module->keep_alive(std::shared_ptr<nvinfer1::ICudaEngine>(
        raw_engine, [](nvinfer1::ICudaEngine* p) { delete p; }));
    result.module->keep_alive(stream);
    return result;
}

LoadedModule try_load_trt_module_from_plan(
    const std::vector<char>* plan, const char* label,
    std::shared_ptr<CudaStream> shared_stream)
{
    if (!plan || plan->empty()) return LoadedModule{};
    try { return load_trt_module_from_plan(plan, label, shared_stream); }
    catch (...) {
        std::cerr << "[trtf] WARNING: failed to load optional engine: " << label << std::endl;
        return LoadedModule{};
    }
}

std::unique_ptr<TrtModule> extract_optional_module(
    const std::vector<char>* plan, const char* label,
    std::shared_ptr<CudaStream> shared_stream)
{
    auto loaded = try_load_trt_module_from_plan(plan, label, shared_stream);
    if (loaded.module && loaded.module->ok())
        return std::move(loaded.module);
    return nullptr;
}

// ─── Config helpers ───

int32_t compute_kv_dim(const BaseConfig& cfg)
{
    if (cfg.attention_size > 0) return cfg.attention_size;
    int32_t hd = (cfg.head_dim > 0) ? cfg.head_dim
        : ((cfg.num_heads > 0) ? cfg.hidden_size / cfg.num_heads : 128);
    return cfg.num_heads * hd;
}

RecurrentGenConfig make_recurrent_gen_config(const BaseConfig& cfg)
{
    RecurrentGenConfig rgc;
    rgc.vocab_size = cfg.vocab_size;
    rgc.id_bos = cfg.id_bos;
    rgc.id_eos = cfg.id_eos;
    return rgc;
}

// ─── Section data conversion ───

std::vector<float> section_to_floats(const std::vector<char>* sec)
{
    if (!sec || sec->empty()) return {};
    std::size_t count = sec->size() / sizeof(float);
    std::vector<float> out(count);
    std::memcpy(out.data(), sec->data(), count * sizeof(float));
    return out;
}

std::vector<int32_t> section_to_int32s(const std::vector<char>* sec)
{
    if (!sec || sec->empty()) return {};
    std::size_t count = sec->size() / sizeof(int32_t);
    std::vector<int32_t> out(count);
    std::memcpy(out.data(), sec->data(), count * sizeof(int32_t));
    return out;
}

bool has_section_data(const std::vector<char>* d)
{
    return d && !d->empty();
}

MelFilterbank load_mel_filterbank(const BundleFile& bundle)
{
    MelFilterbank fb;
    const auto* data = find_section(bundle, "mel_filterbank");
    if (data == nullptr || data->empty())
        return fb;

    // Format: [n_freq_bins(int32), n_mel_bins(int32), float32 data...]
    if (data->size() < 2 * sizeof(int32_t))
        return fb;

    int32_t header[2] = {0, 0};
    std::memcpy(header, data->data(), sizeof(header));
    fb.n_freq_bins = header[0];
    fb.n_mel_bins = header[1];

    if (fb.n_freq_bins <= 0 || fb.n_mel_bins <= 0)
        return fb;

    const auto expected_data_size = static_cast<std::size_t>(fb.n_freq_bins) *
        static_cast<std::size_t>(fb.n_mel_bins) * sizeof(float);
    const auto payload_offset = 2 * sizeof(int32_t);
    if (data->size() < payload_offset + expected_data_size)
    {
        fb.n_freq_bins = 0;
        fb.n_mel_bins = 0;
        return fb;
    }

    fb.data.resize(static_cast<std::size_t>(fb.n_freq_bins) * fb.n_mel_bins);
    std::memcpy(fb.data.data(), data->data() + payload_offset,
                expected_data_size);
    return fb;
}

std::unique_ptr<ITokenizer> create_clip_tokenizer_from_bundle(
    const BundleFile& bundle)
{
    auto* tok_data = find_section(bundle, "clip_tokenizer.json");
    if (!tok_data || tok_data->empty())
        return nullptr;
    try {
        auto tok = CreateBpeTokenizer(
            tok_data->data(), tok_data->size(), /*add_special_tokens=*/true);
        if (tok)
            std::cerr << "[trtf] Using native BPE CLIP tokenizer" << std::endl;
        return tok;
    } catch (const std::exception& e) {
        std::cerr << "[trtf] WARNING: CLIP tokenizer failed: " << e.what() << std::endl;
    }
    return nullptr;
}

#endif // TRTF_HAS_TRT

} // namespace trtf
