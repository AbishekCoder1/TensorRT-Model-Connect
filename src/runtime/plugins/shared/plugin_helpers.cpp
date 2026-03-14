#include "runtime/plugins/shared/plugin_helpers.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
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

std::shared_ptr<ITokenizer> create_tokenizer_from_bundle(
    const BundleFile& bundle, const std::string& hf_python)
{
    bool add_special = detect_add_special_tokens(bundle);

    // TRTF_NATIVE_TOKENIZER=1  → force native, throw on failure (no fallback)
    // TRTF_NATIVE_TOKENIZER=0  → force HF Python, skip native entirely
    // unset                    → try native first, fallback to HF Python
    const char* env = std::getenv("TRTF_NATIVE_TOKENIZER");
    bool native_disabled = false;
    bool native_forced = false;
    if (env) {
        native_disabled = std::string(env) == "0";
        native_forced = !native_disabled;
    }

    if (!native_disabled) {
        auto tok = try_create_native_bpe(bundle, add_special, native_forced);
        if (tok) return tok;
    }

    // Fall back to HfPythonTokenizer
    if (hf_python.empty()) return nullptr;
    try {
        auto result = extract_tokenizer_from_bundle(bundle, hf_python, add_special);
        if (result.tokenizer) return std::move(result.tokenizer);
    } catch (...) {}
    return nullptr;
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

// ─── Migrated from bundle_helpers (Phase A1: MR-3) ───

namespace {

bool has_non_empty_section(const std::vector<char>* data)
{
    return data != nullptr && !data->empty();
}

std::filesystem::path create_tokenizer_temp_dir(const char* pattern_template)
{
    // mkdtemp modifies the buffer in-place, so copy the template.
    std::string temp_pattern(pattern_template);
    char* created = mkdtemp(temp_pattern.data());
    if (created == nullptr)
    {
        throw std::runtime_error(
            std::string("Failed to create temp dir: ") + pattern_template);
    }
    return std::filesystem::path(created);
}

void write_optional_section(
    const std::filesystem::path& dir,
    const char* filename,
    const std::vector<char>* data)
{
    if (!has_non_empty_section(data))
        return;

    std::ofstream out(dir / filename, std::ios::binary | std::ios::trunc);
    if (!out)
        return;

    out.write(data->data(), static_cast<std::streamsize>(data->size()));
}

} // namespace

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

TokenizerResult extract_tokenizer_from_bundle(
    const BundleFile& bundle,
    const std::string& hf_python,
    bool add_special_tokens)
{
    const auto* tok_json = find_section(bundle, "tokenizer.json");
    const auto* vocab_json = find_section(bundle, "vocab.json");
    const auto* tok_model = find_section(bundle, "tokenizer.model");

    bool has_tokenizer = has_non_empty_section(tok_json)
        || has_non_empty_section(vocab_json)
        || has_non_empty_section(tok_model);
    if (!has_tokenizer)
    {
        throw std::runtime_error("Bundle has no tokenizer files");
    }

    const std::filesystem::path temp_dir =
        create_tokenizer_temp_dir("/tmp/trtfb_tok_XXXXXX");
    std::string temp_dir_str = temp_dir.string();

    write_optional_section(temp_dir, "tokenizer.json", tok_json);
    write_optional_section(temp_dir, "tokenizer_config.json",
                           find_section(bundle, "tokenizer_config.json"));
    write_optional_section(temp_dir, "vocab.json", vocab_json);
    write_optional_section(temp_dir, "merges.txt",
                           find_section(bundle, "merges.txt"));
    write_optional_section(temp_dir, "special_tokens_map.json",
                           find_section(bundle, "special_tokens_map.json"));
    write_optional_section(temp_dir, "tokenizer.model", tok_model);
    write_optional_section(temp_dir, "preprocessor_config.json",
                           find_section(bundle, "preprocessor_config.json"));

    std::cerr << "[trtf] Initializing HF tokenizer from bundle ..." << std::endl;
    auto ttok0 = std::chrono::steady_clock::now();
    auto tokenizer = CreateHfPythonTokenizer(temp_dir_str, hf_python, add_special_tokens);
    auto ttok1 = std::chrono::steady_clock::now();
    std::cerr << "[trtf] Tokenizer ready ["
              << std::chrono::duration_cast<std::chrono::milliseconds>(ttok1 - ttok0).count()
              << " ms]" << std::endl;

    return TokenizerResult{std::move(tokenizer), std::move(temp_dir_str)};
}

TokenizerResult extract_clip_tokenizer_from_bundle(
    const BundleFile& bundle,
    const std::string& hf_python)
{
    const auto* clip_vocab = find_section(bundle, "clip_vocab.json");
    bool has_clip = (clip_vocab != nullptr && !clip_vocab->empty());
    if (!has_clip)
    {
        throw std::runtime_error("Bundle has no CLIP tokenizer files");
    }

    const std::filesystem::path temp_dir =
        create_tokenizer_temp_dir("/tmp/trtfb_clip_XXXXXX");
    std::string temp_dir_str = temp_dir.string();

    // Write CLIP tokenizer files with standard names (HfPythonTokenizer expects them)
    write_optional_section(temp_dir, "vocab.json", clip_vocab);
    write_optional_section(temp_dir, "merges.txt",
                           find_section(bundle, "clip_merges.txt"));
    write_optional_section(temp_dir, "tokenizer_config.json",
                           find_section(bundle, "clip_tokenizer_config.json"));
    write_optional_section(temp_dir, "special_tokens_map.json",
                           find_section(bundle, "clip_special_tokens_map.json"));

    std::cerr << "[trtf] Initializing CLIP tokenizer from bundle ..." << std::endl;
    auto t0 = std::chrono::steady_clock::now();
    auto tokenizer = CreateHfPythonTokenizer(temp_dir_str, hf_python, /*add_special_tokens=*/true);
    auto t1 = std::chrono::steady_clock::now();
    std::cerr << "[trtf] CLIP tokenizer ready ["
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
              << " ms]" << std::endl;

    return TokenizerResult{std::move(tokenizer), std::move(temp_dir_str)};
}

#endif // TRTF_HAS_TRT

} // namespace trtf
