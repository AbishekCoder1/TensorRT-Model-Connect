#include "utils/trt/engine_cache.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string_view>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace trtf {
namespace {

constexpr uint64_t kFnvOffsetBasis = 1469598103934665603ULL;
constexpr uint64_t kFnvPrime = 1099511628211ULL;

bool cache_disabled()
{
    const char* env = std::getenv("TRTF_DISABLE_ENGINE_CACHE");
    if (env == nullptr || env[0] == '\0')
    {
        return false;
    }
    return std::strcmp(env, "0") != 0;
}

std::filesystem::path default_cache_dir()
{
    if (const char* env = std::getenv("TRTF_TRT_ENGINE_CACHE_DIR"); env != nullptr && env[0] != '\0')
    {
        return std::filesystem::path(env);
    }

    if (const char* home = std::getenv("HOME"); home != nullptr && home[0] != '\0')
    {
        return std::filesystem::path(home) / ".cache" / "trtf" / "trt_engine_plans";
    }

    std::error_code ec;
    const std::filesystem::path temp_dir = std::filesystem::temp_directory_path(ec);
    if (!ec)
    {
        return temp_dir / "trtf" / "trt_engine_plans";
    }
    return std::filesystem::path("/tmp") / "trtf" / "trt_engine_plans";
}

std::filesystem::path plan_path_for_key(const std::string& key)
{
    return default_cache_dir() / (key + ".plan");
}

void hash_bytes(uint64_t& hash, const void* data, std::size_t size)
{
    const auto* bytes = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < size; ++i)
    {
        hash ^= static_cast<uint64_t>(bytes[i]);
        hash *= kFnvPrime;
    }
}

template <typename T>
void hash_scalar(uint64_t& hash, const T& value)
{
    hash_bytes(hash, &value, sizeof(T));
}

void hash_string(uint64_t& hash, std::string_view value)
{
    const uint64_t size = static_cast<uint64_t>(value.size());
    hash_scalar(hash, size);
    if (!value.empty())
    {
        hash_bytes(hash, value.data(), value.size());
    }
}

void hash_vector(uint64_t& hash, const std::vector<float>& value)
{
    const uint64_t size = static_cast<uint64_t>(value.size());
    hash_scalar(hash, size);
    if (value.empty())
    {
        return;
    }

    constexpr std::size_t kDirectHashLimit = 2048;
    constexpr std::size_t kWindow = 256;
    constexpr std::size_t kStrideSamples = 512;

    if (value.size() <= kDirectHashLimit)
    {
        hash_bytes(hash, value.data(), value.size() * sizeof(float));
        return;
    }

    const auto hash_slice = [&](std::size_t start, std::size_t count) {
        const std::size_t clamped_start = std::min(start, value.size());
        const std::size_t clamped_count = std::min(count, value.size() - clamped_start);
        if (clamped_count == 0)
        {
            return;
        }
        hash_bytes(hash, value.data() + static_cast<std::ptrdiff_t>(clamped_start),
            clamped_count * sizeof(float));
    };

    hash_slice(0, kWindow);
    hash_slice((value.size() - std::min(kWindow, value.size())) / 2, kWindow);
    hash_slice(value.size() - std::min(kWindow, value.size()), kWindow);

    const std::size_t step = std::max<std::size_t>(std::size_t{1}, value.size() / kStrideSamples);
    for (std::size_t i = 0; i < value.size(); i += step)
    {
        hash_scalar(hash, value[i]);
    }
}

void hash_extra_tensors(uint64_t& hash, const std::unordered_map<std::string, std::vector<float>>& tensors)
{
    std::vector<std::string> keys;
    keys.reserve(tensors.size());
    for (const auto& kv : tensors)
    {
        keys.push_back(kv.first);
    }
    std::sort(keys.begin(), keys.end());
    const uint64_t count = static_cast<uint64_t>(keys.size());
    hash_scalar(hash, count);
    for (const std::string& key : keys)
    {
        hash_string(hash, key);
        hash_vector(hash, tensors.at(key));
    }
}

void hash_layer(uint64_t& hash, const TrtDecoderLayerDefinition& layer)
{
    hash_vector(hash, layer.input_norm);
    hash_vector(hash, layer.q_norm);
    hash_vector(hash, layer.k_norm);
    hash_vector(hash, layer.w_q);
    hash_vector(hash, layer.w_k);
    hash_vector(hash, layer.w_v);
    hash_vector(hash, layer.w_o);
    hash_vector(hash, layer.post_attn_norm);
    hash_vector(hash, layer.w_gate);
    hash_vector(hash, layer.w_up);
    hash_vector(hash, layer.w_down);
    hash_extra_tensors(hash, layer.extra_tensors);
}

} // namespace

std::string BuildTrtEngineCacheKey(const TrtDecoderDefinition& definition, const TrtEngineCacheKeyParams& params)
{
    uint64_t hash = kFnvOffsetBasis;

    hash_string(hash, "trtf-trt-plan-v4");

    hash_scalar(hash, definition.vocab_size);
    hash_scalar(hash, definition.hidden_size);
    hash_scalar(hash, definition.attention_size);
    hash_scalar(hash, definition.mlp_size);
    hash_scalar(hash, definition.max_cache_length);
    hash_scalar(hash, definition.id_bos);
    hash_scalar(hash, definition.id_eos);
    hash_scalar(hash, definition.has_decoder_layers ? 1 : 0);
    hash_scalar(hash, definition.rms_norm_eps);
    hash_scalar(hash, definition.num_attention_heads);
    hash_scalar(hash, definition.num_key_value_heads);
    hash_scalar(hash, definition.rope_theta);

    hash_scalar(hash, params.requires_position_input ? 1 : 0);
    hash_scalar(hash, params.num_layers);

    hash_vector(hash, definition.embedding);
    hash_vector(hash, definition.w_q);
    hash_vector(hash, definition.w_k);
    hash_vector(hash, definition.w_v);
    hash_vector(hash, definition.w1);
    hash_vector(hash, definition.b1);
    hash_vector(hash, definition.w2);
    hash_vector(hash, definition.b2);
    hash_vector(hash, definition.w_out);
    hash_vector(hash, definition.b_out);
    hash_vector(hash, definition.final_norm);

    const uint64_t num_decoder_layers = static_cast<uint64_t>(definition.decoder_layers.size());
    hash_scalar(hash, num_decoder_layers);
    for (const TrtDecoderLayerDefinition& layer : definition.decoder_layers)
    {
        hash_layer(hash, layer);
    }

    // Hash extra params (sorted for determinism)
    {
        std::vector<std::string> int_keys;
        int_keys.reserve(definition.extra_int_params.size());
        for (const auto& kv : definition.extra_int_params)
        {
            int_keys.push_back(kv.first);
        }
        std::sort(int_keys.begin(), int_keys.end());
        const uint64_t int_count = static_cast<uint64_t>(int_keys.size());
        hash_scalar(hash, int_count);
        for (const std::string& key : int_keys)
        {
            hash_string(hash, key);
            hash_scalar(hash, definition.extra_int_params.at(key));
        }

        std::vector<std::string> float_keys;
        float_keys.reserve(definition.extra_float_params.size());
        for (const auto& kv : definition.extra_float_params)
        {
            float_keys.push_back(kv.first);
        }
        std::sort(float_keys.begin(), float_keys.end());
        const uint64_t float_count = static_cast<uint64_t>(float_keys.size());
        hash_scalar(hash, float_count);
        for (const std::string& key : float_keys)
        {
            hash_string(hash, key);
            hash_scalar(hash, definition.extra_float_params.at(key));
        }

        hash_extra_tensors(hash, definition.extra_tensors);
    }

    std::ostringstream oss;
    oss << std::hex << std::setfill('0') << std::setw(16) << hash;
    return oss.str();
}

std::optional<std::vector<char>> LoadTrtEnginePlanFromCache(const std::string& cache_key)
{
    if (cache_disabled() || cache_key.empty())
    {
        return std::nullopt;
    }

    const std::filesystem::path path = plan_path_for_key(cache_key);
    const int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0)
    {
        return std::nullopt;
    }

    struct stat st{};
    if (fstat(fd, &st) != 0 || st.st_size <= 0)
    {
        close(fd);
        return std::nullopt;
    }

    const std::size_t file_size = static_cast<std::size_t>(st.st_size);

    // mmap the file for fast access — avoids copying 2+GB through userspace read()
    void* mapped = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (mapped == MAP_FAILED)
    {
        return std::nullopt;
    }

    // Advise sequential access for read-ahead
    madvise(mapped, file_size, MADV_SEQUENTIAL);

    std::vector<char> bytes(static_cast<const char*>(mapped),
        static_cast<const char*>(mapped) + file_size);
    munmap(mapped, file_size);

    return bytes;
}

void SaveTrtEnginePlanToCache(const std::string& cache_key, const void* plan_data, std::size_t plan_size)
{
    if (cache_disabled() || cache_key.empty() || plan_data == nullptr || plan_size == 0)
    {
        return;
    }

    const std::filesystem::path plan_path = plan_path_for_key(cache_key);

    std::error_code ec;
    std::filesystem::create_directories(plan_path.parent_path(), ec);
    if (ec)
    {
        return;
    }

    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path tmp_path = plan_path.string() + ".tmp." + std::to_string(now);

    {
        std::ofstream out(tmp_path, std::ios::binary | std::ios::trunc);
        if (!out)
        {
            return;
        }
        out.write(static_cast<const char*>(plan_data), static_cast<std::streamsize>(plan_size));
        if (!out)
        {
            std::error_code remove_ec;
            std::filesystem::remove(tmp_path, remove_ec);
            return;
        }
    }

    std::filesystem::rename(tmp_path, plan_path, ec);
    if (ec)
    {
        std::error_code remove_ec;
        std::filesystem::remove(tmp_path, remove_ec);
    }
}

std::string BuildModelDirIndexKey(const std::string& model_dir, int32_t max_cache_length)
{
    uint64_t hash = kFnvOffsetBasis;
    hash_string(hash, "trtf-model-dir-idx-v1");

    // Hash the canonical model directory path
    std::error_code ec;
    const std::string canonical = std::filesystem::canonical(model_dir, ec).string();
    hash_string(hash, ec ? model_dir : canonical);

    // Hash config.json content (captures all architecture params)
    const std::filesystem::path config_path = std::filesystem::path(model_dir) / "config.json";
    {
        std::ifstream in(config_path);
        if (in)
        {
            const std::string content((std::istreambuf_iterator<char>(in)),
                std::istreambuf_iterator<char>());
            hash_string(hash, content);
        }
    }

    // Hash safetensors file sizes (catches weight changes without reading data)
    const std::filesystem::path st_path = std::filesystem::path(model_dir) / "model.safetensors";
    const std::filesystem::path idx_path = std::filesystem::path(model_dir) / "model.safetensors.index.json";
    if (std::filesystem::exists(st_path))
    {
        const uint64_t sz = static_cast<uint64_t>(std::filesystem::file_size(st_path, ec));
        hash_scalar(hash, sz);
    }
    if (std::filesystem::exists(idx_path))
    {
        std::ifstream in(idx_path);
        if (in)
        {
            const std::string content((std::istreambuf_iterator<char>(in)),
                std::istreambuf_iterator<char>());
            hash_string(hash, content);
        }
    }

    hash_scalar(hash, max_cache_length);

    std::ostringstream oss;
    oss << std::hex << std::setfill('0') << std::setw(16) << hash;
    return oss.str();
}

void SaveModelDirIndex(const std::string& index_key, const std::string& cache_key)
{
    if (cache_disabled() || index_key.empty() || cache_key.empty())
    {
        return;
    }

    const std::filesystem::path idx_path = default_cache_dir() / (index_key + ".idx");

    std::error_code ec;
    std::filesystem::create_directories(idx_path.parent_path(), ec);
    if (ec)
    {
        return;
    }

    std::ofstream out(idx_path, std::ios::trunc);
    if (out)
    {
        out << cache_key;
    }
}

std::optional<std::string> LookupModelDirIndex(const std::string& index_key)
{
    if (cache_disabled() || index_key.empty())
    {
        return std::nullopt;
    }

    const std::filesystem::path idx_path = default_cache_dir() / (index_key + ".idx");
    std::ifstream in(idx_path);
    if (!in)
    {
        return std::nullopt;
    }

    std::string cache_key;
    std::getline(in, cache_key);
    if (cache_key.empty())
    {
        return std::nullopt;
    }

    // Verify the plan file still exists
    if (!std::filesystem::exists(plan_path_for_key(cache_key)))
    {
        return std::nullopt;
    }

    return cache_key;
}

} // namespace trtf
