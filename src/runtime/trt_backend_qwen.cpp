#include "trtf/backend.h"
#include "trtf/model.h"
#include "trtf/tokenizer.h"
#include "model/trt_model_definition.h"
#include "utils/trt/engine_cache.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if TRTF_HAS_TRT
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>
#endif

namespace trtf {
namespace {

#if TRTF_HAS_TRT
const char* trt_severity_name(nvinfer1::ILogger::Severity severity)
{
    switch (severity)
    {
    case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
        return "INTERNAL_ERROR";
    case nvinfer1::ILogger::Severity::kERROR:
        return "ERROR";
    case nvinfer1::ILogger::Severity::kWARNING:
        return "WARNING";
    case nvinfer1::ILogger::Severity::kINFO:
        return "INFO";
    case nvinfer1::ILogger::Severity::kVERBOSE:
        return "VERBOSE";
    default:
        return "UNKNOWN";
    }
}

bool trt_log_to_stderr_enabled()
{
    static const bool enabled = [] {
        const char* env = std::getenv("TRTF_TRT_LOG_STDERR");
        if (env == nullptr || env[0] == '\0')
        {
            return false;
        }
        return std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

nvinfer1::ILogger::Severity trt_log_stderr_min_severity()
{
    static const nvinfer1::ILogger::Severity severity = [] {
        const char* env = std::getenv("TRTF_TRT_LOG_MIN_SEVERITY");
        if (env == nullptr || env[0] == '\0')
        {
            return nvinfer1::ILogger::Severity::kINFO;
        }

        std::string value(env);
        std::transform(value.begin(), value.end(), value.begin(),
            [](unsigned char c) { return static_cast<char>(std::toupper(c)); });

        if (value == "INTERNAL_ERROR")
        {
            return nvinfer1::ILogger::Severity::kINTERNAL_ERROR;
        }
        if (value == "ERROR")
        {
            return nvinfer1::ILogger::Severity::kERROR;
        }
        if (value == "WARNING")
        {
            return nvinfer1::ILogger::Severity::kWARNING;
        }
        if (value == "VERBOSE")
        {
            return nvinfer1::ILogger::Severity::kVERBOSE;
        }
        return nvinfer1::ILogger::Severity::kINFO;
    }();
    return severity;
}

class TrtLogger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity <= Severity::kERROR && msg != nullptr)
        {
            mLastError = msg;
        }

        if (msg != nullptr && trt_log_to_stderr_enabled() && severity <= trt_log_stderr_min_severity())
        {
            std::cerr << "TRT_LOG[" << trt_severity_name(severity) << "] " << msg << '\n';
        }
    }

    const std::string& last_error() const
    {
        return mLastError;
    }

    void clear_error()
    {
        mLastError.clear();
    }

private:
    std::string mLastError;
};

template <typename T>
struct TrtDeleter {
    void operator()(T* ptr) const noexcept
    {
        if (ptr == nullptr)
        {
            return;
        }
#if NV_TENSORRT_MAJOR >= 10
        delete ptr;
#else
        ptr->destroy();
#endif
    }
};

template <typename T>
using TrtUniquePtr = std::unique_ptr<T, TrtDeleter<T>>;

class CudaStream final {
public:
    CudaStream()
    {
        mStatus = cudaStreamCreate(&mStream);
    }

    ~CudaStream()
    {
        if (mStream != nullptr)
        {
            cudaStreamDestroy(mStream);
        }
    }

    bool ok() const
    {
        return mStatus == cudaSuccess;
    }

    cudaStream_t get() const
    {
        return mStream;
    }

private:
    cudaStream_t mStream{nullptr};
    cudaError_t mStatus{cudaSuccess};
};

class CudaBuffer final {
public:
    explicit CudaBuffer(std::size_t bytes)
        : mBytes(bytes)
    {
        if (mBytes == 0)
        {
            return;
        }
        mStatus = cudaMalloc(&mPtr, mBytes);
    }

    ~CudaBuffer()
    {
        if (mPtr != nullptr)
        {
            cudaFree(mPtr);
        }
    }

    bool ok() const
    {
        return mStatus == cudaSuccess;
    }

    void* data() const
    {
        return mPtr;
    }

private:
    void* mPtr{nullptr};
    std::size_t mBytes{0};
    cudaError_t mStatus{cudaSuccess};
};

constexpr int32_t kDefaultMaxCacheLength = 32;
constexpr float kMaskedScore = -1.0e4F;

struct DecoderStepEngine {
    TrtUniquePtr<nvinfer1::ICudaEngine> engine;
    TrtUniquePtr<nvinfer1::IExecutionContext> context;

    std::string token_input_name{"token_id"};
    std::string position_input_name{"position_id"};
    std::string mask_input_name{"attention_mask"};
    std::string logits_output_name{"logits"};
    std::vector<std::string> cache_k_input_names;
    std::vector<std::string> cache_v_input_names;
    std::vector<std::string> present_k_output_names;
    std::vector<std::string> present_v_output_names;

    int32_t num_layers{1};
    bool requires_position_input{false};
    int32_t vocab_size{0};
    int32_t hidden_size{0};
    int32_t cache_state_size{0};
    int32_t attention_mask_size{0};
    int32_t max_cache_length{kDefaultMaxCacheLength};
};

nvinfer1::Dims make_dims_1d(int32_t d0)
{
    nvinfer1::Dims dims{};
    dims.nbDims = 1;
    dims.d[0] = d0;
    return dims;
}

nvinfer1::Dims make_dims_2d(int32_t d0, int32_t d1)
{
    nvinfer1::Dims dims{};
    dims.nbDims = 2;
    dims.d[0] = d0;
    dims.d[1] = d1;
    return dims;
}

nvinfer1::Dims make_dims_3d(int32_t d0, int32_t d1, int32_t d2)
{
    nvinfer1::Dims dims{};
    dims.nbDims = 3;
    dims.d[0] = d0;
    dims.d[1] = d1;
    dims.d[2] = d2;
    return dims;
}

std::string layer_tensor_name(const char* stem, int32_t layer)
{
    return std::string(stem) + "_" + std::to_string(layer);
}

nvinfer1::ITensor* add_constant_tensor(
    nvinfer1::INetworkDefinition& network, nvinfer1::Dims dims, const std::vector<float>& values)
{
    const nvinfer1::Weights weights{
        nvinfer1::DataType::kFLOAT, values.data(), static_cast<int64_t>(values.size())};
    auto* layer = network.addConstant(dims, weights);
    if (layer == nullptr)
    {
        return nullptr;
    }
    return layer->getOutput(0);
}

nvinfer1::ITensor* add_matmul_rhs_constant(nvinfer1::INetworkDefinition& network, nvinfer1::ITensor& lhs,
    int32_t lhs_width, int32_t rhs_width, const std::vector<float>& rhs_weights)
{
    nvinfer1::ITensor* rhs = add_constant_tensor(network, make_dims_2d(lhs_width, rhs_width), rhs_weights);
    if (rhs == nullptr)
    {
        return nullptr;
    }

    auto* matmul
        = network.addMatrixMultiply(lhs, nvinfer1::MatrixOperation::kNONE, *rhs, nvinfer1::MatrixOperation::kNONE);
    if (matmul == nullptr)
    {
        return nullptr;
    }
    return matmul->getOutput(0);
}

nvinfer1::ITensor* add_bias_sum(
    nvinfer1::INetworkDefinition& network, nvinfer1::ITensor& input, int32_t width, const std::vector<float>& bias)
{
    nvinfer1::ITensor* bias_tensor = add_constant_tensor(network, make_dims_2d(1, width), bias);
    if (bias_tensor == nullptr)
    {
        return nullptr;
    }

    auto* sum = network.addElementWise(input, *bias_tensor, nvinfer1::ElementWiseOperation::kSUM);
    if (sum == nullptr)
    {
        return nullptr;
    }
    return sum->getOutput(0);
}

nvinfer1::ITensor* add_rms_norm(nvinfer1::INetworkDefinition& network, nvinfer1::ITensor& input, int32_t hidden_size,
    const std::vector<float>& gamma, nvinfer1::ITensor& eps_tensor)
{
    auto* sq = network.addElementWise(input, input, nvinfer1::ElementWiseOperation::kPROD);
    if (sq == nullptr)
    {
        return nullptr;
    }

    auto* mean = network.addReduce(*sq->getOutput(0), nvinfer1::ReduceOperation::kAVG, 1U << 1, true);
    if (mean == nullptr)
    {
        return nullptr;
    }

    auto* denom_in = network.addElementWise(*mean->getOutput(0), eps_tensor, nvinfer1::ElementWiseOperation::kSUM);
    if (denom_in == nullptr)
    {
        return nullptr;
    }

    auto* sqrt_layer = network.addUnary(*denom_in->getOutput(0), nvinfer1::UnaryOperation::kSQRT);
    if (sqrt_layer == nullptr)
    {
        return nullptr;
    }

    auto* recip = network.addUnary(*sqrt_layer->getOutput(0), nvinfer1::UnaryOperation::kRECIP);
    if (recip == nullptr)
    {
        return nullptr;
    }

    auto* normalized = network.addElementWise(input, *recip->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);
    if (normalized == nullptr)
    {
        return nullptr;
    }

    nvinfer1::ITensor* gamma_tensor = add_constant_tensor(network, make_dims_2d(1, hidden_size), gamma);
    if (gamma_tensor == nullptr)
    {
        return nullptr;
    }

    auto* scaled
        = network.addElementWise(*normalized->getOutput(0), *gamma_tensor, nvinfer1::ElementWiseOperation::kPROD);
    if (scaled == nullptr)
    {
        return nullptr;
    }
    return scaled->getOutput(0);
}

nvinfer1::ITensor* add_rms_norm_per_head(nvinfer1::INetworkDefinition& network, nvinfer1::ITensor& input,
    int32_t num_heads, int32_t head_dim, const std::vector<float>& gamma, nvinfer1::ITensor& eps_tensor)
{
    if (num_heads <= 0 || head_dim <= 0 || gamma.size() != static_cast<std::size_t>(num_heads) * static_cast<std::size_t>(head_dim))
    {
        return nullptr;
    }

    auto* reshape_in = network.addShuffle(input);
    if (reshape_in == nullptr)
    {
        return nullptr;
    }
    reshape_in->setReshapeDimensions(make_dims_2d(num_heads, head_dim));

    nvinfer1::ITensor* reshaped = reshape_in->getOutput(0);
    auto* sq = network.addElementWise(*reshaped, *reshaped, nvinfer1::ElementWiseOperation::kPROD);
    if (sq == nullptr)
    {
        return nullptr;
    }

    auto* mean = network.addReduce(*sq->getOutput(0), nvinfer1::ReduceOperation::kAVG, 1U << 1, true);
    if (mean == nullptr)
    {
        return nullptr;
    }

    auto* denom_in = network.addElementWise(*mean->getOutput(0), eps_tensor, nvinfer1::ElementWiseOperation::kSUM);
    if (denom_in == nullptr)
    {
        return nullptr;
    }

    auto* sqrt_layer = network.addUnary(*denom_in->getOutput(0), nvinfer1::UnaryOperation::kSQRT);
    if (sqrt_layer == nullptr)
    {
        return nullptr;
    }

    auto* recip = network.addUnary(*sqrt_layer->getOutput(0), nvinfer1::UnaryOperation::kRECIP);
    if (recip == nullptr)
    {
        return nullptr;
    }

    auto* normalized = network.addElementWise(*reshaped, *recip->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);
    if (normalized == nullptr)
    {
        return nullptr;
    }

    nvinfer1::ITensor* gamma_tensor = add_constant_tensor(network, make_dims_2d(num_heads, head_dim), gamma);
    if (gamma_tensor == nullptr)
    {
        return nullptr;
    }

    auto* scaled
        = network.addElementWise(*normalized->getOutput(0), *gamma_tensor, nvinfer1::ElementWiseOperation::kPROD);
    if (scaled == nullptr)
    {
        return nullptr;
    }

    auto* reshape_out = network.addShuffle(*scaled->getOutput(0));
    if (reshape_out == nullptr)
    {
        return nullptr;
    }
    reshape_out->setReshapeDimensions(make_dims_2d(1, num_heads * head_dim));
    return reshape_out->getOutput(0);
}

std::vector<float> make_rope_table(int32_t max_cache_length, int32_t hidden_size, int32_t num_attention_heads,
    float rope_theta, bool cosine)
{
    std::vector<float> table(
        static_cast<std::size_t>(max_cache_length) * static_cast<std::size_t>(hidden_size), cosine ? 1.0F : 0.0F);
    if (max_cache_length <= 0 || hidden_size <= 0 || num_attention_heads <= 0 || hidden_size % num_attention_heads != 0)
    {
        return table;
    }

    const int32_t head_dim = hidden_size / num_attention_heads;
    const int32_t half_head_dim = head_dim / 2;
    if (half_head_dim <= 0 || rope_theta <= 0.0F)
    {
        return table;
    }

    for (int32_t pos = 0; pos < max_cache_length; ++pos)
    {
        for (int32_t head = 0; head < num_attention_heads; ++head)
        {
            // HF Qwen3 rotary embedding builds emb = cat(freqs, freqs) across the head dimension.
            const int32_t rope_dims = half_head_dim * 2;
            for (int32_t dim = 0; dim < rope_dims; ++dim)
            {
                const int32_t freq_idx = dim % half_head_dim;
                const float exponent = (2.0F * static_cast<float>(freq_idx)) / static_cast<float>(head_dim);
                const float inv_freq = std::pow(rope_theta, -exponent);
                const float angle = static_cast<float>(pos) * inv_freq;
                const float value = cosine ? std::cos(angle) : std::sin(angle);
                const std::size_t offset = static_cast<std::size_t>(pos) * static_cast<std::size_t>(hidden_size)
                    + static_cast<std::size_t>(head) * static_cast<std::size_t>(head_dim)
                    + static_cast<std::size_t>(dim);
                table[offset] = value;
            }
        }
    }

    return table;
}

std::vector<float> make_rotate_half_matrix(int32_t hidden_size, int32_t num_attention_heads)
{
    std::vector<float> matrix(
        static_cast<std::size_t>(hidden_size) * static_cast<std::size_t>(hidden_size), 0.0F);
    if (hidden_size <= 0 || num_attention_heads <= 0 || hidden_size % num_attention_heads != 0)
    {
        for (int32_t i = 0; i < hidden_size; ++i)
        {
            matrix[static_cast<std::size_t>(i) * static_cast<std::size_t>(hidden_size) + static_cast<std::size_t>(i)]
                = 1.0F;
        }
        return matrix;
    }

    const int32_t head_dim = hidden_size / num_attention_heads;
    const int32_t half_head_dim = head_dim / 2;

    for (int32_t head = 0; head < num_attention_heads; ++head)
    {
        const int32_t base = head * head_dim;
        for (int32_t i = 0; i < half_head_dim; ++i)
        {
            const int32_t out_left = base + i;
            const int32_t out_right = base + half_head_dim + i;

            // For row-vector matmul (x * M), set coefficients so output is rotate_half(x).
            matrix[static_cast<std::size_t>(out_left) * static_cast<std::size_t>(hidden_size)
                + static_cast<std::size_t>(out_right)] = 1.0F;
            matrix[static_cast<std::size_t>(out_right) * static_cast<std::size_t>(hidden_size)
                + static_cast<std::size_t>(out_left)] = -1.0F;
        }

        if (head_dim % 2 != 0)
        {
            const int32_t tail = base + (2 * half_head_dim);
            matrix[static_cast<std::size_t>(tail) * static_cast<std::size_t>(hidden_size)
                + static_cast<std::size_t>(tail)] = 1.0F;
        }
    }
    return matrix;
}

nvinfer1::ITensor* add_apply_rope(nvinfer1::INetworkDefinition& network, nvinfer1::ITensor& input,
    nvinfer1::ITensor& position_id, nvinfer1::ITensor& cos_table, nvinfer1::ITensor& sin_table,
    nvinfer1::ITensor& rotate_half_matrix)
{
    auto* cos_gather = network.addGather(cos_table, position_id, 0);
    auto* sin_gather = network.addGather(sin_table, position_id, 0);
    if (cos_gather == nullptr || sin_gather == nullptr)
    {
        return nullptr;
    }

    auto* rotated = network.addMatrixMultiply(
        input, nvinfer1::MatrixOperation::kNONE, rotate_half_matrix, nvinfer1::MatrixOperation::kNONE);
    if (rotated == nullptr)
    {
        return nullptr;
    }

    auto* x_cos = network.addElementWise(
        input, *cos_gather->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);
    if (x_cos == nullptr)
    {
        return nullptr;
    }

    auto* rot_sin = network.addElementWise(
        *rotated->getOutput(0), *sin_gather->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);
    if (rot_sin == nullptr)
    {
        return nullptr;
    }

    auto* sum = network.addElementWise(
        *x_cos->getOutput(0), *rot_sin->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
    if (sum == nullptr)
    {
        return nullptr;
    }
    return sum->getOutput(0);
}

bool has_io_tensor(const nvinfer1::ICudaEngine& engine, const std::string& tensor_name)
{
    const int32_t count = engine.getNbIOTensors();
    for (int32_t i = 0; i < count; ++i)
    {
        const char* candidate = engine.getIOTensorName(i);
        if (candidate != nullptr && tensor_name == candidate)
        {
            return true;
        }
    }
    return false;
}

bool has_all_required_tensors(const DecoderStepEngine& engine)
{
    if (!has_io_tensor(*engine.engine, engine.token_input_name) || !has_io_tensor(*engine.engine, engine.mask_input_name)
        || !has_io_tensor(*engine.engine, engine.logits_output_name))
    {
        return false;
    }
    if (engine.requires_position_input && !has_io_tensor(*engine.engine, engine.position_input_name))
    {
        return false;
    }

    for (int32_t i = 0; i < engine.num_layers; ++i)
    {
        if (!has_io_tensor(*engine.engine, engine.cache_k_input_names[static_cast<std::size_t>(i)])
            || !has_io_tensor(*engine.engine, engine.cache_v_input_names[static_cast<std::size_t>(i)])
            || !has_io_tensor(*engine.engine, engine.present_k_output_names[static_cast<std::size_t>(i)])
            || !has_io_tensor(*engine.engine, engine.present_v_output_names[static_cast<std::size_t>(i)]))
        {
            return false;
        }
    }
    return true;
}

std::unique_ptr<DecoderStepEngine> finalize_decoder_step_engine(nvinfer1::IBuilder& builder,
    nvinfer1::INetworkDefinition& network, nvinfer1::IBuilderConfig& config, TrtLogger& logger,
    const TrtDecoderDefinition& weights, const std::vector<std::string>& cache_k_input_names,
    const std::vector<std::string>& cache_v_input_names, const std::vector<std::string>& present_k_output_names,
    const std::vector<std::string>& present_v_output_names, bool requires_position_input)
{
    auto runtime = TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
    if (!runtime)
    {
        return nullptr;
    }

    TrtEngineCacheKeyParams cache_key_params;
    cache_key_params.requires_position_input = requires_position_input;
    cache_key_params.num_layers = static_cast<int32_t>(cache_k_input_names.size());
    const std::string cache_key = BuildTrtEngineCacheKey(weights, cache_key_params);

    TrtUniquePtr<nvinfer1::ICudaEngine> trt_engine;
    if (const auto cached_plan = LoadTrtEnginePlanFromCache(cache_key))
    {
        trt_engine = TrtUniquePtr<nvinfer1::ICudaEngine>(
            runtime->deserializeCudaEngine(cached_plan->data(), cached_plan->size()));
    }

    if (!trt_engine)
    {
        auto plan = TrtUniquePtr<nvinfer1::IHostMemory>(builder.buildSerializedNetwork(network, config));
        if (!plan)
        {
            return nullptr;
        }

        trt_engine = TrtUniquePtr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan->data(), plan->size()));
        if (!trt_engine)
        {
            return nullptr;
        }

        SaveTrtEnginePlanToCache(cache_key, plan->data(), plan->size());
    }

    auto execution_context = TrtUniquePtr<nvinfer1::IExecutionContext>(trt_engine->createExecutionContext());
    if (!execution_context)
    {
        return nullptr;
    }

    auto out = std::make_unique<DecoderStepEngine>();
    out->engine = std::move(trt_engine);
    out->context = std::move(execution_context);
    out->vocab_size = weights.vocab_size;
    out->hidden_size = weights.hidden_size;
    out->cache_state_size = (weights.has_qwen_layers && weights.attention_size > 0)
        ? weights.attention_size
        : weights.hidden_size;
    out->attention_mask_size = requires_position_input ? (weights.max_cache_length + 1) : weights.max_cache_length;
    out->max_cache_length = weights.max_cache_length;
    out->num_layers = static_cast<int32_t>(cache_k_input_names.size());
    out->requires_position_input = requires_position_input;
    out->cache_k_input_names = cache_k_input_names;
    out->cache_v_input_names = cache_v_input_names;
    out->present_k_output_names = present_k_output_names;
    out->present_v_output_names = present_v_output_names;

    if (!has_all_required_tensors(*out))
    {
        return nullptr;
    }
    return out;
}

std::unique_ptr<DecoderStepEngine> create_decoder_step_engine_legacy(
    const TrtDecoderDefinition& weights, TrtLogger& logger)
{
    auto builder = TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    if (!builder)
    {
        return nullptr;
    }

    uint32_t flags = 0U;
#if NV_TENSORRT_MAJOR < 10
    flags = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
#endif
    auto network = TrtUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(flags));
    auto config = TrtUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!network || !config)
    {
        return nullptr;
    }

#if NV_TENSORRT_MAJOR >= 8
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1U << 20);
    config->clearFlag(nvinfer1::BuilderFlag::kTF32);
#endif

    const std::string cache_k_name = "cache_k";
    const std::string cache_v_name = "cache_v";
    const std::string present_k_name = "present_k";
    const std::string present_v_name = "present_v";

    auto* token_id = network->addInput("token_id", nvinfer1::DataType::kINT32, make_dims_1d(1));
    auto* cache_k = network->addInput(
        cache_k_name.c_str(), nvinfer1::DataType::kFLOAT, make_dims_2d(weights.max_cache_length, weights.hidden_size));
    auto* cache_v = network->addInput(
        cache_v_name.c_str(), nvinfer1::DataType::kFLOAT, make_dims_2d(weights.max_cache_length, weights.hidden_size));
    auto* attention_mask = network->addInput(
        "attention_mask", nvinfer1::DataType::kFLOAT, make_dims_2d(1, weights.max_cache_length));
    if (token_id == nullptr || cache_k == nullptr || cache_v == nullptr || attention_mask == nullptr)
    {
        return nullptr;
    }

    nvinfer1::ITensor* embedding_table
        = add_constant_tensor(*network, make_dims_2d(weights.vocab_size, weights.hidden_size), weights.embedding);
    if (embedding_table == nullptr)
    {
        return nullptr;
    }

    auto* gather = network->addGather(*embedding_table, *token_id, 0);
    if (gather == nullptr)
    {
        return nullptr;
    }
    nvinfer1::ITensor* token_hidden = gather->getOutput(0);

    nvinfer1::ITensor* q
        = add_matmul_rhs_constant(*network, *token_hidden, weights.hidden_size, weights.hidden_size, weights.w_q);
    nvinfer1::ITensor* k
        = add_matmul_rhs_constant(*network, *token_hidden, weights.hidden_size, weights.hidden_size, weights.w_k);
    nvinfer1::ITensor* v
        = add_matmul_rhs_constant(*network, *token_hidden, weights.hidden_size, weights.hidden_size, weights.w_v);
    if (q == nullptr || k == nullptr || v == nullptr)
    {
        return nullptr;
    }

    auto* score_matmul
        = network->addMatrixMultiply(*q, nvinfer1::MatrixOperation::kNONE, *cache_k, nvinfer1::MatrixOperation::kTRANSPOSE);
    if (score_matmul == nullptr)
    {
        return nullptr;
    }

    const float scale = 1.0F / std::sqrt(static_cast<float>(weights.hidden_size));
    const std::vector<float> score_scale{scale};
    nvinfer1::ITensor* scale_tensor = add_constant_tensor(*network, make_dims_2d(1, 1), score_scale);
    if (scale_tensor == nullptr)
    {
        return nullptr;
    }

    auto* scaled_scores
        = network->addElementWise(*score_matmul->getOutput(0), *scale_tensor, nvinfer1::ElementWiseOperation::kPROD);
    if (scaled_scores == nullptr)
    {
        return nullptr;
    }

    auto* masked_scores
        = network->addElementWise(*scaled_scores->getOutput(0), *attention_mask, nvinfer1::ElementWiseOperation::kSUM);
    if (masked_scores == nullptr)
    {
        return nullptr;
    }

    auto* softmax = network->addSoftMax(*masked_scores->getOutput(0));
    if (softmax == nullptr)
    {
        return nullptr;
    }
    softmax->setAxes(1U << 1);

    auto* context_layer
        = network->addMatrixMultiply(*softmax->getOutput(0), nvinfer1::MatrixOperation::kNONE, *cache_v, nvinfer1::MatrixOperation::kNONE);
    if (context_layer == nullptr)
    {
        return nullptr;
    }

    auto* residual1
        = network->addElementWise(*token_hidden, *context_layer->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
    if (residual1 == nullptr)
    {
        return nullptr;
    }

    nvinfer1::ITensor* mlp_fc1
        = add_matmul_rhs_constant(*network, *residual1->getOutput(0), weights.hidden_size, weights.mlp_size, weights.w1);
    if (mlp_fc1 == nullptr)
    {
        return nullptr;
    }

    nvinfer1::ITensor* mlp_fc1_bias = add_bias_sum(*network, *mlp_fc1, weights.mlp_size, weights.b1);
    if (mlp_fc1_bias == nullptr)
    {
        return nullptr;
    }

    auto* mlp_relu = network->addActivation(*mlp_fc1_bias, nvinfer1::ActivationType::kRELU);
    if (mlp_relu == nullptr)
    {
        return nullptr;
    }

    nvinfer1::ITensor* mlp_fc2
        = add_matmul_rhs_constant(*network, *mlp_relu->getOutput(0), weights.mlp_size, weights.hidden_size, weights.w2);
    if (mlp_fc2 == nullptr)
    {
        return nullptr;
    }

    nvinfer1::ITensor* mlp_fc2_bias = add_bias_sum(*network, *mlp_fc2, weights.hidden_size, weights.b2);
    if (mlp_fc2_bias == nullptr)
    {
        return nullptr;
    }

    auto* residual2
        = network->addElementWise(*residual1->getOutput(0), *mlp_fc2_bias, nvinfer1::ElementWiseOperation::kSUM);
    if (residual2 == nullptr)
    {
        return nullptr;
    }

    nvinfer1::ITensor* logits
        = add_matmul_rhs_constant(*network, *residual2->getOutput(0), weights.hidden_size, weights.vocab_size, weights.w_out);
    if (logits == nullptr)
    {
        return nullptr;
    }

    nvinfer1::ITensor* logits_bias = add_bias_sum(*network, *logits, weights.vocab_size, weights.b_out);
    if (logits_bias == nullptr)
    {
        return nullptr;
    }

    logits_bias->setName("logits");
    k->setName(present_k_name.c_str());
    v->setName(present_v_name.c_str());
    network->markOutput(*logits_bias);
    network->markOutput(*k);
    network->markOutput(*v);

    return finalize_decoder_step_engine(*builder, *network, *config, logger, weights,
        {cache_k_name}, {cache_v_name}, {present_k_name}, {present_v_name}, false);
}

struct QwenLayerTensors {
    nvinfer1::ITensor* hidden{nullptr};
    nvinfer1::ITensor* present_k{nullptr};
    nvinfer1::ITensor* present_v{nullptr};
};

QwenLayerTensors add_qwen_layer_block(nvinfer1::INetworkDefinition& network, const TrtDecoderDefinition& weights,
    const TrtDecoderLayerDefinition& layer, nvinfer1::ITensor& hidden, nvinfer1::ITensor& cache_k,
    nvinfer1::ITensor& cache_v,
    nvinfer1::ITensor& attention_mask, nvinfer1::ITensor& position_id, nvinfer1::ITensor& cos_table,
    nvinfer1::ITensor& sin_table, nvinfer1::ITensor& rotate_half_matrix,
    nvinfer1::ITensor& attention_scale_tensor, nvinfer1::ITensor& eps_tensor)
{
    QwenLayerTensors out;
    const int32_t attention_size = weights.attention_size > 0 ? weights.attention_size : weights.hidden_size;
    if (attention_size <= 0 || weights.num_attention_heads <= 0 || attention_size % weights.num_attention_heads != 0)
    {
        return out;
    }

    const int32_t head_dim = attention_size / weights.num_attention_heads;
    const int32_t attention_window = weights.max_cache_length + 1;

    nvinfer1::ITensor* norm1 = add_rms_norm(network, hidden, weights.hidden_size, layer.input_norm, eps_tensor);
    if (norm1 == nullptr)
    {
        return out;
    }

    nvinfer1::ITensor* q
        = add_matmul_rhs_constant(network, *norm1, weights.hidden_size, attention_size, layer.w_q);
    nvinfer1::ITensor* k
        = add_matmul_rhs_constant(network, *norm1, weights.hidden_size, attention_size, layer.w_k);
    nvinfer1::ITensor* v
        = add_matmul_rhs_constant(network, *norm1, weights.hidden_size, attention_size, layer.w_v);
    if (q == nullptr || k == nullptr || v == nullptr)
    {
        return out;
    }

    if (!layer.q_norm.empty())
    {
        q = add_rms_norm_per_head(
            network, *q, weights.num_attention_heads, head_dim, layer.q_norm, eps_tensor);
    }
    if (!layer.k_norm.empty())
    {
        k = add_rms_norm_per_head(
            network, *k, weights.num_attention_heads, head_dim, layer.k_norm, eps_tensor);
    }
    if (q == nullptr || k == nullptr)
    {
        return out;
    }

    q = add_apply_rope(network, *q, position_id, cos_table, sin_table, rotate_half_matrix);
    k = add_apply_rope(network, *k, position_id, cos_table, sin_table, rotate_half_matrix);
    if (q == nullptr || k == nullptr)
    {
        return out;
    }

    auto* current_k_reshape = network.addShuffle(*k);
    auto* current_v_reshape = network.addShuffle(*v);
    if (current_k_reshape == nullptr || current_v_reshape == nullptr)
    {
        return out;
    }
    current_k_reshape->setReshapeDimensions(make_dims_2d(1, attention_size));
    current_v_reshape->setReshapeDimensions(make_dims_2d(1, attention_size));

    nvinfer1::ITensor* all_k_inputs[] = {&cache_k, current_k_reshape->getOutput(0)};
    nvinfer1::ITensor* all_v_inputs[] = {&cache_v, current_v_reshape->getOutput(0)};
    auto* all_k_concat = network.addConcatenation(all_k_inputs, 2);
    auto* all_v_concat = network.addConcatenation(all_v_inputs, 2);
    if (all_k_concat == nullptr || all_v_concat == nullptr)
    {
        return out;
    }
    all_k_concat->setAxis(0);
    all_v_concat->setAxis(0);

    auto* q_heads = network.addShuffle(*q);
    if (q_heads == nullptr)
    {
        return out;
    }
    q_heads->setReshapeDimensions(make_dims_3d(weights.num_attention_heads, 1, head_dim));

    auto* k_heads = network.addShuffle(*all_k_concat->getOutput(0));
    auto* v_heads = network.addShuffle(*all_v_concat->getOutput(0));
    if (k_heads == nullptr || v_heads == nullptr)
    {
        return out;
    }
    k_heads->setReshapeDimensions(make_dims_3d(attention_window, weights.num_attention_heads, head_dim));
    v_heads->setReshapeDimensions(make_dims_3d(attention_window, weights.num_attention_heads, head_dim));

    nvinfer1::Permutation seq_to_head_major{};
    seq_to_head_major.order[0] = 1;
    seq_to_head_major.order[1] = 0;
    seq_to_head_major.order[2] = 2;
    k_heads->setSecondTranspose(seq_to_head_major);
    v_heads->setSecondTranspose(seq_to_head_major);

    auto* score = network.addMatrixMultiply(
        *q_heads->getOutput(0), nvinfer1::MatrixOperation::kNONE,
        *k_heads->getOutput(0), nvinfer1::MatrixOperation::kTRANSPOSE);
    if (score == nullptr)
    {
        return out;
    }

    auto* scaled
        = network.addElementWise(*score->getOutput(0), attention_scale_tensor, nvinfer1::ElementWiseOperation::kPROD);
    if (scaled == nullptr)
    {
        return out;
    }

    auto* mask3d = network.addShuffle(attention_mask);
    if (mask3d == nullptr)
    {
        return out;
    }
    mask3d->setReshapeDimensions(make_dims_3d(1, 1, attention_window));

    auto* masked = network.addElementWise(*scaled->getOutput(0), *mask3d->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
    if (masked == nullptr)
    {
        return out;
    }

    auto* softmax = network.addSoftMax(*masked->getOutput(0));
    if (softmax == nullptr)
    {
        return out;
    }
    softmax->setAxes(1U << 2);

    auto* context_heads = network.addMatrixMultiply(
        *softmax->getOutput(0), nvinfer1::MatrixOperation::kNONE,
        *v_heads->getOutput(0), nvinfer1::MatrixOperation::kNONE);
    if (context_heads == nullptr)
    {
        return out;
    }

    auto* context_flat = network.addShuffle(*context_heads->getOutput(0));
    if (context_flat == nullptr)
    {
        return out;
    }
    context_flat->setReshapeDimensions(make_dims_2d(1, attention_size));

    nvinfer1::ITensor* attn_out
        = add_matmul_rhs_constant(network, *context_flat->getOutput(0), attention_size, weights.hidden_size, layer.w_o);
    if (attn_out == nullptr)
    {
        return out;
    }

    auto* residual1 = network.addElementWise(hidden, *attn_out, nvinfer1::ElementWiseOperation::kSUM);
    if (residual1 == nullptr)
    {
        return out;
    }

    nvinfer1::ITensor* norm2 = add_rms_norm(
        network, *residual1->getOutput(0), weights.hidden_size, layer.post_attn_norm, eps_tensor);
    if (norm2 == nullptr)
    {
        return out;
    }

    nvinfer1::ITensor* gate
        = add_matmul_rhs_constant(network, *norm2, weights.hidden_size, weights.mlp_size, layer.w_gate);
    nvinfer1::ITensor* up = add_matmul_rhs_constant(network, *norm2, weights.hidden_size, weights.mlp_size, layer.w_up);
    if (gate == nullptr || up == nullptr)
    {
        return out;
    }

    auto* sigmoid = network.addActivation(*gate, nvinfer1::ActivationType::kSIGMOID);
    if (sigmoid == nullptr)
    {
        return out;
    }

    auto* swish = network.addElementWise(*gate, *sigmoid->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);
    if (swish == nullptr)
    {
        return out;
    }

    auto* gated = network.addElementWise(*swish->getOutput(0), *up, nvinfer1::ElementWiseOperation::kPROD);
    if (gated == nullptr)
    {
        return out;
    }

    nvinfer1::ITensor* down
        = add_matmul_rhs_constant(network, *gated->getOutput(0), weights.mlp_size, weights.hidden_size, layer.w_down);
    if (down == nullptr)
    {
        return out;
    }

    auto* residual2 = network.addElementWise(
        *residual1->getOutput(0), *down, nvinfer1::ElementWiseOperation::kSUM);
    if (residual2 == nullptr)
    {
        return out;
    }

    out.hidden = residual2->getOutput(0);
    out.present_k = k;
    out.present_v = v;
    return out;
}

std::unique_ptr<DecoderStepEngine> create_decoder_step_engine_qwen(
    const TrtDecoderDefinition& weights, TrtLogger& logger)
{
    const int32_t attention_size = weights.attention_size > 0 ? weights.attention_size : weights.hidden_size;
    if (attention_size <= 0 || weights.num_attention_heads <= 0
        || attention_size % weights.num_attention_heads != 0)
    {
        return nullptr;
    }

    const int32_t head_dim = attention_size / weights.num_attention_heads;
    const int32_t attention_window = weights.max_cache_length + 1;

    auto builder = TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    if (!builder)
    {
        return nullptr;
    }

    uint32_t flags = 0U;
#if NV_TENSORRT_MAJOR < 10
    flags = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
#endif
    auto network = TrtUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(flags));
    auto config = TrtUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!network || !config)
    {
        return nullptr;
    }

#if NV_TENSORRT_MAJOR >= 8
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1U << 20);
    config->clearFlag(nvinfer1::BuilderFlag::kTF32);
#endif

    auto* token_id = network->addInput("token_id", nvinfer1::DataType::kINT32, make_dims_1d(1));
    auto* position_id = network->addInput("position_id", nvinfer1::DataType::kINT32, make_dims_1d(1));
    auto* attention_mask = network->addInput(
        "attention_mask", nvinfer1::DataType::kFLOAT, make_dims_2d(1, attention_window));
    if (token_id == nullptr || position_id == nullptr || attention_mask == nullptr)
    {
        return nullptr;
    }

    std::vector<nvinfer1::ITensor*> cache_k_inputs;
    std::vector<nvinfer1::ITensor*> cache_v_inputs;
    std::vector<std::string> cache_k_names;
    std::vector<std::string> cache_v_names;
    std::vector<std::string> present_k_names;
    std::vector<std::string> present_v_names;

    const int32_t num_layers = static_cast<int32_t>(weights.qwen_layers.size());
    cache_k_inputs.reserve(static_cast<std::size_t>(num_layers));
    cache_v_inputs.reserve(static_cast<std::size_t>(num_layers));
    cache_k_names.reserve(static_cast<std::size_t>(num_layers));
    cache_v_names.reserve(static_cast<std::size_t>(num_layers));
    present_k_names.reserve(static_cast<std::size_t>(num_layers));
    present_v_names.reserve(static_cast<std::size_t>(num_layers));

    for (int32_t layer = 0; layer < num_layers; ++layer)
    {
        cache_k_names.push_back(layer_tensor_name("cache_k", layer));
        cache_v_names.push_back(layer_tensor_name("cache_v", layer));
        present_k_names.push_back(layer_tensor_name("present_k", layer));
        present_v_names.push_back(layer_tensor_name("present_v", layer));

        auto* cache_k = network->addInput(cache_k_names.back().c_str(), nvinfer1::DataType::kFLOAT,
            make_dims_2d(weights.max_cache_length, attention_size));
        auto* cache_v = network->addInput(cache_v_names.back().c_str(), nvinfer1::DataType::kFLOAT,
            make_dims_2d(weights.max_cache_length, attention_size));
        if (cache_k == nullptr || cache_v == nullptr)
        {
            return nullptr;
        }
        cache_k_inputs.push_back(cache_k);
        cache_v_inputs.push_back(cache_v);
    }

    nvinfer1::ITensor* embedding_table
        = add_constant_tensor(*network, make_dims_2d(weights.vocab_size, weights.hidden_size), weights.embedding);
    if (embedding_table == nullptr)
    {
        return nullptr;
    }

    auto* gather = network->addGather(*embedding_table, *token_id, 0);
    if (gather == nullptr)
    {
        return nullptr;
    }
    nvinfer1::ITensor* hidden = gather->getOutput(0);

    const std::vector<float> cos_table
        = make_rope_table(attention_window, attention_size, weights.num_attention_heads, weights.rope_theta, true);
    const std::vector<float> sin_table
        = make_rope_table(attention_window, attention_size, weights.num_attention_heads, weights.rope_theta, false);
    const std::vector<float> rotate_half = make_rotate_half_matrix(attention_size, weights.num_attention_heads);
    const std::vector<float> eps_data{weights.rms_norm_eps};
    const float attention_scale = 1.0F / std::sqrt(static_cast<float>(std::max(head_dim, 1)));
    const std::vector<float> attention_scale_data{attention_scale};

    nvinfer1::ITensor* cos_tensor
        = add_constant_tensor(*network, make_dims_2d(attention_window, attention_size), cos_table);
    nvinfer1::ITensor* sin_tensor
        = add_constant_tensor(*network, make_dims_2d(attention_window, attention_size), sin_table);
    nvinfer1::ITensor* rotate_half_tensor
        = add_constant_tensor(*network, make_dims_2d(attention_size, attention_size), rotate_half);
    nvinfer1::ITensor* eps_tensor = add_constant_tensor(*network, make_dims_2d(1, 1), eps_data);
    nvinfer1::ITensor* attention_scale_tensor
        = add_constant_tensor(*network, make_dims_3d(1, 1, 1), attention_scale_data);
    if (cos_tensor == nullptr || sin_tensor == nullptr || rotate_half_tensor == nullptr || eps_tensor == nullptr
        || attention_scale_tensor == nullptr)
    {
        return nullptr;
    }

    std::vector<nvinfer1::ITensor*> present_k_outputs(static_cast<std::size_t>(num_layers), nullptr);
    std::vector<nvinfer1::ITensor*> present_v_outputs(static_cast<std::size_t>(num_layers), nullptr);

    for (int32_t layer_idx = 0; layer_idx < num_layers; ++layer_idx)
    {
        const TrtDecoderLayerDefinition& layer = weights.qwen_layers[static_cast<std::size_t>(layer_idx)];
        QwenLayerTensors layer_tensors = add_qwen_layer_block(
            *network, weights, layer, *hidden, *cache_k_inputs[static_cast<std::size_t>(layer_idx)],
            *cache_v_inputs[static_cast<std::size_t>(layer_idx)], *attention_mask, *position_id,
            *cos_tensor, *sin_tensor, *rotate_half_tensor, *attention_scale_tensor, *eps_tensor);
        if (layer_tensors.hidden == nullptr || layer_tensors.present_k == nullptr || layer_tensors.present_v == nullptr)
        {
            return nullptr;
        }

        hidden = layer_tensors.hidden;
        present_k_outputs[static_cast<std::size_t>(layer_idx)] = layer_tensors.present_k;
        present_v_outputs[static_cast<std::size_t>(layer_idx)] = layer_tensors.present_v;
    }

    if (!weights.final_norm.empty())
    {
        hidden = add_rms_norm(*network, *hidden, weights.hidden_size, weights.final_norm, *eps_tensor);
        if (hidden == nullptr)
        {
            return nullptr;
        }
    }

    nvinfer1::ITensor* logits
        = add_matmul_rhs_constant(*network, *hidden, weights.hidden_size, weights.vocab_size, weights.w_out);
    if (logits == nullptr)
    {
        return nullptr;
    }
    nvinfer1::ITensor* logits_bias = add_bias_sum(*network, *logits, weights.vocab_size, weights.b_out);
    if (logits_bias == nullptr)
    {
        return nullptr;
    }

    logits_bias->setName("logits");
    network->markOutput(*logits_bias);

    for (int32_t layer = 0; layer < num_layers; ++layer)
    {
        nvinfer1::ITensor* pk = present_k_outputs[static_cast<std::size_t>(layer)];
        nvinfer1::ITensor* pv = present_v_outputs[static_cast<std::size_t>(layer)];
        if (pk == nullptr || pv == nullptr)
        {
            return nullptr;
        }
        pk->setName(present_k_names[static_cast<std::size_t>(layer)].c_str());
        pv->setName(present_v_names[static_cast<std::size_t>(layer)].c_str());
        network->markOutput(*pk);
        network->markOutput(*pv);
    }

    return finalize_decoder_step_engine(*builder, *network, *config, logger, weights,
        cache_k_names, cache_v_names, present_k_names, present_v_names, true);
}

std::unique_ptr<DecoderStepEngine> create_decoder_step_engine(const TrtDecoderDefinition& weights, TrtLogger& logger)
{
    if (weights.has_qwen_layers && !weights.qwen_layers.empty())
    {
        return create_decoder_step_engine_qwen(weights, logger);
    }
    return create_decoder_step_engine_legacy(weights, logger);
}

int32_t select_argmax_token(const std::vector<float>& logits)
{
    if (logits.empty())
    {
        return 0;
    }
    const auto it = std::max_element(logits.begin(), logits.end());
    return static_cast<int32_t>(std::distance(logits.begin(), it));
}

int32_t parse_positive_env_int(const char* env_name, int32_t fallback)
{
    const char* env = std::getenv(env_name);
    if (env == nullptr || env[0] == '\0')
    {
        return fallback;
    }

    char* end = nullptr;
    const long parsed = std::strtol(env, &end, 10);
    if (end == env || (end != nullptr && *end != '\0') || parsed <= 0 || parsed > 1000)
    {
        return fallback;
    }
    return static_cast<int32_t>(parsed);
}

std::vector<int32_t> select_topk_tokens(const std::vector<float>& logits, int32_t k)
{
    if (logits.empty() || k <= 0)
    {
        return {};
    }

    const int32_t capped = std::min(k, static_cast<int32_t>(logits.size()));
    std::vector<int32_t> indices(logits.size(), 0);
    for (std::size_t i = 0; i < indices.size(); ++i)
    {
        indices[i] = static_cast<int32_t>(i);
    }

    std::partial_sort(indices.begin(), indices.begin() + capped, indices.end(),
        [&](int32_t a, int32_t b) { return logits[static_cast<std::size_t>(a)] > logits[static_cast<std::size_t>(b)]; });
    indices.resize(static_cast<std::size_t>(capped));
    return indices;
}

std::vector<float> build_attention_mask(int32_t cache_length, int32_t max_cache_length, bool include_current_slot)
{
    const int32_t width = max_cache_length + (include_current_slot ? 1 : 0);
    if (width <= 0)
    {
        return {};
    }

    std::vector<float> mask(static_cast<std::size_t>(width), kMaskedScore);
    const int32_t valid = std::max(0, std::min(cache_length, max_cache_length));
    for (int32_t i = 0; i < valid; ++i)
    {
        mask[static_cast<std::size_t>(i)] = 0.0F;
    }

    if (include_current_slot)
    {
        mask.back() = 0.0F;
    }
    else if (valid <= 0)
    {
        mask[0] = 0.0F;
    }

    return mask;
}

void append_cache_state(
    std::vector<float>& cache, const std::vector<float>& state, int32_t hidden_size, int32_t max_cache_length,
    int32_t write_index)
{
    if (static_cast<int32_t>(state.size()) != hidden_size || max_cache_length <= 0)
    {
        return;
    }

    if (write_index < max_cache_length)
    {
        const std::size_t start = static_cast<std::size_t>(write_index) * static_cast<std::size_t>(hidden_size);
        std::copy(state.begin(), state.end(), cache.begin() + static_cast<std::ptrdiff_t>(start));
        return;
    }

    const std::size_t row_size = static_cast<std::size_t>(hidden_size);
    const std::size_t bytes_to_move
        = (static_cast<std::size_t>(max_cache_length - 1) * row_size) * sizeof(float);
    std::memmove(cache.data(), cache.data() + static_cast<std::ptrdiff_t>(row_size), bytes_to_move);

    const std::size_t tail = static_cast<std::size_t>(max_cache_length - 1) * row_size;
    std::copy(state.begin(), state.end(), cache.begin() + static_cast<std::ptrdiff_t>(tail));
}

bool run_decoder_step(const DecoderStepEngine& engine, int32_t token_id, int32_t position_id,
    const std::vector<std::vector<float>>& cache_k_by_layer, const std::vector<std::vector<float>>& cache_v_by_layer,
    const std::vector<float>& attention_mask, std::vector<float>& logits,
    std::vector<std::vector<float>>& present_k_by_layer, std::vector<std::vector<float>>& present_v_by_layer,
    std::string& error)
{
    auto fail = [&error](std::string_view stage) {
        error = std::string(stage);
        return false;
    };

    if (static_cast<int32_t>(cache_k_by_layer.size()) != engine.num_layers
        || static_cast<int32_t>(cache_v_by_layer.size()) != engine.num_layers
        || attention_mask.size() != static_cast<std::size_t>(engine.attention_mask_size))
    {
        return fail("invalid cache layer count");
    }

    const std::size_t expected_cache
        = static_cast<std::size_t>(engine.max_cache_length) * static_cast<std::size_t>(engine.cache_state_size);
    for (int32_t i = 0; i < engine.num_layers; ++i)
    {
        if (cache_k_by_layer[static_cast<std::size_t>(i)].size() != expected_cache
            || cache_v_by_layer[static_cast<std::size_t>(i)].size() != expected_cache)
        {
            return fail("invalid cache tensor size");
        }
    }

    logits.assign(static_cast<std::size_t>(engine.vocab_size), 0.0F);
    present_k_by_layer.assign(static_cast<std::size_t>(engine.num_layers),
        std::vector<float>(static_cast<std::size_t>(engine.cache_state_size), 0.0F));
    present_v_by_layer.assign(static_cast<std::size_t>(engine.num_layers),
        std::vector<float>(static_cast<std::size_t>(engine.cache_state_size), 0.0F));

    CudaStream stream;
    if (!stream.ok())
    {
        return fail("cudaStreamCreate failed");
    }

    struct PendingCopy {
        void* host_ptr{nullptr};
        void* device_ptr{nullptr};
        std::size_t bytes{0};
    };

    std::vector<std::unique_ptr<CudaBuffer>> device_buffers;
    std::vector<PendingCopy> output_copies;

    auto bind_input = [&](const std::string& name, const void* host_ptr, std::size_t bytes) -> bool {
        const auto location = engine.engine->getTensorLocation(name.c_str());
        if (location == nvinfer1::TensorLocation::kHOST)
        {
            return engine.context->setTensorAddress(name.c_str(), const_cast<void*>(host_ptr));
        }

        auto buffer = std::make_unique<CudaBuffer>(bytes);
        if (!buffer->ok())
        {
            return false;
        }
        if (cudaMemcpyAsync(buffer->data(), host_ptr, bytes, cudaMemcpyHostToDevice, stream.get()) != cudaSuccess)
        {
            return false;
        }

        void* device_ptr = buffer->data();
        if (!engine.context->setTensorAddress(name.c_str(), device_ptr))
        {
            return false;
        }

        device_buffers.push_back(std::move(buffer));
        return true;
    };

    auto bind_output = [&](const std::string& name, void* host_ptr, std::size_t bytes) -> bool {
        const auto location = engine.engine->getTensorLocation(name.c_str());
        if (location == nvinfer1::TensorLocation::kHOST)
        {
            return engine.context->setTensorAddress(name.c_str(), host_ptr);
        }

        auto buffer = std::make_unique<CudaBuffer>(bytes);
        if (!buffer->ok())
        {
            return false;
        }

        void* device_ptr = buffer->data();
        if (!engine.context->setTensorAddress(name.c_str(), device_ptr))
        {
            return false;
        }

        output_copies.push_back(PendingCopy{host_ptr, device_ptr, bytes});
        device_buffers.push_back(std::move(buffer));
        return true;
    };

    const std::size_t cache_bytes = expected_cache * sizeof(float);
    const std::size_t mask_bytes = attention_mask.size() * sizeof(float);
    const std::size_t logits_bytes = logits.size() * sizeof(float);
    const std::size_t state_bytes = static_cast<std::size_t>(engine.cache_state_size) * sizeof(float);

    if (!bind_input(engine.token_input_name, &token_id, sizeof(token_id)))
    {
        return fail("bind input token failed");
    }
    if (engine.requires_position_input
        && !bind_input(engine.position_input_name, &position_id, sizeof(position_id)))
    {
        return fail("bind input position_id failed");
    }
    if (const char* debug_mask = std::getenv("TRTF_DEBUG_MASK"); debug_mask != nullptr && debug_mask[0] != '\0')
    {
        std::cerr << "TRTF_DEBUG_MASK size=" << attention_mask.size();
        if (!attention_mask.empty())
        {
            std::cerr << " first=" << attention_mask.front() << " last=" << attention_mask.back();
        }
        std::cerr << '\n';
    }

    if (!bind_input(engine.mask_input_name, attention_mask.data(), mask_bytes))
    {
        return fail("bind input attention_mask failed");
    }

    for (int32_t layer = 0; layer < engine.num_layers; ++layer)
    {
        const std::size_t idx = static_cast<std::size_t>(layer);
        if (!bind_input(engine.cache_k_input_names[idx], cache_k_by_layer[idx].data(), cache_bytes))
        {
            return fail("bind input cache_k failed");
        }
        if (!bind_input(engine.cache_v_input_names[idx], cache_v_by_layer[idx].data(), cache_bytes))
        {
            return fail("bind input cache_v failed");
        }
    }

    if (!bind_output(engine.logits_output_name, logits.data(), logits_bytes))
    {
        return fail("bind output logits failed");
    }

    for (int32_t layer = 0; layer < engine.num_layers; ++layer)
    {
        const std::size_t idx = static_cast<std::size_t>(layer);
        if (!bind_output(engine.present_k_output_names[idx], present_k_by_layer[idx].data(), state_bytes))
        {
            return fail("bind output present_k failed");
        }
        if (!bind_output(engine.present_v_output_names[idx], present_v_by_layer[idx].data(), state_bytes))
        {
            return fail("bind output present_v failed");
        }
    }

    if (!engine.context->enqueueV3(stream.get()))
    {
        return fail("enqueueV3 failed");
    }

    for (const PendingCopy& copy : output_copies)
    {
        if (cudaMemcpyAsync(copy.host_ptr, copy.device_ptr, copy.bytes, cudaMemcpyDeviceToHost, stream.get())
            != cudaSuccess)
        {
            return fail("cudaMemcpyAsync output failed");
        }
    }

    if (cudaStreamSynchronize(stream.get()) != cudaSuccess)
    {
        return fail("cudaStreamSynchronize failed");
    }

    return true;
}
#endif

class TrtBackend final : public IGenerationBackend {
public:
    TrtBackend(const ITokenizer& tokenizer, const DecoderModel& model)
        : mTokenizer(tokenizer)
    {
#if TRTF_HAS_TRT
        const cudaError_t init_err = cudaFree(nullptr);
        if (init_err != cudaSuccess)
        {
            mInitError = std::string("cudaFree(nullptr) failed with cudaError=")
                + std::to_string(static_cast<int>(init_err));
            mAvailable = false;
            return;
        }

        try
        {
            mLogger.clear_error();
            mWeights = BuildTrtDecoderWeights(mTokenizer, model);
            mDecoderStepEngine = create_decoder_step_engine(mWeights, mLogger);
            mAvailable = static_cast<bool>(mDecoderStepEngine);
            if (!mAvailable)
            {
                mInitError = mLogger.last_error();
                if (mInitError.empty())
                {
                    mInitError = "create_decoder_step_engine returned null";
                }
            }
        }
        catch (const std::exception& e)
        {
            mInitError = e.what();
            mAvailable = false;
        }
#else
        mInitError = "TRTF_HAS_TRT=0";
        mAvailable = false;
#endif
    }

    bool is_available() const override
    {
        return mAvailable;
    }

    const char* name() const override
    {
        return "trt";
    }

    const char* unavailable_reason() const override
    {
        return mInitError.c_str();
    }

    std::vector<int32_t> generate(const std::vector<int32_t>& input_ids, const GenerationConfig& config) override
    {
#if TRTF_HAS_TRT
        if (!mAvailable || !mDecoderStepEngine)
        {
            throw std::runtime_error("TRT backend is unavailable: " + mInitError);
        }

        std::vector<int32_t> output = input_ids;
        if (config.max_new_tokens == 0)
        {
            return output;
        }

        const int32_t cache_state_size = mDecoderStepEngine->cache_state_size;
        const int32_t max_cache_length = mDecoderStepEngine->max_cache_length;
        const int32_t num_layers = mDecoderStepEngine->num_layers;
        const bool include_current_slot = mDecoderStepEngine->requires_position_input;
        const int32_t position_limit = include_current_slot ? max_cache_length : std::max(max_cache_length - 1, 0);
        const std::size_t cache_elems
            = static_cast<std::size_t>(max_cache_length) * static_cast<std::size_t>(cache_state_size);

        std::vector<std::vector<float>> cache_k(
            static_cast<std::size_t>(num_layers), std::vector<float>(cache_elems, 0.0F));
        std::vector<std::vector<float>> cache_v(
            static_cast<std::size_t>(num_layers), std::vector<float>(cache_elems, 0.0F));
        int32_t cache_length = 0;

        std::vector<float> logits;
        std::vector<std::vector<float>> present_k;
        std::vector<std::vector<float>> present_v;
        const int32_t debug_logits_topk = parse_positive_env_int("TRTF_DEBUG_LOGITS_TOPK", 0);

        if (input_ids.size() > 1)
        {
            for (std::size_t i = 0; i + 1 < input_ids.size(); ++i)
            {
                const std::vector<float> mask = build_attention_mask(cache_length, max_cache_length, include_current_slot);
                const int32_t position_id = std::min(cache_length, position_limit);
                std::string error;
                if (!run_decoder_step(*mDecoderStepEngine, input_ids[i], position_id, cache_k, cache_v, mask, logits,
                        present_k,
                        present_v, error))
                {
                    throw std::runtime_error("prefill step failed: " + error);
                }
                for (int32_t layer = 0; layer < num_layers; ++layer)
                {
                    const std::size_t idx = static_cast<std::size_t>(layer);
                    append_cache_state(cache_k[idx], present_k[idx], cache_state_size, max_cache_length, cache_length);
                    append_cache_state(cache_v[idx], present_v[idx], cache_state_size, max_cache_length, cache_length);
                }
                cache_length = std::min(cache_length + 1, max_cache_length);
            }
        }

        int32_t current_token = input_ids.empty() ? mWeights.id_bos : input_ids.back();

        for (std::size_t step = 0; step < config.max_new_tokens; ++step)
        {
            const std::vector<float> mask = build_attention_mask(cache_length, max_cache_length, include_current_slot);
            const int32_t position_id = std::min(cache_length, position_limit);
            std::string error;
            if (!run_decoder_step(*mDecoderStepEngine, current_token, position_id, cache_k, cache_v, mask, logits, present_k,
                    present_v, error))
            {
                throw std::runtime_error("decode step failed: " + error);
            }
            for (int32_t layer = 0; layer < num_layers; ++layer)
            {
                const std::size_t idx = static_cast<std::size_t>(layer);
                append_cache_state(cache_k[idx], present_k[idx], cache_state_size, max_cache_length, cache_length);
                append_cache_state(cache_v[idx], present_v[idx], cache_state_size, max_cache_length, cache_length);
            }
            cache_length = std::min(cache_length + 1, max_cache_length);

            if (debug_logits_topk > 0)
            {
                const std::vector<int32_t> top_ids = select_topk_tokens(logits, debug_logits_topk);
                std::cerr << "TRTF_DEBUG_LOGITS step=" << step;
                for (int32_t token_id : top_ids)
                {
                    const std::size_t idx = static_cast<std::size_t>(token_id);
                    std::cerr << ' ' << token_id << ':' << logits[idx];
                }
                std::cerr << '\n';
            }

            const int32_t next_token = select_argmax_token(logits);
            output.push_back(next_token);
            current_token = next_token;

            if (next_token == mWeights.id_eos)
            {
                break;
            }
        }

        return output;
#else
        (void) input_ids;
        (void) config;
        throw std::runtime_error("TRT backend is unavailable because this build has no TensorRT support.");
#endif
    }

private:
    const ITokenizer& mTokenizer;
    bool mAvailable{false};
    std::string mInitError;

#if TRTF_HAS_TRT
    TrtLogger mLogger;
    TrtDecoderDefinition mWeights;
    std::unique_ptr<DecoderStepEngine> mDecoderStepEngine;
#endif
};

} // namespace

std::unique_ptr<IGenerationBackend> CreateTrtQwenBackend(const ITokenizer& tokenizer, const DecoderModel& model)
{
    return std::make_unique<TrtBackend>(tokenizer, model);
}

} // namespace trtf
