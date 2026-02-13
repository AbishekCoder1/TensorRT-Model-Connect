#include "trtf/backend.h"
#include "trtf/model.h"
#include "trtf/tokenizer.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
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
class TrtLogger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity <= Severity::kERROR && msg != nullptr)
        {
            mLastError = msg;
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

struct DecoderWeights {
    int32_t vocab_size{0};
    int32_t hidden_size{0};
    int32_t mlp_size{0};
    int32_t max_cache_length{kDefaultMaxCacheLength};
    int32_t id_bos{0};
    int32_t id_eos{0};

    std::vector<float> embedding;
    std::vector<float> w_q;
    std::vector<float> w_k;
    std::vector<float> w_v;
    std::vector<float> w1;
    std::vector<float> b1;
    std::vector<float> w2;
    std::vector<float> b2;
    std::vector<float> w_out;
    std::vector<float> b_out;
};

struct DecoderStepEngine {
    TrtUniquePtr<nvinfer1::ICudaEngine> engine;
    TrtUniquePtr<nvinfer1::IExecutionContext> context;

    std::string token_input_name{"token_id"};
    std::string cache_k_input_name{"cache_k"};
    std::string cache_v_input_name{"cache_v"};
    std::string mask_input_name{"attention_mask"};

    std::string logits_output_name{"logits"};
    std::string present_k_output_name{"present_k"};
    std::string present_v_output_name{"present_v"};

    int32_t vocab_size{0};
    int32_t hidden_size{0};
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

std::vector<float> make_identity_matrix(int32_t dim)
{
    std::vector<float> matrix(static_cast<std::size_t>(dim) * static_cast<std::size_t>(dim), 0.0F);
    for (int32_t i = 0; i < dim; ++i)
    {
        matrix[static_cast<std::size_t>(i) * static_cast<std::size_t>(dim) + static_cast<std::size_t>(i)] = 1.0F;
    }
    return matrix;
}

void set_transition_row(std::vector<float>& matrix, int32_t vocab_size, int32_t from_token, int32_t to_token)
{
    if (from_token < 0 || to_token < 0 || from_token >= vocab_size || to_token >= vocab_size)
    {
        return;
    }

    float* row = matrix.data() + static_cast<std::size_t>(from_token) * static_cast<std::size_t>(vocab_size);
    std::fill(row, row + vocab_size, -1.0F);
    row[to_token] = 8.0F;
}

DecoderWeights create_decoder_weights(const ITokenizer& tokenizer, const DecoderModel& model)
{
    int32_t max_id = 0;
    for (const std::string& token : model.vocab)
    {
        max_id = std::max(max_id, tokenizer.id_for_token(token));
    }

    DecoderWeights weights;
    weights.vocab_size = max_id + 1;
    weights.max_cache_length = std::max(model.max_cache_length, 1);
    weights.id_bos = tokenizer.id_for_token("<bos>");
    weights.id_eos = tokenizer.id_for_token("<eos>");

    const auto expect_size = [](const std::vector<float>& tensor, std::size_t expected, const char* name) {
        if (tensor.size() != expected)
        {
            throw std::runtime_error(
                std::string("Invalid checkpoint tensor size for ") + name + ": expected "
                + std::to_string(expected) + ", got " + std::to_string(tensor.size()));
        }
    };

    if (model.has_checkpoint)
    {
        weights.hidden_size = model.checkpoint.hidden_size;
        weights.mlp_size = model.checkpoint.mlp_size;
        if (weights.hidden_size <= 0 || weights.mlp_size <= 0)
        {
            throw std::runtime_error("Checkpoint has invalid hidden_size/mlp_size.");
        }

        const std::size_t vocab = static_cast<std::size_t>(weights.vocab_size);
        const std::size_t hidden = static_cast<std::size_t>(weights.hidden_size);
        const std::size_t mlp = static_cast<std::size_t>(weights.mlp_size);

        expect_size(model.checkpoint.embedding, vocab * hidden, "embedding");
        expect_size(model.checkpoint.w_q, hidden * hidden, "w_q");
        expect_size(model.checkpoint.w_k, hidden * hidden, "w_k");
        expect_size(model.checkpoint.w_v, hidden * hidden, "w_v");
        expect_size(model.checkpoint.w1, hidden * mlp, "w1");
        expect_size(model.checkpoint.b1, mlp, "b1");
        expect_size(model.checkpoint.w2, mlp * hidden, "w2");
        expect_size(model.checkpoint.b2, hidden, "b2");
        expect_size(model.checkpoint.w_out, hidden * vocab, "w_out");
        expect_size(model.checkpoint.b_out, vocab, "b_out");

        weights.embedding = model.checkpoint.embedding;
        weights.w_q = model.checkpoint.w_q;
        weights.w_k = model.checkpoint.w_k;
        weights.w_v = model.checkpoint.w_v;
        weights.w1 = model.checkpoint.w1;
        weights.b1 = model.checkpoint.b1;
        weights.w2 = model.checkpoint.w2;
        weights.b2 = model.checkpoint.b2;
        weights.w_out = model.checkpoint.w_out;
        weights.b_out = model.checkpoint.b_out;
        return weights;
    }

    // Compatibility path for older model directories without tensor checkpoint.
    weights.hidden_size = weights.vocab_size;
    weights.mlp_size = weights.hidden_size * 2;

    const std::size_t vocab = static_cast<std::size_t>(weights.vocab_size);
    const std::size_t hidden = static_cast<std::size_t>(weights.hidden_size);
    const std::size_t mlp = static_cast<std::size_t>(weights.mlp_size);

    weights.embedding.assign(vocab * hidden, 0.0F);
    for (std::size_t i = 0; i < vocab; ++i)
    {
        weights.embedding[i * hidden + i] = 1.0F;
    }

    weights.w_q = make_identity_matrix(weights.hidden_size);
    weights.w_k = make_identity_matrix(weights.hidden_size);
    weights.w_v.assign(hidden * hidden, 0.0F);

    weights.w1.assign(hidden * mlp, 0.0F);
    weights.b1.assign(mlp, 0.0F);
    weights.w2.assign(mlp * hidden, 0.0F);
    weights.b2.assign(hidden, 0.0F);

    const int32_t default_next_id = tokenizer.id_for_token(model.default_next_token);
    weights.w_out.assign(hidden * vocab, -1.0F);
    for (int32_t token_id = 0; token_id < weights.vocab_size; ++token_id)
    {
        set_transition_row(weights.w_out, weights.vocab_size, token_id, default_next_id);
    }

    for (const auto& transition : model.transitions)
    {
        const int32_t from_id = tokenizer.id_for_token(transition.first);
        const int32_t to_id = tokenizer.id_for_token(transition.second);
        set_transition_row(weights.w_out, weights.vocab_size, from_id, to_id);
    }

    weights.b_out.assign(vocab, 0.0F);
    return weights;
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

std::unique_ptr<DecoderStepEngine> create_decoder_step_engine(const DecoderWeights& weights, TrtLogger& logger)
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
#endif

    auto* token_id = network->addInput("token_id", nvinfer1::DataType::kINT32, make_dims_1d(1));
    auto* cache_k
        = network->addInput("cache_k", nvinfer1::DataType::kFLOAT, make_dims_2d(weights.max_cache_length, weights.hidden_size));
    auto* cache_v
        = network->addInput("cache_v", nvinfer1::DataType::kFLOAT, make_dims_2d(weights.max_cache_length, weights.hidden_size));
    auto* attention_mask
        = network->addInput("attention_mask", nvinfer1::DataType::kFLOAT, make_dims_2d(1, weights.max_cache_length));
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
    k->setName("present_k");
    v->setName("present_v");

    network->markOutput(*logits_bias);
    network->markOutput(*k);
    network->markOutput(*v);

    auto plan = TrtUniquePtr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    if (!plan)
    {
        return nullptr;
    }

    auto runtime = TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
    if (!runtime)
    {
        return nullptr;
    }

    auto engine = TrtUniquePtr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan->data(), plan->size()));
    if (!engine)
    {
        return nullptr;
    }

    auto execution_context = TrtUniquePtr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
    if (!execution_context)
    {
        return nullptr;
    }

    auto out = std::make_unique<DecoderStepEngine>();
    out->engine = std::move(engine);
    out->context = std::move(execution_context);
    out->vocab_size = weights.vocab_size;
    out->hidden_size = weights.hidden_size;
    out->max_cache_length = weights.max_cache_length;

    const bool has_all_tensors = has_io_tensor(*out->engine, out->token_input_name)
        && has_io_tensor(*out->engine, out->cache_k_input_name)
        && has_io_tensor(*out->engine, out->cache_v_input_name)
        && has_io_tensor(*out->engine, out->mask_input_name)
        && has_io_tensor(*out->engine, out->logits_output_name)
        && has_io_tensor(*out->engine, out->present_k_output_name)
        && has_io_tensor(*out->engine, out->present_v_output_name);

    if (!has_all_tensors)
    {
        return nullptr;
    }

    return out;
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

std::vector<float> build_attention_mask(int32_t cache_length, int32_t max_cache_length)
{
    std::vector<float> mask(static_cast<std::size_t>(max_cache_length), kMaskedScore);
    if (cache_length <= 0)
    {
        mask[0] = 0.0F;
        return mask;
    }

    const int32_t valid = std::min(cache_length, max_cache_length);
    for (int32_t i = 0; i < valid; ++i)
    {
        mask[static_cast<std::size_t>(i)] = 0.0F;
    }
    return mask;
}

void append_cache_state(
    std::vector<float>& cache, const std::vector<float>& state, int32_t hidden_size, int32_t max_cache_length,
    int32_t& cache_length)
{
    if (static_cast<int32_t>(state.size()) != hidden_size)
    {
        return;
    }

    if (cache_length < max_cache_length)
    {
        const std::size_t start = static_cast<std::size_t>(cache_length) * static_cast<std::size_t>(hidden_size);
        std::copy(state.begin(), state.end(), cache.begin() + static_cast<std::ptrdiff_t>(start));
        ++cache_length;
        return;
    }

    const std::size_t row_size = static_cast<std::size_t>(hidden_size);
    const std::size_t bytes_to_move
        = (static_cast<std::size_t>(max_cache_length - 1) * row_size) * sizeof(float);
    std::memmove(cache.data(), cache.data() + static_cast<std::ptrdiff_t>(row_size), bytes_to_move);

    const std::size_t tail = static_cast<std::size_t>(max_cache_length - 1) * row_size;
    std::copy(state.begin(), state.end(), cache.begin() + static_cast<std::ptrdiff_t>(tail));
}

bool run_decoder_step(const DecoderStepEngine& engine, int32_t token_id, const std::vector<float>& cache_k,
    const std::vector<float>& cache_v, const std::vector<float>& attention_mask, std::vector<float>& logits,
    std::vector<float>& present_k, std::vector<float>& present_v, std::string& error)
{
    auto fail = [&error](std::string_view stage) {
        error = std::string(stage);
        return false;
    };

    if (cache_k.size() != static_cast<std::size_t>(engine.max_cache_length) * static_cast<std::size_t>(engine.hidden_size)
        || cache_v.size() != static_cast<std::size_t>(engine.max_cache_length) * static_cast<std::size_t>(engine.hidden_size)
        || attention_mask.size() != static_cast<std::size_t>(engine.max_cache_length))
    {
        return fail("invalid input tensor size");
    }

    logits.assign(static_cast<std::size_t>(engine.vocab_size), 0.0F);
    present_k.assign(static_cast<std::size_t>(engine.hidden_size), 0.0F);
    present_v.assign(static_cast<std::size_t>(engine.hidden_size), 0.0F);

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

    const std::size_t cache_bytes = cache_k.size() * sizeof(float);
    const std::size_t mask_bytes = attention_mask.size() * sizeof(float);
    const std::size_t logits_bytes = logits.size() * sizeof(float);
    const std::size_t state_bytes = present_k.size() * sizeof(float);

    if (!bind_input(engine.token_input_name, &token_id, sizeof(token_id)))
    {
        return fail("bind input token failed");
    }
    if (!bind_input(engine.cache_k_input_name, cache_k.data(), cache_bytes))
    {
        return fail("bind input cache_k failed");
    }
    if (!bind_input(engine.cache_v_input_name, cache_v.data(), cache_bytes))
    {
        return fail("bind input cache_v failed");
    }
    if (!bind_input(engine.mask_input_name, attention_mask.data(), mask_bytes))
    {
        return fail("bind input attention_mask failed");
    }

    if (!bind_output(engine.logits_output_name, logits.data(), logits_bytes))
    {
        return fail("bind output logits failed");
    }
    if (!bind_output(engine.present_k_output_name, present_k.data(), state_bytes))
    {
        return fail("bind output present_k failed");
    }
    if (!bind_output(engine.present_v_output_name, present_v.data(), state_bytes))
    {
        return fail("bind output present_v failed");
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
            mWeights = create_decoder_weights(mTokenizer, model);
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

        const int32_t hidden_size = mDecoderStepEngine->hidden_size;
        const int32_t max_cache_length = mDecoderStepEngine->max_cache_length;
        std::vector<float> cache_k(
            static_cast<std::size_t>(max_cache_length) * static_cast<std::size_t>(hidden_size), 0.0F);
        std::vector<float> cache_v(
            static_cast<std::size_t>(max_cache_length) * static_cast<std::size_t>(hidden_size), 0.0F);
        int32_t cache_length = 0;

        std::vector<float> logits;
        std::vector<float> present_k;
        std::vector<float> present_v;

        if (input_ids.size() > 1)
        {
            for (std::size_t i = 0; i + 1 < input_ids.size(); ++i)
            {
                const std::vector<float> mask = build_attention_mask(cache_length, max_cache_length);
                std::string error;
                if (!run_decoder_step(*mDecoderStepEngine, input_ids[i], cache_k, cache_v, mask, logits, present_k,
                        present_v, error))
                {
                    throw std::runtime_error("prefill step failed: " + error);
                }

                append_cache_state(cache_k, present_k, hidden_size, max_cache_length, cache_length);
                append_cache_state(cache_v, present_v, hidden_size, max_cache_length, cache_length);
            }
        }

        int32_t current_token = input_ids.empty() ? mWeights.id_bos : input_ids.back();

        for (std::size_t step = 0; step < config.max_new_tokens; ++step)
        {
            const std::vector<float> mask = build_attention_mask(cache_length, max_cache_length);
            std::string error;
            if (!run_decoder_step(
                    *mDecoderStepEngine, current_token, cache_k, cache_v, mask, logits, present_k, present_v, error))
            {
                throw std::runtime_error("decode step failed: " + error);
            }

            append_cache_state(cache_k, present_k, hidden_size, max_cache_length, cache_length);
            append_cache_state(cache_v, present_v, hidden_size, max_cache_length, cache_length);

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
    DecoderWeights mWeights;
    std::unique_ptr<DecoderStepEngine> mDecoderStepEngine;
#endif
};

} // namespace

std::unique_ptr<IGenerationBackend> CreateTrtBackend(const ITokenizer& tokenizer, const DecoderModel& model)
{
    return std::make_unique<TrtBackend>(tokenizer, model);
}

} // namespace trtf
