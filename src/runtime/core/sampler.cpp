// ISampler implementations: GreedySampler, TopKSampler, and factory.
//
// GreedySampler wraps the same std::max_element logic as the original
// TextGenerationPipeline::argmax() and select_argmax_token(), producing
// bit-identical token sequences.
//
// TopKSampler wraps the same temperature-scaled top-k logic as
// sample_token_topk(), with internal xorshift64 RNG state.

#include "trtf/runtime/sampler.h"

#include "trtf/pipeline.h"
#if TRTF_HAS_CUDA_KERNELS
#include "runtime/core/argmax_kernel.h"

#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace trtf {

// Shared argmax helper — returns {token_id, logprob} for the highest logit.
static SampleResult argmax_over_logits(const float* logits, int32_t vocab_size,
                                       int32_t eos_token_id) {
    SampleResult result;
    const float* best = logits;
    for (int32_t i = 1; i < vocab_size; ++i) {
        if (logits[i] > *best)
            best = logits + i;
    }
    result.token_id = static_cast<int32_t>(best - logits);
    result.logprob = *best;
    result.is_eos = (result.token_id == eos_token_id);
    return result;
}

// ─────────────────────────────────────────────────────────────
// GreedySampler: deterministic argmax (identical to select_argmax_token)
// ─────────────────────────────────────────────────────────────

class GreedySampler final : public ISampler {
  public:
    SampleResult sample(const float* logits, int32_t vocab_size,
                        const SamplingParams& params) override {
        if (vocab_size <= 0 || logits == nullptr) {
            SampleResult result;
            result.token_id = 0;
            result.is_eos = (0 == params.eos_token_id);
            return result;
        }

        return argmax_over_logits(logits, vocab_size, params.eos_token_id);
    }

    LogitsLocation logits_location() const override { return LogitsLocation::HOST; }
    const char* sampler_type() const override { return "greedy"; }
};

// ─────────────────────────────────────────────────────────────
// TopKSampler: temperature-scaled top-k with xorshift64 RNG
// (identical logic to sample_token_topk)
// ─────────────────────────────────────────────────────────────

class TopKSampler final : public ISampler {
  public:
    explicit TopKSampler(uint64_t initial_seed)
        : rng_state_(initial_seed == 0 ? 1 : initial_seed),
          initial_seed_(initial_seed == 0 ? 1 : initial_seed) {}

    SampleResult sample(const float* logits, int32_t vocab_size,
                        const SamplingParams& params) override {
        SampleResult result;

        if (vocab_size <= 0 || logits == nullptr) {
            result.token_id = 0;
            result.is_eos = (0 == params.eos_token_id);
            return result;
        }

        const float temperature = params.temperature;

        // Fallback to argmax if temperature is near zero
        if (temperature < 1e-6F)
            return argmax_over_logits(logits, vocab_size, params.eos_token_id);

        const auto n = vocab_size;

        // Build index array sorted by logit value (descending)
        std::vector<int32_t> indices(static_cast<std::size_t>(n));
        std::iota(indices.begin(), indices.end(), 0);
        const int32_t k = std::min(std::max(params.top_k, 1), n);
        std::partial_sort(
            indices.begin(), indices.begin() + k, indices.end(), [&](int32_t a, int32_t b) {
                return logits[static_cast<std::size_t>(a)] > logits[static_cast<std::size_t>(b)];
            });

        // Temperature-scaled softmax over top-k
        float max_logit = logits[static_cast<std::size_t>(indices[0])];
        std::vector<float> probs(static_cast<std::size_t>(k));
        float sum = 0.0F;
        for (int32_t i = 0; i < k; ++i) {
            float scaled = (logits[static_cast<std::size_t>(indices[i])] - max_logit) / temperature;
            probs[i] = std::exp(scaled);
            sum += probs[i];
        }

        // Normalize
        if (sum > 0.0F) {
            for (int32_t i = 0; i < k; ++i)
                probs[i] /= sum;
        } else {
            for (int32_t i = 0; i < k; ++i)
                probs[i] = 1.0F / static_cast<float>(k);
        }

        // xorshift64 random number generation
        rng_state_ ^= rng_state_ << 13;
        rng_state_ ^= rng_state_ >> 7;
        rng_state_ ^= rng_state_ << 17;
        float u = static_cast<float>(rng_state_ & 0xFFFFFFFF) / 4294967296.0F;

        // Sample from cumulative distribution
        float cumulative = 0.0F;
        for (int32_t i = 0; i < k; ++i) {
            cumulative += probs[i];
            if (u < cumulative) {
                result.token_id = indices[i];
                result.logprob = std::log(probs[i]);
                result.is_eos = (result.token_id == params.eos_token_id);
                return result;
            }
        }

        result.token_id = indices[k - 1];
        result.logprob = std::log(probs[k - 1]);
        result.is_eos = (result.token_id == params.eos_token_id);
        return result;
    }

    LogitsLocation logits_location() const override { return LogitsLocation::HOST; }
    const char* sampler_type() const override { return "top_k"; }

    void reset() override { rng_state_ = initial_seed_; }

  private:
    uint64_t rng_state_;
    uint64_t initial_seed_;
};

// ─────────────────────────────────────────────────────────────
// GpuGreedySampler: on-device argmax (no D2H logit transfer)
// ─────────────────────────────────────────────────────────────

#if TRTF_HAS_CUDA_KERNELS
class GpuGreedySampler final : public ISampler {
  public:
    explicit GpuGreedySampler(cudaStream_t stream) : stream_(stream) {
        cudaMalloc(&d_token_id_, sizeof(int32_t));
        cudaMalloc(&d_logit_val_, sizeof(float));
    }

    ~GpuGreedySampler() override {
        cudaFree(d_token_id_);
        cudaFree(d_logit_val_);
    }

    GpuGreedySampler(const GpuGreedySampler&) = delete;
    GpuGreedySampler& operator=(const GpuGreedySampler&) = delete;

    SampleResult sample(const float* logits, int32_t vocab_size,
                        const SamplingParams& params) override {
        SampleResult result;
        if (vocab_size <= 0 || logits == nullptr) {
            result.token_id = 0;
            result.is_eos = (0 == params.eos_token_id);
            return result;
        }

        // logits is a device pointer — run GPU argmax kernel
        gpu_argmax(logits, vocab_size, d_token_id_, d_logit_val_, stream_);

        // D2H: copy only token_id + logit (8 bytes total vs vocab_size*4 bytes)
        cudaMemcpyAsync(&h_token_id_, d_token_id_, sizeof(int32_t), cudaMemcpyDeviceToHost,
                        stream_);
        cudaMemcpyAsync(&h_logit_val_, d_logit_val_, sizeof(float), cudaMemcpyDeviceToHost,
                        stream_);
        cudaStreamSynchronize(stream_);

        result.token_id = h_token_id_;
        result.logprob = h_logit_val_;
        result.is_eos = (result.token_id == params.eos_token_id);
        return result;
    }

    LogitsLocation logits_location() const override { return LogitsLocation::DEVICE; }
    const char* sampler_type() const override { return "gpu_greedy"; }

  private:
    cudaStream_t stream_{nullptr};
    int32_t* d_token_id_{nullptr};
    float* d_logit_val_{nullptr};
    int32_t h_token_id_{0};
    float h_logit_val_{0.0f};
};
#endif // TRTF_HAS_CUDA_KERNELS

// ─────────────────────────────────────────────────────────────
// Factory
// ─────────────────────────────────────────────────────────────

SamplingParams sampling_params_from_config(const GenerateConfig& cfg, int32_t default_eos) {
    SamplingParams p;
    p.temperature = cfg.temperature;
    p.top_k = cfg.top_k;
    p.seed = cfg.seed;
    p.eos_token_id = (cfg.eos_token_id >= 0) ? cfg.eos_token_id : default_eos;
    return p;
}

std::unique_ptr<ISampler> create_sampler(const SamplingParams& params) {
    // Greedy when top_k == 1 and no explicit random seed
    if (params.top_k <= 1 && params.seed < 0) {
        return std::make_unique<GreedySampler>();
    }

    // TopK sampler with xorshift64 RNG
    uint64_t seed = (params.seed >= 0) ? static_cast<uint64_t>(params.seed)
                                       : 42ULL; // deterministic default for reproducibility
    return std::make_unique<TopKSampler>(seed);
}

std::unique_ptr<ISampler> create_gpu_greedy_sampler(void* stream) {
#if TRTF_HAS_CUDA_KERNELS
    return std::make_unique<GpuGreedySampler>(static_cast<cudaStream_t>(stream));
#else
    (void)stream;
    return nullptr;
#endif
}

} // namespace trtf
