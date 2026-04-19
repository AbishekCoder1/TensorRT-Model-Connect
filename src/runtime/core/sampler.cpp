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

#if TRTF_HAS_LIBTORCH_MULTINOMIAL
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <string_view>
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

struct FilteredDistribution {
    std::vector<int32_t> indices;
    std::vector<float> probs;
    int32_t keep{0};
};

static FilteredDistribution build_filtered_distribution(const float* logits, int32_t vocab_size,
                                                        const SamplingParams& params) {
    const int32_t n = vocab_size;
    const int32_t k = std::min(std::max(params.top_k, 1), n);

    FilteredDistribution dist;
    dist.indices.resize(static_cast<std::size_t>(n));
    std::iota(dist.indices.begin(), dist.indices.end(), 0);
    std::partial_sort(
        dist.indices.begin(), dist.indices.begin() + k, dist.indices.end(), [&](int32_t a, int32_t b) {
            return logits[static_cast<std::size_t>(a)] > logits[static_cast<std::size_t>(b)];
        });

    const float max_logit = logits[static_cast<std::size_t>(dist.indices[0])];
    dist.probs.resize(static_cast<std::size_t>(k));
    float sum = 0.0F;
    for (int32_t i = 0; i < k; ++i) {
        const float scaled =
            (logits[static_cast<std::size_t>(dist.indices[static_cast<std::size_t>(i)])] - max_logit)
            / params.temperature;
        dist.probs[static_cast<std::size_t>(i)] = std::exp(scaled);
        sum += dist.probs[static_cast<std::size_t>(i)];
    }

    if (sum > 0.0F) {
        for (int32_t i = 0; i < k; ++i)
            dist.probs[static_cast<std::size_t>(i)] /= sum;
    } else {
        for (int32_t i = 0; i < k; ++i)
            dist.probs[static_cast<std::size_t>(i)] = 1.0F / static_cast<float>(k);
    }

    int32_t keep = k;
    const float max_prob = dist.probs.empty() ? 1.0F : dist.probs[0];
    if (params.min_p > 0.0F && max_prob > 0.0F) {
        const float min_prob = params.min_p * max_prob;
        keep = 0;
        while (keep < k && dist.probs[static_cast<std::size_t>(keep)] >= min_prob)
            ++keep;
        keep = std::max(keep, 1);
    }
    if (params.top_p < 1.0F) {
        float cumulative = 0.0F;
        int32_t top_p_keep = 0;
        while (top_p_keep < keep) {
            cumulative += dist.probs[static_cast<std::size_t>(top_p_keep)];
            ++top_p_keep;
            if (cumulative >= params.top_p)
                break;
        }
        keep = std::max(top_p_keep, 1);
    }

    if (keep < k) {
        float kept_sum = 0.0F;
        for (int32_t i = 0; i < keep; ++i)
            kept_sum += dist.probs[static_cast<std::size_t>(i)];
        if (kept_sum > 0.0F) {
            for (int32_t i = 0; i < keep; ++i)
                dist.probs[static_cast<std::size_t>(i)] /= kept_sum;
        } else {
            for (int32_t i = 0; i < keep; ++i)
                dist.probs[static_cast<std::size_t>(i)] = 1.0F / static_cast<float>(keep);
        }
    }

    dist.keep = keep;
    return dist;
}

static bool use_torch_multinomial_sampler() {
#if TRTF_HAS_LIBTORCH_MULTINOMIAL
    const char* env = std::getenv("TRTF_USE_TORCH_MULTINOMIAL");
    if (env == nullptr)
        return true;
    const std::string_view value(env);
    return value != "0" && value != "false" && value != "FALSE";
#else
    return false;
#endif
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

        const FilteredDistribution dist = build_filtered_distribution(logits, vocab_size, params);

        // xorshift64 random number generation
        rng_state_ ^= rng_state_ << 13;
        rng_state_ ^= rng_state_ >> 7;
        rng_state_ ^= rng_state_ << 17;
        float u = static_cast<float>(rng_state_ & 0xFFFFFFFF) / 4294967296.0F;

        // Sample from cumulative distribution
        float cumulative = 0.0F;
        for (int32_t i = 0; i < dist.keep; ++i) {
            cumulative += dist.probs[static_cast<std::size_t>(i)];
            if (u < cumulative) {
                result.token_id = dist.indices[static_cast<std::size_t>(i)];
                result.logprob = std::log(dist.probs[static_cast<std::size_t>(i)]);
                result.is_eos = (result.token_id == params.eos_token_id);
                return result;
            }
        }

        result.token_id = dist.indices[static_cast<std::size_t>(dist.keep - 1)];
        result.logprob = std::log(dist.probs[static_cast<std::size_t>(dist.keep - 1)]);
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

#if TRTF_HAS_LIBTORCH_MULTINOMIAL
class TorchCudaMultinomialSampler final : public ISampler {
  public:
    explicit TorchCudaMultinomialSampler(uint64_t initial_seed)
        : initial_seed_(initial_seed == 0 ? 1 : initial_seed),
          generator_(at::cuda::detail::createCUDAGenerator()) {
        reset();
    }

    SampleResult sample(const float* logits, int32_t vocab_size,
                        const SamplingParams& params) override {
        SampleResult result;

        if (vocab_size <= 0 || logits == nullptr) {
            result.token_id = 0;
            result.is_eos = (0 == params.eos_token_id);
            return result;
        }

        if (params.temperature < 1e-6F)
            return argmax_over_logits(logits, vocab_size, params.eos_token_id);

        const FilteredDistribution dist = build_filtered_distribution(logits, vocab_size, params);
        ensure_probability_buffer(vocab_size);
        probability_buffer_.zero_();

        std::vector<int64_t> kept_indices(static_cast<std::size_t>(dist.keep));
        std::vector<float> kept_probs(static_cast<std::size_t>(dist.keep));
        for (int32_t i = 0; i < dist.keep; ++i) {
            kept_indices[static_cast<std::size_t>(i)] =
                static_cast<int64_t>(dist.indices[static_cast<std::size_t>(i)]);
            kept_probs[static_cast<std::size_t>(i)] = dist.probs[static_cast<std::size_t>(i)];
        }

        at::Tensor indices = at::from_blob(
                                 kept_indices.data(), {dist.keep},
                                 at::TensorOptions().dtype(at::kLong).device(at::kCPU))
                                 .clone()
                                 .to(at::TensorOptions().dtype(at::kLong).device(at::kCUDA));
        at::Tensor probs = at::from_blob(
                               kept_probs.data(), {dist.keep},
                               at::TensorOptions().dtype(at::kFloat).device(at::kCPU))
                               .clone()
                               .to(at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));
        probability_buffer_.index_put_({indices}, probs, false);

        at::Tensor choice = at::multinomial(probability_buffer_, 1, false, generator_);
        result.token_id = static_cast<int32_t>(choice.item<int64_t>());
        const float picked_prob = probability_buffer_.index({result.token_id}).item<float>();
        result.logprob = std::log(std::max(picked_prob, 1e-20F));
        result.is_eos = (result.token_id == params.eos_token_id);
        return result;
    }

    LogitsLocation logits_location() const override { return LogitsLocation::HOST; }
    const char* sampler_type() const override { return "torch_multinomial"; }

    void reset() override {
        generator_.set_current_seed(initial_seed_);
        generator_.set_offset(0);
    }

  private:
    void ensure_probability_buffer(int32_t vocab_size) {
        if (!probability_buffer_.defined() || probability_buffer_.numel() != vocab_size) {
            probability_buffer_ = at::zeros(
                {vocab_size}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));
        }
    }

    uint64_t initial_seed_;
    at::Generator generator_;
    at::Tensor probability_buffer_;
};
#endif

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
    p.top_p = cfg.top_p;
    p.min_p = cfg.min_p;
    p.seed = cfg.seed;
    p.eos_token_id = (cfg.eos_token_id >= 0) ? cfg.eos_token_id : default_eos;
    return p;
}

std::unique_ptr<ISampler> create_sampler(const SamplingParams& params) {
    // Greedy when sampling is fully disabled and no explicit random seed is set.
    if (params.top_k <= 1 && params.top_p >= 1.0F && params.min_p <= 0.0F && params.seed < 0) {
        return std::make_unique<GreedySampler>();
    }

    uint64_t seed = (params.seed >= 0) ? static_cast<uint64_t>(params.seed)
                                       : 42ULL; // deterministic default for reproducibility
#if TRTF_HAS_LIBTORCH_MULTINOMIAL
    if (use_torch_multinomial_sampler())
        return std::make_unique<TorchCudaMultinomialSampler>(seed);
#endif

    // TopK sampler with xorshift64 RNG
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
