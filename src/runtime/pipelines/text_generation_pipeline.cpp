#include "runtime/pipelines/text_generation_pipeline.h"

#if TRTF_HAS_TRT

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

namespace trtf {

namespace {
bool env_flag_set(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && v[0] == '1';
}
} // namespace

TextGenerationPipeline::TextGenerationPipeline(std::unique_ptr<TrtModule> decoder,
                                               std::unique_ptr<IInferenceState> state,
                                               TextGenConfig config, cudaStream_t stream,
                                               std::shared_ptr<ITokenizer> tokenizer,
                                               std::string model_id_str,
                                               std::unique_ptr<ISampler> sampler)
    : decoder_(std::move(decoder)), state_(std::move(state)), config_(config), stream_(stream),
      tokenizer_(std::move(tokenizer)), model_id_(std::move(model_id_str)),
      sampler_(std::move(sampler)), logits_output_name_(config.logits_output_name) {
    if (!decoder_ || !decoder_->ok()) {
        throw std::runtime_error("TextGenerationPipeline: invalid decoder module");
    }
    if (!state_ || !state_->ok()) {
        throw std::runtime_error("TextGenerationPipeline: invalid inference state");
    }

    // CUDA Graphs: capture TRT kernels on first step, replay on subsequent steps.
    // Disable with TRTF_DISABLE_CUDA_GRAPH=1.
    if (!env_flag_set("TRTF_DISABLE_CUDA_GRAPH"))
        decoder_->enable_cuda_graph();

    // GPU-side argmax: eliminates D2H transfer of full logit vector.
    // Enable with TRTF_GPU_ARGMAX=1 (greedy decoding only).
    if (!sampler_ && env_flag_set("TRTF_GPU_ARGMAX")) {
        auto gpu_sampler = create_gpu_greedy_sampler(stream_);
        if (gpu_sampler)
            sampler_ = std::move(gpu_sampler);
    }
}

// Encode a prompt, optionally applying a chat template first.
// Deduplicates the leading BOS token that chat templates embed but
// the tokenizer's add_special_tokens may also prepend.
static std::vector<int32_t> encode_prompt(const ITokenizer& tokenizer, const TextGenConfig& config,
                                          const std::string& prompt, const GenerateConfig& cfg) {
    std::string effective = prompt;
    bool templated = false;
    if (cfg.use_chat_template && config.chat_template_format != ChatTemplateFormat::kNone) {
        effective = apply_chat_template(config.chat_template_format, prompt, cfg.enable_thinking);
        templated = true;
    }
    auto ids = tokenizer.encode(effective);
    if (templated && ids.size() >= 2 && config.id_bos >= 0 && ids[0] == config.id_bos &&
        ids[1] == config.id_bos) {
        ids.erase(ids.begin());
    }
    return ids;
}

TextResult TextGenerationPipeline::generate(const std::string& prompt, const GenerateConfig& cfg) {
    if (!tokenizer_) {
        throw std::runtime_error("TextGenerationPipeline: no tokenizer configured");
    }

    auto input_ids = encode_prompt(*tokenizer_, config_, prompt, cfg);
    int32_t max_new = (cfg.max_new_tokens > 0) ? cfg.max_new_tokens : 128;
    int32_t eos = (cfg.eos_token_id >= 0) ? cfg.eos_token_id : config_.id_eos;

    auto sp = sampling_params_from_config(cfg, eos);
    auto timed = generate_from_ids(input_ids, max_new, sp, cfg.collect_timing);

    // Decode only the NEW tokens (skip input)
    std::vector<int32_t> new_tokens(timed.token_ids.begin() +
                                        static_cast<std::ptrdiff_t>(input_ids.size()),
                                    timed.token_ids.end());
    std::string text = tokenizer_->decode(new_tokens);

    return TextResult{std::move(text), std::move(new_tokens), timed.prefill_ms, timed.decode_ms};
}

TextGenerationPipeline::GenerationResult
TextGenerationPipeline::generate_ids(const std::vector<int32_t>& input_ids,
                                     const GenerateConfig& cfg) {
    int32_t max_new = cfg.max_new_tokens; // honour exact value (0 = no generation)
    int32_t eos = (cfg.eos_token_id >= 0) ? cfg.eos_token_id : config_.id_eos;
    auto sp = sampling_params_from_config(cfg, eos);
    return GenerationResult{generate_from_ids(input_ids, max_new, sp).token_ids};
}

TextGenerationPipeline::TimedGenResult
TextGenerationPipeline::generate_from_ids(const std::vector<int32_t>& input_ids,
                                          int32_t max_new_tokens, const SamplingParams& params,
                                          bool collect_timing) {
    using Clock = std::chrono::steady_clock;

    if (max_new_tokens == 0 || input_ids.empty()) {
        return TimedGenResult{input_ids, 0.0, 0.0};
    }

    // Create a per-call sampler if none was injected at construction time.
    ISampler* active_sampler = sampler_.get();
    std::unique_ptr<ISampler> local_sampler;
    if (!active_sampler) {
        local_sampler = create_sampler(params);
        active_sampler = local_sampler.get();
    }
    active_sampler->reset();

    state_->reset();
    state_->bind_to(*decoder_);

    std::vector<float> logits;
    const bool gpu_sampling = (active_sampler->logits_location() == LogitsLocation::DEVICE);

    auto t0 = Clock::now();

    // Prefill: process all input tokens except the last.
    for (std::size_t i = 0; i + 1 < input_ids.size(); ++i) {
        if (gpu_sampling)
            run_step_device(input_ids[i]);
        else
            run_step(input_ids[i], logits);
    }

    // Start decode from last input token
    int32_t current_token = input_ids.back();
    if (gpu_sampling)
        run_step_device(current_token);
    else
        run_step(current_token, logits);

    auto t1 = Clock::now();

    // Decode: autoregressive loop
    std::vector<int32_t> output = input_ids;
    int32_t decode_steps =
        run_decode_loop(active_sampler, params, output, logits, max_new_tokens, gpu_sampling);

    auto t2 = Clock::now();

    double prefill_ms =
        collect_timing ? std::chrono::duration<double, std::milli>(t1 - t0).count() : 0.0;
    double decode_ms =
        collect_timing ? std::chrono::duration<double, std::milli>(t2 - t1).count() : 0.0;

    return TimedGenResult{std::move(output), prefill_ms, decode_ms};
}

int32_t TextGenerationPipeline::run_decode_loop(ISampler* sampler, const SamplingParams& params,
                                                std::vector<int32_t>& output,
                                                std::vector<float>& logits, int32_t max_new_tokens,
                                                bool gpu_sampling) {
    const int32_t vocab_size =
        gpu_sampling ? config_.vocab_size : static_cast<int32_t>(logits.size());
    auto decode_start = std::chrono::steady_clock::now();
    int32_t steps = 0;
    for (int32_t step = 0; step < max_new_tokens; ++step) {
        const float* sample_ptr = gpu_sampling ? d_logits_ptr_ : logits.data();
        SampleResult result = sampler->sample(sample_ptr, vocab_size, params);
        output.push_back(result.token_id);
        if (result.is_eos)
            break;
        if (gpu_sampling)
            run_step_device(result.token_id);
        else
            run_step(result.token_id, logits);
        ++steps;
    }
    auto decode_end = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(decode_end - decode_start).count();
    if (steps > 0 && trt_log_to_stderr_enabled()) {
        double tps = steps * 1000.0 / ms;
        std::cerr << "[trtf] Decode: " << steps << " tokens, " << ms << " ms, " << tps << " tok/s"
                  << (decoder_->cuda_graph_active() ? " [CUDA Graph ON]" : "") << '\n';
    }
    return steps;
}

void TextGenerationPipeline::run_step(int32_t token_id, std::vector<float>& logits) {
    TensorMap inputs;

    Tensor token_tensor;
    token_tensor.data = &token_id;
    token_tensor.shape = {1};
    token_tensor.dtype = DType::kInt32;
    inputs[config_.token_id_name] = token_tensor;

    state_->prepare_step(inputs);

    TensorMap outputs = decoder_->forward(inputs);

    auto it = outputs.find(logits_output_name_);
    if (it == outputs.end()) {
        throw std::runtime_error("TextGenerationPipeline: no '" + logits_output_name_ + "' output");
    }

    const auto& logits_tensor = it->second;
    auto num_logits = logits_tensor.numel();
    logits.resize(static_cast<std::size_t>(num_logits));
    std::memcpy(logits.data(), logits_tensor.data, num_logits * sizeof(float));

    state_->advance();
}

void TextGenerationPipeline::run_step_device(int32_t token_id) {
    TensorMap inputs;

    Tensor token_tensor;
    token_tensor.data = &token_id;
    token_tensor.shape = {1};
    token_tensor.dtype = DType::kInt32;
    inputs[config_.token_id_name] = token_tensor;

    state_->prepare_step(inputs);

    // Use forward_async + sync instead of forward() to skip the D2H output copy.
    // The GPU argmax kernel reads logits directly from the device buffer.
    decoder_->forward_async(inputs);
    decoder_->sync();

    // Get device pointer to logits output buffer (still on GPU).
    d_logits_ptr_ = static_cast<const float*>(decoder_->device_ptr(logits_output_name_));

    state_->advance();
}

int32_t TextGenerationPipeline::argmax(const std::vector<float>& logits) {
    if (logits.empty())
        return 0;
    return static_cast<int32_t>(
        std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
}

} // namespace trtf

#endif // TRTF_HAS_TRT
