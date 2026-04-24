#include "runtime/pipelines/text_generation_pipeline.h"

#include "runtime/core/trt_common.h"
#include "runtime/core/trt_engine_lifecycle.h"
#include "trtf/runtime/kv_cache.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace trtf {

namespace {
bool env_flag_set(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && v[0] == '1';
}
} // namespace

TextGenerationPipeline::TextGenerationPipeline(
    std::unique_ptr<TrtModule> decoder, std::unique_ptr<IInferenceState> state,
    TextGenConfig config, cudaStream_t stream, std::shared_ptr<ITokenizer> tokenizer,
    std::string model_id_str, std::unique_ptr<ISampler> sampler, std::unique_ptr<TrtModule> prefill)
    : decoder_(std::move(decoder)), prefill_(std::move(prefill)), state_(std::move(state)),
      config_(config), stream_(stream), tokenizer_(std::move(tokenizer)),
      model_id_(std::move(model_id_str)), sampler_(std::move(sampler)),
      logits_output_name_(config.logits_output_name) {
    if (!decoder_ || !decoder_->ok()) {
        throw std::runtime_error("TextGenerationPipeline: invalid decoder module");
    }
    if (!state_ || !state_->ok()) {
        throw std::runtime_error("TextGenerationPipeline: invalid inference state");
    }
    // prefill_ is an optional module — the plugin loader only hands us a
    // fully-constructed one (nullptr otherwise), so no separate ok() check.

    // CUDA Graphs: capture TRT kernels on first step, replay on subsequent steps.
    // Disable with TRTF_DISABLE_CUDA_GRAPH=1.
    // Not enabled on the prefill module — it uses dynamic shapes, which
    // CUDA Graphs cannot capture.
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
    // Bind shared cache buffers on the prefill context too so the
    // batched-sequence engine reads/writes the same KvCache memory as
    // the decode context. Present K/V are NOT bound for prefill — they'd
    // need a (Sq, kv_dim) buffer which doesn't fit KvCache's one-row
    // present slot; the runtime reads prefill outputs directly from the
    // prefill module's own allocations via device_ptr().
    if (prefill_) {
        auto* kv = dynamic_cast<KvCache*>(state_.get());
        if (kv)
            kv->bind_cache_inputs(*prefill_);
    }

    std::vector<float> logits;
    const bool gpu_sampling = (active_sampler->logits_location() == LogitsLocation::DEVICE);

    auto t0 = Clock::now();

    // Dispatch the prompt through either the batched prefill engine (when
    // available + supported) or the legacy token-by-token fallback. After
    // this call, the KV cache is populated and `logits` holds the logits
    // for the last prompt token.
    run_prefill_stage(input_ids, logits, gpu_sampling);

    auto t1 = Clock::now();
    if (trt_log_to_stderr_enabled()) {
        double pms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cerr << "[trtf] Prefill: " << input_ids.size() << " tokens, " << pms << " ms\n";
    }

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

namespace {

// Check preconditions for running the batched prefill engine. Returns the
// KvCache* when all checks pass, nullptr otherwise. Kept narrow so the main
// run_prefill_batched body stays simple enough for the CCN gate.
KvCache* prefill_preflight(TrtModule* prefill, const TextGenConfig& cfg, IInferenceState* state,
                           int32_t sq) {
    if (prefill == nullptr || sq <= 0)
        return nullptr;
    if (cfg.prefill_max_length > 0 && sq > cfg.prefill_max_length)
        return nullptr;
    if (cfg.num_layers <= 0 || cfg.vocab_size <= 0)
        return nullptr;
    return dynamic_cast<KvCache*>(state);
}

// Gather per-layer prefill present-K/V device pointers. Returns false if any
// layer name is not present in the prefill module (caller falls back).
bool gather_prefill_kv_pointers(TrtModule& prefill, const TextGenConfig& cfg,
                                std::vector<const void*>& pk, std::vector<const void*>& pv) {
    pk.resize(static_cast<std::size_t>(cfg.num_layers));
    pv.resize(static_cast<std::size_t>(cfg.num_layers));
    for (int32_t i = 0; i < cfg.num_layers; ++i) {
        const auto li = static_cast<std::size_t>(i);
        pk[li] = prefill.device_ptr(expand_layer_name(cfg.present_k_pattern, i));
        pv[li] = prefill.device_ptr(expand_layer_name(cfg.present_v_pattern, i));
        if (pk[li] == nullptr || pv[li] == nullptr)
            return false;
    }
    return true;
}

} // namespace

void TextGenerationPipeline::run_prefill_stage(const std::vector<int32_t>& input_ids,
                                               std::vector<float>& logits, bool gpu_sampling) {
    if (prefill_ && !gpu_sampling && run_prefill_batched(input_ids, logits))
        return;

    // Fallback: token-by-token prefill on the decode engine, then one decode
    // step on the last input token to produce its logits.
    for (std::size_t i = 0; i + 1 < input_ids.size(); ++i) {
        if (gpu_sampling)
            run_step_device(input_ids[i]);
        else
            run_step(input_ids[i], logits);
    }
    const int32_t current_token = input_ids.back();
    if (gpu_sampling)
        run_step_device(current_token);
    else
        run_step(current_token, logits);
}

bool TextGenerationPipeline::run_prefill_batched(const std::vector<int32_t>& input_ids,
                                                 std::vector<float>& logits) {
    const auto sq = static_cast<int32_t>(input_ids.size());
    KvCache* kv = prefill_preflight(prefill_.get(), config_, state_.get(), sq);
    if (kv == nullptr)
        return false;

    TensorMap inputs;
    Tensor tok_t{const_cast<int32_t*>(input_ids.data()), {static_cast<int64_t>(sq)}, DType::kInt32};
    inputs[config_.token_id_name] = tok_t;
    // KvCache builds position_id (seq_len,) and the causal-with-cache
    // attention_mask (seq_len, max_cache + seq_len).
    kv->prepare_step(inputs, sq);

    TensorMap outputs = prefill_->forward(inputs);
    auto logits_it = outputs.find(config_.logits_output_name);
    if (logits_it == outputs.end())
        return false;

    // The dual-profile builder slices hidden_state to the last row before
    // the LM head, so the logits output is always (1, vocab) regardless
    // of the active profile's Sq — matching the contract of the legacy
    // single-token decode engine.
    const auto vocab = static_cast<std::size_t>(config_.vocab_size);
    const auto& lt = logits_it->second;
    if (static_cast<std::size_t>(lt.numel()) < vocab)
        return false;
    logits.resize(vocab);
    std::memcpy(logits.data(), lt.data, vocab * sizeof(float));

    std::vector<const void*> pk, pv;
    if (!gather_prefill_kv_pointers(*prefill_, config_, pk, pv))
        return false;
    kv->write_prefill_kv(pk, pv, sq);
    if (trt_log_to_stderr_enabled())
        std::cerr << "[trtf] Batched prefill (profile 0): " << sq << " tokens in one call\n";
    return true;
}

int32_t TextGenerationPipeline::argmax(const std::vector<float>& logits) {
    if (logits.empty())
        return 0;
    return static_cast<int32_t>(
        std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
}

} // namespace trtf
