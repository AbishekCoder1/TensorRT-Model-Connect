#include "runtime/pipelines/text_generation_pipeline.h"

#if TRTF_HAS_TRT

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace trtf {

TextGenerationPipeline::TextGenerationPipeline(
    std::unique_ptr<TrtModule> decoder,
    std::unique_ptr<IInferenceState> state,
    TextGenConfig config,
    cudaStream_t stream,
    std::shared_ptr<ITokenizer> tokenizer,
    std::string model_id_str,
    std::unique_ptr<ISampler> sampler)
    : decoder_(std::move(decoder))
    , state_(std::move(state))
    , config_(config)
    , stream_(stream)
    , tokenizer_(std::move(tokenizer))
    , model_id_(std::move(model_id_str))
    , sampler_(std::move(sampler))
{
    if (!decoder_ || !decoder_->ok())
    {
        throw std::runtime_error("TextGenerationPipeline: invalid decoder module");
    }
    if (!state_ || !state_->ok())
    {
        throw std::runtime_error("TextGenerationPipeline: invalid inference state");
    }
}

TextResult TextGenerationPipeline::generate(
    const std::string& prompt, const GenerateConfig& cfg)
{
    if (!tokenizer_)
    {
        throw std::runtime_error("TextGenerationPipeline: no tokenizer configured");
    }

    auto input_ids = tokenizer_->encode(prompt);
    int32_t max_new = (cfg.max_new_tokens > 0) ? cfg.max_new_tokens : 128;
    int32_t eos = (cfg.eos_token_id >= 0) ? cfg.eos_token_id : config_.id_eos;

    auto sp = sampling_params_from_config(cfg, eos);
    auto output_ids = generate_from_ids(input_ids, max_new, sp);

    // Decode only the NEW tokens (skip input)
    std::vector<int32_t> new_tokens(
        output_ids.begin() + static_cast<std::ptrdiff_t>(input_ids.size()),
        output_ids.end());
    std::string text = tokenizer_->decode(new_tokens);

    return TextResult{std::move(text), std::move(new_tokens)};
}

TextGenerationPipeline::GenerationResult TextGenerationPipeline::generate_ids(
    const std::vector<int32_t>& input_ids,
    const GenerateConfig& cfg)
{
    int32_t max_new = cfg.max_new_tokens;  // honour exact value (0 = no generation)
    int32_t eos = (cfg.eos_token_id >= 0) ? cfg.eos_token_id : config_.id_eos;
    auto sp = sampling_params_from_config(cfg, eos);
    return GenerationResult{generate_from_ids(input_ids, max_new, sp)};
}

std::vector<int32_t> TextGenerationPipeline::generate_from_ids(
    const std::vector<int32_t>& input_ids,
    int32_t max_new_tokens,
    const SamplingParams& params)
{
    if (max_new_tokens == 0 || input_ids.empty())
    {
        return input_ids;
    }

    // Create a per-call sampler if none was injected at construction time.
    ISampler* active_sampler = sampler_.get();
    std::unique_ptr<ISampler> local_sampler;
    if (!active_sampler)
    {
        local_sampler = create_sampler(params);
        active_sampler = local_sampler.get();
    }
    active_sampler->reset();

    state_->reset();
    state_->bind_to(*decoder_);

    std::vector<float> logits;

    // Prefill: process all input tokens except the last
    for (std::size_t i = 0; i + 1 < input_ids.size(); ++i)
    {
        run_step(input_ids[i], logits);
    }

    // Start decode from last input token
    int32_t current_token = input_ids.back();
    run_step(current_token, logits);

    // Decode: autoregressive loop
    std::vector<int32_t> output = input_ids;
    const int32_t vocab_size = static_cast<int32_t>(logits.size());

    for (int32_t step = 0; step < max_new_tokens; ++step)
    {
        SampleResult result = active_sampler->sample(
            logits.data(), vocab_size, params);
        output.push_back(result.token_id);

        if (result.is_eos)
        {
            break;
        }

        run_step(result.token_id, logits);
    }

    return output;
}

void TextGenerationPipeline::run_step(int32_t token_id, std::vector<float>& logits)
{
    TensorMap inputs;

    Tensor token_tensor;
    token_tensor.data = &token_id;
    token_tensor.shape = {1};
    token_tensor.dtype = DType::kInt32;
    inputs["token_id"] = token_tensor;

    state_->prepare_step(inputs);

    TensorMap outputs = decoder_->forward(inputs);

    auto it = outputs.find("logits");
    if (it == outputs.end())
    {
        throw std::runtime_error("TextGenerationPipeline: no 'logits' output");
    }

    const auto& logits_tensor = it->second;
    auto num_logits = logits_tensor.numel();
    logits.resize(static_cast<std::size_t>(num_logits));
    std::memcpy(logits.data(), logits_tensor.data,
                num_logits * sizeof(float));

    state_->advance();
}

int32_t TextGenerationPipeline::argmax(const std::vector<float>& logits)
{
    if (logits.empty()) return 0;
    return static_cast<int32_t>(
        std::distance(logits.begin(),
                      std::max_element(logits.begin(), logits.end())));
}

} // namespace trtf

#endif // TRTF_HAS_TRT
