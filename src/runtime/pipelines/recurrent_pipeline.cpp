#include "runtime/pipelines/recurrent_pipeline.h"

#if TRTF_HAS_TRT

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace trtf {

RecurrentPipeline::RecurrentPipeline(
    std::unique_ptr<TrtModule> decoder,
    std::unique_ptr<IStateManager> state,
    RecurrentGenConfig config,
    cudaStream_t stream,
    const char* name,
    std::shared_ptr<ITokenizer> tokenizer,
    std::string model_id_str)
    : decoder_(std::move(decoder))
    , state_(std::move(state))
    , config_(config)
    , stream_(stream)
    , name_(name)
    , tokenizer_(std::move(tokenizer))
    , model_id_(std::move(model_id_str))
{
    if (!decoder_ || !decoder_->ok())
        throw std::runtime_error(std::string(name_) + ": invalid decoder module");
}

TextResult RecurrentPipeline::generate(
    const std::string& prompt, const GenerateConfig& cfg)
{
    if (!tokenizer_)
        throw std::runtime_error(std::string(name_) + ": no tokenizer configured");

    auto input_ids = tokenizer_->encode(prompt);
    int32_t max_new = (cfg.max_new_tokens > 0) ? cfg.max_new_tokens : 128;
    int32_t eos = (cfg.eos_token_id >= 0) ? cfg.eos_token_id : config_.id_eos;

    auto output_ids = generate_from_ids(input_ids, max_new, eos);

    std::vector<int32_t> new_tokens(
        output_ids.begin() + static_cast<std::ptrdiff_t>(input_ids.size()),
        output_ids.end());
    std::string text = tokenizer_->decode(new_tokens);

    return TextResult{std::move(text), std::move(new_tokens)};
}

RecurrentPipeline::GenerationResult RecurrentPipeline::generate_ids(
    const std::vector<int32_t>& input_ids,
    const GenerateConfig& cfg)
{
    int32_t max_new = cfg.max_new_tokens;
    int32_t eos = (cfg.eos_token_id >= 0) ? cfg.eos_token_id : config_.id_eos;
    return GenerationResult{generate_from_ids(input_ids, max_new, eos)};
}

std::vector<int32_t> RecurrentPipeline::generate_from_ids(
    const std::vector<int32_t>& input_ids,
    int32_t max_new_tokens,
    int32_t eos_token_id)
{
    if (max_new_tokens == 0 || input_ids.empty())
        return input_ids;

    state_->reset();
    state_->bind_to(*decoder_);

    std::vector<float> logits;

    for (std::size_t i = 0; i + 1 < input_ids.size(); ++i)
        run_step(input_ids[i], logits);

    run_step(input_ids.back(), logits);

    std::vector<int32_t> output = input_ids;

    for (int32_t step = 0; step < max_new_tokens; ++step)
    {
        int32_t next = argmax(logits);
        output.push_back(next);
        if (next == eos_token_id) break;
        run_step(next, logits);
    }

    return output;
}

void RecurrentPipeline::run_step(int32_t token_id, std::vector<float>& logits)
{
    Tensor token_t;
    token_t.data = &token_id;
    token_t.shape = {1};
    token_t.dtype = DType::kInt32;

    TensorMap inputs;
    inputs["token_id"] = token_t;

    int32_t pos = state_->position();
    std::vector<float> mask;

    if (state_->has_mask())
    {
        state_->build_mask(mask);

        Tensor pos_t;
        pos_t.data = &pos;
        pos_t.shape = {1};
        pos_t.dtype = DType::kInt32;
        inputs["position_id"] = pos_t;

        Tensor mask_t;
        mask_t.data = mask.data();
        mask_t.shape = {static_cast<int64_t>(mask.size())};
        mask_t.dtype = DType::kFloat32;
        inputs["attention_mask"] = mask_t;
    }

    TensorMap outputs = decoder_->forward(inputs);

    auto it = outputs.find("logits");
    if (it == outputs.end())
        throw std::runtime_error(std::string(name_) + ": no 'logits' output");

    auto num = it->second.numel();
    logits.resize(static_cast<std::size_t>(num));
    std::memcpy(logits.data(), it->second.data, num * sizeof(float));

    state_->advance();
}

int32_t RecurrentPipeline::argmax(const std::vector<float>& logits)
{
    if (logits.empty()) return 0;
    return static_cast<int32_t>(
        std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
}

} // namespace trtf

#endif // TRTF_HAS_TRT
