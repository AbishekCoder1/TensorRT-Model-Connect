#include "runtime/services/text/generation_text_service.h"

#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace trtf::runtime::services::text {

GenerationTextService::GenerationTextService(
    std::shared_ptr<trtf::ITokenizer> tokenizer,
    std::unique_ptr<common::ITextGenerationPort> backend,
    std::shared_ptr<common::ScopedTempDirOwner> tokenizer_temp_dir,
    trtf::GenerationConfig generation_config)
    : mTokenizerTempDir(std::move(tokenizer_temp_dir))
    , mTokenizer(std::move(tokenizer))
    , mBackend(std::move(backend))
    , mGenerationConfig(generation_config)
{
}

const char* GenerationTextService::generate(const char* prompt, std::size_t max_new_tokens)
{
    if (prompt == nullptr)
    {
        return clear_and_return_empty_output();
    }
    return generate_text_only(prompt, resolve_generation_config(max_new_tokens));
}

const char* GenerationTextService::generate(
    const char* prompt,
    const adapters::io::DecodedImage& image,
    std::size_t max_new_tokens)
{
    if (prompt == nullptr)
    {
        return clear_and_return_empty_output();
    }
    if (!supports_vision() || image.empty())
    {
        return generate(prompt, max_new_tokens);
    }
    if (mTokenizer == nullptr || mBackend == nullptr)
    {
        return clear_and_return_empty_output();
    }

    std::vector<float> image_features;
    int32_t num_features = 0;
    int32_t feature_dim = 0;
    std::string error;
    if (!mBackend->prepare_image(image, image_features, num_features, feature_dim, error))
    {
        std::cerr << "[trtf] VL image preparation failed: " << error
                  << ", falling back to text-only" << std::endl;
        return generate(prompt, max_new_tokens);
    }

    const std::string formatted = mBackend->format_vision_prompt(std::string(prompt));
    const auto input_ids = mTokenizer->encode(formatted);
    const auto output_ids = mBackend->generate_with_image(
        input_ids,
        image_features.data(),
        num_features,
        feature_dim,
        resolve_generation_config(max_new_tokens));
    mLastOutput = mTokenizer->decode(output_ids);
    return mLastOutput.c_str();
}

bool GenerationTextService::supports_vision() const
{
    return mBackend != nullptr && mBackend->supports_vision();
}

trtf::GenerationConfig GenerationTextService::resolve_generation_config(std::size_t max_new_tokens) const
{
    trtf::GenerationConfig config = mGenerationConfig;
    if (max_new_tokens > 0)
    {
        config.max_new_tokens = max_new_tokens;
    }
    return config;
}

const char* GenerationTextService::clear_and_return_empty_output()
{
    mLastOutput.clear();
    return mLastOutput.c_str();
}

const char* GenerationTextService::generate_text_only(
    const char* prompt,
    const trtf::GenerationConfig& config)
{
    if (mTokenizer == nullptr || mBackend == nullptr)
    {
        return clear_and_return_empty_output();
    }

    const auto input_ids = mTokenizer->encode(prompt);
    const auto output_ids = mBackend->generate_text(input_ids, config);
    mLastOutput = mTokenizer->decode(output_ids);
    return mLastOutput.c_str();
}

} // namespace trtf::runtime::services::text
