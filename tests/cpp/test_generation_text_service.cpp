#include "runtime/services/text/generation_text_service.h"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

int g_failures = 0;

void check(bool condition, const char* name)
{
    if (!condition)
    {
        std::cerr << "FAIL: " << name << '\n';
        ++g_failures;
    }
}

class FakeTokenizer final : public trtf::ITokenizer {
public:
    std::vector<int32_t> encode(const std::string& text) const override
    {
        ++encode_calls;
        last_encoded_text = text;
        return next_encode;
    }

    std::string decode(const std::vector<int32_t>& ids) const override
    {
        ++decode_calls;
        last_decoded_ids = ids;
        return next_decode;
    }

    int32_t id_for_token(std::string_view token) const override
    {
        return static_cast<int32_t>(token.size());
    }

    std::string token_for_id(int32_t id) const override
    {
        return std::to_string(id);
    }

    mutable int encode_calls{0};
    mutable int decode_calls{0};
    mutable std::string last_encoded_text;
    mutable std::vector<int32_t> last_decoded_ids;
    std::vector<int32_t> next_encode{1, 2, 3};
    std::string next_decode{"decoded"};
};

class FakeTextPort final : public trtf::runtime::services::common::ITextGenerationPort {
public:
    std::vector<int32_t> generate_text(
        const std::vector<int32_t>& input_ids,
        const trtf::GenerationConfig& config) override
    {
        ++generate_text_calls;
        last_text_ids = input_ids;
        last_text_max_new_tokens = static_cast<int32_t>(config.max_new_tokens);
        return text_result;
    }

    bool supports_vision() const override
    {
        return vision_enabled;
    }

    bool prepare_image(
        const trtf::runtime::adapters::io::DecodedImage& image,
        std::vector<float>& image_features,
        int32_t& num_features,
        int32_t& feature_dim,
        std::string& error) override
    {
        ++prepare_image_calls;
        last_image_width = image.width;
        last_image_height = image.height;
        if (!prepare_success)
        {
            error = prepare_error;
            return false;
        }
        image_features = next_image_features;
        num_features = next_num_features;
        feature_dim = next_feature_dim;
        return true;
    }

    std::string format_vision_prompt(const std::string& prompt) const override
    {
        return vision_prompt_prefix + prompt;
    }

    std::vector<int32_t> generate_with_image(
        const std::vector<int32_t>& input_ids,
        const float* image_features,
        int32_t num_features,
        int32_t feature_dim,
        const trtf::GenerationConfig& config) override
    {
        ++generate_with_image_calls;
        last_image_ids = input_ids;
        last_image_feature_count = num_features;
        last_image_feature_dim = feature_dim;
        last_image_max_new_tokens = static_cast<int32_t>(config.max_new_tokens);
        captured_features.assign(image_features, image_features + (num_features * feature_dim));
        return image_result;
    }

    bool vision_enabled{false};
    bool prepare_success{true};
    int generate_text_calls{0};
    int prepare_image_calls{0};
    int generate_with_image_calls{0};
    int32_t last_text_max_new_tokens{0};
    int32_t last_image_feature_count{0};
    int32_t last_image_feature_dim{0};
    int32_t last_image_max_new_tokens{0};
    int32_t last_image_width{0};
    int32_t last_image_height{0};
    std::string prepare_error{"bad-image"};
    std::string vision_prompt_prefix{"[vl] "};
    std::vector<int32_t> last_text_ids;
    std::vector<int32_t> last_image_ids;
    std::vector<float> captured_features;
    std::vector<float> next_image_features{1.0F, 2.0F, 3.0F, 4.0F};
    int32_t next_num_features{2};
    int32_t next_feature_dim{2};
    std::vector<int32_t> text_result{7, 8};
    std::vector<int32_t> image_result{9, 10};
};

void test_text_only_generation_uses_port_and_tokenizer()
{
    auto tokenizer = std::make_shared<FakeTokenizer>();
    auto* tokenizer_ptr = tokenizer.get();
    auto port = std::make_unique<FakeTextPort>();
    auto* port_ptr = port.get();

    trtf::GenerationConfig config;
    config.max_new_tokens = 11;
    trtf::runtime::services::text::GenerationTextService service(
        tokenizer,
        std::move(port),
        std::make_shared<trtf::runtime::services::common::ScopedTempDirOwner>(),
        config);

    const char* output = service.generate("hello", 5);
    check(std::strcmp(output, "decoded") == 0, "text generation returns decoded output");
    check(tokenizer_ptr->encode_calls == 1, "text generation encodes prompt");
    check(tokenizer_ptr->last_encoded_text == "hello", "text generation encodes original prompt");
    check(port_ptr->generate_text_calls == 1, "text generation uses text port");
    check(port_ptr->last_text_ids == std::vector<int32_t>({1, 2, 3}), "text generation forwards token ids");
    check(port_ptr->last_text_max_new_tokens == 5, "text generation overrides max tokens");
}

void test_null_prompt_returns_empty_output()
{
    auto tokenizer = std::make_shared<FakeTokenizer>();
    auto port = std::make_unique<FakeTextPort>();
    trtf::runtime::services::text::GenerationTextService service(
        tokenizer,
        std::move(port),
        std::make_shared<trtf::runtime::services::common::ScopedTempDirOwner>());

    const char* output = service.generate(nullptr, 4);
    check(std::strcmp(output, "") == 0, "null prompt returns empty output");
}

void test_vision_generation_uses_vision_port()
{
    auto tokenizer = std::make_shared<FakeTokenizer>();
    auto* tokenizer_ptr = tokenizer.get();
    auto port = std::make_unique<FakeTextPort>();
    auto* port_ptr = port.get();
    port_ptr->vision_enabled = true;
    tokenizer_ptr->next_decode = "vision-output";

    trtf::runtime::services::text::GenerationTextService service(
        tokenizer,
        std::move(port),
        std::make_shared<trtf::runtime::services::common::ScopedTempDirOwner>());

    const char* output = service.generate("caption", {{255, 0, 0}, 1, 1, 3}, 6);
    check(std::strcmp(output, "vision-output") == 0, "vision generation returns decoded output");
    check(port_ptr->prepare_image_calls == 1, "vision generation prepares image");
    check(port_ptr->last_image_width == 1 && port_ptr->last_image_height == 1,
        "vision generation forwards decoded image");
    check(port_ptr->generate_with_image_calls == 1, "vision generation uses image path");
    check(tokenizer_ptr->last_encoded_text == "[vl] caption", "vision prompt is formatted before tokenization");
    check(port_ptr->captured_features == std::vector<float>({1.0F, 2.0F, 3.0F, 4.0F}),
        "vision generation forwards image features");
    check(port_ptr->last_image_max_new_tokens == 6, "vision generation forwards max tokens");
}

void test_vision_prepare_failure_falls_back_to_text()
{
    auto tokenizer = std::make_shared<FakeTokenizer>();
    auto* tokenizer_ptr = tokenizer.get();
    auto port = std::make_unique<FakeTextPort>();
    auto* port_ptr = port.get();
    port_ptr->vision_enabled = true;
    port_ptr->prepare_success = false;

    trtf::runtime::services::text::GenerationTextService service(
        tokenizer,
        std::move(port),
        std::make_shared<trtf::runtime::services::common::ScopedTempDirOwner>());

    const char* output = service.generate("fallback", {{255, 0, 0}, 1, 1, 3}, 3);
    check(std::strcmp(output, "decoded") == 0, "vision prepare failure falls back to text output");
    check(port_ptr->generate_with_image_calls == 0, "vision prepare failure skips image generation");
    check(port_ptr->generate_text_calls == 1, "vision prepare failure falls back to text generation");
    check(tokenizer_ptr->last_encoded_text == "fallback", "fallback uses original prompt");
}

} // namespace

int main()
{
    test_text_only_generation_uses_port_and_tokenizer();
    test_null_prompt_returns_empty_output();
    test_vision_generation_uses_vision_port();
    test_vision_prepare_failure_falls_back_to_text();

    if (g_failures != 0)
    {
        std::cerr << g_failures << " test(s) failed\n";
        return 1;
    }
    return 0;
}
