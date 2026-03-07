#pragma once

#include "trtf/runtime/contracts/services.h"
#include "trtf/generation.h"
#include "trtf/tokenizer.h"
#include "runtime/services/common/runtime_service_ports.h"
#include "runtime/services/common/scoped_temp_dir.h"

#include <cstddef>
#include <memory>
#include <string>

namespace trtf::runtime::services::text {

class GenerationTextService final : public trtf::runtime::ITextService {
public:
    GenerationTextService(
        std::shared_ptr<trtf::ITokenizer> tokenizer,
        std::unique_ptr<common::ITextGenerationPort> backend,
        std::shared_ptr<common::ScopedTempDirOwner> tokenizer_temp_dir,
        trtf::GenerationConfig generation_config = {});

    const char* generate(const char* prompt, std::size_t max_new_tokens) override;
    const char* generate(
        const char* prompt,
        const adapters::io::DecodedImage& image,
        std::size_t max_new_tokens) override;
    bool supports_vision() const override;

private:
    trtf::GenerationConfig resolve_generation_config(std::size_t max_new_tokens) const;
    const char* clear_and_return_empty_output();
    const char* generate_text_only(const char* prompt, const trtf::GenerationConfig& config);

    std::shared_ptr<common::ScopedTempDirOwner> mTokenizerTempDir;
    std::shared_ptr<trtf::ITokenizer> mTokenizer;
    std::unique_ptr<common::ITextGenerationPort> mBackend;
    trtf::GenerationConfig mGenerationConfig;
    std::string mLastOutput;
};

} // namespace trtf::runtime::services::text
