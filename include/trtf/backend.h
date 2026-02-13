#pragma once

#include "trtf/generation.h"
#include "trtf/model.h"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace trtf {

class ITokenizer;

class IGenerationBackend {
public:
    virtual ~IGenerationBackend() = default;

    virtual bool is_available() const = 0;
    virtual const char* name() const = 0;

    virtual const char* unavailable_reason() const
    {
        return "";
    }

    virtual std::vector<int32_t> generate(
        const std::vector<int32_t>& input_ids,
        const GenerationConfig& config) = 0;

    virtual bool supports_text_generation() const
    {
        return false;
    }

    virtual std::string generate_text(const std::string& prompt, const GenerationConfig& config)
    {
        (void) prompt;
        (void) config;
        throw std::runtime_error("Backend does not support direct text generation.");
    }
};

std::unique_ptr<IGenerationBackend> CreateCpuReferenceBackend(const ITokenizer& tokenizer, const DecoderModel& model);
std::unique_ptr<IGenerationBackend> CreateTrtBackend(const ITokenizer& tokenizer, const DecoderModel& model);
std::unique_ptr<IGenerationBackend> CreateHfPythonBackend(const std::string& model_dir);

} // namespace trtf
