#pragma once

// Legacy Pipeline class -- retained for backward compatibility with existing examples.
// New code should use IPipeline + trtf_create_pipeline() from <trtf/pipeline.h>.

#include "trtf/backend.h"
#include "trtf/generation.h"
#include "trtf/tokenizer.h"

#include <memory>
#include <string>
#include <vector>

namespace trtf {

class Pipeline {
public:
    static Pipeline CreateTextGeneration(
        const std::string& model_id, bool prefer_trt = true, bool force_trt = false);
    static Pipeline LoadModel(const std::string& model_id, bool prefer_trt = true, bool force_trt = false);

    std::vector<GenerationResult> operator()(const std::string& prompt) const;
    std::string generate(const std::string& prompt) const;

    const std::string& task() const;
    const std::string& model_id() const;
    const std::string& backend_name() const;

    // Accessors for internals (used by IPipeline impl and bundle save).
    ITokenizer* tokenizer() const { return mTokenizer.get(); }
    IGenerationBackend* backend() const { return mBackend.get(); }
    const GenerationConfig& generation_config() const { return mGenerationConfig; }

private:
    Pipeline(std::string task, std::string model_id, std::unique_ptr<ITokenizer> tokenizer,
        std::unique_ptr<IGenerationBackend> backend, std::string backend_name, GenerationConfig generation_config);

    std::string mTask;
    std::string mModelId;
    std::unique_ptr<ITokenizer> mTokenizer;
    std::unique_ptr<IGenerationBackend> mBackend;
    std::string mBackendName;
    GenerationConfig mGenerationConfig;
};

Pipeline loadModel(const std::string& model_id, bool prefer_trt = true, bool force_trt = false);

} // namespace trtf
