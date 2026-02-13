#include "trtf/pipeline.h"
#include "trtf/model_resolver.h"
#include "trtf/runtime_factory.h"

#include <cstdlib>
#include <stdexcept>
#include <utility>

namespace trtf {
namespace {

std::size_t resolve_max_new_tokens(std::size_t fallback)
{
    const char* env = std::getenv("TRTF_MAX_NEW_TOKENS");
    if (env == nullptr || env[0] == '\0')
    {
        return fallback;
    }

    try
    {
        const long long parsed = std::stoll(env);
        if (parsed > 0)
        {
            return static_cast<std::size_t>(parsed);
        }
    }
    catch (const std::exception&)
    {
    }

    return fallback;
}

} // namespace

Pipeline::Pipeline(std::string task, std::string model_id, std::unique_ptr<ITokenizer> tokenizer,
    std::unique_ptr<IGenerationBackend> backend, std::string backend_name, GenerationConfig generation_config)
    : mTask(std::move(task))
    , mModelId(std::move(model_id))
    , mTokenizer(std::move(tokenizer))
    , mBackend(std::move(backend))
    , mBackendName(std::move(backend_name))
    , mGenerationConfig(generation_config)
{
}

Pipeline Pipeline::CreateTextGeneration(const std::string& model_id, bool prefer_trt, bool force_trt)
{
    const ResolvedModelSpec model_spec = ResolveTextGenerationModel(model_id);
    BackendSelection selection;
    selection.prefer_trt = prefer_trt;
    selection.force_trt = force_trt;

    RuntimeAssembly runtime = BuildRuntimeForTextGeneration(model_spec, selection);

    GenerationConfig generation_config{};
    generation_config.max_new_tokens = resolve_max_new_tokens(generation_config.max_new_tokens);

    return Pipeline("text-generation", model_id, std::move(runtime.tokenizer), std::move(runtime.backend),
        std::move(runtime.backend_name), generation_config);
}

Pipeline Pipeline::LoadModel(const std::string& model_id, bool prefer_trt, bool force_trt)
{
    return CreateTextGeneration(model_id, prefer_trt, force_trt);
}

std::vector<GenerationResult> Pipeline::operator()(const std::string& prompt) const
{
    if (!mBackend)
    {
        throw std::runtime_error("Pipeline is not initialized.");
    }

    if (mBackend->supports_text_generation())
    {
        return {GenerationResult{mBackend->generate_text(prompt, mGenerationConfig)}};
    }

    if (!mTokenizer)
    {
        throw std::runtime_error("Tokenizer is not initialized.");
    }

    auto input_ids = mTokenizer->encode(prompt);
    auto output_ids = mBackend->generate(input_ids, mGenerationConfig);

    return {GenerationResult{mTokenizer->decode(output_ids)}};
}

std::string Pipeline::generate(const std::string& prompt) const
{
    const std::vector<GenerationResult> out = (*this)(prompt);
    if (out.empty())
    {
        return "";
    }
    return out.front().generated_text;
}

const std::string& Pipeline::task() const
{
    return mTask;
}

const std::string& Pipeline::model_id() const
{
    return mModelId;
}

const std::string& Pipeline::backend_name() const
{
    return mBackendName;
}

Pipeline loadModel(const std::string& model_id, bool prefer_trt, bool force_trt)
{
    return Pipeline::LoadModel(model_id, prefer_trt, force_trt);
}

} // namespace trtf
