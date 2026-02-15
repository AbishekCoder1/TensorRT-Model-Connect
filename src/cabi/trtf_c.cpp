#include "trtf/pipeline.h"
#include "trtf/bundle.h"
#include "trtf/pipeline_legacy.h"
#include "trtf/model_resolver.h"
#include "trtf/runtime_factory.h"
#include "bundle/bundle_format.h"

#include <chrono>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include <utility>

#ifndef TRTF_VERSION_STRING
#define TRTF_VERSION_STRING "0.1.0"
#endif

namespace {

thread_local std::string g_last_error;

void set_last_error(const std::string& msg)
{
    g_last_error = msg;
}

void clear_last_error()
{
    g_last_error.clear();
}

class PipelineImpl final : public trtf::IPipeline {
public:
    PipelineImpl(std::string model_id, std::unique_ptr<trtf::ITokenizer> tokenizer,
        std::unique_ptr<trtf::IGenerationBackend> backend, std::string backend_name,
        trtf::GenerationConfig gen_config)
        : mModelId(std::move(model_id))
        , mTokenizer(std::move(tokenizer))
        , mBackend(std::move(backend))
        , mBackendName(std::move(backend_name))
        , mGenConfig(gen_config)
    {
    }

    const char* generate(const char* prompt, std::size_t max_new_tokens) override
    {
        if (prompt == nullptr)
        {
            mLastOutput = "";
            return mLastOutput.c_str();
        }

        trtf::GenerationConfig config = mGenConfig;
        if (max_new_tokens > 0)
        {
            config.max_new_tokens = max_new_tokens;
        }

        if (mBackend->supports_text_generation())
        {
            mLastOutput = mBackend->generate_text(prompt, config);
            return mLastOutput.c_str();
        }

        auto input_ids = mTokenizer->encode(prompt);
        auto output_ids = mBackend->generate(input_ids, config);
        mLastOutput = mTokenizer->decode(output_ids);
        return mLastOutput.c_str();
    }

    const char* model_id() const override
    {
        return mModelId.c_str();
    }

    const char* backend_name() const override
    {
        return mBackendName.c_str();
    }

    bool save_bundle(const char* output_path) override
    {
        // Bundle save is implemented in Phase 3 when TRT engine serialization is ready.
        // For now, return false.
        (void) output_path;
        return false;
    }

private:
    std::string mModelId;
    std::unique_ptr<trtf::ITokenizer> mTokenizer;
    std::unique_ptr<trtf::IGenerationBackend> mBackend;
    std::string mBackendName;
    trtf::GenerationConfig mGenConfig;
    std::string mLastOutput;
};

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

extern "C" {

trtf::IPipeline* trtf_create_pipeline(const char* model_or_bundle, int flags)
{
    clear_last_error();

    if (model_or_bundle == nullptr || model_or_bundle[0] == '\0')
    {
        set_last_error("model_or_bundle must not be null or empty");
        return nullptr;
    }

    try
    {
        const std::string input(model_or_bundle);

        // Check if input is a .trtfb bundle
        if (trtf::IsBundle(input))
        {
            // Bundle loading will be implemented in Phase 3
            set_last_error("Bundle loading not yet implemented");
            return nullptr;
        }

        // Normal model loading path
        bool prefer_trt = true;
        bool force_trt = false;
        switch (flags)
        {
        case TRTF_FORCE_TRT:
            force_trt = true;
            break;
        case TRTF_CPU_ONLY:
            prefer_trt = false;
            break;
        case TRTF_PREFER_TRT:
        default:
            break;
        }

        auto t0 = std::chrono::steady_clock::now();

        std::cerr << "[trtf] Resolving model: " << input << " ..." << std::endl;
        const trtf::ResolvedModelSpec model_spec = trtf::ResolveTextGenerationModel(input);
        auto t1 = std::chrono::steady_clock::now();
        std::cerr << "[trtf] Model resolved (kind="
                  << (model_spec.kind == trtf::ResolvedModelKind::kDecoderDefinition ? "decoder-definition"
                      : model_spec.kind == trtf::ResolvedModelKind::kHuggingFaceLocal ? "hf-local" : "custom")
                  << ") ["
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " ms]"
                  << std::endl;

        trtf::BackendSelection selection;
        selection.prefer_trt = prefer_trt;
        selection.force_trt = force_trt;

        std::cerr << "[trtf] Building runtime (backend="
                  << (force_trt ? "force-trt" : prefer_trt ? "prefer-trt" : "cpu-only")
                  << ") ..." << std::endl;
        trtf::RuntimeAssembly runtime = trtf::BuildRuntimeForTextGeneration(model_spec, selection);
        auto t2 = std::chrono::steady_clock::now();
        std::cerr << "[trtf] Runtime ready (backend=" << runtime.backend_name
                  << ") [" << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms]"
                  << std::endl;

        trtf::GenerationConfig gen_config{};
        gen_config.max_new_tokens = resolve_max_new_tokens(gen_config.max_new_tokens);

        return new PipelineImpl(
            input,
            std::move(runtime.tokenizer),
            std::move(runtime.backend),
            std::move(runtime.backend_name),
            gen_config);
    }
    catch (const std::exception& e)
    {
        set_last_error(e.what());
        return nullptr;
    }
    catch (...)
    {
        set_last_error("Unknown error creating pipeline");
        return nullptr;
    }
}

const char* trtf_last_error(void)
{
    return g_last_error.c_str();
}

const char* trtf_version(void)
{
    return TRTF_VERSION_STRING;
}

int trtf_has_trt(void)
{
#if TRTF_HAS_TRT
    return 1;
#else
    return 0;
#endif
}

} // extern "C"
