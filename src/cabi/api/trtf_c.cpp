#include "trtf/pipeline.h"
#include "trtf/bundle.h"
#include "trtf/runtime/pipeline_factory.h"

#include <chrono>
#include <exception>
#include <iostream>
#include <string>

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

} // namespace

extern "C" {

trtf::IPipeline* trtf_create_pipeline(const char* bundle_path, int flags)
{
    (void) flags;
    TrtfPipelineOptions opts{};
    opts.max_new_tokens = 0;
    opts.hf_python = nullptr;
    opts.image_path = nullptr;
    return trtf_create_pipeline_ex(bundle_path, &opts);
}

trtf::IPipeline* trtf_create_pipeline_ex(const char* bundle_path, const TrtfPipelineOptions* options)
{
    clear_last_error();

    if (bundle_path == nullptr || bundle_path[0] == '\0')
    {
        set_last_error("bundle_path must not be null or empty");
        return nullptr;
    }

    try
    {
        const std::string path(bundle_path);
        if (!trtf::IsBundle(path))
        {
            set_last_error("Not a valid .trtfb bundle: " + path);
            return nullptr;
        }

        std::string hf_python;
        if (options != nullptr && options->hf_python != nullptr)
            hf_python = options->hf_python;

        auto t0 = std::chrono::steady_clock::now();

        auto pipeline = trtf::PipelineFactory::from_bundle(path, hf_python);

        auto t1 = std::chrono::steady_clock::now();
        std::cerr << "[trtf] Runtime ready (strategy="
                  << pipeline->pipeline_type() << ") ["
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
                  << " ms]" << std::endl;

        return pipeline.release();
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
