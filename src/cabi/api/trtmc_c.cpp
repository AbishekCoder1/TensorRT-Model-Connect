#include "trtmc/bundle.h"
#include "trtmc/pipeline.h"
#include "trtmc/runtime/pipeline_factory.h"

#include <chrono>
#include <exception>
#include <iostream>
#include <string>

#ifndef TRTMC_VERSION_STRING
#define TRTMC_VERSION_STRING "0.1.0"
#endif

namespace {

thread_local std::string g_last_error;

struct PipelineCreateArgs {
    std::string hf_python;
    std::string runtime_cache;
    bool cuda_graphs{false};
};

void set_last_error(const std::string& msg) {
    g_last_error = msg;
}

void clear_last_error() {
    g_last_error.clear();
}

PipelineCreateArgs parse_pipeline_options(const TrtmcPipelineOptions* options) {
    PipelineCreateArgs args;
    if (options != nullptr && options->hf_python != nullptr)
        args.hf_python = options->hf_python;
    if (options != nullptr && options->runtime_cache != nullptr)
        args.runtime_cache = options->runtime_cache;
    args.cuda_graphs = (options != nullptr && options->cuda_graphs != 0);
    return args;
}

} // namespace

extern "C" {

trtmc::IPipeline* trtmc_create_pipeline(const char* bundle_path, int flags) {
    (void)flags;
    TrtmcPipelineOptions opts{};
    opts.max_new_tokens = 0;
    opts.hf_python = nullptr;
    opts.image_path = nullptr;
    opts.runtime_cache = nullptr;
    opts.cuda_graphs = 0;
    return trtmc_create_pipeline_ex(bundle_path, &opts);
}

trtmc::IPipeline* trtmc_create_pipeline_ex(const char* bundle_path,
                                         const TrtmcPipelineOptions* options) {
    clear_last_error();

    if (bundle_path == nullptr || bundle_path[0] == '\0') {
        set_last_error("bundle_path must not be null or empty");
        return nullptr;
    }

    try {
        const std::string path(bundle_path);
        if (!trtmc::IsBundle(path)) {
            set_last_error("Not a valid .trtfb bundle: " + path);
            return nullptr;
        }

        const PipelineCreateArgs args = parse_pipeline_options(options);

        auto t0 = std::chrono::steady_clock::now();

        auto pipeline = trtmc::PipelineFactory::from_bundle(path, args.hf_python, args.runtime_cache,
                                                           args.cuda_graphs);

        auto t1 = std::chrono::steady_clock::now();
        std::cerr << "[trtmc] Runtime ready (strategy=" << pipeline->pipeline_type() << ") ["
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
                  << " ms]" << std::endl;

        return pipeline.release();
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return nullptr;
    } catch (...) {
        set_last_error("Unknown error creating pipeline");
        return nullptr;
    }
}

const char* trtmc_last_error(void) {
    return g_last_error.c_str();
}

const char* trtmc_version(void) {
    return TRTMC_VERSION_STRING;
}

int trtmc_has_trt(void) {
    return 1;
}

} // extern "C"
