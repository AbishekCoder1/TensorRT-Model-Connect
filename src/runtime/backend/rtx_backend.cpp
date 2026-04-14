// RtxBackend: IBackend implementation for TensorRT-RTX.
// Compiled into libtrtf_backend_rtx.so. Links libtensorrt_rtx.so.
//
// Uses the RTX-specific NvInfer.h headers which declare IRuntimeCache,
// CudaGraphStrategy, and DynamicShapesKernelSpecializationStrategy.

#include "trtf/runtime/trt_backend.h"
#include "trt_module_impl.h"
#include "trt_logger.h"
#include "runtime/core/cuda_common.h"

#include <NvInfer.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace trtf {

namespace {

struct StreamSetup {
    cudaStream_t stream{nullptr};
    std::shared_ptr<void> owner;
};

StreamSetup resolve_stream(cudaStream_t requested_stream)
{
    if (requested_stream) {
        return StreamSetup{requested_stream, {}};
    }

    auto owned = std::make_shared<CudaStream>();
    if (!owned->ok()) {
        throw std::runtime_error("[trtf] Failed to create CUDA stream");
    }

    return StreamSetup{owned->get(), owned};
}

} // namespace

class RtxBackend final : public IBackend {
public:
    RtxBackend() : runtime_(create_trt_runtime()) {
        if (!runtime_)
            throw std::runtime_error("[trtf] Failed to create TRT-RTX runtime");
    }

    ~RtxBackend() override {
        flush_runtime_cache();
        delete runtime_cache_;
    }

    std::unique_ptr<ITrtModule> create_module(
        const void* plan_data, size_t plan_size,
        const ModuleCreateOptions& options) override
    {
        auto* engine = runtime_->deserializeCudaEngine(plan_data, plan_size);
        if (!engine)
            throw std::runtime_error("[trtf] Failed to deserialize engine (RTX)");

        // Create IRuntimeConfig with RTX-specific features
        auto* rt_config = engine->createRuntimeConfig();
        if (!rt_config) {
            delete engine;
            throw std::runtime_error("[trtf] Failed to create RTX runtime config");
        }

        // JIT kernel cache
        if (options.runtime_cache_path && options.runtime_cache_path[0] != '\0') {
            ensure_runtime_cache(rt_config, options.runtime_cache_path);
        }

        // CUDA graph capture
        if (options.cuda_graphs) {
            rt_config->setCudaGraphStrategy(
                nvinfer1::CudaGraphStrategy::kWHOLE_GRAPH_CAPTURE);
            std::cerr << "[trtf] CUDA graphs enabled (whole-graph capture)\n";
        }

        auto* ctx = engine->createExecutionContext(rt_config);
        delete rt_config;
        if (!ctx) {
            delete engine;
            throw std::runtime_error("[trtf] Failed to create RTX execution context");
        }

        StreamSetup stream_setup;
        try {
            stream_setup = resolve_stream(options.stream);
        } catch (...) {
            delete ctx;
            delete engine;
            throw;
        }

        auto module = std::make_unique<TrtModuleImpl>(engine, ctx, stream_setup.stream);
        if (!module->ok()) {
            delete engine;
            throw std::runtime_error("[trtf] TrtModuleImpl creation failed (RTX)");
        }

        module->keep_alive(std::shared_ptr<nvinfer1::ICudaEngine>(
            engine, [](nvinfer1::ICudaEngine* p) { delete p; }));
        if (stream_setup.owner)
            module->keep_alive(stream_setup.owner);

        return module;
    }

    const char* name() const override { return "trt_rtx"; }

private:
    TrtUniquePtr<nvinfer1::IRuntime> runtime_;
    nvinfer1::IRuntimeCache* runtime_cache_{nullptr};
    std::string cache_path_;

    void ensure_runtime_cache(nvinfer1::IRuntimeConfig* cfg, const char* path) {
        if (!runtime_cache_) {
            runtime_cache_ = cfg->createRuntimeCache();
            cache_path_ = path;
            std::ifstream ifs(path, std::ios::binary | std::ios::ate);
            if (ifs) {
                auto sz = ifs.tellg();
                if (sz > 0) {
                    std::vector<char> buf(static_cast<size_t>(sz));
                    ifs.seekg(0);
                    ifs.read(buf.data(), sz);
                    runtime_cache_->deserialize(buf.data(), buf.size());
                    std::cerr << "[trtf] RTX runtime cache loaded: "
                              << path << " (" << sz << " bytes)\n";
                }
            }
        }
        cfg->setRuntimeCache(*runtime_cache_);
    }

    void flush_runtime_cache() {
        if (!runtime_cache_ || cache_path_.empty()) return;
        auto* mem = runtime_cache_->serialize();
        if (mem && mem->size() > 0) {
            std::ofstream ofs(cache_path_, std::ios::binary | std::ios::trunc);
            if (ofs) {
                ofs.write(static_cast<const char*>(mem->data()),
                          static_cast<std::streamsize>(mem->size()));
                std::cerr << "[trtf] RTX runtime cache saved: "
                          << cache_path_ << " (" << mem->size() << " bytes)\n";
            }
            delete mem;
        }
    }
};

} // namespace trtf

extern "C" trtf::IBackend* trtf_create_backend()
{
    try { return new trtf::RtxBackend(); }
    catch (const std::exception& e) {
        std::cerr << "[trtf] RTX backend init failed: " << e.what() << std::endl;
        return nullptr;
    }
}

extern "C" void trtf_destroy_backend(trtf::IBackend* b) { delete b; }
