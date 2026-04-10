#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>

#if TRTF_HAS_TRT
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#endif

namespace trtf {

#if TRTF_HAS_TRT

const char* trt_severity_name(nvinfer1::ILogger::Severity severity);
bool trt_log_to_stderr_enabled();
nvinfer1::ILogger::Severity trt_log_stderr_min_severity();

class TrtLogger final : public nvinfer1::ILogger {
  public:
    void log(Severity severity, const char* msg) noexcept override;
    const std::string& last_error() const;
    void clear_error();

  private:
    std::string mLastError;
};

template <typename T>
struct TrtDeleter {
    void operator()(T* ptr) const noexcept {
        if (ptr == nullptr) {
            return;
        }
#if NV_TENSORRT_MAJOR >= 10
        delete ptr;
#else
        ptr->destroy();
#endif
    }
};

template <typename T>
using TrtUniquePtr = std::unique_ptr<T, TrtDeleter<T>>;

// TensorRT runtime factory with a process-lifetime logger.
// TensorRT keeps a reference to ILogger, so logger lifetime must outlive
// all runtime/engine/context objects created from it.
TrtUniquePtr<nvinfer1::IRuntime> create_trt_runtime();

class CudaStream final {
  public:
    CudaStream();
    ~CudaStream();

    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    CudaStream(CudaStream&& other) noexcept;
    CudaStream& operator=(CudaStream&& other) noexcept;

    bool ok() const;
    cudaStream_t get() const;

  private:
    cudaStream_t mStream{nullptr};
    cudaError_t mStatus{cudaSuccess};
};

class CudaBuffer final {
  public:
    explicit CudaBuffer(std::size_t bytes);
    ~CudaBuffer();

    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    CudaBuffer(CudaBuffer&& other) noexcept;
    CudaBuffer& operator=(CudaBuffer&& other) noexcept;

    bool ok() const;
    void* data() const;
    std::size_t size() const;

  private:
    void* mPtr{nullptr};
    std::size_t mBytes{0};
    cudaError_t mStatus{cudaSuccess};
};

// RAII wrapper for CUDA graph + executable graph.
// Captures a stream region and replays it without per-kernel launch overhead.
class CudaGraphExec final {
  public:
    CudaGraphExec() = default;
    ~CudaGraphExec();

    CudaGraphExec(const CudaGraphExec&) = delete;
    CudaGraphExec& operator=(const CudaGraphExec&) = delete;
    CudaGraphExec(CudaGraphExec&& other) noexcept;
    CudaGraphExec& operator=(CudaGraphExec&& other) noexcept;

    // Begin capturing on the given stream. All subsequent CUDA operations
    // on this stream will be recorded into the graph until end_capture().
    bool begin_capture(cudaStream_t stream);

    // End capture, instantiate the executable graph. Returns true on success.
    bool end_capture(cudaStream_t stream);

    // Launch the captured graph on the given stream.
    bool launch(cudaStream_t stream) const;

    // True if a graph has been successfully captured and instantiated.
    bool ready() const;

    // Reset — destroy captured graph and executable.
    void reset();

  private:
    cudaGraph_t graph_{nullptr};
    cudaGraphExec_t exec_{nullptr};
};

#endif // TRTF_HAS_TRT

} // namespace trtf
