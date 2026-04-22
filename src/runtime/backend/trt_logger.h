#pragma once

// TRT logger and runtime factory — compiled into backend DSOs only.

#include <NvInfer.h>
#include <memory>
#include <string>

namespace trtf {

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
        if (ptr)
            delete ptr;
    }
};

template <typename T>
using TrtUniquePtr = std::unique_ptr<T, TrtDeleter<T>>;

// TensorRT runtime factory with a process-lifetime logger.
// TensorRT keeps a reference to ILogger, so logger lifetime must outlive
// all runtime/engine/context objects created from it.
TrtUniquePtr<nvinfer1::IRuntime> create_trt_runtime();

} // namespace trtf
