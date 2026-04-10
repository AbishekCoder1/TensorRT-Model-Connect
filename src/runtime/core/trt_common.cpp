// trt_common.cpp — TensorRT logger/runtime helpers plus CUDA graph support.
// CudaStream/CudaBuffer moved to cuda_common.cpp.

#include "runtime/core/trt_common.h"

#include <algorithm>

namespace trtf {

#if TRTF_HAS_TRT

const char* trt_severity_name(nvinfer1::ILogger::Severity severity) {
    switch (severity) {
    case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
        return "INTERNAL_ERROR";
    case nvinfer1::ILogger::Severity::kERROR:
        return "ERROR";
    case nvinfer1::ILogger::Severity::kWARNING:
        return "WARNING";
    case nvinfer1::ILogger::Severity::kINFO:
        return "INFO";
    case nvinfer1::ILogger::Severity::kVERBOSE:
        return "VERBOSE";
    default:
        return "UNKNOWN";
    }
}

bool trt_log_to_stderr_enabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("TRTF_TRT_LOG_STDERR");
        if (env == nullptr || env[0] == '\0') {
            return false;
        }
        return std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

nvinfer1::ILogger::Severity trt_log_stderr_min_severity() {
    static const nvinfer1::ILogger::Severity severity = [] {
        const char* env = std::getenv("TRTF_TRT_LOG_MIN_SEVERITY");
        if (env == nullptr || env[0] == '\0') {
            return nvinfer1::ILogger::Severity::kINFO;
        }

        std::string value(env);
        std::transform(value.begin(), value.end(), value.begin(),
                       [](unsigned char c) { return static_cast<char>(std::toupper(c)); });

        if (value == "INTERNAL_ERROR") {
            return nvinfer1::ILogger::Severity::kINTERNAL_ERROR;
        }
        if (value == "ERROR") {
            return nvinfer1::ILogger::Severity::kERROR;
        }
        if (value == "WARNING") {
            return nvinfer1::ILogger::Severity::kWARNING;
        }
        if (value == "VERBOSE") {
            return nvinfer1::ILogger::Severity::kVERBOSE;
        }
        return nvinfer1::ILogger::Severity::kINFO;
    }();
    return severity;
}

void TrtLogger::log(Severity severity, const char* msg) noexcept {
    if (severity <= Severity::kERROR && msg != nullptr) {
        mLastError = msg;
    }

    if (msg == nullptr) {
        return;
    }

    if (trt_log_to_stderr_enabled() && severity <= trt_log_stderr_min_severity()) {
        std::cerr << "TRT_LOG[" << trt_severity_name(severity) << "] " << msg << '\n';
    } else if (severity <= Severity::kWARNING) {
        std::cerr << "[trt] " << trt_severity_name(severity) << ": " << msg << '\n';
    }
}

const std::string& TrtLogger::last_error() const {
    return mLastError;
}

void TrtLogger::clear_error() {
    mLastError.clear();
}

TrtUniquePtr<nvinfer1::IRuntime> create_trt_runtime() {
    static TrtLogger logger;
    return TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
}

CudaGraphExec::~CudaGraphExec() {
    reset();
}

CudaGraphExec::CudaGraphExec(CudaGraphExec&& other) noexcept
    : graph_(other.graph_), exec_(other.exec_) {
    other.graph_ = nullptr;
    other.exec_ = nullptr;
}

CudaGraphExec& CudaGraphExec::operator=(CudaGraphExec&& other) noexcept {
    if (this != &other) {
        reset();
        graph_ = other.graph_;
        exec_ = other.exec_;
        other.graph_ = nullptr;
        other.exec_ = nullptr;
    }
    return *this;
}

bool CudaGraphExec::begin_capture(cudaStream_t stream) {
    reset();
    auto err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    if (err != cudaSuccess) {
        std::cerr << "[cuda_graph] begin_capture failed: " << cudaGetErrorString(err) << '\n';
        return false;
    }
    return true;
}

bool CudaGraphExec::end_capture(cudaStream_t stream) {
    auto err = cudaStreamEndCapture(stream, &graph_);
    if (err != cudaSuccess || graph_ == nullptr) {
        std::cerr << "[cuda_graph] end_capture failed: "
                  << (err != cudaSuccess ? cudaGetErrorString(err) : "null graph") << '\n';
        graph_ = nullptr;
        return false;
    }

    err = cudaGraphInstantiate(&exec_, graph_, 0);
    if (err != cudaSuccess) {
        std::cerr << "[cuda_graph] instantiate failed: " << cudaGetErrorString(err) << '\n';
        cudaGraphDestroy(graph_);
        graph_ = nullptr;
        return false;
    }
    return true;
}

bool CudaGraphExec::launch(cudaStream_t stream) const {
    if (exec_ == nullptr) {
        return false;
    }
    return cudaGraphLaunch(exec_, stream) == cudaSuccess;
}

bool CudaGraphExec::ready() const {
    return exec_ != nullptr;
}

void CudaGraphExec::reset() {
    if (exec_ != nullptr) {
        cudaGraphExecDestroy(exec_);
        exec_ = nullptr;
    }
    if (graph_ != nullptr) {
        cudaGraphDestroy(graph_);
        graph_ = nullptr;
    }
}

#endif // TRTF_HAS_TRT

} // namespace trtf
