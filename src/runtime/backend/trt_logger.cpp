#include "trt_logger.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>

namespace trtf {

const char* trt_severity_name(nvinfer1::ILogger::Severity severity)
{
    switch (severity)
    {
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

bool trt_log_to_stderr_enabled()
{
    static const bool enabled = [] {
        const char* env = std::getenv("TRTF_TRT_LOG_STDERR");
        if (env == nullptr || env[0] == '\0')
        {
            return false;
        }
        return std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

nvinfer1::ILogger::Severity trt_log_stderr_min_severity()
{
    static const nvinfer1::ILogger::Severity severity = [] {
        const char* env = std::getenv("TRTF_TRT_LOG_MIN_SEVERITY");
        if (env == nullptr || env[0] == '\0')
        {
            return nvinfer1::ILogger::Severity::kINFO;
        }

        std::string value(env);
        std::transform(value.begin(), value.end(), value.begin(),
            [](unsigned char c) { return static_cast<char>(std::toupper(c)); });

        if (value == "INTERNAL_ERROR")
        {
            return nvinfer1::ILogger::Severity::kINTERNAL_ERROR;
        }
        if (value == "ERROR")
        {
            return nvinfer1::ILogger::Severity::kERROR;
        }
        if (value == "WARNING")
        {
            return nvinfer1::ILogger::Severity::kWARNING;
        }
        if (value == "VERBOSE")
        {
            return nvinfer1::ILogger::Severity::kVERBOSE;
        }
        return nvinfer1::ILogger::Severity::kINFO;
    }();
    return severity;
}

void TrtLogger::log(Severity severity, const char* msg) noexcept
{
    if (severity <= Severity::kERROR && msg != nullptr)
    {
        mLastError = msg;
    }

    if (msg == nullptr)
    {
        return;
    }

    if (trt_log_to_stderr_enabled() && severity <= trt_log_stderr_min_severity())
    {
        std::cerr << "TRT_LOG[" << trt_severity_name(severity) << "] " << msg << '\n';
    }
    else if (severity <= Severity::kWARNING)
    {
        // Always show warnings and errors even without TRTF_TRT_LOG_STDERR
        std::cerr << "[trt] " << trt_severity_name(severity) << ": " << msg << '\n';
    }
}

const std::string& TrtLogger::last_error() const
{
    return mLastError;
}

void TrtLogger::clear_error()
{
    mLastError.clear();
}

TrtUniquePtr<nvinfer1::IRuntime> create_trt_runtime()
{
    // Keep logger alive for entire process lifetime because TensorRT runtime
    // stores ILogger by reference.
    static TrtLogger logger;
    return TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
}

} // namespace trtf
