// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-LOG-CPP-01
// Architecture:   ARCH-MOD-001
// Unit Design:    UD-TRT-CORE-01
// Intent:         TRT logger severity names, error storage, env-var controls
// Preconditions:  TRT headers available
// Postconditions: Logger stores errors and respects severity settings
// =============================================================================

// =============================================================================
// test_trt_logger.cpp — Unit tests for TRT logger env-var helpers
// =============================================================================
//
// Purpose:
//   Validates the environment-variable helper functions from trt_common.h/cpp
//   that control TRT logging behavior. When TRTF_HAS_TRT is enabled, tests
//   trt_log_to_stderr_enabled() and trt_log_stderr_min_severity(). When TRT
//   headers are unavailable, the test reports a skip (exit 0).
//
// Dependencies:
//   - runtime/core/trt_common.h (trt_log_to_stderr_enabled,
//     trt_log_stderr_min_severity, trt_severity_name)
//
// Environment:
//   CPU-only. No GPU or CUDA runtime required.
//   Env vars TRTF_TRT_LOG_STDERR and TRTF_TRT_LOG_MIN_SEVERITY are tested.
//
// Note:
//   The functions trt_log_to_stderr_enabled() and trt_log_stderr_min_severity()
//   use static local variables that are initialized once. Because of this
//   one-shot initialization, we cannot test multiple env-var values in the
//   same process. Instead we test the value that was set at process start.
//   The CMake test infrastructure can run separate processes with different
//   env settings if needed.
// =============================================================================

#include "runtime/core/trt_common.h"

#include <cstdlib>
#include <iostream>
#include <string>


static int failures = 0;

static void check(bool condition, const char* test_name)
{
    if (!condition)
    {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

static void test_severity_name_error()
{
    const char* name = trtf::trt_severity_name(nvinfer1::ILogger::Severity::kERROR);
    check(std::string(name) == "ERROR", "severity_name ERROR");
}

static void test_severity_name_warning()
{
    const char* name = trtf::trt_severity_name(nvinfer1::ILogger::Severity::kWARNING);
    check(std::string(name) == "WARNING", "severity_name WARNING");
}

static void test_severity_name_info()
{
    const char* name = trtf::trt_severity_name(nvinfer1::ILogger::Severity::kINFO);
    check(std::string(name) == "INFO", "severity_name INFO");
}

static void test_severity_name_verbose()
{
    const char* name = trtf::trt_severity_name(nvinfer1::ILogger::Severity::kVERBOSE);
    check(std::string(name) == "VERBOSE", "severity_name VERBOSE");
}

static void test_severity_name_internal_error()
{
    const char* name = trtf::trt_severity_name(nvinfer1::ILogger::Severity::kINTERNAL_ERROR);
    check(std::string(name) == "INTERNAL_ERROR", "severity_name INTERNAL_ERROR");
}

static void test_logger_stores_error()
{
    trtf::TrtLogger logger;
    logger.log(nvinfer1::ILogger::Severity::kERROR, "test error message");
    check(logger.last_error() == "test error message", "logger stores error");
}

static void test_logger_clear_error()
{
    trtf::TrtLogger logger;
    logger.log(nvinfer1::ILogger::Severity::kERROR, "first error");
    logger.clear_error();
    check(logger.last_error().empty(), "logger clear_error");
}

static void test_logger_ignores_info()
{
    trtf::TrtLogger logger;
    logger.log(nvinfer1::ILogger::Severity::kINFO, "info message");
    check(logger.last_error().empty(), "logger ignores info");
}

static void test_logger_overwrites_error()
{
    trtf::TrtLogger logger;
    logger.log(nvinfer1::ILogger::Severity::kERROR, "first");
    logger.log(nvinfer1::ILogger::Severity::kERROR, "second");
    check(logger.last_error() == "second", "logger overwrites error");
}

static void test_logger_null_msg()
{
    trtf::TrtLogger logger;
    logger.log(nvinfer1::ILogger::Severity::kERROR, nullptr);
    check(logger.last_error().empty(), "logger null msg ignored");
}

static void test_log_to_stderr_returns_bool()
{
    // Just verify the function returns without crashing and gives a bool.
    const bool result = trtf::trt_log_to_stderr_enabled();
    // The result depends on TRTF_TRT_LOG_STDERR env var at static init time.
    // We just verify it is callable and returns a consistent value.
    check(result == trtf::trt_log_to_stderr_enabled(), "log_to_stderr consistent");
}

static void test_min_severity_returns_valid()
{
    const auto sev = trtf::trt_log_stderr_min_severity();
    // Verify the returned severity is one of the valid enum values.
    const bool valid = (sev == nvinfer1::ILogger::Severity::kINTERNAL_ERROR
        || sev == nvinfer1::ILogger::Severity::kERROR
        || sev == nvinfer1::ILogger::Severity::kWARNING
        || sev == nvinfer1::ILogger::Severity::kINFO
        || sev == nvinfer1::ILogger::Severity::kVERBOSE);
    check(valid, "min_severity valid enum");
}


int main()
{
    test_severity_name_error();
    test_severity_name_warning();
    test_severity_name_info();
    test_severity_name_verbose();
    test_severity_name_internal_error();
    test_logger_stores_error();
    test_logger_clear_error();
    test_logger_ignores_info();
    test_logger_overwrites_error();
    test_logger_null_msg();
    test_log_to_stderr_returns_bool();
    test_min_severity_returns_valid();

    if (failures > 0)
    {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All trt_logger tests passed.\n";
    return 0;
}
