#pragma once

#include "trtf/runtime/ports/bundle_port.h"
#include "trtf/runtime/ports/trt_port.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace trtf::runtime::testhelpers {

inline int g_failures = 0;

inline void check(bool condition, const std::string& test_name)
{
    if (!condition)
    {
        std::cerr << "FAIL: " << test_name << '\n';
        ++g_failures;
    }
}

inline nvinfer1::IRuntime* fake_runtime_ptr()
{
    static std::uint8_t token = 0;
    return reinterpret_cast<nvinfer1::IRuntime*>(&token);
}

class FakeBundlePort final : public IBundlePort {
public:
    bool has_section(std::string_view section_name) const override
    {
        return sections.find(std::string(section_name)) != sections.end();
    }

    BundlePortResult<std::vector<char>> fetch_section_bytes(std::string_view section_name) const override
    {
        const auto it = sections.find(std::string(section_name));
        if (it == sections.end())
        {
            return BundlePortResult<std::vector<char>>::missing(
                "missing section: " + std::string(section_name));
        }
        if (it->second.empty())
        {
            return BundlePortResult<std::vector<char>>::invalid(
                "empty section: " + std::string(section_name));
        }
        return BundlePortResult<std::vector<char>>::success(it->second);
    }

    BundlePortResult<FastPathModelConfig> parse_fast_path_config(int32_t max_cache_length_override) const override
    {
        (void) max_cache_length_override;
        if (parse_status == BundlePortStatus::kOk)
        {
            return BundlePortResult<FastPathModelConfig>::success(config);
        }

        if (parse_status == BundlePortStatus::kMissingSection)
        {
            return BundlePortResult<FastPathModelConfig>::missing(parse_message);
        }

        return BundlePortResult<FastPathModelConfig>::invalid(parse_message);
    }

    std::map<std::string, std::vector<char>> sections;
    BundlePortStatus parse_status{BundlePortStatus::kOk};
    std::string parse_message{"parse failure"};
    FastPathModelConfig config{};
};

class FakeTrtPort final : public ITrtPort {
public:
    TrtPortResult<nvinfer1::ICudaEngine*> deserialize_engine(
        nvinfer1::IRuntime* runtime,
        const void* serialized_bytes,
        std::size_t serialized_size) const override
    {
        ++deserialize_calls;
        last_runtime = runtime;
        last_bytes = serialized_bytes;
        last_size = serialized_size;

        if (fail_deserialize_on_call > 0 && deserialize_calls == fail_deserialize_on_call)
        {
            return TrtPortResult<nvinfer1::ICudaEngine*>::failure(
                TrtPortStatus::kDeserializeFailed,
                "deserialize failed");
        }

        return TrtPortResult<nvinfer1::ICudaEngine*>::success(fake_engine());
    }

    TrtPortResult<nvinfer1::IExecutionContext*> create_execution_context(
        nvinfer1::ICudaEngine* engine) const override
    {
        ++create_context_calls;
        last_engine = engine;

        if (fail_context_on_call > 0 && create_context_calls == fail_context_on_call)
        {
            return TrtPortResult<nvinfer1::IExecutionContext*>::failure(
                TrtPortStatus::kContextCreationFailed,
                "context failed");
        }

        return TrtPortResult<nvinfer1::IExecutionContext*>::success(fake_context());
    }

    TrtPortResult<bool> has_io_tensor_named(
        const nvinfer1::ICudaEngine* engine,
        const std::string& tensor_name) const override
    {
        ++has_io_calls;
        last_engine = const_cast<nvinfer1::ICudaEngine*>(engine);
        last_tensor_name = tensor_name;

        if (metadata_failure)
        {
            return TrtPortResult<bool>::failure(
                TrtPortStatus::kMetadataFailed,
                "metadata failure");
        }

        return TrtPortResult<bool>::success(has_requested_tensor);
    }

    static nvinfer1::ICudaEngine* fake_engine()
    {
        static std::uint8_t token = 0;
        return reinterpret_cast<nvinfer1::ICudaEngine*>(&token);
    }

    static nvinfer1::IExecutionContext* fake_context()
    {
        static std::uint8_t token = 0;
        return reinterpret_cast<nvinfer1::IExecutionContext*>(&token);
    }

    mutable int deserialize_calls{0};
    mutable int create_context_calls{0};
    mutable int has_io_calls{0};
    mutable nvinfer1::IRuntime* last_runtime{nullptr};
    mutable nvinfer1::ICudaEngine* last_engine{nullptr};
    mutable const void* last_bytes{nullptr};
    mutable std::size_t last_size{0};
    mutable std::string last_tensor_name;

    int fail_deserialize_on_call{0};
    int fail_context_on_call{0};
    bool metadata_failure{false};
    bool has_requested_tensor{true};
};

inline void print_summary_and_exit_if_failures(const char* suite_name)
{
    if (g_failures > 0)
    {
        std::cerr << "[" << suite_name << "] " << g_failures << " failure(s)\n";
        std::exit(1);
    }
    std::cerr << "[" << suite_name << "] all checks passed\n";
}

} // namespace trtf::runtime::testhelpers
