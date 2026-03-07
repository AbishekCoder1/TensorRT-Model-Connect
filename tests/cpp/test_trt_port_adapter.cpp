#include "runtime/adapters/trt/trt_port_adapter.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const std::string& test_name)
{
    if (!condition)
    {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

template <typename T>
T* opaque_ptr()
{
    static std::uint8_t token = 0;
    return reinterpret_cast<T*>(&token);
}

struct FakeApiState {
    int deserialize_calls{0};
    int create_context_calls{0};
    int get_nb_io_tensors_calls{0};
    int get_io_tensor_name_calls{0};

    nvinfer1::IRuntime* last_runtime{nullptr};
    nvinfer1::ICudaEngine* last_engine{nullptr};
    const void* last_bytes{nullptr};
    std::size_t last_size{0};

    nvinfer1::ICudaEngine* deserialize_result{nullptr};
    nvinfer1::IExecutionContext* create_context_result{nullptr};
    int32_t io_tensor_count{0};
    std::vector<const char*> io_tensor_names;
};

FakeApiState* gFakeApiState = nullptr;

nvinfer1::ICudaEngine* fake_deserialize_engine(
    nvinfer1::IRuntime* runtime,
    const void* serialized_bytes,
    std::size_t serialized_size)
{
    ++gFakeApiState->deserialize_calls;
    gFakeApiState->last_runtime = runtime;
    gFakeApiState->last_bytes = serialized_bytes;
    gFakeApiState->last_size = serialized_size;
    return gFakeApiState->deserialize_result;
}

nvinfer1::IExecutionContext* fake_create_execution_context(nvinfer1::ICudaEngine* engine)
{
    ++gFakeApiState->create_context_calls;
    gFakeApiState->last_engine = engine;
    return gFakeApiState->create_context_result;
}

int32_t fake_get_nb_io_tensors(const nvinfer1::ICudaEngine* engine)
{
    ++gFakeApiState->get_nb_io_tensors_calls;
    gFakeApiState->last_engine = const_cast<nvinfer1::ICudaEngine*>(engine);
    return gFakeApiState->io_tensor_count;
}

const char* fake_get_io_tensor_name(const nvinfer1::ICudaEngine* engine, int32_t index)
{
    ++gFakeApiState->get_io_tensor_name_calls;
    gFakeApiState->last_engine = const_cast<nvinfer1::ICudaEngine*>(engine);
    if (index < 0 || static_cast<std::size_t>(index) >= gFakeApiState->io_tensor_names.size())
    {
        return nullptr;
    }
    return gFakeApiState->io_tensor_names[static_cast<std::size_t>(index)];
}

trtf::runtime::adapters::trt::TrtPortAdapter make_adapter(FakeApiState& state)
{
    gFakeApiState = &state;
    trtf::runtime::adapters::trt::TrtPortAdapter::Api api;
    api.deserialize_engine = fake_deserialize_engine;
    api.create_execution_context = fake_create_execution_context;
    api.get_nb_io_tensors = fake_get_nb_io_tensors;
    api.get_io_tensor_name = fake_get_io_tensor_name;
    return trtf::runtime::adapters::trt::TrtPortAdapter(api);
}

void test_deserialize_null_bytes_returns_invalid_argument()
{
    FakeApiState state;
    auto adapter = make_adapter(state);

    const auto result = adapter.deserialize_engine(opaque_ptr<nvinfer1::IRuntime>(), nullptr, 16);
    check(!result.ok(), "deserialize_null_bytes: failure expected");
    check(result.status == trtf::runtime::TrtPortStatus::kInvalidArgument,
        "deserialize_null_bytes: invalid argument");
    check(result.message.find("pointer") != std::string::npos,
        "deserialize_null_bytes: error message");
    check(state.deserialize_calls == 0, "deserialize_null_bytes: api not invoked");
}

void test_deserialize_failure_maps_error()
{
    FakeApiState state;
    auto adapter = make_adapter(state);
    std::uint8_t bytes[3] = {1, 2, 3};

    const auto result =
        adapter.deserialize_engine(opaque_ptr<nvinfer1::IRuntime>(), bytes, sizeof(bytes));
    check(!result.ok(), "deserialize_failure: failure expected");
    check(result.status == trtf::runtime::TrtPortStatus::kDeserializeFailed,
        "deserialize_failure: deserialize error");
    check(result.message.find("deserializeCudaEngine") != std::string::npos,
        "deserialize_failure: message");
    check(state.deserialize_calls == 1, "deserialize_failure: api invoked once");
    check(state.last_runtime == opaque_ptr<nvinfer1::IRuntime>(),
        "deserialize_failure: runtime passed through");
    check(state.last_bytes == bytes, "deserialize_failure: bytes passed through");
    check(state.last_size == sizeof(bytes), "deserialize_failure: size passed through");
}

void test_create_execution_context_failure_maps_error()
{
    FakeApiState state;
    auto adapter = make_adapter(state);

    const auto result = adapter.create_execution_context(opaque_ptr<nvinfer1::ICudaEngine>());
    check(!result.ok(), "create_context_failure: failure expected");
    check(result.status == trtf::runtime::TrtPortStatus::kContextCreationFailed,
        "create_context_failure: context error");
    check(result.message.find("createExecutionContext") != std::string::npos,
        "create_context_failure: message");
    check(state.create_context_calls == 1, "create_context_failure: api invoked once");
}

void test_success_path_and_metadata_lookup()
{
    FakeApiState state;
    state.deserialize_result = opaque_ptr<nvinfer1::ICudaEngine>();
    state.create_context_result = opaque_ptr<nvinfer1::IExecutionContext>();
    state.io_tensor_count = 2;
    state.io_tensor_names = {"token_id", "logits"};
    auto adapter = make_adapter(state);

    std::uint8_t bytes[2] = {9, 7};
    const auto engine_result =
        adapter.deserialize_engine(opaque_ptr<nvinfer1::IRuntime>(), bytes, sizeof(bytes));
    check(engine_result.ok(), "success: deserialize ok");
    check(engine_result.value == state.deserialize_result, "success: deserialize value");

    const auto context_result = adapter.create_execution_context(engine_result.value);
    check(context_result.ok(), "success: context ok");
    check(context_result.value == state.create_context_result, "success: context value");

    const auto has_logits = adapter.has_io_tensor_named(engine_result.value, "logits");
    check(has_logits.ok(), "success: metadata lookup ok");
    check(has_logits.value, "success: metadata lookup true");

    const auto has_missing = adapter.has_io_tensor_named(engine_result.value, "missing");
    check(has_missing.ok(), "success: metadata miss lookup ok");
    check(!has_missing.value, "success: metadata miss false");

    check(state.get_nb_io_tensors_calls == 2, "success: count queried twice");
    check(state.get_io_tensor_name_calls == 4, "success: names iterated");
}

} // namespace

int main()
{
    test_deserialize_null_bytes_returns_invalid_argument();
    test_deserialize_failure_maps_error();
    test_create_execution_context_failure_maps_error();
    test_success_path_and_metadata_lookup();

    if (failures > 0)
    {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }

    std::cerr << "All trt_port_adapter tests passed.\n";
    return 0;
}
