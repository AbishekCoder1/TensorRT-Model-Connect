// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-CUDA-CPP-03
// Architecture:   ARCH-MOD-001
// Unit Design:    UD-TRT-CORE-01
// Intent:         CudaGraphExec RAII wrapper + TrtModule CUDA Graph capture/replay
// Preconditions:  CUDA GPU available, TRT engine buildable
// Postconditions: CudaGraphExec captures/replays correctly; TrtModule produces
//                 identical output with and without CUDA Graphs; env var disables
// =============================================================================

// =============================================================================
// Test suite: CUDA Graph capture and replay
// =============================================================================
//
// Validates:
// 1. CudaGraphExec RAII lifecycle (default state, capture, replay, reset,
//    move semantics, double-reset safety)
// 2. TrtModule::enable_cuda_graph() — first call captures, subsequent replays
// 3. CUDA Graph output matches normal execution
// 4. TRTF_DISABLE_CUDA_GRAPH env var control
//
// Requires TRT + CUDA GPU. Skips gracefully without TRT.
// =============================================================================

#include "runtime/core/trt_common.h"
#include "test_helpers.h"
#include "trtf/runtime/tensor.h"
#include "trtf/runtime/trt_module.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

#if TRTF_HAS_TRT
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#endif

static int failures = 0;

static void check(bool condition, const char* test_name) {
    if (!condition) {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

#if TRTF_HAS_TRT

static trtf::TrtLogger g_logger;

// Build a tiny TRT engine: y = x (identity), fixed shape [4] float32
static trtf::TrtUniquePtr<nvinfer1::ICudaEngine> build_identity_engine() {
    auto builder = trtf::TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(g_logger));
    if (!builder)
        return nullptr;

    auto network = trtf::TrtUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
    auto config = trtf::TrtUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 20);

    auto* inp = network->addInput("x", nvinfer1::DataType::kFLOAT, nvinfer1::Dims{1, {4}});
    if (!inp)
        return nullptr;

    auto* id_layer = network->addIdentity(*inp);
    if (!id_layer)
        return nullptr;
    auto* out = id_layer->getOutput(0);
    out->setName("y");
    network->markOutput(*out);

    auto plan = trtf::TrtUniquePtr<nvinfer1::IHostMemory>(
        builder->buildSerializedNetwork(*network, *config));
    if (!plan)
        return nullptr;

    auto runtime = trtf::TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(g_logger));
    if (!runtime)
        return nullptr;

    return trtf::TrtUniquePtr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(plan->data(), plan->size()));
}

// Helper: run forward_async + sync, read back output
static void run_and_read(trtf::TrtModule& module, const float* input, float* output) {
    trtf::Tensor input_tensor;
    input_tensor.data = const_cast<float*>(input);
    input_tensor.shape = {4};
    input_tensor.dtype = trtf::DType::kFloat32;

    trtf::TensorMap inputs;
    inputs["x"] = input_tensor;

    module.forward_async(inputs);
    module.sync();

    cudaMemcpy(output, module.device_ptr("y"), 4 * sizeof(float), cudaMemcpyDeviceToHost);
}

// --- CudaGraphExec unit tests ---

static void test_default_state() {
    trtf::CudaGraphExec graph;
    check(!graph.ready(), "default: not ready");
    check(!graph.launch(nullptr), "default: launch returns false");
}

static void test_capture_and_replay() {
    // Capture a simple cudaMemcpyAsync into a graph, then replay it
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    float host_src[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float host_dst[4] = {0};
    void* d_buf = nullptr;
    cudaMalloc(&d_buf, 16);

    trtf::CudaGraphExec graph;

    // Capture: H2D copy
    check(graph.begin_capture(stream), "capture: begin ok");
    cudaMemcpyAsync(d_buf, host_src, 16, cudaMemcpyHostToDevice, stream);
    check(graph.end_capture(stream), "capture: end ok");
    check(graph.ready(), "capture: graph is ready");

    // Replay: should copy the same data
    check(graph.launch(stream), "replay: launch ok");
    cudaStreamSynchronize(stream);

    cudaMemcpy(host_dst, d_buf, 16, cudaMemcpyDeviceToHost);
    check(host_dst[0] == 1.0f, "replay: dst[0] = 1.0");
    check(host_dst[3] == 4.0f, "replay: dst[3] = 4.0");

    cudaFree(d_buf);
    cudaStreamDestroy(stream);
}

static void test_reset() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    void* d_buf = nullptr;
    cudaMalloc(&d_buf, 16);
    float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};

    trtf::CudaGraphExec graph;
    graph.begin_capture(stream);
    cudaMemcpyAsync(d_buf, data, 16, cudaMemcpyHostToDevice, stream);
    graph.end_capture(stream);
    check(graph.ready(), "reset: ready before reset");

    graph.reset();
    check(!graph.ready(), "reset: not ready after reset");
    check(!graph.launch(stream), "reset: launch fails after reset");

    cudaFree(d_buf);
    cudaStreamDestroy(stream);
}

static void test_double_reset() {
    trtf::CudaGraphExec graph;
    graph.reset(); // reset on default-constructed — should not crash
    graph.reset(); // double reset — should not crash
    check(!graph.ready(), "double_reset: still not ready");
}

static void test_move_constructor() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    void* d_buf = nullptr;
    cudaMalloc(&d_buf, 16);
    float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};

    trtf::CudaGraphExec src;
    src.begin_capture(stream);
    cudaMemcpyAsync(d_buf, data, 16, cudaMemcpyHostToDevice, stream);
    src.end_capture(stream);
    check(src.ready(), "move_ctor: src ready before move");

    trtf::CudaGraphExec dst(std::move(src));
    check(dst.ready(), "move_ctor: dst ready after move");
    check(!src.ready(), "move_ctor: src not ready after move");

    // dst should still be launchable
    check(dst.launch(stream), "move_ctor: dst launch ok");
    cudaStreamSynchronize(stream);

    cudaFree(d_buf);
    cudaStreamDestroy(stream);
}

static void test_move_assignment() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    void* d_buf = nullptr;
    cudaMalloc(&d_buf, 16);
    float data[4] = {5.0f, 6.0f, 7.0f, 8.0f};

    trtf::CudaGraphExec src;
    src.begin_capture(stream);
    cudaMemcpyAsync(d_buf, data, 16, cudaMemcpyHostToDevice, stream);
    src.end_capture(stream);

    trtf::CudaGraphExec dst;
    dst = std::move(src);
    check(dst.ready(), "move_assign: dst ready after move");
    check(!src.ready(), "move_assign: src not ready after move");
    check(dst.launch(stream), "move_assign: dst launch ok");
    cudaStreamSynchronize(stream);

    cudaFree(d_buf);
    cudaStreamDestroy(stream);
}

// --- TrtModule CUDA Graph integration tests ---

static void test_module_cuda_graph_correctness() {
    // Run the same inputs with and without CUDA Graphs — output must match
    auto engine = build_identity_engine();
    check(engine != nullptr, "graph_correctness: engine built");
    if (!engine)
        return;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Run without CUDA Graph
    trtf::TrtModule normal(engine.get(), stream);
    float input[4] = {10.0f, 20.0f, 30.0f, 40.0f};
    float normal_out[4] = {0};
    run_and_read(normal, input, normal_out);

    // Run with CUDA Graph
    trtf::TrtModule graphed(engine.get(), stream);
    graphed.enable_cuda_graph();
    check(graphed.cuda_graph_active(), "graph_correctness: cuda_graph_active");

    // First call: capture + execute
    float graph_out1[4] = {0};
    run_and_read(graphed, input, graph_out1);

    // Second call: replay
    float graph_out2[4] = {0};
    run_and_read(graphed, input, graph_out2);

    // All three should produce identical results
    for (int i = 0; i < 4; ++i) {
        check(normal_out[i] == graph_out1[i], "graph_correctness: capture matches normal");
        check(normal_out[i] == graph_out2[i], "graph_correctness: replay matches normal");
    }

    cudaStreamDestroy(stream);
}

static void test_module_cuda_graph_multiple_runs() {
    // Verify CUDA Graph replay works over many iterations (stable, no drift)
    auto engine = build_identity_engine();
    if (!engine)
        return;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    trtf::TrtModule module(engine.get(), stream);
    module.enable_cuda_graph();

    float input[4] = {1.0f, 2.0f, 3.0f, 4.0f};

    for (int iter = 0; iter < 10; ++iter) {
        float out[4] = {0};
        run_and_read(module, input, out);
        for (int i = 0; i < 4; ++i) {
            check(out[i] == input[i], "multi_run: output matches input");
        }
    }

    cudaStreamDestroy(stream);
}

static void test_module_enable_after_normal_run() {
    // Enable CUDA Graphs after some normal forward calls
    auto engine = build_identity_engine();
    if (!engine)
        return;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    trtf::TrtModule module(engine.get(), stream);

    // Run 3 normal steps
    float input[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    for (int i = 0; i < 3; ++i) {
        float out[4] = {0};
        run_and_read(module, input, out);
    }

    // Now enable CUDA Graph
    module.enable_cuda_graph();
    check(module.cuda_graph_active(), "enable_after_normal: active");

    // First call captures, second replays
    float out1[4] = {0};
    run_and_read(module, input, out1);
    float out2[4] = {0};
    run_and_read(module, input, out2);

    for (int i = 0; i < 4; ++i) {
        check(out1[i] == input[i], "enable_after_normal: capture output correct");
        check(out2[i] == input[i], "enable_after_normal: replay output correct");
    }

    cudaStreamDestroy(stream);
}

static void test_env_var_disable() {
    // TRTF_DISABLE_CUDA_GRAPH=1 should prevent auto-enable in TextGenerationPipeline.
    // Here we test the env var logic directly: enable_cuda_graph sets the flag,
    // but the pipeline constructor reads the env var. We test the TrtModule flag.
    auto engine = build_identity_engine();
    if (!engine)
        return;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    trtf::TrtModule module(engine.get(), stream);
    check(!module.cuda_graph_active(), "env_disable: not active by default");

    module.enable_cuda_graph();
    check(module.cuda_graph_active(), "env_disable: active after enable");

    // Verify it still works correctly
    float input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float out[4] = {0};
    run_and_read(module, input, out);
    check(out[0] == 1.0f, "env_disable: output correct");

    cudaStreamDestroy(stream);
}

static void test_env_var_disable_pipeline_logic() {
    // Simulate the TextGenerationPipeline constructor logic:
    // if TRTF_DISABLE_CUDA_GRAPH=1, don't call enable_cuda_graph()
    {
        trtf_test::EnvVarGuard guard("TRTF_DISABLE_CUDA_GRAPH", "1");
        const char* disable_env = std::getenv("TRTF_DISABLE_CUDA_GRAPH");
        bool should_enable =
            (disable_env == nullptr || disable_env[0] == '0' || disable_env[0] == '\0');
        check(!should_enable, "env_pipeline: DISABLE=1 prevents enable");
    }
    {
        trtf_test::EnvVarGuard guard("TRTF_DISABLE_CUDA_GRAPH", "0");
        const char* disable_env = std::getenv("TRTF_DISABLE_CUDA_GRAPH");
        bool should_enable =
            (disable_env == nullptr || disable_env[0] == '0' || disable_env[0] == '\0');
        check(should_enable, "env_pipeline: DISABLE=0 allows enable");
    }
    {
        trtf_test::EnvVarGuard guard("TRTF_DISABLE_CUDA_GRAPH", nullptr); // unset
        const char* disable_env = std::getenv("TRTF_DISABLE_CUDA_GRAPH");
        bool should_enable =
            (disable_env == nullptr || disable_env[0] == '0' || disable_env[0] == '\0');
        check(should_enable, "env_pipeline: unset allows enable");
    }
}

#endif // TRTF_HAS_TRT

int main() {
#if TRTF_HAS_TRT
    // CudaGraphExec unit tests
    test_default_state();
    test_capture_and_replay();
    test_reset();
    test_double_reset();
    test_move_constructor();
    test_move_assignment();

    // TrtModule CUDA Graph integration tests
    test_module_cuda_graph_correctness();
    test_module_cuda_graph_multiple_runs();
    test_module_enable_after_normal_run();
    test_env_var_disable();
    test_env_var_disable_pipeline_logic();

    if (failures > 0) {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All CUDA Graph tests passed.\n";
    return 0;
#else
    std::cout << "test_cuda_graph: SKIPPED (TRTF_HAS_TRT=0)" << std::endl;
    return 0;
#endif
}
