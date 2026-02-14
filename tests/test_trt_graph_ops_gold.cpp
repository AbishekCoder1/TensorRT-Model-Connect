// Per-op gold tensor tests for TRT graph ops.
// Loads gold .safetensors fixtures, builds single-op TRT networks, runs inference,
// and compares against expected outputs.
// GPU-only test, only compiled when TRTF_HAS_TRT is enabled.

#include "trtf/model.h"
#include "model/trt_model_definition.h"
#include "runtime/trt/trt_common.h"
#include "runtime/trt/trt_graph_ops.h"
#include "model/safetensors_loader.h"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#ifndef TRTF_SOURCE_DIR
#define TRTF_SOURCE_DIR "."
#endif

#if TRTF_HAS_TRT
#include <NvInfer.h>
#endif

namespace {

std::filesystem::path gold_dir()
{
    return std::filesystem::path(TRTF_SOURCE_DIR) / "tests" / "gold";
}

bool check_close(const std::vector<float>& actual, const std::vector<float>& expected,
    float atol, const std::string& op_name)
{
    if (actual.size() != expected.size())
    {
        std::cerr << op_name << ": size mismatch actual=" << actual.size()
                  << " expected=" << expected.size() << std::endl;
        return false;
    }

    float max_diff = 0.0F;
    for (std::size_t i = 0; i < actual.size(); ++i)
    {
        const float diff = std::abs(actual[i] - expected[i]);
        if (diff > max_diff)
        {
            max_diff = diff;
        }
    }

    if (max_diff > atol)
    {
        std::cerr << op_name << ": max_abs_diff=" << max_diff << " exceeds atol=" << atol << std::endl;
        return false;
    }

    std::cout << "  " << op_name << ": max_abs_diff=" << max_diff << " (atol=" << atol << ") PASS" << std::endl;
    return true;
}

#if TRTF_HAS_TRT

bool test_rms_norm()
{
    const std::filesystem::path path = gold_dir() / "rms_norm.safetensors";
    if (!std::filesystem::exists(path))
    {
        std::cout << "  rms_norm: SKIPPED (no gold file)" << std::endl;
        return true;
    }

    trtf::SafetensorReader reader(path.string());
    const std::vector<float> input = reader.load_f32("input");
    const std::vector<float> gamma = reader.load_f32("gamma");
    const std::vector<float> expected = reader.load_f32("output");
    const std::vector<float> eps_vec = reader.load_f32("eps");
    const float eps = eps_vec[0];

    const int32_t hidden_size = static_cast<int32_t>(gamma.size());

    trtf::TrtLogger logger;
    auto builder = trtf::TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    uint32_t flags = 0U;
#if NV_TENSORRT_MAJOR < 10
    flags = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
#endif
    auto network = trtf::TrtUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(flags));
    auto config = trtf::TrtUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
#if NV_TENSORRT_MAJOR >= 8
    config->clearFlag(nvinfer1::BuilderFlag::kTF32);
#endif

    auto* input_tensor = network->addInput("input", nvinfer1::DataType::kFLOAT,
        trtf::make_dims_2d(1, hidden_size));
    nvinfer1::ITensor* eps_tensor = trtf::add_constant_tensor(*network, trtf::make_dims_2d(1, 1), {eps});

    nvinfer1::ITensor* output = trtf::add_rms_norm(*network, *input_tensor, hidden_size, gamma, *eps_tensor);
    if (output == nullptr)
    {
        std::cerr << "rms_norm: failed to build network" << std::endl;
        return false;
    }
    output->setName("output");
    network->markOutput(*output);

    auto plan = trtf::TrtUniquePtr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    if (!plan)
    {
        std::cerr << "rms_norm: failed to build plan" << std::endl;
        return false;
    }

    auto runtime = trtf::TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
    auto engine = trtf::TrtUniquePtr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(plan->data(), plan->size()));
    auto context = trtf::TrtUniquePtr<nvinfer1::IExecutionContext>(engine->createExecutionContext());

    trtf::CudaBuffer input_buf(input.size() * sizeof(float));
    trtf::CudaBuffer output_buf(expected.size() * sizeof(float));
    cudaMemcpy(input_buf.data(), input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice);

    context->setTensorAddress("input", input_buf.data());
    context->setTensorAddress("output", output_buf.data());

    trtf::CudaStream stream;
    context->enqueueV3(stream.get());
    cudaStreamSynchronize(stream.get());

    std::vector<float> actual(expected.size());
    cudaMemcpy(actual.data(), output_buf.data(), actual.size() * sizeof(float), cudaMemcpyDeviceToHost);

    return check_close(actual, expected, 1e-5F, "rms_norm");
}

bool test_matmul_rhs_constant()
{
    const std::filesystem::path path = gold_dir() / "matmul_rhs_constant.safetensors";
    if (!std::filesystem::exists(path))
    {
        std::cout << "  matmul_rhs_constant: SKIPPED (no gold file)" << std::endl;
        return true;
    }

    trtf::SafetensorReader reader(path.string());
    const std::vector<float> input = reader.load_f32("input");
    const std::vector<float> weight = reader.load_f32("weight");
    const std::vector<float> expected = reader.load_f32("output");

    const trtf::SafetensorEntry& w_entry = reader.entry("weight");
    const int32_t k = static_cast<int32_t>(w_entry.shape[0]);
    const int32_t n = static_cast<int32_t>(w_entry.shape[1]);

    trtf::TrtLogger logger;
    auto builder_ptr = trtf::TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    uint32_t flags = 0U;
#if NV_TENSORRT_MAJOR < 10
    flags = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
#endif
    auto network = trtf::TrtUniquePtr<nvinfer1::INetworkDefinition>(builder_ptr->createNetworkV2(flags));
    auto config = trtf::TrtUniquePtr<nvinfer1::IBuilderConfig>(builder_ptr->createBuilderConfig());
#if NV_TENSORRT_MAJOR >= 8
    config->clearFlag(nvinfer1::BuilderFlag::kTF32);
#endif

    auto* input_tensor = network->addInput("input", nvinfer1::DataType::kFLOAT,
        trtf::make_dims_2d(1, k));
    nvinfer1::ITensor* output = trtf::add_matmul_rhs_constant(*network, *input_tensor, k, n, weight);
    if (output == nullptr)
    {
        std::cerr << "matmul_rhs_constant: failed to build network" << std::endl;
        return false;
    }
    output->setName("output");
    network->markOutput(*output);

    auto plan = trtf::TrtUniquePtr<nvinfer1::IHostMemory>(builder_ptr->buildSerializedNetwork(*network, *config));
    auto runtime = trtf::TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
    auto engine = trtf::TrtUniquePtr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(plan->data(), plan->size()));
    auto context = trtf::TrtUniquePtr<nvinfer1::IExecutionContext>(engine->createExecutionContext());

    trtf::CudaBuffer input_buf(input.size() * sizeof(float));
    trtf::CudaBuffer output_buf(expected.size() * sizeof(float));
    cudaMemcpy(input_buf.data(), input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice);

    context->setTensorAddress("input", input_buf.data());
    context->setTensorAddress("output", output_buf.data());

    trtf::CudaStream stream;
    context->enqueueV3(stream.get());
    cudaStreamSynchronize(stream.get());

    std::vector<float> actual(expected.size());
    cudaMemcpy(actual.data(), output_buf.data(), actual.size() * sizeof(float), cudaMemcpyDeviceToHost);

    return check_close(actual, expected, 1e-5F, "matmul_rhs_constant");
}

#endif // TRTF_HAS_TRT

} // namespace

int main()
{
#if TRTF_HAS_TRT
    const cudaError_t init_err = cudaFree(nullptr);
    if (init_err != cudaSuccess)
    {
        std::cout << "test_trt_graph_ops_gold: SKIPPED (no CUDA device)" << std::endl;
        return 0;
    }

    bool all_passed = true;
    std::cout << "test_trt_graph_ops_gold:" << std::endl;

    all_passed &= test_rms_norm();
    all_passed &= test_matmul_rhs_constant();

    if (all_passed)
    {
        std::cout << "test_trt_graph_ops_gold passed" << std::endl;
        return 0;
    }
    std::cerr << "test_trt_graph_ops_gold FAILED" << std::endl;
    return 1;
#else
    std::cout << "test_trt_graph_ops_gold: SKIPPED (TRTF_HAS_TRT=0)" << std::endl;
    return 0;
#endif
}
