#include "runtime/services/vision/vision_runtime_services.h"

#include <exception>
#include <utility>

namespace trtf::runtime::services::vision {

SegmentationService::SegmentationService(std::unique_ptr<common::ISegmentationPort> backend)
    : mBackend(std::move(backend))
{
}

SegmentationResult SegmentationService::segment(const SegmentationRequest& request)
{
    if (mBackend == nullptr || request.image.empty())
    {
        return SegmentationResult::Failure(RuntimeServiceStatus::kInvalidArgument, "segmentation backend or image input missing");
    }

    try
    {
        return SegmentationResult::Success(mBackend->segment_image(request.image.decoded));
    }
    catch (const std::exception& e)
    {
        return SegmentationResult::Failure(RuntimeServiceStatus::kRuntimeError, e.what());
    }
}

PromptedSegmentationService::PromptedSegmentationService(std::unique_ptr<common::IPromptedSegmentationPort> backend)
    : mBackend(std::move(backend))
{
}

SegmentationResult PromptedSegmentationService::segment(const SegmentationRequest& request)
{
    (void) request;
    return SegmentationResult::Failure(RuntimeServiceStatus::kUnsupported, "prompted segmentation only supports point prompts");
}

bool PromptedSegmentationService::supports_prompted() const
{
    return mBackend != nullptr;
}

PromptedSegmentationResult PromptedSegmentationService::segment_prompt(const PromptedSegmentationRequest& request)
{
    if (mBackend == nullptr || request.image.empty())
    {
        return PromptedSegmentationResult::Failure(
            RuntimeServiceStatus::kInvalidArgument, "prompted segmentation backend or image input missing");
    }

    try
    {
        if (!mBackend->encode_image(request.image.decoded))
        {
            return PromptedSegmentationResult::Failure(RuntimeServiceStatus::kRuntimeError, "SAM: Failed to encode image");
        }
        return PromptedSegmentationResult::Success(
            mBackend->segment_point(request.point_x, request.point_y, request.is_foreground));
    }
    catch (const std::exception& e)
    {
        return PromptedSegmentationResult::Failure(RuntimeServiceStatus::kRuntimeError, e.what());
    }
}

DetectionService::DetectionService(std::unique_ptr<common::IDetectionPort> backend)
    : mBackend(std::move(backend))
{
}

DetectionResult DetectionService::detect(const DetectionRequest& request)
{
    if (mBackend == nullptr || request.image.empty())
    {
        return DetectionResult::Failure(RuntimeServiceStatus::kInvalidArgument, "detection backend or image input missing");
    }

    try
    {
        return DetectionResult::Success(mBackend->detect_image(request.image.decoded));
    }
    catch (const std::exception& e)
    {
        return DetectionResult::Failure(RuntimeServiceStatus::kRuntimeError, e.what());
    }
}

NeuralOperatorService::NeuralOperatorService(std::unique_ptr<common::INeuralOperatorPort> backend)
    : mBackend(std::move(backend))
{
}

const float* NeuralOperatorService::solve(
    const float* branch_input,
    int32_t branch_len,
    const float* trunk_input,
    int32_t trunk_len,
    int32_t* out_dim)
{
    if (mBackend == nullptr || branch_input == nullptr || trunk_input == nullptr)
    {
        return nullptr;
    }

    try
    {
        mLastResult = mBackend->solve(branch_input, branch_len, trunk_input, trunk_len);
        if (out_dim != nullptr)
        {
            *out_dim = mLastResult.output_dim;
        }
        return mLastResult.output.data();
    }
    catch (const std::exception&)
    {
        return nullptr;
    }
}

const float* NeuralOperatorService::solve_field(
    const float* field_input,
    int32_t input_size,
    int32_t* out_channels,
    int32_t* out_h,
    int32_t* out_w)
{
    if (mBackend == nullptr || field_input == nullptr)
    {
        return nullptr;
    }

    try
    {
        mLastResult = mBackend->solve_field(field_input, input_size);
        if (out_channels != nullptr)
        {
            *out_channels = mLastResult.out_channels;
        }
        if (out_h != nullptr)
        {
            *out_h = mLastResult.height;
        }
        if (out_w != nullptr)
        {
            *out_w = mLastResult.width;
        }
        return mLastResult.output.data();
    }
    catch (const std::exception&)
    {
        return nullptr;
    }
}

} // namespace trtf::runtime::services::vision
