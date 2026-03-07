#pragma once

#include "trtf/runtime/contracts/services.h"
#include "runtime/services/common/runtime_service_ports.h"

#include <cstdint>
#include <memory>

namespace trtf::runtime::services::vision {

class SegmentationService final : public trtf::runtime::ISegmentationService {
public:
    explicit SegmentationService(std::unique_ptr<common::ISegmentationPort> backend);

    SegmentationResult segment(const SegmentationRequest& request) override;

private:
    std::unique_ptr<common::ISegmentationPort> mBackend;
};

class PromptedSegmentationService final : public trtf::runtime::ISegmentationService {
public:
    explicit PromptedSegmentationService(std::unique_ptr<common::IPromptedSegmentationPort> backend);

    SegmentationResult segment(const SegmentationRequest& request) override;
    bool supports_prompted() const override;
    PromptedSegmentationResult segment_prompt(const PromptedSegmentationRequest& request) override;

private:
    std::unique_ptr<common::IPromptedSegmentationPort> mBackend;
};

class DetectionService final : public trtf::runtime::IDetectionService {
public:
    explicit DetectionService(std::unique_ptr<common::IDetectionPort> backend);

    DetectionResult detect(const DetectionRequest& request) override;

private:
    std::unique_ptr<common::IDetectionPort> mBackend;
};

class NeuralOperatorService final : public trtf::runtime::ISolveService {
public:
    explicit NeuralOperatorService(std::unique_ptr<common::INeuralOperatorPort> backend);

    const float* solve(const float* branch_input, int32_t branch_len, const float* trunk_input,
        int32_t trunk_len, int32_t* out_dim) override;
    const float* solve_field(
        const float* field_input, int32_t input_size, int32_t* out_channels, int32_t* out_h, int32_t* out_w)
        override;

private:
    std::unique_ptr<common::INeuralOperatorPort> mBackend;
    common::NeuralOperatorOutput mLastResult;
};

} // namespace trtf::runtime::services::vision
