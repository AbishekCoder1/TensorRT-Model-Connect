#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace trtf::runtime::adapters::io {

enum class DetectionIoStatus {
    kOk,
    kInvalidArgument,
    kIoError,
};

struct DetectionRecord {
    int32_t class_id{0};
    float confidence{0.0F};
    float x1{0.0F};
    float y1{0.0F};
    float x2{0.0F};
    float y2{0.0F};
};

struct DetectionArtifact {
    std::vector<DetectionRecord> detections;
};

struct DetectionWriteResult {
    DetectionIoStatus status{DetectionIoStatus::kOk};
    int32_t written{0};
    std::string message;

    [[nodiscard]] bool ok() const
    {
        return status == DetectionIoStatus::kOk;
    }
};

class IDetectionArtifactWriter {
public:
    virtual ~IDetectionArtifactWriter() = default;

    virtual DetectionWriteResult write_json(
        const DetectionArtifact& artifact,
        const char* output_path,
        float conf_threshold) const
        = 0;
};

class DetectionJsonFileWriter final : public IDetectionArtifactWriter {
public:
    DetectionWriteResult write_json(
        const DetectionArtifact& artifact,
        const char* output_path,
        float conf_threshold) const override;
};

} // namespace trtf::runtime::adapters::io
