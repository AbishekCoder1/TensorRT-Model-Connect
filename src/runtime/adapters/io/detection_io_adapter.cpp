#include "trtf/runtime/adapters/io/detection_io_adapter.h"

#include <cstdio>
#include <fstream>
#include <string>

namespace trtf::runtime::adapters::io {
namespace {

std::string build_detection_json(
    const DetectionArtifact& artifact,
    float conf_threshold,
    int32_t& written)
{
    const bool apply_threshold = conf_threshold >= 0.0F;
    std::string json = "[\n";
    written = 0;

    for (const auto& detection : artifact.detections)
    {
        if (apply_threshold && detection.confidence < conf_threshold)
        {
            continue;
        }
        if (written > 0)
        {
            json += ",\n";
        }

        char buffer[256];
        std::snprintf(buffer, sizeof(buffer),
            "  {\"class_id\": %d, \"confidence\": %.4f, "
            "\"x1\": %.1f, \"y1\": %.1f, \"x2\": %.1f, \"y2\": %.1f}",
            detection.class_id,
            detection.confidence,
            detection.x1,
            detection.y1,
            detection.x2,
            detection.y2);
        json += buffer;
        ++written;
    }

    json += "\n]\n";
    return json;
}

} // namespace

DetectionWriteResult DetectionJsonFileWriter::write_json(
    const DetectionArtifact& artifact,
    const char* output_path,
    float conf_threshold) const
{
    if (output_path == nullptr || output_path[0] == '\0')
    {
        return {DetectionIoStatus::kInvalidArgument, 0, "Detection output path is required"};
    }

    int32_t written = 0;
    const std::string json = build_detection_json(artifact, conf_threshold, written);

    std::ofstream out(output_path, std::ios::binary | std::ios::trunc);
    if (!out)
    {
        return {DetectionIoStatus::kIoError, 0,
            std::string("Failed to open detection output: ") + output_path};
    }

    out.write(json.data(), static_cast<std::streamsize>(json.size()));
    if (!out)
    {
        return {DetectionIoStatus::kIoError, 0,
            std::string("Failed to write detection output: ") + output_path};
    }

    return {DetectionIoStatus::kOk, written, {}};
}

} // namespace trtf::runtime::adapters::io
