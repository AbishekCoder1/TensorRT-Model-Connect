#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace trtf::runtime::adapters::io {

enum class ImageIoStatus {
    kOk,
    kInvalidArgument,
    kIoError,
};

struct ImageWriteResult {
    ImageIoStatus status{ImageIoStatus::kOk};
    int32_t written{0};
    std::string message;

    [[nodiscard]] bool ok() const
    {
        return status == ImageIoStatus::kOk;
    }
};

struct SegmentationArtifact {
    std::vector<int32_t> class_map;
    int32_t width{0};
    int32_t height{0};
};

struct SamMaskArtifact {
    std::vector<float> masks;
    std::vector<float> iou_scores;
    int32_t num_masks{0};
    int32_t mask_width{0};
    int32_t mask_height{0};
};

struct VideoFrameArtifact {
    std::vector<float> frames;
    int32_t num_frames{0};
    int32_t width{0};
    int32_t height{0};
};

class ISegmentationArtifactWriter {
public:
    virtual ~ISegmentationArtifactWriter() = default;

    virtual ImageWriteResult write_png(const SegmentationArtifact& artifact, const char* output_path) const = 0;
};

class SegmentationPngFileWriter final : public ISegmentationArtifactWriter {
public:
    ImageWriteResult write_png(const SegmentationArtifact& artifact, const char* output_path) const override;
};

class ISamMaskArtifactWriter {
public:
    virtual ~ISamMaskArtifactWriter() = default;

    virtual ImageWriteResult write_masks(const SamMaskArtifact& artifact, const char* output_dir) const = 0;
};

class SamMaskDirectoryWriter final : public ISamMaskArtifactWriter {
public:
    ImageWriteResult write_masks(const SamMaskArtifact& artifact, const char* output_dir) const override;
};

class IVideoFrameArtifactWriter {
public:
    virtual ~IVideoFrameArtifactWriter() = default;

    virtual ImageWriteResult write_png_frames(const VideoFrameArtifact& artifact, const char* output_dir) const = 0;
};

class VideoFrameDirectoryWriter final : public IVideoFrameArtifactWriter {
public:
    ImageWriteResult write_png_frames(const VideoFrameArtifact& artifact, const char* output_dir) const override;
};

} // namespace trtf::runtime::adapters::io
