#include "trtf/runtime/adapters/io/image_io_adapter.h"

#include "stb_image_write.h"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace trtf::runtime::adapters::io {
namespace {

bool has_valid_output_path(const char* path)
{
    return path != nullptr && path[0] != '\0';
}

std::vector<uint8_t> to_u8_pixels(const float* data, std::size_t count)
{
    std::vector<uint8_t> pixels(count);
    for (std::size_t i = 0; i < count; ++i)
    {
        pixels[i] = static_cast<uint8_t>(std::max(0.0F, std::min(255.0F, data[i] * 255.0F)));
    }
    return pixels;
}

std::vector<uint8_t> build_segmentation_pixels(const SegmentationArtifact& artifact)
{
    const auto pixel_count = static_cast<std::size_t>(artifact.width) * artifact.height;
    std::vector<uint8_t> pixels(pixel_count);
    for (std::size_t i = 0; i < pixel_count; ++i)
    {
        pixels[i] = static_cast<uint8_t>(std::min(artifact.class_map[i], 255));
    }
    return pixels;
}

std::vector<uint8_t> build_sam_mask_pixels(const SamMaskArtifact& artifact, int32_t mask_index)
{
    const auto mask_size = static_cast<std::size_t>(artifact.mask_width) * artifact.mask_height;
    const auto offset = static_cast<std::size_t>(mask_index) * mask_size;
    std::vector<uint8_t> pixels(mask_size);
    for (std::size_t i = 0; i < mask_size; ++i)
    {
        pixels[i] = artifact.masks[offset + i] > 0.0F ? static_cast<uint8_t>(255) : static_cast<uint8_t>(0);
    }
    return pixels;
}

bool ensure_directory(const char* output_dir, std::string& error)
{
    std::error_code ec;
    std::filesystem::create_directories(output_dir, ec);
    if (ec)
    {
        error = std::string("Failed to create directory: ") + output_dir;
        return false;
    }
    return true;
}

} // namespace

ImageWriteResult SegmentationPngFileWriter::write_png(
    const SegmentationArtifact& artifact,
    const char* output_path) const
{
    if (!has_valid_output_path(output_path))
    {
        return {ImageIoStatus::kInvalidArgument, 0, "Segmentation output path is required"};
    }
    if (artifact.width <= 0 || artifact.height <= 0)
    {
        return {ImageIoStatus::kInvalidArgument, 0, "Segmentation image dimensions must be positive"};
    }

    const auto expected_pixels = static_cast<std::size_t>(artifact.width) * artifact.height;
    if (artifact.class_map.size() < expected_pixels)
    {
        return {ImageIoStatus::kInvalidArgument, 0, "Segmentation class map size does not match image dimensions"};
    }

    auto pixels = build_segmentation_pixels(artifact);
    if (stbi_write_png(output_path, artifact.width, artifact.height, 1, pixels.data(), artifact.width) == 0)
    {
        return {ImageIoStatus::kIoError, 0, std::string("Failed to write segmentation output: ") + output_path};
    }

    return {ImageIoStatus::kOk, 1, {}};
}

ImageWriteResult SamMaskDirectoryWriter::write_masks(
    const SamMaskArtifact& artifact,
    const char* output_dir) const
{
    if (!has_valid_output_path(output_dir))
    {
        return {ImageIoStatus::kInvalidArgument, 0, "SAM output directory is required"};
    }
    if (artifact.num_masks < 0 || artifact.mask_width <= 0 || artifact.mask_height <= 0)
    {
        return {ImageIoStatus::kInvalidArgument, 0, "SAM mask dimensions must be positive"};
    }

    const auto mask_size = static_cast<std::size_t>(artifact.mask_width) * artifact.mask_height;
    const auto expected_values = static_cast<std::size_t>(artifact.num_masks) * mask_size;
    if (artifact.masks.size() < expected_values || artifact.iou_scores.size() < static_cast<std::size_t>(artifact.num_masks))
    {
        return {ImageIoStatus::kInvalidArgument, 0, "SAM artifact size does not match mask metadata"};
    }

    std::string error;
    if (!ensure_directory(output_dir, error))
    {
        return {ImageIoStatus::kIoError, 0, std::move(error)};
    }

    for (int32_t mask_index = 0; mask_index < artifact.num_masks; ++mask_index)
    {
        auto pixels = build_sam_mask_pixels(artifact, mask_index);
        char file_name[64];
        std::snprintf(file_name, sizeof(file_name), "/mask_%d_iou_%.4f.png", mask_index, artifact.iou_scores[mask_index]);
        const std::string output_path = std::string(output_dir) + file_name;
        if (stbi_write_png(output_path.c_str(), artifact.mask_width, artifact.mask_height, 1, pixels.data(), artifact.mask_width) == 0)
        {
            return {ImageIoStatus::kIoError, mask_index,
                std::string("Failed to write SAM mask output: ") + output_path};
        }
    }

    return {ImageIoStatus::kOk, artifact.num_masks, {}};
}

ImageWriteResult VideoFrameDirectoryWriter::write_png_frames(
    const VideoFrameArtifact& artifact,
    const char* output_dir) const
{
    if (!has_valid_output_path(output_dir))
    {
        return {ImageIoStatus::kInvalidArgument, 0, "Video output directory is required"};
    }
    if (artifact.num_frames <= 0 || artifact.width <= 0 || artifact.height <= 0)
    {
        return {ImageIoStatus::kInvalidArgument, 0, "Video frame dimensions and count must be positive"};
    }

    const auto frame_size = static_cast<std::size_t>(artifact.width) * artifact.height * 3;
    const auto expected_values = static_cast<std::size_t>(artifact.num_frames) * frame_size;
    if (artifact.frames.size() < expected_values)
    {
        return {ImageIoStatus::kInvalidArgument, 0, "Video artifact size does not match frame metadata"};
    }

    std::string error;
    if (!ensure_directory(output_dir, error))
    {
        return {ImageIoStatus::kIoError, 0, std::move(error)};
    }

    for (int32_t frame_index = 0; frame_index < artifact.num_frames; ++frame_index)
    {
        const float* frame = artifact.frames.data() + static_cast<std::size_t>(frame_index) * frame_size;
        auto pixels = to_u8_pixels(frame, frame_size);
        char file_name[64];
        std::snprintf(file_name, sizeof(file_name), "/frame_%04d.png", frame_index);
        const std::string output_path = std::string(output_dir) + file_name;
        if (stbi_write_png(output_path.c_str(), artifact.width, artifact.height, 3, pixels.data(), artifact.width * 3) == 0)
        {
            return {ImageIoStatus::kIoError, frame_index,
                std::string("Failed to write video frame output: ") + output_path};
        }
    }

    return {ImageIoStatus::kOk, artifact.num_frames, {}};
}

} // namespace trtf::runtime::adapters::io
