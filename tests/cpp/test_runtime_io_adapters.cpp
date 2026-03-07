#include "trtf/runtime/adapters/io/audio_io_adapter.h"
#include "trtf/runtime/adapters/io/detection_io_adapter.h"
#include "trtf/runtime/adapters/io/image_io_adapter.h"
#include "trtf/runtime/adapters/io/media_io_adapter.h"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using trtf::runtime::adapters::io::AudioArtifact;
using trtf::runtime::adapters::io::AudioIoStatus;
using trtf::runtime::adapters::io::AudioWavFileWriter;
using trtf::runtime::adapters::io::DetectionArtifact;
using trtf::runtime::adapters::io::DetectionIoStatus;
using trtf::runtime::adapters::io::DetectionJsonFileWriter;
using trtf::runtime::adapters::io::DetectionRecord;
using trtf::runtime::adapters::io::FileImageInputLoader;
using trtf::runtime::adapters::io::ImageIoStatus;
using trtf::runtime::adapters::io::SamMaskArtifact;
using trtf::runtime::adapters::io::SamMaskDirectoryWriter;
using trtf::runtime::adapters::io::SegmentationArtifact;
using trtf::runtime::adapters::io::SegmentationPngFileWriter;
using trtf::runtime::adapters::io::VideoFrameArtifact;
using trtf::runtime::adapters::io::VideoFrameDirectoryWriter;
using trtf::runtime::adapters::io::WavFileAudioInputLoader;
using trtf::runtime::adapters::io::MediaIoStatus;

namespace {

int g_failures = 0;

void check(bool condition, const char* name)
{
    if (!condition)
    {
        std::cerr << "FAIL: " << name << '\n';
        ++g_failures;
    }
}

std::string read_text_file(const std::filesystem::path& path)
{
    std::ifstream input(path, std::ios::binary);
    return std::string(
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>());
}

void check_close(float actual, float expected, float tolerance, const char* name)
{
    if (std::fabs(actual - expected) > tolerance)
    {
        std::cerr << "FAIL: " << name << " actual=" << actual << " expected=" << expected << '\n';
        ++g_failures;
    }
}

DetectionArtifact make_detection_artifact()
{
    DetectionArtifact artifact;
    artifact.detections = {
        DetectionRecord{1, 0.95F, 10.0F, 20.0F, 30.0F, 40.0F},
        DetectionRecord{7, 0.25F, 1.0F, 2.0F, 3.0F, 4.0F},
    };
    return artifact;
}

std::filesystem::path prepare_temp_dir(const char* name)
{
    const auto dir = std::filesystem::temp_directory_path() / name;
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    std::filesystem::create_directories(dir, ec);
    return dir;
}

void test_detection_writer_filters_and_formats_json()
{
    DetectionJsonFileWriter writer;
    const auto dir = prepare_temp_dir("trtf_runtime_io_adapter_detection_success");
    const auto output_path = dir / "detections.json";

    const auto result = writer.write_json(make_detection_artifact(), output_path.string().c_str(), 0.5F);
    check(result.ok(), "detection writer success status");
    check(result.status == DetectionIoStatus::kOk, "detection writer ok enum");
    check(result.written == 1, "detection writer filtered count");

    const std::string contents = read_text_file(output_path);
    check(
        contents
            == "[\n"
               "  {\"class_id\": 1, \"confidence\": 0.9500, \"x1\": 10.0, \"y1\": 20.0, \"x2\": 30.0, \"y2\": 40.0}\n"
               "]\n",
        "detection writer filtered json format");

    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
}

void test_detection_writer_writes_empty_array_when_all_filtered()
{
    DetectionJsonFileWriter writer;
    const auto dir = prepare_temp_dir("trtf_runtime_io_adapter_detection_empty");
    const auto output_path = dir / "detections.json";

    const auto result = writer.write_json(make_detection_artifact(), output_path.string().c_str(), 0.99F);
    check(result.ok(), "detection writer empty success status");
    check(result.written == 0, "detection writer empty count");
    check(read_text_file(output_path) == "[\n\n]\n", "detection writer empty json format");

    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
}

void test_detection_writer_rejects_missing_output_path()
{
    DetectionJsonFileWriter writer;
    const auto result = writer.write_json(make_detection_artifact(), "", -1.0F);

    check(!result.ok(), "detection writer invalid path status");
    check(result.status == DetectionIoStatus::kInvalidArgument, "detection writer invalid path enum");
    check(result.written == 0, "detection writer invalid path count");
}

void test_detection_writer_reports_io_failures()
{
    DetectionJsonFileWriter writer;
    const auto dir = std::filesystem::temp_directory_path() / "trtf_runtime_io_adapter_detection_missing_parent";
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    const auto output_path = dir / "nested" / "detections.json";

    const auto result = writer.write_json(make_detection_artifact(), output_path.string().c_str(), -1.0F);
    check(!result.ok(), "detection writer io failure status");
    check(result.status == DetectionIoStatus::kIoError, "detection writer io failure enum");
    check(result.written == 0, "detection writer io failure count");
}

void test_audio_writer_round_trips_with_wav_loader()
{
    AudioWavFileWriter writer;
    WavFileAudioInputLoader loader;
    const auto dir = prepare_temp_dir("trtf_runtime_io_adapter_audio");
    const auto output_path = dir / "audio.wav";

    AudioArtifact artifact;
    artifact.waveform = {0.0F, 0.5F, -0.5F, 0.25F};
    artifact.sample_rate = 16000;
    artifact.num_samples = static_cast<int32_t>(artifact.waveform.size());

    const auto write_result = writer.write_wav(artifact, output_path.string().c_str());
    check(write_result.ok(), "audio writer success status");
    check(write_result.status == AudioIoStatus::kOk, "audio writer ok enum");
    check(write_result.written == artifact.num_samples, "audio writer sample count");

    const auto load_result = loader.load(output_path.string().c_str());
    check(load_result.ok(), "audio loader success status");
    check(load_result.value.sample_rate == artifact.sample_rate, "audio loader preserves sample rate");
    check(load_result.value.samples.size() == artifact.waveform.size(), "audio loader sample count");
    check_close(load_result.value.samples[1], artifact.waveform[1], 1e-3F, "audio loader sample round trip");

    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
}

void test_audio_writer_validates_arguments()
{
    AudioWavFileWriter writer;
    AudioArtifact artifact;
    artifact.waveform = {0.1F, 0.2F};
    artifact.sample_rate = 16000;
    artifact.num_samples = 3;

    const auto bad_path = writer.write_wav(artifact, "");
    check(!bad_path.ok(), "audio writer rejects empty path");
    check(bad_path.status == AudioIoStatus::kInvalidArgument, "audio writer empty path enum");

    const auto bad_size = writer.write_wav(artifact, "/tmp/ignored.wav");
    check(!bad_size.ok(), "audio writer rejects mismatched metadata");
    check(bad_size.status == AudioIoStatus::kInvalidArgument, "audio writer mismatched size enum");

    WavFileAudioInputLoader loader;
    const auto bad_load_path = loader.load("");
    check(!bad_load_path.ok(), "audio loader rejects empty path");
    check(bad_load_path.status == MediaIoStatus::kInvalidArgument, "audio loader empty path enum");
}

void test_segmentation_writer_and_image_loader_round_trip()
{
    SegmentationPngFileWriter writer;
    FileImageInputLoader loader;
    const auto dir = prepare_temp_dir("trtf_runtime_io_adapter_segmentation");
    const auto output_path = dir / "mask.png";

    SegmentationArtifact artifact;
    artifact.width = 2;
    artifact.height = 2;
    artifact.class_map = {0, 64, 128, 255};

    const auto write_result = writer.write_png(artifact, output_path.string().c_str());
    check(write_result.ok(), "segmentation writer success status");
    check(write_result.written == 1, "segmentation writer wrote one image");

    const auto load_result = loader.load(output_path.string().c_str());
    check(load_result.ok(), "image loader reads segmentation png");
    check(load_result.value.width == artifact.width, "image loader width");
    check(load_result.value.height == artifact.height, "image loader height");
    check(!load_result.value.empty(), "image loader returns pixels");

    SegmentationArtifact invalid_artifact;
    invalid_artifact.width = 2;
    invalid_artifact.height = 2;
    invalid_artifact.class_map = {1, 2};
    const auto bad_result = writer.write_png(invalid_artifact, output_path.string().c_str());
    check(!bad_result.ok(), "segmentation writer rejects short class map");
    check(bad_result.status == ImageIoStatus::kInvalidArgument, "segmentation writer invalid enum");

    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
}

void test_sam_and_video_writers_validate_and_emit_artifacts()
{
    SamMaskDirectoryWriter sam_writer;
    VideoFrameDirectoryWriter video_writer;
    const auto sam_dir = prepare_temp_dir("trtf_runtime_io_adapter_sam");
    const auto video_dir = prepare_temp_dir("trtf_runtime_io_adapter_video");

    SamMaskArtifact sam_artifact;
    sam_artifact.num_masks = 2;
    sam_artifact.mask_width = 2;
    sam_artifact.mask_height = 2;
    sam_artifact.iou_scores = {0.9F, 0.1F};
    sam_artifact.masks = {
        1.0F, 0.0F, 0.0F, 1.0F,
        0.0F, 1.0F, 1.0F, 0.0F,
    };
    const auto sam_result = sam_writer.write_masks(sam_artifact, sam_dir.string().c_str());
    check(sam_result.ok(), "sam writer success status");
    check(sam_result.written == 2, "sam writer wrote two masks");
    check(std::filesystem::exists(sam_dir / "mask_0_iou_0.9000.png"), "sam writer first mask exists");
    check(std::filesystem::exists(sam_dir / "mask_1_iou_0.1000.png"), "sam writer second mask exists");

    SamMaskArtifact bad_sam = sam_artifact;
    bad_sam.iou_scores.resize(1);
    const auto bad_sam_result = sam_writer.write_masks(bad_sam, sam_dir.string().c_str());
    check(!bad_sam_result.ok(), "sam writer rejects mismatched metadata");
    check(bad_sam_result.status == ImageIoStatus::kInvalidArgument, "sam writer invalid enum");

    VideoFrameArtifact video_artifact;
    video_artifact.num_frames = 2;
    video_artifact.width = 1;
    video_artifact.height = 1;
    video_artifact.frames = {
        1.0F, 0.0F, 0.0F,
        0.0F, 1.0F, 0.0F,
    };
    const auto video_result = video_writer.write_png_frames(video_artifact, video_dir.string().c_str());
    check(video_result.ok(), "video writer success status");
    check(video_result.written == 2, "video writer wrote two frames");
    check(std::filesystem::exists(video_dir / "frame_0000.png"), "video writer first frame exists");
    check(std::filesystem::exists(video_dir / "frame_0001.png"), "video writer second frame exists");

    VideoFrameArtifact bad_video = video_artifact;
    bad_video.frames.resize(5);
    const auto bad_video_result = video_writer.write_png_frames(bad_video, video_dir.string().c_str());
    check(!bad_video_result.ok(), "video writer rejects short frame buffer");
    check(bad_video_result.status == ImageIoStatus::kInvalidArgument, "video writer invalid enum");

    std::error_code ec;
    std::filesystem::remove_all(sam_dir, ec);
    std::filesystem::remove_all(video_dir, ec);
}

void test_media_loaders_report_invalid_and_io_errors()
{
    FileImageInputLoader image_loader;
    WavFileAudioInputLoader audio_loader;

    const auto bad_image_path = image_loader.load("");
    check(!bad_image_path.ok(), "image loader rejects empty path");
    check(bad_image_path.status == MediaIoStatus::kInvalidArgument, "image loader empty path enum");

    const auto bad_image_io = image_loader.load("/tmp/does_not_exist_runtime_io_adapter.png");
    check(!bad_image_io.ok(), "image loader reports decode failure");
    check(bad_image_io.status == MediaIoStatus::kIoError, "image loader io enum");

    const auto bad_audio_io = audio_loader.load("/tmp/does_not_exist_runtime_io_adapter.wav");
    check(!bad_audio_io.ok(), "audio loader reports decode failure");
    check(bad_audio_io.status == MediaIoStatus::kIoError, "audio loader io enum");
}

} // namespace

int main()
{
    test_detection_writer_filters_and_formats_json();
    test_detection_writer_writes_empty_array_when_all_filtered();
    test_detection_writer_rejects_missing_output_path();
    test_detection_writer_reports_io_failures();
    test_audio_writer_round_trips_with_wav_loader();
    test_audio_writer_validates_arguments();
    test_segmentation_writer_and_image_loader_round_trip();
    test_sam_and_video_writers_validate_and_emit_artifacts();
    test_media_loaders_report_invalid_and_io_errors();

    if (g_failures > 0)
    {
        std::cerr << g_failures << " test(s) FAILED\n";
        return 1;
    }

    std::cerr << "All runtime_io_adapter tests passed.\n";
    return 0;
}
