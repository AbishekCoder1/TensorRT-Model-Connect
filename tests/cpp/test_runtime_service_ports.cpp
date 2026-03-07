#include "runtime/services/common/runtime_service_ports.h"
#include "runtime/trt/audio/bark_backend.h"
#include "runtime/trt/audio/magpie_tts_backend.h"
#include "runtime/trt/audio/omni_backend.h"
#include "runtime/trt/audio/speech_backend.h"
#include "runtime/trt/audio/whisper_backend.h"
#include "runtime/trt/perception/detection_backend.h"
#include "runtime/trt/perception/neural_operator_backend.h"
#include "runtime/trt/perception/sam_backend.h"
#include "runtime/trt/perception/segmentation_backend.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

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

trtf::runtime::adapters::io::DecodedImage make_decoded_image()
{
    trtf::runtime::adapters::io::DecodedImage image;
    image.pixels = {255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255};
    image.width = 2;
    image.height = 2;
    image.channels = 3;
    return image;
}

class FakeGenerationBackend final : public trtf::IGenerationBackend {
public:
    bool is_available() const override
    {
        return true;
    }

    const char* name() const override
    {
        return "fake";
    }

    std::vector<int32_t> generate(
        const std::vector<int32_t>& input_ids,
        const trtf::GenerationConfig& config) override
    {
        ++generate_calls;
        last_input_ids = input_ids;
        last_max_new_tokens = static_cast<int32_t>(config.max_new_tokens);
        return next_output;
    }

    bool supports_vision() const override
    {
        return vision_enabled;
    }

    int generate_calls{0};
    bool vision_enabled{true};
    int32_t last_max_new_tokens{0};
    std::vector<int32_t> last_input_ids;
    std::vector<int32_t> next_output{7, 8};
};

class FakeTextBackendAdapter final : public trtf::runtime::services::common::ITextGenerationBackendAdapter {
public:
    std::vector<int32_t> generate_text(
        const std::vector<int32_t>& input_ids,
        const trtf::GenerationConfig& config) override
    {
        ++generate_text_calls;
        last_text_ids = input_ids;
        last_text_max_new_tokens = static_cast<int32_t>(config.max_new_tokens);
        return text_output;
    }

    bool supports_vision() const override
    {
        return vision_enabled;
    }

    bool prepare_image(
        const trtf::runtime::adapters::io::DecodedImage& image,
        std::vector<float>& image_features,
        int32_t& num_features,
        int32_t& feature_dim,
        std::string& error) override
    {
        ++prepare_image_calls;
        last_image_width = image.width;
        last_image_height = image.height;
        if (!prepare_success)
        {
            error = prepare_error;
            return false;
        }
        image_features = next_image_features;
        num_features = next_num_features;
        feature_dim = next_feature_dim;
        return true;
    }

    std::string format_vision_prompt(const std::string& prompt) const override
    {
        return prompt_prefix + prompt;
    }

    std::vector<int32_t> generate_with_image(
        const std::vector<int32_t>& input_ids,
        const float* image_features,
        int32_t num_features,
        int32_t feature_dim,
        const trtf::GenerationConfig& config) override
    {
        ++generate_with_image_calls;
        last_image_ids = input_ids;
        last_image_feature_count = num_features;
        last_image_feature_dim = feature_dim;
        last_image_max_new_tokens = static_cast<int32_t>(config.max_new_tokens);
        captured_features.assign(image_features, image_features + (num_features * feature_dim));
        return image_output;
    }

    bool vision_enabled{true};
    bool prepare_success{true};
    int generate_text_calls{0};
    int prepare_image_calls{0};
    int generate_with_image_calls{0};
    int32_t last_text_max_new_tokens{0};
    int32_t last_image_feature_count{0};
    int32_t last_image_feature_dim{0};
    int32_t last_image_max_new_tokens{0};
    int32_t last_image_width{0};
    int32_t last_image_height{0};
    std::string prepare_error{"bad-image"};
    std::string prompt_prefix{"[vl] "};
    std::vector<int32_t> last_text_ids;
    std::vector<int32_t> last_image_ids;
    std::vector<float> captured_features;
    std::vector<float> next_image_features{1.0F, 2.0F, 3.0F, 4.0F};
    int32_t next_num_features{2};
    int32_t next_feature_dim{2};
    std::vector<int32_t> text_output{1, 2};
    std::vector<int32_t> image_output{3, 4};
};

class FakeAudioBackendAdapter final : public trtf::runtime::services::common::IAudioGenerationBackendAdapter {
public:
    trtf::runtime::adapters::io::AudioArtifact generate_audio(
        const std::vector<int32_t>& input_ids,
        int32_t max_tokens) override
    {
        ++calls;
        last_input_ids = input_ids;
        last_max_tokens = max_tokens;
        return next_artifact;
    }

    int calls{0};
    int32_t last_max_tokens{0};
    std::vector<int32_t> last_input_ids;
    trtf::runtime::adapters::io::AudioArtifact next_artifact{{0.1F, 0.2F}, 24000, 2};
};

class FakeTranscriptionBackendAdapter final : public trtf::runtime::services::common::ITranscriptionBackendAdapter {
public:
    trtf::runtime::services::common::TranscriptionOutput transcribe(
        const float* mel_data,
        int32_t mel_bins,
        int32_t mel_length,
        int32_t max_new_tokens) override
    {
        ++calls;
        last_mel.assign(mel_data, mel_data + (mel_bins * mel_length));
        last_mel_bins = mel_bins;
        last_mel_length = mel_length;
        last_max_new_tokens = max_new_tokens;
        return next_output;
    }

    int calls{0};
    int32_t last_mel_bins{0};
    int32_t last_mel_length{0};
    int32_t last_max_new_tokens{0};
    std::vector<float> last_mel;
    trtf::runtime::services::common::TranscriptionOutput next_output{"decoded", {9, 10}, 2};
};

class FakeSpeechBackendAdapter final : public trtf::runtime::services::common::ISpeechSynthesisBackendAdapter {
public:
    trtf::runtime::services::common::SpeechInputPolicy input_policy() const override
    {
        return policy;
    }

    trtf::runtime::adapters::io::AudioArtifact process_audio(
        const float* input_samples,
        int32_t num_input_samples,
        int32_t max_output_frames,
        int32_t input_sample_rate,
        int32_t tail_frames) override
    {
        ++calls;
        last_samples.assign(input_samples, input_samples + num_input_samples);
        last_max_output_frames = max_output_frames;
        last_input_sample_rate = input_sample_rate;
        last_tail_frames = tail_frames;
        return next_artifact;
    }

    mutable int calls{0};
    mutable int32_t last_max_output_frames{0};
    mutable int32_t last_input_sample_rate{0};
    mutable int32_t last_tail_frames{0};
    mutable std::vector<float> last_samples;
    trtf::runtime::services::common::SpeechInputPolicy policy{16000, true};
    trtf::runtime::adapters::io::AudioArtifact next_artifact{{0.9F, 1.0F}, 16000, 2};
};

class FakeSegmentationBackendAdapter final : public trtf::runtime::services::common::ISegmentationBackendAdapter {
public:
    trtf::runtime::adapters::io::SegmentationArtifact segment_image(
        const trtf::runtime::adapters::io::DecodedImage& image) override
    {
        ++calls;
        last_image_width = image.width;
        last_image_height = image.height;
        return next_artifact;
    }

    int calls{0};
    int32_t last_image_width{0};
    int32_t last_image_height{0};
    trtf::runtime::adapters::io::SegmentationArtifact next_artifact{{1, 2, 3, 4}, 2, 2};
};

class FakeSamBackendAdapter final : public trtf::runtime::services::common::IPromptedSegmentationBackendAdapter {
public:
    bool encode_image(const trtf::runtime::adapters::io::DecodedImage& image) override
    {
        ++encode_calls;
        last_image_width = image.width;
        last_image_height = image.height;
        return encode_success;
    }

    trtf::runtime::adapters::io::SamMaskArtifact segment_point(float point_x, float point_y, bool is_foreground) override
    {
        ++segment_calls;
        last_point_x = point_x;
        last_point_y = point_y;
        last_is_foreground = is_foreground;
        return next_artifact;
    }

    bool encode_success{true};
    int encode_calls{0};
    int segment_calls{0};
    float last_point_x{0.0F};
    float last_point_y{0.0F};
    bool last_is_foreground{false};
    int32_t last_image_width{0};
    int32_t last_image_height{0};
    trtf::runtime::adapters::io::SamMaskArtifact next_artifact{{1.0F, 0.0F, 0.0F, 1.0F}, {0.9F}, 1, 2, 2};
};

class FakeDetectionBackendAdapter final : public trtf::runtime::services::common::IDetectionBackendAdapter {
public:
    trtf::runtime::adapters::io::DetectionArtifact detect_image(
        const trtf::runtime::adapters::io::DecodedImage& image) override
    {
        ++calls;
        last_image_width = image.width;
        last_image_height = image.height;
        return next_artifact;
    }

    int calls{0};
    int32_t last_image_width{0};
    int32_t last_image_height{0};
    trtf::runtime::adapters::io::DetectionArtifact next_artifact{{{2, 0.75F, 1.0F, 2.0F, 3.0F, 4.0F}}};
};

class FakeNeuralBackendAdapter final : public trtf::runtime::services::common::INeuralOperatorBackendAdapter {
public:
    trtf::runtime::services::common::NeuralOperatorOutput solve(
        const float* branch_input,
        int32_t branch_len,
        const float* trunk_input,
        int32_t trunk_len) override
    {
        ++solve_calls;
        last_branch.assign(branch_input, branch_input + branch_len);
        last_trunk.assign(trunk_input, trunk_input + trunk_len);
        return solve_output;
    }

    trtf::runtime::services::common::NeuralOperatorOutput solve_field(
        const float* field_input,
        int32_t input_size) override
    {
        ++solve_field_calls;
        last_field.assign(field_input, field_input + input_size);
        return field_output;
    }

    int solve_calls{0};
    int solve_field_calls{0};
    std::vector<float> last_branch;
    std::vector<float> last_trunk;
    std::vector<float> last_field;
    trtf::runtime::services::common::NeuralOperatorOutput solve_output{{1.0F, 2.0F}, 2, 0, 0, 0};
    trtf::runtime::services::common::NeuralOperatorOutput field_output{{3.0F, 4.0F, 5.0F, 6.0F}, 4, 1, 2, 2};
};

class FakeDetectionBackend final : public trtf::DetectionBackend {
public:
    trtf::DetectionResult detect_image(const trtf::runtime::adapters::io::DecodedImage& image) override
    {
        ++calls;
        last_image_width = image.width;
        last_image_height = image.height;
        return next_result;
    }

    int calls{0};
    int32_t last_image_width{0};
    int32_t last_image_height{0};
    trtf::DetectionResult next_result{true, {{7, 0.8F, 1.0F, 2.0F, 3.0F, 4.0F}}};
};

class FakeNeuralBackend final : public trtf::NeuralOperatorBackend {
public:
    trtf::NeuralOperatorResult solve(
        const float* branch,
        int32_t branch_len,
        const float* trunk,
        int32_t trunk_len) override
    {
        ++solve_calls;
        last_branch.assign(branch, branch + branch_len);
        last_trunk.assign(trunk, trunk + trunk_len);
        return solve_result;
    }

    trtf::NeuralOperatorResult solve_field(
        const float* field,
        int32_t size) override
    {
        ++solve_field_calls;
        last_field.assign(field, field + size);
        return field_result;
    }

    int solve_calls{0};
    int solve_field_calls{0};
    std::vector<float> last_branch;
    std::vector<float> last_trunk;
    std::vector<float> last_field;
    trtf::NeuralOperatorResult solve_result{{8.0F, 9.0F}, 2, 0, 0, 0};
    trtf::NeuralOperatorResult field_result{{1.0F, 2.0F, 3.0F, 4.0F}, 4, 1, 2, 2};
};

void test_generation_backend_port_wraps_generation_backend()
{
    auto backend = std::make_unique<FakeGenerationBackend>();
    auto* backend_ptr = backend.get();
    trtf::runtime::services::common::GenerationBackendPort port(std::move(backend));

    trtf::GenerationConfig config;
    config.max_new_tokens = 6;

    const auto output = port.generate_text({5, 6}, config);
    check(output == std::vector<int32_t>({7, 8}), "generation port returns backend tokens");
    check(backend_ptr->generate_calls == 1, "generation port calls backend generate");
    check(backend_ptr->last_input_ids == std::vector<int32_t>({5, 6}), "generation port forwards input ids");
    check(backend_ptr->last_max_new_tokens == 6, "generation port forwards max_new_tokens");
    check(port.supports_vision(), "generation port mirrors supports_vision");

    std::vector<float> image_features;
    int32_t num_features = 0;
    int32_t feature_dim = 0;
    std::string error;
    check(!port.prepare_image(make_decoded_image(), image_features, num_features, feature_dim, error),
        "generation port returns false when backend lacks VL fast path");
    check(port.format_vision_prompt("caption") == "caption",
        "generation port keeps prompt unchanged without VL fast path");
    const auto fallback = port.generate_with_image({1, 2}, nullptr, 0, 0, config);
    check(fallback == std::vector<int32_t>({7, 8}), "generation port falls back to text generate without VL fast path");
    check(backend_ptr->generate_calls == 2, "image fallback reuses backend generate");
}

void test_text_generation_ports_delegate_custom_backend_adapters()
{
    auto generation_adapter = std::make_unique<FakeTextBackendAdapter>();
    auto* generation_ptr = generation_adapter.get();
    trtf::runtime::services::common::GenerationBackendPort generation_port(std::move(generation_adapter));

    trtf::GenerationConfig config;
    config.max_new_tokens = 9;
    std::vector<float> image_features;
    int32_t num_features = 0;
    int32_t feature_dim = 0;
    std::string error;

    check(generation_port.supports_vision(), "custom text adapter reports vision support");
    check(generation_port.prepare_image(make_decoded_image(), image_features, num_features, feature_dim, error),
        "custom text adapter prepares image");
    check(image_features == std::vector<float>({1.0F, 2.0F, 3.0F, 4.0F}), "custom text adapter returns image features");
    check(num_features == 2 && feature_dim == 2, "custom text adapter returns image feature shape");
    check(generation_port.format_vision_prompt("prompt") == "[vl] prompt",
        "custom text adapter formats vision prompt");
    const auto vision_output = generation_port.generate_with_image({4, 5}, image_features.data(), num_features, feature_dim, config);
    check(vision_output == std::vector<int32_t>({3, 4}), "custom text adapter returns image generation tokens");
    check(generation_ptr->generate_with_image_calls == 1, "custom text adapter receives image generation call");
    check(generation_ptr->captured_features == image_features, "custom text adapter receives flattened features");

    auto omni_adapter = std::make_unique<FakeTextBackendAdapter>();
    auto* omni_ptr = omni_adapter.get();
    trtf::runtime::services::common::OmniTextGenerationPort omni_port(std::move(omni_adapter));
    const auto omni_output = omni_port.generate_text({8, 9}, config);
    check(omni_output == std::vector<int32_t>({1, 2}), "omni text port returns adapter tokens");
    check(omni_ptr->generate_text_calls == 1, "omni text port delegates text generation");

    trtf::runtime::services::common::OmniTextGenerationPort null_port(
        std::unique_ptr<trtf::runtime::services::common::ITextGenerationBackendAdapter>{});
    check(null_port.generate_text({1}, config).empty(), "omni text port returns empty output for null adapter");
}

void test_audio_generation_ports_delegate_custom_adapters()
{
    auto bark_adapter = std::make_unique<FakeAudioBackendAdapter>();
    auto* bark_ptr = bark_adapter.get();
    trtf::runtime::services::common::BarkAudioGenerationPort bark_port(std::move(bark_adapter));
    const auto bark = bark_port.generate_audio({1, 2}, 5);
    check(bark.num_samples == 2 && bark.sample_rate == 24000, "bark audio port returns adapter artifact");
    check(bark_ptr->calls == 1 && bark_ptr->last_max_tokens == 5, "bark audio port forwards request");

    auto magpie_adapter = std::make_unique<FakeAudioBackendAdapter>();
    auto* magpie_ptr = magpie_adapter.get();
    trtf::runtime::services::common::MagpieAudioGenerationPort magpie_port(std::move(magpie_adapter));
    const auto magpie = magpie_port.generate_audio({3, 4}, 7);
    check(magpie.num_samples == 2, "magpie audio port returns adapter artifact");
    check(magpie_ptr->last_input_ids == std::vector<int32_t>({3, 4}), "magpie audio port forwards ids");

    auto omni_adapter = std::make_unique<FakeAudioBackendAdapter>();
    auto* omni_ptr = omni_adapter.get();
    trtf::runtime::services::common::OmniAudioGenerationPort omni_port(std::move(omni_adapter));
    const auto omni = omni_port.generate_audio({5, 6}, 8);
    check(omni.num_samples == 2, "omni audio port returns adapter artifact");
    check(omni_ptr->last_max_tokens == 8, "omni audio port forwards max tokens");

    trtf::runtime::services::common::BarkAudioGenerationPort null_port(
        std::unique_ptr<trtf::runtime::services::common::IAudioGenerationBackendAdapter>{});
    check(null_port.generate_audio({1}, 2).waveform.empty(), "audio port returns empty artifact for null adapter");
}

void test_transcription_and_speech_ports_delegate_custom_adapters()
{
    auto transcription_adapter = std::make_unique<FakeTranscriptionBackendAdapter>();
    auto* transcription_ptr = transcription_adapter.get();
    trtf::runtime::services::common::WhisperTranscriptionPort transcription_port(std::move(transcription_adapter));
    const float mel[]{0.1F, 0.2F, 0.3F, 0.4F};
    const auto transcription = transcription_port.transcribe(mel, 2, 2, 11);
    check(transcription.text == "decoded", "transcription port returns text");
    check(transcription.output_ids == std::vector<int32_t>({9, 10}), "transcription port returns token ids");
    check(transcription_ptr->calls == 1 && transcription_ptr->last_max_new_tokens == 11,
        "transcription port forwards mel request");

    auto speech_adapter = std::make_unique<FakeSpeechBackendAdapter>();
    auto* speech_ptr = speech_adapter.get();
    trtf::runtime::services::common::SpeechSynthesisPort speech_port(std::move(speech_adapter));
    const auto policy = speech_port.input_policy();
    check(policy.sample_rate == 16000 && policy.prefers_python_resampler,
        "speech port returns backend input policy");
    const float samples[]{0.5F, 0.25F};
    const auto speech = speech_port.process_audio(samples, 2, 13, 22050, 4);
    check(speech.num_samples == 2 && speech.sample_rate == 16000, "speech port returns audio artifact");
    check(speech_ptr->calls == 1, "speech port delegates process_audio");
    check(speech_ptr->last_input_sample_rate == 22050 && speech_ptr->last_tail_frames == 4,
        "speech port forwards sample rate and tail frames");

    trtf::runtime::services::common::SpeechSynthesisPort null_port(
        std::unique_ptr<trtf::runtime::services::common::ISpeechSynthesisBackendAdapter>{});
    check(null_port.process_audio(samples, 2, 1, 16000, 0).waveform.empty(),
        "speech port returns empty artifact for null adapter");
}

void test_vision_ports_delegate_custom_adapters()
{
    auto segmentation_adapter = std::make_unique<FakeSegmentationBackendAdapter>();
    auto* segmentation_ptr = segmentation_adapter.get();
    trtf::runtime::services::common::SegmentationBackendPort segmentation_port(std::move(segmentation_adapter));
    const auto segmentation = segmentation_port.segment_image(make_decoded_image());
    check(segmentation.width == 2 && segmentation.height == 2, "segmentation port returns artifact shape");
    check(segmentation_ptr->calls == 1, "segmentation port delegates image segmentation");

    auto sam_adapter = std::make_unique<FakeSamBackendAdapter>();
    auto* sam_ptr = sam_adapter.get();
    trtf::runtime::services::common::SamBackendPort sam_port(std::move(sam_adapter));
    check(sam_port.encode_image(make_decoded_image()), "sam port delegates image encode");
    const auto sam_masks = sam_port.segment_point(0.25F, 0.75F, false);
    check(sam_masks.num_masks == 1, "sam port returns mask artifact");
    check(sam_ptr->segment_calls == 1 && !sam_ptr->last_is_foreground,
        "sam port forwards point segmentation request");

    auto detection_adapter = std::make_unique<FakeDetectionBackendAdapter>();
    auto* detection_ptr = detection_adapter.get();
    trtf::runtime::services::common::DetectionBackendPort detection_port(std::move(detection_adapter));
    const auto detection = detection_port.detect_image(make_decoded_image());
    check(detection.detections.size() == 1, "detection port returns detection artifact");
    check(detection_ptr->calls == 1, "detection port delegates image detection");

    auto neural_adapter = std::make_unique<FakeNeuralBackendAdapter>();
    auto* neural_ptr = neural_adapter.get();
    trtf::runtime::services::common::NeuralOperatorBackendPort neural_port(std::move(neural_adapter));
    const float branch[]{1.0F, 2.0F};
    const float trunk[]{3.0F, 4.0F, 5.0F};
    const auto solve = neural_port.solve(branch, 2, trunk, 3);
    check(solve.output_dim == 2 && solve.output == std::vector<float>({1.0F, 2.0F}),
        "neural port returns solve output");
    const float field[]{6.0F, 7.0F, 8.0F, 9.0F};
    const auto solve_field = neural_port.solve_field(field, 4);
    check(solve_field.out_channels == 1 && solve_field.height == 2 && solve_field.width == 2,
        "neural port returns field output shape");
    check(neural_ptr->solve_calls == 1 && neural_ptr->solve_field_calls == 1,
        "neural port delegates both solve paths");

    trtf::runtime::services::common::DetectionBackendPort null_detection(
        std::unique_ptr<trtf::runtime::services::common::IDetectionBackendAdapter>{});
    check(null_detection.detect_image(make_decoded_image()).detections.empty(),
        "detection port returns empty artifact for null adapter");
}

void test_concrete_runtime_ports_handle_null_backends()
{
    trtf::GenerationConfig config;
    config.max_new_tokens = 4;
    const float audio_samples[]{0.1F, 0.2F};

    trtf::runtime::services::common::BarkAudioGenerationPort bark_port(std::unique_ptr<trtf::BarkBackend>{});
    check(bark_port.generate_audio({1, 2}, 3).waveform.empty(),
        "bark concrete port handles null backend");

    trtf::runtime::services::common::MagpieAudioGenerationPort magpie_port(std::unique_ptr<trtf::MagpieTTSBackend>{});
    check(magpie_port.generate_audio({1, 2}, 3).waveform.empty(),
        "magpie concrete port handles null backend");

    trtf::runtime::services::common::OmniTextGenerationPort omni_text_port(std::shared_ptr<trtf::OmniBackend>{});
    check(omni_text_port.generate_text({1, 2}, config).empty(),
        "omni concrete text port handles null backend");

    trtf::runtime::services::common::OmniAudioGenerationPort omni_audio_port(std::shared_ptr<trtf::OmniBackend>{});
    check(omni_audio_port.generate_audio({1, 2}, 3).waveform.empty(),
        "omni concrete audio port handles null backend");

    trtf::runtime::services::common::WhisperTranscriptionPort whisper_port(std::unique_ptr<trtf::WhisperBackend>{});
    check(whisper_port.transcribe(audio_samples, 1, 2, 5).text.empty(),
        "whisper concrete port handles null backend");

    trtf::runtime::services::common::SpeechSynthesisPort speech_port(std::unique_ptr<trtf::SpeechToSpeechBackend>{});
    check(speech_port.input_policy().sample_rate == 0,
        "speech concrete port returns default policy for null backend");
    check(speech_port.process_audio(audio_samples, 2, 4, 16000, 0).waveform.empty(),
        "speech concrete port handles null backend");

    trtf::runtime::services::common::SegmentationBackendPort segmentation_port(
        std::unique_ptr<trtf::SegmentationBackend>{});
    check(segmentation_port.segment_image(make_decoded_image()).class_map.empty(),
        "segmentation concrete port handles null backend");

    trtf::runtime::services::common::SamBackendPort sam_port(std::unique_ptr<trtf::SamBackend>{});
    check(!sam_port.encode_image(make_decoded_image()),
        "sam concrete port encode returns false for null backend");
    check(sam_port.segment_point(0.2F, 0.3F, true).masks.empty(),
        "sam concrete port handles null backend");
}

void test_concrete_detection_and_neural_ports_convert_backend_results()
{
    auto detection_backend = std::make_unique<FakeDetectionBackend>();
    auto* detection_ptr = detection_backend.get();
    trtf::runtime::services::common::DetectionBackendPort detection_port(std::move(detection_backend));
    const auto detection = detection_port.detect_image(make_decoded_image());
    check(detection.detections.size() == 1, "concrete detection port converts detection result");
    check(detection.detections[0].class_id == 7, "concrete detection port preserves class id");
    check(detection_ptr->calls == 1 && detection_ptr->last_image_width == 2,
        "concrete detection port forwards decoded image");

    auto neural_backend = std::make_unique<FakeNeuralBackend>();
    auto* neural_ptr = neural_backend.get();
    trtf::runtime::services::common::NeuralOperatorBackendPort neural_port(std::move(neural_backend));
    const float branch[]{1.0F, 2.0F};
    const float trunk[]{3.0F, 4.0F};
    const auto solve = neural_port.solve(branch, 2, trunk, 2);
    check(solve.output == std::vector<float>({8.0F, 9.0F}),
        "concrete neural port converts solve output");
    const float field[]{5.0F, 6.0F, 7.0F, 8.0F};
    const auto solve_field = neural_port.solve_field(field, 4);
    check(solve_field.out_channels == 1 && solve_field.height == 2 && solve_field.width == 2,
        "concrete neural port converts field output");
    check(neural_ptr->solve_calls == 1 && neural_ptr->solve_field_calls == 1,
        "concrete neural port delegates both calls");
}

void test_default_runtime_helper_ports_use_runtime_helpers()
{
    trtf::runtime::services::common::DefaultAudioResampler resampler;
    const float resample_src[]{0.0F, 1.0F, 0.0F, -1.0F, 0.0F, 1.0F, 0.0F, -1.0F};
    const auto half_rate = resampler.resample(resample_src, 8, 16000, 8000);
    check(half_rate.size() == 4, "default resampler halves sample count");
    const float identity_src[]{0.1F, 0.2F, 0.3F};
    const auto identity = resampler.resample(identity_src, 3, 16000, 16000);
    check(identity.size() == 3 && std::abs(identity[1] - 0.2F) < 1e-6F,
        "default resampler preserves identical sample rate");

    trtf::runtime::services::common::DefaultMelSpectrogramPort mel_port;
    trtf::runtime::services::common::MelSpectrogramConfig mel_config;
    mel_config.filterbank = {
        1.0F, 0.0F,
        0.0F, 1.0F,
        0.5F, 0.5F,
    };
    mel_config.n_freq_bins = 3;
    mel_config.n_mel_bins = 2;
    mel_config.n_fft = 4;
    mel_config.hop_length = 2;
    mel_config.chunk_length_s = 1;
    mel_config.sample_rate = 16;
    const float silence[16] = {};
    const auto mel = mel_port.extract(silence, 16, mel_config);
    check(mel.n_mels == 2, "default mel port returns configured mel bins");
    check(mel.n_frames == 8, "default mel port returns expected frame count");
    check(static_cast<int32_t>(mel.data.size()) == 16, "default mel port returns full mel buffer");
}

} // namespace

int main()
{
    test_generation_backend_port_wraps_generation_backend();
    test_text_generation_ports_delegate_custom_backend_adapters();
    test_audio_generation_ports_delegate_custom_adapters();
    test_transcription_and_speech_ports_delegate_custom_adapters();
    test_vision_ports_delegate_custom_adapters();
    test_concrete_runtime_ports_handle_null_backends();
    test_concrete_detection_and_neural_ports_convert_backend_results();
    test_default_runtime_helper_ports_use_runtime_helpers();

    if (g_failures != 0)
    {
        std::cerr << g_failures << " test(s) failed\n";
        return 1;
    }
    return 0;
}
