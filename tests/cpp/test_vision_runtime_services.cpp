#include "runtime/services/vision/vision_runtime_services.h"

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

class FakeSegmentationPort final : public trtf::runtime::services::common::ISegmentationPort {
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

class FakeSamPort final : public trtf::runtime::services::common::IPromptedSegmentationPort {
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

class FakeDetectionPort final : public trtf::runtime::services::common::IDetectionPort {
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
    trtf::runtime::adapters::io::DetectionArtifact next_artifact{{{1, 0.95F, 0.0F, 1.0F, 2.0F, 3.0F}}};
};

class FakeNeuralPort final : public trtf::runtime::services::common::INeuralOperatorPort {
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

void test_segmentation_service_returns_artifact()
{
    auto port = std::make_unique<FakeSegmentationPort>();
    auto* port_ptr = port.get();
    trtf::runtime::services::vision::SegmentationService service(std::move(port));

    const auto result = service.segment({make_decoded_image()});
    check(result.ok(), "segmentation service returns success result");
    check(port_ptr->calls == 1, "segmentation service calls backend port");
    check(port_ptr->last_image_width == 2 && port_ptr->last_image_height == 2,
        "segmentation service forwards decoded image");
    check(result.value.width == 2 && result.value.height == 2, "segmentation service returns artifact shape");
}

void test_prompted_segmentation_service_handles_encode_failure_and_success()
{
    auto failing_port = std::make_unique<FakeSamPort>();
    auto* failing_port_ptr = failing_port.get();
    failing_port_ptr->encode_success = false;
    trtf::runtime::services::vision::PromptedSegmentationService failing_service(std::move(failing_port));
    check(!failing_service.segment({make_decoded_image()}).ok(), "prompted segmentation plain segment remains unsupported");
    check(failing_service.supports_prompted(), "prompted segmentation advertises support");
    check(!failing_service.segment_prompt({make_decoded_image(), 0.1F, 0.2F, true}).ok(),
        "prompted segmentation returns error when encode fails");

    auto success_port = std::make_unique<FakeSamPort>();
    auto* success_port_ptr = success_port.get();
    trtf::runtime::services::vision::PromptedSegmentationService success_service(std::move(success_port));
    const auto success = success_service.segment_prompt({make_decoded_image(), 0.3F, 0.4F, false});
    check(success.ok(), "prompted segmentation returns success result");
    check(success_port_ptr->segment_calls == 1, "prompted segmentation runs point segmentation");
    check(success_port_ptr->last_point_x == 0.3F && success_port_ptr->last_point_y == 0.4F,
        "prompted segmentation forwards point coordinates");
    check(!success_port_ptr->last_is_foreground, "prompted segmentation forwards foreground flag");
    check(success.value.num_masks == 1, "prompted segmentation returns mask artifact");
}

void test_detection_service_and_neural_operator_service()
{
    auto detection_port = std::make_unique<FakeDetectionPort>();
    auto* detection_port_ptr = detection_port.get();
    trtf::runtime::services::vision::DetectionService detection_service(std::move(detection_port));
    const auto detection = detection_service.detect({make_decoded_image(), 0.6F});
    check(detection.ok(), "detection service returns success result");
    check(detection_port_ptr->calls == 1, "detection service calls detection port");
    check(detection.value.detections.size() == 1, "detection service returns detection artifact");

    auto neural_port = std::make_unique<FakeNeuralPort>();
    auto* neural_port_ptr = neural_port.get();
    trtf::runtime::services::vision::NeuralOperatorService neural_service(std::move(neural_port));
    const float branch[]{1.0F, 2.0F};
    const float trunk[]{3.0F, 4.0F, 5.0F};
    int32_t out_dim = 0;
    const float* solve_output = neural_service.solve(branch, 2, trunk, 3, &out_dim);
    check(solve_output != nullptr && out_dim == 2, "neural operator solve returns output and dimension");
    check(neural_port_ptr->solve_calls == 1, "neural operator solve calls backend port");

    const float field[]{6.0F, 7.0F, 8.0F, 9.0F};
    int32_t out_channels = 0;
    int32_t out_h = 0;
    int32_t out_w = 0;
    const float* field_output = neural_service.solve_field(field, 4, &out_channels, &out_h, &out_w);
    check(field_output != nullptr, "neural operator solve_field returns output");
    check(out_channels == 1 && out_h == 2 && out_w == 2, "neural operator solve_field returns output shape");
    check(neural_port_ptr->solve_field_calls == 1, "neural operator solve_field calls backend port");
}

} // namespace

int main()
{
    test_segmentation_service_returns_artifact();
    test_prompted_segmentation_service_handles_encode_failure_and_success();
    test_detection_service_and_neural_operator_service();

    if (g_failures != 0)
    {
        std::cerr << g_failures << " test(s) failed\n";
        return 1;
    }
    return 0;
}
