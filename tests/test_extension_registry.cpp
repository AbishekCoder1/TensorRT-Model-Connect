// Test: Custom extension points for model resolution and runtime assembly.
// Verifies: RegisterTextGenerationModelResolver and RegisterTextGenerationRuntimeAssembler
// with a mock backend produce expected pipeline output.

#include "trtf/model_resolver.h"
#include "trtf/pipeline.h"
#include "trtf/runtime_factory.h"

#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace {

class MockTextBackend final : public trtf::IGenerationBackend {
public:
    bool is_available() const override
    {
        return true;
    }

    const char* name() const override
    {
        return "mock-custom";
    }

    std::vector<int32_t> generate(const std::vector<int32_t>& input_ids, const trtf::GenerationConfig& config) override
    {
        (void) config;
        return input_ids;
    }

    bool supports_text_generation() const override
    {
        return true;
    }

    std::string generate_text(const std::string& prompt, const trtf::GenerationConfig& config) override
    {
        (void) config;
        return prompt + " ::mock::";
    }
};

} // namespace

int main()
{
    trtf::RegisterTextGenerationModelResolver([](const std::string& model_id) -> std::optional<trtf::ResolvedModelSpec> {
        if (model_id != "mock/custom-v1")
        {
            return std::nullopt;
        }

        trtf::ResolvedModelSpec spec;
        spec.model_id = model_id;
        spec.kind = trtf::ResolvedModelKind::kCustom;
        spec.custom_type = "mock";
        return spec;
    });

    trtf::RegisterTextGenerationRuntimeAssembler(
        [](const trtf::ResolvedModelSpec& model_spec, const trtf::BackendSelection& selection)
            -> std::optional<trtf::RuntimeAssembly> {
            (void) selection;
            if (model_spec.kind != trtf::ResolvedModelKind::kCustom || model_spec.custom_type != "mock")
            {
                return std::nullopt;
            }

            trtf::RuntimeAssembly assembly;
            assembly.backend = std::make_unique<MockTextBackend>();
            assembly.backend_name = "mock-custom";
            return assembly;
        });

    auto pipeline = trtf::Pipeline::CreateTextGeneration("mock/custom-v1");
    if (pipeline.backend_name() != "mock-custom")
    {
        std::cerr << "unexpected backend for custom extension path: " << pipeline.backend_name() << std::endl;
        return 1;
    }

    const auto out = pipeline("hello");
    if (out.size() != 1 || out[0].generated_text != "hello ::mock::")
    {
        std::cerr << "unexpected output for custom extension path" << std::endl;
        return 1;
    }

    std::cout << "test_extension_registry passed" << std::endl;
    return 0;
}
