#include "trtf/runtime_factory.h"

#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace trtf {
namespace {

std::mutex& custom_assemblers_mutex()
{
    static std::mutex m;
    return m;
}

std::vector<TextGenerationRuntimeAssembler>& custom_assemblers()
{
    static std::vector<TextGenerationRuntimeAssembler> assemblers;
    return assemblers;
}

} // namespace

void RegisterTextGenerationRuntimeAssembler(TextGenerationRuntimeAssembler assembler)
{
    if (!assembler)
    {
        throw std::invalid_argument("RegisterTextGenerationRuntimeAssembler requires a valid assembler.");
    }

    std::lock_guard<std::mutex> lock(custom_assemblers_mutex());
    custom_assemblers().push_back(std::move(assembler));
}

RuntimeAssembly BuildRuntimeForTextGeneration(const ResolvedModelSpec& model_spec, const BackendSelection& selection)
{
    if (selection.force_trt && !selection.prefer_trt)
    {
        throw std::invalid_argument("force_trt requires prefer_trt=true.");
    }

    std::vector<TextGenerationRuntimeAssembler> assemblers_snapshot;
    {
        std::lock_guard<std::mutex> lock(custom_assemblers_mutex());
        assemblers_snapshot = custom_assemblers();
    }

    for (const auto& assembler : assemblers_snapshot)
    {
        std::optional<RuntimeAssembly> assembly = assembler(model_spec, selection);
        if (assembly.has_value())
        {
            if (!assembly->backend)
            {
                throw std::runtime_error("Custom runtime assembler returned no backend.");
            }
            if (assembly->backend_name.empty())
            {
                assembly->backend_name = assembly->backend->name();
            }
            return std::move(*assembly);
        }
    }

    RuntimeAssembly assembly;

    switch (model_spec.kind)
    {
    case ResolvedModelKind::kHuggingFaceLocal:
    {
        if (selection.force_trt)
        {
            throw std::runtime_error("force_trt is not supported for raw Hugging Face model directories.");
        }

        auto hf_backend = CreateHfPythonBackend(model_spec.huggingface_model_dir);
        if (!hf_backend || !hf_backend->is_available())
        {
            throw std::runtime_error(
                "Hugging Face backend is unavailable for model directory: " + model_spec.huggingface_model_dir);
        }
        assembly.backend_name = hf_backend->name();
        assembly.backend = std::move(hf_backend);
        return assembly;
    }

    case ResolvedModelKind::kDecoderDefinition:
    {
        const DecoderModel& model = model_spec.decoder_model;
        if (model.prefer_hf_tokenizer && !model.hf_tokenizer_dir.empty())
        {
            try
            {
                assembly.tokenizer = CreateHfPythonTokenizer(model.hf_tokenizer_dir);
            }
            catch (const std::exception& e)
            {
                if (selection.force_trt)
                {
                    throw std::runtime_error("Failed to initialize HF tokenizer for TRT path: " + std::string(e.what()));
                }
            }
        }

        if (!assembly.tokenizer)
        {
            assembly.tokenizer = CreateVocabTokenizer(model.vocab);
        }

        std::unique_ptr<IGenerationBackend> trt_backend;
        if (selection.prefer_trt)
        {
            trt_backend = CreateTrtBackend(*assembly.tokenizer, model);
            if (trt_backend && trt_backend->is_available())
            {
                assembly.backend_name = trt_backend->name();
                assembly.backend = std::move(trt_backend);
            }
        }

        if (!assembly.backend && selection.force_trt)
        {
            std::string reason;
            if (trt_backend)
            {
                const char* unavailable = trt_backend->unavailable_reason();
                if (unavailable != nullptr)
                {
                    reason = unavailable;
                }
            }

            if (!reason.empty())
            {
                throw std::runtime_error(
                    "TRT backend requested with force_trt=true but is unavailable: " + reason);
            }
            throw std::runtime_error("TRT backend requested with force_trt=true but is unavailable.");
        }

        if (!assembly.backend)
        {
            auto cpu_backend = CreateCpuReferenceBackend(*assembly.tokenizer, model);
            if (!cpu_backend || !cpu_backend->is_available())
            {
                throw std::runtime_error("No available generation backend.");
            }
            assembly.backend_name = cpu_backend->name();
            assembly.backend = std::move(cpu_backend);
        }

        return assembly;
    }

    case ResolvedModelKind::kCustom:
        break;
    }

    throw std::runtime_error("Unsupported resolved model kind (no runtime assembler registered).");
}

} // namespace trtf
