#include "trtf/runtime_factory.h"

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace trtf {

RuntimeAssembly BuildRuntimeForTextGeneration(const ResolvedModelSpec& model_spec, const BackendSelection& selection)
{
    if (selection.force_trt && !selection.prefer_trt)
    {
        throw std::invalid_argument("force_trt requires prefer_trt=true.");
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

        auto hf_backend = CreateHfPythonBackend(model_spec.huggingface_model_dir, selection.hf_python);
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
                auto ttok0 = std::chrono::steady_clock::now();
                std::cerr << "[trtf] Initializing HF tokenizer ..." << std::endl;
                assembly.tokenizer = CreateHfPythonTokenizer(model.hf_tokenizer_dir, selection.hf_python);
                auto ttok1 = std::chrono::steady_clock::now();
                std::cerr << "[trtf] Tokenizer ready ["
                          << std::chrono::duration_cast<std::chrono::milliseconds>(ttok1 - ttok0).count()
                          << " ms]" << std::endl;
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
            if (!model.hf_tokenizer_dir.empty())
            {
                auto hf_backend = CreateHfPythonBackend(model.hf_tokenizer_dir, selection.hf_python);
                if (hf_backend && hf_backend->is_available())
                {
                    assembly.backend_name = hf_backend->name();
                    assembly.backend = std::move(hf_backend);
                }
            }
            if (!assembly.backend)
            {
                throw std::runtime_error("No available generation backend.");
            }
        }

        return assembly;
    }
    }

    throw std::runtime_error("Unsupported resolved model kind.");
}

} // namespace trtf
