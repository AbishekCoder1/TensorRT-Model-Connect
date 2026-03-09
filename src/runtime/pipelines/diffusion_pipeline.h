#pragma once

// Diffusion pipelines: FluxPipeline, WanPipeline, ZImagePipeline.
// Each holds an old-style IDiffusionBackend and delegates inference to it.

#include "trtf/pipeline.h"
#include "trtf/tokenizer.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#if TRTF_HAS_TRT

// Forward-declare old diffusion backend to avoid pulling in heavy headers.
namespace trtf {
class IDiffusionBackend;
} // namespace trtf

namespace trtf {

class FluxPipeline final : public IPipeline {
public:
    FluxPipeline(
        std::unique_ptr<IDiffusionBackend> backend,
        std::shared_ptr<ITokenizer> tokenizer,
        int32_t image_height,
        int32_t image_width,
        std::string model_id_str = "");

    ~FluxPipeline() override;

    ImageResult generate_image(const std::string& prompt, const GenerateConfig& cfg = {}) override;

    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "FluxPipeline"; }

private:
    std::unique_ptr<IDiffusionBackend> backend_;
    std::shared_ptr<ITokenizer> tokenizer_;
    int32_t image_height_;
    int32_t image_width_;
    std::string model_id_;
};

class WanPipeline final : public IPipeline {
public:
    WanPipeline(
        std::unique_ptr<IDiffusionBackend> backend,
        std::shared_ptr<ITokenizer> tokenizer,
        int32_t image_height,
        int32_t image_width,
        std::string model_id_str = "");

    ~WanPipeline() override;

    ImageResult generate_image(const std::string& prompt, const GenerateConfig& cfg = {}) override;

    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "WanPipeline"; }

private:
    std::unique_ptr<IDiffusionBackend> backend_;
    std::shared_ptr<ITokenizer> tokenizer_;
    int32_t image_height_;
    int32_t image_width_;
    std::string model_id_;
};

class ZImagePipeline final : public IPipeline {
public:
    ZImagePipeline(
        std::unique_ptr<IDiffusionBackend> backend,
        std::shared_ptr<ITokenizer> tokenizer,
        int32_t image_height,
        int32_t image_width,
        std::string model_id_str = "");

    ~ZImagePipeline() override;

    ImageResult generate_image(const std::string& prompt, const GenerateConfig& cfg = {}) override;

    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "ZImagePipeline"; }

private:
    std::unique_ptr<IDiffusionBackend> backend_;
    std::shared_ptr<ITokenizer> tokenizer_;
    int32_t image_height_;
    int32_t image_width_;
    std::string model_id_;
};

} // namespace trtf

#endif // TRTF_HAS_TRT
