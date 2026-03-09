#include "runtime/pipelines/encoder_pipeline.h"

#if TRTF_HAS_TRT

#include <cmath>
#include <cstring>
#include <stdexcept>

namespace trtf {

namespace {

// Infer the hidden dimension from the last axis of the first output tensor.
int32_t infer_output_hidden_dim(const TrtModule& module)
{
    for (const auto& info : module.output_info())
        if (!info.shape.empty())
            return static_cast<int32_t>(info.shape.back());
    return 0;
}

// Mean-pool [seq_len, hidden] over the first actual_len positions,
// then L2-normalize. Returns the pooled vector of size hidden.
std::vector<float> mean_pool_and_normalize(
    const float* data, int32_t actual_len, int32_t hidden)
{
    std::vector<float> pooled(static_cast<std::size_t>(hidden), 0.0f);
    const float inv_len = 1.0f / static_cast<float>(actual_len);
    for (int32_t s = 0; s < actual_len; ++s)
        for (int32_t h = 0; h < hidden; ++h)
            pooled[h] += data[static_cast<std::size_t>(s) * hidden + h];
    for (int32_t h = 0; h < hidden; ++h)
        pooled[h] *= inv_len;

    float norm = 0.0f;
    for (int32_t h = 0; h < hidden; ++h)
        norm += pooled[h] * pooled[h];
    norm = std::sqrt(norm);
    if (norm > 1e-12f)
        for (int32_t h = 0; h < hidden; ++h)
            pooled[h] /= norm;

    return pooled;
}

// Check whether the engine's attention_mask input expects int32.
bool engine_mask_is_int32(const TrtModule& module)
{
    for (const auto& info : module.input_info())
        if (info.name == "attention_mask")
            return info.dtype == DType::kInt32;
    return false;
}

} // namespace

// ─── EncoderPipeline ───

EncoderPipeline::EncoderPipeline(
    std::unique_ptr<TrtModule> encoder, std::string mode,
    std::shared_ptr<ITokenizer> tokenizer,
    std::string model_id_str)
    : encoder_(std::move(encoder))
    , mode_(std::move(mode))
    , tokenizer_(std::move(tokenizer))
    , model_id_(std::move(model_id_str))
{
    if (!encoder_ || !encoder_->ok())
        throw std::runtime_error("EncoderPipeline: invalid encoder module");
}

EmbeddingResult EncoderPipeline::embed(const std::string& text)
{
    if (!tokenizer_)
        throw std::runtime_error("EncoderPipeline: no tokenizer configured");
    auto ids = tokenizer_->encode(text);
    auto raw = encode_ids(ids);

    // For embedding models: the TRT engine returns [max_seq, hidden] hidden
    // states. Mean-pool over actual input positions and L2-normalize.
    if (mode_ != "embedding" || raw.data.empty())
        return raw;

    const auto actual_len = static_cast<int32_t>(ids.size());
    const int32_t hidden = infer_output_hidden_dim(*encoder_);
    if (hidden <= 0 || actual_len <= 0 || raw.dim < actual_len * hidden)
        return raw;

    auto pooled = mean_pool_and_normalize(raw.data.data(), actual_len, hidden);
    raw.data = std::move(pooled);
    raw.dim = hidden;
    return raw;
}

EmbeddingResult EncoderPipeline::encode(const std::string& text)
{
    if (!tokenizer_)
        throw std::runtime_error("EncoderPipeline: no tokenizer configured");
    auto ids = tokenizer_->encode(text);
    auto raw = encode_ids(ids);

    // Extract CLS token (first hidden_dim values) from the full hidden state
    // matrix [max_seq, hidden]. This matches HF model.encode() behavior for
    // encoder-only models (BERT, RoBERTa, etc.).
    const int32_t hidden = infer_output_hidden_dim(*encoder_);
    if (hidden > 0 && raw.dim > hidden)
    {
        raw.data.resize(static_cast<std::size_t>(hidden));
        raw.dim = hidden;
    }
    return raw;
}

float EncoderPipeline::rerank(const std::string& query, const std::string& document)
{
    if (!tokenizer_)
        throw std::runtime_error("EncoderPipeline: no tokenizer configured");
    std::string combined = query + " [SEP] " + document;
    auto ids = tokenizer_->encode(combined);
    auto result = encode_ids(ids);
    return result.data.empty() ? 0.0f : result.data[0];
}

EmbeddingResult EncoderPipeline::encode_ids(const std::vector<int32_t>& input_ids)
{
    const auto n = input_ids.size();
    std::vector<int32_t> mask_i32(n, 1);
    std::vector<float> mask_f32(n, 1.0f);

    auto ids_copy = input_ids;
    Tensor ids_t;
    ids_t.data = ids_copy.data();
    ids_t.shape = {static_cast<int64_t>(n)};
    ids_t.dtype = DType::kInt32;

    // Match the engine's expected dtype for the attention mask.
    Tensor mask_t;
    if (engine_mask_is_int32(*encoder_))
    {
        mask_t.data = mask_i32.data();
        mask_t.shape = {static_cast<int64_t>(n)};
        mask_t.dtype = DType::kInt32;
    }
    else
    {
        mask_t.data = mask_f32.data();
        mask_t.shape = {static_cast<int64_t>(n)};
        mask_t.dtype = DType::kFloat32;
    }

    TensorMap inputs;
    inputs["input_ids"] = ids_t;
    inputs["attention_mask"] = mask_t;

    auto outputs = encoder_->forward(inputs);

    EmbeddingResult result;
    for (auto& [name, tensor] : outputs)
    {
        if (name.find("logits") != std::string::npos ||
            name.find("embed") != std::string::npos ||
            name.find("output") != std::string::npos ||
            name.find("hidden") != std::string::npos ||
            name.find("score") != std::string::npos)
        {
            auto n = tensor.numel();
            result.data.resize(static_cast<std::size_t>(n));
            std::memcpy(result.data.data(), tensor.data, n * sizeof(float));
            result.dim = static_cast<int32_t>(n);
            break;
        }
    }

    return result;
}

// ─── Segmentation helpers ───

namespace {

// Argmax over class dimension: logits[C, H, W] -> class_map[H, W].
void argmax_class_map(
    const float* logits, int32_t num_classes,
    int32_t out_h, int32_t out_w,
    std::vector<int32_t>& class_map)
{
    const auto plane_size = static_cast<std::size_t>(out_h) * out_w;
    class_map.resize(plane_size);

    for (std::size_t px = 0; px < plane_size; ++px)
    {
        int32_t best_class = 0;
        float best_val = -1e30F;
        for (int32_t c = 0; c < num_classes; ++c)
        {
            const float val = logits[static_cast<std::size_t>(c) * plane_size + px];
            if (val > best_val)
            {
                best_val = val;
                best_class = c;
            }
        }
        class_map[px] = best_class;
    }
}

// Extract (num_classes, H, W) from output shape, handling optional batch dim.
bool parse_segmentation_shape(
    const std::vector<int64_t>& shape,
    int32_t& num_classes, int32_t& out_h, int32_t& out_w)
{
    if (shape.size() == 4)
    {
        num_classes = static_cast<int32_t>(shape[1]);
        out_h = static_cast<int32_t>(shape[2]);
        out_w = static_cast<int32_t>(shape[3]);
    }
    else if (shape.size() == 3)
    {
        num_classes = static_cast<int32_t>(shape[0]);
        out_h = static_cast<int32_t>(shape[1]);
        out_w = static_cast<int32_t>(shape[2]);
    }
    else
    {
        return false;
    }
    return num_classes > 1 && out_h > 0 && out_w > 0;
}

// Find the logits/output tensor from the model output map.
const Tensor* find_segmentation_output(const TensorMap& outputs)
{
    for (const auto& [name, tensor] : outputs)
    {
        if (name.find("logits") != std::string::npos ||
            name.find("output") != std::string::npos ||
            outputs.size() == 1)
            return &tensor;
    }
    return nullptr;
}

} // namespace

// ─── SegmentPipeline ───

SegmentPipeline::SegmentPipeline(std::unique_ptr<TrtModule> model, std::string model_id_str)
    : model_(std::move(model)), model_id_(std::move(model_id_str))
{
    if (!model_ || !model_->ok())
        throw std::runtime_error("SegmentPipeline: invalid model");
}

SegmentResult SegmentPipeline::segment(
    const float* pixels, int32_t height, int32_t width)
{
    Tensor img_t;
    img_t.data = const_cast<float*>(pixels);
    img_t.shape = {3, height, width};
    img_t.dtype = DType::kFloat32;

    auto outputs = model_->forward({{"pixel_values", img_t}});
    SegmentResult result;

    const Tensor* out_tensor = find_segmentation_output(outputs);
    if (!out_tensor) return result;

    const auto* data = static_cast<const float*>(out_tensor->data);
    int32_t num_classes = 0, out_h = 0, out_w = 0;

    if (parse_segmentation_shape(out_tensor->shape, num_classes, out_h, out_w))
    {
        result.height = out_h;
        result.width = out_w;
        argmax_class_map(data, num_classes, out_h, out_w, result.mask);
    }
    else
    {
        auto n = out_tensor->numel();
        result.height = height;
        result.width = width;
        result.mask.resize(static_cast<std::size_t>(n));
        for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i)
            result.mask[i] = static_cast<int32_t>(data[i]);
    }

    return result;
}

// ─── SamPipeline ───

SamPipeline::SamPipeline(
    std::unique_ptr<TrtModule> image_encoder,
    std::unique_ptr<TrtModule> mask_decoder,
    std::string model_id_str)
    : image_encoder_(std::move(image_encoder))
    , mask_decoder_(std::move(mask_decoder))
    , model_id_(std::move(model_id_str))
{
    if (!image_encoder_ || !image_encoder_->ok())
        throw std::runtime_error("SamPipeline: invalid image_encoder");
    if (!mask_decoder_ || !mask_decoder_->ok())
        throw std::runtime_error("SamPipeline: invalid mask_decoder");
}

SegmentResult SamPipeline::segment(
    const float* pixels, int32_t height, int32_t width)
{
    Tensor img_t;
    img_t.data = const_cast<float*>(pixels);
    img_t.shape = {3, height, width};
    img_t.dtype = DType::kFloat32;

    auto enc_out = image_encoder_->forward({{"pixel_values", img_t}});

    TensorMap decoder_inputs;
    for (auto& [name, tensor] : enc_out)
        decoder_inputs[name] = tensor;

    auto dec_out = mask_decoder_->forward(decoder_inputs);

    SegmentResult result;
    result.height = height;
    result.width = width;

    for (auto& [name, tensor] : dec_out)
    {
        if (name.find("mask") != std::string::npos || name.find("output") != std::string::npos)
        {
            auto n = tensor.numel();
            result.mask.resize(static_cast<std::size_t>(n));
            const auto* data = static_cast<const float*>(tensor.data);
            for (int64_t i = 0; i < n; ++i)
                result.mask[static_cast<std::size_t>(i)] = static_cast<int32_t>(data[i]);
            break;
        }
    }

    return result;
}

} // namespace trtf

#endif // TRTF_HAS_TRT
