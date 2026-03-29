#include "runtime/pipelines/vl_pipeline.h"

#if TRTF_HAS_TRT

#include "runtime/domains/multimodal/image_preprocessor.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace trtf {

VLPipeline::VLPipeline(
    std::unique_ptr<TrtModule> text_decoder,
    std::unique_ptr<TrtModule> vision_encoder,
    std::unique_ptr<KvCache> cache,
    VLConfig config,
    VLPreprocessConfig vl_preprocess,
    cudaStream_t stream,
    std::shared_ptr<ITokenizer> tokenizer,
    std::string model_id_str)
    : text_decoder_(std::move(text_decoder))
    , vision_encoder_(std::move(vision_encoder))
    , cache_(std::move(cache))
    , config_(config)
    , vl_preprocess_(std::move(vl_preprocess))
    , stream_(stream)
    , tokenizer_(std::move(tokenizer))
    , model_id_(std::move(model_id_str))
{
    if (!text_decoder_ || !text_decoder_->ok())
        throw std::runtime_error("VLPipeline: invalid text decoder");
    if (!cache_ || !cache_->ok())
        throw std::runtime_error("VLPipeline: invalid KvCache");

    // Sync image_token_id from VLPreprocessConfig if not set in VLConfig
    if (config_.image_token_id < 0 && vl_preprocess_.image_token_id >= 0)
        config_.image_token_id = vl_preprocess_.image_token_id;
    if (config_.vision_output_dim <= 0 && vl_preprocess_.vision_output_dim > 0)
        config_.vision_output_dim = vl_preprocess_.vision_output_dim;
}

TextResult VLPipeline::generate(
    const std::string& prompt, const GenerateConfig& cfg)
{
    if (!tokenizer_)
        throw std::runtime_error("VLPipeline: no tokenizer configured");

    auto input_ids = tokenizer_->encode(prompt);
    auto [max_new, eos] = resolve_gen_limits(cfg);
    auto output_ids = generate_from_ids(input_ids, max_new, eos);

    std::vector<int32_t> new_tokens(
        output_ids.begin() + static_cast<std::ptrdiff_t>(input_ids.size()),
        output_ids.end());
    std::string text = tokenizer_->decode(new_tokens);

    return TextResult{std::move(text), std::move(new_tokens)};
}

namespace {

runtime::adapters::io::DecodedImage convert_float_to_decoded(
    const float* pixels, int32_t height, int32_t width)
{
    runtime::adapters::io::DecodedImage decoded;
    decoded.width = width;
    decoded.height = height;
    decoded.channels = 3;
    auto n = static_cast<std::size_t>(width) * height * 3;
    decoded.pixels.resize(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        float v = std::max(0.0F, std::min(255.0F, pixels[i] * 255.0F));
        decoded.pixels[i] = static_cast<uint8_t>(v + 0.5F);
    }
    return decoded;
}

int32_t infer_feature_dim(const TrtModule& encoder, int32_t configured_dim)
{
    if (configured_dim > 0)
        return configured_dim;
    for (const auto& info : encoder.output_info())
    {
        if (info.name == "image_features" && info.shape.size() >= 2)
            return static_cast<int32_t>(info.shape.back());
    }
    return 0;
}

} // namespace

std::pair<int32_t, int32_t> VLPipeline::resolve_gen_limits(
    const GenerateConfig& cfg) const
{
    int32_t max_new = (cfg.max_new_tokens > 0) ? cfg.max_new_tokens : 128;
    int32_t eos = (cfg.eos_token_id >= 0) ? cfg.eos_token_id : config_.id_eos;
    return {max_new, eos};
}

TextResult VLPipeline::generate(
    const std::string& prompt, const float* image_pixels,
    int32_t image_height, int32_t image_width,
    const GenerateConfig& cfg)
{
    if (!tokenizer_)
        throw std::runtime_error("VLPipeline: no tokenizer configured");

    bool valid = image_pixels && image_height > 0 && image_width > 0;
    if (!valid || !vision_encoder_)
        return generate(prompt, cfg);

    // Preprocess and encode the image
    auto decoded = convert_float_to_decoded(image_pixels, image_height, image_width);
    auto preprocessed = preprocess_decoded_image(decoded, vl_preprocess_);
    if (!preprocessed.ok)
        throw std::runtime_error("VLPipeline: image preprocessing failed");

    std::vector<float> features;
    if (!run_vision_encoder(preprocessed.pixel_values.data(),
                            preprocessed.pixel_values.size(), features))
        throw std::runtime_error("VLPipeline: vision encoder failed");

    int32_t dim = infer_feature_dim(*vision_encoder_, config_.vision_output_dim);
    if (dim <= 0)
        throw std::runtime_error("VLPipeline: cannot determine vision feature dim");
    int32_t nf = static_cast<int32_t>(features.size() / static_cast<std::size_t>(dim));

    // Format prompt, tokenize, generate with vision features
    auto input_ids = tokenizer_->encode(format_vl_prompt(prompt, vl_preprocess_));
    auto [max_new, eos] = resolve_gen_limits(cfg);
    auto out = generate_vl_from_ids(input_ids, features, nf, dim, max_new, eos);

    std::vector<int32_t> new_tokens(
        out.begin() + static_cast<std::ptrdiff_t>(input_ids.size()), out.end());
    return TextResult{tokenizer_->decode(new_tokens), std::move(new_tokens)};
}

VLPipeline::GenerationResult VLPipeline::generate_ids(
    const std::vector<int32_t>& input_ids,
    const GenerateConfig& cfg)
{
    int32_t max_new = cfg.max_new_tokens;
    int32_t eos = (cfg.eos_token_id >= 0) ? cfg.eos_token_id : config_.id_eos;
    return GenerationResult{generate_from_ids(input_ids, max_new, eos)};
}

std::vector<int32_t> VLPipeline::generate_from_ids(
    const std::vector<int32_t>& input_ids,
    int32_t max_new_tokens,
    int32_t eos_token_id)
{
    if (max_new_tokens == 0 || input_ids.empty())
        return input_ids;

    cache_->reset();
    cache_->bind_to(*text_decoder_);

    std::vector<float> logits;

    for (std::size_t i = 0; i + 1 < input_ids.size(); ++i)
        run_text_step(input_ids[i], logits);

    run_text_step(input_ids.back(), logits);

    std::vector<int32_t> output = input_ids;

    for (int32_t step = 0; step < max_new_tokens; ++step)
    {
        int32_t next = argmax(logits);
        output.push_back(next);
        if (next == eos_token_id) break;
        run_text_step(next, logits);
    }

    return output;
}

std::vector<int32_t> VLPipeline::generate_vl_from_ids(
    const std::vector<int32_t>& input_ids,
    const std::vector<float>& image_features,
    int32_t num_features,
    int32_t feature_dim,
    int32_t max_new_tokens,
    int32_t eos_token_id)
{
    if (max_new_tokens == 0 || input_ids.empty())
        return input_ids;

    cache_->reset();
    cache_->bind_to(*text_decoder_);

    std::vector<float> logits;
    int32_t feature_index = 0;
    int32_t image_token = config_.image_token_id;

    // Prefill: run each token with vision embedding injection at image tokens.
    auto prefill_one = [&](int32_t tid) {
        if (tid == image_token && feature_index < num_features)
        {
            const float* embed = image_features.data() +
                static_cast<std::size_t>(feature_index) * feature_dim;
            run_text_step_with_embed(tid, embed, 1.0F, logits);
            ++feature_index;
        }
        else
        {
            run_text_step_with_embed(tid, nullptr, 0.0F, logits);
        }
    };

    for (const auto& tid : input_ids)
        prefill_one(tid);

    // Autoregressive decode
    std::vector<int32_t> output = input_ids;
    for (int32_t step = 0; step < max_new_tokens; ++step)
    {
        int32_t next = argmax(logits);
        output.push_back(next);
        if (next == eos_token_id) break;
        run_text_step(next, logits);
    }

    return output;
}

void VLPipeline::run_text_step(int32_t token_id, std::vector<float>& logits)
{
    std::vector<float> mask;
    cache_->build_attention_mask(mask);

    int32_t position = cache_->position();

    Tensor token_t;
    token_t.data = &token_id;
    token_t.shape = {1};
    token_t.dtype = DType::kInt32;

    Tensor pos_t;
    pos_t.data = &position;
    pos_t.shape = {1};
    pos_t.dtype = DType::kInt32;

    Tensor mask_t;
    mask_t.data = mask.data();
    mask_t.shape = {static_cast<int64_t>(mask.size())};
    mask_t.dtype = DType::kFloat32;

    TensorMap inputs;
    inputs["token_id"] = token_t;
    if (config_.has_position_input && text_decoder_->has_input("position_id"))
        inputs["position_id"] = pos_t;
    inputs["attention_mask"] = mask_t;

    auto outputs = text_decoder_->forward(inputs);

    auto it = outputs.find("logits");
    if (it == outputs.end())
        throw std::runtime_error("VLPipeline: no 'logits' output");

    auto n = it->second.numel();
    logits.resize(static_cast<std::size_t>(n));
    std::memcpy(logits.data(), it->second.data, n * sizeof(float));

    cache_->advance();
}

void VLPipeline::run_text_step_with_embed(
    int32_t token_id,
    const float* input_embed,
    float use_input_embed,
    std::vector<float>& logits)
{
    // If the text decoder does not have the input_embed input, fall back
    // to the normal step (embed_input mode not enabled for this engine).
    if (!text_decoder_->has_input("input_embed"))
    {
        run_text_step(token_id, logits);
        return;
    }

    std::vector<float> mask;
    cache_->build_attention_mask(mask);

    int32_t position = cache_->position();

    Tensor token_t;
    token_t.data = &token_id;
    token_t.shape = {1};
    token_t.dtype = DType::kInt32;

    Tensor pos_t;
    pos_t.data = &position;
    pos_t.shape = {1};
    pos_t.dtype = DType::kInt32;

    Tensor mask_t;
    mask_t.data = mask.data();
    mask_t.shape = {static_cast<int64_t>(mask.size())};
    mask_t.dtype = DType::kFloat32;

    // use_input_embed scalar: 1.0 = use input_embed, 0.0 = use token embedding
    Tensor use_embed_t;
    use_embed_t.data = &use_input_embed;
    use_embed_t.shape = {1};
    use_embed_t.dtype = DType::kFloat32;

    TensorMap inputs;
    inputs["token_id"] = token_t;
    if (config_.has_position_input && text_decoder_->has_input("position_id"))
        inputs["position_id"] = pos_t;
    inputs["attention_mask"] = mask_t;
    inputs["use_input_embed"] = use_embed_t;

    // Provide the input embedding vector
    int32_t embed_dim = config_.vision_output_dim;
    std::vector<float> zero_embed;
    if (input_embed == nullptr)
    {
        // Provide a zero embed when not injecting
        zero_embed.resize(static_cast<std::size_t>(embed_dim), 0.0F);
        input_embed = zero_embed.data();
    }

    Tensor embed_t;
    embed_t.data = const_cast<float*>(input_embed);
    embed_t.shape = {static_cast<int64_t>(embed_dim)};
    embed_t.dtype = DType::kFloat32;
    inputs["input_embed"] = embed_t;

    // DeepStack: if the engine has deepstack inputs, provide inactive zeros.
    // Full DeepStack support with per-level features would require the vision
    // encoder to output multiple feature levels (deepstack_features_0..N).
    // For now we set deepstack_active=0 which disables the DeepStack injection
    // in the TRT graph (the text-only embedding path is used for non-image tokens,
    // and the standard input_embed path is used for image tokens).
    if (text_decoder_->has_input("deepstack_active"))
    {
        float ds_active = 0.0F;
        Tensor ds_active_t;
        ds_active_t.data = &ds_active;
        ds_active_t.shape = {1};
        ds_active_t.dtype = DType::kFloat32;
        inputs["deepstack_active"] = ds_active_t;
    }

    auto outputs = text_decoder_->forward(inputs);

    auto it = outputs.find("logits");
    if (it == outputs.end())
        throw std::runtime_error("VLPipeline: no 'logits' output");

    auto n = it->second.numel();
    logits.resize(static_cast<std::size_t>(n));
    std::memcpy(logits.data(), it->second.data, n * sizeof(float));

    cache_->advance();
}

bool VLPipeline::run_vision_encoder(
    const float* pixel_values,
    std::size_t pixel_count,
    std::vector<float>& image_features)
{
    if (!vision_encoder_ || !vision_encoder_->ok())
        return false;

    // Build input tensor for pixel values
    Tensor pixel_t;
    pixel_t.data = const_cast<float*>(pixel_values);
    // Get the shape from the vision engine's input
    auto inputs_info = vision_encoder_->input_info();
    for (const auto& info : inputs_info)
    {
        if (info.name == "pixel_values")
        {
            pixel_t.shape = info.shape;
            break;
        }
    }
    if (pixel_t.shape.empty())
    {
        // Fallback: use the pixel_count as a flat shape
        pixel_t.shape = {static_cast<int64_t>(pixel_count)};
    }
    pixel_t.dtype = DType::kFloat32;

    TensorMap inputs;
    inputs["pixel_values"] = pixel_t;

    auto outputs = vision_encoder_->forward(inputs);

    // Extract image_features output
    auto it = outputs.find("image_features");
    if (it == outputs.end())
    {
        std::cerr << "[trtf] Vision encoder has no 'image_features' output"
                  << std::endl;
        return false;
    }

    auto n = it->second.numel();
    image_features.resize(static_cast<std::size_t>(n));
    std::memcpy(image_features.data(), it->second.data, n * sizeof(float));

    return true;
}

int32_t VLPipeline::argmax(const std::vector<float>& logits)
{
    if (logits.empty()) return 0;
    return static_cast<int32_t>(
        std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
}

} // namespace trtf

#endif // TRTF_HAS_TRT
