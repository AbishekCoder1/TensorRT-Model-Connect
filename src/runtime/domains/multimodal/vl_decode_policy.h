#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace trtmc {

struct VlTokenEmbedding
{
    const float* input_embed{nullptr};
    float use_input_embed{0.0F};
    std::vector<const float*> deepstack_embeds;
    float deepstack_active{0.0F};
};

inline int32_t current_vl_token(const std::vector<int32_t>& input_ids, int32_t bos_token_id)
{
    return input_ids.empty() ? bos_token_id : input_ids.back();
}

inline VlTokenEmbedding build_vl_token_embedding(
    int32_t token_id,
    int32_t image_token_id,
    const float* image_features,
    int32_t num_features,
    int32_t feature_dim,
    int32_t& feature_index,
    const std::vector<std::vector<float>>& deepstack_features)
{
    VlTokenEmbedding embedding;
    if (token_id != image_token_id || feature_index >= num_features)
    {
        return embedding;
    }

    const int32_t used_feature_index = feature_index;
    embedding.input_embed = image_features + static_cast<std::size_t>(feature_index) * feature_dim;
    embedding.use_input_embed = 1.0F;
    ++feature_index;

    if (deepstack_features.empty())
    {
        return embedding;
    }

    embedding.deepstack_embeds.reserve(deepstack_features.size());
    for (const auto& deepstack : deepstack_features)
    {
        const int32_t deepstack_feature_count = static_cast<int32_t>(deepstack.size() / feature_dim);
        if (used_feature_index < deepstack_feature_count)
        {
            embedding.deepstack_embeds.push_back(
                deepstack.data() + static_cast<std::size_t>(used_feature_index) * feature_dim);
        }
        else
        {
            embedding.deepstack_embeds.push_back(nullptr);
        }
    }
    embedding.deepstack_active = 1.0F;
    return embedding;
}

template <typename StepFn, typename SelectFn>
inline bool run_vl_decode_loop(
    std::size_t max_new_tokens,
    int32_t eos_token_id,
    std::vector<float>& logits,
    std::vector<int32_t>& output_ids,
    std::string& error,
    StepFn&& run_step,
    SelectFn&& select_next_token)
{
    for (std::size_t step = 0; step < max_new_tokens; ++step)
    {
        const int32_t next_token = select_next_token(logits);
        output_ids.push_back(next_token);
        if (next_token == eos_token_id)
        {
            return true;
        }

        if (!run_step(next_token, logits, error))
        {
            return false;
        }
    }

    return true;
}

} // namespace trtmc
