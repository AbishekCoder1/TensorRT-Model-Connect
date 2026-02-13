#include "trt_model_definition.h"
#include "qwen3_trt_model_definition.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace trtf {
namespace {

std::vector<float> make_identity_matrix(int32_t dim)
{
    std::vector<float> matrix(static_cast<std::size_t>(dim) * static_cast<std::size_t>(dim), 0.0F);
    for (int32_t i = 0; i < dim; ++i)
    {
        matrix[static_cast<std::size_t>(i) * static_cast<std::size_t>(dim) + static_cast<std::size_t>(i)] = 1.0F;
    }
    return matrix;
}

void set_transition_row(std::vector<float>& matrix, int32_t vocab_size, int32_t from_token, int32_t to_token)
{
    if (from_token < 0 || to_token < 0 || from_token >= vocab_size || to_token >= vocab_size)
    {
        return;
    }

    float* row = matrix.data() + static_cast<std::size_t>(from_token) * static_cast<std::size_t>(vocab_size);
    std::fill(row, row + vocab_size, -1.0F);
    row[to_token] = 8.0F;
}

} // namespace

TrtDecoderDefinition BuildTrtDecoderWeights(const ITokenizer& tokenizer, const DecoderModel& model)
{
    TrtDecoderDefinition definition;
    definition.vocab_size = static_cast<int32_t>(model.vocab.size());
    if (definition.vocab_size <= 0)
    {
        throw std::runtime_error("Model vocabulary must not be empty.");
    }
    definition.max_cache_length = std::max(model.max_cache_length, 1);
    definition.id_bos = (model.architecture.bos_token_id >= 0)
        ? model.architecture.bos_token_id
        : tokenizer.id_for_token("<bos>");
    definition.id_eos = (model.architecture.eos_token_id >= 0)
        ? model.architecture.eos_token_id
        : tokenizer.id_for_token("<eos>");

    const auto expect_size = [](const std::vector<float>& tensor, std::size_t expected, const char* name) {
        if (tensor.size() != expected)
        {
            throw std::runtime_error(
                std::string("Invalid checkpoint tensor size for ") + name + ": expected "
                + std::to_string(expected) + ", got " + std::to_string(tensor.size()));
        }
    };

    if (model.has_checkpoint)
    {
        definition.hidden_size = model.checkpoint.hidden_size;
        definition.attention_size = model.checkpoint.attention_size > 0
            ? model.checkpoint.attention_size
            : definition.hidden_size;
        definition.mlp_size = model.checkpoint.mlp_size;
        if (definition.hidden_size <= 0 || definition.attention_size <= 0 || definition.mlp_size <= 0)
        {
            throw std::runtime_error("Checkpoint has invalid hidden_size/attention_size/mlp_size.");
        }

        const std::size_t vocab = static_cast<std::size_t>(definition.vocab_size);
        const std::size_t hidden = static_cast<std::size_t>(definition.hidden_size);
        const std::size_t mlp = static_cast<std::size_t>(definition.mlp_size);

        expect_size(model.checkpoint.embedding, vocab * hidden, "embedding");
        expect_size(model.checkpoint.w_out, hidden * vocab, "w_out");
        expect_size(model.checkpoint.b_out, vocab, "b_out");

        definition.embedding = model.checkpoint.embedding;
        definition.w_out = model.checkpoint.w_out;
        definition.b_out = model.checkpoint.b_out;

        if (PopulateQwen3TrtModelDefinition(definition, model))
        {
            return definition;
        }

        definition.attention_size = definition.hidden_size;
        expect_size(model.checkpoint.w_q, hidden * hidden, "w_q");
        expect_size(model.checkpoint.w_k, hidden * hidden, "w_k");
        expect_size(model.checkpoint.w_v, hidden * hidden, "w_v");
        expect_size(model.checkpoint.w1, hidden * mlp, "w1");
        expect_size(model.checkpoint.b1, mlp, "b1");
        expect_size(model.checkpoint.w2, mlp * hidden, "w2");
        expect_size(model.checkpoint.b2, hidden, "b2");

        definition.w_q = model.checkpoint.w_q;
        definition.w_k = model.checkpoint.w_k;
        definition.w_v = model.checkpoint.w_v;
        definition.w1 = model.checkpoint.w1;
        definition.b1 = model.checkpoint.b1;
        definition.w2 = model.checkpoint.w2;
        definition.b2 = model.checkpoint.b2;
        return definition;
    }

    // Compatibility path for older model directories without tensor checkpoint.
    definition.hidden_size = definition.vocab_size;
    definition.attention_size = definition.hidden_size;
    definition.mlp_size = definition.hidden_size * 2;

    const std::size_t vocab = static_cast<std::size_t>(definition.vocab_size);
    const std::size_t hidden = static_cast<std::size_t>(definition.hidden_size);
    const std::size_t mlp = static_cast<std::size_t>(definition.mlp_size);

    definition.embedding.assign(vocab * hidden, 0.0F);
    for (std::size_t i = 0; i < vocab; ++i)
    {
        definition.embedding[i * hidden + i] = 1.0F;
    }

    definition.w_q = make_identity_matrix(definition.hidden_size);
    definition.w_k = make_identity_matrix(definition.hidden_size);
    definition.w_v.assign(hidden * hidden, 0.0F);

    definition.w1.assign(hidden * mlp, 0.0F);
    definition.b1.assign(mlp, 0.0F);
    definition.w2.assign(mlp * hidden, 0.0F);
    definition.b2.assign(hidden, 0.0F);

    const int32_t default_next_id = tokenizer.id_for_token(model.default_next_token);
    definition.w_out.assign(hidden * vocab, -1.0F);
    for (int32_t token_id = 0; token_id < definition.vocab_size; ++token_id)
    {
        set_transition_row(definition.w_out, definition.vocab_size, token_id, default_next_id);
    }

    for (const auto& transition : model.transitions)
    {
        const int32_t from_id = tokenizer.id_for_token(transition.first);
        const int32_t to_id = tokenizer.id_for_token(transition.second);
        set_transition_row(definition.w_out, definition.vocab_size, from_id, to_id);
    }

    definition.b_out.assign(vocab, 0.0F);
    return definition;
}

} // namespace trtf
