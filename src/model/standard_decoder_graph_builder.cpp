#include "model/standard_decoder_graph_builder.h"
#include "runtime/trt/trt_common.h"
#include "runtime/trt/trt_graph_ops.h"
#include "runtime/trt/trt_engine_lifecycle.h"
#include "model/trt_model_definition.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#if TRTF_HAS_TRT
#include <NvInfer.h>
#endif

namespace trtf {

#if TRTF_HAS_TRT

namespace {

struct DecoderLayerTensors {
    nvinfer1::ITensor* hidden{nullptr};
    nvinfer1::ITensor* present_k{nullptr};
    nvinfer1::ITensor* present_v{nullptr};
};

DecoderLayerTensors add_standard_decoder_layer_block(nvinfer1::INetworkDefinition& network,
    const TrtDecoderDefinition& weights,
    const TrtDecoderLayerDefinition& layer, nvinfer1::ITensor& hidden, nvinfer1::ITensor& cache_k,
    nvinfer1::ITensor& cache_v,
    nvinfer1::ITensor& attention_mask, nvinfer1::ITensor& position_id, nvinfer1::ITensor& cos_table,
    nvinfer1::ITensor& sin_table, nvinfer1::ITensor& rotate_half_matrix,
    nvinfer1::ITensor& attention_scale_tensor, nvinfer1::ITensor& eps_tensor)
{
    DecoderLayerTensors out;
    const int32_t attention_size = weights.attention_size > 0 ? weights.attention_size : weights.hidden_size;
    if (attention_size <= 0 || weights.num_attention_heads <= 0 || attention_size % weights.num_attention_heads != 0)
    {
        return out;
    }

    const int32_t head_dim = attention_size / weights.num_attention_heads;
    const int32_t attention_window = weights.max_cache_length + 1;

    nvinfer1::ITensor* norm1 = add_rms_norm(network, hidden, weights.hidden_size, layer.input_norm, eps_tensor);
    if (norm1 == nullptr)
    {
        return out;
    }

    nvinfer1::ITensor* q
        = add_matmul_rhs_constant(network, *norm1, weights.hidden_size, attention_size, layer.w_q);
    nvinfer1::ITensor* k
        = add_matmul_rhs_constant(network, *norm1, weights.hidden_size, attention_size, layer.w_k);
    nvinfer1::ITensor* v
        = add_matmul_rhs_constant(network, *norm1, weights.hidden_size, attention_size, layer.w_v);
    if (q == nullptr || k == nullptr || v == nullptr)
    {
        return out;
    }

    // Optional QKV biases (Qwen2, etc.). Empty = skip.
    if (!layer.q_bias.empty())
    {
        q = add_bias_sum(network, *q, attention_size, layer.q_bias);
    }
    if (!layer.k_bias.empty())
    {
        k = add_bias_sum(network, *k, attention_size, layer.k_bias);
    }
    if (!layer.v_bias.empty())
    {
        v = add_bias_sum(network, *v, attention_size, layer.v_bias);
    }
    if (q == nullptr || k == nullptr || v == nullptr)
    {
        return out;
    }

    if (!layer.q_norm.empty())
    {
        q = add_rms_norm_per_head(
            network, *q, weights.num_attention_heads, head_dim, layer.q_norm, eps_tensor);
    }
    if (!layer.k_norm.empty())
    {
        k = add_rms_norm_per_head(
            network, *k, weights.num_attention_heads, head_dim, layer.k_norm, eps_tensor);
    }
    if (q == nullptr || k == nullptr)
    {
        return out;
    }

    q = add_apply_rope(network, *q, position_id, cos_table, sin_table, rotate_half_matrix);
    k = add_apply_rope(network, *k, position_id, cos_table, sin_table, rotate_half_matrix);
    if (q == nullptr || k == nullptr)
    {
        return out;
    }

    auto* current_k_reshape = network.addShuffle(*k);
    auto* current_v_reshape = network.addShuffle(*v);
    if (current_k_reshape == nullptr || current_v_reshape == nullptr)
    {
        return out;
    }
    current_k_reshape->setReshapeDimensions(make_dims_2d(1, attention_size));
    current_v_reshape->setReshapeDimensions(make_dims_2d(1, attention_size));

    nvinfer1::ITensor* all_k_inputs[] = {&cache_k, current_k_reshape->getOutput(0)};
    nvinfer1::ITensor* all_v_inputs[] = {&cache_v, current_v_reshape->getOutput(0)};
    auto* all_k_concat = network.addConcatenation(all_k_inputs, 2);
    auto* all_v_concat = network.addConcatenation(all_v_inputs, 2);
    if (all_k_concat == nullptr || all_v_concat == nullptr)
    {
        return out;
    }
    all_k_concat->setAxis(0);
    all_v_concat->setAxis(0);

    auto* q_heads = network.addShuffle(*q);
    if (q_heads == nullptr)
    {
        return out;
    }
    q_heads->setReshapeDimensions(make_dims_3d(weights.num_attention_heads, 1, head_dim));

    auto* k_heads = network.addShuffle(*all_k_concat->getOutput(0));
    auto* v_heads = network.addShuffle(*all_v_concat->getOutput(0));
    if (k_heads == nullptr || v_heads == nullptr)
    {
        return out;
    }
    k_heads->setReshapeDimensions(make_dims_3d(attention_window, weights.num_attention_heads, head_dim));
    v_heads->setReshapeDimensions(make_dims_3d(attention_window, weights.num_attention_heads, head_dim));

    nvinfer1::Permutation seq_to_head_major{};
    seq_to_head_major.order[0] = 1;
    seq_to_head_major.order[1] = 0;
    seq_to_head_major.order[2] = 2;
    k_heads->setSecondTranspose(seq_to_head_major);
    v_heads->setSecondTranspose(seq_to_head_major);

    auto* score = network.addMatrixMultiply(
        *q_heads->getOutput(0), nvinfer1::MatrixOperation::kNONE,
        *k_heads->getOutput(0), nvinfer1::MatrixOperation::kTRANSPOSE);
    if (score == nullptr)
    {
        return out;
    }

    auto* scaled
        = network.addElementWise(*score->getOutput(0), attention_scale_tensor, nvinfer1::ElementWiseOperation::kPROD);
    if (scaled == nullptr)
    {
        return out;
    }

    auto* mask3d = network.addShuffle(attention_mask);
    if (mask3d == nullptr)
    {
        return out;
    }
    mask3d->setReshapeDimensions(make_dims_3d(1, 1, attention_window));

    auto* masked = network.addElementWise(*scaled->getOutput(0), *mask3d->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
    if (masked == nullptr)
    {
        return out;
    }

    auto* softmax = network.addSoftMax(*masked->getOutput(0));
    if (softmax == nullptr)
    {
        return out;
    }
    softmax->setAxes(1U << 2);

    auto* context_heads = network.addMatrixMultiply(
        *softmax->getOutput(0), nvinfer1::MatrixOperation::kNONE,
        *v_heads->getOutput(0), nvinfer1::MatrixOperation::kNONE);
    if (context_heads == nullptr)
    {
        return out;
    }

    auto* context_flat = network.addShuffle(*context_heads->getOutput(0));
    if (context_flat == nullptr)
    {
        return out;
    }
    context_flat->setReshapeDimensions(make_dims_2d(1, attention_size));

    nvinfer1::ITensor* attn_out
        = add_matmul_rhs_constant(network, *context_flat->getOutput(0), attention_size, weights.hidden_size, layer.w_o);
    if (attn_out == nullptr)
    {
        return out;
    }

    auto* residual1 = network.addElementWise(hidden, *attn_out, nvinfer1::ElementWiseOperation::kSUM);
    if (residual1 == nullptr)
    {
        return out;
    }

    nvinfer1::ITensor* norm2 = add_rms_norm(
        network, *residual1->getOutput(0), weights.hidden_size, layer.post_attn_norm, eps_tensor);
    if (norm2 == nullptr)
    {
        return out;
    }

    nvinfer1::ITensor* gate
        = add_matmul_rhs_constant(network, *norm2, weights.hidden_size, weights.mlp_size, layer.w_gate);
    nvinfer1::ITensor* up = add_matmul_rhs_constant(network, *norm2, weights.hidden_size, weights.mlp_size, layer.w_up);
    if (gate == nullptr || up == nullptr)
    {
        return out;
    }

    auto* sigmoid = network.addActivation(*gate, nvinfer1::ActivationType::kSIGMOID);
    if (sigmoid == nullptr)
    {
        return out;
    }

    auto* swish = network.addElementWise(*gate, *sigmoid->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);
    if (swish == nullptr)
    {
        return out;
    }

    auto* gated = network.addElementWise(*swish->getOutput(0), *up, nvinfer1::ElementWiseOperation::kPROD);
    if (gated == nullptr)
    {
        return out;
    }

    nvinfer1::ITensor* down
        = add_matmul_rhs_constant(network, *gated->getOutput(0), weights.mlp_size, weights.hidden_size, layer.w_down);
    if (down == nullptr)
    {
        return out;
    }

    auto* residual2 = network.addElementWise(
        *residual1->getOutput(0), *down, nvinfer1::ElementWiseOperation::kSUM);
    if (residual2 == nullptr)
    {
        return out;
    }

    out.hidden = residual2->getOutput(0);
    out.present_k = k;
    out.present_v = v;
    return out;
}

std::unique_ptr<DecoderStepEngine> create_decoder_step_engine_legacy(
    const TrtDecoderDefinition& weights, TrtLogger& logger)
{
    auto builder = TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    if (!builder)
    {
        return nullptr;
    }

    uint32_t flags = 0U;
#if NV_TENSORRT_MAJOR < 10
    flags = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
#endif
    auto network = TrtUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(flags));
    auto config = TrtUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!network || !config)
    {
        return nullptr;
    }

#if NV_TENSORRT_MAJOR >= 8
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);
    config->clearFlag(nvinfer1::BuilderFlag::kTF32);
#endif

    const std::string cache_k_name = "cache_k";
    const std::string cache_v_name = "cache_v";
    const std::string present_k_name = "present_k";
    const std::string present_v_name = "present_v";

    auto* token_id = network->addInput("token_id", nvinfer1::DataType::kINT32, make_dims_1d(1));
    auto* cache_k = network->addInput(
        cache_k_name.c_str(), nvinfer1::DataType::kFLOAT, make_dims_2d(weights.max_cache_length, weights.hidden_size));
    auto* cache_v = network->addInput(
        cache_v_name.c_str(), nvinfer1::DataType::kFLOAT, make_dims_2d(weights.max_cache_length, weights.hidden_size));
    auto* attention_mask = network->addInput(
        "attention_mask", nvinfer1::DataType::kFLOAT, make_dims_2d(1, weights.max_cache_length));
    if (token_id == nullptr || cache_k == nullptr || cache_v == nullptr || attention_mask == nullptr)
    {
        return nullptr;
    }

    nvinfer1::ITensor* embedding_table
        = add_constant_tensor(*network, make_dims_2d(weights.vocab_size, weights.hidden_size), weights.embedding);
    if (embedding_table == nullptr)
    {
        return nullptr;
    }

    auto* gather = network->addGather(*embedding_table, *token_id, 0);
    if (gather == nullptr)
    {
        return nullptr;
    }
    nvinfer1::ITensor* token_hidden = gather->getOutput(0);

    nvinfer1::ITensor* q
        = add_matmul_rhs_constant(*network, *token_hidden, weights.hidden_size, weights.hidden_size, weights.w_q);
    nvinfer1::ITensor* k
        = add_matmul_rhs_constant(*network, *token_hidden, weights.hidden_size, weights.hidden_size, weights.w_k);
    nvinfer1::ITensor* v
        = add_matmul_rhs_constant(*network, *token_hidden, weights.hidden_size, weights.hidden_size, weights.w_v);
    if (q == nullptr || k == nullptr || v == nullptr)
    {
        return nullptr;
    }

    auto* score_matmul
        = network->addMatrixMultiply(*q, nvinfer1::MatrixOperation::kNONE, *cache_k, nvinfer1::MatrixOperation::kTRANSPOSE);
    if (score_matmul == nullptr)
    {
        return nullptr;
    }

    const float scale = 1.0F / std::sqrt(static_cast<float>(weights.hidden_size));
    const std::vector<float> score_scale{scale};
    nvinfer1::ITensor* scale_tensor = add_constant_tensor(*network, make_dims_2d(1, 1), score_scale);
    if (scale_tensor == nullptr)
    {
        return nullptr;
    }

    auto* scaled_scores
        = network->addElementWise(*score_matmul->getOutput(0), *scale_tensor, nvinfer1::ElementWiseOperation::kPROD);
    if (scaled_scores == nullptr)
    {
        return nullptr;
    }

    auto* masked_scores
        = network->addElementWise(*scaled_scores->getOutput(0), *attention_mask, nvinfer1::ElementWiseOperation::kSUM);
    if (masked_scores == nullptr)
    {
        return nullptr;
    }

    auto* softmax = network->addSoftMax(*masked_scores->getOutput(0));
    if (softmax == nullptr)
    {
        return nullptr;
    }
    softmax->setAxes(1U << 1);

    auto* context_layer
        = network->addMatrixMultiply(*softmax->getOutput(0), nvinfer1::MatrixOperation::kNONE, *cache_v, nvinfer1::MatrixOperation::kNONE);
    if (context_layer == nullptr)
    {
        return nullptr;
    }

    auto* residual1
        = network->addElementWise(*token_hidden, *context_layer->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
    if (residual1 == nullptr)
    {
        return nullptr;
    }

    nvinfer1::ITensor* mlp_fc1
        = add_matmul_rhs_constant(*network, *residual1->getOutput(0), weights.hidden_size, weights.mlp_size, weights.w1);
    if (mlp_fc1 == nullptr)
    {
        return nullptr;
    }

    nvinfer1::ITensor* mlp_fc1_bias = add_bias_sum(*network, *mlp_fc1, weights.mlp_size, weights.b1);
    if (mlp_fc1_bias == nullptr)
    {
        return nullptr;
    }

    auto* mlp_relu = network->addActivation(*mlp_fc1_bias, nvinfer1::ActivationType::kRELU);
    if (mlp_relu == nullptr)
    {
        return nullptr;
    }

    nvinfer1::ITensor* mlp_fc2
        = add_matmul_rhs_constant(*network, *mlp_relu->getOutput(0), weights.mlp_size, weights.hidden_size, weights.w2);
    if (mlp_fc2 == nullptr)
    {
        return nullptr;
    }

    nvinfer1::ITensor* mlp_fc2_bias = add_bias_sum(*network, *mlp_fc2, weights.hidden_size, weights.b2);
    if (mlp_fc2_bias == nullptr)
    {
        return nullptr;
    }

    auto* residual2
        = network->addElementWise(*residual1->getOutput(0), *mlp_fc2_bias, nvinfer1::ElementWiseOperation::kSUM);
    if (residual2 == nullptr)
    {
        return nullptr;
    }

    nvinfer1::ITensor* logits
        = add_matmul_rhs_constant(*network, *residual2->getOutput(0), weights.hidden_size, weights.vocab_size, weights.w_out);
    if (logits == nullptr)
    {
        return nullptr;
    }

    nvinfer1::ITensor* logits_bias = add_bias_sum(*network, *logits, weights.vocab_size, weights.b_out);
    if (logits_bias == nullptr)
    {
        return nullptr;
    }

    logits_bias->setName("logits");
    k->setName(present_k_name.c_str());
    v->setName(present_v_name.c_str());
    network->markOutput(*logits_bias);
    network->markOutput(*k);
    network->markOutput(*v);

    return finalize_decoder_step_engine(*builder, *network, *config, logger, weights,
        {cache_k_name}, {cache_v_name}, {present_k_name}, {present_v_name}, false);
}

std::unique_ptr<DecoderStepEngine> create_decoder_step_engine_multi_layer(
    const TrtDecoderDefinition& weights, TrtLogger& logger)
{
    const int32_t attention_size = weights.attention_size > 0 ? weights.attention_size : weights.hidden_size;
    if (attention_size <= 0 || weights.num_attention_heads <= 0
        || attention_size % weights.num_attention_heads != 0)
    {
        return nullptr;
    }

    // Fast path: try loading a cached engine before building the graph.
    const int32_t num_layers = static_cast<int32_t>(weights.decoder_layers.size());
    auto cached = try_load_cached_engine(logger, weights, num_layers, /*requires_position_input=*/true);
    if (cached)
    {
        return cached;
    }

    const int32_t head_dim = attention_size / weights.num_attention_heads;
    const int32_t attention_window = weights.max_cache_length + 1;

    auto builder = TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    if (!builder)
    {
        return nullptr;
    }

    uint32_t flags = 0U;
#if NV_TENSORRT_MAJOR < 10
    flags = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
#endif
    auto network = TrtUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(flags));
    auto config = TrtUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!network || !config)
    {
        return nullptr;
    }

#if NV_TENSORRT_MAJOR >= 8
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);
    config->clearFlag(nvinfer1::BuilderFlag::kTF32);
#endif

    auto* token_id = network->addInput("token_id", nvinfer1::DataType::kINT32, make_dims_1d(1));
    auto* position_id = network->addInput("position_id", nvinfer1::DataType::kINT32, make_dims_1d(1));
    auto* attention_mask = network->addInput(
        "attention_mask", nvinfer1::DataType::kFLOAT, make_dims_2d(1, attention_window));
    if (token_id == nullptr || position_id == nullptr || attention_mask == nullptr)
    {
        return nullptr;
    }

    std::vector<nvinfer1::ITensor*> cache_k_inputs;
    std::vector<nvinfer1::ITensor*> cache_v_inputs;
    std::vector<std::string> cache_k_names;
    std::vector<std::string> cache_v_names;
    std::vector<std::string> present_k_names;
    std::vector<std::string> present_v_names;

    cache_k_inputs.reserve(static_cast<std::size_t>(num_layers));
    cache_v_inputs.reserve(static_cast<std::size_t>(num_layers));
    cache_k_names.reserve(static_cast<std::size_t>(num_layers));
    cache_v_names.reserve(static_cast<std::size_t>(num_layers));
    present_k_names.reserve(static_cast<std::size_t>(num_layers));
    present_v_names.reserve(static_cast<std::size_t>(num_layers));

    for (int32_t layer = 0; layer < num_layers; ++layer)
    {
        cache_k_names.push_back(layer_tensor_name("cache_k", layer));
        cache_v_names.push_back(layer_tensor_name("cache_v", layer));
        present_k_names.push_back(layer_tensor_name("present_k", layer));
        present_v_names.push_back(layer_tensor_name("present_v", layer));

        auto* ck = network->addInput(cache_k_names.back().c_str(), nvinfer1::DataType::kFLOAT,
            make_dims_2d(weights.max_cache_length, attention_size));
        auto* cv = network->addInput(cache_v_names.back().c_str(), nvinfer1::DataType::kFLOAT,
            make_dims_2d(weights.max_cache_length, attention_size));
        if (ck == nullptr || cv == nullptr)
        {
            return nullptr;
        }
        cache_k_inputs.push_back(ck);
        cache_v_inputs.push_back(cv);
    }

    nvinfer1::ITensor* embedding_table
        = add_constant_tensor(*network, make_dims_2d(weights.vocab_size, weights.hidden_size), weights.embedding);
    if (embedding_table == nullptr)
    {
        return nullptr;
    }

    auto* gather = network->addGather(*embedding_table, *token_id, 0);
    if (gather == nullptr)
    {
        return nullptr;
    }
    nvinfer1::ITensor* hidden = gather->getOutput(0);

    const std::vector<float> cos_table
        = make_rope_table(attention_window, attention_size, weights.num_attention_heads, weights.rope_theta, true);
    const std::vector<float> sin_table
        = make_rope_table(attention_window, attention_size, weights.num_attention_heads, weights.rope_theta, false);
    const std::vector<float> rotate_half = make_rotate_half_matrix(attention_size, weights.num_attention_heads);
    const std::vector<float> eps_data{weights.rms_norm_eps};
    const float attention_scale = 1.0F / std::sqrt(static_cast<float>(std::max(head_dim, 1)));
    const std::vector<float> attention_scale_data{attention_scale};

    nvinfer1::ITensor* cos_tensor
        = add_constant_tensor(*network, make_dims_2d(attention_window, attention_size), cos_table);
    nvinfer1::ITensor* sin_tensor
        = add_constant_tensor(*network, make_dims_2d(attention_window, attention_size), sin_table);
    nvinfer1::ITensor* rotate_half_tensor
        = add_constant_tensor(*network, make_dims_2d(attention_size, attention_size), rotate_half);
    nvinfer1::ITensor* eps_tensor = add_constant_tensor(*network, make_dims_2d(1, 1), eps_data);
    nvinfer1::ITensor* attention_scale_tensor
        = add_constant_tensor(*network, make_dims_3d(1, 1, 1), attention_scale_data);
    if (cos_tensor == nullptr || sin_tensor == nullptr || rotate_half_tensor == nullptr || eps_tensor == nullptr
        || attention_scale_tensor == nullptr)
    {
        return nullptr;
    }

    std::vector<nvinfer1::ITensor*> present_k_outputs(static_cast<std::size_t>(num_layers), nullptr);
    std::vector<nvinfer1::ITensor*> present_v_outputs(static_cast<std::size_t>(num_layers), nullptr);

    for (int32_t layer_idx = 0; layer_idx < num_layers; ++layer_idx)
    {
        const TrtDecoderLayerDefinition& layer = weights.decoder_layers[static_cast<std::size_t>(layer_idx)];
        DecoderLayerTensors layer_tensors = add_standard_decoder_layer_block(
            *network, weights, layer, *hidden, *cache_k_inputs[static_cast<std::size_t>(layer_idx)],
            *cache_v_inputs[static_cast<std::size_t>(layer_idx)], *attention_mask, *position_id,
            *cos_tensor, *sin_tensor, *rotate_half_tensor, *attention_scale_tensor, *eps_tensor);
        if (layer_tensors.hidden == nullptr || layer_tensors.present_k == nullptr || layer_tensors.present_v == nullptr)
        {
            return nullptr;
        }

        hidden = layer_tensors.hidden;
        present_k_outputs[static_cast<std::size_t>(layer_idx)] = layer_tensors.present_k;
        present_v_outputs[static_cast<std::size_t>(layer_idx)] = layer_tensors.present_v;
    }

    if (!weights.final_norm.empty())
    {
        hidden = add_rms_norm(*network, *hidden, weights.hidden_size, weights.final_norm, *eps_tensor);
        if (hidden == nullptr)
        {
            return nullptr;
        }
    }

    nvinfer1::ITensor* logits
        = add_matmul_rhs_constant(*network, *hidden, weights.hidden_size, weights.vocab_size, weights.w_out);
    if (logits == nullptr)
    {
        return nullptr;
    }
    nvinfer1::ITensor* logits_bias = add_bias_sum(*network, *logits, weights.vocab_size, weights.b_out);
    if (logits_bias == nullptr)
    {
        return nullptr;
    }

    logits_bias->setName("logits");
    network->markOutput(*logits_bias);

    for (int32_t layer = 0; layer < num_layers; ++layer)
    {
        nvinfer1::ITensor* pk = present_k_outputs[static_cast<std::size_t>(layer)];
        nvinfer1::ITensor* pv = present_v_outputs[static_cast<std::size_t>(layer)];
        if (pk == nullptr || pv == nullptr)
        {
            return nullptr;
        }
        pk->setName(present_k_names[static_cast<std::size_t>(layer)].c_str());
        pv->setName(present_v_names[static_cast<std::size_t>(layer)].c_str());
        network->markOutput(*pk);
        network->markOutput(*pv);
    }

    return finalize_decoder_step_engine(*builder, *network, *config, logger, weights,
        cache_k_names, cache_v_names, present_k_names, present_v_names, true);
}

} // namespace

std::unique_ptr<DecoderStepEngine> StandardDecoderGraphBuilder::build_decoder_step_engine(
    const TrtDecoderDefinition& weights, TrtLogger& logger)
{
    if (weights.has_decoder_layers && !weights.decoder_layers.empty())
    {
        return create_decoder_step_engine_multi_layer(weights, logger);
    }
    return create_decoder_step_engine_legacy(weights, logger);
}

#endif // TRTF_HAS_TRT

} // namespace trtf
