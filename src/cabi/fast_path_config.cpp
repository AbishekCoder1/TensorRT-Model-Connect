#include "cabi/fast_path_config.h"
#include "utils/json_helpers.h"

#include <algorithm>

namespace trtf {

FastPathModelConfig parse_fast_path_config(const std::string& config_text, int32_t max_cache_length_override)
{
    FastPathModelConfig cfg;

    cfg.vocab_size = extract_json_int(config_text, "vocab_size", 0);

    // Support config aliases from various model families:
    //   GPT-2:  n_embd, n_layer, n_head
    //   XGLM:   d_model, num_layers, attention_heads
    //   Bloom:   n_embed, n_layer, num_attention_heads (standard)
    cfg.hidden_size = extract_json_int(config_text, "hidden_size", 0);
    if (cfg.hidden_size == 0)
        cfg.hidden_size = extract_json_int(config_text, "n_embd", 0);
    if (cfg.hidden_size == 0)
        cfg.hidden_size = extract_json_int(config_text, "d_model", 0);
    if (cfg.hidden_size == 0)
        cfg.hidden_size = extract_json_int(config_text, "n_embed", 0);

    int32_t raw_layers = extract_json_int(config_text, "num_hidden_layers", 0);
    if (raw_layers == 0)
        raw_layers = extract_json_int(config_text, "n_layer", 0);
    if (raw_layers == 0)
        raw_layers = extract_json_int(config_text, "num_layers", 0);
    cfg.num_layers = std::max(raw_layers, 1);

    int32_t raw_heads = extract_json_int(config_text, "num_attention_heads", 0);
    if (raw_heads == 0)
        raw_heads = extract_json_int(config_text, "n_head", 0);
    if (raw_heads == 0)
        raw_heads = extract_json_int(config_text, "attention_heads", 0);
    if (raw_heads == 0)
        raw_heads = extract_json_int(config_text, "num_heads", 0);
    cfg.num_heads = std::max(raw_heads, 1);

    cfg.num_kv_heads = std::max(extract_json_int(config_text, "num_key_value_heads", cfg.num_heads), 1);
    cfg.head_dim = extract_json_int(config_text, "head_dim", cfg.hidden_size / cfg.num_heads);
    cfg.attention_size = cfg.num_heads * cfg.head_dim;

    if (max_cache_length_override > 0)
    {
        cfg.max_cache_length = max_cache_length_override;
    }
    else
    {
        cfg.max_cache_length = extract_json_int(config_text, "max_position_embeddings", 32);
        if (cfg.max_cache_length > 4096)
        {
            cfg.max_cache_length = 4096;
        }
    }

    cfg.id_bos = extract_json_int_or_first_array(config_text, "bos_token_id", -1);
    cfg.id_eos = extract_json_int_or_first_array(config_text, "eos_token_id", -1);

    // Runtime strategy: determines backend type. Default is "decoder_kv_cache".
    // MoE models use the same KV cache strategy (routing is in the TRT graph).
    cfg.runtime_strategy = extract_json_string(config_text, "runtime_strategy", "decoder_kv_cache");

    // Mamba/SSM-specific fields
    if (cfg.runtime_strategy == "ssm_recurrent")
    {
        cfg.d_inner = extract_json_int(config_text, "intermediate_size", 0);
        if (cfg.d_inner == 0)
            cfg.d_inner = extract_json_int(config_text, "d_inner", cfg.hidden_size * 2);
        cfg.state_size = extract_json_int(config_text, "state_size", 16);
        cfg.conv_kernel = extract_json_int(config_text, "conv_kernel", 4);
    }

    // Vision-Language fields
    if (cfg.runtime_strategy == "vision_language")
    {
        cfg.has_vision_engine = extract_json_int(config_text, "has_vision_engine", 0) != 0;
        cfg.embed_input = extract_json_int(config_text, "embed_input", 0) != 0;
        cfg.image_token_id = extract_json_int(config_text, "image_token_id", -1);
        cfg.vision_output_dim = extract_json_int(config_text, "vision_output_dim", 0);
        cfg.fixed_image_size = extract_json_int(config_text, "fixed_image_size", 448);
        cfg.num_image_pad_tokens = extract_json_int(config_text, "num_image_pad_tokens", 0);
        cfg.vl_prompt_template = extract_json_string(config_text, "vl_prompt_template", "");
        cfg.image_token_str = extract_json_string(config_text, "image_token_str", "");
    }

    // Diffusion fields
    if (cfg.runtime_strategy == "diffusion")
    {
        cfg.num_text_encoders = extract_json_int(config_text, "num_text_encoders", 0);
        cfg.scheduler = extract_json_string(config_text, "scheduler", "flow_match_euler");
        cfg.num_inference_steps = extract_json_int(config_text, "num_inference_steps", 50);
        cfg.guidance_scale = extract_json_float(config_text, "guidance_scale", 5.0F);
        cfg.flow_shift = extract_json_float(config_text, "flow_shift", 1.0F);
        cfg.video_height = extract_json_int(config_text, "video_height", 480);
        cfg.video_width = extract_json_int(config_text, "video_width", 832);
        cfg.video_num_frames = extract_json_int(config_text, "video_num_frames", 81);
        cfg.z_dim = extract_json_int(config_text, "z_dim", 16);
        cfg.scale_factor_temporal = extract_json_int(config_text, "scale_factor_temporal", 4);
        cfg.scale_factor_spatial = extract_json_int(config_text, "scale_factor_spatial", 8);
        cfg.dit_dim = extract_json_int(config_text, "dit_dim", 1536);
        cfg.dit_num_heads = extract_json_int(config_text, "dit_num_heads", 12);
        cfg.dit_num_layers = extract_json_int(config_text, "dit_num_layers", 30);
        cfg.freq_dim = extract_json_int(config_text, "freq_dim", 256);
        cfg.num_vae_caches = extract_json_int(config_text, "num_vae_caches", 0);
        cfg.text_encoder_dim = extract_json_int(config_text, "text_encoder_dim", 4096);
        cfg.text_seq_len = extract_json_int(config_text, "text_seq_len", 512);
        cfg.latents_mean = extract_json_float_array(config_text, "latents_mean");
        cfg.latents_std = extract_json_float_array(config_text, "latents_std");
        cfg.patch_size = extract_json_int_array(config_text, "patch_size");
        cfg.vae_model_id = extract_json_string(config_text, "vae_model_id", "");
        cfg.diffusion_backend_type = extract_json_string(config_text, "diffusion_backend_type", "wan_3d");
    }

    return cfg;
}

} // namespace trtf
