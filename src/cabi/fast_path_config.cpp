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
    if (raw_heads == 0)
        raw_heads = extract_json_int(config_text, "decoder_attention_heads", 0);
    cfg.num_heads = std::max(raw_heads, 1);

    // Whisper uses decoder_layers for layer count
    if (raw_layers == 0)
    {
        int32_t dl = extract_json_int(config_text, "decoder_layers", 0);
        if (dl > 0) cfg.num_layers = dl;
    }

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

    // Tokenizer: whether to add BOS/EOS special tokens during prompt encoding.
    // Determined at bundle build time and stored in the bundle's config.json.
    // When not present (old bundles), the C++ runtime defaults to true to
    // match HF's tokenizer.encode() default behavior.
    {
        int32_t raw = extract_json_int(config_text, "tokenizer_add_special_tokens", -1);
        if (raw >= 0) {
            cfg.tokenizer_add_special_tokens = (raw != 0);
            cfg.tokenizer_add_special_tokens_present = true;
        }
    }

    // Mamba/SSM-specific fields
    if (cfg.runtime_strategy == "ssm_recurrent")
    {
        cfg.d_inner = extract_json_int(config_text, "intermediate_size", 0);
        if (cfg.d_inner == 0)
            cfg.d_inner = extract_json_int(config_text, "d_inner", cfg.hidden_size * 2);
        cfg.state_size = extract_json_int(config_text, "state_size", 16);
        cfg.conv_kernel = extract_json_int(config_text, "conv_kernel", 4);
    }

    // Hybrid Mamba-Attention fields
    if (cfg.runtime_strategy == "hybrid_mamba_attention")
    {
        cfg.num_mamba_layers = extract_json_int(config_text, "num_mamba_layers", 0);
        cfg.num_attention_layers = extract_json_int(config_text, "num_attention_layers", 0);
        cfg.d_inner = extract_json_int(config_text, "d_inner", cfg.hidden_size * 2);
        cfg.mamba_d_state = extract_json_int(config_text, "mamba_d_state", 128);
        cfg.mamba_d_conv = extract_json_int(config_text, "mamba_d_conv", 4);
        cfg.mamba_nheads = extract_json_int(config_text, "mamba_nheads", 0);
        cfg.mamba_head_dim = extract_json_int(config_text, "mamba_head_dim", 0);
        cfg.conv_dim = extract_json_int(config_text, "conv_dim", cfg.d_inner);
        cfg.layer_types = extract_json_string_array(config_text, "layer_types");
    }

    // Whisper/speech-to-text fields
    if (cfg.runtime_strategy == "speech_to_text")
    {
        cfg.num_mel_bins = extract_json_int(config_text, "num_mel_bins", 80);
        cfg.max_source_positions = extract_json_int(config_text, "max_source_positions", 1500);
        cfg.max_target_positions = extract_json_int(config_text, "max_target_positions", 448);
        cfg.encoder_layers = extract_json_int(config_text, "encoder_layers", cfg.num_layers);
        cfg.decoder_layers = extract_json_int(config_text, "decoder_layers", cfg.num_layers);
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

    // Segmentation fields
    if (cfg.runtime_strategy == "segmentation")
    {
        cfg.num_classes = extract_json_int(config_text, "num_classes", 150);
        cfg.input_image_h = extract_json_int(config_text, "input_image_h", 512);
        cfg.input_image_w = extract_json_int(config_text, "input_image_w", 512);
        cfg.output_h = extract_json_int(config_text, "output_h", 128);
        cfg.output_w = extract_json_int(config_text, "output_w", 128);
        cfg.seg_image_mean = extract_json_float_array(config_text, "image_mean");
        cfg.seg_image_std = extract_json_float_array(config_text, "image_std");
    }

    // SAM prompted segmentation fields
    if (cfg.runtime_strategy == "prompted_segmentation")
    {
        cfg.input_image_h = extract_json_int(config_text, "input_image_h", 1024);
        cfg.input_image_w = extract_json_int(config_text, "input_image_w", 1024);
        cfg.sam_image_embedding_size = extract_json_int(config_text, "sam_image_embedding_size", 64);
        cfg.sam_decoder_hidden_size = extract_json_int(config_text, "sam_decoder_hidden_size", 256);
        cfg.sam_num_mask_outputs = extract_json_int(config_text, "sam_num_mask_outputs", 4);
        cfg.sam_num_multimask_outputs = extract_json_int(config_text, "sam_num_multimask_outputs", 3);
        cfg.seg_image_mean = extract_json_float_array(config_text, "image_mean");
        cfg.seg_image_std = extract_json_float_array(config_text, "image_std");
        cfg.sam_point_embed_fg = extract_json_float_array(config_text, "sam_point_embed_1");
        cfg.sam_point_embed_bg = extract_json_float_array(config_text, "sam_point_embed_0");
        cfg.sam_not_a_point_embed = extract_json_float_array(config_text, "sam_not_a_point_embed");
        cfg.sam_shared_image_pe = extract_json_float_array(config_text, "sam_shared_image_pe");
    }

    // Audio/Bark fields
    if (cfg.runtime_strategy == "text_to_audio")
    {
        cfg.audio_sample_rate = extract_json_int(config_text, "sample_rate", 24000);
        cfg.semantic_vocab_size = extract_json_int(config_text, "semantic_vocab_size", 10000);
        cfg.coarse_vocab_size = extract_json_int(config_text, "coarse_vocab_size", 1024);
        cfg.fine_vocab_size = extract_json_int(config_text, "fine_vocab_size", 1024);
        cfg.n_coarse_codebooks = extract_json_int(config_text, "n_coarse_codebooks", 2);
        cfg.n_fine_codebooks = extract_json_int(config_text, "n_fine_codebooks", 8);
        cfg.semantic_pad_token = extract_json_int(config_text, "semantic_pad_token", 10000);
        cfg.semantic_infer_token = extract_json_int(config_text, "semantic_infer_token", 129599);
        cfg.coarse_semantic_pad_token = extract_json_int(config_text, "coarse_semantic_pad_token", 12048);
        cfg.coarse_infer_token = extract_json_int(config_text, "coarse_infer_token", 12050);
        cfg.text_encoding_offset = extract_json_int(config_text, "text_encoding_offset", 10048);
        cfg.text_pad_token = extract_json_int(config_text, "text_pad_token", 129595);
        cfg.semantic_input_vocab = extract_json_int(config_text, "semantic_input_vocab", 129600);
        cfg.coarse_input_vocab = extract_json_int(config_text, "coarse_input_vocab", 12096);
        cfg.codebook_size = extract_json_int(config_text, "codebook_size", 1024);
        // Coarse sub-model dimensions (defaults to main model dims if not specified)
        cfg.coarse_hidden_size = extract_json_int(config_text, "coarse_hidden_size", cfg.hidden_size);
        cfg.coarse_num_layers = extract_json_int(config_text, "coarse_num_layers", cfg.num_layers);
        cfg.coarse_num_heads = extract_json_int(config_text, "coarse_num_heads", cfg.num_heads);
        cfg.coarse_max_cache_length = extract_json_int(config_text, "coarse_max_cache_length", 1024);
        // Codec engine config
        cfg.codec_seq_length = extract_json_int(config_text, "codec_seq_length", 0);
        cfg.codec_upsample_factor = extract_json_int(config_text, "codec_upsample_factor", 320);
        cfg.codec_n_codebooks = extract_json_int(config_text, "codec_n_codebooks", 8);
        // Fine model config
        cfg.fine_hidden_size = extract_json_int(config_text, "fine_hidden_size", cfg.hidden_size);
        cfg.fine_num_layers = extract_json_int(config_text, "fine_num_layers", 12);
        cfg.fine_num_heads = extract_json_int(config_text, "fine_num_heads", 12);
        cfg.fine_codebook_size = extract_json_int(config_text, "fine_codebook_size", 1056);
        cfg.fine_n_lm_heads = extract_json_int(config_text, "fine_n_lm_heads", 7);
        cfg.fine_seq_length = extract_json_int(config_text, "fine_seq_length", 0);

        // MagpieTTS fields (detected by "magpie_tts": true in config)
        cfg.is_magpie_tts = extract_json_int(config_text, "magpie_tts", 0) != 0;
        if (cfg.is_magpie_tts)
        {
            cfg.magpie_num_codebooks = extract_json_int(config_text, "magpie_num_codebooks", 8);
            cfg.magpie_codebook_size = extract_json_int(config_text, "magpie_codebook_size", 2024);
            cfg.magpie_fps = extract_json_float(config_text, "magpie_fps", 21.5F);
            cfg.magpie_num_speakers = extract_json_int(config_text, "magpie_num_speakers", 5);
            cfg.magpie_encoder_layers = extract_json_int(config_text, "magpie_encoder_layers", 6);
            cfg.magpie_decoder_layers = extract_json_int(config_text, "magpie_decoder_layers", 12);
            cfg.magpie_hidden_size = extract_json_int(config_text, "magpie_hidden_size", cfg.hidden_size);
            cfg.magpie_text_vocab_size = extract_json_int(config_text, "magpie_text_vocab_size", 0);
            cfg.magpie_max_source_positions = extract_json_int(config_text, "magpie_max_source_positions", 2048);
            cfg.magpie_xa_n_heads = extract_json_int(config_text, "magpie_xa_n_heads", 1);
            cfg.magpie_xa_d_head = extract_json_int(config_text, "magpie_xa_d_head", 128);
            cfg.magpie_nemo_path = extract_json_string(config_text, "magpie_nemo_path", "");
            cfg.audio_sample_rate = extract_json_int(config_text, "sample_rate", 22050);
        }
    }

    // Object detection fields
    if (cfg.runtime_strategy == "object_detection")
    {
        cfg.det_num_classes = extract_json_int(config_text, "det_num_classes", 80);
        cfg.det_input_h = extract_json_int(config_text, "det_input_h", 640);
        cfg.det_input_w = extract_json_int(config_text, "det_input_w", 640);
        cfg.det_conf_threshold = extract_json_float(config_text, "det_conf_threshold", 0.5F);
        cfg.det_nms_threshold = extract_json_float(config_text, "det_nms_threshold", 0.45F);
        cfg.det_image_mean = extract_json_float_array(config_text, "image_mean");
        cfg.det_image_std = extract_json_float_array(config_text, "image_std");
    }

    // Neural operator fields
    if (cfg.runtime_strategy == "neural_operator")
    {
        cfg.operator_type = extract_json_string(config_text, "operator_type", "deeponet");
        cfg.num_sensors = extract_json_int(config_text, "num_sensors", 100);
        cfg.spatial_dim = extract_json_int(config_text, "spatial_dim", 2);
        cfg.output_dim = extract_json_int(config_text, "output_dim", 1);
        // FNO-specific fields
        cfg.fno_in_channels = extract_json_int(config_text, "in_channels", 3);
        cfg.fno_out_channels = extract_json_int(config_text, "out_channels", 1);
        cfg.fno_hidden_channels = extract_json_int(config_text, "hidden_channels", 64);
        cfg.fno_grid_h = extract_json_int(config_text, "grid_h", 64);
        cfg.fno_grid_w = extract_json_int(config_text, "grid_w", 64);
    }

    // Encoder-only fields
    if (cfg.runtime_strategy == "encoder_only")
    {
        cfg.type_vocab_size = extract_json_int(config_text, "type_vocab_size", 2);
    }

    // Embedding fields
    if (cfg.runtime_strategy == "embedding")
    {
        cfg.embedding_dim = extract_json_int(config_text, "embedding_dim", cfg.hidden_size);
    }

    // Reranking fields
    if (cfg.runtime_strategy == "reranking")
    {
        cfg.is_reranker = extract_json_int(config_text, "is_reranker", 0) != 0;
    }

    // Speech-to-speech fields
    if (cfg.runtime_strategy == "speech_to_speech")
    {
        cfg.speech_sample_rate = extract_json_int(config_text, "sample_rate", 24000);
        cfg.speech_num_codebooks = extract_json_int(config_text, "num_codebooks", 8);
        cfg.speech_codebook_size = extract_json_int(config_text, "codebook_size", 2048);
        cfg.speech_frame_rate = extract_json_float(config_text, "frame_rate", 12.5F);
        cfg.speech_depth_hidden_size = extract_json_int(config_text, "depth_hidden_size", cfg.hidden_size);
        cfg.speech_depth_num_layers = extract_json_int(config_text, "depth_num_layers", 6);
        cfg.speech_depth_num_heads = extract_json_int(config_text, "depth_num_attention_heads", cfg.num_heads);
        cfg.speech_depth_num_kv_heads = extract_json_int(config_text, "depth_num_key_value_heads", cfg.speech_depth_num_heads);
        // Delay pattern and initial token IDs
        cfg.speech_delays = extract_json_int_array(config_text, "delays", 32);
        cfg.speech_text_initial_token_id = extract_json_int(config_text, "text_initial_token_id", 32000);
        cfg.speech_audio_initial_token_id = extract_json_int(config_text, "audio_initial_token_id", 2048);
        cfg.speech_text_padding_id = extract_json_int(config_text, "text_padding_id", 3);
        cfg.speech_depth_temperature = extract_json_float(config_text, "speech_depth_temperature", 0.0F);
        cfg.speech_depth_top_k = extract_json_int(config_text, "speech_depth_top_k", 0);
        cfg.speech_system_prompt = extract_json_string(config_text, "speech_system_prompt", "");
        // Parse pre-tokenized text prompt IDs: "speech_text_prompt_ids": [493, 298, ...]
        {
            auto pos = config_text.find("\"speech_text_prompt_ids\"");
            if (pos != std::string::npos) {
                auto bracket = config_text.find('[', pos);
                auto end_bracket = config_text.find(']', bracket);
                if (bracket != std::string::npos && end_bracket != std::string::npos) {
                    std::string arr = config_text.substr(bracket + 1, end_bracket - bracket - 1);
                    std::size_t p = 0;
                    while (p < arr.size()) {
                        auto num_start = arr.find_first_of("0123456789", p);
                        if (num_start == std::string::npos) break;
                        cfg.speech_text_prompt_ids.push_back(std::stoi(arr.substr(num_start)));
                        p = arr.find_first_of(",]", num_start);
                        if (p == std::string::npos) break;
                        p++;
                    }
                }
            }
        }
        // Reuse codebook fields for the C++ SpeechConfig
        cfg.codec_n_codebooks = cfg.speech_num_codebooks;
        cfg.codebook_size = cfg.speech_codebook_size;
        cfg.audio_sample_rate = cfg.speech_sample_rate;
        cfg.fine_num_layers = cfg.speech_depth_num_layers;
        cfg.fine_hidden_size = cfg.speech_depth_hidden_size;
        cfg.fine_num_heads = cfg.speech_depth_num_heads;
    }

    // Omni multimodal fields
    if (cfg.runtime_strategy == "omni_multimodal")
    {
        cfg.omni_sample_rate = extract_json_int(config_text, "audio_sample_rate", 24000);
        cfg.omni_num_experts = extract_json_int(config_text, "num_local_experts", 8);
        cfg.omni_num_experts_per_tok = extract_json_int(config_text, "num_experts_per_tok", 2);

        // Talker config
        cfg.omni_talker_hidden_size = extract_json_int(config_text, "omni_talker_hidden_size", 0);
        cfg.omni_talker_num_layers = extract_json_int(config_text, "omni_talker_num_layers", 0);
        cfg.omni_talker_max_cache_length = extract_json_int(config_text, "omni_talker_max_cache_length", 1024);
        cfg.omni_n_codebooks = extract_json_int(config_text, "omni_n_codebooks", 8);
        cfg.omni_codebook_size = extract_json_int(config_text, "omni_codebook_size", 2048);

        // Audio encoder config
        cfg.omni_audio_embed_dim = extract_json_int(config_text, "omni_audio_embed_dim", 1280);
        cfg.omni_audio_num_mel = extract_json_int(config_text, "omni_audio_num_mel", 128);
        cfg.omni_audio_num_layers = extract_json_int(config_text, "omni_audio_num_layers", 0);
        cfg.omni_audio_num_frames = extract_json_int(config_text, "omni_audio_num_frames", 1500);

        // VL fields (Omni also supports vision)
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
        cfg.use_dynamic_shifting = extract_json_int(config_text, "use_dynamic_shifting", 0) != 0;
        cfg.base_shift = extract_json_float(config_text, "base_shift", 0.5F);
        cfg.max_shift = extract_json_float(config_text, "max_shift", 1.15F);
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
        cfg.guidance_embeds = extract_json_int(config_text, "guidance_embeds", 0) != 0;
        cfg.use_rope = extract_json_int(config_text, "use_rope", 1) != 0;
        cfg.vae_scaling_factor = extract_json_float(config_text, "vae_scaling_factor", 0.0F);
        cfg.diffusion_backend_type = extract_json_string(config_text, "diffusion_backend_type", "wan_3d");
    }

    // Embedding fields
    if (cfg.runtime_strategy == "embedding")
    {
        cfg.embedding_dim = extract_json_int(config_text, "embedding_dim", cfg.hidden_size);
    }

    // Reranking fields
    if (cfg.runtime_strategy == "reranking")
    {
        cfg.is_reranker = extract_json_int(config_text, "is_reranker", 0) != 0;
    }

    return cfg;
}

} // namespace trtf
