#include "cabi/config/fast_path_config.h"
#include "utils/json_helpers.h"

#include <algorithm>
#include <initializer_list>

namespace trtf {
namespace {

int32_t extract_first_nonzero_int(const std::string& config_text, std::initializer_list<const char*> keys)
{
    for (const char* key : keys)
    {
        int32_t value = extract_json_int(config_text, key, 0);
        if (value != 0)
        {
            return value;
        }
    }
    return 0;
}

void parse_model_dimensions(const std::string& config_text, FastPathModelConfig& cfg)
{
    cfg.vocab_size = extract_json_int(config_text, "vocab_size", 0);

    cfg.hidden_size = extract_first_nonzero_int(config_text, {"hidden_size", "n_embd", "d_model", "n_embed", "dim"});

    cfg.num_layers = std::max(extract_first_nonzero_int(config_text, {"num_hidden_layers", "n_layer", "num_layers", "n_layers"}), 1);
    cfg.num_heads = std::max(extract_first_nonzero_int(
                                 config_text, {"num_attention_heads", "n_head", "attention_heads", "num_heads", "n_heads", "decoder_attention_heads"}),
        1);

    int32_t decoder_layers = extract_json_int(config_text, "decoder_layers", 0);
    if (decoder_layers > 0)
    {
        cfg.num_layers = decoder_layers;
    }

    cfg.num_kv_heads = std::max(extract_json_int(config_text, "num_key_value_heads", cfg.num_heads), 1);
    cfg.head_dim = extract_json_int(config_text, "head_dim", cfg.hidden_size / cfg.num_heads);
    cfg.attention_size = cfg.num_heads * cfg.head_dim;
}

void parse_cache_and_token_ids(const std::string& config_text, int32_t max_cache_length_override, FastPathModelConfig& cfg)
{
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
}

void parse_runtime_strategy_and_tokenizer_flags(const std::string& config_text, FastPathModelConfig& cfg)
{
    cfg.runtime_strategy = extract_json_string(config_text, "runtime_strategy", "decoder_kv_cache");

    int32_t raw = extract_json_int(config_text, "tokenizer_add_special_tokens", -1);
    if (raw >= 0)
    {
        cfg.tokenizer_add_special_tokens = (raw != 0);
        cfg.tokenizer_add_special_tokens_present = true;
    }
}

void parse_ssm_recurrent_config(const std::string& config_text, FastPathModelConfig& cfg)
{
    cfg.d_inner = extract_json_int(config_text, "intermediate_size", 0);
    if (cfg.d_inner == 0)
    {
        cfg.d_inner = extract_json_int(config_text, "d_inner", cfg.hidden_size * 2);
    }
    cfg.state_size = extract_json_int(config_text, "state_size", 16);
    cfg.conv_kernel = extract_json_int(config_text, "conv_kernel", 4);
}

void parse_hybrid_mamba_attention_config(const std::string& config_text, FastPathModelConfig& cfg)
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

void parse_speech_to_text_config(const std::string& config_text, FastPathModelConfig& cfg)
{
    cfg.num_mel_bins = extract_json_int(config_text, "num_mel_bins", 80);
    cfg.max_source_positions = extract_json_int(config_text, "max_source_positions", 1500);
    cfg.max_target_positions = extract_json_int(config_text, "max_target_positions", 448);
    cfg.encoder_layers = extract_json_int(config_text, "encoder_layers", cfg.num_layers);
    cfg.decoder_layers = extract_json_int(config_text, "decoder_layers", cfg.num_layers);
    cfg.mel_n_fft = extract_json_int(config_text, "mel_n_fft", 400);
    cfg.mel_hop_length = extract_json_int(config_text, "mel_hop_length", 160);
    cfg.mel_chunk_length = extract_json_int(config_text, "mel_chunk_length", 30);
    cfg.mel_sampling_rate = extract_json_int(config_text, "mel_sampling_rate", 16000);
    cfg.mel_length = extract_json_int(config_text, "mel_length", 0);
    cfg.eot_token_id = extract_json_int(config_text, "eot_token_id", -1);
    cfg.decoder_start_token_ids = extract_json_int_array(config_text, "decoder_start_token_ids");
}

void parse_vision_language_fields(const std::string& config_text, FastPathModelConfig& cfg)
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

void parse_vision_language_config(const std::string& config_text, FastPathModelConfig& cfg)
{
    parse_vision_language_fields(config_text, cfg);
}

void parse_segmentation_config(const std::string& config_text, FastPathModelConfig& cfg)
{
    cfg.num_classes = extract_json_int(config_text, "num_classes", 150);
    cfg.input_image_h = extract_json_int(config_text, "input_image_h", 512);
    cfg.input_image_w = extract_json_int(config_text, "input_image_w", 512);
    cfg.output_h = extract_json_int(config_text, "output_h", 128);
    cfg.output_w = extract_json_int(config_text, "output_w", 128);
    cfg.seg_image_mean = extract_json_float_array(config_text, "image_mean");
    cfg.seg_image_std = extract_json_float_array(config_text, "image_std");
}

void parse_prompted_segmentation_config(const std::string& config_text, FastPathModelConfig& cfg)
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

void parse_magpie_tts_config(const std::string& config_text, FastPathModelConfig& cfg)
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
    cfg.magpie_temperature = extract_json_float(config_text, "magpie_temperature", 0.6F);
    cfg.magpie_cfg_scale = extract_json_float(config_text, "magpie_cfg_scale", 2.5F);
    cfg.magpie_finished_limit_with_eot = extract_json_int(config_text, "magpie_finished_limit_with_eot", 0);
    cfg.audio_sample_rate = extract_json_int(config_text, "sample_rate", 22050);
}

void parse_text_to_audio_config(const std::string& config_text, FastPathModelConfig& cfg)
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
    cfg.coarse_hidden_size = extract_json_int(config_text, "coarse_hidden_size", cfg.hidden_size);
    cfg.coarse_num_layers = extract_json_int(config_text, "coarse_num_layers", cfg.num_layers);
    cfg.coarse_num_heads = extract_json_int(config_text, "coarse_num_heads", cfg.num_heads);
    cfg.coarse_max_cache_length = extract_json_int(config_text, "coarse_max_cache_length", 1024);
    cfg.codec_seq_length = extract_json_int(config_text, "codec_seq_length", 0);
    cfg.codec_upsample_factor = extract_json_int(config_text, "codec_upsample_factor", 320);
    cfg.codec_n_codebooks = extract_json_int(config_text, "codec_n_codebooks", 8);
    cfg.fine_hidden_size = extract_json_int(config_text, "fine_hidden_size", cfg.hidden_size);
    cfg.fine_num_layers = extract_json_int(config_text, "fine_num_layers", 12);
    cfg.fine_num_heads = extract_json_int(config_text, "fine_num_heads", 12);
    cfg.fine_codebook_size = extract_json_int(config_text, "fine_codebook_size", 1056);
    cfg.fine_n_lm_heads = extract_json_int(config_text, "fine_n_lm_heads", 7);
    cfg.fine_seq_length = extract_json_int(config_text, "fine_seq_length", 0);
    cfg.is_magpie_tts = extract_json_int(config_text, "magpie_tts", 0) != 0;
    if (cfg.is_magpie_tts)
    {
        parse_magpie_tts_config(config_text, cfg);
    }
}

void parse_object_detection_config(const std::string& config_text, FastPathModelConfig& cfg)
{
    cfg.det_num_classes = extract_json_int(config_text, "det_num_classes", 80);
    cfg.det_input_h = extract_json_int(config_text, "det_input_h", 640);
    cfg.det_input_w = extract_json_int(config_text, "det_input_w", 640);
    cfg.det_conf_threshold = extract_json_float(config_text, "det_conf_threshold", 0.5F);
    cfg.det_nms_threshold = extract_json_float(config_text, "det_nms_threshold", 0.45F);
    cfg.det_image_mean = extract_json_float_array(config_text, "image_mean");
    cfg.det_image_std = extract_json_float_array(config_text, "image_std");
}

void parse_neural_operator_config(const std::string& config_text, FastPathModelConfig& cfg)
{
    cfg.operator_type = extract_json_string(config_text, "operator_type", "deeponet");
    cfg.num_sensors = extract_json_int(config_text, "num_sensors", 100);
    cfg.spatial_dim = extract_json_int(config_text, "spatial_dim", 2);
    cfg.output_dim = extract_json_int(config_text, "output_dim", 1);
    cfg.fno_in_channels = extract_json_int(config_text, "in_channels", 3);
    cfg.fno_out_channels = extract_json_int(config_text, "out_channels", 1);
    cfg.fno_hidden_channels = extract_json_int(config_text, "hidden_channels", 64);
    cfg.fno_grid_h = extract_json_int(config_text, "grid_h", 64);
    cfg.fno_grid_w = extract_json_int(config_text, "grid_w", 64);
}

void parse_encoder_only_config(const std::string& config_text, FastPathModelConfig& cfg)
{
    cfg.type_vocab_size = extract_json_int(config_text, "type_vocab_size", 2);
}

void parse_embedding_config(const std::string& config_text, FastPathModelConfig& cfg)
{
    cfg.embedding_dim = extract_json_int(config_text, "embedding_dim", cfg.hidden_size);
}

void parse_reranking_config(const std::string& config_text, FastPathModelConfig& cfg)
{
    cfg.is_reranker = extract_json_int(config_text, "is_reranker", 0) != 0;
}

std::vector<int32_t> parse_speech_text_prompt_ids(const std::string& config_text)
{
    std::vector<int32_t> prompt_ids;
    const std::size_t pos = config_text.find("\"speech_text_prompt_ids\"");
    if (pos == std::string::npos)
    {
        return prompt_ids;
    }

    const std::size_t bracket = config_text.find('[', pos);
    const std::size_t end_bracket = config_text.find(']', bracket);
    if (bracket == std::string::npos || end_bracket == std::string::npos)
    {
        return prompt_ids;
    }

    std::string arr = config_text.substr(bracket + 1, end_bracket - bracket - 1);
    std::size_t p = 0;
    while (p < arr.size())
    {
        const std::size_t num_start = arr.find_first_of("0123456789", p);
        if (num_start == std::string::npos)
        {
            break;
        }
        prompt_ids.push_back(std::stoi(arr.substr(num_start)));
        p = arr.find_first_of(",]", num_start);
        if (p == std::string::npos)
        {
            break;
        }
        ++p;
    }
    return prompt_ids;
}

void parse_speech_to_speech_config(const std::string& config_text, FastPathModelConfig& cfg)
{
    cfg.speech_sample_rate = extract_json_int(config_text, "sample_rate", 24000);
    cfg.speech_num_codebooks = extract_json_int(config_text, "num_codebooks", 8);
    cfg.speech_codebook_size = extract_json_int(config_text, "codebook_size", 2048);
    cfg.speech_frame_rate = extract_json_float(config_text, "frame_rate", 12.5F);
    cfg.speech_depth_hidden_size = extract_json_int(config_text, "depth_hidden_size", cfg.hidden_size);
    cfg.speech_depth_num_layers = extract_json_int(config_text, "depth_num_layers", 6);
    cfg.speech_depth_num_heads = extract_json_int(config_text, "depth_num_attention_heads", cfg.num_heads);
    cfg.speech_depth_num_kv_heads = extract_json_int(config_text, "depth_num_key_value_heads", cfg.speech_depth_num_heads);
    cfg.speech_delays = extract_json_int_array(config_text, "delays", 32);
    cfg.speech_text_initial_token_id = extract_json_int(config_text, "text_initial_token_id", 32000);
    cfg.speech_audio_initial_token_id = extract_json_int(config_text, "audio_initial_token_id", 2048);
    cfg.speech_text_padding_id = extract_json_int(config_text, "text_padding_id", 3);
    cfg.speech_depth_temperature = extract_json_float(config_text, "speech_depth_temperature", 0.0F);
    cfg.speech_depth_top_k = extract_json_int(config_text, "speech_depth_top_k", 0);
    cfg.speech_system_prompt = extract_json_string(config_text, "speech_system_prompt", "");
    cfg.speech_text_prompt_ids = parse_speech_text_prompt_ids(config_text);
    cfg.codec_n_codebooks = cfg.speech_num_codebooks;
    cfg.codebook_size = cfg.speech_codebook_size;
    cfg.audio_sample_rate = cfg.speech_sample_rate;
    cfg.fine_num_layers = cfg.speech_depth_num_layers;
    cfg.fine_hidden_size = cfg.speech_depth_hidden_size;
    cfg.fine_num_heads = cfg.speech_depth_num_heads;
}

void parse_omni_multimodal_config(const std::string& config_text, FastPathModelConfig& cfg)
{
    cfg.omni_sample_rate = extract_json_int(config_text, "audio_sample_rate", 24000);
    cfg.omni_num_experts = extract_json_int(config_text, "num_local_experts", 8);
    cfg.omni_num_experts_per_tok = extract_json_int(config_text, "num_experts_per_tok", 2);
    cfg.omni_talker_hidden_size = extract_json_int(config_text, "omni_talker_hidden_size", 0);
    cfg.omni_talker_num_layers = extract_json_int(config_text, "omni_talker_num_layers", 0);
    cfg.omni_talker_max_cache_length = extract_json_int(config_text, "omni_talker_max_cache_length", 1024);
    cfg.omni_n_codebooks = extract_json_int(config_text, "omni_n_codebooks", 8);
    cfg.omni_codebook_size = extract_json_int(config_text, "omni_codebook_size", 2048);
    cfg.omni_audio_embed_dim = extract_json_int(config_text, "omni_audio_embed_dim", 1280);
    cfg.omni_audio_num_mel = extract_json_int(config_text, "omni_audio_num_mel", 128);
    cfg.omni_audio_num_layers = extract_json_int(config_text, "omni_audio_num_layers", 0);
    cfg.omni_audio_num_frames = extract_json_int(config_text, "omni_audio_num_frames", 1500);
    parse_vision_language_fields(config_text, cfg);
}

void parse_diffusion_config(const std::string& config_text, FastPathModelConfig& cfg)
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
    cfg.axes_dims_rope = extract_json_int_array(config_text, "axes_dims_rope");
    cfg.rope_theta = extract_json_float(config_text, "rope_theta", 10000.0F);
    cfg.vae_model_id = extract_json_string(config_text, "vae_model_id", "");
    cfg.guidance_embeds = extract_json_int(config_text, "guidance_embeds", 0) != 0;
    cfg.use_rope = extract_json_int(config_text, "use_rope", 1) != 0;
    cfg.vae_scaling_factor = extract_json_float(config_text, "vae_scaling_factor", 0.0F);
    cfg.diffusion_backend_type = extract_json_string(config_text, "diffusion_backend_type", "wan_3d");
}

using StrategyParser = void (*)(const std::string&, FastPathModelConfig&);

struct StrategyHandler {
    const char* strategy;
    StrategyParser parser;
};

void parse_strategy_specific_fields(const std::string& config_text, FastPathModelConfig& cfg)
{
    static constexpr StrategyHandler kHandlers[] = {
        {"ssm_recurrent", parse_ssm_recurrent_config},
        {"hybrid_mamba_attention", parse_hybrid_mamba_attention_config},
        {"speech_to_text", parse_speech_to_text_config},
        {"vision_language", parse_vision_language_config},
        {"segmentation", parse_segmentation_config},
        {"prompted_segmentation", parse_prompted_segmentation_config},
        {"text_to_audio", parse_text_to_audio_config},
        {"object_detection", parse_object_detection_config},
        {"neural_operator", parse_neural_operator_config},
        {"encoder_only", parse_encoder_only_config},
        {"embedding", parse_embedding_config},
        {"reranking", parse_reranking_config},
        {"speech_to_speech", parse_speech_to_speech_config},
        {"omni_multimodal", parse_omni_multimodal_config},
        {"diffusion", parse_diffusion_config},
    };

    for (const auto& handler : kHandlers)
    {
        if (cfg.runtime_strategy == handler.strategy)
        {
            handler.parser(config_text, cfg);
            return;
        }
    }
}

} // namespace

FastPathModelConfig parse_fast_path_config(const std::string& config_text, int32_t max_cache_length_override)
{
    FastPathModelConfig cfg;
    parse_model_dimensions(config_text, cfg);
    parse_cache_and_token_ids(config_text, max_cache_length_override, cfg);
    parse_runtime_strategy_and_tokenizer_flags(config_text, cfg);
    parse_strategy_specific_fields(config_text, cfg);
    return cfg;
}

} // namespace trtf
