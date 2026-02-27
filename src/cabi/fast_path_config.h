#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace trtf {

struct FastPathModelConfig {
    int32_t vocab_size{0};
    int32_t hidden_size{0};
    int32_t num_layers{1};
    int32_t num_heads{1};
    int32_t num_kv_heads{1};
    int32_t head_dim{0};
    int32_t attention_size{0};
    int32_t max_cache_length{32};
    int32_t id_bos{-1};
    int32_t id_eos{-1};

    // Runtime strategy determines which backend and state management to use.
    // "decoder_kv_cache" (default): standard attention-based decoder with KV cache
    // "decoder_moe":  MoE decoder (same KV cache, routing in TRT graph)
    // "ssm_recurrent": Mamba/SSM (conv_state + ssm_state, no KV cache)
    // "rwkv_recurrent": RWKV (5 recurrent state vectors per layer, no KV cache)
    // "vision_language": two-engine (vision encoder + text decoder)
    std::string runtime_strategy{"decoder_kv_cache"};

    // Mamba/SSM-specific fields (used when runtime_strategy == "ssm_recurrent")
    int32_t d_inner{0};      // intermediate_size / d_inner
    int32_t state_size{16};  // SSM state dimension
    int32_t conv_kernel{4};  // causal conv1d kernel size

    // Hybrid Mamba-Attention fields (used when runtime_strategy == "hybrid_mamba_attention")
    int32_t num_mamba_layers{0};
    int32_t num_attention_layers{0};
    int32_t mamba_d_state{128};
    int32_t mamba_d_conv{4};
    int32_t mamba_nheads{0};
    int32_t mamba_head_dim{0};
    int32_t conv_dim{0};          // d_inner + 2*n_groups*d_state (conv1d channels)
    std::vector<std::string> layer_types;  // "mamba2", "mlp", "attention"

    // Whisper/speech-to-text fields (used when runtime_strategy == "speech_to_text")
    int32_t num_mel_bins{80};
    int32_t max_source_positions{1500};
    int32_t max_target_positions{448};
    int32_t encoder_layers{0};
    int32_t decoder_layers{0};

    // Vision-Language fields (used when runtime_strategy == "vision_language")
    bool has_vision_engine{false};  // bundle contains vision_engine_plan
    bool embed_input{false};        // text decoder uses input_embed mode
    int32_t image_token_id{-1};     // special token ID for <|image_pad|>
    int32_t vision_output_dim{0};   // vision encoder output feature dim (= text hidden)
    int32_t fixed_image_size{448};  // image size the vision engine was compiled for
    int32_t num_image_pad_tokens{0}; // number of image pad tokens per image
    std::string vl_prompt_template; // prompt template with {image_pads} and {prompt}
    std::string image_token_str;    // string for one image pad token (e.g. "<|image_pad|>")
    bool tokenizer_add_special_tokens{false}; // whether tokenizer should add BOS/EOS
    bool tokenizer_add_special_tokens_present{false}; // true if bundle explicitly sets the field

    // Segmentation fields (used when runtime_strategy == "segmentation")
    int32_t num_classes{150};
    int32_t input_image_h{512};
    int32_t input_image_w{512};
    int32_t output_h{128};
    int32_t output_w{128};
    std::vector<float> seg_image_mean;
    std::vector<float> seg_image_std;

    // SAM prompted segmentation fields (used when runtime_strategy == "prompted_segmentation")
    int32_t sam_image_embedding_size{64};
    int32_t sam_decoder_hidden_size{256};
    int32_t sam_num_mask_outputs{4};
    int32_t sam_num_multimask_outputs{3};
    std::vector<float> sam_point_embed_fg;
    std::vector<float> sam_point_embed_bg;
    std::vector<float> sam_not_a_point_embed;
    std::vector<float> sam_shared_image_pe;

    // Audio/Bark fields (used when runtime_strategy == "text_to_audio")
    int32_t audio_sample_rate{24000};
    int32_t semantic_vocab_size{10000};
    int32_t coarse_vocab_size{1024};
    int32_t fine_vocab_size{1024};
    int32_t n_coarse_codebooks{2};
    int32_t n_fine_codebooks{8};
    int32_t semantic_pad_token{10000};
    int32_t semantic_infer_token{129599};
    int32_t coarse_semantic_pad_token{12048};
    int32_t coarse_infer_token{12050};
    int32_t text_encoding_offset{10048};
    int32_t text_pad_token{129595};
    int32_t semantic_input_vocab{129600};
    int32_t coarse_input_vocab{12096};
    int32_t codebook_size{1024};
    // Sub-model dimensions for coarse engine (semantic uses main hidden/layers/heads)
    int32_t coarse_hidden_size{0};
    int32_t coarse_num_layers{0};
    int32_t coarse_num_heads{0};
    int32_t coarse_max_cache_length{1024};
    // Codec (EnCodec) engine config
    int32_t codec_seq_length{0};       // max frames the codec engine was compiled for
    int32_t codec_upsample_factor{320}; // total upsample ratio (8*5*4*2)
    int32_t codec_n_codebooks{8};      // number of codebooks in codec engine input
    // Fine model config
    int32_t fine_hidden_size{768};
    int32_t fine_num_layers{12};
    int32_t fine_num_heads{12};
    int32_t fine_codebook_size{1056};
    int32_t fine_n_lm_heads{7};
    int32_t fine_seq_length{0};        // 0 = no fine engine

    // Object detection fields (used when runtime_strategy == "object_detection")
    int32_t det_num_classes{80};
    int32_t det_input_h{640};
    int32_t det_input_w{640};
    float det_conf_threshold{0.5F};
    float det_nms_threshold{0.45F};
    std::vector<float> det_image_mean;
    std::vector<float> det_image_std;

    // Neural operator fields (used when runtime_strategy == "neural_operator")
    std::string operator_type{"deeponet"};
    int32_t num_sensors{100};
    int32_t spatial_dim{2};
    int32_t output_dim{1};
    // FNO-specific fields (operator_type == "fno")
    int32_t fno_in_channels{3};
    int32_t fno_out_channels{1};
    int32_t fno_hidden_channels{64};
    int32_t fno_grid_h{64};
    int32_t fno_grid_w{64};

    // Encoder-only fields (used when runtime_strategy == "encoder_only")
    int32_t type_vocab_size{2};  // BERT token type embeddings (segment A/B)

    // Embedding fields (used when runtime_strategy == "embedding")
    int32_t embedding_dim{0};  // output embedding dimension (0 = same as hidden_size)

    // Reranking fields (used when runtime_strategy == "reranking")
    bool is_reranker{false};  // true for reranking models

    // Speech-to-speech fields (used when runtime_strategy == "speech_to_speech")
    int32_t speech_sample_rate{24000};
    int32_t speech_num_codebooks{8};
    int32_t speech_codebook_size{2048};
    float speech_frame_rate{12.5F};
    int32_t speech_depth_hidden_size{0};
    int32_t speech_depth_num_layers{6};
    int32_t speech_depth_num_heads{0};
    int32_t speech_depth_num_kv_heads{0};
    std::vector<int32_t> speech_delays;  // delay pattern per codebook
    int32_t speech_text_initial_token_id{32000};
    int32_t speech_audio_initial_token_id{2048};
    int32_t speech_text_padding_id{3};
    float speech_depth_temperature{0.0F};   // 0 = greedy, 0.8 = official PersonaPlex
    int32_t speech_depth_top_k{0};          // 0 = greedy, 250 = official PersonaPlex
    std::string speech_system_prompt;       // text prompt to inject before user audio
    std::vector<int32_t> speech_text_prompt_ids;  // pre-tokenized prompt IDs

    // Omni multimodal fields (used when runtime_strategy == "omni_multimodal")
    int32_t omni_sample_rate{24000};
    int32_t omni_num_experts{8};
    int32_t omni_num_experts_per_tok{2};
    int32_t omni_talker_hidden_size{0};
    int32_t omni_talker_num_layers{0};
    int32_t omni_talker_max_cache_length{1024};
    int32_t omni_n_codebooks{8};
    int32_t omni_codebook_size{2048};
    int32_t omni_audio_embed_dim{1280};
    int32_t omni_audio_num_mel{128};
    int32_t omni_audio_num_layers{0};
    int32_t omni_audio_num_frames{1500};

    // Diffusion fields (used when runtime_strategy == "diffusion")
    int32_t num_text_encoders{0};
    std::string scheduler;              // "flow_match_euler", etc.
    int32_t num_inference_steps{50};
    float guidance_scale{5.0F};
    float flow_shift{1.0F};
    bool use_dynamic_shifting{false};
    float base_shift{0.5F};
    float max_shift{1.15F};
    int32_t video_height{480};
    int32_t video_width{832};
    int32_t video_num_frames{81};
    int32_t z_dim{16};
    int32_t scale_factor_temporal{4};
    int32_t scale_factor_spatial{8};
    int32_t dit_dim{1536};
    int32_t dit_num_heads{12};
    int32_t dit_num_layers{30};
    int32_t freq_dim{256};
    int32_t num_vae_caches{0};
    int32_t text_encoder_dim{4096};
    int32_t text_seq_len{512};
    std::vector<float> latents_mean;
    std::vector<float> latents_std;
    std::vector<int32_t> patch_size;    // [pt, ph, pw]
    std::string vae_model_id;
    bool guidance_embeds{false};
    std::string diffusion_backend_type{"wan_3d"};
};

// Parse model configuration from config.json text for the fast path.
// max_cache_length_override: value from TRTF_MAX_CACHE_LENGTH env, or -1 to use config.
FastPathModelConfig parse_fast_path_config(const std::string& config_text, int32_t max_cache_length_override);

} // namespace trtf
