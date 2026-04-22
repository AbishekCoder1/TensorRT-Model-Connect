#pragma once

// Standalone config structs for audio pipelines.
// Extracted from backend headers so that plan headers and the new
// TrtModule-based pipelines can reference configs without pulling in
// the full old-style backend class hierarchy.

#include <cstdint>
#include <string>
#include <vector>

namespace trtf {

/// Configuration for the MagpieTTS pipeline.
struct MagpieTTSConfig {
    int32_t sample_rate{22050};
    int32_t hidden_size{0};
    int32_t num_codebooks{8};
    int32_t codebook_size{2024};
    float frames_per_second{21.5F};
    int32_t num_speakers{5};
    int32_t encoder_layers{6};
    int32_t decoder_layers{12};
    int32_t text_vocab_size{0};
    int32_t max_source_positions{2048};
    int32_t xa_n_heads{1};
    int32_t xa_d_head{128};
    // Sampling
    float temperature{0.6F};
    int32_t top_k{80};
    bool greedy{false};
    // CFG (Classifier-Free Guidance)
    float cfg_scale{2.5F};
    int32_t finished_limit_with_eot{0};
    // NeMo's standard do_tts() path does not hard-stop on text completion, so
    // keep the finished-limit guard opt-in for parity/debugging only.
    bool enable_finished_limit_stop{false};
    // audio_magpie.* namespace replaces the TRTF_MAGPIE_{GREEDY, CFG_SCALE,
    // TEMPERATURE, FINISHED_LIMIT, SEED} env vars. magpie_plugin populates
    // these from ctx.runtime_config at construction.
    int64_t seed{-1}; // -1 -> leave RNG at constructed state
};

/// Configuration for the SpeechToSpeech (PersonaPlex) pipeline.
struct SpeechConfig {
    int32_t sample_rate{24000};
    int32_t num_codebooks{8};
    int32_t codebook_size{2048};
    float frame_rate{12.5F};

    // Temporal transformer
    int32_t temporal_hidden_size{0};
    int32_t temporal_num_layers{0};

    // Depth transformer
    int32_t depth_hidden_size{0};
    int32_t depth_num_layers{6};
    int32_t depth_num_heads{0};
    int32_t depth_num_kv_heads{0};
    int32_t depth_max_cache_length{16};

    // Temporal-to-depth projection matrix [depth_hidden, temporal_hidden]
    // Loaded from bundle 'depth_projection' section.
    std::vector<float> depth_projection;
    int32_t temporal_hidden_for_proj{0}; // temporal_hidden (cols of projection)

    // Per-codebook audio embedding tables for temporal transformer input.
    // Layout: [num_codebooks, audio_vocab_size, temporal_hidden_size] as float32.
    // The temporal input for each frame is the SUM of all codebook embeddings.
    std::vector<float> audio_embeddings;
    int32_t audio_vocab_size{2049}; // per-codebook vocab (Mimi: 2049)

    // Temporal text embedding table [text_vocab, temporal_hidden] as float32.
    // The official Moshi code sums text_emb(text_token) + audio embeddings.
    // During generation, the text token is the PAD token (text_padding_id).
    std::vector<float> temporal_text_embedding;
    int32_t temporal_text_vocab{0};
    int32_t text_padding_id{3}; // PAD token for text during generation

    // Depth decoder text embedding table [depth_text_vocab, depth_hidden] as float32.
    // Used at depth position 0 (text token step) before audio codebook generation.
    std::vector<float> depth_text_embedding;
    int32_t depth_text_vocab{0};

    // Number of output codebooks to send to Mimi decoder (default: 8).
    // The depth transformer generates num_codebooks (16) tokens, but
    // only the first mimi_decode_codebooks (moshi stream) are decoded.
    int32_t mimi_decode_codebooks{8};

    // Depth decoder per-codebook audio embedding tables.
    // Layout: [num_depformer_emb, audio_vocab_size, depth_hidden] as float32.
    // depformer_emb.{i} is used at depth position i+1.
    std::vector<float> depth_audio_embeddings;
    int32_t num_depformer_emb{0};

    // Delay pattern for temporal transformer input alignment.
    // Official Moshi: [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]
    // Length = num_codebooks + 1 (text + 16 audio).
    // delay[k]=0: token from current frame; delay[k]=1: token from previous frame.
    std::vector<int32_t> delays;
    int32_t text_initial_token_id{32000}; // BOS for text stream at first step
    int32_t audio_initial_token_id{2048}; // BOS for audio streams at first step

    // Depth decoder sampling parameters (greedy if temperature <= 0)
    float depth_temperature{0.0F}; // 0 = greedy, 0.8 = official PersonaPlex
    int32_t depth_top_k{0};        // 0 = greedy, 250 = official PersonaPlex

    // Optional text EOS token used to stop long-form speech generation early.
    // -1 disables EOS-based early stop.
    int32_t text_eos_token_id{-1};

    // System prompt for text prompt injection (primes temporal KV cache)
    std::string system_prompt;
    // Pre-tokenized system prompt IDs (avoids runtime tokenization)
    std::vector<int32_t> text_prompt_ids;

    // Optional Python path used only for runtime text tokenization fallback.
    std::string hf_python;
};

/// Configuration for the Qwen3-Omni multimodal pipeline.
struct OmniConfig {
    int32_t sample_rate{24000};

    // Thinker MoE config (text decoder)
    int32_t thinker_hidden_size{0};
    int32_t thinker_num_layers{0};
    int32_t thinker_num_heads{0};
    int32_t num_experts{8};
    int32_t num_experts_per_tok{2};

    // Audio encoder config
    int32_t audio_embed_dim{1280};
    int32_t audio_num_mel{128};
    int32_t audio_num_layers{0};
    int32_t audio_num_frames{1500};

    // Talker config
    int32_t talker_hidden_size{0};
    int32_t talker_num_layers{0};
    int32_t talker_n_codebooks{8};
    int32_t talker_codebook_size{2048};

    // Code2Wav config
    int32_t code2wav_upsample_factor{320};
    int32_t code2wav_max_frames{256};

    // Generation parameters
    bool greedy{false};
    float temperature{0.7F};
    int32_t top_k{50};
};

} // namespace trtf
