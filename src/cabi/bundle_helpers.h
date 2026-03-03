#pragma once

// Shared helpers for bundle-based pipeline creation.
// Extracts duplicated tokenizer/engine setup logic from trtf_c.cpp
// so each backend factory can reuse it without copy-paste.

#include "trtf/tokenizer.h"
#include "bundle/bundle_format.h"
#include "cabi/fast_path_config.h"
#include "runtime/trt/trt_engine_lifecycle.h"
#include "runtime/trt/trt_common.h"

#include <memory>
#include <string>
#include <vector>

namespace trtf {

#if TRTF_HAS_TRT

// Pointers into a BundleFile's sections. Non-owning — the BundleFile must
// outlive any use of these pointers.
struct BundleSections {
    const std::vector<char>* plan_data{nullptr};
    const std::vector<char>* vision_plan_data{nullptr};
    const std::vector<char>* config_json_data{nullptr};
    const std::vector<char>* preprocessor_config_data{nullptr};
    const std::vector<char>* tokenizer_json_data{nullptr};
    const std::vector<char>* tokenizer_config_data{nullptr};
    const std::vector<char>* vocab_json_data{nullptr};
    const std::vector<char>* merges_txt_data{nullptr};
    const std::vector<char>* special_tokens_data{nullptr};
    const std::vector<char>* tokenizer_model_data{nullptr};

    // Bark/multi-engine sections
    const std::vector<char>* coarse_engine_plan_data{nullptr};
    const std::vector<char>* fine_engine_plan_data{nullptr};
    const std::vector<char>* codec_engine_plan_data{nullptr};
    const std::vector<char>* semantic_embed_data{nullptr};
    const std::vector<char>* coarse_embed_data{nullptr};
    const std::vector<char>* fine_embed_data{nullptr};
    const std::vector<char>* fine_position_embed_data{nullptr};

    // MagpieTTS sections
    const std::vector<char>* magpie_audio_embed_data{nullptr};
    const std::vector<char>* magpie_text_embed_data{nullptr};
    const std::vector<char>* magpie_context_embed_data{nullptr};
    const std::vector<char>* magpie_context_lengths_data{nullptr};

    // MagpieTTS native IPA tokenizer sections
    const std::vector<char>* magpie_ipa_phoneme_dict_data{nullptr};
    const std::vector<char>* magpie_ipa_heteronyms_data{nullptr};
    const std::vector<char>* magpie_ipa_vocab_data{nullptr};
    const std::vector<char>* magpie_ipa_config_data{nullptr};

    // Omni multimodal sections
    const std::vector<char>* audio_encoder_plan_data{nullptr};
    const std::vector<char>* talker_engine_plan_data{nullptr};
    const std::vector<char>* code2wav_engine_plan_data{nullptr};

    // Speech-to-speech (PersonaPlex/Moshi) sections
    const std::vector<char>* depth_engine_plan_data{nullptr};
    std::vector<const std::vector<char>*> depth_engine_plans;  // per-codebook: depth_engine_plan_0, ...
    const std::vector<char>* mimi_encoder_plan_data{nullptr};
    const std::vector<char>* mimi_decoder_plan_data{nullptr};
    const std::vector<char>* depth_projection_data{nullptr};
    const std::vector<char>* audio_embeddings_data{nullptr};
    const std::vector<char>* temporal_text_embedding_data{nullptr};
    const std::vector<char>* depth_text_embedding_data{nullptr};
    const std::vector<char>* depth_audio_embeddings_data{nullptr};

    // Diffusion sections
    std::vector<const std::vector<char>*> text_encoder_plans;  // text_encoder_0_plan, ...
    const std::vector<char>* denoiser_plan_data{nullptr};
    const std::vector<char>* vae_decoder_plan_data{nullptr};
    const std::vector<char>* preprocessor_weights_data{nullptr};

    // CLIP tokenizer sections (for FLUX dual-tokenizer: CLIP + T5)
    const std::vector<char>* clip_tokenizer_config_data{nullptr};
    const std::vector<char>* clip_vocab_json_data{nullptr};
    const std::vector<char>* clip_merges_txt_data{nullptr};
    const std::vector<char>* clip_special_tokens_data{nullptr};

    // Whisper mel filterbank (baked at build time)
    const std::vector<char>* mel_filterbank_data{nullptr};
};

// Scan bundle sections and populate pointers by name.
BundleSections find_bundle_sections(const BundleFile& bundle);

// Mel filterbank loaded from bundle (for Whisper native mel extraction).
struct MelFilterbank {
    std::vector<float> data;  // [n_freq_bins * n_mel_bins] row-major
    int32_t n_freq_bins{0};
    int32_t n_mel_bins{0};
};

// Load mel filterbank from the "mel_filterbank" bundle section.
// Returns empty MelFilterbank if section is not present (old bundles).
MelFilterbank load_mel_filterbank(const BundleSections& sections);

// Result of extracting a tokenizer from a bundle.
struct TokenizerResult {
    std::unique_ptr<ITokenizer> tokenizer;
    std::string temp_dir;  // caller must transfer ownership to PipelineImpl
};

// Write tokenizer files from bundle sections to a temp dir, then create
// an HfPythonTokenizer. Throws on failure.
TokenizerResult extract_tokenizer_from_bundle(
    const BundleSections& sections,
    const std::string& hf_python,
    bool add_special_tokens = false);

// Extract a CLIP tokenizer from bundle sections (for dual-tokenizer models).
// Writes clip_vocab.json, clip_merges.txt, clip_tokenizer_config.json,
// clip_special_tokens_map.json to a temp dir and creates an HfPythonTokenizer.
TokenizerResult extract_clip_tokenizer_from_bundle(
    const BundleSections& sections,
    const std::string& hf_python);

// Build a DecoderStepEngine from a TRT engine + config.
std::unique_ptr<DecoderStepEngine> make_decoder_engine(
    TrtUniquePtr<nvinfer1::ICudaEngine> trt_engine,
    TrtUniquePtr<nvinfer1::IExecutionContext> exec_ctx,
    const FastPathModelConfig& cfg);

#endif // TRTF_HAS_TRT

} // namespace trtf
