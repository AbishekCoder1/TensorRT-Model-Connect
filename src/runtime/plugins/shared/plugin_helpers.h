#pragma once

// Shared helper functions for pipeline plugins.
// Extracted from pipeline_factory.cpp's anonymous namespace so all
// strategy plugins can reuse TRT module loading, tokenizer creation,
// KV-dim computation, and data-section conversion utilities.

#include "bundle/bundle_format.h"
#include "bundle/bundle_view.h"
#include "runtime/core/trt_common.h"
#include "runtime/pipelines/recurrent_pipeline.h"
#include "trtf/runtime/inference_state.h"
#include "trtf/runtime/kv_cache.h"
#include "trtf/runtime/pipeline_plugin.h"
#include "trtf/runtime/trt_module.h"
#include "trtf/tokenizer.h"

#include <memory>
#include <string>
#include <vector>

namespace trtf {

#if TRTF_HAS_TRT

// A loaded TRT engine + its CUDA stream, ready for inference.
struct LoadedModule {
    std::unique_ptr<TrtModule> module;
    std::shared_ptr<CudaStream> stream;
};

// Load a TRT engine from a serialized plan. Throws on failure.
// If shared_stream is provided, reuses it; otherwise creates a new one.
// profile_idx selects which optimization profile to use (default 0).
LoadedModule load_trt_module_from_plan(const std::vector<char>* plan, const char* label,
                                       std::shared_ptr<CudaStream> shared_stream = nullptr,
                                       int32_t profile_idx = 0);

// Like load_trt_module_from_plan but returns empty LoadedModule on failure
// instead of throwing (for optional engines).
LoadedModule try_load_trt_module_from_plan(const std::vector<char>* plan, const char* label,
                                           std::shared_ptr<CudaStream> shared_stream,
                                           int32_t profile_idx = 0);

// Load an optional TRT module, returning nullptr if the plan is absent.
// On deserialization failure, returns nullptr (does not throw).
std::unique_ptr<TrtModule> extract_optional_module(const std::vector<char>* plan, const char* label,
                                                   std::shared_ptr<CudaStream> shared_stream);

// Detect whether the bundle's config requests add_special_tokens for the tokenizer.
bool detect_add_special_tokens(const BundleFile& bundle);

// Check if the bundle's tokenizer.json describes a BPE model.
bool is_bpe_tokenizer_json(const BundleFile& bundle);

// Try to create a native C++ BPE tokenizer from the bundle's tokenizer.json.
// Returns nullptr if the section is absent or the model is non-BPE.
// If throw_on_failure is true, throws instead of returning nullptr on BPE parse errors.
std::shared_ptr<ITokenizer> try_create_native_bpe(const BundleFile& bundle, bool add_special,
                                                  bool throw_on_failure);

// Try to create a native C++ tokenizer from the bundle's tokenizer.json.
// Attempts: BPE → WordPiece → Unigram. Returns nullptr if none match.
std::shared_ptr<ITokenizer> try_create_native_tokenizer(const BundleFile& bundle,
                                                        bool add_special_tokens);

// Create a native tokenizer from bundle. Tries BPE → WordPiece → Unigram.
// Returns nullptr if no native tokenizer matches.
std::shared_ptr<ITokenizer> create_tokenizer_from_bundle(const BundleFile& bundle);

// Compute the KV cache dimension from model config.
int32_t compute_kv_dim(const BaseConfig& cfg);

// Convert the BaseConfig precision string ("fp16", "bf16", "fp32") to a DType
// for use as KV cache element type.
DType cache_dtype_from_precision(const std::string& precision);

// Build a RecurrentGenConfig from model config fields.
RecurrentGenConfig make_recurrent_gen_config(const BaseConfig& cfg);

// Reinterpret a raw char section as a vector of floats.
std::vector<float> section_to_floats(const std::vector<char>* sec);

// Reinterpret a raw char section as a vector of int32_t.
std::vector<int32_t> section_to_int32s(const std::vector<char>* sec);

// Return true if the section pointer is non-null and non-empty.
bool has_section_data(const std::vector<char>* d);

// ─── BundleFile-based helpers ───

// Mel filterbank loaded from bundle (for Whisper native mel extraction).
struct MelFilterbank {
    std::vector<float> data; // [n_freq_bins * n_mel_bins] row-major
    int32_t n_freq_bins{0};
    int32_t n_mel_bins{0};
};

// Load mel filterbank from the "mel_filterbank" bundle section.
// Returns empty MelFilterbank if section is not present (old bundles).
MelFilterbank load_mel_filterbank(const BundleFile& bundle);

// Create a native BPE tokenizer from the CLIP tokenizer sections in the bundle.
// Used for dual-tokenizer models (e.g., FLUX: CLIP + T5).
// Returns nullptr if clip_tokenizer.json section is absent.
std::unique_ptr<ITokenizer> create_clip_tokenizer_from_bundle(const BundleFile& bundle);

#endif // TRTF_HAS_TRT

} // namespace trtf
