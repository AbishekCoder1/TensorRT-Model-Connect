#pragma once

// Pipeline plugin interface for registry-based pipeline dispatch.
// Each plugin handles one or more runtime_strategy values, parsing its own
// config from raw JSON and extracting its own bundle sections.

#include "trtf/pipeline.h"

#include <cstdint>
#include <memory>
#include <string>

namespace trtf {

// Forward declaration — defined in src/bundle/bundle_format.h (internal).
// PipelineContext holds a const reference, so no full definition needed here.
struct BundleFile;

// Universal base config — the ~10 fields every pipeline needs.
// Plugins parse strategy-specific fields directly from config_json.
struct BaseConfig {
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
    std::string runtime_strategy{"decoder_kv_cache"};
    bool tokenizer_add_special_tokens{false};
    bool tokenizer_add_special_tokens_present{false};
};

// Parse universal base config from config.json text.
BaseConfig parse_base_config(const std::string& config_text, int32_t max_cache_length_override);

// Context passed to each plugin's create() method. Non-owning references
// to the bundle and parsed base config. The BundleFile must outlive the
// pipeline being created (it does — PipelineFactory::from_bundle() holds it).
struct PipelineContext {
    const BundleFile& bundle;
    const BaseConfig& config;
    const std::string& config_json;     // raw JSON text from bundle
    const std::string& hf_python;       // path to HF Python interpreter
    const std::string& bundle_path;     // filesystem path to .trtfb file
};

// Plugin interface. Each plugin registers itself with the PipelineRegistry
// via the REGISTER_PIPELINE_PLUGIN macro. The registry calls create() when
// the bundle's runtime_strategy matches one of the plugin's registered keys.
class IPipelinePlugin {
public:
    virtual ~IPipelinePlugin() = default;

    // Create a pipeline from the given context. The plugin is responsible for:
    // 1. Parsing strategy-specific config from ctx.config_json
    // 2. Extracting needed sections from ctx.bundle via find_section()
    // 3. Loading TRT engines, creating caches, tokenizers, etc.
    // 4. Returning the fully constructed pipeline
    virtual std::unique_ptr<IPipeline> create(const PipelineContext& ctx) = 0;
};

} // namespace trtf
