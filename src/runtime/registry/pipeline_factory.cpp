#include "trtf/runtime/pipeline_factory.h"
#include "trtf/runtime/pipeline_plugin.h"
#include "trtf/runtime/pipeline_registry.h"
#include "trtf/runtime/trt_backend.h"
#include "runtime/backend/backend_loader.h"
#include "bundle/bundle_format.h"
#include "utils/json_helpers.h"

#include <iostream>
#include <stdexcept>
#include <string>

namespace trtf {

namespace {

#if TRTF_HAS_TRT

// Rewrite legacy ambiguous strategy strings from old bundles into their
// unambiguous per-model equivalents. New bundles already use the split strings.
bool json_field_is_truthy(const std::string& config_text, const char* key) {
    auto pos = config_text.find(std::string("\"") + key + "\"");
    if (pos == std::string::npos)
        return false;
    auto colon = config_text.find(':', pos);
    if (colon == std::string::npos)
        return false;
    auto rest = config_text.substr(colon + 1, 20);
    return rest.find("true") != std::string::npos || rest.find('1') != std::string::npos;
}

std::string json_field_substr(const std::string& config_text, const char* key) {
    auto pos = config_text.find(std::string("\"") + key + "\"");
    if (pos == std::string::npos)
        return "";
    auto colon = config_text.find(':', pos);
    if (colon == std::string::npos)
        return "";
    return config_text.substr(colon, 40);
}

std::string normalize_legacy_strategy(const std::string& strategy, const std::string& config_text) {
    if (strategy == "text_to_audio") {
        return json_field_is_truthy(config_text, "magpie_tts") ? "text_to_audio_magpie"
                                                               : "text_to_audio_bark";
    }
    if (strategy == "diffusion") {
        auto bt = json_field_substr(config_text, "diffusion_backend_type");
        if (bt.find("flux") != std::string::npos)
            return "diffusion_flux";
        if (bt.find("z_image") != std::string::npos)
            return "diffusion_zimage";
        if (bt.find("pixart") != std::string::npos)
            return "diffusion_pixart";
        return "diffusion_wan";
    }
    // Legacy torch-trt diffusion bundles -> pixart torch-trt
    if (strategy == "torchtrt_diffusion") {
        return "diffusion_pixart_torchtrt";
    }
    return strategy;
}

#endif // TRTF_HAS_TRT

} // namespace

std::unique_ptr<IPipeline> PipelineFactory::from_bundle(
    const std::string& bundle_path, const std::string& hf_python,
    const std::string& runtime_cache_path, bool cuda_graphs)
{
#if TRTF_HAS_TRT
    BundleFile bundle = ReadBundleFile(bundle_path);
    if (bundle.sections.empty())
        throw std::runtime_error("Failed to read bundle: " + bundle_path);

    // Extract config JSON from bundle
    std::string config_text;
    for (const auto& section : bundle.sections) {
        if (section.name == "config.json" && !section.data.empty()) {
            config_text.assign(section.data.begin(), section.data.end());
            break;
        }
    }

    // Parse runtime_strategy and normalize legacy strings
    std::string strategy = extract_json_string(config_text, "runtime_strategy", "decoder_kv_cache");
    if (strategy.empty())
        strategy = "decoder_kv_cache";
    strategy = normalize_legacy_strategy(strategy, config_text);

    // Load backend DSO based on bundle metadata
    std::string backend_name = extract_json_string(config_text, "engine_backend", "trt");
    IBackend* backend = BackendLoader::load(backend_name);

    // Look up plugin in registry
    auto* plugin = PipelineRegistry::instance().lookup(strategy);
    if (!plugin) {
        auto registered = PipelineRegistry::instance().registered_strategies();
        std::string available;
        for (const auto& s : registered) {
            if (!available.empty())
                available += ", ";
            available += s;
        }
        throw std::runtime_error("No plugin registered for runtime_strategy: " + strategy +
                                 " (available: " + available + ")");
    }

    // Parse base config and dispatch to plugin
    BaseConfig base_cfg = parse_base_config(config_text, bundle.info.max_cache_length);
    base_cfg.runtime_strategy = strategy; // use normalized strategy

    PipelineContext ctx{bundle, base_cfg, config_text, hf_python, bundle_path,
                        backend, runtime_cache_path, cuda_graphs};
    auto pipeline = plugin->create(ctx);

    std::cerr << "[trtf] Pipeline loaded (strategy=" << strategy << ", backend=trt_new_runtime)"
              << std::endl;
    return pipeline;
#else
    (void)bundle_path;
    (void)hf_python;
    (void)runtime_cache_path;
    (void)cuda_graphs;
    throw std::runtime_error("Bundle loading requires TRT support (compile with TRT)");
#endif
}

std::unique_ptr<IPipeline> load(const std::string& bundle_path, const std::string& hf_python,
                                const std::string& runtime_cache_path, bool cuda_graphs)
{
    return PipelineFactory::from_bundle(bundle_path, hf_python, runtime_cache_path, cuda_graphs);
}

} // namespace trtf
