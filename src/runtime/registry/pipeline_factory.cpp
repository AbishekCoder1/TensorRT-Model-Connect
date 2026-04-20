#include "trtf/runtime/pipeline_factory.h"

#include "bundle/bundle_format.h"
#include "runtime/backend/backend_loader.h"
#include "runtime/core/trt_common.h"
#include "trtf/config/cli_support.h"
#include "trtf/config/config_bundle.h"
#include "trtf/config/schema_registry.h"
#include "trtf/runtime/pipeline_plugin.h"
#include "trtf/runtime/pipeline_registry.h"
#include "trtf/runtime/trt_backend.h"
#include "utils/data_dir.h"
#include "utils/json_helpers.h"

#include <exception>
#include <iostream>
#include <optional>
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

IPipelinePlugin* lookup_plugin_or_throw(const std::string& strategy)
{
    auto* plugin = PipelineRegistry::instance().lookup(strategy);
    if (plugin != nullptr) return plugin;
    std::string available;
    for (const auto& s : PipelineRegistry::instance().registered_strategies())
    {
        if (!available.empty()) available += ", ";
        available += s;
    }
    throw std::runtime_error(
        "No plugin registered for runtime_strategy: " + strategy +
        " (available: " + available + ")");
}

// Apply platform.* values to their process-wide sinks. Replaces the old
// TRTF_DATA_DIR and TRTF_TRT_LOG_{STDERR,MIN_SEVERITY} env-var reads.
// Called from try_resolve_runtime_config once a bundle has resolved.
void apply_platform_config(const config::ConfigBundle& bundle)
{
    try
    {
        const std::string source = bundle.get<std::string>("platform", "source_dir");
        if (!source.empty()) set_source_dir_override(source);
        const bool verbose_stderr = bundle.get<bool>("platform", "trt_log_stderr");
        const std::string severity = bundle.get<std::string>("platform", "trt_log_min_severity");
        configure_trt_logger(verbose_stderr, severity);
    }
    catch (const std::exception&)
    {
        // Schema absent or type mismatch — leave sinks at defaults.
    }
}

std::optional<config::ConfigBundle> try_resolve_runtime_config(
    const std::string& config_text,
    const std::string& bundle_path,
    const std::string& config_path,
    const std::vector<std::string>& set_tokens)
{
    try
    {
        auto resolution = config::resolve_pipeline_config(
            config_text, config_path, set_tokens);
        config::write_effective_config_next_to(resolution.bundle, bundle_path);
        apply_platform_config(resolution.bundle);
        return std::move(resolution.bundle);
    }
    catch (const std::exception& e)
    {
        std::cerr << "[trtf.config] Failed to resolve runtime config: "
                  << e.what() << "\n          Proceeding with schema defaults.\n";
        return std::nullopt;
    }
}

#endif // TRTF_HAS_TRT

} // namespace

std::unique_ptr<IPipeline> PipelineFactory::from_bundle(const std::string& bundle_path,
                                                        const std::string& hf_python,
                                                        const std::string& runtime_cache_path,
                                                        bool cuda_graphs) {
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

    auto* plugin = lookup_plugin_or_throw(strategy);

    // Parse base config and dispatch to plugin
    BaseConfig base_cfg = parse_base_config(config_text, bundle.info.max_cache_length);
    base_cfg.runtime_strategy = strategy; // use normalized strategy

    // Resolve the layered runtime config (BUNDLE_DEFAULT + SESSION_REQUEST).
    // Best-effort: a malformed input prints to stderr and falls back to
    // schema defaults so plugin construction isn't blocked.
    std::optional<config::ConfigBundle> resolved =
        try_resolve_runtime_config(config_text, bundle_path, /*config_path=*/"",
                                   /*set_tokens=*/{});

    PipelineContext ctx{bundle,
                        base_cfg,
                        config_text,
                        hf_python,
                        bundle_path,
                        backend,
                        runtime_cache_path,
                        cuda_graphs,
                        /*kv_cache_size_bytes=*/0,
                        resolved ? &*resolved : nullptr};
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
                                const std::string& runtime_cache_path, bool cuda_graphs) {
    return PipelineFactory::from_bundle(bundle_path, hf_python, runtime_cache_path, cuda_graphs);
}

} // namespace trtf
