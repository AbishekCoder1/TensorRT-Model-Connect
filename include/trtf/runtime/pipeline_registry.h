#pragma once

// Pipeline registry: singleton that maps runtime_strategy strings to
// IPipelinePlugin instances. Plugins self-register at static-init time
// via the REGISTER_PIPELINE_PLUGIN macro.

#include "trtf/runtime/pipeline_plugin.h"

#include <initializer_list>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace trtf {

class PipelineRegistry {
  public:
    static PipelineRegistry& instance();

    // Register a plugin for one or more strategy strings.
    // Called at static-init time by REGISTER_PIPELINE_PLUGIN.
    void register_plugin(const std::string& strategy, IPipelinePlugin* plugin);

    // Look up the plugin for a given strategy string.
    // Returns nullptr if no plugin is registered for the strategy.
    IPipelinePlugin* lookup(const std::string& strategy) const;

    // Return all registered strategy strings (for diagnostics / testing).
    std::vector<std::string> registered_strategies() const;

  private:
    PipelineRegistry() = default;
    std::unordered_map<std::string, IPipelinePlugin*> registry_;
};

// Helper: registers a static plugin instance for one strategy at static-init time.
struct PluginRegistrar {
    PluginRegistrar(const std::string& strategy, IPipelinePlugin* plugin) {
        PipelineRegistry::instance().register_plugin(strategy, plugin);
    }
};

// Macro: declare a static plugin instance and register it for one strategy.
// Usage (at file scope in a plugin .cpp):
//   REGISTER_PIPELINE_PLUGIN("decoder_kv_cache", DecoderPlugin);
//
// For plugins handling multiple strategies, use the multi variant.
#define REGISTER_PIPELINE_PLUGIN(strategy, PluginClass)                                            \
    REGISTER_PIPELINE_PLUGIN_MULTI(PluginClass, strategy)

// Multi-strategy variant: register same instance for multiple strategies.
// Usage:
//   REGISTER_PIPELINE_PLUGIN_MULTI(DecoderPlugin, "decoder_kv_cache", "decoder_moe");
#define REGISTER_PIPELINE_PLUGIN_MULTI(PluginClass, ...)                                           \
    static PluginClass g_##PluginClass##_instance;                                                 \
    namespace {                                                                                    \
    struct PluginClass##_MultiReg {                                                                \
        PluginClass##_MultiReg() {                                                                 \
            for (const char* s : std::initializer_list<const char*>{__VA_ARGS__})                  \
                ::trtf::PipelineRegistry::instance().register_plugin(s,                            \
                                                                     &g_##PluginClass##_instance); \
        }                                                                                          \
    };                                                                                             \
    static PluginClass##_MultiReg g_##PluginClass##_multi_reg;                                     \
    }

// Define a force-link symbol and register the plugin strategies in one place.
// The symbol name must match cmake/trtf_pipeline_plugins.cmake.
#define REGISTER_PIPELINE_PLUGIN_WITH_FORCE_LINK(ForceLinkSymbol, PluginClass, ...)                \
    volatile int ForceLinkSymbol = 0;                                                              \
    REGISTER_PIPELINE_PLUGIN_MULTI(PluginClass, __VA_ARGS__)

} // namespace trtf
