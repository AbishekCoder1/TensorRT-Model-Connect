// Registration for the "runtime" namespace schema.
// Mirrors trtf_build/trtf_build/runtime_config/schemas/runtime.py.

#include "trtf/config/schemas/runtime.h"

#include <any>
#include <set>

namespace trtf::config::schemas {

Schema make_runtime_schema() {
    const std::set<Layer> session = {Layer::SessionRequest, Layer::PlatformProfile};
    return Schema{
        "runtime",
        {
            ConfigField{"disable_cuda_graph", "bool", std::any{false}, session, nullptr},
            ConfigField{"prefer_gpu_greedy", "bool", std::any{false}, session, nullptr},
        },
    };
}

namespace {
struct RuntimeRegistrar {
    RuntimeRegistrar() { SchemaRegistry::instance().register_schema(make_runtime_schema()); }
};
RuntimeRegistrar g_runtime_registrar{};
} // namespace
} // namespace trtf::config::schemas

namespace trtf::config::schemas {
volatile int kForceLink_runtime = 0;
} // namespace trtf::config::schemas
