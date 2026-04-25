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

REGISTER_CONFIG_SCHEMA_FACTORY_WITH_FORCE_LINK(kForceLink_runtime, make_runtime_schema);
} // namespace trtf::config::schemas
