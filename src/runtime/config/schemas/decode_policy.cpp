// Registration for the "decode_policy" namespace schema.
//
// Mirrors trtf_build/trtf_build/runtime_config/schemas/decode_policy.py.
// Only edit both sides together; the cross-language match test gates on it.

#include "trtf/config/schemas/decode_policy.h"

#include <any>
#include <set>
#include <string>

namespace trtf::config::schemas {

Schema make_decode_policy_schema()
{
    const std::set<Layer> build_and_bundle = {
        Layer::BuildTime, Layer::BundleDefault,
    };
    return Schema{
        "decode_policy",
        {
            // Build-time only. Chosen when the bundle is built and baked
            // into the engine graph — session/platform layers cannot
            // retroactively toggle it.
            ConfigField{
                "force_manual_attention", "bool",
                std::any{false}, build_and_bundle, nullptr,
            },
        },
    };
}

namespace {
struct DecodePolicyRegistrar {
    DecodePolicyRegistrar()
    {
        SchemaRegistry::instance().register_schema(make_decode_policy_schema());
    }
};
DecodePolicyRegistrar g_decode_policy_registrar{};
} // namespace
} // namespace trtf::config::schemas

// Force-link anchor.
namespace trtf::config::schemas {
volatile int kForceLink_decode_policy = 0;
} // namespace trtf::config::schemas
