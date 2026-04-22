// Registration for the "platform" namespace schema.
// Mirrors trtf_build/trtf_build/runtime_config/schemas/platform.py.

#include "trtf/config/schemas/platform.h"

#include <any>
#include <set>
#include <string>

namespace trtf::config::schemas {

namespace {
bool is_valid_severity(const std::any& v) {
    if (v.type() != typeid(std::string))
        return false;
    const auto& s = std::any_cast<const std::string&>(v);
    return s == "INTERNAL_ERROR" || s == "ERROR" || s == "WARNING" || s == "INFO" || s == "VERBOSE";
}
} // namespace

Schema make_platform_schema() {
    const std::set<Layer> session = {Layer::SessionRequest, Layer::PlatformProfile};
    return Schema{
        "platform",
        {
            ConfigField{"source_dir", "string", std::any{std::string{}}, session, nullptr},
            ConfigField{"trt_log_stderr", "bool", std::any{false}, session, nullptr},
            ConfigField{"trt_log_min_severity", "string", std::any{std::string{"INFO"}}, session,
                        is_valid_severity},
        },
    };
}

namespace {
struct PlatformRegistrar {
    PlatformRegistrar() { SchemaRegistry::instance().register_schema(make_platform_schema()); }
};
PlatformRegistrar g_platform_registrar{};
} // namespace
} // namespace trtf::config::schemas

namespace trtf::config::schemas {
volatile int kForceLink_platform = 0;
} // namespace trtf::config::schemas
