// Registration for the "text_trace" namespace schema.
//
// Mirrors trtf_build/trtf_build/runtime_config/schemas/text_trace.py.

#include "trtf/config/schemas/text_trace.h"

#include <any>
#include <cstdint>
#include <limits>
#include <set>
#include <string>

namespace trtf::config::schemas {

namespace {
bool is_nonneg_int32(const std::any& v) {
    if (v.type() != typeid(std::int32_t)) return false;
    return std::any_cast<std::int32_t>(v) >= 0;
}
bool is_positive_int32(const std::any& v) {
    if (v.type() != typeid(std::int32_t)) return false;
    return std::any_cast<std::int32_t>(v) >= 1;
}
} // namespace

Schema make_text_trace_schema()
{
    const std::set<Layer> session = {Layer::SessionRequest, Layer::PlatformProfile};
    return Schema{
        "text_trace",
        {
            ConfigField{"step_trace_path", "string", std::any{std::string{}},
                        session, nullptr},
            ConfigField{"step_trace_start_pos", "int32", std::any{std::int32_t{0}},
                        session, is_nonneg_int32},
            ConfigField{"step_trace_end_pos", "int32",
                        std::any{std::int32_t{2'000'000'000}},
                        session, is_nonneg_int32},
            ConfigField{"step_trace_topk", "int32", std::any{std::int32_t{8}},
                        session, is_positive_int32},
        },
    };
}

namespace {
struct TextTraceRegistrar {
    TextTraceRegistrar()
    {
        SchemaRegistry::instance().register_schema(make_text_trace_schema());
    }
};
TextTraceRegistrar g_text_trace_registrar{};
} // namespace
} // namespace trtf::config::schemas

namespace trtf::config::schemas {
volatile int kForceLink_text_trace = 0;
} // namespace trtf::config::schemas
