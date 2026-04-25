// Registration for the "audio_magpie" namespace schema.
// Mirrors trtf_build/trtf_build/runtime_config/schemas/audio_magpie.py.

#include "trtf/config/schemas/audio_magpie.h"

#include <any>
#include <cstdint>
#include <set>

namespace trtf::config::schemas {

namespace {
bool is_nonneg_float(const std::any& v) {
    if (v.type() == typeid(float))
        return std::any_cast<float>(v) >= 0.0F;
    if (v.type() == typeid(double))
        return std::any_cast<double>(v) >= 0.0;
    return false;
}
bool is_nonneg_int32(const std::any& v) {
    if (v.type() != typeid(std::int32_t))
        return false;
    return std::any_cast<std::int32_t>(v) >= 0;
}
} // namespace

Schema make_audio_magpie_schema() {
    const std::set<Layer> session = {Layer::SessionRequest, Layer::PlatformProfile};
    const std::set<Layer> build_bundle = {Layer::BuildTime, Layer::BundleDefault};
    return Schema{
        "audio_magpie",
        {
            ConfigField{"greedy", "bool", std::any{false}, session, nullptr},
            ConfigField{"cfg_scale", "float", std::any{0.0F}, session, is_nonneg_float},
            ConfigField{"temperature", "float", std::any{0.0F}, session, is_nonneg_float},
            ConfigField{"finished_limit", "int32", std::any{std::int32_t{-1}}, session, nullptr},
            ConfigField{"seed", "int64", std::any{std::int64_t{-1}}, session, nullptr},
            ConfigField{"max_source_positions", "int32", std::any{std::int32_t{0}}, build_bundle,
                        is_nonneg_int32},
        },
    };
}

REGISTER_CONFIG_SCHEMA_FACTORY_WITH_FORCE_LINK(kForceLink_audio_magpie, make_audio_magpie_schema);
} // namespace trtf::config::schemas
