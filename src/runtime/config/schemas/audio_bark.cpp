// Registration for the "audio_bark" namespace schema.
// Mirrors trtf_build/trtf_build/runtime_config/schemas/audio_bark.py.

#include "trtf/config/schemas/audio_bark.h"

#include <any>
#include <cstdint>
#include <set>
#include <string>

namespace trtf::config::schemas {

Schema make_audio_bark_schema() {
    const std::set<Layer> session = {Layer::SessionRequest, Layer::PlatformProfile};
    return Schema{
        "audio_bark",
        {
            ConfigField{"dump_path", "string", std::any{std::string{}}, session, nullptr},
            ConfigField{"greedy", "bool", std::any{false}, session, nullptr},
            ConfigField{"seed", "int64", std::any{std::int64_t{-1}}, session, nullptr},
        },
    };
}

namespace {
struct AudioBarkRegistrar {
    AudioBarkRegistrar() { SchemaRegistry::instance().register_schema(make_audio_bark_schema()); }
};
AudioBarkRegistrar g_audio_bark_registrar{};
} // namespace
} // namespace trtf::config::schemas

namespace trtf::config::schemas {
volatile int kForceLink_audio_bark = 0;
} // namespace trtf::config::schemas
