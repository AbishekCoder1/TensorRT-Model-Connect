#include "cabi/registry/backend_registry_dispatch.h"

#include "cabi/registry/backend_registry_strategy_plugins.h"

#include <mutex>

namespace trtf {
namespace cabi {

#if TRTF_HAS_TRT

void register_builtin_backend_factories_once()
{
    static std::once_flag once;
    std::call_once(once, [] {
        register_text_strategy_backend_factories();
        register_vision_strategy_backend_factories();
        register_encoder_strategy_backend_factories();
        register_audio_strategy_backend_factories();
        register_misc_strategy_backend_factories();
    });
}

#endif // TRTF_HAS_TRT

} // namespace cabi
} // namespace trtf
