#include "cabi/registry/backend_registry_strategy_plugins.h"

#include "cabi/registry/backend_registry.h"
#include "cabi/registry/backend_registry_strategy_wrappers.h"

namespace trtf {
namespace cabi {

#if TRTF_HAS_TRT

void register_audio_strategy_backend_factories()
{
    (void) register_backend_factory("speech_to_text", &create_whisper_pipeline_via_registry);
    (void) register_backend_factory("text_to_audio", &create_text_to_audio_pipeline_via_registry);
    (void) register_backend_factory("speech_to_speech", &create_speech_pipeline_via_registry);
}

#endif // TRTF_HAS_TRT

} // namespace cabi
} // namespace trtf
