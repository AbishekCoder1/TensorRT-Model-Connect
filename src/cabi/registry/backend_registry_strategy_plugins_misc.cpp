#include "cabi/registry/backend_registry_strategy_plugins.h"

#include "cabi/registry/backend_registry.h"
#include "cabi/registry/backend_registry_strategy_wrappers.h"

namespace trtf {
namespace cabi {

#if TRTF_HAS_TRT

void register_misc_strategy_backend_factories()
{
    (void) register_backend_factory("neural_operator", &create_neural_operator_pipeline_via_registry);
    (void) register_backend_factory("omni_multimodal", &create_omni_pipeline_via_registry);
}

#endif // TRTF_HAS_TRT

} // namespace cabi
} // namespace trtf
