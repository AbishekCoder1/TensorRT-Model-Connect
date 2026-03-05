#include "cabi/registry/backend_registry_strategy_plugins.h"

#include "cabi/registry/backend_registry.h"
#include "cabi/registry/backend_registry_strategy_wrappers.h"

namespace trtf {
namespace cabi {

#if TRTF_HAS_TRT

void register_vision_strategy_backend_factories()
{
    (void) register_backend_factory("segmentation", &create_segmentation_pipeline_via_registry);
    (void) register_backend_factory("vision_language", &create_vl_pipeline_via_registry);
    (void) register_backend_factory("object_detection", &create_detection_pipeline_via_registry);
    (void) register_backend_factory("prompted_segmentation", &create_sam_pipeline_via_registry);
}

#endif // TRTF_HAS_TRT

} // namespace cabi
} // namespace trtf
