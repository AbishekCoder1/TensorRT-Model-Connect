#include "cabi/registry/backend_registry_strategy_plugins.h"

#include "cabi/registry/backend_registry.h"
#include "cabi/registry/backend_registry_strategy_wrappers.h"

namespace trtf {
namespace cabi {

#if TRTF_HAS_TRT

void register_encoder_strategy_backend_factories()
{
    (void) register_backend_factory("encoder_only", &create_encoder_pipeline_via_registry);
    (void) register_backend_factory("embedding", &create_embedding_pipeline_via_registry);
    (void) register_backend_factory("reranking", &create_reranking_pipeline_via_registry);
}

#endif // TRTF_HAS_TRT

} // namespace cabi
} // namespace trtf
