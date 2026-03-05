#include "cabi/registry/backend_registry_strategy_plugins.h"

#include "cabi/registry/backend_registry.h"
#include "cabi/registry/backend_registry_strategy_wrappers.h"

namespace trtf {
namespace cabi {

#if TRTF_HAS_TRT

void register_text_strategy_backend_factories()
{
    (void) register_backend_factory("decoder_kv_cache", &create_decoder_pipeline_via_registry);
    (void) register_backend_factory("decoder_moe", &create_decoder_pipeline_via_registry);
    (void) register_backend_factory("ssm_recurrent", &create_mamba_pipeline_via_registry);
    (void) register_backend_factory("rwkv_recurrent", &create_rwkv_pipeline_via_registry);
    (void) register_backend_factory("hybrid_mamba_attention", &create_hybrid_pipeline_via_registry);
}

#endif // TRTF_HAS_TRT

} // namespace cabi
} // namespace trtf
