#pragma once

#include "cabi/registry/backend_registry_dispatch.h"
#include "trtf/pipeline.h"

namespace trtf {
namespace cabi {

#if TRTF_HAS_TRT

trtf::IPipeline* create_segmentation_pipeline_via_registry(void* opaque_context);
trtf::IPipeline* create_mamba_pipeline_via_registry(void* opaque_context);
trtf::IPipeline* create_rwkv_pipeline_via_registry(void* opaque_context);
trtf::IPipeline* create_decoder_pipeline_via_registry(void* opaque_context);
trtf::IPipeline* create_whisper_pipeline_via_registry(void* opaque_context);
trtf::IPipeline* create_vl_pipeline_via_registry(void* opaque_context);
trtf::IPipeline* create_encoder_pipeline_via_registry(void* opaque_context);
trtf::IPipeline* create_embedding_pipeline_via_registry(void* opaque_context);
trtf::IPipeline* create_reranking_pipeline_via_registry(void* opaque_context);
trtf::IPipeline* create_text_to_audio_pipeline_via_registry(void* opaque_context);
trtf::IPipeline* create_hybrid_pipeline_via_registry(void* opaque_context);
trtf::IPipeline* create_detection_pipeline_via_registry(void* opaque_context);
trtf::IPipeline* create_sam_pipeline_via_registry(void* opaque_context);
trtf::IPipeline* create_neural_operator_pipeline_via_registry(void* opaque_context);
trtf::IPipeline* create_omni_pipeline_via_registry(void* opaque_context);
trtf::IPipeline* create_speech_pipeline_via_registry(void* opaque_context);

#endif // TRTF_HAS_TRT

} // namespace cabi
} // namespace trtf
