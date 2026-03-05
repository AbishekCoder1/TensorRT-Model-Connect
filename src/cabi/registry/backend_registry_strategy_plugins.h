#pragma once

namespace trtf {
namespace cabi {

#if TRTF_HAS_TRT

void register_text_strategy_backend_factories();
void register_vision_strategy_backend_factories();
void register_encoder_strategy_backend_factories();
void register_audio_strategy_backend_factories();
void register_misc_strategy_backend_factories();

#endif // TRTF_HAS_TRT

} // namespace cabi
} // namespace trtf
