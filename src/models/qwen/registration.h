#pragma once

namespace trtf {
namespace qwen {

// Register Qwen model family: HF family matcher + TRT graph builder.
// Called once during static initialization or from RegisterBuiltinHfModelFamilies().
void RegisterQwenFamily();

} // namespace qwen
} // namespace trtf
