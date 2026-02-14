#pragma once

namespace trtf {
namespace llama {

// Register LLaMA model family: HF family matcher + TRT graph builder.
// Called once during static initialization or from RegisterBuiltinHfModelFamilies().
void RegisterLlamaFamily();

} // namespace llama
} // namespace trtf
