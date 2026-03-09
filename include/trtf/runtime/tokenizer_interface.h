#pragma once

// ITokenizer: shared tokenizer interface.
// HF equivalent: AutoTokenizer / PreTrainedTokenizer.
//
// Existing VocabTokenizer and HfPythonTokenizer already implement these
// operations — this interface formalizes the contract so pipelines can
// use either without knowing the concrete type.

#include <cstdint>
#include <string>
#include <vector>

namespace trtf {

class ITokenizer {
public:
    virtual ~ITokenizer() = default;

    // Encode text → token IDs.
    virtual std::vector<int32_t> encode(const std::string& text) = 0;

    // Decode token IDs → text.
    virtual std::string decode(const std::vector<int32_t>& ids) = 0;
};

} // namespace trtf
