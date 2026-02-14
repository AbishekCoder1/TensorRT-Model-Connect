// Test: ToyTokenizer encode/decode round-trip and unknown token handling.
// Verifies: Token encoding from built-in vocabulary, decode back to text, and
// unknown token fallback using the built-in tiny-cake-v1 vocabulary.

#include "trtf/tokenizer.h"

#include <iostream>
#include <vector>

int main()
{
    auto tokenizer = trtf::CreateToyTokenizer();

    const std::string prompt = "the secret to baking a really good cake is.";
    const auto ids = tokenizer->encode(prompt);

    if (ids.empty())
    {
        std::cerr << "tokenizer produced no tokens" << std::endl;
        return 1;
    }

    const auto decoded = tokenizer->decode(ids);
    if (decoded != "the secret to baking a really good cake is.")
    {
        std::cerr << "decode mismatch: " << decoded << std::endl;
        return 1;
    }

    const auto unknown = tokenizer->encode("nonexistenttoken");
    if (unknown.size() != 1)
    {
        std::cerr << "unexpected unknown token size" << std::endl;
        return 1;
    }

    std::cout << "test_tokenizer passed" << std::endl;
    return 0;
}
