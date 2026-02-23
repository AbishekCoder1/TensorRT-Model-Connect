#pragma once

// Internal helpers for HfPythonTokenizer, extracted for unit testing.
// These functions are used by hf_python_tokenizer.cpp and tested by
// tests/cpp/test_hf_python_tokenizer.cpp.

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace trtf {
namespace hf_tok_detail {

std::string trim_trailing_newlines(std::string text);
std::string shell_quote(std::string_view value);
std::vector<int32_t> parse_int_list(const std::string& text);
std::string join_ids_csv(const std::vector<int32_t>& ids);
std::string sanitize_hf_output(const std::string& text);

} // namespace hf_tok_detail
} // namespace trtf
