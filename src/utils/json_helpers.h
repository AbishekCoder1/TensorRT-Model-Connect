#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace trtf {

std::string extract_json_string(const std::string& text, const std::string& key, const std::string& fallback);
std::vector<std::string> extract_json_string_array(const std::string& text, const std::string& key);
int32_t extract_json_int(const std::string& text, const std::string& key, int32_t fallback);
int32_t extract_json_int_or_first_array(const std::string& text, const std::string& key, int32_t fallback);
float extract_json_float(const std::string& text, const std::string& key, float fallback);
std::vector<float> extract_json_float_array(const std::string& text, const std::string& key, std::size_t max_count = 16);
std::vector<int32_t> extract_json_int_array(const std::string& text, const std::string& key, std::size_t max_count = 16);
int32_t parse_positive_env_int(const char* env_name, int32_t fallback);

} // namespace trtf
