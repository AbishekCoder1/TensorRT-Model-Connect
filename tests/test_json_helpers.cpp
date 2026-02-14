// Unit tests for src/utils/json_helpers.cpp (JSON extraction functions).
// CPU-only, no TRT/CUDA deps.

#include "utils/json_helpers.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

bool test_extract_json_string_present()
{
    const std::string json = R"({"model_type": "qwen3", "other": "value"})";
    const std::string result = trtf::extract_json_string(json, "model_type", "");
    if (result != "qwen3")
    {
        std::cerr << "extract_json_string_present: got '" << result << "'" << std::endl;
        return false;
    }
    return true;
}

bool test_extract_json_string_absent()
{
    const std::string json = R"({"other": "value"})";
    const std::string result = trtf::extract_json_string(json, "model_type", "fallback");
    if (result != "fallback")
    {
        std::cerr << "extract_json_string_absent: got '" << result << "'" << std::endl;
        return false;
    }
    return true;
}

bool test_extract_json_string_nested_braces()
{
    const std::string json = R"({"config": {"inner": 1}, "model_type": "llama"})";
    const std::string result = trtf::extract_json_string(json, "model_type", "");
    if (result != "llama")
    {
        std::cerr << "extract_json_string_nested: got '" << result << "'" << std::endl;
        return false;
    }
    return true;
}

bool test_extract_json_int_positive()
{
    const std::string json = R"({"hidden_size": 768})";
    const int32_t result = trtf::extract_json_int(json, "hidden_size", -1);
    if (result != 768)
    {
        std::cerr << "extract_json_int_positive: got " << result << std::endl;
        return false;
    }
    return true;
}

bool test_extract_json_int_negative()
{
    const std::string json = R"({"offset": -42})";
    const int32_t result = trtf::extract_json_int(json, "offset", 0);
    if (result != -42)
    {
        std::cerr << "extract_json_int_negative: got " << result << std::endl;
        return false;
    }
    return true;
}

bool test_extract_json_int_missing()
{
    const std::string json = R"({"other": 5})";
    const int32_t result = trtf::extract_json_int(json, "hidden_size", -99);
    if (result != -99)
    {
        std::cerr << "extract_json_int_missing: got " << result << std::endl;
        return false;
    }
    return true;
}

bool test_extract_json_int_float_value()
{
    // Float values should return fallback (parser stops at '.')
    const std::string json = R"({"hidden_size": 3.14})";
    const int32_t result = trtf::extract_json_int(json, "hidden_size", -1);
    // Parser reads "3" then stops at '.' — returns 3
    if (result != 3)
    {
        std::cerr << "extract_json_int_float: got " << result << std::endl;
        return false;
    }
    return true;
}

bool test_extract_json_int_or_first_array_scalar()
{
    const std::string json = R"({"bos_token_id": 123})";
    const int32_t result = trtf::extract_json_int_or_first_array(json, "bos_token_id", -1);
    if (result != 123)
    {
        std::cerr << "int_or_first_array_scalar: got " << result << std::endl;
        return false;
    }
    return true;
}

bool test_extract_json_int_or_first_array_array()
{
    const std::string json = R"({"bos_token_id": [456, 789]})";
    const int32_t result = trtf::extract_json_int_or_first_array(json, "bos_token_id", -1);
    if (result != 456)
    {
        std::cerr << "int_or_first_array_array: got " << result << std::endl;
        return false;
    }
    return true;
}

bool test_extract_json_int_or_first_array_empty_array()
{
    const std::string json = R"({"bos_token_id": []})";
    const int32_t result = trtf::extract_json_int_or_first_array(json, "bos_token_id", -1);
    if (result != -1)
    {
        std::cerr << "int_or_first_array_empty: got " << result << std::endl;
        return false;
    }
    return true;
}

bool test_extract_json_int_or_first_array_missing()
{
    const std::string json = R"({"other": 5})";
    const int32_t result = trtf::extract_json_int_or_first_array(json, "bos_token_id", -1);
    if (result != -1)
    {
        std::cerr << "int_or_first_array_missing: got " << result << std::endl;
        return false;
    }
    return true;
}

bool test_extract_json_float_basic()
{
    const std::string json = R"({"rope_theta": 3.14})";
    const float result = trtf::extract_json_float(json, "rope_theta", 0.0F);
    if (std::abs(result - 3.14F) > 0.01F)
    {
        std::cerr << "extract_json_float_basic: got " << result << std::endl;
        return false;
    }
    return true;
}

bool test_extract_json_float_scientific()
{
    const std::string json = R"({"eps": 1e-5})";
    const float result = trtf::extract_json_float(json, "eps", 0.0F);
    if (std::abs(result - 1e-5F) > 1e-8F)
    {
        std::cerr << "extract_json_float_scientific: got " << result << std::endl;
        return false;
    }
    return true;
}

bool test_extract_json_float_missing()
{
    const std::string json = R"({"other": 5})";
    const float result = trtf::extract_json_float(json, "eps", -1.0F);
    if (std::abs(result - (-1.0F)) > 1e-6F)
    {
        std::cerr << "extract_json_float_missing: got " << result << std::endl;
        return false;
    }
    return true;
}

bool test_extract_json_string_array_basic()
{
    const std::string json = R"({"architectures": ["QwenForCausalLM", "Qwen2ForCausalLM"]})";
    const auto result = trtf::extract_json_string_array(json, "architectures");
    if (result.size() != 2 || result[0] != "QwenForCausalLM" || result[1] != "Qwen2ForCausalLM")
    {
        std::cerr << "string_array_basic: size=" << result.size() << std::endl;
        return false;
    }
    return true;
}

bool test_extract_json_string_array_empty()
{
    const std::string json = R"({"architectures": []})";
    const auto result = trtf::extract_json_string_array(json, "architectures");
    if (!result.empty())
    {
        std::cerr << "string_array_empty: size=" << result.size() << std::endl;
        return false;
    }
    return true;
}

bool test_extract_json_string_array_missing()
{
    const std::string json = R"({"other": 5})";
    const auto result = trtf::extract_json_string_array(json, "architectures");
    if (!result.empty())
    {
        std::cerr << "string_array_missing: size=" << result.size() << std::endl;
        return false;
    }
    return true;
}

} // namespace

int main()
{
    bool all_passed = true;
    std::cout << "test_json_helpers:" << std::endl;

    const auto run = [&](const char* name, bool (*fn)()) {
        const bool ok = fn();
        std::cout << "  " << name << ": " << (ok ? "PASS" : "FAIL") << std::endl;
        all_passed &= ok;
    };

    run("extract_json_string_present", test_extract_json_string_present);
    run("extract_json_string_absent", test_extract_json_string_absent);
    run("extract_json_string_nested", test_extract_json_string_nested_braces);
    run("extract_json_int_positive", test_extract_json_int_positive);
    run("extract_json_int_negative", test_extract_json_int_negative);
    run("extract_json_int_missing", test_extract_json_int_missing);
    run("extract_json_int_float_value", test_extract_json_int_float_value);
    run("int_or_first_array_scalar", test_extract_json_int_or_first_array_scalar);
    run("int_or_first_array_array", test_extract_json_int_or_first_array_array);
    run("int_or_first_array_empty", test_extract_json_int_or_first_array_empty_array);
    run("int_or_first_array_missing", test_extract_json_int_or_first_array_missing);
    run("extract_json_float_basic", test_extract_json_float_basic);
    run("extract_json_float_scientific", test_extract_json_float_scientific);
    run("extract_json_float_missing", test_extract_json_float_missing);
    run("string_array_basic", test_extract_json_string_array_basic);
    run("string_array_empty", test_extract_json_string_array_empty);
    run("string_array_missing", test_extract_json_string_array_missing);

    if (all_passed)
    {
        std::cout << "test_json_helpers passed" << std::endl;
        return 0;
    }
    std::cerr << "test_json_helpers FAILED" << std::endl;
    return 1;
}
