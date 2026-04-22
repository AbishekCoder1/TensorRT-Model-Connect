// Unit tests for runtime plugin helper parsing.
// Focus: tokenizer_add_special_tokens detection from bundle config.

#include "runtime/plugins/shared/plugin_helpers.h"

#include <iostream>
#include <string>

static int failures = 0;

static void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

static trtf::BundleFile make_bundle_with_config(const std::string& config) {
    trtf::BundleFile bundle;
    trtf::BundleSection sec;
    sec.name = "config.json";
    sec.data.assign(config.begin(), config.end());
    bundle.sections.push_back(std::move(sec));
    return bundle;
}

static void test_missing_field_defaults_true() {
    auto bundle = make_bundle_with_config(R"({"runtime_strategy":"decoder_kv_cache"})");
    check(trtf::detect_add_special_tokens(bundle) == true,
          "detect_add_special_tokens: missing field defaults true");
}

static void test_integer_false_parsed() {
    auto bundle = make_bundle_with_config(R"({"tokenizer_add_special_tokens":0})");
    check(trtf::detect_add_special_tokens(bundle) == false,
          "detect_add_special_tokens: integer 0 parsed as false");
}

static void test_integer_true_parsed() {
    auto bundle = make_bundle_with_config(R"({"tokenizer_add_special_tokens":1})");
    check(trtf::detect_add_special_tokens(bundle) == true,
          "detect_add_special_tokens: integer 1 parsed as true");
}

static void test_bool_false_parsed() {
    auto bundle = make_bundle_with_config(R"({"tokenizer_add_special_tokens":false})");
    check(trtf::detect_add_special_tokens(bundle) == false,
          "detect_add_special_tokens: bool false parsed as false");
}

static void test_bool_true_parsed() {
    auto bundle = make_bundle_with_config(R"({"tokenizer_add_special_tokens":true})");
    check(trtf::detect_add_special_tokens(bundle) == true,
          "detect_add_special_tokens: bool true parsed as true");
}

int main() {
    test_missing_field_defaults_true();
    test_integer_false_parsed();
    test_integer_true_parsed();
    test_bool_false_parsed();
    test_bool_true_parsed();

    if (failures > 0) {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All plugin helper tests passed.\n";
    return 0;
}
