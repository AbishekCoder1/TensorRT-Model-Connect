// =============================================================================
// test_hf_python_tokenizer.cpp — Unit tests for HfPythonTokenizer helpers
// =============================================================================
//
// Purpose:
//   Validates the string utility functions used by HfPythonTokenizer:
//   trim_trailing_newlines, shell_quote, parse_int_list, join_ids_csv, and
//   sanitize_hf_output. These are pure string operations that do not require
//   a Python subprocess or any external dependencies.
//
// Dependencies:
//   - tokenizer/hf_python_tokenizer_helpers.h
//
// Environment:
//   CPU-only. No GPU, CUDA, TRT, Python, or filesystem access required.
// =============================================================================

#include "tokenizer/hf_python_tokenizer_helpers.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

static int failures = 0;

static void check(bool condition, const char* test_name)
{
    if (!condition)
    {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

// ---------------------------------------------------------------------------
// trim_trailing_newlines tests
// ---------------------------------------------------------------------------

static void test_trim_trailing_newlines_basic()
{
    const std::string result = trtf::hf_tok_detail::trim_trailing_newlines("hello\n\n");
    check(result == "hello", "trim_trailing_newlines basic");
}

static void test_trim_trailing_newlines_cr()
{
    const std::string result = trtf::hf_tok_detail::trim_trailing_newlines("hello\r\n");
    check(result == "hello", "trim_trailing_newlines \\r\\n");
}

static void test_trim_trailing_newlines_none()
{
    const std::string result = trtf::hf_tok_detail::trim_trailing_newlines("hello");
    check(result == "hello", "trim_trailing_newlines no newlines");
}

static void test_trim_trailing_newlines_empty()
{
    const std::string result = trtf::hf_tok_detail::trim_trailing_newlines("");
    check(result.empty(), "trim_trailing_newlines empty");
}

static void test_trim_trailing_newlines_only_newlines()
{
    const std::string result = trtf::hf_tok_detail::trim_trailing_newlines("\n\n\n");
    check(result.empty(), "trim_trailing_newlines only newlines");
}

static void test_trim_trailing_newlines_preserves_internal()
{
    const std::string result = trtf::hf_tok_detail::trim_trailing_newlines("hello\nworld\n");
    check(result == "hello\nworld", "trim_trailing_newlines preserves internal");
}

// ---------------------------------------------------------------------------
// shell_quote tests
// ---------------------------------------------------------------------------

static void test_shell_quote_simple()
{
    const std::string result = trtf::hf_tok_detail::shell_quote("hello");
    check(result == "'hello'", "shell_quote simple");
}

static void test_shell_quote_with_single_quote()
{
    const std::string result = trtf::hf_tok_detail::shell_quote("it's");
    // Single quotes in shell: close quote, add escaped quote, reopen
    check(result == "'it'\"'\"'s'", "shell_quote single quote");
}

static void test_shell_quote_empty()
{
    const std::string result = trtf::hf_tok_detail::shell_quote("");
    check(result == "''", "shell_quote empty");
}

static void test_shell_quote_spaces()
{
    const std::string result = trtf::hf_tok_detail::shell_quote("hello world");
    check(result == "'hello world'", "shell_quote spaces");
}

static void test_shell_quote_special_chars()
{
    const std::string result = trtf::hf_tok_detail::shell_quote("a$b&c;d");
    check(result == "'a$b&c;d'", "shell_quote special chars");
}

// ---------------------------------------------------------------------------
// parse_int_list tests
// ---------------------------------------------------------------------------

static void test_parse_int_list_basic()
{
    const auto result = trtf::hf_tok_detail::parse_int_list("1 2 3 4");
    check(result.size() == 4, "parse_int_list basic size");
    check(result == (std::vector<int32_t>{1, 2, 3, 4}), "parse_int_list basic values");
}

static void test_parse_int_list_empty()
{
    const auto result = trtf::hf_tok_detail::parse_int_list("");
    check(result.empty(), "parse_int_list empty");
}

static void test_parse_int_list_single()
{
    const auto result = trtf::hf_tok_detail::parse_int_list("42");
    check(result.size() == 1, "parse_int_list single size");
    check(result[0] == 42, "parse_int_list single value");
}

static void test_parse_int_list_with_extra_spaces()
{
    const auto result = trtf::hf_tok_detail::parse_int_list("  10   20   30  ");
    check(result.size() == 3, "parse_int_list extra spaces size");
    check(result == (std::vector<int32_t>{10, 20, 30}), "parse_int_list extra spaces values");
}

static void test_parse_int_list_negative()
{
    const auto result = trtf::hf_tok_detail::parse_int_list("-1 0 1");
    check(result.size() == 3, "parse_int_list negative size");
    check(result[0] == -1, "parse_int_list negative first");
}

static void test_parse_int_list_skips_non_numeric()
{
    const auto result = trtf::hf_tok_detail::parse_int_list("1 abc 3");
    // "abc" cannot be parsed as int, so it is skipped
    check(result.size() == 2, "parse_int_list skips non-numeric size");
    check(result[0] == 1, "parse_int_list skips non-numeric [0]");
    check(result[1] == 3, "parse_int_list skips non-numeric [1]");
}

// ---------------------------------------------------------------------------
// join_ids_csv tests
// ---------------------------------------------------------------------------

static void test_join_ids_csv_basic()
{
    const std::string result = trtf::hf_tok_detail::join_ids_csv({1, 2, 3});
    check(result == "1,2,3", "join_ids_csv basic");
}

static void test_join_ids_csv_single()
{
    const std::string result = trtf::hf_tok_detail::join_ids_csv({42});
    check(result == "42", "join_ids_csv single");
}

static void test_join_ids_csv_empty()
{
    const std::string result = trtf::hf_tok_detail::join_ids_csv({});
    check(result.empty(), "join_ids_csv empty");
}

static void test_join_ids_csv_negative()
{
    const std::string result = trtf::hf_tok_detail::join_ids_csv({-1, 0, 100});
    check(result == "-1,0,100", "join_ids_csv negative");
}

// ---------------------------------------------------------------------------
// sanitize_hf_output tests
// ---------------------------------------------------------------------------

static void test_sanitize_hf_output_clean()
{
    const std::string result = trtf::hf_tok_detail::sanitize_hf_output("42 43 44\n");
    check(result == "42 43 44", "sanitize_hf_output clean");
}

static void test_sanitize_hf_output_strips_warnings()
{
    const std::string input =
        "None of PyTorch, TensorFlow >= 2.0, or Flax have been found.\n"
        "42 43 44\n";
    const std::string result = trtf::hf_tok_detail::sanitize_hf_output(input);
    check(result == "42 43 44", "sanitize_hf_output strips warnings");
}

static void test_sanitize_hf_output_strips_cpp_ext_warning()
{
    const std::string input =
        "Skipping import of cpp extensions due to incompatible torch version\n"
        "hello world\n";
    const std::string result = trtf::hf_tok_detail::sanitize_hf_output(input);
    check(result == "hello world", "sanitize_hf_output strips cpp ext warning");
}

static void test_sanitize_hf_output_empty_lines()
{
    const std::string input = "\n\n42\n\n";
    const std::string result = trtf::hf_tok_detail::sanitize_hf_output(input);
    check(result == "42", "sanitize_hf_output empty lines");
}

static void test_sanitize_hf_output_preserves_multiline()
{
    const std::string input = "line1\nline2\n";
    const std::string result = trtf::hf_tok_detail::sanitize_hf_output(input);
    check(result == "line1\nline2", "sanitize_hf_output preserves multiline");
}

int main()
{
    test_trim_trailing_newlines_basic();
    test_trim_trailing_newlines_cr();
    test_trim_trailing_newlines_none();
    test_trim_trailing_newlines_empty();
    test_trim_trailing_newlines_only_newlines();
    test_trim_trailing_newlines_preserves_internal();

    test_shell_quote_simple();
    test_shell_quote_with_single_quote();
    test_shell_quote_empty();
    test_shell_quote_spaces();
    test_shell_quote_special_chars();

    test_parse_int_list_basic();
    test_parse_int_list_empty();
    test_parse_int_list_single();
    test_parse_int_list_with_extra_spaces();
    test_parse_int_list_negative();
    test_parse_int_list_skips_non_numeric();

    test_join_ids_csv_basic();
    test_join_ids_csv_single();
    test_join_ids_csv_empty();
    test_join_ids_csv_negative();

    test_sanitize_hf_output_clean();
    test_sanitize_hf_output_strips_warnings();
    test_sanitize_hf_output_strips_cpp_ext_warning();
    test_sanitize_hf_output_empty_lines();
    test_sanitize_hf_output_preserves_multiline();

    if (failures > 0)
    {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All hf_python_tokenizer helper tests passed.\n";
    return 0;
}
