// =============================================================================
// test_hf_python_tokenizer.cpp — Unit tests for HfPythonTokenizer
// =============================================================================
//
// Coverage split:
//   1) Pure helper utilities in hf_tok_detail namespace.
//   2) Deterministic runtime behavior of CreateHfPythonTokenizer and
//      ITokenizer methods through a local fake executable (no Python/HF deps).
// =============================================================================

#include "tokenizer/hf_python_tokenizer_helpers.h"
#include "trtf/tokenizer.h"
#include "test_helpers.h"

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
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

static void check_contains(const std::string& text, const std::string& needle, const char* test_name)
{
    check(text.find(needle) != std::string::npos, test_name);
}

static void check_ids_equal(const std::vector<int32_t>& actual, std::initializer_list<int32_t> expected,
    const char* test_name)
{
    check(actual == std::vector<int32_t>(expected), test_name);
}

// ---------------------------------------------------------------------------
// Runtime harness (fake executable + temp asset layout)
// ---------------------------------------------------------------------------

static std::string fake_python_bridge_script()
{
    return R"SCRIPT(#!/bin/sh
if [ "$#" -gt 0 ]; then
  shift
fi

check_mode=0
op=""
text_file=""
ids=""
token=""
id=""
add_special=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --check)
      check_mode=1
      ;;
    --op)
      shift
      op="$1"
      ;;
    --text-file)
      shift
      text_file="$1"
      ;;
    --ids)
      shift
      ids="$1"
      ;;
    --token)
      shift
      token="$1"
      ;;
    --id)
      shift
      id="$1"
      ;;
    --add-special-tokens)
      add_special=1
      ;;
  esac
  shift
done

warn1="None of PyTorch, TensorFlow >= 2.0, or Flax have been found (test)"
warn2="Skipping import of cpp extensions due to incompatible torch version (test)"
mode="${HF_TEST_MODE:-ok}"

if [ "$check_mode" -eq 1 ]; then
  echo "$warn1"
  if [ "$mode" = "check_fail" ]; then
    echo "check failed output"
    exit 70
  fi
  echo "check passed"
  exit 0
fi

case "$op" in
  encode)
    if [ "$mode" = "encode_fail" ]; then
      echo "encode failed output"
      exit 71
    fi
    payload="$(cat "$text_file")"
    if [ "$payload" = "include_special" ]; then
      if [ "$add_special" -eq 1 ]; then
        echo "101 11 12 102"
      else
        echo "11 12"
      fi
    elif [ "$payload" = "warnings" ]; then
      echo "  $warn1"
      echo "   $warn2"
      echo "7 8 9"
    elif [ "$payload" = "mixed" ]; then
      echo "1 bad 2x 3"
    else
      echo "42"
    fi
    ;;
  decode)
    if [ "$mode" = "decode_fail" ]; then
      echo "decode failed output"
      exit 72
    fi
    if [ "$ids" = "1,2,3" ]; then
      echo "  $warn1"
      echo "decoded text"
    else
      echo "decoded:$ids"
    fi
    ;;
  id-for-token)
    if [ "$mode" = "id_fail" ]; then
      echo "id failed output"
      exit 73
    fi
    if [ "$token" = "blank" ]; then
      echo "  $warn1"
      echo ""
    elif [ "$token" = "bad" ]; then
      echo "not-a-number"
    elif [ "$token" = "a'b c" ]; then
      echo "321"
    else
      echo "123"
    fi
    ;;
  token-for-id)
    if [ "$mode" = "token_fail" ]; then
      echo "token failed output"
      exit 74
    fi
    if [ "$id" = "77" ]; then
      echo "   $warn2"
      echo "tok77"
    else
      echo "tok$id"
    fi
    ;;
  *)
    echo "unknown op: $op"
    exit 75
    ;;
esac
)SCRIPT";
}

class HfTokenizerHarness {
public:
    HfTokenizerHarness()
        : mTempDir()
        , mRoot(mTempDir.path())
        , mDataRoot(mRoot / "hf data 'root'")
        , mModelDir(mRoot / "model dir 'quoted'")
        , mPythonCmd(mRoot / "fake python 'shim'.sh")
    {
        std::filesystem::create_directories(mDataRoot / "scripts");
        std::filesystem::create_directories(mModelDir);

        mDataRootString = mDataRoot.string();
        mDataDirGuard = std::make_unique<trtf_test::EnvVarGuard>("TRTF_DATA_DIR", mDataRootString.c_str());

        trtf_test::write_file(mDataRoot / "scripts" / "hf_tokenizer.py", "#!/usr/bin/env python3\n");
        trtf_test::write_file(mPythonCmd, fake_python_bridge_script());
        std::filesystem::permissions(
            mPythonCmd, std::filesystem::perms::owner_exec, std::filesystem::perm_options::add);
    }

    std::unique_ptr<trtf::ITokenizer> create(bool add_special_tokens = false) const
    {
        return trtf::CreateHfPythonTokenizer(mModelDir.string(), mPythonCmd.string(), add_special_tokens);
    }

    const std::filesystem::path& model_dir() const { return mModelDir; }
    const std::filesystem::path& python_cmd() const { return mPythonCmd; }

private:
    trtf_test::TempDirGuard mTempDir;
    std::filesystem::path mRoot;
    std::filesystem::path mDataRoot;
    std::filesystem::path mModelDir;
    std::filesystem::path mPythonCmd;
    std::string mDataRootString;
    std::unique_ptr<trtf_test::EnvVarGuard> mDataDirGuard;
};

// ---------------------------------------------------------------------------
// Helper-function tests
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

static void test_shell_quote_simple()
{
    const std::string result = trtf::hf_tok_detail::shell_quote("hello");
    check(result == "'hello'", "shell_quote simple");
}

static void test_shell_quote_with_single_quote()
{
    const std::string result = trtf::hf_tok_detail::shell_quote("it's");
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
    check(result.size() == 2, "parse_int_list skips non-numeric size");
    check(result[0] == 1, "parse_int_list skips non-numeric [0]");
    check(result[1] == 3, "parse_int_list skips non-numeric [1]");
}

static void test_parse_int_list_skips_partial_tokens()
{
    const auto result = trtf::hf_tok_detail::parse_int_list("10 20x +30 -40a 50");
    check(result == (std::vector<int32_t>{10, 30, 50}), "parse_int_list skips partial tokens");
}

static void test_parse_int_list_skips_overflow_token()
{
    const auto result = trtf::hf_tok_detail::parse_int_list("999999999999999999999999 5");
    check(result == (std::vector<int32_t>{5}), "parse_int_list skips overflow token");
}

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

static void test_sanitize_hf_output_strips_indented_warning_lines()
{
    const std::string input =
        "   None of PyTorch, TensorFlow >= 2.0, or Flax have been found.\n"
        "\tSkipping import of cpp extensions due to incompatible torch version\n"
        "payload\n";
    const std::string result = trtf::hf_tok_detail::sanitize_hf_output(input);
    check(result == "payload", "sanitize_hf_output strips indented warning lines");
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

// ---------------------------------------------------------------------------
// Runtime-behavior tests (no external deps)
// ---------------------------------------------------------------------------

static void test_constructor_missing_script_throws()
{
    trtf_test::TempDirGuard temp;
    const std::filesystem::path root(temp.path());
    const std::filesystem::path data_root = root / "empty data root";
    const std::filesystem::path model_dir = root / "existing model";
    std::filesystem::create_directories(data_root);
    std::filesystem::create_directories(model_dir);

    trtf_test::EnvVarGuard data_guard("TRTF_DATA_DIR", data_root.string().c_str());

    bool threw = false;
    std::string msg;
    try
    {
        auto tok = trtf::CreateHfPythonTokenizer(model_dir.string(), "/bin/sh", false);
        (void) tok;
    }
    catch (const std::runtime_error& e)
    {
        threw = true;
        msg = e.what();
    }
    check(threw, "constructor missing script throws");
    check_contains(msg, "Missing HF tokenizer script", "constructor missing script message");
}

static void test_constructor_missing_model_dir_throws()
{
    HfTokenizerHarness env;
    const std::filesystem::path missing_model = env.model_dir().parent_path() / "missing model";

    bool threw = false;
    std::string msg;
    try
    {
        auto tok = trtf::CreateHfPythonTokenizer(missing_model.string(), env.python_cmd().string(), false);
        (void) tok;
    }
    catch (const std::runtime_error& e)
    {
        threw = true;
        msg = e.what();
    }
    check(threw, "constructor missing model throws");
    check_contains(msg, "HF tokenizer model directory does not exist", "constructor missing model message");
}

static void test_constructor_check_failure_throws_trimmed_output()
{
    HfTokenizerHarness env;
    trtf_test::EnvVarGuard mode("HF_TEST_MODE", "check_fail");

    bool threw = false;
    std::string msg;
    try
    {
        auto tok = env.create();
        (void) tok;
    }
    catch (const std::runtime_error& e)
    {
        threw = true;
        msg = e.what();
    }
    check(threw, "constructor check failure throws");
    check_contains(msg, "HF tokenizer check failed:", "constructor check failure prefix");
    check_contains(msg, "check failed output", "constructor check failure message content");
    check(!msg.empty() && msg.back() != '\n' && msg.back() != '\r', "constructor check message trimmed");
}

static void test_encode_success_sanitizes_and_parses()
{
    HfTokenizerHarness env;
    auto tok = env.create();

    check_ids_equal(tok->encode("warnings"), {7, 8, 9}, "encode sanitizes warning lines");
    check_ids_equal(tok->encode("mixed"), {1, 3}, "encode skips invalid/partial numeric tokens");
}

static void test_encode_add_special_tokens_flag()
{
    HfTokenizerHarness env;
    auto tok_plain = env.create(false);
    auto tok_special = env.create(true);

    check_ids_equal(tok_plain->encode("include_special"), {11, 12}, "encode without special tokens");
    check_ids_equal(tok_special->encode("include_special"), {101, 11, 12, 102}, "encode with special tokens");
}

static void test_encode_failure_throws()
{
    HfTokenizerHarness env;
    auto tok = env.create();
    trtf_test::EnvVarGuard mode("HF_TEST_MODE", "encode_fail");

    bool threw = false;
    std::string msg;
    try
    {
        (void) tok->encode("anything");
    }
    catch (const std::runtime_error& e)
    {
        threw = true;
        msg = e.what();
    }
    check(threw, "encode failure throws");
    check_contains(msg, "HF tokenizer encode failed:", "encode failure prefix");
    check_contains(msg, "encode failed output", "encode failure message content");
}

static void test_decode_success_sanitizes_and_roundtrips_ids_csv()
{
    HfTokenizerHarness env;
    auto tok = env.create();

    const std::string sanitized = tok->decode({1, 2, 3});
    check(sanitized == "decoded text", "decode sanitizes warning lines");
    check(tok->decode({4, -5, 0}) == "decoded:4,-5,0", "decode joins ids csv including negatives");
}

static void test_decode_failure_throws()
{
    HfTokenizerHarness env;
    auto tok = env.create();
    trtf_test::EnvVarGuard mode("HF_TEST_MODE", "decode_fail");

    bool threw = false;
    std::string msg;
    try
    {
        (void) tok->decode({1, 2});
    }
    catch (const std::runtime_error& e)
    {
        threw = true;
        msg = e.what();
    }
    check(threw, "decode failure throws");
    check_contains(msg, "HF tokenizer decode failed:", "decode failure prefix");
    check_contains(msg, "decode failed output", "decode failure message content");
}

static void test_id_for_token_success_shell_quoting_and_empty_output_fallback()
{
    HfTokenizerHarness env;
    auto tok = env.create();

    check(tok->id_for_token("a'b c") == 321, "id_for_token handles shell quoting");
    check(tok->id_for_token("blank") == 0, "id_for_token empty sanitized output defaults to 0");
}

static void test_id_for_token_failure_throws()
{
    HfTokenizerHarness env;
    auto tok = env.create();
    trtf_test::EnvVarGuard mode("HF_TEST_MODE", "id_fail");

    bool threw = false;
    std::string msg;
    try
    {
        (void) tok->id_for_token("abc");
    }
    catch (const std::runtime_error& e)
    {
        threw = true;
        msg = e.what();
    }
    check(threw, "id_for_token failure throws");
    check_contains(msg, "HF tokenizer id-for-token failed:", "id_for_token failure prefix");
    check_contains(msg, "id failed output", "id_for_token failure message content");
}

static void test_id_for_token_invalid_number_throws()
{
    HfTokenizerHarness env;
    auto tok = env.create();

    bool threw = false;
    try
    {
        (void) tok->id_for_token("bad");
    }
    catch (const std::invalid_argument&)
    {
        threw = true;
    }
    check(threw, "id_for_token invalid number throws invalid_argument");
}

static void test_token_for_id_success_sanitizes()
{
    HfTokenizerHarness env;
    auto tok = env.create();

    check(tok->token_for_id(77) == "tok77", "token_for_id sanitizes warning lines");
    check(tok->token_for_id(8) == "tok8", "token_for_id basic");
}

static void test_token_for_id_failure_throws()
{
    HfTokenizerHarness env;
    auto tok = env.create();
    trtf_test::EnvVarGuard mode("HF_TEST_MODE", "token_fail");

    bool threw = false;
    std::string msg;
    try
    {
        (void) tok->token_for_id(1);
    }
    catch (const std::runtime_error& e)
    {
        threw = true;
        msg = e.what();
    }
    check(threw, "token_for_id failure throws");
    check_contains(msg, "HF tokenizer token-for-id failed:", "token_for_id failure prefix");
    check_contains(msg, "token failed output", "token_for_id failure message content");
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
    test_parse_int_list_skips_partial_tokens();
    test_parse_int_list_skips_overflow_token();

    test_join_ids_csv_basic();
    test_join_ids_csv_single();
    test_join_ids_csv_empty();
    test_join_ids_csv_negative();

    test_sanitize_hf_output_clean();
    test_sanitize_hf_output_strips_warnings();
    test_sanitize_hf_output_strips_cpp_ext_warning();
    test_sanitize_hf_output_strips_indented_warning_lines();
    test_sanitize_hf_output_empty_lines();
    test_sanitize_hf_output_preserves_multiline();

    test_constructor_missing_script_throws();
    test_constructor_missing_model_dir_throws();
    test_constructor_check_failure_throws_trimmed_output();
    test_encode_success_sanitizes_and_parses();
    test_encode_add_special_tokens_flag();
    test_encode_failure_throws();
    test_decode_success_sanitizes_and_roundtrips_ids_csv();
    test_decode_failure_throws();
    test_id_for_token_success_shell_quoting_and_empty_output_fallback();
    test_id_for_token_failure_throws();
    test_id_for_token_invalid_number_throws();
    test_token_for_id_success_sanitizes();
    test_token_for_id_failure_throws();

    if (failures > 0)
    {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All hf_python_tokenizer tests passed.\n";
    return 0;
}
