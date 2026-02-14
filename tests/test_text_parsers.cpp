// Unit tests for src/utils/text_parsers.cpp (string parsing utilities).
// CPU-only, no TRT/CUDA deps.

#include "utils/text_parsers.h"

#include <iostream>
#include <string>
#include <vector>

namespace {

bool test_starts_with_match()
{
    return trtf::starts_with("hello world", "hello");
}

bool test_starts_with_no_match()
{
    return !trtf::starts_with("hello world", "world");
}

bool test_starts_with_empty_prefix()
{
    return trtf::starts_with("hello", "");
}

bool test_starts_with_empty_string()
{
    return !trtf::starts_with("", "hello");
}

bool test_ends_with_match()
{
    return trtf::ends_with("hello world", "world");
}

bool test_ends_with_no_match()
{
    return !trtf::ends_with("hello world", "hello");
}

bool test_ends_with_empty_suffix()
{
    return trtf::ends_with("hello", "");
}

bool test_to_lower_ascii()
{
    const std::string result = trtf::to_lower_ascii("Hello World 123!");
    if (result != "hello world 123!")
    {
        std::cerr << "to_lower_ascii: got '" << result << "'" << std::endl;
        return false;
    }
    return true;
}

bool test_trim_leading()
{
    const std::string result = trtf::trim("  hello");
    if (result != "hello")
    {
        std::cerr << "trim_leading: got '" << result << "'" << std::endl;
        return false;
    }
    return true;
}

bool test_trim_trailing()
{
    const std::string result = trtf::trim("hello   ");
    if (result != "hello")
    {
        std::cerr << "trim_trailing: got '" << result << "'" << std::endl;
        return false;
    }
    return true;
}

bool test_trim_both()
{
    const std::string result = trtf::trim("  hello   ");
    if (result != "hello")
    {
        std::cerr << "trim_both: got '" << result << "'" << std::endl;
        return false;
    }
    return true;
}

bool test_trim_empty()
{
    const std::string result = trtf::trim("");
    if (!result.empty())
    {
        std::cerr << "trim_empty: got '" << result << "'" << std::endl;
        return false;
    }
    return true;
}

bool test_trim_no_whitespace()
{
    const std::string result = trtf::trim("hello");
    if (result != "hello")
    {
        std::cerr << "trim_no_whitespace: got '" << result << "'" << std::endl;
        return false;
    }
    return true;
}

bool test_strip_inline_comment_with_comment()
{
    const std::string result = trtf::strip_inline_comment("hello # comment");
    if (result != "hello")
    {
        std::cerr << "strip_comment: got '" << result << "'" << std::endl;
        return false;
    }
    return true;
}

bool test_strip_inline_comment_no_comment()
{
    const std::string result = trtf::strip_inline_comment("hello world");
    if (result != "hello world")
    {
        std::cerr << "strip_no_comment: got '" << result << "'" << std::endl;
        return false;
    }
    return true;
}

bool test_strip_inline_comment_only_comment()
{
    const std::string result = trtf::strip_inline_comment("# comment only");
    if (!result.empty())
    {
        std::cerr << "strip_only_comment: got '" << result << "'" << std::endl;
        return false;
    }
    return true;
}

bool test_split_words_basic()
{
    const auto result = trtf::split_words("hello world");
    if (result.size() != 2 || result[0] != "hello" || result[1] != "world")
    {
        std::cerr << "split_words_basic: size=" << result.size() << std::endl;
        return false;
    }
    return true;
}

bool test_split_words_empty()
{
    const auto result = trtf::split_words("");
    if (!result.empty())
    {
        std::cerr << "split_words_empty: size=" << result.size() << std::endl;
        return false;
    }
    return true;
}

bool test_split_words_multiple_spaces()
{
    const auto result = trtf::split_words("  a   b  c  ");
    if (result.size() != 3 || result[0] != "a" || result[1] != "b" || result[2] != "c")
    {
        std::cerr << "split_words_spaces: size=" << result.size() << std::endl;
        return false;
    }
    return true;
}

bool test_iequals_ascii_match()
{
    return trtf::iequals_ascii("Hello", "hello");
}

bool test_iequals_ascii_no_match()
{
    return !trtf::iequals_ascii("Hello", "world");
}

bool test_iequals_ascii_different_length()
{
    return !trtf::iequals_ascii("Hello", "Hell");
}

} // namespace

int main()
{
    bool all_passed = true;
    std::cout << "test_text_parsers:" << std::endl;

    const auto run = [&](const char* name, bool (*fn)()) {
        const bool ok = fn();
        std::cout << "  " << name << ": " << (ok ? "PASS" : "FAIL") << std::endl;
        all_passed &= ok;
    };

    run("starts_with_match", test_starts_with_match);
    run("starts_with_no_match", test_starts_with_no_match);
    run("starts_with_empty_prefix", test_starts_with_empty_prefix);
    run("starts_with_empty_string", test_starts_with_empty_string);
    run("ends_with_match", test_ends_with_match);
    run("ends_with_no_match", test_ends_with_no_match);
    run("ends_with_empty_suffix", test_ends_with_empty_suffix);
    run("to_lower_ascii", test_to_lower_ascii);
    run("trim_leading", test_trim_leading);
    run("trim_trailing", test_trim_trailing);
    run("trim_both", test_trim_both);
    run("trim_empty", test_trim_empty);
    run("trim_no_whitespace", test_trim_no_whitespace);
    run("strip_comment", test_strip_inline_comment_with_comment);
    run("strip_no_comment", test_strip_inline_comment_no_comment);
    run("strip_only_comment", test_strip_inline_comment_only_comment);
    run("split_words_basic", test_split_words_basic);
    run("split_words_empty", test_split_words_empty);
    run("split_words_spaces", test_split_words_multiple_spaces);
    run("iequals_match", test_iequals_ascii_match);
    run("iequals_no_match", test_iequals_ascii_no_match);
    run("iequals_diff_length", test_iequals_ascii_different_length);

    if (all_passed)
    {
        std::cout << "test_text_parsers passed" << std::endl;
        return 0;
    }
    std::cerr << "test_text_parsers FAILED" << std::endl;
    return 1;
}
