// =============================================================================
// test_text_parsers.cpp — Unit tests for src/utils/text_parsers.cpp
// =============================================================================
//
// Purpose:
//   Validates the string parsing utility functions used throughout the codebase
//   for tasks such as prefix/suffix matching, case-insensitive comparison,
//   whitespace trimming, comment stripping, word splitting, and ASCII case
//   conversion. These utilities support vocabulary loading, config parsing,
//   and model ID matching.
//
// Dependencies:
//   - utils/text_parsers.h (starts_with, ends_with, to_lower_ascii, trim,
//     strip_inline_comment, split_words, iequals_ascii)
//
// Approach:
//   Each test calls a single text_parsers function with a carefully chosen
//   input and checks that the output matches expectations. Tests cover both
//   typical usage and edge cases (empty strings, strings with only whitespace,
//   strings with only comments, multiple consecutive delimiters, differing
//   string lengths for comparison).
//
// Environment:
//   CPU-only, no TRT/CUDA dependencies. No filesystem access required.
// =============================================================================

#include "utils/text_parsers.h"

#include <iostream>
#include <string>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// starts_with tests
// ---------------------------------------------------------------------------

// Intention: Verify starts_with returns true for a matching prefix.
// Setup:     Input string "hello world", prefix "hello".
// Mechanism: Calls starts_with, expects true.
bool test_starts_with_match()
{
    return trtf::starts_with("hello world", "hello");
}

// Intention: Verify starts_with returns false when the prefix does not match
//            the beginning of the string (even though "world" appears later).
// Setup:     Input string "hello world", prefix "world".
// Mechanism: Calls starts_with, expects false.
bool test_starts_with_no_match()
{
    return !trtf::starts_with("hello world", "world");
}

// Intention: Verify that an empty prefix matches any string (every string
//            starts with the empty string).
// Setup:     Input string "hello", prefix "".
// Mechanism: Calls starts_with, expects true.
bool test_starts_with_empty_prefix()
{
    return trtf::starts_with("hello", "");
}

// Intention: Verify that an empty string does not start with a non-empty
//            prefix.
// Setup:     Input string "", prefix "hello".
// Mechanism: Calls starts_with, expects false.
bool test_starts_with_empty_string()
{
    return !trtf::starts_with("", "hello");
}

// ---------------------------------------------------------------------------
// ends_with tests
// ---------------------------------------------------------------------------

// Intention: Verify ends_with returns true for a matching suffix.
// Setup:     Input string "hello world", suffix "world".
// Mechanism: Calls ends_with, expects true.
bool test_ends_with_match()
{
    return trtf::ends_with("hello world", "world");
}

// Intention: Verify ends_with returns false when the suffix does not match
//            the end of the string (even though "hello" appears at the start).
// Setup:     Input string "hello world", suffix "hello".
// Mechanism: Calls ends_with, expects false.
bool test_ends_with_no_match()
{
    return !trtf::ends_with("hello world", "hello");
}

// Intention: Verify that an empty suffix matches any string (every string
//            ends with the empty string).
// Setup:     Input string "hello", suffix "".
// Mechanism: Calls ends_with, expects true.
bool test_ends_with_empty_suffix()
{
    return trtf::ends_with("hello", "");
}

// ---------------------------------------------------------------------------
// to_lower_ascii test
// ---------------------------------------------------------------------------

// Intention: Verify that to_lower_ascii converts uppercase ASCII letters to
//            lowercase while leaving non-letter characters unchanged.
// Setup:     Mixed-case input "Hello World 123!" with uppercase, lowercase,
//            digits, and punctuation.
// Mechanism: Calls to_lower_ascii, checks the result is "hello world 123!".
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

// ---------------------------------------------------------------------------
// trim tests
// ---------------------------------------------------------------------------

// Intention: Verify that trim removes leading whitespace.
// Setup:     String "  hello" with two leading spaces and no trailing spaces.
// Mechanism: Calls trim, checks the result is "hello".
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

// Intention: Verify that trim removes trailing whitespace.
// Setup:     String "hello   " with three trailing spaces and no leading spaces.
// Mechanism: Calls trim, checks the result is "hello".
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

// Intention: Verify that trim removes both leading and trailing whitespace
//            simultaneously.
// Setup:     String "  hello   " with whitespace on both sides.
// Mechanism: Calls trim, checks the result is "hello".
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

// Intention: Verify that trim handles an empty string without error.
// Setup:     Empty string "".
// Mechanism: Calls trim, checks the result is still empty.
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

// Intention: Verify that trim returns the input unchanged when there is no
//            leading or trailing whitespace.
// Setup:     String "hello" with no whitespace.
// Mechanism: Calls trim, checks the result is "hello" (identity case).
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

// ---------------------------------------------------------------------------
// strip_inline_comment tests
// ---------------------------------------------------------------------------

// Intention: Verify that strip_inline_comment removes a '#' comment and
//            trims trailing whitespace from the content portion.
// Setup:     String "hello # comment" — content is "hello", comment follows '#'.
// Mechanism: Calls strip_inline_comment, checks the result is "hello" (the
//            trailing space before '#' is trimmed).
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

// Intention: Verify that strip_inline_comment returns the full string when
//            no '#' character is present.
// Setup:     String "hello world" with no comment marker.
// Mechanism: Calls strip_inline_comment, checks the result is "hello world".
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

// Intention: Verify that a line consisting entirely of a comment yields an
//            empty string.
// Setup:     String "# comment only" — the '#' is the first character.
// Mechanism: Calls strip_inline_comment, checks the result is empty.
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

// ---------------------------------------------------------------------------
// split_words tests
// ---------------------------------------------------------------------------

// Intention: Verify basic whitespace-delimited word splitting.
// Setup:     String "hello world" — two words separated by a single space.
// Mechanism: Calls split_words, checks the result has exactly 2 elements:
//            "hello" and "world".
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

// Intention: Verify that split_words returns an empty vector for an empty
//            input string.
// Setup:     Empty string "".
// Mechanism: Calls split_words, checks the result is empty.
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

// Intention: Verify that split_words handles multiple consecutive spaces,
//            leading spaces, and trailing spaces correctly without producing
//            empty "words".
// Setup:     String "  a   b  c  " with irregular spacing.
// Mechanism: Calls split_words, checks the result has exactly 3 elements:
//            "a", "b", and "c".
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

// ---------------------------------------------------------------------------
// iequals_ascii tests
// ---------------------------------------------------------------------------

// Intention: Verify case-insensitive ASCII string comparison for strings
//            that differ only in case.
// Setup:     Strings "Hello" and "hello" — same letters, different case.
// Mechanism: Calls iequals_ascii, expects true.
bool test_iequals_ascii_match()
{
    return trtf::iequals_ascii("Hello", "hello");
}

// Intention: Verify that iequals_ascii returns false for strings with
//            different content (not just a case difference).
// Setup:     Strings "Hello" and "world" — entirely different words.
// Mechanism: Calls iequals_ascii, expects false.
bool test_iequals_ascii_no_match()
{
    return !trtf::iequals_ascii("Hello", "world");
}

// Intention: Verify that iequals_ascii returns false when strings have
//            different lengths, even if one is a prefix of the other.
// Setup:     Strings "Hello" (length 5) and "Hell" (length 4).
// Mechanism: Calls iequals_ascii, expects false.
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
