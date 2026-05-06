#include "utils/json_helpers.h"

#include <cctype>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace trtmc {
namespace {

enum class ArrayParseState { kReady, kEnd };

bool is_digit_char(char c) {
    return std::isdigit(static_cast<unsigned char>(c)) != 0;
}

bool is_space_char(char c) {
    return std::isspace(static_cast<unsigned char>(c)) != 0;
}

bool is_space_or_comma(char c) {
    if (c == ',') {
        return true;
    }
    return is_space_char(c);
}

bool is_int_char(char c) {
    if (is_digit_char(c)) {
        return true;
    }
    return c == '-';
}

bool is_float_char(char c) {
    if (is_digit_char(c)) {
        return true;
    }
    return c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E';
}

std::size_t skip_whitespace(const std::string& text, std::size_t pos) {
    while (pos < text.size() && is_space_char(text[pos])) {
        ++pos;
    }
    return pos;
}

std::size_t skip_space_or_commas(const std::string& text, std::size_t pos) {
    while (pos < text.size() && is_space_or_comma(text[pos])) {
        ++pos;
    }
    return pos;
}

bool find_key_colon(const std::string& text, const std::string& key, std::size_t& colon) {
    const std::string needle = "\"" + key + "\"";
    const std::size_t key_pos = text.find(needle);
    if (key_pos == std::string::npos) {
        return false;
    }

    colon = text.find(':', key_pos);
    return colon != std::string::npos;
}

bool find_array_start(const std::string& text, std::size_t colon, std::size_t& open_bracket) {
    open_bracket = text.find('[', colon + 1);
    return open_bracket != std::string::npos;
}

std::size_t scan_while(const std::string& text, std::size_t pos, bool (*is_allowed)(char)) {
    std::size_t end = pos;
    while (end < text.size() && is_allowed(text[end])) {
        ++end;
    }
    return end;
}

ArrayParseState advance_array_pos(const std::string& text, std::size_t& pos) {
    pos = skip_space_or_commas(text, pos);
    if (pos >= text.size()) {
        return ArrayParseState::kEnd;
    }
    if (text[pos] == ']') {
        return ArrayParseState::kEnd;
    }
    return ArrayParseState::kReady;
}

bool read_quoted_token(const std::string& text, std::size_t& pos, std::string& out) {
    if (pos >= text.size() || text[pos] != '"') {
        return false;
    }

    const std::size_t first_quote = pos;
    const std::size_t second_quote = text.find('"', first_quote + 1);
    if (second_quote == std::string::npos || second_quote <= first_quote + 1) {
        return false;
    }

    out = text.substr(first_quote + 1, second_quote - first_quote - 1);
    pos = second_quote + 1;
    return true;
}

template <typename T, typename Parser>
std::vector<T> extract_numeric_array_impl(const std::string& text, const std::string& key,
                                          std::size_t max_count, bool (*is_allowed)(char),
                                          Parser parse) {
    std::size_t colon = 0;
    if (!find_key_colon(text, key, colon)) {
        return {};
    }

    std::size_t open_bracket = 0;
    if (!find_array_start(text, colon, open_bracket)) {
        return {};
    }

    std::vector<T> out;
    std::size_t pos = open_bracket + 1;
    while (pos < text.size() && out.size() < max_count) {
        if (advance_array_pos(text, pos) != ArrayParseState::kReady) {
            break;
        }

        const std::size_t end = scan_while(text, pos, is_allowed);
        if (end == pos) {
            break;
        }

        if (!parse(text.substr(pos, end - pos), out)) {
            break;
        }
        pos = end;
    }

    return out;
}

} // namespace

std::string extract_json_string(const std::string& text, const std::string& key,
                                const std::string& fallback) {
    std::size_t colon = 0;
    if (!find_key_colon(text, key, colon)) {
        return fallback;
    }

    const std::size_t first_quote = text.find('"', colon + 1);
    if (first_quote == std::string::npos) {
        return fallback;
    }

    std::size_t pos = first_quote;
    std::string parsed;
    if (!read_quoted_token(text, pos, parsed)) {
        return fallback;
    }
    return parsed;
}

std::vector<std::string> extract_json_string_array(const std::string& text,
                                                   const std::string& key) {
    std::size_t colon = 0;
    if (!find_key_colon(text, key, colon)) {
        return {};
    }

    std::size_t open_bracket = 0;
    if (!find_array_start(text, colon, open_bracket)) {
        return {};
    }

    std::vector<std::string> out;
    std::size_t pos = open_bracket + 1;
    while (pos < text.size()) {
        if (advance_array_pos(text, pos) != ArrayParseState::kReady) {
            break;
        }

        std::string parsed;
        if (!read_quoted_token(text, pos, parsed)) {
            break;
        }
        out.push_back(parsed);
    }

    return out;
}

int32_t extract_json_int(const std::string& text, const std::string& key, int32_t fallback) {
    std::size_t colon = 0;
    if (!find_key_colon(text, key, colon)) {
        return fallback;
    }

    const std::size_t pos = skip_whitespace(text, colon + 1);
    const std::size_t end = scan_while(text, pos, is_int_char);

    if (end == pos) {
        return fallback;
    }

    return static_cast<int32_t>(std::stoi(text.substr(pos, end - pos)));
}

int32_t extract_json_int_or_first_array(const std::string& text, const std::string& key,
                                        int32_t fallback) {
    std::size_t colon = 0;
    if (!find_key_colon(text, key, colon)) {
        return fallback;
    }

    std::size_t pos = skip_whitespace(text, colon + 1);

    if (pos < text.size() && text[pos] == '[') {
        pos = skip_whitespace(text, pos + 1);
    }

    const std::size_t end = scan_while(text, pos, is_int_char);

    if (end == pos) {
        return fallback;
    }

    return static_cast<int32_t>(std::stoi(text.substr(pos, end - pos)));
}

float extract_json_float(const std::string& text, const std::string& key, float fallback) {
    std::size_t colon = 0;
    if (!find_key_colon(text, key, colon)) {
        return fallback;
    }

    const std::size_t pos = skip_whitespace(text, colon + 1);
    const std::size_t end = scan_while(text, pos, is_float_char);

    if (end == pos) {
        return fallback;
    }

    try {
        return std::stof(text.substr(pos, end - pos));
    } catch (const std::exception&) {
        return fallback;
    }
}

std::vector<float> extract_json_float_array(const std::string& text, const std::string& key,
                                            std::size_t max_count) {
    auto parse_float = [](const std::string& token, std::vector<float>& out) {
        try {
            out.push_back(std::stof(token));
            return true;
        } catch (const std::exception&) {
            return false;
        }
    };
    return extract_numeric_array_impl<float>(text, key, max_count, is_float_char, parse_float);
}

std::vector<int32_t> extract_json_int_array(const std::string& text, const std::string& key,
                                            std::size_t max_count) {
    auto parse_int = [](const std::string& token, std::vector<int32_t>& out) {
        try {
            out.push_back(static_cast<int32_t>(std::stoi(token)));
            return true;
        } catch (const std::exception&) {
            return false;
        }
    };
    return extract_numeric_array_impl<int32_t>(text, key, max_count, is_int_char, parse_int);
}

} // namespace trtmc
