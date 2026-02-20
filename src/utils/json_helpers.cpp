#include "utils/json_helpers.h"

#include <cctype>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace trtf {

std::string extract_json_string(const std::string& text, const std::string& key, const std::string& fallback)
{
    const std::string needle = "\"" + key + "\"";
    const std::size_t key_pos = text.find(needle);
    if (key_pos == std::string::npos)
    {
        return fallback;
    }

    const std::size_t colon = text.find(':', key_pos);
    if (colon == std::string::npos)
    {
        return fallback;
    }

    const std::size_t first_quote = text.find('"', colon + 1);
    if (first_quote == std::string::npos)
    {
        return fallback;
    }
    const std::size_t second_quote = text.find('"', first_quote + 1);
    if (second_quote == std::string::npos || second_quote <= first_quote + 1)
    {
        return fallback;
    }
    return text.substr(first_quote + 1, second_quote - first_quote - 1);
}

std::vector<std::string> extract_json_string_array(const std::string& text, const std::string& key)
{
    const std::string needle = "\"" + key + "\"";
    const std::size_t key_pos = text.find(needle);
    if (key_pos == std::string::npos)
    {
        return {};
    }

    const std::size_t colon = text.find(':', key_pos);
    if (colon == std::string::npos)
    {
        return {};
    }

    const std::size_t open_bracket = text.find('[', colon + 1);
    if (open_bracket == std::string::npos)
    {
        return {};
    }

    std::vector<std::string> out;
    std::size_t pos = open_bracket + 1;
    while (pos < text.size())
    {
        while (pos < text.size() && (std::isspace(static_cast<unsigned char>(text[pos])) != 0 || text[pos] == ','))
        {
            ++pos;
        }
        if (pos >= text.size() || text[pos] == ']')
        {
            break;
        }
        if (text[pos] != '"')
        {
            break;
        }

        const std::size_t first_quote = pos;
        const std::size_t second_quote = text.find('"', first_quote + 1);
        if (second_quote == std::string::npos || second_quote <= first_quote + 1)
        {
            break;
        }

        out.push_back(text.substr(first_quote + 1, second_quote - first_quote - 1));
        pos = second_quote + 1;
    }

    return out;
}

int32_t extract_json_int(const std::string& text, const std::string& key, int32_t fallback)
{
    const std::string needle = "\"" + key + "\"";
    const std::size_t key_pos = text.find(needle);
    if (key_pos == std::string::npos)
    {
        return fallback;
    }

    const std::size_t colon = text.find(':', key_pos);
    if (colon == std::string::npos)
    {
        return fallback;
    }

    std::size_t pos = colon + 1;
    while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos])) != 0)
    {
        ++pos;
    }

    std::size_t end = pos;
    while (end < text.size() && (std::isdigit(static_cast<unsigned char>(text[end])) != 0 || text[end] == '-'))
    {
        ++end;
    }

    if (end == pos)
    {
        return fallback;
    }

    return static_cast<int32_t>(std::stoi(text.substr(pos, end - pos)));
}

int32_t extract_json_int_or_first_array(const std::string& text, const std::string& key, int32_t fallback)
{
    const std::string needle = "\"" + key + "\"";
    const std::size_t key_pos = text.find(needle);
    if (key_pos == std::string::npos)
    {
        return fallback;
    }

    const std::size_t colon = text.find(':', key_pos);
    if (colon == std::string::npos)
    {
        return fallback;
    }

    std::size_t pos = colon + 1;
    while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos])) != 0)
    {
        ++pos;
    }

    if (pos < text.size() && text[pos] == '[')
    {
        ++pos;
        while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos])) != 0)
        {
            ++pos;
        }
    }

    std::size_t end = pos;
    while (end < text.size() && (std::isdigit(static_cast<unsigned char>(text[end])) != 0 || text[end] == '-'))
    {
        ++end;
    }

    if (end == pos)
    {
        return fallback;
    }

    return static_cast<int32_t>(std::stoi(text.substr(pos, end - pos)));
}

float extract_json_float(const std::string& text, const std::string& key, float fallback)
{
    const std::string needle = "\"" + key + "\"";
    const std::size_t key_pos = text.find(needle);
    if (key_pos == std::string::npos)
    {
        return fallback;
    }

    const std::size_t colon = text.find(':', key_pos);
    if (colon == std::string::npos)
    {
        return fallback;
    }

    std::size_t pos = colon + 1;
    while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos])) != 0)
    {
        ++pos;
    }

    std::size_t end = pos;
    while (end < text.size())
    {
        const char c = text[end];
        const bool numeric = (std::isdigit(static_cast<unsigned char>(c)) != 0) || c == '-' || c == '+'
            || c == '.' || c == 'e' || c == 'E';
        if (!numeric)
        {
            break;
        }
        ++end;
    }

    if (end == pos)
    {
        return fallback;
    }

    try
    {
        return std::stof(text.substr(pos, end - pos));
    }
    catch (const std::exception&)
    {
        return fallback;
    }
}

std::vector<float> extract_json_float_array(const std::string& text, const std::string& key, std::size_t max_count)
{
    const std::string needle = "\"" + key + "\"";
    const std::size_t key_pos = text.find(needle);
    if (key_pos == std::string::npos)
    {
        return {};
    }

    const std::size_t colon = text.find(':', key_pos);
    if (colon == std::string::npos)
    {
        return {};
    }

    const std::size_t open_bracket = text.find('[', colon + 1);
    if (open_bracket == std::string::npos)
    {
        return {};
    }

    std::vector<float> out;
    std::size_t pos = open_bracket + 1;
    while (pos < text.size() && out.size() < max_count)
    {
        while (pos < text.size() && (std::isspace(static_cast<unsigned char>(text[pos])) != 0 || text[pos] == ','))
        {
            ++pos;
        }
        if (pos >= text.size() || text[pos] == ']')
        {
            break;
        }

        std::size_t end = pos;
        while (end < text.size())
        {
            const char c = text[end];
            const bool numeric = (std::isdigit(static_cast<unsigned char>(c)) != 0) || c == '-' || c == '+'
                || c == '.' || c == 'e' || c == 'E';
            if (!numeric)
            {
                break;
            }
            ++end;
        }

        if (end == pos)
        {
            break;
        }

        try
        {
            out.push_back(std::stof(text.substr(pos, end - pos)));
        }
        catch (const std::exception&)
        {
            break;
        }
        pos = end;
    }

    return out;
}

std::vector<int32_t> extract_json_int_array(const std::string& text, const std::string& key, std::size_t max_count)
{
    const std::string needle = "\"" + key + "\"";
    const std::size_t key_pos = text.find(needle);
    if (key_pos == std::string::npos)
    {
        return {};
    }

    const std::size_t colon = text.find(':', key_pos);
    if (colon == std::string::npos)
    {
        return {};
    }

    const std::size_t open_bracket = text.find('[', colon + 1);
    if (open_bracket == std::string::npos)
    {
        return {};
    }

    std::vector<int32_t> out;
    std::size_t pos = open_bracket + 1;
    while (pos < text.size() && out.size() < max_count)
    {
        while (pos < text.size() && (std::isspace(static_cast<unsigned char>(text[pos])) != 0 || text[pos] == ','))
        {
            ++pos;
        }
        if (pos >= text.size() || text[pos] == ']')
        {
            break;
        }

        std::size_t end = pos;
        while (end < text.size() && (std::isdigit(static_cast<unsigned char>(text[end])) != 0 || text[end] == '-'))
        {
            ++end;
        }

        if (end == pos)
        {
            break;
        }

        try
        {
            out.push_back(static_cast<int32_t>(std::stoi(text.substr(pos, end - pos))));
        }
        catch (const std::exception&)
        {
            break;
        }
        pos = end;
    }

    return out;
}

int32_t parse_positive_env_int(const char* env_name, int32_t fallback)
{
    const char* env = std::getenv(env_name);
    if (env == nullptr || env[0] == '\0')
    {
        return fallback;
    }

    try
    {
        std::size_t parsed = 0;
        const int value = std::stoi(env, &parsed);
        if (parsed != std::strlen(env) || value <= 0)
        {
            return fallback;
        }
        return static_cast<int32_t>(value);
    }
    catch (const std::exception&)
    {
        return fallback;
    }
}

} // namespace trtf
