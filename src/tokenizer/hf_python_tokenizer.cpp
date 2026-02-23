#include "trtf/tokenizer.h"
#include "tokenizer/hf_python_tokenizer_helpers.h"
#include "utils/data_dir.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

namespace trtf {

// --- Exported helper functions (hf_tok_detail namespace) ---

namespace hf_tok_detail {

std::string trim_trailing_newlines(std::string text)
{
    while (!text.empty() && (text.back() == '\n' || text.back() == '\r'))
    {
        text.pop_back();
    }
    return text;
}

std::string shell_quote(std::string_view value)
{
    std::string out;
    out.reserve(value.size() + 2);
    out.push_back('\'');
    for (char ch : value)
    {
        if (ch == '\'')
        {
            out += "'\"'\"'";
        }
        else
        {
            out.push_back(ch);
        }
    }
    out.push_back('\'');
    return out;
}

std::vector<int32_t> parse_int_list(const std::string& text)
{
    std::vector<int32_t> out;
    std::istringstream iss(text);
    std::string token;
    while (iss >> token)
    {
        try
        {
            std::size_t parsed = 0;
            const long long value = std::stoll(token, &parsed);
            if (parsed == token.size())
            {
                out.push_back(static_cast<int32_t>(value));
            }
        }
        catch (const std::exception&)
        {
        }
    }
    return out;
}

std::string join_ids_csv(const std::vector<int32_t>& ids)
{
    std::ostringstream oss;
    for (std::size_t i = 0; i < ids.size(); ++i)
    {
        if (i != 0)
        {
            oss << ',';
        }
        oss << ids[i];
    }
    return oss.str();
}

namespace {

std::string trim_ascii_whitespace(std::string text)
{
    const auto is_space = [](unsigned char c) { return std::isspace(c) != 0; };
    text.erase(text.begin(), std::find_if(text.begin(), text.end(), [&](unsigned char c) { return !is_space(c); }));
    text.erase(std::find_if(text.rbegin(), text.rend(), [&](unsigned char c) { return !is_space(c); }).base(), text.end());
    return text;
}

bool starts_with_ascii(std::string_view value, std::string_view prefix)
{
    return value.size() >= prefix.size() && value.compare(0, prefix.size(), prefix) == 0;
}

bool is_ignorable_hf_warning_line(const std::string& line)
{
    const std::string trimmed = trim_ascii_whitespace(line);
    if (trimmed.empty())
    {
        return true;
    }
    return starts_with_ascii(trimmed, "None of PyTorch, TensorFlow >= 2.0, or Flax have been found")
        || starts_with_ascii(trimmed, "Skipping import of cpp extensions due to incompatible torch version");
}

} // namespace

std::string sanitize_hf_output(const std::string& text)
{
    std::istringstream iss(text);
    std::string line;
    std::ostringstream filtered;
    bool first = true;
    while (std::getline(iss, line))
    {
        if (is_ignorable_hf_warning_line(line))
        {
            continue;
        }

        if (!first)
        {
            filtered << '\n';
        }
        filtered << line;
        first = false;
    }

    return trim_trailing_newlines(filtered.str());
}

} // namespace hf_tok_detail

namespace {

struct CommandResult {
    int exit_code{1};
    std::string output;
};

CommandResult run_command_capture(std::string command)
{
    command += " 2>&1";
    FILE* pipe = popen(command.c_str(), "r");
    if (pipe == nullptr)
    {
        return CommandResult{1, "popen failed"};
    }

    std::array<char, 512> buffer{};
    std::string output;
    while (true)
    {
        const std::size_t n = std::fread(buffer.data(), 1, buffer.size(), pipe);
        if (n > 0)
        {
            output.append(buffer.data(), n);
        }
        if (n < buffer.size())
        {
            if (std::feof(pipe) != 0 || std::ferror(pipe) != 0)
            {
                break;
            }
        }
    }

    const int status = pclose(pipe);
    int exit_code = 1;
    if (WIFEXITED(status))
    {
        exit_code = WEXITSTATUS(status);
    }
    return CommandResult{exit_code, output};
}

std::string find_python_command(const std::string& override_path)
{
    if (!override_path.empty())
    {
        return override_path;
    }

    const std::filesystem::path venv_python("/opt/hf-venv/bin/python");
    if (std::filesystem::exists(venv_python))
    {
        return venv_python.string();
    }

    return "python3";
}

std::filesystem::path tokenizer_script_path()
{
    return std::filesystem::path(script_path("hf_tokenizer.py"));
}

class HfPythonTokenizer final : public ITokenizer {
public:
    HfPythonTokenizer(std::string model_dir, std::string python_path, bool add_special_tokens)
        : mModelDir(std::move(model_dir))
        , mAddSpecialTokens(add_special_tokens)
    {
        mPythonCommand = find_python_command(python_path);
        mScript = tokenizer_script_path();

        if (!std::filesystem::exists(mScript))
        {
            throw std::runtime_error("Missing HF tokenizer script: " + mScript.string());
        }
        if (!std::filesystem::exists(mModelDir))
        {
            throw std::runtime_error("HF tokenizer model directory does not exist: " + mModelDir);
        }

        const std::string check_cmd = hf_tok_detail::shell_quote(mPythonCommand) + " "
            + hf_tok_detail::shell_quote(mScript.string())
            + " --check --model-dir " + hf_tok_detail::shell_quote(mModelDir);
        const CommandResult check = run_command_capture(check_cmd);
        if (check.exit_code != 0)
        {
            throw std::runtime_error("HF tokenizer check failed: "
                + hf_tok_detail::trim_trailing_newlines(check.output));
        }
    }

    std::vector<int32_t> encode(const std::string& text) const override
    {
        char temp_path[] = "/tmp/trtf_hf_tokenizer_encode_XXXXXX";
        const int fd = mkstemp(temp_path);
        if (fd < 0)
        {
            throw std::runtime_error("mkstemp failed with errno=" + std::to_string(errno));
        }

        {
            std::ofstream out(temp_path, std::ios::binary | std::ios::trunc);
            if (!out)
            {
                close(fd);
                std::filesystem::remove(temp_path);
                throw std::runtime_error("Failed to write temporary tokenizer input file.");
            }
            out << text;
        }
        close(fd);

        std::string cmd = hf_tok_detail::shell_quote(mPythonCommand) + " "
            + hf_tok_detail::shell_quote(mScript.string())
            + " --model-dir " + hf_tok_detail::shell_quote(mModelDir)
            + " --op encode"
            + " --text-file " + hf_tok_detail::shell_quote(temp_path);
        if (mAddSpecialTokens)
        {
            cmd += " --add-special-tokens";
        }

        const CommandResult result = run_command_capture(cmd);
        std::error_code remove_ec;
        std::filesystem::remove(temp_path, remove_ec);

        if (result.exit_code != 0)
        {
            throw std::runtime_error("HF tokenizer encode failed: "
                + hf_tok_detail::trim_trailing_newlines(result.output));
        }

        return hf_tok_detail::parse_int_list(hf_tok_detail::sanitize_hf_output(result.output));
    }

    std::string decode(const std::vector<int32_t>& ids) const override
    {
        const std::string cmd = hf_tok_detail::shell_quote(mPythonCommand) + " "
            + hf_tok_detail::shell_quote(mScript.string())
            + " --model-dir " + hf_tok_detail::shell_quote(mModelDir)
            + " --op decode"
            + " --ids " + hf_tok_detail::shell_quote(hf_tok_detail::join_ids_csv(ids));

        const CommandResult result = run_command_capture(cmd);
        if (result.exit_code != 0)
        {
            throw std::runtime_error("HF tokenizer decode failed: "
                + hf_tok_detail::trim_trailing_newlines(result.output));
        }
        return hf_tok_detail::sanitize_hf_output(result.output);
    }

    int32_t id_for_token(std::string_view token) const override
    {
        const std::string cmd = hf_tok_detail::shell_quote(mPythonCommand) + " "
            + hf_tok_detail::shell_quote(mScript.string())
            + " --model-dir " + hf_tok_detail::shell_quote(mModelDir)
            + " --op id-for-token"
            + " --token " + hf_tok_detail::shell_quote(std::string(token));

        const CommandResult result = run_command_capture(cmd);
        if (result.exit_code != 0)
        {
            throw std::runtime_error("HF tokenizer id-for-token failed: "
                + hf_tok_detail::trim_trailing_newlines(result.output));
        }

        const std::string out = hf_tok_detail::sanitize_hf_output(result.output);
        if (out.empty())
        {
            return 0;
        }
        return static_cast<int32_t>(std::stoi(out));
    }

    std::string token_for_id(int32_t id) const override
    {
        const std::string cmd = hf_tok_detail::shell_quote(mPythonCommand) + " "
            + hf_tok_detail::shell_quote(mScript.string())
            + " --model-dir " + hf_tok_detail::shell_quote(mModelDir)
            + " --op token-for-id"
            + " --id " + std::to_string(id);

        const CommandResult result = run_command_capture(cmd);
        if (result.exit_code != 0)
        {
            throw std::runtime_error("HF tokenizer token-for-id failed: "
                + hf_tok_detail::trim_trailing_newlines(result.output));
        }
        return hf_tok_detail::sanitize_hf_output(result.output);
    }

private:
    std::string mModelDir;
    std::string mPythonCommand;
    std::filesystem::path mScript;
    bool mAddSpecialTokens{false};
};

} // namespace

std::unique_ptr<ITokenizer> CreateHfPythonTokenizer(std::string model_dir, std::string python_path,
                                                    bool add_special_tokens)
{
    return std::make_unique<HfPythonTokenizer>(std::move(model_dir), std::move(python_path), add_special_tokens);
}

} // namespace trtf
