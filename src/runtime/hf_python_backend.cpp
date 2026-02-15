#include "trtf/backend.h"
#include "utils/data_dir.h"

#include <array>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

namespace trtf {
namespace {

struct CommandResult {
    int exit_code{1};
    std::string output;
};

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

std::filesystem::path runner_script_path()
{
    return std::filesystem::path(script_path("hf_generate.py"));
}

class HfPythonBackend final : public IGenerationBackend {
public:
    HfPythonBackend(std::string model_dir, std::string python_path)
        : mModelDir(std::move(model_dir))
    {
        mPythonCommand = find_python_command(python_path);
        mRunnerScript = runner_script_path();

        if (!std::filesystem::exists(mRunnerScript))
        {
            mInitError = "Missing HF runner script: " + mRunnerScript.string();
            return;
        }

        if (!std::filesystem::exists(mModelDir))
        {
            mInitError = "Model directory does not exist: " + mModelDir;
            return;
        }

        const std::string check_cmd = shell_quote(mPythonCommand) + " " + shell_quote(mRunnerScript.string())
            + " --check --model-dir " + shell_quote(mModelDir);
        const CommandResult check = run_command_capture(check_cmd);
        if (check.exit_code != 0)
        {
            mInitError = "HF runner check failed: " + trim_trailing_newlines(check.output);
            return;
        }

        mAvailable = true;
    }

    bool is_available() const override
    {
        return mAvailable;
    }

    const char* name() const override
    {
        return "hf-transformers";
    }

    std::vector<int32_t> generate(const std::vector<int32_t>& input_ids, const GenerationConfig& config) override
    {
        (void) input_ids;
        (void) config;
        throw std::runtime_error("hf-transformers backend supports only direct text generation.");
    }

    bool supports_text_generation() const override
    {
        return true;
    }

    std::string generate_text(const std::string& prompt, const GenerationConfig& config) override
    {
        if (!mAvailable)
        {
            throw std::runtime_error("hf-transformers backend is unavailable: " + mInitError);
        }

        char temp_path[] = "/tmp/trtf_hf_prompt_XXXXXX";
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
                throw std::runtime_error("Failed to write temporary prompt file.");
            }
            out << prompt;
        }
        close(fd);

        const std::string cmd = shell_quote(mPythonCommand) + " " + shell_quote(mRunnerScript.string())
            + " --model-dir " + shell_quote(mModelDir)
            + " --prompt-file " + shell_quote(temp_path)
            + " --max-new-tokens " + std::to_string(config.max_new_tokens)
            + " --do-sample " + std::to_string(config.do_sample ? 1 : 0)
            + " --temperature " + std::to_string(config.temperature);

        const CommandResult result = run_command_capture(cmd);

        std::error_code remove_ec;
        std::filesystem::remove(temp_path, remove_ec);

        if (result.exit_code != 0)
        {
            throw std::runtime_error("hf-transformers generation failed: " + trim_trailing_newlines(result.output));
        }

        return trim_trailing_newlines(result.output);
    }

private:
    std::string mModelDir;
    std::string mPythonCommand;
    std::filesystem::path mRunnerScript;
    bool mAvailable{false};
    std::string mInitError;
};

} // namespace

std::unique_ptr<IGenerationBackend> CreateHfPythonBackend(const std::string& model_dir, const std::string& python_path)
{
    return std::make_unique<HfPythonBackend>(model_dir, python_path);
}

} // namespace trtf
