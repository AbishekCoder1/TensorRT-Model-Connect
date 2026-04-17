#include "trtf/pipeline.h"

#include <cctype>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Sample {
    std::string sample_id;
    std::string answer;
    std::string prompt;
};

std::string trim(std::string value) {
    std::size_t start = 0;
    while (start < value.size() && std::isspace(static_cast<unsigned char>(value[start])))
        ++start;
    std::size_t end = value.size();
    while (end > start && std::isspace(static_cast<unsigned char>(value[end - 1])))
        --end;
    return value.substr(start, end - start);
}

std::string unescape_json_string(std::string_view raw) {
    std::string out;
    out.reserve(raw.size());
    for (std::size_t i = 0; i < raw.size(); ++i) {
        char ch = raw[i];
        if (ch != '\\') {
            out.push_back(ch);
            continue;
        }
        if (i + 1 >= raw.size())
            throw std::runtime_error("Invalid trailing escape in JSON string");
        char esc = raw[++i];
        switch (esc) {
        case '\\':
            out.push_back('\\');
            break;
        case '"':
            out.push_back('"');
            break;
        case 'n':
            out.push_back('\n');
            break;
        case 'r':
            out.push_back('\r');
            break;
        case 't':
            out.push_back('\t');
            break;
        default:
            throw std::runtime_error(std::string("Unsupported JSON escape: \\") + esc);
        }
    }
    return out;
}

bool extract_json_field(const std::string& line, const std::string& key, std::string& value) {
    const std::string needle = "\"" + key + "\"";
    std::size_t pos = line.find(needle);
    if (pos == std::string::npos)
        return false;
    pos = line.find(':', pos + needle.size());
    if (pos == std::string::npos)
        throw std::runtime_error("Malformed JSON line: missing ':' for key " + key);
    ++pos;
    while (pos < line.size() && std::isspace(static_cast<unsigned char>(line[pos])))
        ++pos;
    if (pos >= line.size() || line[pos] != '"')
        throw std::runtime_error("Malformed JSON line: expected string value for key " + key);
    ++pos;
    std::string raw;
    bool escaped = false;
    for (; pos < line.size(); ++pos) {
        char ch = line[pos];
        if (escaped) {
            raw.push_back(ch);
            escaped = false;
            continue;
        }
        if (ch == '\\') {
            raw.push_back(ch);
            escaped = true;
            continue;
        }
        if (ch == '"') {
            value = unescape_json_string(raw);
            return true;
        }
        raw.push_back(ch);
    }
    throw std::runtime_error("Malformed JSON line: unterminated string for key " + key);
}

std::vector<Sample> load_samples(const std::string& dataset_path) {
    std::ifstream input(dataset_path);
    if (!input)
        throw std::runtime_error("Failed to open dataset file: " + dataset_path);

    std::vector<Sample> samples;
    std::string line;
    std::size_t line_no = 0;
    while (std::getline(input, line)) {
        ++line_no;
        if (trim(line).empty())
            continue;
        Sample sample;
        if (!extract_json_field(line, "sample_id", sample.sample_id) ||
            !extract_json_field(line, "answer", sample.answer) ||
            !extract_json_field(line, "prompt", sample.prompt)) {
            throw std::runtime_error("Dataset line missing required fields at line " +
                                     std::to_string(line_no));
        }
        samples.push_back(std::move(sample));
    }
    return samples;
}

std::string json_escape(const std::string& text) {
    std::string out;
    out.reserve(text.size() + 16);
    for (char ch : text) {
        switch (ch) {
        case '\\':
            out += "\\\\";
            break;
        case '"':
            out += "\\\"";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            out.push_back(ch);
            break;
        }
    }
    return out;
}

void usage() {
    std::cerr << "Usage: trtf_dataset_benchmark <bundle.trtfb> <dataset.jsonl> <output.jsonl> "
                 "[--max-new-tokens N] [--hf-python PATH] [--kv-cache-size SIZE] "
                 "[--temperature F] [--top-k N] [--top-p F] [--min-p F] [--seed N] "
                 "[--chat-template] [--no-thinking] [--stop-on-answer] "
                 "[--stop-check-interval N]\n";
}

std::uint64_t parse_size_bytes(const std::string& text) {
    if (text.empty())
        throw std::runtime_error("Empty kv-cache-size");
    std::size_t idx = 0;
    const double value = std::stod(text, &idx);
    std::string suffix = text.substr(idx);
    for (char& ch : suffix)
        ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
    double multiplier = 1.0;
    if (suffix.empty() || suffix == "B") {
        multiplier = 1.0;
    } else if (suffix == "K" || suffix == "KB" || suffix == "KIB") {
        multiplier = 1024.0;
    } else if (suffix == "M" || suffix == "MB" || suffix == "MIB") {
        multiplier = 1024.0 * 1024.0;
    } else if (suffix == "G" || suffix == "GB" || suffix == "GIB") {
        multiplier = 1024.0 * 1024.0 * 1024.0;
    } else if (suffix == "T" || suffix == "TB" || suffix == "TIB") {
        multiplier = 1024.0 * 1024.0 * 1024.0 * 1024.0;
    } else {
        throw std::runtime_error("Unsupported kv-cache-size suffix: " + suffix);
    }
    return static_cast<std::uint64_t>(value * multiplier);
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        usage();
        return 1;
    }

    std::string bundle_path = argv[1];
    std::string dataset_path = argv[2];
    std::string output_path = argv[3];
    int32_t max_new_tokens = 12000;
    trtf::LoadOptions load_options;
    float temperature = 1.0F;
    int32_t top_k = 1;
    float top_p = 1.0F;
    float min_p = 0.0F;
    int32_t seed = -1;
    bool use_chat_template = false;
    bool enable_thinking = true;
    bool stop_on_answer = false;
    int32_t stop_check_interval = 16;

    for (int i = 4; i < argc; ++i) {
        std::string arg = argv[i];
        auto need_value = [&](const std::string& flag) -> const char* {
            if (i + 1 >= argc)
                throw std::runtime_error("Missing value for " + flag);
            return argv[++i];
        };
        if (arg == "--max-new-tokens") {
            max_new_tokens = std::stoi(need_value(arg));
        } else if (arg == "--hf-python") {
            load_options.hf_python = need_value(arg);
        } else if (arg == "--kv-cache-size") {
            load_options.kv_cache_size_bytes = parse_size_bytes(need_value(arg));
        } else if (arg == "--temperature") {
            temperature = std::stof(need_value(arg));
        } else if (arg == "--top-k") {
            top_k = std::stoi(need_value(arg));
        } else if (arg == "--top-p") {
            top_p = std::stof(need_value(arg));
        } else if (arg == "--min-p") {
            min_p = std::stof(need_value(arg));
        } else if (arg == "--seed") {
            seed = std::stoi(need_value(arg));
        } else if (arg == "--chat-template") {
            use_chat_template = true;
        } else if (arg == "--no-thinking") {
            enable_thinking = false;
        } else if (arg == "--stop-on-answer") {
            stop_on_answer = true;
        } else if (arg == "--stop-check-interval") {
            stop_check_interval = std::stoi(need_value(arg));
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    auto samples = load_samples(dataset_path);
    auto pipeline = trtf::load(bundle_path, load_options);
    if (!pipeline)
        throw std::runtime_error("Failed to load bundle: " + bundle_path);

    trtf::GenerateConfig cfg;
    cfg.max_new_tokens = max_new_tokens;
    cfg.temperature = temperature;
    cfg.top_k = top_k;
    cfg.top_p = top_p;
    cfg.min_p = min_p;
    cfg.seed = seed;
    cfg.collect_timing = true;
    cfg.use_chat_template = use_chat_template;
    cfg.enable_thinking = enable_thinking;
    cfg.stop_on_boxed_answer = stop_on_answer;
    cfg.stop_check_interval = stop_check_interval;

    std::ofstream output(output_path);
    if (!output)
        throw std::runtime_error("Failed to open output file: " + output_path);

    for (const auto& sample : samples) {
        auto wall_start = std::chrono::steady_clock::now();
        trtf::TextResult result = pipeline->generate(sample.prompt, cfg);
        auto wall_end = std::chrono::steady_clock::now();
        const double wall_ms =
            std::chrono::duration<double, std::milli>(wall_end - wall_start).count();
        const std::size_t generated_tokens = result.token_ids.size();
        const double tok_per_sec =
            (result.decode_ms > 0.0 && generated_tokens > 0)
                ? (static_cast<double>(generated_tokens) / (result.decode_ms / 1000.0))
                : 0.0;

        output << "{\"sample_id\":\"" << json_escape(sample.sample_id) << "\""
               << ",\"answer\":\"" << json_escape(sample.answer) << "\""
               << ",\"generated_tokens\":" << generated_tokens
               << ",\"prefill_ms\":" << std::fixed << std::setprecision(6) << result.prefill_ms
               << ",\"decode_ms\":" << std::fixed << std::setprecision(6) << result.decode_ms
               << ",\"wall_ms\":" << std::fixed << std::setprecision(6) << wall_ms
               << ",\"tokens_per_sec\":" << std::fixed << std::setprecision(6) << tok_per_sec
               << ",\"text\":\"" << json_escape(result.text) << "\"}\n";
        output.flush();

        std::cerr << "[trtf.dataset_benchmark] sample=" << sample.sample_id
                  << " generated_tokens=" << generated_tokens
                  << " decode_ms=" << result.decode_ms
                  << " tok/s=" << tok_per_sec << '\n';
    }

    return 0;
}
