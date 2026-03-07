#pragma once

#include "cabi/config/fast_path_config.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace trtf::runtime {

enum class BundlePortStatus {
    kOk = 0,
    kMissingSection,
    kInvalidSection,
};

template <typename T>
struct BundlePortResult {
    BundlePortStatus status{BundlePortStatus::kOk};
    std::string message;
    T value{};

    bool ok() const noexcept
    {
        return status == BundlePortStatus::kOk;
    }

    explicit operator bool() const noexcept
    {
        return ok();
    }

    static BundlePortResult success(T result_value)
    {
        BundlePortResult result;
        result.value = std::move(result_value);
        return result;
    }

    static BundlePortResult missing(std::string message_text)
    {
        BundlePortResult result;
        result.status = BundlePortStatus::kMissingSection;
        result.message = std::move(message_text);
        return result;
    }

    static BundlePortResult invalid(std::string message_text)
    {
        BundlePortResult result;
        result.status = BundlePortStatus::kInvalidSection;
        result.message = std::move(message_text);
        return result;
    }
};

class IBundlePort {
public:
    virtual ~IBundlePort() = default;

    virtual bool has_section(std::string_view section_name) const = 0;
    virtual BundlePortResult<std::vector<char>> fetch_section_bytes(std::string_view section_name) const = 0;
    virtual BundlePortResult<FastPathModelConfig> parse_fast_path_config(int32_t max_cache_length_override) const = 0;
};

} // namespace trtf::runtime
