#pragma once

#include "bundle/bundle_format.h"
#include "trtf/runtime/ports/bundle_port.h"

#include <string_view>
#include <vector>

namespace trtf::runtime {

class BundlePortAdapter final : public IBundlePort {
public:
    explicit BundlePortAdapter(const BundleFile& bundle);

    bool has_section(std::string_view section_name) const override;
    BundlePortResult<std::vector<char>> fetch_section_bytes(std::string_view section_name) const override;
    BundlePortResult<FastPathModelConfig> parse_fast_path_config(int32_t max_cache_length_override) const override;

private:
    const std::vector<char>* find_section(std::string_view section_name) const;

    const BundleFile& mBundle;
};

} // namespace trtf::runtime
