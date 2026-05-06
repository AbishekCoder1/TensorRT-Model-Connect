#pragma once

#include <cstddef>
#include <cstdint>

namespace trtmc::detail {

struct CacheRowUpdatePlan {
    bool shift_existing_rows{false};
    std::size_t append_offset_bytes{0};
    std::size_t shift_source_offset_bytes{0};
    std::size_t shift_copy_bytes{0};
    std::size_t tail_offset_bytes{0};
    int32_t next_cache_length{0};
};

CacheRowUpdatePlan plan_cache_row_update(
    int32_t cache_length,
    int32_t max_cache_length,
    std::size_t row_bytes);

} // namespace trtmc::detail
