#pragma once

#include "trtf/pipeline.h"

#include <cstddef>

namespace trtf {
namespace cabi {
namespace detail {

#if TRTF_HAS_TRT

void set_default_max_new_tokens(
    trtf::IPipeline* pipeline,
    std::size_t max_new_tokens);

#endif // TRTF_HAS_TRT

} // namespace detail
} // namespace cabi
} // namespace trtf
