#pragma once

// Internal definition of DecodedImage.
// This header replaces the deleted public
// include/trtmc/runtime/adapters/io/media_io_adapter.h.
// It is used only inside src/runtime/domains/ for image preprocessing,
// perception, and embedding backends.

#include <cstdint>
#include <cstddef>
#include <vector>

namespace trtmc {
namespace runtime {
namespace adapters {
namespace io {

/// Raw decoded image: pixel buffer in HWC layout, uint8 per channel.
struct DecodedImage {
    std::vector<uint8_t> pixels;  // [height * width * channels]
    int32_t width{0};
    int32_t height{0};
    int32_t channels{0};

    [[nodiscard]] bool empty() const
    {
        return pixels.empty() || width <= 0 || height <= 0 || channels <= 0;
    }
};

} // namespace io
} // namespace adapters
} // namespace runtime
} // namespace trtmc
