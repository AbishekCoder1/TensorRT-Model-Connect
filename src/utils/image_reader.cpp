// image_reader.cpp — implements trtmc::io::read_image() using stb_image.

#include "trtmc/trtmc_io.hpp"

#include "stb_image.h"

#include <cstddef>
#include <stdexcept>

namespace trtmc::io {

LoadedImage read_image(const std::string& path)
{
    LoadedImage result;

    int width = 0;
    int height = 0;
    int channels = 0;
    unsigned char* raw = stbi_load(path.c_str(), &width, &height, &channels, 3);
    if (raw == nullptr)
    {
        return result;  // empty = decode failure
    }

    result.width = static_cast<int32_t>(width);
    result.height = static_cast<int32_t>(height);

    // Convert uint8 [0, 255] to float [0, 1] in HWC layout
    auto npixels = static_cast<std::size_t>(width) * height * 3;
    result.pixels.resize(npixels);
    for (std::size_t i = 0; i < npixels; ++i)
    {
        result.pixels[i] = static_cast<float>(raw[i]) / 255.0F;
    }

    stbi_image_free(raw);
    return result;
}

} // namespace trtmc::io
