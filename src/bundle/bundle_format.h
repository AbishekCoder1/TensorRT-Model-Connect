#pragma once

// Internal bundle format: read/write .trtfb files.
// Format:
//   Bytes 0-7:    Magic "TRTFB\x00\x01\x00"
//   Bytes 8-15:   uint64_t json_header_length (LE)
//   Bytes 16..N:  JSON metadata header (UTF-8)
//   Bytes N..EOF: Binary sections referenced by offset in the header

#include "trtf/bundle.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace trtf {

// Magic bytes for .trtfb files.
static constexpr unsigned char kBundleMagic[8] = {'T', 'R', 'T', 'F', 'B', '\0', '\x01', '\0'};
static constexpr std::size_t kBundleHeaderOffset = 16; // 8 magic + 8 length

struct BundleSection {
    std::string name;
    std::vector<char> data;
};

struct BundleFile {
    BundleInfo info;
    std::vector<BundleSection> sections;
};

// Write a bundle to disk.
void WriteBundleFile(const std::string& path, const BundleFile& bundle);

// Read a complete bundle from disk.
BundleFile ReadBundleFile(const std::string& path);

// Read just the header metadata (no section data loaded).
BundleInfo ReadBundleHeader(const std::string& path);

// Check magic bytes without reading full file.
bool HasBundleMagic(const std::string& path);

// JSON serialization for BundleInfo.
std::string BundleInfoToJson(const BundleInfo& info,
    const std::vector<std::pair<std::string, std::size_t>>& section_offsets_and_sizes);
BundleInfo BundleInfoFromJson(const std::string& json,
    std::vector<std::pair<std::string, std::pair<std::size_t, std::size_t>>>& sections_out);

} // namespace trtf
