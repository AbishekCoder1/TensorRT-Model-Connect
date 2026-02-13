#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace trtf {

struct SafetensorEntry {
    std::string dtype;
    std::vector<int64_t> shape;
    uint64_t data_begin{0};
    uint64_t data_end{0};
};

class SafetensorReader {
public:
    explicit SafetensorReader(std::filesystem::path path);

    bool has(const std::string& name) const;
    const SafetensorEntry& entry(const std::string& name) const;
    std::vector<float> load_f32(const std::string& name) const;

private:
    std::filesystem::path mPath;
    uint64_t mDataBaseOffset{0};
    std::unordered_map<std::string, SafetensorEntry> mEntries;
};

} // namespace trtf
