#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
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

bool is_safetensors_index_file(const std::filesystem::path& path);

class TensorSource {
public:
    explicit TensorSource(const std::filesystem::path& path);

    bool has(const std::string& name) const;
    const SafetensorEntry& entry(const std::string& name) const;
    std::vector<float> load_f32(const std::string& name) const;

private:
    const SafetensorReader& shard_reader_for(const std::string& name) const;
    void init_from_index(const std::filesystem::path& index_path);

    std::filesystem::path mSinglePath;
    std::unique_ptr<SafetensorReader> mSingleReader;
    std::vector<std::unique_ptr<SafetensorReader>> mShardReaders;
    std::vector<std::filesystem::path> mShardPaths;
    std::unordered_map<std::string, std::size_t> mTensorToShard;
};

} // namespace trtf
