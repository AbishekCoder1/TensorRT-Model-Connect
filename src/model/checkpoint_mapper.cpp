#include "model/checkpoint_mapper.h"

#include <algorithm>
#include <mutex>
#include <vector>

namespace trtf {
namespace {

struct CheckpointMapperEntry {
    std::string family;
    int priority{0};
    std::unique_ptr<ICheckpointMapper> mapper;
};

std::mutex& mappers_mutex()
{
    static std::mutex m;
    return m;
}

std::vector<CheckpointMapperEntry>& mappers()
{
    static std::vector<CheckpointMapperEntry> registry;
    return registry;
}

} // namespace

void RegisterCheckpointMapper(const std::string& family, int priority,
                              std::unique_ptr<ICheckpointMapper> mapper)
{
    std::lock_guard<std::mutex> lock(mappers_mutex());
    mappers().push_back({family, priority, std::move(mapper)});
    std::stable_sort(mappers().begin(), mappers().end(),
        [](const CheckpointMapperEntry& a, const CheckpointMapperEntry& b) {
            return a.priority > b.priority;
        });
}

ICheckpointMapper* FindCheckpointMapper(const DecoderArchitectureConfig& architecture)
{
    std::lock_guard<std::mutex> lock(mappers_mutex());
    for (auto& entry : mappers())
    {
        if (entry.mapper && entry.mapper->can_map(architecture))
        {
            return entry.mapper.get();
        }
    }
    return nullptr;
}

} // namespace trtf
