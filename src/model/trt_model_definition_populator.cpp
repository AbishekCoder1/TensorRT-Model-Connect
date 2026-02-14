#include "model/trt_model_definition_populator.h"

#include <algorithm>
#include <mutex>
#include <vector>

namespace trtf {
namespace {

struct PopulatorEntry {
    std::string family;
    int priority{0};
    std::unique_ptr<ITrtModelDefinitionPopulator> populator;
};

std::mutex& populators_mutex()
{
    static std::mutex m;
    return m;
}

std::vector<PopulatorEntry>& populators()
{
    static std::vector<PopulatorEntry> registry;
    return registry;
}

} // namespace

void RegisterTrtModelDefinitionPopulator(const std::string& family, int priority,
    std::unique_ptr<ITrtModelDefinitionPopulator> populator)
{
    std::lock_guard<std::mutex> lock(populators_mutex());
    populators().push_back({family, priority, std::move(populator)});
    std::stable_sort(populators().begin(), populators().end(),
        [](const PopulatorEntry& a, const PopulatorEntry& b) {
            return a.priority > b.priority;
        });
}

bool PopulateViaRegistry(TrtDecoderDefinition& definition, const DecoderModel& model)
{
    std::lock_guard<std::mutex> lock(populators_mutex());
    for (auto& entry : populators())
    {
        if (entry.populator && entry.populator->can_populate(model))
        {
            return entry.populator->populate(definition, model);
        }
    }
    return false;
}

} // namespace trtf
