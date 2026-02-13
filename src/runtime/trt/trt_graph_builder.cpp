#include "runtime/trt/trt_graph_builder.h"

#include <mutex>
#include <unordered_map>

namespace trtf {

#if TRTF_HAS_TRT

namespace {

std::mutex& builders_mutex()
{
    static std::mutex m;
    return m;
}

std::unordered_map<std::string, std::unique_ptr<ITrtGraphBuilder>>& builders()
{
    static std::unordered_map<std::string, std::unique_ptr<ITrtGraphBuilder>> registry;
    return registry;
}

} // namespace

void RegisterTrtGraphBuilder(const std::string& family, std::unique_ptr<ITrtGraphBuilder> builder)
{
    std::lock_guard<std::mutex> lock(builders_mutex());
    builders()[family] = std::move(builder);
}

ITrtGraphBuilder* FindTrtGraphBuilder(const std::string& family)
{
    std::lock_guard<std::mutex> lock(builders_mutex());
    const auto it = builders().find(family);
    if (it == builders().end())
    {
        return nullptr;
    }
    return it->second.get();
}

#endif // TRTF_HAS_TRT

} // namespace trtf
