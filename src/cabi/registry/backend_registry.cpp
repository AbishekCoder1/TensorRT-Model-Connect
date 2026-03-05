#include "cabi/registry/backend_registry.h"

#include <mutex>
#include <unordered_map>

namespace trtf {
namespace cabi {
namespace {

class BackendRegistry final {
public:
    bool register_factory(const std::string& strategy, BackendFactoryFn factory)
    {
        if (strategy.empty() || factory == nullptr)
        {
            return false;
        }

        std::lock_guard<std::mutex> lock(mMutex);
        return mFactories.emplace(strategy, factory).second;
    }

    bool unregister_factory(const std::string& strategy)
    {
        std::lock_guard<std::mutex> lock(mMutex);
        return mFactories.erase(strategy) > 0;
    }

    BackendFactoryFn find_factory(const std::string& strategy)
    {
        std::lock_guard<std::mutex> lock(mMutex);
        const auto it = mFactories.find(strategy);
        return it != mFactories.end() ? it->second : nullptr;
    }

    std::size_t size() const
    {
        std::lock_guard<std::mutex> lock(mMutex);
        return mFactories.size();
    }

private:
    mutable std::mutex mMutex;
    std::unordered_map<std::string, BackendFactoryFn> mFactories;
};

BackendRegistry& registry()
{
    static BackendRegistry instance;
    return instance;
}

} // namespace

bool register_backend_factory(const std::string& strategy, BackendFactoryFn factory)
{
    return registry().register_factory(strategy, factory);
}

bool unregister_backend_factory(const std::string& strategy)
{
    return registry().unregister_factory(strategy);
}

BackendFactoryFn find_backend_factory(const std::string& strategy)
{
    return registry().find_factory(strategy);
}

trtf::IPipeline* try_create_pipeline_from_registry(const std::string& strategy, void* context)
{
    BackendFactoryFn factory = registry().find_factory(strategy);
    if (factory == nullptr)
    {
        return nullptr;
    }
    return factory(context);
}

std::size_t backend_factory_count()
{
    return registry().size();
}

} // namespace cabi
} // namespace trtf
