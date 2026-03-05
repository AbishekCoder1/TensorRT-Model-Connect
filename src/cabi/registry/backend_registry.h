#pragma once

#include "trtf/pipeline.h"

#include <cstddef>
#include <string>

namespace trtf {
namespace cabi {

// Factory function for creating a pipeline from an opaque context.
using BackendFactoryFn = trtf::IPipeline* (*)(void* context);

// Registers a factory for runtime_strategy.
// Returns false if strategy is empty, factory is null, or strategy already exists.
bool register_backend_factory(const std::string& strategy, BackendFactoryFn factory);

// Unregisters a factory by runtime_strategy. Returns true if a factory was removed.
bool unregister_backend_factory(const std::string& strategy);

// Looks up a factory by runtime_strategy. Returns nullptr if not found.
BackendFactoryFn find_backend_factory(const std::string& strategy);

// Calls a registered factory for runtime_strategy with the provided context.
// Returns nullptr when no factory is registered for strategy.
trtf::IPipeline* try_create_pipeline_from_registry(const std::string& strategy, void* context);

// Returns the number of registered factories.
std::size_t backend_factory_count();

} // namespace cabi
} // namespace trtf
