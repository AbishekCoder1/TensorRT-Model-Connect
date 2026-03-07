#pragma once

#include "trtf/runtime/contracts/services.h"

#include <string>
#include <utility>
#include <vector>

namespace trtf::runtime {

struct BuildContext {
    std::string model_id;
    std::string strategy;
    std::string hf_python;
    std::string bundle_path;

    // Optional non-owning references consumed by strategy builders.
    const void* config{nullptr};
    const void* sections{nullptr};

    // Reserved for builder-specific test hooks. Production call sites should
    // prefer the explicit metadata fields above over opaque handles.
    std::vector<void*> runtime_handles;
};

enum class BuildStatus {
    kUnspecified = 0,
    kOk,
    kInvalidArgument,
    kUnsupportedStrategy,
    kMissingDependency,
    kRuntimeError
};

struct BuildResult {
    BuildStatus status{BuildStatus::kUnspecified};
    std::string message;
    PipelineServices services;

    [[nodiscard]] bool ok() const
    {
        return status == BuildStatus::kOk;
    }

    static BuildResult Success(PipelineServices services_in)
    {
        BuildResult result;
        result.status = BuildStatus::kOk;
        result.services = std::move(services_in);
        return result;
    }

    static BuildResult Failure(BuildStatus status_in, std::string message_in)
    {
        BuildResult result;
        result.status = status_in;
        result.message = std::move(message_in);
        return result;
    }
};

} // namespace trtf::runtime
