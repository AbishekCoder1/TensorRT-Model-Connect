#pragma once

#include "trtf/runtime/contracts/build.h"

namespace trtf::runtime {

class IStrategyBuilder {
public:
    virtual ~IStrategyBuilder() = default;

    virtual BuildResult build(const BuildContext& context) = 0;
};

} // namespace trtf::runtime
