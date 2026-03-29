// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-VL-CPP-02
// Architecture:   ARCH-FAC-001
// Unit Design:    UD-VL-01
// Intent:         Vision execution plan: feature count, DeepStack name collection, copy plan
// Preconditions:  Vision config with known feature levels
// Postconditions: Feature counts match config, DeepStack names collected, copy plan succeeds/fails correctly
// =============================================================================

#include "runtime/domains/multimodal/vision_execution_plan.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void check(bool condition, const char* name)
{
    if (!condition)
    {
        std::cerr << "FAIL: " << name << '\n';
        ++g_failures;
    }
}

void test_feature_count_and_deepstack_name_collection()
{
    check(trtf::vision_output_feature_count(4, 8) == 32,
        "vision execution plan computes output feature count");

    const auto names = trtf::collect_vision_deepstack_output_names(
        [](const std::string& name)
        {
            return name == "deepstack_features_0" || name == "deepstack_features_1";
        });
    check(names == std::vector<std::string>({"deepstack_features_0", "deepstack_features_1"}),
        "vision execution plan collects sequential deepstack outputs");
}

void test_run_vision_copy_plan_success_and_failures()
{
    std::vector<trtf::VisionPendingCopy> copies = {
        {reinterpret_cast<void*>(1), reinterpret_cast<void*>(2), 16},
        {reinterpret_cast<void*>(3), reinterpret_cast<void*>(4), 32},
    };
    std::string error;
    int enqueue_calls = 0;
    int copy_calls = 0;
    int sync_calls = 0;

    bool ok = trtf::run_vision_copy_plan(
        copies,
        error,
        [&enqueue_calls]()
        {
            ++enqueue_calls;
            return true;
        },
        [&copy_calls](const trtf::VisionPendingCopy& copy)
        {
            ++copy_calls;
            return copy.bytes > 0;
        },
        [&sync_calls]()
        {
            ++sync_calls;
            return true;
        });
    check(ok, "vision execution plan succeeds when enqueue/copy/sync succeed");
    check(error.empty(), "vision execution plan leaves error empty on success");
    check(enqueue_calls == 1, "vision execution plan enqueues once");
    check(copy_calls == 2, "vision execution plan copies each output");
    check(sync_calls == 1, "vision execution plan synchronizes once");

    error.clear();
    ok = trtf::run_vision_copy_plan(
        copies,
        error,
        []()
        {
            return false;
        },
        [](const trtf::VisionPendingCopy&)
        {
            return true;
        },
        []()
        {
            return true;
        });
    check(!ok, "vision execution plan reports enqueue failure");
    check(error == "vision enqueueV3 failed",
        "vision execution plan uses enqueue failure message");

    error.clear();
    ok = trtf::run_vision_copy_plan(
        copies,
        error,
        []()
        {
            return true;
        },
        [](const trtf::VisionPendingCopy& copy)
        {
            return copy.bytes == 16;
        },
        []()
        {
            return true;
        });
    check(!ok, "vision execution plan reports output copy failure");
    check(error == "vision cudaMemcpyAsync output failed",
        "vision execution plan uses copy failure message");

    error.clear();
    ok = trtf::run_vision_copy_plan(
        copies,
        error,
        []()
        {
            return true;
        },
        [](const trtf::VisionPendingCopy&)
        {
            return true;
        },
        []()
        {
            return false;
        });
    check(!ok, "vision execution plan reports sync failure");
    check(error == "vision cudaStreamSynchronize failed",
        "vision execution plan uses sync failure message");
}

} // namespace

int main()
{
    test_feature_count_and_deepstack_name_collection();
    test_run_vision_copy_plan_success_and_failures();

    if (g_failures != 0)
    {
        std::cerr << g_failures << " vision execution plan test(s) failed\n";
        return 1;
    }
    return 0;
}
