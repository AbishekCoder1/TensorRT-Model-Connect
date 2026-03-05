#include "cabi/registry/backend_registry.h"

#include <atomic>
#include <cstddef>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

static int failures = 0;

static void check(bool condition, const char* test_name)
{
    if (!condition)
    {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

namespace {

class DummyPipeline final : public trtf::IPipeline {
public:
    const char* generate(const char* /* prompt */, std::size_t /* max_new_tokens */) override
    {
        return "";
    }

    const char* model_id() const override
    {
        return "dummy-model";
    }

    const char* backend_name() const override
    {
        return "dummy-backend";
    }
};

struct DummyContext {
    int value{0};
};

trtf::IPipeline* context_increment_factory(void* opaque)
{
    auto* ctx = static_cast<DummyContext*>(opaque);
    if (ctx != nullptr)
    {
        ctx->value += 1;
    }

    static DummyPipeline pipeline;
    return &pipeline;
}

trtf::IPipeline* null_factory(void* /* opaque */)
{
    return nullptr;
}

} // namespace

static void test_registry_register_find_and_dispatch()
{
    const std::string key = "unit_test_registry_dispatch";
    check(trtf::cabi::register_backend_factory(key, &context_increment_factory),
          "register backend factory");

    const auto found = trtf::cabi::find_backend_factory(key);
    check(found == &context_increment_factory, "find backend factory");

    DummyContext ctx{41};
    auto* pipeline = trtf::cabi::try_create_pipeline_from_registry(key, &ctx);
    check(pipeline != nullptr, "dispatch returns non-null pipeline");
    check(ctx.value == 42, "dispatch passes opaque context");

    check(trtf::cabi::unregister_backend_factory(key), "unregister backend factory");
}

static void test_registry_rejects_duplicate_registration()
{
    const std::string key = "unit_test_registry_duplicate";
    check(trtf::cabi::register_backend_factory(key, &null_factory),
          "initial registration succeeds");
    check(!trtf::cabi::register_backend_factory(key, &context_increment_factory),
          "duplicate registration rejected");
    check(trtf::cabi::unregister_backend_factory(key),
          "cleanup duplicate test registration");
}

static void test_registry_invalid_inputs_and_miss()
{
    check(!trtf::cabi::register_backend_factory("", &null_factory),
          "empty strategy rejected");
    check(!trtf::cabi::register_backend_factory("unit_test_null_factory", nullptr),
          "null factory rejected");
    check(trtf::cabi::find_backend_factory("unit_test_missing") == nullptr,
          "find missing factory returns nullptr");
    check(trtf::cabi::try_create_pipeline_from_registry("unit_test_missing", nullptr) == nullptr,
          "dispatch missing strategy returns nullptr");
}

static void test_registry_concurrent_registration()
{
    const std::size_t base_count = trtf::cabi::backend_factory_count();
    constexpr int kThreads = 8;

    std::atomic<int> success_count{0};
    std::vector<std::thread> threads;
    std::vector<std::string> keys;
    keys.reserve(kThreads);
    for (int i = 0; i < kThreads; ++i)
    {
        keys.push_back("unit_test_registry_thread_" + std::to_string(i));
    }

    threads.reserve(kThreads);
    for (int i = 0; i < kThreads; ++i)
    {
        threads.emplace_back([&, i] {
            if (trtf::cabi::register_backend_factory(keys[i], &null_factory))
            {
                success_count.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    for (auto& thread : threads)
    {
        thread.join();
    }

    check(success_count.load(std::memory_order_relaxed) == kThreads,
          "all concurrent registrations succeed");
    check(trtf::cabi::backend_factory_count() == base_count + static_cast<std::size_t>(kThreads),
          "registry count updated after concurrent registrations");

    for (const auto& key : keys)
    {
        check(trtf::cabi::unregister_backend_factory(key),
              "cleanup concurrent test registration");
    }
    check(trtf::cabi::backend_factory_count() == base_count,
          "registry count restored after cleanup");
}

int main()
{
    test_registry_register_find_and_dispatch();
    test_registry_rejects_duplicate_registration();
    test_registry_invalid_inputs_and_miss();
    test_registry_concurrent_registration();

    if (failures > 0)
    {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }

    std::cerr << "All backend_registry tests passed.\n";
    return 0;
}
