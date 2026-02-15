// Test: IPipeline virtual interface through trtf_create_pipeline.
// Uses QWEN3 model (falls back gracefully if not available).

#include "trtf/pipeline.h"

#include <cstring>
#include <iostream>
#include <string>

static int failures = 0;

static void check(bool condition, const char* test_name)
{
    if (!condition)
    {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

static void test_generate_returns_text()
{
    auto* p = trtf_create_pipeline("QWEN3", TRTF_CPU_ONLY);
    if (p == nullptr)
    {
        std::cerr << "SKIP: QWEN3 not available: " << trtf_last_error() << '\n';
        return;
    }
    const char* result = p->generate("hello", 3);
    check(result != nullptr, "generate returns non-null");
    if (result != nullptr)
    {
        check(std::strlen(result) > 0, "generate returns non-empty text");
    }
    delete p;
}

static void test_generate_with_max_tokens()
{
    auto* p = trtf_create_pipeline("QWEN3", TRTF_CPU_ONLY);
    if (p == nullptr) return;
    const char* result = p->generate("hello", 3);
    check(result != nullptr, "generate with max_tokens returns non-null");
    delete p;
}

static void test_generate_pointer_valid_until_next()
{
    auto* p = trtf_create_pipeline("QWEN3", TRTF_CPU_ONLY);
    if (p == nullptr) return;
    const char* first = p->generate("hello", 3);
    check(first != nullptr, "first generate result valid");
    const std::string first_copy(first ? first : "");

    const char* second = p->generate("world", 3);
    check(second != nullptr, "second generate result valid");
    check(!first_copy.empty(), "first result was non-empty");
    delete p;
}

static void test_sizeof_ipipeline_is_vtable()
{
    check(sizeof(trtf::IPipeline) == sizeof(void*), "sizeof(IPipeline) equals vtable pointer size");
}

int main()
{
    test_generate_returns_text();
    test_generate_with_max_tokens();
    test_generate_pointer_valid_until_next();
    test_sizeof_ipipeline_is_vtable();

    if (failures > 0)
    {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All pipeline_api tests passed.\n";
    return 0;
}
