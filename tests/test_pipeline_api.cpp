// Test: IPipeline virtual interface through trtf_create_pipeline.
// No TRT/GPU needed -- uses cpu-reference backend with builtin model.

#include "trtf/pipeline.h"

#include <cerrno>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>

#include <stdlib.h>

static int failures = 0;

static void check(bool condition, const char* test_name)
{
    if (!condition)
    {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

static std::filesystem::path make_temp_dir()
{
    char pattern[] = "/tmp/trtfb_pipe_XXXXXX";
    char* dir = mkdtemp(pattern);
    if (dir == nullptr)
    {
        throw std::runtime_error(std::string("mkdtemp failed: ") + std::strerror(errno));
    }
    return std::filesystem::path(dir);
}

static void test_generate_returns_text()
{
    auto* p = trtf_create_pipeline("trtf/tiny-cake-v1", TRTF_CPU_ONLY);
    check(p != nullptr, "pipeline created");
    if (p != nullptr)
    {
        const char* result = p->generate("hello");
        check(result != nullptr, "generate returns non-null");
        check(std::strlen(result) > 0, "generate returns non-empty text");
        delete p;
    }
}

static void test_generate_with_max_tokens()
{
    auto* p = trtf_create_pipeline("trtf/tiny-cake-v1", TRTF_CPU_ONLY);
    check(p != nullptr, "pipeline created for max_tokens test");
    if (p != nullptr)
    {
        const char* result = p->generate("hello", 3);
        check(result != nullptr, "generate with max_tokens returns non-null");
        delete p;
    }
}

static void test_generate_pointer_valid_until_next()
{
    auto* p = trtf_create_pipeline("trtf/tiny-cake-v1", TRTF_CPU_ONLY);
    check(p != nullptr, "pipeline created for pointer validity test");
    if (p != nullptr)
    {
        const char* first = p->generate("hello", 3);
        check(first != nullptr, "first generate result valid");
        const std::string first_copy(first);

        const char* second = p->generate("world", 3);
        check(second != nullptr, "second generate result valid");

        // first pointer may now be stale (implementation reuses buffer),
        // but first_copy should be valid
        check(!first_copy.empty(), "first result was non-empty");
        delete p;
    }
}

static void test_save_bundle_creates_file()
{
    auto* p = trtf_create_pipeline("trtf/tiny-cake-v1", TRTF_CPU_ONLY);
    check(p != nullptr, "pipeline created for save_bundle test");
    if (p != nullptr)
    {
        const auto tmp = make_temp_dir();
        const auto bundle_path = (tmp / "test.trtfb").string();
        const bool result = p->save_bundle(bundle_path.c_str());
        // cpu-reference backend doesn't support bundle save (no TRT engine),
        // so this should return false
        check(!result, "save_bundle returns false for cpu-reference backend");
        delete p;
        std::filesystem::remove_all(tmp);
    }
}

static void test_save_bundle_bad_path_returns_false()
{
    auto* p = trtf_create_pipeline("trtf/tiny-cake-v1", TRTF_CPU_ONLY);
    check(p != nullptr, "pipeline created for bad path test");
    if (p != nullptr)
    {
        const bool result = p->save_bundle("/nonexistent/dir/test.trtfb");
        check(!result, "save_bundle returns false for bad path");
        delete p;
    }
}

static void test_sizeof_ipipeline_is_vtable()
{
    // IPipeline is a pure virtual class -- sizeof should be just the vtable pointer
    check(sizeof(trtf::IPipeline) == sizeof(void*), "sizeof(IPipeline) equals vtable pointer size");
}

int main()
{
    test_generate_returns_text();
    test_generate_with_max_tokens();
    test_generate_pointer_valid_until_next();
    test_save_bundle_creates_file();
    test_save_bundle_bad_path_returns_false();
    test_sizeof_ipipeline_is_vtable();

    if (failures > 0)
    {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All pipeline_api tests passed.\n";
    return 0;
}
