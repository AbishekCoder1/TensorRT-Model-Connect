// =============================================================================
// Test suite: Diffusion pipeline headers compile and types are correct.
// These pipelines now delegate to old-style IDiffusionBackend which requires
// real TRT engines, so construction tests are limited to compile checks.
// Full integration is validated by E2E tests.
// =============================================================================

#include "runtime/pipelines/diffusion_pipeline.h"

#include <cstdint>
#include <iostream>
#include <string>

static int failures = 0;
static void check(bool c, const char* n) { if (!c) { std::cerr << "FAIL: " << n << '\n'; ++failures; } }

#if TRTF_HAS_TRT

static void test_pipeline_types_compile()
{
    check(sizeof(trtf::FluxPipeline) > 0, "FluxPipeline defined");
    check(sizeof(trtf::WanPipeline) > 0, "WanPipeline defined");
    check(sizeof(trtf::ZImagePipeline) > 0, "ZImagePipeline defined");
}

#endif

int main()
{
#if TRTF_HAS_TRT
    test_pipeline_types_compile();
#else
    std::cerr << "TRT not available, skipping\n";
#endif
    if (failures > 0) std::cerr << failures << " FAILED\n";
    return failures;
}
