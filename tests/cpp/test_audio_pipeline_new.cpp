// =============================================================================
// Test suite: Audio pipeline headers compile and types are correct.
// These pipelines now delegate to old-style backends which require real TRT
// engines, so construction tests are limited to compile checks and type
// assertions.  Full integration is validated by E2E tests.
// =============================================================================

#include "runtime/pipelines/audio_pipeline.h"

#include <cstdint>
#include <iostream>
#include <string>

static int failures = 0;
static void check(bool c, const char* n) { if (!c) { std::cerr << "FAIL: " << n << '\n'; ++failures; } }

#if TRTF_HAS_TRT

static void test_pipeline_types_compile()
{
    // Verify the pipeline classes exist and have the expected interface
    // (compile-time check — no construction since backends need real TRT)
    check(sizeof(trtf::WhisperPipeline) > 0, "WhisperPipeline defined");
    check(sizeof(trtf::BarkPipeline) > 0, "BarkPipeline defined");
    check(sizeof(trtf::MagpiePipeline) > 0, "MagpiePipeline defined");
    check(sizeof(trtf::SpeechPipeline) > 0, "SpeechPipeline defined");
    check(sizeof(trtf::OmniPipeline) > 0, "OmniPipeline defined");
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
