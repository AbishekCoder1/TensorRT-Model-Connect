// =============================================================================
// Test suite: Pipeline C ABI -- IPipeline virtual interface via trtf_create_pipeline
// =============================================================================
//
// Purpose:
//   Validates the public C ABI entry point trtf_create_pipeline() and the
//   IPipeline virtual interface it returns. Tests cover null/invalid input
//   handling, version queries, and ABI stability guarantees. Since the
//   runtime is now bundle-only (requires pre-built .trtfb files), tests
//   focus on error paths and compile-time interface properties rather than
//   successful generation.
//
// Dependencies:
//   - trtf/pipeline.h (IPipeline, trtf_create_pipeline, trtf_last_error,
//     trtf_version, trtf_has_trt)
//   - No TRT, GPU, or model files required.
//
// Approach:
//   Each test function exercises the C ABI entry points with invalid inputs
//   and checks return values. A simple check() helper tracks failure count.
//   Tests are designed to be order-independent but run sequentially in main().
// =============================================================================

#include "trtf/pipeline.h"

#include <cstring>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

static int failures = 0;

// -----------------------------------------------------------------------------
// Helper: Assert a boolean condition and report failure with a descriptive name.
// Increments the global failure counter on false. Does not abort -- all tests
// run to completion so the full failure picture is visible.
// -----------------------------------------------------------------------------
static void check(bool condition, const char* test_name)
{
    if (!condition)
    {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

class DummyPipeline final : public trtf::IPipeline {
public:
    using trtf::IPipeline::generate;

    const char* generate(const char* prompt, std::size_t max_new_tokens = 0) override
    {
        last_prompt = (prompt != nullptr) ? prompt : "";
        last_max_new_tokens = max_new_tokens;
        return "dummy-generate";
    }

    const char* model_id() const override
    {
        return "dummy-model";
    }

    const char* backend_name() const override
    {
        return "dummy-backend";
    }

    std::string last_prompt;
    std::size_t last_max_new_tokens{0};
};

// -----------------------------------------------------------------------------
// Test: null input returns nullptr with error message
//
// Intention:
//   Verify that passing nullptr as the bundle path to trtf_create_pipeline
//   returns nullptr and sets a descriptive error via trtf_last_error().
//
// Setup: None.
//
// Mechanism:
//   Calls trtf_create_pipeline(nullptr, 0) and checks the return and error.
// -----------------------------------------------------------------------------
static void test_null_input_returns_null()
{
    auto* p = trtf_create_pipeline(nullptr, 0);
    check(p == nullptr, "null input returns nullptr");
    const char* err = trtf_last_error();
    check(err != nullptr && std::strlen(err) > 0, "error set after null input");
}

// -----------------------------------------------------------------------------
// Test: invalid path returns nullptr with error message
//
// Intention:
//   Verify that a nonexistent bundle path is rejected with a clear error.
//
// Setup: None.
//
// Mechanism:
//   Calls trtf_create_pipeline with a path that does not exist, checks nullptr
//   return and non-empty error message.
// -----------------------------------------------------------------------------
static void test_invalid_path_returns_null()
{
    auto* p = trtf_create_pipeline("/nonexistent/path/to/bundle.trtfb", 0);
    check(p == nullptr, "invalid path returns nullptr");
    const char* err = trtf_last_error();
    check(err != nullptr && std::strlen(err) > 0, "error set after invalid path");
}

// -----------------------------------------------------------------------------
// Test: version string is available and non-empty
//
// Intention:
//   Verify that trtf_version() returns a usable version string.
//
// Setup: None.
//
// Mechanism:
//   Calls trtf_version(), checks non-null and non-empty.
// -----------------------------------------------------------------------------
static void test_version_available()
{
    const char* ver = trtf_version();
    check(ver != nullptr, "version is non-null");
    check(std::strlen(ver) > 0, "version is non-empty");
}

// -----------------------------------------------------------------------------
// Test: trtf_has_trt returns a valid boolean
//
// Intention:
//   Verify that the TRT detection function returns 0 or 1.
//
// Setup: None.
//
// Mechanism:
//   Calls trtf_has_trt(), checks the value is 0 or 1.
// -----------------------------------------------------------------------------
static void test_has_trt_returns_bool()
{
    const int val = trtf_has_trt();
    check(val == 0 || val == 1, "trtf_has_trt returns 0 or 1");
}

// -----------------------------------------------------------------------------
// Test: sizeof(IPipeline) equals a single vtable pointer
//
// Intention:
//   Verify the ABI stability guarantee that IPipeline is a pure abstract
//   interface with no data members -- its size must equal exactly one pointer
//   (the vtable pointer). This ensures the C ABI boundary is safe: callers
//   compiled with different compilers/settings can use IPipeline* as long
//   as the vtable layout is stable.
//
// Setup:
//   None -- this is a compile-time/static property check.
//
// Mechanism:
//   Compares sizeof(trtf::IPipeline) against sizeof(void*). If they differ,
//   it means data members or multiple inheritance have been added, breaking
//   the ABI contract.
// -----------------------------------------------------------------------------
static void test_sizeof_ipipeline_is_vtable()
{
    check(sizeof(trtf::IPipeline) == sizeof(void*), "sizeof(IPipeline) equals vtable pointer size");
}

// -----------------------------------------------------------------------------
// Test: delete null IPipeline is safe
//
// Intention:
//   Verify that deleting a null IPipeline pointer does not crash.
//
// Setup: A null IPipeline pointer.
//
// Mechanism:
//   Calls delete on the null pointer, asserts true if we reach this point.
// -----------------------------------------------------------------------------
static void test_delete_null_safe()
{
    trtf::IPipeline* p = nullptr;
    delete p;
    check(true, "delete null IPipeline is safe");
}

// -----------------------------------------------------------------------------
// Test: IPipeline default virtual methods return documented fallback values
//
// Intention:
//   Exercise inline default virtual implementations in include/trtf/pipeline.h
//   so ABI callers can rely on stable fallback behavior when a backend does not
//   implement optional capabilities.
//
// Setup:
//   A minimal concrete DummyPipeline that only overrides required pure virtuals.
//
// Mechanism:
//   Calls each default method and checks return values (false/-1/null/0), plus
//   validates that generate(prompt,image,max) forwards to generate(prompt,max).
// -----------------------------------------------------------------------------
static void test_ipipeline_default_virtuals()
{
    DummyPipeline pipeline;

    const char* generated = pipeline.generate("hello", "image.png", 7);
    check(generated != nullptr && std::string(generated) == "dummy-generate",
        "generate(prompt,image,max) forwards to text generate");
    check(pipeline.last_prompt == "hello", "generate forwarding preserves prompt");
    check(pipeline.last_max_new_tokens == 7, "generate forwarding preserves max_new_tokens");

    check(!pipeline.supports_vision(), "default supports_vision is false");
    check(!pipeline.supports_video(), "default supports_video is false");
    check(pipeline.generate_video("p", "/tmp/out") == -1, "default generate_video returns -1");

    check(!pipeline.supports_segmentation(), "default supports_segmentation is false");
    check(pipeline.segment("img.png", "mask.png") == -1, "default segment returns -1");

    check(!pipeline.supports_encoding(), "default supports_encoding is false");
    int32_t seq_len = -1;
    int32_t hidden = -1;
    check(pipeline.encode("text", &seq_len, &hidden) == nullptr, "default encode returns nullptr");

    check(!pipeline.supports_solve(), "default supports_solve is false");
    int32_t out_dim = -1;
    check(pipeline.solve(nullptr, 0, nullptr, 0, &out_dim) == nullptr, "default solve returns nullptr");
    int32_t out_c = -1;
    int32_t out_h = -1;
    int32_t out_w = -1;
    check(pipeline.solve_field(nullptr, 0, &out_c, &out_h, &out_w) == nullptr,
        "default solve_field returns nullptr");

    check(!pipeline.supports_detection(), "default supports_detection is false");
    check(pipeline.detect("img.png", "detections.json") == -1, "default detect returns -1");

    check(!pipeline.supports_transcription(), "default supports_transcription is false");
    check(pipeline.transcribe("audio.wav") == nullptr, "default transcribe returns nullptr");

    check(!pipeline.supports_audio(), "default supports_audio is false");
    check(pipeline.generate_audio("hello", "audio.wav") == -1, "default generate_audio returns -1");

    check(!pipeline.supports_embedding(), "default supports_embedding is false");
    check(pipeline.embed("hello", &out_dim) == nullptr, "default embed returns nullptr");
    check(pipeline.embed_image("image.png", &out_dim) == nullptr, "default embed_image returns nullptr");
    check(pipeline.embed_image_text("hello", "image.png", &out_dim) == nullptr,
        "default embed_image_text returns nullptr");

    check(!pipeline.supports_reranking(), "default supports_reranking is false");
    check(pipeline.rerank("q", "doc") == 0.0F, "default rerank returns 0.0");

    check(!pipeline.supports_prompted_segmentation(), "default supports_prompted_segmentation is false");
    check(pipeline.segment_sam("image.png", "/tmp", 0.5F, 0.5F, true) == -1,
        "default segment_sam returns -1");

    check(!pipeline.supports_speech(), "default supports_speech is false");
    check(pipeline.speak("in.wav", "out.wav") == -1, "default speak returns -1");
}

int main()
{
    test_null_input_returns_null();
    test_invalid_path_returns_null();
    test_version_available();
    test_has_trt_returns_bool();
    test_sizeof_ipipeline_is_vtable();
    test_delete_null_safe();
    test_ipipeline_default_virtuals();

    if (failures > 0)
    {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All pipeline_api tests passed.\n";
    return 0;
}
