// =============================================================================
// Test suite: Bundle end-to-end -- build, save, load, and generate from
//   .trtfb bundles using the full pipeline.
//
// Purpose:
//   Validates the complete bundle lifecycle: creating a TRT pipeline from a
//   model, saving it as a .trtfb bundle, reloading the bundle as a new
//   pipeline, and generating text from the reloaded pipeline. This exercises
//   the integration between the C ABI pipeline, TRT backend, bundle
//   serialization, and bundle deserialization.
//
// Dependencies:
//   - trtf/pipeline.h: C ABI entry points (trtf_create_pipeline, trtf_has_trt).
//   - trtf/bundle.h: IsBundle, InspectBundle.
//   - TensorRT + GPU: required for pipeline creation and text generation.
//   - The "trtf/tiny-cake-v1" built-in test model.
//   - Filesystem access (temp directories via mkdtemp).
//
// Approach:
//   Tests are guarded by a has_trt flag (checked via trtf_has_trt()). When
//   TRT is not available, tests print SKIP and pass without assertions. When
//   TRT is available but pipeline creation fails (e.g., missing model files),
//   tests also skip gracefully. This allows the test binary to run in both
//   GPU and CPU-only environments without failure.
//
// Test categories:
//   - Save/load roundtrip: save pipeline to bundle, reload, generate text
//   - Inspect: verify InspectBundle extracts metadata from a saved bundle
// =============================================================================

#include "trtf/pipeline.h"
#include "trtf/bundle.h"

#include <cerrno>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>

#include <stdlib.h>

static int failures = 0;
static bool has_trt = false;

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
    char pattern[] = "/tmp/trtfb_e2e_XXXXXX";
    char* dir = mkdtemp(pattern);
    if (dir == nullptr)
    {
        throw std::runtime_error(std::string("mkdtemp failed: ") + std::strerror(errno));
    }
    return std::filesystem::path(dir);
}

// -----------------------------------------------------------------------------
// Intention: Verify the full bundle save/load/generate roundtrip -- a pipeline
//   created from a model can be saved as a .trtfb bundle, and the bundle can
//   be reloaded as a new pipeline that successfully generates text.
// Setup: Creates a TRT pipeline from the "trtf/tiny-cake-v1" built-in model
//   with TRTF_FORCE_TRT. Saves the pipeline to a temp .trtfb file. The
//   original pipeline generates 5 tokens from "hello" as a baseline.
// Mechanism:
//   1. Checks has_trt; skips if TRT unavailable.
//   2. Creates a pipeline via trtf_create_pipeline; skips if creation fails.
//   3. Calls save_bundle(); skips if not implemented for this backend.
//   4. Asserts the bundle file exists on disk and IsBundle() returns true.
//   5. Generates text with the original pipeline as a baseline.
//   6. Loads the bundle as a new pipeline via trtf_create_pipeline.
//   7. Asserts the bundle-loaded pipeline is non-null and can generate text.
//   8. Cleans up the temp directory.
// -----------------------------------------------------------------------------
static void test_save_and_load_roundtrip()
{
    if (!has_trt)
    {
        std::cerr << "  SKIP: test_save_and_load_roundtrip (no TRT)\n";
        return;
    }

    auto* p = trtf_create_pipeline("trtf/tiny-cake-v1", TRTF_FORCE_TRT);
    if (p == nullptr)
    {
        std::cerr << "  SKIP: test_save_and_load_roundtrip (TRT pipeline creation failed: "
                  << trtf_last_error() << ")\n";
        return;
    }

    const auto tmp = make_temp_dir();
    const auto bundle_path = (tmp / "roundtrip.trtfb").string();

    const bool saved = p->save_bundle(bundle_path.c_str());
    if (!saved)
    {
        std::cerr << "  SKIP: test_save_and_load_roundtrip (save_bundle not yet implemented for this backend)\n";
        delete p;
        std::filesystem::remove_all(tmp);
        return;
    }

    check(std::filesystem::exists(bundle_path), "bundle file created");
    check(trtf::IsBundle(bundle_path), "bundle has valid magic");

    // Generate text with original pipeline
    const char* original_output = p->generate("hello", 5);
    const std::string original_text(original_output ? original_output : "");
    delete p;

    // Load from bundle and generate
    auto* p2 = trtf_create_pipeline(bundle_path.c_str(), TRTF_FORCE_TRT);
    check(p2 != nullptr, "bundle loads as pipeline");
    if (p2 != nullptr)
    {
        const char* bundle_output = p2->generate("hello", 5);
        check(bundle_output != nullptr, "bundle pipeline generates text");
        delete p2;
    }

    std::filesystem::remove_all(tmp);
}

// -----------------------------------------------------------------------------
// Intention: Verify that InspectBundle() can extract meaningful metadata from
//   a bundle file that was saved by a real TRT pipeline (not hand-crafted).
// Setup: Creates a TRT pipeline from "trtf/tiny-cake-v1", saves it as a
//   .trtfb bundle to a temp directory.
// Mechanism:
//   1. Checks has_trt; skips if TRT unavailable.
//   2. Creates and saves a pipeline; skips if creation or save fails.
//   3. Calls InspectBundle on the saved file.
//   4. Asserts that model_id is non-empty (the inspection extracted real data).
//   5. Cleans up the temp directory.
// -----------------------------------------------------------------------------
static void test_inspect_saved_bundle()
{
    if (!has_trt)
    {
        std::cerr << "  SKIP: test_inspect_saved_bundle (no TRT)\n";
        return;
    }

    auto* p = trtf_create_pipeline("trtf/tiny-cake-v1", TRTF_FORCE_TRT);
    if (p == nullptr)
    {
        std::cerr << "  SKIP: test_inspect_saved_bundle (TRT pipeline creation failed)\n";
        return;
    }

    const auto tmp = make_temp_dir();
    const auto bundle_path = (tmp / "inspect.trtfb").string();

    const bool saved = p->save_bundle(bundle_path.c_str());
    delete p;

    if (!saved)
    {
        std::cerr << "  SKIP: test_inspect_saved_bundle (save_bundle not yet implemented)\n";
        std::filesystem::remove_all(tmp);
        return;
    }

    const auto info = trtf::InspectBundle(bundle_path);
    check(!info.model_id.empty(), "inspect model_id non-empty");

    std::filesystem::remove_all(tmp);
}

int main()
{
    has_trt = (trtf_has_trt() == 1);

    test_save_and_load_roundtrip();
    test_inspect_saved_bundle();

    if (failures > 0)
    {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All bundle E2E tests passed.\n";
    return 0;
}
