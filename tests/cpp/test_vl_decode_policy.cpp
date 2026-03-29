// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-VL-CPP-03
// Architecture:   ARCH-FAC-001
// Unit Design:    UD-VL-01
// Intent:         VL decode policy: token/embedding selection, decode loop EOS stopping
// Preconditions:  VL decode policy with configured token and embedding sequences
// Postconditions: Current token and embedding selected correctly, loop stops on EOS, failures reported
// =============================================================================

#include "runtime/domains/multimodal/vl_decode_policy.h"

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

void test_current_vl_token_and_embedding_selection()
{
    check(trtf::current_vl_token({}, 7) == 7, "vl current token falls back to bos");
    check(trtf::current_vl_token({1, 2, 3}, 7) == 3, "vl current token uses last prompt token");

    const std::vector<float> image_features = {
        1.0F, 2.0F,
        3.0F, 4.0F,
    };
    const std::vector<std::vector<float>> deepstack = {
        {
            10.0F, 11.0F,
            12.0F, 13.0F,
        },
        {
            20.0F, 21.0F,
        },
    };
    int32_t feature_index = 0;
    auto embedding = trtf::build_vl_token_embedding(
        99,
        99,
        image_features.data(),
        2,
        2,
        feature_index,
        deepstack);

    check(embedding.use_input_embed == 1.0F, "vl embedding activates image feature injection");
    check(embedding.input_embed == image_features.data(), "vl embedding points at current image feature");
    check(feature_index == 1, "vl embedding advances feature index");
    check(embedding.deepstack_active == 1.0F, "vl embedding activates deepstack when present");
    check(embedding.deepstack_embeds.size() == 2, "vl embedding carries deepstack per level");
    check(embedding.deepstack_embeds[0] == deepstack[0].data(), "vl embedding selects matching deepstack level");
    check(embedding.deepstack_embeds[1] == deepstack[1].data(), "vl embedding allows sparse deepstack coverage");

    embedding = trtf::build_vl_token_embedding(
        5,
        99,
        image_features.data(),
        2,
        2,
        feature_index,
        deepstack);
    check(embedding.use_input_embed == 0.0F, "vl embedding skips non-image tokens");
    check(feature_index == 1, "vl embedding leaves feature index unchanged for non-image tokens");
}

void test_vl_decode_loop_stops_on_eos_and_reports_failures()
{
    int step_calls = 0;
    std::vector<float> logits = {0.1F, 0.2F};
    std::vector<int32_t> output;
    std::string error;

    bool ok = trtf::run_vl_decode_loop(
        4,
        77,
        logits,
        output,
        error,
        [&step_calls](int32_t next_token, std::vector<float>& next_logits, std::string&)
        {
            ++step_calls;
            next_logits = next_token == 11 ? std::vector<float>{1.0F, 0.0F} : std::vector<float>{0.0F, 1.0F};
            return true;
        },
        [](const std::vector<float>& next_logits)
        {
            return next_logits[1] > next_logits[0] ? 11 : 77;
        });

    check(ok, "vl decode loop succeeds on clean eos path");
    check(output == std::vector<int32_t>({11, 77}), "vl decode loop emits tokens until eos");
    check(step_calls == 1, "vl decode loop skips step callback after eos");

    logits = {1.0F, 0.0F};
    output.clear();
    error.clear();
    step_calls = 0;
    ok = trtf::run_vl_decode_loop(
        2,
        77,
        logits,
        output,
        error,
        [&step_calls](int32_t, std::vector<float>&, std::string& step_error)
        {
            ++step_calls;
            step_error = "decode-fail";
            return false;
        },
        [](const std::vector<float>&)
        {
            return 11;
        });

    check(!ok, "vl decode loop surfaces decode-step failure");
    check(error == "decode-fail", "vl decode loop forwards failure message");
    check(output == std::vector<int32_t>({11}), "vl decode loop preserves emitted token before failure");
    check(step_calls == 1, "vl decode loop stops immediately after failure");
}

} // namespace

int main()
{
    test_current_vl_token_and_embedding_selection();
    test_vl_decode_loop_stops_on_eos_and_reports_failures();

    if (g_failures != 0)
    {
        std::cerr << g_failures << " VL decode policy test(s) failed\n";
        return 1;
    }
    return 0;
}
