#include "runtime/trt/audio/magpie_decode_policy.h"

#include <cmath>
#include <cstdint>
#include <iostream>
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

void test_greedy_decode_uses_audio_range_but_still_detects_eos()
{
    const int32_t num_cb = 2;
    const int32_t cb_size = trtf::kMagpieEosToken + 1;
    std::vector<float> logits(static_cast<std::size_t>(num_cb) * cb_size, -10.0F);

    logits[10] = 4.0F;
    logits[trtf::kMagpieEosToken] = 5.0F;

    const auto cb1_offset = static_cast<std::size_t>(cb_size);
    logits[cb1_offset + 7] = 3.0F;

    int sampler_calls = 0;
    const auto result = trtf::decode_magpie_frame_codes(
        logits,
        num_cb,
        cb_size,
        true,
        0.8F,
        80,
        [&sampler_calls](const float*, int32_t, float, int32_t)
        {
            ++sampler_calls;
            return -1;
        });

    check(result.eos, "greedy decode detects eos from full argmax");
    check(result.frame_codes == std::vector<int32_t>({10, 7}),
        "greedy decode samples from audio range only");
    check(sampler_calls == 0, "greedy decode skips sampler");
}

void test_sampling_decode_uses_sampler()
{
    const int32_t num_cb = 2;
    const int32_t cb_size = trtf::kMagpieEosToken + 1;
    std::vector<float> logits(static_cast<std::size_t>(num_cb) * cb_size, -5.0F);
    logits[trtf::kMagpieEosToken] = 2.0F;

    int sampler_calls = 0;
    const auto result = trtf::decode_magpie_frame_codes(
        logits,
        num_cb,
        cb_size,
        false,
        0.6F,
        32,
        [&sampler_calls](const float*, int32_t vocab_size, float temperature, int32_t top_k)
        {
            ++sampler_calls;
            check(vocab_size == trtf::kMagpieAudioRange, "sampler path uses audio range vocab");
            check(std::fabs(temperature - 0.6F) < 1e-6F, "sampler path forwards temperature");
            check(top_k == 32, "sampler path forwards top-k");
            return 40 + sampler_calls;
        });

    check(result.eos, "sampling decode still detects eos");
    check(result.frame_codes == std::vector<int32_t>({41, 42}),
        "sampling decode returns sampler-selected ids");
    check(sampler_calls == 2, "sampling decode invokes sampler per codebook");
}

void test_repetition_helpers_and_stop_rules()
{
    const std::vector<int32_t> repeated = {
        1, 2,
        3, 4,
        3, 4,
        3, 4,
    };
    const std::vector<int32_t> distinct = {
        1, 2,
        3, 4,
        5, 6,
    };

    check(trtf::magpie_has_repeated_tail_frames(repeated, 2, 3),
        "repetition helper detects repeated tail");
    check(!trtf::magpie_has_repeated_tail_frames(distinct, 2, 3),
        "repetition helper ignores distinct tail");
    check(trtf::magpie_trimmed_frame_count_for_repetition(5, 3) == 3,
        "trimmed frame count removes repeated suffix");
    check(trtf::should_run_magpie_periodic_check(15, 4, 16),
        "periodic helper triggers on configured interval");
    check(!trtf::should_run_magpie_periodic_check(3, 4, 16),
        "periodic helper skips early frames");
    check(trtf::should_stop_magpie_on_eos(true, trtf::kMagpieMinFrames, trtf::kMagpieMinFrames),
        "eos helper stops after minimum frames");
    check(!trtf::should_stop_magpie_on_eos(true, 1, trtf::kMagpieMinFrames),
        "eos helper keeps short outputs alive");
}

} // namespace

int main()
{
    test_greedy_decode_uses_audio_range_but_still_detects_eos();
    test_sampling_decode_uses_sampler();
    test_repetition_helpers_and_stop_rules();

    if (g_failures != 0)
    {
        std::cerr << g_failures << " Magpie decode policy test(s) failed\n";
        return 1;
    }
    return 0;
}
