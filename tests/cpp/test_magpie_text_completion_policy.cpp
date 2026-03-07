#include "runtime/trt/audio/magpie_text_completion_policy.h"

#include <cstdint>
#include <iostream>

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

void test_cross_attention_tracking_updates_high_water_mark()
{
    const float xattn[] = {0.1F, 0.2F, 0.9F, 0.4F};
    int32_t max_peak_pos = 1;
    bool text_consumed = false;
    const bool changed = trtf::update_magpie_text_consumed_from_cross_attn(
        xattn, 4, 2, max_peak_pos, text_consumed);

    check(changed, "cross attn marks text consumed");
    check(text_consumed, "cross attn flips text consumed flag");
    check(max_peak_pos == 2, "cross attn updates high water mark");
}

void test_heuristic_marks_text_consumed_after_estimate()
{
    bool text_consumed = false;
    check(!trtf::update_magpie_text_consumed_from_heuristic(5, 4, text_consumed),
        "heuristic waits before threshold");
    check(trtf::update_magpie_text_consumed_from_heuristic(5, 5, text_consumed),
        "heuristic triggers at estimate");
    check(text_consumed, "heuristic sets text consumed");
}

void test_finished_limit_only_advances_after_text_consumed()
{
    int32_t frames_past = 0;
    check(!trtf::advance_magpie_finished_limit(false, 2, frames_past),
        "finished limit idle before text consumed");
    check(frames_past == 0, "finished limit does not advance early");
    check(!trtf::advance_magpie_finished_limit(true, 2, frames_past),
        "finished limit waits for second frame");
    check(frames_past == 1, "finished limit increments counter");
    check(trtf::advance_magpie_finished_limit(true, 2, frames_past),
        "finished limit stops at threshold");
}

} // namespace

int main()
{
    test_cross_attention_tracking_updates_high_water_mark();
    test_heuristic_marks_text_consumed_after_estimate();
    test_finished_limit_only_advances_after_text_consumed();

    if (g_failures != 0)
    {
        std::cerr << g_failures << " magpie text completion policy test(s) failed\n";
        return 1;
    }
    return 0;
}
