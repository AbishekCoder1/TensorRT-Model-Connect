#pragma once

#include <cstdint>

namespace trtf {

struct SpeechDecodeStopState
{
    int32_t text_eos_streak{0};
    int32_t text_pad_streak{0};
    bool stop_requested{false};
    int32_t stop_collect_until_offset{-1};
};

enum class SpeechDecodeStopReason
{
    kNone,
    kTextEos,
    kTextPadFallback,
    kContinuationCap
};

struct SpeechDecodeStopInput
{
    int32_t text_eos_token_id{-1};
    int32_t text_padding_id{0};
    int32_t effective_frames{0};
    int32_t extra_tail{0};
    int32_t target_pos{0};
    int32_t sampled_text_token{0};
    int32_t offset{0};
    int32_t max_delay{0};
    bool text_provided{false};
};

struct SpeechDecodeStopDecision
{
    SpeechDecodeStopState state;
    SpeechDecodeStopReason reason{SpeechDecodeStopReason::kNone};
    bool should_break{false};
};

inline constexpr int32_t kSpeechMinConsecutiveTextEos = 2;
inline constexpr int32_t kSpeechMinConsecutiveTextPadAfterInput = 16;
inline constexpr int32_t kSpeechMaxContinuationFramesAfterInput = 16;

inline SpeechDecodeStopDecision UpdateSpeechDecodeStopState(
    SpeechDecodeStopState state,
    const SpeechDecodeStopInput& input)
{
    SpeechDecodeStopDecision decision;
    decision.state = state;

    const auto request_stop = [&](SpeechDecodeStopReason reason) {
        decision.state.stop_requested = true;
        decision.state.stop_collect_until_offset = input.offset + input.max_delay;
        decision.reason = reason;
    };

    if (input.text_eos_token_id < 0
        || input.text_provided
        || input.target_pos < input.effective_frames
        || input.sampled_text_token != input.text_eos_token_id)
    {
        decision.state.text_eos_streak = 0;
    }
    else
    {
        ++decision.state.text_eos_streak;
        if (!decision.state.stop_requested
            && decision.state.text_eos_streak >= kSpeechMinConsecutiveTextEos)
        {
            request_stop(SpeechDecodeStopReason::kTextEos);
        }
    }

    if (decision.state.stop_requested
        || input.extra_tail <= 0
        || input.text_provided
        || input.target_pos < input.effective_frames
        || input.sampled_text_token != input.text_padding_id)
    {
        decision.state.text_pad_streak = 0;
    }
    else
    {
        ++decision.state.text_pad_streak;
        if (decision.state.text_pad_streak >= kSpeechMinConsecutiveTextPadAfterInput)
        {
            request_stop(SpeechDecodeStopReason::kTextPadFallback);
        }
    }

    const bool should_break_before_cap = decision.state.stop_requested
        && input.offset >= decision.state.stop_collect_until_offset;
    if (!should_break_before_cap
        && !decision.state.stop_requested
        && input.extra_tail > 0
        && input.target_pos
            >= (input.effective_frames + kSpeechMaxContinuationFramesAfterInput))
    {
        request_stop(SpeechDecodeStopReason::kContinuationCap);
    }

    decision.should_break = decision.state.stop_requested
        && input.offset >= decision.state.stop_collect_until_offset;
    return decision;
}

} // namespace trtf
