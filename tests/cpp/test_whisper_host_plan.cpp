// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-AUD-CPP-19
// Architecture:   ARCH-FAC-001
// Unit Design:    UD-AUD-01
// Intent:         Whisper host plan: mel length, initial tokens, encoder planning, cross-KV apply
// Preconditions:  Whisper config with valid mel/encoder parameters
// Postconditions: Mel length and tokens correct, encoder mask planned, cross-KV copy operations tracked
// =============================================================================

#include "runtime/trt/audio/whisper_cross_kv_apply.h"
#include "runtime/trt/audio/whisper_cross_kv_plan.h"
#include "runtime/trt/audio/whisper_host_plan.h"

#include <cstdint>
#include <cstring>
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

void test_expected_mel_length_and_initial_tokens()
{
    trtf::WhisperConfig cfg;
    cfg.max_source_positions = 1500;
    check(trtf::resolve_whisper_expected_mel_length(cfg) == 3000,
        "whisper expected mel length defaults to max_source_positions * 2");

    cfg.mel_length = 4096;
    check(trtf::resolve_whisper_expected_mel_length(cfg) == 4096,
        "whisper expected mel length honors explicit override");

    cfg.decoder_start_token_ids = {1, 2, 3};
    check(trtf::make_whisper_initial_decoder_tokens(cfg) == std::vector<int32_t>({1, 2, 3}),
        "whisper initial tokens honor custom sequence");

    cfg.decoder_start_token_ids.clear();
    cfg.decoder_start_token_id = 10;
    cfg.language_token_id = 11;
    cfg.transcribe_token_id = 12;
    cfg.notimestamps_token_id = 13;
    check(trtf::make_whisper_initial_decoder_tokens(cfg) == std::vector<int32_t>({10, 11, 12, 13}),
        "whisper initial tokens fall back to default special tokens");
}

void test_encoder_length_planning()
{
    check(trtf::count_whisper_stride2_stages(3000, 1500) == 1,
        "whisper stride stage count handles one stride-2 stage");
    check(trtf::count_whisper_stride2_stages(4000, 1000) == 2,
        "whisper stride stage count handles repeated stride-2 stages");
    check(trtf::apply_whisper_stride2_subsampling(3000, 1) == 1500,
        "whisper stride-2 subsampling halves even length");
    check(trtf::apply_whisper_stride2_subsampling(2999, 1) == 1500,
        "whisper stride-2 subsampling rounds like convolution output");

    check(trtf::compute_whisper_actual_encoder_length(2000, 3000, 1500) == 1000,
        "whisper actual encoder length subsamples partial mel length");
    check(trtf::compute_whisper_actual_encoder_length(3000, 3000, 1500) == 0,
        "whisper actual encoder length returns zero for full-length mel");
    check(trtf::compute_whisper_actual_encoder_length(100, 0, 1500) == 0,
        "whisper actual encoder length returns zero for invalid full mel length");
}

void test_mel_padding_and_truncation_are_row_major()
{
    const float mel_data[] = {
        1.0F, 2.0F, 3.0F,
        4.0F, 5.0F, 6.0F,
    };
    const auto padded = trtf::build_whisper_padded_mel_input(mel_data, 2, 3, 5);
    check(padded.size() == 10, "whisper padded mel buffer uses expected size");
    check(padded == std::vector<float>({1.0F, 2.0F, 3.0F, 0.0F, 0.0F, 4.0F, 5.0F, 6.0F, 0.0F, 0.0F}),
        "whisper padded mel buffer pads each mel row independently");

    const auto truncated = trtf::build_whisper_padded_mel_input(mel_data, 2, 3, 2);
    check(truncated == std::vector<float>({1.0F, 2.0F, 4.0F, 5.0F}),
        "whisper padded mel buffer truncates each mel row independently");

    check(trtf::build_whisper_padded_mel_input(nullptr, 2, 3, 5).empty(),
        "whisper padded mel buffer returns empty for null input");
}

void test_encoder_mask_and_cross_kv_plan()
{
    const auto full_mask = trtf::build_whisper_encoder_mask_values(4, 4);
    check(full_mask == std::vector<float>({0.0F, 0.0F, 0.0F, 0.0F}),
        "whisper encoder mask leaves valid sequence unmasked");

    const auto partial_mask = trtf::build_whisper_encoder_mask_values(5, 2);
    check(partial_mask == std::vector<float>({0.0F, 0.0F, -10000.0F, -10000.0F, -10000.0F}),
        "whisper encoder mask marks padded encoder positions");

    const auto clamped_mask = trtf::build_whisper_encoder_mask_values(3, -2);
    check(clamped_mask == std::vector<float>({-10000.0F, -10000.0F, -10000.0F}),
        "whisper encoder mask clamps negative actual length");

    const auto no_mask = trtf::build_whisper_encoder_mask_values(0, 0);
    check(no_mask.empty(), "whisper encoder mask returns empty for invalid sequence length");

    const auto full_plan = trtf::make_whisper_cross_kv_plan(1500, 4, 0);
    check(full_plan.buffer_bytes == static_cast<std::size_t>(1500 * 4 * static_cast<int>(sizeof(float))),
        "whisper cross-kv plan computes full buffer size");
    check(!full_plan.zero_pad_encoder_output && full_plan.pad_bytes == 0,
        "whisper cross-kv plan skips zero-padding when actual length is unknown");

    const auto partial_plan = trtf::make_whisper_cross_kv_plan(1500, 4, 1000);
    check(partial_plan.zero_pad_encoder_output, "whisper cross-kv plan marks partial encoder output for zero-padding");
    check(partial_plan.valid_bytes == static_cast<std::size_t>(1000 * 4 * static_cast<int>(sizeof(float))),
        "whisper cross-kv plan computes valid byte span");
    check(partial_plan.pad_bytes == partial_plan.buffer_bytes - partial_plan.valid_bytes,
        "whisper cross-kv plan computes padded byte span");

    const auto invalid_plan = trtf::make_whisper_cross_kv_plan(0, 4, 0);
    check(invalid_plan.buffer_bytes == 0 && !invalid_plan.zero_pad_encoder_output,
        "whisper cross-kv plan returns empty plan for invalid shape");
}

void test_cross_kv_apply_tracks_zero_and_copy_operations()
{
    const auto plan = trtf::make_whisper_cross_kv_plan(8, 4, 3);
    trtf::WhisperCrossKvApplyStats stats;
    std::vector<std::size_t> zero_calls;
    std::vector<std::pair<std::size_t, trtf::WhisperCrossKvBufferKind>> copy_calls;
    std::string error;

    const bool ok = trtf::apply_whisper_cross_kv_plan(
        plan,
        2,
        [&zero_calls](std::size_t valid_bytes, std::size_t pad_bytes)
        {
            zero_calls.emplace_back(valid_bytes + pad_bytes);
            return true;
        },
        [&copy_calls](std::size_t layer, trtf::WhisperCrossKvBufferKind kind, std::size_t bytes)
        {
            copy_calls.emplace_back(layer, kind);
            return bytes > 0;
        },
        error,
        &stats);

    check(ok, "whisper cross-kv apply succeeds for valid plan");
    check(error.empty(), "whisper cross-kv apply leaves error empty on success");
    check(zero_calls.size() == 1, "whisper cross-kv apply zeroes encoder padding once");
    check(copy_calls.size() == 4, "whisper cross-kv apply copies k and v per layer");
    check(stats.zero_ops == 1, "whisper cross-kv apply counts zero operations");
    check(stats.copy_ops == 4, "whisper cross-kv apply counts copy operations");
    check(copy_calls[0] == std::make_pair<std::size_t, trtf::WhisperCrossKvBufferKind>(
            0, trtf::WhisperCrossKvBufferKind::K),
        "whisper cross-kv apply copies cross_k first");
    check(copy_calls[1] == std::make_pair<std::size_t, trtf::WhisperCrossKvBufferKind>(
            0, trtf::WhisperCrossKvBufferKind::V),
        "whisper cross-kv apply copies cross_v second");
}

void test_cross_kv_apply_reports_failures()
{
    const auto plan = trtf::make_whisper_cross_kv_plan(8, 4, 3);
    std::string error;

    bool ok = trtf::apply_whisper_cross_kv_plan(
        plan,
        1,
        [](std::size_t, std::size_t)
        {
            return false;
        },
        [](std::size_t, trtf::WhisperCrossKvBufferKind, std::size_t)
        {
            return true;
        },
        error);
    check(!ok, "whisper cross-kv apply fails when zeroing fails");
    check(error == "failed to zero whisper encoder padding",
        "whisper cross-kv apply reports zeroing failure");

    error.clear();
    ok = trtf::apply_whisper_cross_kv_plan(
        plan,
        1,
        [](std::size_t, std::size_t)
        {
            return true;
        },
        [](std::size_t, trtf::WhisperCrossKvBufferKind kind, std::size_t)
        {
            return kind == trtf::WhisperCrossKvBufferKind::K;
        },
        error);
    check(!ok, "whisper cross-kv apply fails when copy fails");
    check(error == "failed to copy whisper cross_v",
        "whisper cross-kv apply reports copy failure");

    error.clear();
    ok = trtf::apply_whisper_cross_kv_plan(
        trtf::WhisperCrossKvPlan{},
        1,
        [](std::size_t, std::size_t)
        {
            return true;
        },
        [](std::size_t, trtf::WhisperCrossKvBufferKind, std::size_t)
        {
            return true;
        },
        error);
    check(!ok, "whisper cross-kv apply rejects empty plan");
    check(error == "invalid whisper cross-kv plan",
        "whisper cross-kv apply reports invalid plan");
}

} // namespace

int main()
{
    test_expected_mel_length_and_initial_tokens();
    test_encoder_length_planning();
    test_mel_padding_and_truncation_are_row_major();
    test_encoder_mask_and_cross_kv_plan();
    test_cross_kv_apply_tracks_zero_and_copy_operations();
    test_cross_kv_apply_reports_failures();

    if (g_failures != 0)
    {
        std::cerr << g_failures << " whisper host plan test(s) failed\n";
        return 1;
    }
    return 0;
}
