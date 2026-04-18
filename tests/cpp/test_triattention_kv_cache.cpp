// =============================================================================
// TriAttentionKvCache native runtime tests
// =============================================================================

#include "trtf/runtime/triattention_kv_cache.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#if TRTF_HAS_TRT
#include <cuda_runtime_api.h>
#endif

static int failures = 0;

static void check(bool condition, const char* name)
{
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

#if TRTF_HAS_TRT

namespace {

trtf::TriAttentionConfig make_config(int32_t kv_budget, int32_t recent_window = 1,
                                     bool protect_prefill = false)
{
    trtf::TriAttentionConfig cfg;
    cfg.enabled = true;
    cfg.kv_budget = kv_budget;
    cfg.count_prompt_tokens = true;
    cfg.recent_window = recent_window;
    cfg.protect_prefill = protect_prefill;
    cfg.disable_trig = true;
    cfg.disable_mlr = false;
    cfg.score_aggregation = trtf::TriAttentionScoreAggregation::kMean;
    return cfg;
}

trtf::TriAttentionStats make_stats()
{
    trtf::TriAttentionStats stats;
    stats.head_dim = 4;
    stats.rope_style = trtf::TriAttentionRopeStyle::kHalf;
    stats.num_attention_heads = 1;
    stats.num_key_value_heads = 1;
    stats.stats_head_count = 1;
    stats.num_layers = 1;
    stats.inv_freq = {1.0F, 0.1F};

    trtf::TriAttentionHeadStats head;
    head.q_mean_real = {0.0F, 0.0F};
    head.q_mean_imag = {0.0F, 0.0F};
    head.q_abs_mean = {1.0F, 0.0F};
    head.freq_scale_sq = {1.0F, 1.0F};
    stats.layer_stats.push_back(std::move(head));
    return stats;
}

void write_present_row(trtf::TriAttentionKvCache& cache, const std::vector<float>& row)
{
    cache.present_k(0).copy_from_host(row.data());
    cache.present_v(0).copy_from_host(row.data());
    cudaStreamSynchronize(cache.present_k(0).stream());
}

void test_absolute_position_and_mask()
{
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    {
        trtf::TriAttentionKvCache cache(1, 1, 3, 4, stream, make_config(3), make_stats());
        for (int i = 0; i < 6; ++i) {
            write_present_row(cache, {1.0F + static_cast<float>(i), 0.0F, 0.0F, 0.0F});
            cache.advance();
        }

        check(cache.position() == 6, "absolute position keeps growing");
        check(cache.active_length() == 3, "active cache length capped at kv_budget");

        std::vector<float> mask;
        cache.build_attention_mask(mask);
        check(mask.size() == 4, "mask size matches max_length + 1");
        check(mask[0] == 0.0F, "mask[0] visible");
        check(mask[1] == 0.0F, "mask[1] visible");
        check(mask[2] == 0.0F, "mask[2] visible");
        check(mask[3] == 0.0F, "mask current slot visible");
    }

    cudaStreamDestroy(stream);
}

void test_score_based_compaction()
{
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    trtf::TriAttentionKvCache cache(1, 1, 3, 4, stream, make_config(3), make_stats());

    write_present_row(cache, {10.0F, 0.0F, 0.0F, 0.0F});
    cache.advance(); // pos 0
    write_present_row(cache, {1.0F, 0.0F, 0.0F, 0.0F});
    cache.advance(); // pos 1
    write_present_row(cache, {5.0F, 0.0F, 0.0F, 0.0F});
    cache.advance(); // pos 2
    write_present_row(cache, {2.0F, 0.0F, 0.0F, 0.0F});
    cache.advance(); // compact [0,1,2] -> keep [0,2], append 3

    const auto& positions = cache.cache_positions();
    check(positions.size() == 3, "score compaction keeps kv_budget positions");
    check(positions[0] == 0, "highest-score early token retained");
    check(positions[1] == 2, "second-best token retained");
    check(positions[2] == 3, "newest token appended");

    cudaStreamDestroy(stream);
}

void test_protect_prefill()
{
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    trtf::TriAttentionKvCache cache(1, 1, 3, 4, stream, make_config(3, 1, true), make_stats());

    write_present_row(cache, {1.0F, 0.0F, 0.0F, 0.0F});
    cache.advance(); // pos 0
    write_present_row(cache, {2.0F, 0.0F, 0.0F, 0.0F});
    cache.advance(); // pos 1
    write_present_row(cache, {3.0F, 0.0F, 0.0F, 0.0F});
    cache.advance(); // pos 2
    cache.mark_prefill_complete();

    write_present_row(cache, {4.0F, 0.0F, 0.0F, 0.0F});
    cache.advance();
    write_present_row(cache, {5.0F, 0.0F, 0.0F, 0.0F});
    cache.advance();

    const auto& positions = cache.cache_positions();
    check(positions.size() == 3, "prefill protection keeps kv_budget positions");
    check(positions[0] == 0, "prefill token 0 preserved");
    check(positions[1] == 1, "prefill token 1 preserved");
    check(positions[2] == 4, "latest generated token appended");
    check(cache.prompt_end_position() == 3, "prompt boundary recorded");

    cudaStreamDestroy(stream);
}

void test_prefill_protection_during_overflow()
{
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    trtf::TriAttentionKvCache cache(1, 1, 3, 4, stream, make_config(3, 1, true), make_stats());
    cache.set_prompt_length(5);

    for (int i = 0; i < 5; ++i) {
        write_present_row(cache, {1.0F + static_cast<float>(i), 0.0F, 0.0F, 0.0F});
        cache.advance();
    }

    const auto& positions = cache.cache_positions();
    check(positions.size() == 3, "overflow prefill keeps kv_budget positions");
    check(positions[0] == 0, "earliest prompt token survives overflow");
    check(positions[1] == 1, "second prompt token survives overflow");
    check(positions[2] == 4, "latest prompt token appended");

    cudaStreamDestroy(stream);
}

void test_slack_window_delays_compaction()
{
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    trtf::TriAttentionKvCache cache(1, 1, 5, 4, stream, make_config(3, 1, false), make_stats());

    for (int i = 0; i < 5; ++i) {
        write_present_row(cache, {1.0F + static_cast<float>(i), 0.0F, 0.0F, 0.0F});
        cache.advance();
    }

    check(cache.active_length() == 3,
          "slack-threshold compaction now fires as soon as the physical window is reached");
    check(cache.cache_positions().size() == 3,
          "cache positions shrink to the logical budget at the threshold-crossing step");

    write_present_row(cache, {6.0F, 0.0F, 0.0F, 0.0F});
    cache.advance();

    check(cache.active_length() == 4,
          "compaction keeps the full logical budget before appending the new token");
    check(cache.cache_positions().size() == 4,
          "cache positions reflect logical budget plus the just-appended token");

    cudaStreamDestroy(stream);
}

void test_prefill_overflow_uses_physical_slack()
{
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    trtf::TriAttentionKvCache cache(1, 1, 5, 4, stream, make_config(3, 1, false), make_stats());
    cache.set_prompt_length(6);

    for (int i = 0; i < 6; ++i) {
        write_present_row(cache, {1.0F + static_cast<float>(i), 0.0F, 0.0F, 0.0F});
        cache.advance();
    }

    check(cache.position() == 6, "prefill overflow advances absolute position");
    check(cache.active_length() == 3,
          "final prompt token compacts to the logical budget once prompt_length is reached");
    check(cache.cache_positions().size() == 3,
          "prompt-overflow positions shrink to the logical budget at prompt completion");

    cache.mark_prefill_complete();
    write_present_row(cache, {7.0F, 0.0F, 0.0F, 0.0F});
    cache.advance();

    check(cache.active_length() == 4,
          "decode after prompt completion appends one token on top of the compacted prompt state");
    check(cache.cache_positions().size() == 4,
          "decode after prompt completion reflects the compacted prompt plus the new decode token");

    cudaStreamDestroy(stream);
}

void test_slack_window_uses_full_logical_budget()
{
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    trtf::TriAttentionKvCache cache(1, 1, 5, 4, stream, make_config(3, 1, false), make_stats());

    for (int i = 0; i < 6; ++i) {
        write_present_row(cache, {1.0F + static_cast<float>(i), 0.0F, 0.0F, 0.0F});
        cache.advance();
    }

    const auto& positions = cache.cache_positions();
    check(positions.size() == 4,
          "slack-window compaction keeps kv_budget old rows plus the appended token");
    check(positions[0] == 2, "highest-scoring candidate survives first slack compaction");
    check(positions[1] == 3, "second-highest-scoring candidate survives first slack compaction");
    check(positions[2] == 4, "recent protected token remains before append");
    check(positions[3] == 5, "new token is appended after compaction");

    cudaStreamDestroy(stream);
}

void test_bundle_parsing()
{
    const std::string config_json = R"json(
{
  "triattention": {
    "enabled": true,
    "kv_budget": 64,
    "divide_length": 16,
    "recent_window": 8,
    "score_aggregation": "max",
    "per_layer_aggregation": "max",
    "count_prompt_tokens": false,
    "protect_prefill": true,
    "disable_mlr": true,
    "disable_trig": false,
    "stats_section": "triattention_stats.json"
  }
}
)json";
    const std::string stats_json = R"json(
{
  "head_dim": 4,
  "rope_style": "half",
  "rope_theta": 10000.0,
  "num_attention_heads": 2,
  "num_key_value_heads": 1,
  "stats_head_count": 2,
  "num_layers": 1,
  "inv_freq": [1.0, 0.1],
  "sampled_heads": [[0, 1]],
  "layer_stats": {
    "0": {
      "q_mean_real": [[0.0, 0.0], [1.0, 1.0]],
      "q_mean_imag": [[0.0, 0.0], [0.0, 0.0]],
      "q_abs_mean": [[1.0, 1.0], [2.0, 2.0]],
      "freq_scale_sq": [[1.0, 1.0], [1.0, 1.0]]
    }
  }
}
)json";

    auto cfg = trtf::parse_triattention_bundle_config(config_json, 128);
    auto stats = trtf::parse_triattention_stats_json(stats_json, 1, 1, 1);
    check(cfg.enabled, "bundle config enables triattention");
    check(cfg.kv_budget == 64, "bundle config parses kv_budget");
    check(cfg.divide_length == 16, "bundle config parses divide_length");
    check(cfg.recent_window == 8, "bundle config parses recent_window");
    check(!cfg.count_prompt_tokens, "bundle config parses count_prompt_tokens");
    check(cfg.score_aggregation == trtf::TriAttentionScoreAggregation::kMax,
          "bundle config parses score aggregation");
    check(cfg.per_layer_aggregation == trtf::TriAttentionScoreAggregation::kMax,
          "bundle config parses per-layer aggregation");
    check(stats.head_dim == 4, "stats parse head_dim");
    check(stats.layer_stats.size() == 1, "stats parse layer stats");
    check(stats.stats_head_count == 2, "stats parse score head count");
    check(stats.inv_freq.size() == 2 && stats.inv_freq[1] == 0.1F, "stats parse inv_freq");
    check(stats.sampled_score_heads_by_layer.size() == 1,
          "stats parse sampled-head layer mapping");
    check(stats.sampled_score_heads_by_layer[0].size() == 1 &&
              stats.sampled_score_heads_by_layer[0][0] == 1,
          "stats preserve actual sampled score heads");
}

} // namespace

#endif // TRTF_HAS_TRT

int main()
{
#if TRTF_HAS_TRT
    test_absolute_position_and_mask();
    test_score_based_compaction();
    test_protect_prefill();
    test_prefill_protection_during_overflow();
    test_slack_window_delays_compaction();
    test_prefill_overflow_uses_physical_slack();
    test_slack_window_uses_full_logical_budget();
    test_bundle_parsing();
#else
    std::cerr << "TRT not available, skipping TriAttentionKvCache tests\n";
#endif

    if (failures > 0)
        std::cerr << failures << " test(s) FAILED\n";
    return failures;
}
