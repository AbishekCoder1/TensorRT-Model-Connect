// =============================================================================
// Sampling parameter and sampler behavior tests
// =============================================================================

#include "trtf/pipeline.h"
#include "trtf/runtime/sampler.h"

#include <cmath>
#include <cstdlib>
#include <iostream>

static int failures = 0;

static void check(bool condition, const char* test_name) {
    if (!condition) {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

static void test_sampling_params_from_config() {
    trtf::GenerateConfig cfg;
    cfg.temperature = 0.6F;
    cfg.top_k = 20;
    cfg.top_p = 0.95F;
    cfg.min_p = 0.1F;
    cfg.seed = 123;
    cfg.eos_token_id = 99;
    auto params = trtf::sampling_params_from_config(cfg, 42);
    check(params.temperature == 0.6F, "temperature forwarded");
    check(params.top_k == 20, "top_k forwarded");
    check(params.top_p == 0.95F, "top_p forwarded");
    check(params.min_p == 0.1F, "min_p forwarded");
    check(params.seed == 123, "seed forwarded");
    check(params.eos_token_id == 99, "explicit eos forwarded");
}

static void test_create_sampler_greedy_only_when_sampling_disabled() {
    trtf::SamplingParams params;
    params.top_k = 1;
    params.top_p = 1.0F;
    params.min_p = 0.0F;
    params.seed = -1;
    auto sampler = trtf::create_sampler(params);
    check(std::string(sampler->sampler_type()) == "greedy", "greedy factory path");

    params.top_p = 0.95F;
    sampler = trtf::create_sampler(params);
    const std::string sampler_type = sampler->sampler_type();
    check(sampler_type == "top_k" || sampler_type == "torch_multinomial",
          "top_p forces sampling path");
}

static void test_top_p_truncates_tail_tokens() {
    const float logits[] = {10.0F, 9.0F, -10.0F};
    trtf::SamplingParams params;
    params.temperature = 1.0F;
    params.top_k = 3;
    params.top_p = 0.55F;
    params.min_p = 0.0F;
    params.seed = 17;
    auto sampler = trtf::create_sampler(params);
    auto result = sampler->sample(logits, 3, params);
    check(result.token_id == 0, "top_p keeps only the highest-prob token");
}

static void test_min_p_drops_low_probability_tail() {
    const float logits[] = {5.0F, 5.0F, 0.0F};
    trtf::SamplingParams params;
    params.temperature = 1.0F;
    params.top_k = 3;
    params.top_p = 1.0F;
    params.min_p = 0.75F;
    params.seed = 9;
    auto sampler = trtf::create_sampler(params);
    for (int i = 0; i < 32; ++i) {
        auto result = sampler->sample(logits, 3, params);
        check(result.token_id != 2, "min_p excludes low-probability token");
    }
}

#if TRTF_HAS_LIBTORCH_MULTINOMIAL
static void test_torch_multinomial_matches_known_hf_sequence() {
    setenv("TRTF_USE_TORCH_MULTINOMIAL", "1", 1);

    trtf::SamplingParams params;
    params.temperature = 1.0F;
    params.top_k = 20;
    params.top_p = 0.95F;
    params.min_p = 0.0F;
    params.seed = 1235;

    auto sampler = trtf::create_sampler(params);
    check(std::string(sampler->sampler_type()) == "torch_multinomial",
          "torch sampler enabled");

    const float step0[] = {46.041664F, 43.75F, 43.541664F};
    const float step1[] = {46.875F, 45.416664F};
    const float step2[] = {51.666664F, 51.458332F};

    auto result0 = sampler->sample(step0, 3, params);
    auto result1 = sampler->sample(step1, 2, params);
    auto result2 = sampler->sample(step2, 2, params);

    check(result0.token_id == 0, "torch sampler step0 matches HF");
    check(result1.token_id == 0, "torch sampler step1 matches HF");
    check(result2.token_id == 0, "torch sampler step2 matches HF");

    unsetenv("TRTF_USE_TORCH_MULTINOMIAL");
}

static void test_torch_multinomial_uses_full_vocab_semantics() {
    setenv("TRTF_USE_TORCH_MULTINOMIAL", "1", 1);

    trtf::SamplingParams params;
    params.temperature = 1.0F;
    params.top_k = 2;
    params.top_p = 1.0F;
    params.min_p = 0.0F;
    params.seed = 1235;

    auto sampler = trtf::create_sampler(params);
    check(std::string(sampler->sampler_type()) == "torch_multinomial",
          "torch sampler enabled for sparse full-vocab test");

    std::vector<float> logits(100000, -1000.0F);
    logits[279] = 0.0F;
    logits[419] = std::log(0.45458386F / 0.54541614F);

    auto result = sampler->sample(logits.data(), static_cast<int32_t>(logits.size()), params);
    check(result.token_id == 419, "torch sampler matches full-vocab CUDA multinomial");

    unsetenv("TRTF_USE_TORCH_MULTINOMIAL");
}

static void test_torch_multinomial_advances_offset_like_full_vocab_cuda() {
    setenv("TRTF_USE_TORCH_MULTINOMIAL", "1", 1);

    trtf::SamplingParams params;
    params.temperature = 1.0F;
    params.top_k = 2;
    params.top_p = 1.0F;
    params.min_p = 0.0F;
    params.seed = 1235;

    auto sampler = trtf::create_sampler(params);
    check(std::string(sampler->sampler_type()) == "torch_multinomial",
          "torch sampler enabled for offset test");

    std::vector<float> logits(100000, -1000.0F);
    logits[279] = 0.0F;
    logits[419] = std::log(0.45458386F / 0.54541614F);

    const int expected[] = {419, 279, 279, 419, 279, 279, 419, 419};
    for (int token : expected) {
        auto result = sampler->sample(logits.data(), static_cast<int32_t>(logits.size()), params);
        check(result.token_id == token, "torch sampler preserves full-vocab offset progression");
    }

    unsetenv("TRTF_USE_TORCH_MULTINOMIAL");
}

static void test_torch_multinomial_matches_live_step_three_way_case() {
    setenv("TRTF_USE_TORCH_MULTINOMIAL", "1", 1);

    trtf::SamplingParams params;
    params.temperature = 0.6F;
    params.top_k = 20;
    params.top_p = 0.95F;
    params.min_p = 0.0F;
    params.seed = 1235;

    auto sampler = trtf::create_sampler(params);
    check(std::string(sampler->sampler_type()) == "torch_multinomial",
          "torch sampler enabled for live three-way test");

    std::vector<float> logits(151936, -1000.0F);
    logits[2014] = 27.5312F;
    logits[576] = 26.2188F;
    logits[6771] = 26.0938F;

    auto result = sampler->sample(logits.data(), static_cast<int32_t>(logits.size()), params);
    check(result.token_id == 2014, "torch sampler matches three-way live-step synthetic case");

    unsetenv("TRTF_USE_TORCH_MULTINOMIAL");
}
#endif

int main() {
    test_sampling_params_from_config();
    test_create_sampler_greedy_only_when_sampling_disabled();
    test_top_p_truncates_tail_tokens();
    test_min_p_drops_low_probability_tail();
#if TRTF_HAS_LIBTORCH_MULTINOMIAL
    test_torch_multinomial_matches_known_hf_sequence();
    test_torch_multinomial_uses_full_vocab_semantics();
    test_torch_multinomial_advances_offset_like_full_vocab_cuda();
    test_torch_multinomial_matches_live_step_three_way_case();
#endif

    if (failures > 0) {
        std::cerr << failures << " sampler test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All sampler tests passed.\n";
    return 0;
}
