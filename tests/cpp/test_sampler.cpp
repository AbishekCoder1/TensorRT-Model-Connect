// =============================================================================
// Sampling parameter and sampler behavior tests
// =============================================================================

#include "trtf/pipeline.h"
#include "trtf/runtime/sampler.h"

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
    check(std::string(sampler->sampler_type()) == "top_k", "top_p forces sampling path");
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

int main() {
    test_sampling_params_from_config();
    test_create_sampler_greedy_only_when_sampling_disabled();
    test_top_p_truncates_tail_tokens();
    test_min_p_drops_low_probability_tail();

    if (failures > 0) {
        std::cerr << failures << " sampler test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All sampler tests passed.\n";
    return 0;
}
