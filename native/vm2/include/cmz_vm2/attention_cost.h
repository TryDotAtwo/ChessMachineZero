#pragma once

#include <cstdint>
#include <vector>

#include "cmz_vm2/schema.h"

namespace cmz::vm2 {

struct AttentionStageCost {
    std::int64_t stage;
    std::int64_t token_count;
    std::int64_t block_count;
    std::int64_t useful_score_pairs;
    std::int64_t dense_score_pairs;
};

struct AttentionCostReport {
    std::vector<AttentionStageCost> stages;
    std::int64_t useful_score_pairs;
    std::int64_t dense_score_pairs;
};

AttentionCostReport analyze_attention_cost(const ProgramImage& image);

}  // namespace cmz::vm2
