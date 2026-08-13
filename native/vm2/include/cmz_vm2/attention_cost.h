#pragma once

#include <cstdint>
#include <vector>

#include "cmz_vm2/schema.h"
#include "cmz_vm2/attention.h"

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

struct PaddedAttentionBlock {
    std::vector<std::int64_t> queries;
    std::vector<std::int64_t> keys;
    torch::Tensor local_mask;
};

struct PaddedAttentionPlan {
    std::vector<PaddedAttentionBlock> blocks;
    std::int64_t useful_score_pairs;
    std::int64_t padded_score_pairs;
    double estimated_cost;
};

AttentionCostReport analyze_attention_cost(const ProgramImage& image);
PaddedAttentionPlan optimize_attention_blocks(
    const torch::Tensor& mask, std::int64_t block_budget, double launch_cost);
bool covers_attention_mask_exactly(
    const torch::Tensor& mask, const PaddedAttentionPlan& plan);
std::vector<AttentionBlock> materialize_attention_blocks(
    const PaddedAttentionPlan& plan, std::int64_t token_count,
    const torch::TensorOptions& options);

}  // namespace cmz::vm2
