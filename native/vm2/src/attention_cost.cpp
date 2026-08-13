#include "cmz_vm2/attention_cost.h"

#include <cmath>
#include <map>
#include <stdexcept>
#include <limits>
#include <set>

namespace cmz::vm2 {

AttentionCostReport analyze_attention_cost(const ProgramImage& image) {
    AttentionCostReport report{{}, 0, 0};
    report.stages.reserve(kStageCount);
    for (std::int64_t stage = 0; stage < kStageCount; ++stage) {
        const auto mask = image.attention_masks[stage].to(torch::kCPU).contiguous();
        if (mask.dim() != 2 || mask.size(0) != mask.size(1)) {
            throw std::invalid_argument("attention cost requires square masks");
        }
        std::map<std::vector<std::int64_t>, std::int64_t> query_groups;
        std::int64_t useful = 0;
        for (std::int64_t query = 0; query < mask.size(0); ++query) {
            std::vector<std::int64_t> keys;
            for (std::int64_t key = 0; key < mask.size(1); ++key) {
                if (std::isfinite(mask.index({query, key}).item<double>())) {
                    keys.push_back(key);
                }
            }
            if (keys.empty()) {
                throw std::invalid_argument("attention query has no finite key");
            }
            useful += static_cast<std::int64_t>(keys.size());
            ++query_groups[keys];
        }
        const auto dense = mask.size(0) * mask.size(1);
        report.stages.push_back(AttentionStageCost{
            stage, mask.size(0), static_cast<std::int64_t>(query_groups.size()), useful, dense});
        report.useful_score_pairs += useful;
        report.dense_score_pairs += dense;
    }
    return report;
}

PaddedAttentionPlan optimize_attention_blocks(
    const torch::Tensor& input_mask, std::int64_t block_budget, double launch_cost) {
    if (block_budget <= 0 || launch_cost < 0.0) {
        throw std::invalid_argument("invalid attention block cost parameters");
    }
    const auto mask = input_mask.to(torch::kCPU).contiguous();
    std::map<std::vector<std::int64_t>, std::vector<std::int64_t>> groups;
    std::int64_t useful = 0;
    for (std::int64_t query = 0; query < mask.size(0); ++query) {
        std::vector<std::int64_t> keys;
        for (std::int64_t key = 0; key < mask.size(1); ++key) {
            if (std::isfinite(mask.index({query, key}).item<double>())) keys.push_back(key);
        }
        useful += static_cast<std::int64_t>(keys.size());
        groups[keys].push_back(query);
    }
    struct Group { std::vector<std::int64_t> queries; std::vector<std::int64_t> keys; };
    std::vector<Group> work;
    for (auto& [keys, queries] : groups) work.push_back(Group{queries, keys});
    const auto padded = [](const Group& group) {
        return static_cast<std::int64_t>(group.queries.size() * group.keys.size());
    };
    while (work.size() > static_cast<std::size_t>(block_budget)) {
        std::size_t best_left = 0, best_right = 1;
        std::int64_t best_delta = std::numeric_limits<std::int64_t>::max();
        for (std::size_t left = 0; left < work.size(); ++left) {
            for (std::size_t right = left + 1; right < work.size(); ++right) {
                std::vector<std::int64_t> keys;
                std::set_union(work[left].keys.begin(), work[left].keys.end(),
                               work[right].keys.begin(), work[right].keys.end(),
                               std::back_inserter(keys));
                const auto merged = static_cast<std::int64_t>(
                    (work[left].queries.size() + work[right].queries.size()) * keys.size());
                const auto delta = merged - padded(work[left]) - padded(work[right]);
                if (delta < best_delta) {
                    best_delta = delta; best_left = left; best_right = right;
                }
            }
        }
        auto& left = work[best_left];
        const auto& right = work[best_right];
        left.queries.insert(left.queries.end(), right.queries.begin(), right.queries.end());
        std::sort(left.queries.begin(), left.queries.end());
        std::vector<std::int64_t> merged_keys;
        std::set_union(left.keys.begin(), left.keys.end(), right.keys.begin(), right.keys.end(),
                       std::back_inserter(merged_keys));
        left.keys = std::move(merged_keys);
        work.erase(work.begin() + static_cast<std::ptrdiff_t>(best_right));
    }
    PaddedAttentionPlan plan{{}, useful, 0, 0.0};
    for (const auto& group : work) {
        auto local_mask = torch::full(
            {static_cast<std::int64_t>(group.queries.size()), static_cast<std::int64_t>(group.keys.size())},
            -std::numeric_limits<double>::infinity(), mask.options());
        for (std::size_t q = 0; q < group.queries.size(); ++q) {
            for (std::size_t k = 0; k < group.keys.size(); ++k) {
                local_mask.index_put_({static_cast<std::int64_t>(q), static_cast<std::int64_t>(k)},
                                      mask.index({group.queries[q], group.keys[k]}));
            }
        }
        plan.padded_score_pairs += static_cast<std::int64_t>(group.queries.size() * group.keys.size());
        plan.blocks.push_back(PaddedAttentionBlock{group.queries, group.keys,
                                                   local_mask.detach().set_requires_grad(false)});
    }
    plan.estimated_cost = static_cast<double>(plan.padded_score_pairs) +
                          launch_cost * static_cast<double>(plan.blocks.size());
    return plan;
}

bool covers_attention_mask_exactly(
    const torch::Tensor& input_mask, const PaddedAttentionPlan& plan) {
    const auto mask = input_mask.to(torch::kCPU).contiguous();
    auto reconstructed = torch::full_like(mask, -std::numeric_limits<double>::infinity());
    std::vector<bool> seen(static_cast<std::size_t>(mask.size(0)), false);
    for (const auto& block : plan.blocks) {
        for (std::size_t q = 0; q < block.queries.size(); ++q) {
            if (seen[static_cast<std::size_t>(block.queries[q])]) return false;
            seen[static_cast<std::size_t>(block.queries[q])] = true;
            for (std::size_t k = 0; k < block.keys.size(); ++k) {
                reconstructed.index_put_({block.queries[q], block.keys[k]},
                                         block.local_mask.index({static_cast<std::int64_t>(q), static_cast<std::int64_t>(k)}));
            }
        }
    }
    return std::all_of(seen.begin(), seen.end(), [](bool value) { return value; }) &&
           torch::equal(mask, reconstructed);
}

std::vector<AttentionBlock> materialize_attention_blocks(
    const PaddedAttentionPlan& plan, std::int64_t token_count,
    const torch::TensorOptions& options) {
    std::vector<AttentionBlock> blocks;
    blocks.reserve(plan.blocks.size());
    for (const auto& source : plan.blocks) {
        auto query_selection = torch::zeros(
            {static_cast<std::int64_t>(source.queries.size()), token_count}, options);
        auto key_selection = torch::zeros(
            {static_cast<std::int64_t>(source.keys.size()), token_count}, options);
        auto global_key_ids = torch::zeros(
            {static_cast<std::int64_t>(source.keys.size()), 1}, options);
        for (std::size_t local = 0; local < source.queries.size(); ++local) {
            query_selection.index_put_(
                {static_cast<std::int64_t>(local), source.queries[local]}, 1.0);
        }
        for (std::size_t local = 0; local < source.keys.size(); ++local) {
            key_selection.index_put_(
                {static_cast<std::int64_t>(local), source.keys[local]}, 1.0);
            global_key_ids.index_put_(
                {static_cast<std::int64_t>(local), 0}, source.keys[local]);
        }
        blocks.push_back(AttentionBlock{
            query_selection.detach().set_requires_grad(false),
            key_selection.detach().set_requires_grad(false),
            source.local_mask.to(options).detach().set_requires_grad(false),
            global_key_ids.detach().set_requires_grad(false)});
    }
    return blocks;
}

}  // namespace cmz::vm2
