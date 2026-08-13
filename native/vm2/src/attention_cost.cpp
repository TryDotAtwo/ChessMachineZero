#include "cmz_vm2/attention_cost.h"

#include <cmath>
#include <map>
#include <stdexcept>

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

}  // namespace cmz::vm2
