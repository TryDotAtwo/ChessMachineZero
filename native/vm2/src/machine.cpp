#include "cmz_vm2/machine.h"

#include <vector>

#include "cmz_vm2/attention.h"

namespace cmz::vm2 {
torch::Tensor transition(const ProgramImage& image, const torch::Tensor& initial_state) {
    auto state = initial_state;
    for (std::int64_t stage = 0; stage < kStageCount; ++stage) {
        const auto attention = self_attention(
            state,
            image.weights.wq[stage],
            image.weights.wk[stage],
            image.weights.wv[stage],
            image.attention_masks[stage]);
        const auto projected = torch::matmul(attention.output, image.weights.wo[stage]);
        const auto kept = torch::matmul(state.unsqueeze(1), image.keep_projections[stage]).squeeze(1);
        const auto taken = torch::matmul(projected.unsqueeze(1), image.take_projections[stage]).squeeze(1);
        state = kept + taken;
    }
    return state;
}

RunResult run_fixed(const ProgramImage& image, std::int64_t inference_steps) {
    auto state = image.tokens.clone();
    std::vector<torch::Tensor> states{state.clone()};
    for (std::int64_t step = 0; step < inference_steps; ++step) {
        state = transition(image, state);
        states.push_back(state.clone());
    }
    return RunResult{state, torch::stack(states), inference_steps};
}

}  // namespace cmz::vm2
