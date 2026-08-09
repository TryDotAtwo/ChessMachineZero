#include "cmz_vm2/machine.h"

#include <stdexcept>
#include <vector>

#include "cmz_vm2/attention.h"

namespace cmz::vm2 {
namespace {

void validate_state(const torch::Tensor& state) {
    if (!state.defined() || state.dim() != 2 || state.size(0) != kTokenCount || state.size(1) != kModelDim) {
        throw std::invalid_argument("VM state must have shape [token_count, model_dim]");
    }
    if (state.scalar_type() != torch::kFloat64 || !torch::isfinite(state).all().item<bool>()) {
        throw std::invalid_argument("VM state must contain finite float64 values");
    }
}

}  // namespace

torch::Tensor transition(const ProgramImage& image, const torch::Tensor& initial_state) {
    validate_state(initial_state);
    auto state = initial_state;
    for (std::int64_t stage = 0; stage < kStageCount; ++stage) {
        const auto attention = self_attention(
            state,
            image.weights.wq[stage],
            image.weights.wk[stage],
            image.weights.wv[stage],
            image.attention_masks[stage]);
        const auto projected = torch::matmul(attention.output, image.weights.wo[stage]);
        const auto& write = image.write_masks[stage];
        state = state * (1.0 - write) + projected * write;
    }
    return state;
}

RunResult run(const ProgramImage& image, std::int64_t max_steps) {
    if (max_steps <= 0) {
        throw std::invalid_argument("max_steps must be positive");
    }
    auto state = image.tokens.clone();
    std::vector<torch::Tensor> states{state.clone()};
    for (std::int64_t step = 1; step <= max_steps; ++step) {
        state = transition(image, state);
        states.push_back(state.clone());
        if (state[kControlToken][kHaltOffset].item<double>() == 1.0) {
            return RunResult{state, torch::stack(states), step};
        }
    }
    throw std::runtime_error("VM step limit reached before HALT");
}

}  // namespace cmz::vm2
