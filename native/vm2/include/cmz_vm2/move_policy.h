#pragma once

#include <torch/torch.h>

namespace cmz::vm2 {

struct MovePolicy {
    torch::Tensor candidate_projection;
    torch::Tensor legality_projection;
};

struct MovePolicyResult {
    torch::Tensor scores;
    torch::Tensor selection;
};

MovePolicy compile_first_legal_policy(
    std::int64_t candidate_width,
    const torch::TensorOptions& options = torch::TensorOptions().dtype(torch::kFloat64));

MovePolicyResult run_move_policy(
    const MovePolicy& policy,
    const torch::Tensor& trace_candidate_embeddings,
    const torch::Tensor& trace_legality_tokens);

}  // namespace cmz::vm2
