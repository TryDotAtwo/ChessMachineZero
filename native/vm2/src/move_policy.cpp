#include "cmz_vm2/move_policy.h"

#include <ATen/ops/one_hot.h>

namespace cmz::vm2 {

MovePolicy compile_first_legal_policy(
    std::int64_t candidate_width,
    const torch::TensorOptions& options) {
    return MovePolicy{
        torch::zeros({candidate_width, 1}, options).detach().set_requires_grad(false),
        torch::tensor({{-1.0e9}, {0.0}}, options).detach().set_requires_grad(false)};
}

MovePolicyResult run_move_policy(
    const MovePolicy& policy,
    const torch::Tensor& trace_candidate_embeddings,
    const torch::Tensor& trace_legality_tokens) {
    const auto scores = torch::matmul(
        trace_candidate_embeddings, policy.candidate_projection).squeeze(-1) +
        torch::matmul(trace_legality_tokens, policy.legality_projection).squeeze(-1);
    const auto winner = std::get<1>(scores.max(0, false));
    const auto selection = at::one_hot(winner, scores.size(0)).to(scores.scalar_type());
    return MovePolicyResult{scores, selection};
}

}  // namespace cmz::vm2
