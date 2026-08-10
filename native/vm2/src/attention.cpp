#include "cmz_vm2/attention.h"

#include <ATen/ops/one_hot.h>

namespace cmz::vm2 {
AttentionResult self_attention(
    const torch::Tensor& x,
    const torch::Tensor& wq,
    const torch::Tensor& wk,
    const torch::Tensor& wv,
    const torch::Tensor& mask) {
    const auto q = torch::matmul(x, wq);
    const auto k = torch::matmul(x, wk);
    const auto v = torch::matmul(x, wv);
    const auto scores = torch::matmul(q, k.t()) + mask;
    const auto winners = std::get<1>(scores.max(1, false));
    const auto one_hot = at::one_hot(winners, scores.size(1)).to(x.scalar_type());
    const auto output = torch::matmul(one_hot, v);
    return AttentionResult{q, k, v, scores, winners, output};
}

}  // namespace cmz::vm2
