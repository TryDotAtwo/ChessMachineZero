#include "cmz_vm2/attention.h"

#include <stdexcept>
#include <string>

#include <ATen/ops/one_hot.h>

namespace cmz::vm2 {
namespace {

void require_matrix(const torch::Tensor& tensor, const char* name) {
    if (!tensor.defined() || tensor.dim() != 2) {
        throw std::invalid_argument(std::string(name) + " must be a rank-2 tensor");
    }
    if (tensor.scalar_type() != torch::kFloat64) {
        throw std::invalid_argument(std::string(name) + " must use float64");
    }
}

void require_finite(const torch::Tensor& tensor, const char* name) {
    if (!torch::isfinite(tensor).all().item<bool>()) {
        throw std::invalid_argument(std::string(name) + " must contain finite values");
    }
}

}  // namespace

AttentionResult self_attention(
    const torch::Tensor& x,
    const torch::Tensor& wq,
    const torch::Tensor& wk,
    const torch::Tensor& wv,
    const torch::Tensor& mask) {
    require_matrix(x, "x");
    require_matrix(wq, "wq");
    require_matrix(wk, "wk");
    require_matrix(wv, "wv");
    require_matrix(mask, "mask");
    require_finite(x, "x");
    require_finite(wq, "wq");
    require_finite(wk, "wk");
    require_finite(wv, "wv");

    const auto model_dim = x.size(1);
    if (wq.size(0) != model_dim || wk.size(0) != model_dim || wv.size(0) != model_dim) {
        throw std::invalid_argument("projection input dimensions must match x");
    }
    if (wq.size(1) != wk.size(1)) {
        throw std::invalid_argument("Q and K dimensions must match");
    }
    if (mask.size(0) != x.size(0) || mask.size(1) != x.size(0)) {
        throw std::invalid_argument("mask must have shape [token_count, token_count]");
    }
    const auto valid_mask = mask.eq(0) | torch::isneginf(mask);
    if (!valid_mask.all().item<bool>()) {
        throw std::invalid_argument("mask entries must be zero or negative infinity");
    }

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
