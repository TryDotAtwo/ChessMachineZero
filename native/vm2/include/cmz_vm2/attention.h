#pragma once

#include <torch/torch.h>

namespace cmz::vm2 {

struct AttentionResult {
    torch::Tensor q;
    torch::Tensor k;
    torch::Tensor v;
    torch::Tensor scores;
    torch::Tensor winners;
    torch::Tensor output;
};

AttentionResult self_attention(
    const torch::Tensor& x,
    const torch::Tensor& wq,
    const torch::Tensor& wk,
    const torch::Tensor& wv,
    const torch::Tensor& mask);

}  // namespace cmz::vm2
