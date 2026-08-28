#pragma once

#include <torch/torch.h>

namespace cmz {

torch::Tensor support_topk_2d_cuda(
    const torch::Tensor& queries,
    const torch::Tensor& keys,
    const torch::Tensor& hull_indices,
    std::int64_t k);

torch::Tensor hull_attention_2d_ste(
    const torch::Tensor& queries,
    const torch::Tensor& keys,
    const torch::Tensor& values,
    const torch::Tensor& hull_indices,
    std::int64_t competitor_count,
    double temperature);

}  // namespace cmz
