#pragma once

#include <torch/types.h>

namespace cmz {

torch::Tensor support_topk_2d_cuda(
    const torch::Tensor& queries,
    const torch::Tensor& keys,
    const torch::Tensor& hull_indices,
    std::int64_t k);

torch::Tensor support_topk_2d_batched_cuda(
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

torch::Tensor hull_attention_2d_batched_ste(
    const torch::Tensor& queries,
    const torch::Tensor& keys,
    const torch::Tensor& values,
    const torch::Tensor& hull_indices,
    std::int64_t competitor_count,
    double temperature);

torch::Tensor hull_attention_2d_dynamic_batched_ste(
    const torch::Tensor& queries,
    const torch::Tensor& keys,
    const torch::Tensor& values,
    const torch::Tensor& hull_indices,
    std::int64_t competitor_count,
    double temperature);

}  // namespace cmz
