#pragma once

#include <cmz_vm3/program_image.h>

namespace cmz::vm3 {

struct Attention2dOutput {
  torch::Tensor q;
  torch::Tensor k;
  torch::Tensor v;
  torch::Tensor hard_indices;
  torch::Tensor output;
};

struct DenseAttention2dResult {
  Attention2dOutput common;
  torch::Tensor scores;
  torch::Tensor probabilities;
  torch::Tensor hard;
};

DenseAttention2dResult dense_attention2d(
    const torch::Tensor& x,
    const torch::Tensor& wq,
    const torch::Tensor& wk,
    const torch::Tensor& wv,
    const torch::Tensor& fixed_mask,
    const torch::Tensor& fixed_eligibility,
    double temperature);

Attention2dOutput block_attention2d(
    const torch::Tensor& x,
    const torch::Tensor& wq,
    const torch::Tensor& wk,
    const torch::Tensor& wv,
    const Attention2dPlan& plan,
    double temperature);

}  // namespace cmz::vm3
