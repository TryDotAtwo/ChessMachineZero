#pragma once

#include <torch/torch.h>

namespace cmz {

torch::Tensor masked_hardmax_ste(
    const torch::Tensor& logits, const torch::Tensor& mask, double temperature);
torch::Tensor fp4_ste(const torch::Tensor& values, double scale);

}  // namespace cmz
