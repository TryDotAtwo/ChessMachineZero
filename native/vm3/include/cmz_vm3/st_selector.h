#pragma once

#include <torch/torch.h>

namespace cmz::vm3 {

struct StSelection {
  torch::Tensor probabilities;
  torch::Tensor hard_indices;
  torch::Tensor hard;
  torch::Tensor straight_through;
};

StSelection deterministic_st_select(const torch::Tensor& scores,
                                    double temperature);
StSelection deterministic_st_select(const torch::Tensor& scores,
                                    const torch::Tensor& surrogate_eligibility,
                                    double temperature);

}  // namespace cmz::vm3
