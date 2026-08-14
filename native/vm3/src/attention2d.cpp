#include <cmz_vm3/attention2d.h>
#include <cmz_vm3/st_selector.h>
#include <cmz_vm3/tensor_contract.h>

#include <ATen/ops/one_hot.h>

#include <vector>

namespace cmz::vm3 {

DenseAttention2dResult dense_attention2d(
    const torch::Tensor& x,
    const torch::Tensor& wq,
    const torch::Tensor& wk,
    const torch::Tensor& wv,
    const torch::Tensor& fixed_mask,
    const torch::Tensor& fixed_eligibility,
    double temperature) {
  TORCH_CHECK(wq.size(-1) == kLookupHeadDim && wk.size(-1) == kLookupHeadDim,
              "VM3 attention requires physical d_head=2");
  const auto q = torch::matmul(x, wq);
  const auto k = torch::matmul(x, wk);
  const auto v = torch::matmul(x, wv);
  const auto scores = torch::matmul(q, k.transpose(-2, -1)) + fixed_mask;
  const auto selection = deterministic_st_select(
      scores, fixed_eligibility, temperature);
  const auto output = torch::matmul(selection.straight_through, v);
  return {{q, k, v, selection.hard_indices, output},
          scores, selection.probabilities, selection.hard};
}

Attention2dOutput block_attention2d(
    const torch::Tensor& x,
    const torch::Tensor& wq,
    const torch::Tensor& wk,
    const torch::Tensor& wv,
    const Attention2dPlan& plan,
    double temperature) {
  TORCH_CHECK(wq.size(-1) == kLookupHeadDim && wk.size(-1) == kLookupHeadDim,
              "VM3 attention requires physical d_head=2");
  TORCH_CHECK(plan.query_group_offsets.size() >= 2,
              "VM3 block plan requires query groups");
  TORCH_CHECK(plan.query_group_offsets.front() == 0 &&
                  plan.query_group_offsets.back() ==
                      static_cast<std::int64_t>(plan.blocks.size()),
              "VM3 block plan offsets must cover every block");

  const auto q = torch::matmul(x, wq);
  const auto k = torch::matmul(x, wk);
  const auto v = torch::matmul(x, wv);
  std::vector<torch::Tensor> routed_outputs;
  std::vector<torch::Tensor> routed_indices;

  for (std::size_t group = 0;
       group + 1 < plan.query_group_offsets.size(); ++group) {
    const auto begin = plan.query_group_offsets[group];
    const auto end = plan.query_group_offsets[group + 1];
    TORCH_CHECK(begin < end, "VM3 query groups must be non-empty");

    std::vector<torch::Tensor> scores;
    std::vector<torch::Tensor> maxima;
    std::vector<torch::Tensor> local_hard;
    std::vector<torch::Tensor> block_values;
    for (auto index = begin; index < end; ++index) {
      const auto& block = plan.blocks.at(static_cast<std::size_t>(index));
      const auto block_q = torch::matmul(block.query_router, q);
      const auto block_k = torch::matmul(block.key_router, k);
      const auto block_v = torch::matmul(block.key_router, v);
      const auto block_scores =
          torch::matmul(block_q, block_k.transpose(-2, -1)) + block.fixed_mask;
      const auto local_index = std::get<1>(block_scores.max(-1, false));
      scores.push_back(block_scores);
      maxima.push_back(std::get<0>(block_scores.max(-1, false)));
      local_hard.push_back(
          at::one_hot(local_index, block_scores.size(-1)).to(x.scalar_type()));
      block_values.push_back(block_v);
    }

    const auto stacked_maxima = torch::stack(maxima, -1);
    const auto block_selection = deterministic_st_select(
        stacked_maxima, torch::ones_like(stacked_maxima), temperature).hard;
    const auto global_max = std::get<0>(stacked_maxima.max(-1, true));
    std::vector<torch::Tensor> unnormalized;
    for (std::size_t local = 0; local < scores.size(); ++local) {
      const auto& block = plan.blocks.at(
          static_cast<std::size_t>(begin) + local);
      unnormalized.push_back(
          torch::exp((scores[local] - global_max) / temperature) *
          block.fixed_eligibility);
    }
    auto denominator = unnormalized.front().sum(-1, true);
    for (std::size_t local = 1; local < unnormalized.size(); ++local) {
      denominator = denominator + unnormalized[local].sum(-1, true);
    }

    std::vector<torch::Tensor> local_outputs;
    std::vector<torch::Tensor> local_indices;
    for (std::size_t local = 0; local < scores.size(); ++local) {
      const auto& block = plan.blocks.at(
          static_cast<std::size_t>(begin) + local);
      const auto probabilities = unnormalized[local] / denominator;
      const auto winner_weight = block_selection.select(-1, local).unsqueeze(-1);
      const auto hard = winner_weight * local_hard[local];
      const auto selection = exact_hard_soft_selection(hard, probabilities);
      local_outputs.push_back(torch::matmul(
          block.output_router,
          torch::matmul(selection, block_values[local])));
      local_indices.push_back(torch::matmul(
          block.output_router,
          torch::matmul(hard, block.global_key_ids.unsqueeze(-1))));
    }
    auto group_output = local_outputs.front();
    auto group_indices = local_indices.front();
    for (std::size_t local = 1; local < local_outputs.size(); ++local) {
      group_output = group_output + local_outputs[local];
      group_indices = group_indices + local_indices[local];
    }
    routed_outputs.push_back(group_output);
    routed_indices.push_back(group_indices);
  }

  auto output = routed_outputs.front();
  auto indices = routed_indices.front();
  for (std::size_t group = 1; group < routed_outputs.size(); ++group) {
    output = output + routed_outputs[group];
    indices = indices + routed_indices[group];
  }
  return {q, k, v, indices.squeeze(-1).to(torch::kInt64), output};
}

}  // namespace cmz::vm3
