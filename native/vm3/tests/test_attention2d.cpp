#include <cmz_vm3/attention2d.h>

#include <torch/torch.h>

#include <iostream>

namespace {
bool require(bool condition, const char* message) {
  if (!condition) std::cerr << message << '\n';
  return condition;
}

torch::Tensor leaf(const torch::Tensor& value) {
  return value.clone().set_requires_grad(true);
}
}

int main(int argc, char**) {
  torch::manual_seed(19);
  bool ok = true;
  const bool require_cuda = argc > 1;
  TORCH_CHECK(!require_cuda || torch::cuda::is_available(),
              "CUDA attention2d gate requested without an available GPU");
  const auto device = torch::Device(require_cuda ? torch::kCUDA : torch::kCPU);
  const auto options = torch::TensorOptions().dtype(torch::kFloat64).device(device);

  auto x_value = torch::randn({2, 2, 4, 3}, options);
  auto wq_value = torch::randn({3, 2}, options) * 0.2;
  auto wk_value = torch::randn({3, 2}, options) * 0.2;
  auto wv_value = torch::randn({3, 3}, options) * 0.2;
  auto mask = torch::zeros({4, 4}, options);
  mask.index_put_({1, 3}, -50.0);
  auto eligibility = torch::ones({4, 4}, options);
  eligibility.index_put_({1, 3}, 0.0);
  auto upstream = torch::randn({2, 2, 4, 3}, options);

  auto xd = leaf(x_value), wqd = leaf(wq_value), wkd = leaf(wk_value),
       wvd = leaf(wv_value);
  auto dense = cmz::vm3::dense_attention2d(
      xd, wqd, wkd, wvd, mask, eligibility, 0.7);
  dense.common.q.retain_grad();
  dense.common.k.retain_grad();
  dense.common.v.retain_grad();
  (dense.common.output * upstream).sum().backward();

  auto xb = leaf(x_value), wqb = leaf(wq_value), wkb = leaf(wk_value),
       wvb = leaf(wv_value);
  const auto eye = torch::eye(4, options);
  cmz::vm3::Attention2dPlan plan;
  plan.query_group_offsets = {0, 2};
  plan.blocks.push_back({
      eye,
      eye.index({torch::indexing::Slice(0, 2)}),
      eye,
      mask.index({torch::indexing::Slice(), torch::indexing::Slice(0, 2)}),
      eligibility.index({torch::indexing::Slice(), torch::indexing::Slice(0, 2)}),
      torch::tensor({0.0, 1.0}, options)});
  plan.blocks.push_back({
      eye,
      eye.index({torch::indexing::Slice(2, 4)}),
      eye,
      mask.index({torch::indexing::Slice(), torch::indexing::Slice(2, 4)}),
      eligibility.index({torch::indexing::Slice(), torch::indexing::Slice(2, 4)}),
      torch::tensor({2.0, 3.0}, options)});
  auto block = cmz::vm3::block_attention2d(
      xb, wqb, wkb, wvb, plan, 0.7);
  block.q.retain_grad();
  block.k.retain_grad();
  block.v.retain_grad();
  (block.output * upstream).sum().backward();

  ok &= require(dense.common.q.size(-1) == 2 && dense.common.k.size(-1) == 2,
                "attention heads must be physically two-dimensional");
  ok &= require(torch::equal(dense.common.hard_indices, block.hard_indices),
                "dense and streamed hard winners must match");
  ok &= require(torch::equal(dense.common.output, block.output),
                "dense and streamed hard outputs must be exact");
  ok &= require(torch::allclose(xd.grad(), xb.grad(), 1e-12, 0.0),
                "X gradient parity");
  ok &= require(torch::allclose(wqd.grad(), wqb.grad(), 1e-12, 0.0),
                "Wq gradient parity");
  ok &= require(torch::allclose(wkd.grad(), wkb.grad(), 1e-12, 0.0),
                "Wk gradient parity");
  ok &= require(torch::allclose(wvd.grad(), wvb.grad(), 1e-12, 0.0),
                "Wv hard-gradient parity");
  ok &= require(torch::allclose(dense.common.q.grad(), block.q.grad(), 1e-12, 0.0),
                "Q gradient parity");
  ok &= require(torch::allclose(dense.common.k.grad(), block.k.grad(), 1e-12, 0.0),
                "K gradient parity");
  ok &= require(torch::allclose(dense.common.v.grad(), block.v.grad(), 1e-12, 0.0),
                "V hard-gradient parity");

  auto zeros = torch::zeros_like(x_value);
  auto zero_wq = torch::zeros_like(wq_value);
  auto zero_wk = torch::zeros_like(wk_value);
  auto tie_dense = cmz::vm3::dense_attention2d(
      zeros, zero_wq, zero_wk, wv_value,
      torch::zeros_like(mask), torch::ones_like(eligibility), 1.0);
  auto tie_plan = plan;
  for (auto& entry : tie_plan.blocks) {
    entry.fixed_mask = torch::zeros_like(entry.fixed_mask);
    entry.fixed_eligibility = torch::ones_like(entry.fixed_eligibility);
  }
  auto tie_block = cmz::vm3::block_attention2d(
      zeros, zero_wq, zero_wk, wv_value, tie_plan, 1.0);
  ok &= require(torch::equal(tie_dense.common.hard_indices,
                             torch::zeros_like(tie_dense.common.hard_indices)),
                "dense ties must select global key zero");
  ok &= require(torch::equal(tie_dense.common.hard_indices,
                             tie_block.hard_indices),
                "block ties must preserve global lowest-index ordering");

  try {
    (void)cmz::vm3::dense_attention2d(
        zeros, torch::zeros({3, 3}, options), zero_wk, wv_value,
        mask, eligibility, 1.0);
    ok &= require(false, "non-2D query heads must fail closed");
  } catch (const std::exception&) {
  }

  return ok ? 0 : 1;
}
