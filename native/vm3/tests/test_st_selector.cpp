#include <cmz_vm3/st_selector.h>

#include <torch/torch.h>

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
bool require(bool condition, const char* message) {
  if (!condition) std::cerr << message << '\n';
  return condition;
}
}

int main() {
  torch::manual_seed(7);
  bool ok = true;
  const auto base = torch::TensorOptions().dtype(torch::kFloat64);

  auto scores = torch::tensor({{2.0, 2.0, -4.0}}, base.requires_grad(true));
  auto values = torch::tensor({{3.0}, {-2.0}, {5.0}}, base);
  auto out = cmz::vm3::deterministic_st_select(scores, 0.5);
  ok &= require(torch::equal(out.hard, out.straight_through),
                "straight-through forward must be byte-exact hard");
  ok &= require(torch::equal(out.hard, torch::tensor({{1.0, 0.0, 0.0}}, base)),
                "ties must select the lowest index");
  torch::matmul(out.straight_through, values).sum().backward();
  ok &= require(scores.grad().abs().sum().item<double>() > 0.0,
                "hard forward must retain score gradient");

  auto s1 = torch::tensor({{0.5, -0.25, -3.0}}, base.requires_grad(true));
  auto e1 = torch::tensor({{1.0, 0.0, 1.0}}, base.requires_grad(true));
  auto upstream = torch::tensor({{2.0, -1.0, 4.0}}, base);
  auto selected = cmz::vm3::deterministic_st_select(s1, e1, 0.75);
  (selected.straight_through * upstream).sum().backward();
  auto actual_s_grad = s1.grad().clone();
  auto actual_e_grad = e1.grad().clone();

  auto s2 = s1.detach().clone().set_requires_grad(true);
  auto e2 = e1.detach().clone().set_requires_grad(true);
  auto shifted = s2 - std::get<0>(s2.max(-1, true));
  auto unnormalized = torch::exp(shifted / 0.75) * e2;
  auto expected_soft = unnormalized / unnormalized.sum(-1, true);
  (expected_soft * upstream).sum().backward();
  ok &= require(torch::allclose(selected.probabilities, expected_soft.detach(), 1e-12, 0.0),
                "reported probabilities must be the canonical surrogate");
  ok &= require(torch::allclose(actual_s_grad, s2.grad(), 1e-12, 0.0),
                "score gradient must equal soft surrogate gradient");
  ok &= require(torch::allclose(actual_e_grad, e2.grad(), 1e-12, 0.0),
                "eligibility gradient must include the zero boundary");
  ok &= require(torch::isfinite(actual_s_grad).all().item<bool>() &&
                    torch::isfinite(actual_e_grad).all().item<bool>(),
                "selector gradients must remain finite");

  auto sentinel_scores = torch::tensor({{-20.0, -20.0, 4.0}}, base);
  auto sentinel_eligibility = torch::tensor({{0.0, 0.0, 1.0}}, base);
  auto sentinel = cmz::vm3::deterministic_st_select(
      sentinel_scores, sentinel_eligibility, 1.0);
  ok &= require(sentinel.hard_indices.item<std::int64_t>() == 2,
                "no real route must select the certified sentinel row");
  ok &= require(torch::isfinite(sentinel.probabilities).all().item<bool>(),
                "sentinel selector must be finite");

  for (double temperature : {0.0, -1.0,
                             std::numeric_limits<double>::infinity(),
                             std::numeric_limits<double>::quiet_NaN()}) {
    try {
      (void)cmz::vm3::deterministic_st_select(sentinel_scores, temperature);
      ok &= require(false, "invalid temperature must fail closed");
    } catch (const std::exception&) {
    }
  }

  return ok ? 0 : 1;
}
