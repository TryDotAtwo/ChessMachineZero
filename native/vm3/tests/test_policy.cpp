#include <cmz_vm3/policy.h>
#include <cmz_vm3/tensor_contract.h>

#include <torch/torch.h>

#include <cmath>
#include <iostream>
#include <limits>
#include <string>

namespace {
bool require(bool condition, const char* message) {
  if (!condition) std::cerr << message << '\n';
  return condition;
}

class ContractProbe final : public cmz::vm3::PolicyModule {
 public:
  cmz::vm3::BoundedPolicyContract descriptor;
  cmz::vm3::PolicyOperationManifest manifest{true, true, true, true, true, true,
                                              true, true, 1};
  const cmz::vm3::BoundedPolicyContract& contract() const override {
    return descriptor;
  }
  const cmz::vm3::PolicyOperationManifest& operation_manifest() const override {
    return manifest;
  }
  cmz::vm3::PolicyOutput forward(const cmz::vm3::PolicyInput&) override {
    return {};
  }
  cmz::vm3::PolicyKvSlot encode_move(const cmz::vm3::MoveSlot&) override {
    return {};
  }
};

bool preflight_rejects(ContractProbe& probe) {
  try {
    cmz::vm3::validate_policy_preflight(probe);
  } catch (const c10::Error&) {
    return true;
  }
  return false;
}

cmz::vm3::PolicyInput input_with(const torch::Tensor& legal,
                                 const torch::Tensor& claim_now,
                                 const torch::Tensor& claim_after,
                                 const torch::Tensor& auto_claim,
                                 const torch::Tensor& halt,
                                 const cmz::vm3::TrajectoryTape& trajectory) {
  const auto options = legal.options();
  return {torch::zeros({1, 3, 8}, options),
          torch::zeros({1, cmz::vm3::kMoveCandidateCount, 8}, options),
          legal, claim_now, claim_after, auto_claim, halt, trajectory};
}

cmz::vm3::PolicyRouting routing(const torch::TensorOptions& options) {
  auto move = torch::zeros({cmz::vm3::kMoveCandidateCount,
                            cmz::vm3::kCandidateSelectorRouteCount}, options);
  move.index_put_({torch::indexing::Slice(),
                   torch::indexing::Slice(0, cmz::vm3::kMoveCandidateCount)},
                  torch::eye(cmz::vm3::kMoveCandidateCount, options));
  auto move_sentinel = torch::zeros({1, cmz::vm3::kCandidateSelectorRouteCount}, options);
  move_sentinel.index_put_({0, cmz::vm3::kMoveCandidateCount}, 1.0);
  auto control = torch::zeros({cmz::vm3::kPolicyControlCount,
                               cmz::vm3::kControlRouteCount}, options);
  control.index_put_({torch::indexing::Slice(),
                      torch::indexing::Slice(0, cmz::vm3::kPolicyControlCount)},
                     torch::eye(cmz::vm3::kPolicyControlCount, options));
  auto halt = torch::zeros({1, cmz::vm3::kControlRouteCount}, options);
  halt.index_put_({0, 3}, 1.0);
  auto claim_now = torch::zeros({1, cmz::vm3::kControlRouteCount}, options);
  claim_now.index_put_({0, 1}, 1.0);
  return {move, move_sentinel, control, halt, claim_now,
          torch::tensor({{0.0}, {1.0}}, options)};
}

cmz::vm3::TransformerPolicyInitialization transformer_initialization(
    const torch::TensorOptions& options) {
  auto parameter = [&](std::initializer_list<std::int64_t> shape) {
    return torch::randn(shape, options) * 0.2;
  };
  return {parameter({8, 4}), parameter({4, 2}), parameter({8, 2}),
          parameter({8, 4}), parameter({4, 4}), parameter({4, 4}),
          parameter({8, 2}), parameter({8, 4}), parameter({4, 4}),
          parameter({4, 4}), parameter({4, 2}), parameter({3, 4}),
          torch::zeros({2}, options), torch::zeros({4}, options)};
}
}  // namespace

int main(int argc, char** argv) {
  torch::manual_seed(7);
  bool ok = true;
  const bool use_cuda = argc == 2 && std::string(argv[1]) == "--cuda";
  const auto device = use_cuda ? torch::Device(torch::kCUDA, 0)
                               : torch::Device(torch::kCPU);
  const auto options = torch::TensorOptions().dtype(torch::kFloat64).device(device);

  auto raw = torch::tensor({-1.0e300, 0.0, 1.0e300}, options.requires_grad(true));
  const auto bounded = cmz::vm3::bounded_parameter(raw);
  ok &= require(torch::isfinite(bounded).all().item<bool>(),
                "bounded parameters remain finite for extreme finite raw values");
  ok &= require(bounded.min().item<double>() >= -1.0 &&
                    bounded.max().item<double>() <= 1.0,
                "bounded parameters stay in [-1,1]");
  bounded.sum().backward();
  ok &= require(raw.grad().defined() && torch::isfinite(raw.grad()).all().item<bool>(),
                "bounded parameterization preserves a finite gradient path");

  cmz::vm3::ZeroPolicy zero(routing(options), 0.7);
  const auto contract = zero.contract();
  ok &= require(contract.qk_head_dim == 2 &&
                    contract.coordinatewise_signed_binary_softmax &&
                    contract.every_projection_rebounded &&
                    contract.local_accept_reject_softmax &&
                    contract.eligibility_margin > 4.0,
                "policy publishes the mandatory bounded-attention contract");
  cmz::vm3::validate_policy_preflight(zero);
  ContractProbe probe;
  probe.descriptor.coordinatewise_signed_binary_softmax = false;
  ok &= require(preflight_rejects(probe), "preflight rejects unbounded parameters");
  probe.descriptor = {};
  probe.descriptor.every_projection_rebounded = false;
  ok &= require(preflight_rejects(probe), "preflight rejects missing rebounding");
  probe.descriptor = {};
  probe.descriptor.local_accept_reject_softmax = false;
  ok &= require(preflight_rejects(probe), "preflight rejects a flat policy head");
  probe.descriptor = {};
  probe.descriptor.eligibility_margin = 4.0;
  ok &= require(preflight_rejects(probe), "preflight rejects an insufficient margin");
  probe.descriptor = {};
  probe.descriptor.maximum_fan_in = 2;
  probe.descriptor.certified_accumulator_abs_bound = 2.0;
  ok &= require(preflight_rejects(probe), "preflight recomputes rather than trusts fan-in");
  probe.descriptor = {};
  probe.manifest.rebounded_v = false;
  ok &= require(preflight_rejects(probe), "preflight rejects an incomplete operation manifest");

  const auto legal = at::one_hot(torch::tensor({5}, torch::kLong),
                                 cmz::vm3::kMoveCandidateCount).to(options);
  const auto witness = at::one_hot(torch::tensor({7}, torch::kLong),
                                   cmz::vm3::kMoveCandidateCount).to(options);
  const auto zero_bit = torch::zeros({1, 1}, options);
  const auto one_bit = torch::ones({1, 1}, options);
  const cmz::vm3::TrajectoryTape empty_trajectory{};
  const auto running = input_with(legal, zero_bit, witness, zero_bit, zero_bit,
                                  empty_trajectory);
  const auto first = zero.forward(running);
  ok &= require(first.move_selection.argmax(-1).item<std::int64_t>() == 5,
                "ZeroPolicy chooses the first eligible move");
  ok &= require(first.intended_move_selection.argmax(-1).item<std::int64_t>() == 7,
                "intended selector is absolutely gated by witness eligibility");
  ok &= require(first.control_selection.argmax(-1).item<std::int64_t>() == 0,
                "ordinary running state selects COMMIT_MOVE");
  ok &= require(first.candidate_logits.sizes() ==
                    torch::IntArrayRef({1, cmz::vm3::kMoveCandidateCount, 2}) &&
                    first.candidate_probability.sizes() ==
                    torch::IntArrayRef({1, cmz::vm3::kMoveCandidateCount}),
                "candidate scorer is local ACCEPT/REJECT");

  const auto claim = input_with(legal, one_bit, witness, one_bit, zero_bit,
                                empty_trajectory);
  ok &= require(zero.forward(claim).control_selection.argmax(-1).item<std::int64_t>() == 1,
                "auto-claim bonus forces CLAIM_NOW");

  const auto manual_candidate_logits =
      torch::zeros({1, cmz::vm3::kMoveCandidateCount, 2}, options);
  const auto intended_control_logits = torch::tensor(
      {{{10.0, -10.0}, {10.0, -10.0}, {-10.0, 10.0}}}, options);
  const auto intended = cmz::vm3::apply_policy_gate(
      running, routing(options), manual_candidate_logits,
      intended_control_logits, 0.7);
  ok &= require(intended.control_selection.argmax(-1).item<std::int64_t>() == 2,
                "literal policy can select CLAIM_AFTER_INTENDED_MOVE");

  const auto stopped = input_with(torch::zeros_like(legal), zero_bit,
                                  torch::zeros_like(witness), zero_bit, one_bit,
                                  empty_trajectory);
  const auto halt = zero.forward(stopped);
  ok &= require(halt.move_selection.argmax(-1).item<std::int64_t>() ==
                    cmz::vm3::kMoveCandidateCount &&
                    halt.control_selection.argmax(-1).item<std::int64_t>() == 3,
                "all-illegal terminal state routes to zero-effect sentinels");

  cmz::vm3::TransformerPolicy transformer(
      routing(options), transformer_initialization(options), 0.7);
  cmz::vm3::validate_policy_preflight(transformer);
  ok &= require(transformer.contract().maximum_fan_in == 8 &&
                    transformer.contract().certified_accumulator_abs_bound >= 8.0,
                "certificate counts both summed output-projection fan-ins");
  for (const auto& entry : transformer.named_parameters()) {
    ok &= require(entry.value().requires_grad(), "every trainable policy tensor requires grad");
  }
  for (const auto& entry : transformer.named_buffers()) {
    ok &= require(!entry.value().requires_grad(), "frozen routing tensors never require grad");
  }
  auto move0 = torch::randn({1, 8}, options.requires_grad(true));
  auto move1 = torch::randn({1, 8}, options.requires_grad(true));
  cmz::vm3::TrajectoryTape history;
  history.move_slots = {{move0, one_bit}, {move1, one_bit}};
  history.policy_kv_slots = {transformer.encode_move(history.move_slots[0]),
                             transformer.encode_move(history.move_slots[1])};
  auto candidate_tokens = torch::randn(
      {1, cmz::vm3::kMoveCandidateCount, 8}, options) * 0.2;
  auto state_tokens = torch::randn({1, 5, 8}, options) * 0.2;
  const auto learned_legal =
      legal + at::one_hot(torch::tensor({9}, torch::kLong),
                          cmz::vm3::kMoveCandidateCount).to(options);
  const auto learned_witness =
      witness + at::one_hot(torch::tensor({11}, torch::kLong),
                            cmz::vm3::kMoveCandidateCount).to(options);
  cmz::vm3::PolicyInput learned_input{state_tokens, candidate_tokens,
                                      learned_legal, zero_bit, learned_witness,
                                      zero_bit, zero_bit, history};
  auto learned = transformer.forward(learned_input);
  for (const auto& tensor : {learned.candidate_q, learned.state_k,
                             learned.state_v, learned.history_k,
                             learned.history_v}) {
    ok &= require(torch::isfinite(tensor).all().item<bool>() &&
                      tensor.min().item<double>() >= -1.0 &&
                      tensor.max().item<double>() <= 1.0,
                  "every reusable policy projection is finite and re-bounded");
  }
  ok &= require(learned.candidate_q.size(-1) == 2 &&
                    learned.state_k.size(-1) == 2 &&
                    learned.history_k.size(-1) == 2,
                "every policy Q/K head is physically two-dimensional");

  auto candidate_reward = torch::linspace(
      -1.0, 1.0, cmz::vm3::kCandidateSelectorRouteCount, options).unsqueeze(0);
  auto control_reward = torch::tensor({{0.0, 1.0, -1.0, 0.5}}, options);
  learned.candidate_logits.retain_grad();
  learned.control_logits.retain_grad();
  const auto loss = (learned.move_selection * candidate_reward).sum() +
                    (learned.control_selection * control_reward).sum();
  loss.backward();
  ok &= require(move0.grad().defined() && move1.grad().defined() &&
                    move0.grad().abs().sum().item<double>() > 0.0 &&
                    move1.grad().abs().sum().item<double>() > 0.0,
                "gradient reaches every encoded historical K/V slot");
  ok &= require(learned.candidate_logits.grad().defined() &&
                    learned.candidate_logits.grad().abs().sum().item<double>() > 0.0 &&
                    learned.control_logits.grad().defined() &&
                    learned.control_logits.grad().abs().sum().item<double>() > 0.0,
                "candidate and control local logits receive non-zero gradients");

  const auto permutation = torch::randperm(cmz::vm3::kMoveCandidateCount,
                                            options.dtype(torch::kLong));
  cmz::vm3::TrajectoryTape same_history = history;
  cmz::vm3::PolicyInput permuted_input{
      state_tokens, candidate_tokens.index_select(1, permutation),
      learned_legal.index_select(1, permutation), zero_bit,
      learned_witness.index_select(1, permutation), zero_bit, zero_bit,
      same_history};
  const auto permuted = transformer.forward(permuted_input);
  ok &= require(torch::allclose(permuted.candidate_logits,
                                learned.candidate_logits.index_select(1, permutation),
                                1e-12, 1e-12),
                "shared scorer is candidate-permutation equivariant");

  cmz::vm3::TrajectoryTape padded_history = history;
  auto padded_move = torch::randn({1, 8}, options.requires_grad(true));
  padded_history.move_slots.push_back({padded_move, zero_bit});
  padded_history.policy_kv_slots.push_back(
      transformer.encode_move(padded_history.move_slots.back()));
  cmz::vm3::PolicyInput padded_input{state_tokens, candidate_tokens,
                                     learned_legal, zero_bit, learned_witness,
                                     zero_bit, zero_bit, padded_history};
  const auto padded = transformer.forward(padded_input);
  ok &= require(torch::allclose(padded.candidate_logits, learned.candidate_logits,
                                1e-12, 1e-12) &&
                    torch::equal(padded.move_selection, learned.move_selection),
                "inactive structural history padding has exactly zero forward effect");

  cmz::vm3::TrajectoryTape terminal_history;
  terminal_history.move_slots = {{move0, one_bit}, {move1, one_bit}};
  terminal_history.policy_kv_slots = {
      transformer.encode_move(terminal_history.move_slots[0]),
      transformer.encode_move(terminal_history.move_slots[1])};
  cmz::vm3::PolicyInput terminal_input{
      state_tokens, candidate_tokens, learned_legal, zero_bit, learned_witness,
      zero_bit, one_bit, terminal_history};
  auto terminal = transformer.forward(terminal_input);
  terminal.candidate_logits.retain_grad();
  terminal.control_logits.retain_grad();
  ((terminal.move_selection * candidate_reward).sum() +
   (terminal.control_selection * control_reward).sum()).backward();
  ok &= require(terminal.candidate_logits.grad().defined() &&
                    terminal.candidate_logits.grad().abs().sum().item<double>() == 0.0 &&
                    terminal.control_logits.grad().defined() &&
                    terminal.control_logits.grad().abs().sum().item<double>() == 0.0,
                "post-terminal policy logits have exactly zero transition gradient");

  for (auto& parameter : transformer.parameters()) {
    if (parameter.grad().defined()) parameter.grad().zero_();
  }
  auto base_move0 = move0.detach().clone().set_requires_grad(true);
  auto base_move1 = move1.detach().clone().set_requires_grad(true);
  cmz::vm3::TrajectoryTape base_gradient_history;
  base_gradient_history.move_slots = {{base_move0, one_bit}, {base_move1, one_bit}};
  base_gradient_history.policy_kv_slots = {
      transformer.encode_move(base_gradient_history.move_slots[0]),
      transformer.encode_move(base_gradient_history.move_slots[1])};
  cmz::vm3::PolicyInput base_gradient_input{
      state_tokens, candidate_tokens, learned_legal, zero_bit, learned_witness,
      zero_bit, zero_bit, base_gradient_history};
  const auto base_gradient_output = transformer.forward(base_gradient_input);
  ((base_gradient_output.move_selection * candidate_reward).sum() +
   (base_gradient_output.control_selection * control_reward).sum()).backward();
  std::vector<torch::Tensor> base_parameter_gradients;
  for (const auto& parameter : transformer.parameters()) {
    base_parameter_gradients.push_back(parameter.grad().clone());
  }
  const auto base_move0_gradient = base_move0.grad().clone();
  const auto base_move1_gradient = base_move1.grad().clone();
  for (auto& parameter : transformer.parameters()) parameter.grad().zero_();

  auto pad_move0 = move0.detach().clone().set_requires_grad(true);
  auto pad_move1 = move1.detach().clone().set_requires_grad(true);
  auto inactive_move = torch::randn({1, 8}, options.requires_grad(true));
  cmz::vm3::TrajectoryTape padded_gradient_history;
  padded_gradient_history.move_slots = {
      {pad_move0, one_bit}, {pad_move1, one_bit}, {inactive_move, zero_bit}};
  for (const auto& slot : padded_gradient_history.move_slots) {
    padded_gradient_history.policy_kv_slots.push_back(transformer.encode_move(slot));
  }
  cmz::vm3::PolicyInput padded_gradient_input{
      state_tokens, candidate_tokens, learned_legal, zero_bit, learned_witness,
      zero_bit, zero_bit, padded_gradient_history};
  const auto padded_gradient_output = transformer.forward(padded_gradient_input);
  ((padded_gradient_output.move_selection * candidate_reward).sum() +
   (padded_gradient_output.control_selection * control_reward).sum()).backward();
  ok &= require(torch::allclose(base_move0_gradient, pad_move0.grad(), 1e-12, 1e-12) &&
                    torch::allclose(base_move1_gradient, pad_move1.grad(), 1e-12, 1e-12) &&
                    inactive_move.grad().defined() &&
                    inactive_move.grad().abs().sum().item<double>() == 0.0,
                "inactive padding preserves earlier history gradients exactly");
  std::size_t parameter_index = 0;
  for (const auto& parameter : transformer.parameters()) {
    ok &= require(torch::allclose(base_parameter_gradients.at(parameter_index),
                                  parameter.grad(), 1e-12, 1e-12),
                  "inactive padding preserves every pre-terminal policy gradient");
    ++parameter_index;
  }

  cmz::vm3::TransformerPolicy opponent(
      routing(options), transformer_initialization(options), 0.7);
  const auto side_route = torch::tensor({{1.0, 0.0}}, options);
  const cmz::vm3::PolicyPair pair{transformer, opponent};
  const auto paired = cmz::vm3::run_policy_pair(pair, learned_input, side_route);
  const auto paired_kv = cmz::vm3::encode_policy_pair_move(
      pair, side_route, history.move_slots[0]);
  ok &= require(torch::allclose(paired.selected.candidate_logits,
                                learned.candidate_logits, 1e-12, 1e-12),
                "both policies execute and frozen side routing selects one output");
  ok &= require(torch::allclose(paired_kv.key,
                                transformer.encode_move(history.move_slots[0]).key,
                                1e-12, 1e-12),
                "the same tensor side route selects the emitted policy K/V");

  for (const auto& entry : transformer.named_parameters()) {
    for (const auto dimension : entry.value().sizes()) {
      ok &= require(dimension != cmz::vm3::kMoveCandidateCount,
                    "policy has no candidate-axis parameter or bias");
    }
  }
  {
    torch::NoGradGuard guard;
    for (auto& parameter : transformer.parameters()) {
      parameter.fill_(std::numeric_limits<double>::max());
    }
  }
  cmz::vm3::validate_policy_preflight(transformer);
  cmz::vm3::TrajectoryTape extreme_history;
  extreme_history.move_slots = {{move0, one_bit}, {move1, one_bit}};
  extreme_history.policy_kv_slots = {
      transformer.encode_move(extreme_history.move_slots[0]),
      transformer.encode_move(extreme_history.move_slots[1])};
  cmz::vm3::PolicyInput extreme_input{state_tokens, candidate_tokens,
                                      learned_legal, zero_bit, learned_witness,
                                      zero_bit, zero_bit, extreme_history};
  const auto extreme = transformer.forward(extreme_input);
  ok &= require(torch::isfinite(extreme.candidate_logits).all().item<bool>() &&
                    extreme.move_selection.argmax(-1).item<std::int64_t>() == 5,
                "finite extreme raw parameters cannot defeat the absolute legal gate");

  {
    torch::NoGradGuard guard;
    transformer.parameters().front().fill_(
        std::numeric_limits<double>::infinity());
  }
  bool rejected_non_finite = false;
  try {
    cmz::vm3::validate_policy_preflight(transformer);
  } catch (const c10::Error&) {
    rejected_non_finite = true;
  }
  ok &= require(rejected_non_finite,
                "preflight fails closed before launch on non-finite raw parameters");

  return ok ? 0 : 1;
}
