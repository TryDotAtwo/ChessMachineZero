#include <cmz_vm3/policy.h>
#include <cmz_vm3/st_selector.h>

#include <algorithm>
#include <cmath>

namespace cmz::vm3 {
namespace {

torch::Tensor stable_local_probability(const torch::Tensor& logits,
                                       const torch::Tensor& accept_column) {
  const auto shifted = logits - std::get<0>(logits.max(-1, true));
  const auto weights = torch::exp(shifted);
  const auto probabilities = weights / weights.sum(-1, true);
  return torch::matmul(probabilities, accept_column).squeeze(-1);
}

PolicyOutput route_policy(const PolicyInput& input,
                          const PolicyRouting& routing,
                          const torch::Tensor& candidate_logits,
                          const torch::Tensor& control_logits,
                          double temperature,
                          const torch::Tensor& candidate_q = {},
                          const torch::Tensor& state_k = {},
                          const torch::Tensor& state_v = {},
                          const torch::Tensor& history_k = {},
                          const torch::Tensor& history_v = {}) {
  const auto candidate_probability =
      stable_local_probability(candidate_logits, routing.accept_column);
  const auto control_probability =
      stable_local_probability(control_logits, routing.accept_column);
  const auto running = 1.0 - input.halt_required;
  const auto move_allowed = running * input.move_legal;
  const auto witness_allowed = running * input.claim_after_eligible;
  const auto any_legal = std::get<0>(input.move_legal.max(-1, true));
  const auto any_witness =
      std::get<0>(input.claim_after_eligible.max(-1, true));
  const auto move_scores = running * candidate_probability + 2.0 * move_allowed;
  const auto witness_scores =
      running * candidate_probability + 2.0 * witness_allowed;
  const auto move_routes = torch::matmul(move_scores, routing.move_routes) +
                           torch::matmul(4.0 * (1.0 - running * any_legal),
                                         routing.move_sentinel_route);
  const auto witness_routes =
      torch::matmul(witness_scores, routing.move_routes) +
      torch::matmul(4.0 * (1.0 - running * any_witness),
                    routing.move_sentinel_route);
  const auto move_eligibility =
      torch::matmul(move_allowed, routing.move_routes) +
      torch::matmul(1.0 - running * any_legal, routing.move_sentinel_route);
  const auto witness_eligibility =
      torch::matmul(witness_allowed, routing.move_routes) +
      torch::matmul(1.0 - running * any_witness, routing.move_sentinel_route);
  const auto allowed_features = torch::stack(
      {any_legal.squeeze(-1), input.claim_now.squeeze(-1), any_witness.squeeze(-1)},
      -1);
  const auto control_allowed = running * allowed_features;
  const auto control_scores =
      torch::matmul(running * control_probability + 2.0 * control_allowed,
                    routing.control_routes) +
      torch::matmul(4.0 * input.auto_claim * input.claim_now,
                    routing.claim_now_route) +
      torch::matmul(6.0 * input.halt_required, routing.halt_route);
  const auto control_eligibility =
      torch::matmul(control_allowed, routing.control_routes) +
      torch::matmul(input.halt_required, routing.halt_route);
  return {candidate_logits,
          candidate_probability,
          control_logits,
          control_probability,
          deterministic_st_select(move_routes, move_eligibility, temperature)
              .straight_through,
          deterministic_st_select(witness_routes, witness_eligibility, temperature)
              .straight_through,
          deterministic_st_select(control_scores, control_eligibility, temperature)
              .straight_through,
          candidate_q,
          state_k,
          state_v,
          history_k,
          history_v};
}

torch::Tensor bounded_linear(const torch::Tensor& input,
                             const torch::Tensor& raw_weight) {
  return bounded_parameter(torch::matmul(input, bounded_parameter(raw_weight)));
}

struct AttentionResult {
  torch::Tensor output;
  torch::Tensor q;
  torch::Tensor k;
  torch::Tensor v;
};

AttentionResult bounded_attention(const torch::Tensor& query_input,
                                  const torch::Tensor& key_value_input,
                                  const torch::Tensor& q_raw,
                                  const torch::Tensor& k_raw,
                                  const torch::Tensor& v_raw,
                                  const torch::Tensor& eligibility,
                                  double margin,
                                  double temperature) {
  const auto q = bounded_linear(query_input, q_raw);
  const auto k = bounded_linear(key_value_input, k_raw);
  const auto v = bounded_linear(key_value_input, v_raw);
  const auto scores = torch::matmul(q, k.transpose(-2, -1)) +
                      margin * eligibility.unsqueeze(-2);
  const auto selection = deterministic_st_select(scores, eligibility.unsqueeze(-2),
                                                  temperature).straight_through;
  return {torch::matmul(selection, v), q, k, v};
}

}  // namespace

torch::Tensor bounded_parameter(const torch::Tensor& raw) {
  const auto binary = torch::stack({raw, raw * 0.0}, -1);
  const auto shifted = binary - std::get<0>(binary.max(-1, true));
  const auto weights = torch::exp(shifted);
  const auto probability = weights / weights.sum(-1, true);
  return 2.0 * probability.select(-1, 0) - 1.0;
}

PolicyOutput apply_policy_gate(const PolicyInput& input,
                               const PolicyRouting& routing,
                               const torch::Tensor& candidate_logits,
                               const torch::Tensor& control_logits,
                               double temperature) {
  return route_policy(input, routing, candidate_logits, control_logits,
                      temperature);
}

ZeroPolicy::ZeroPolicy(PolicyRouting routing, double temperature)
    : routing_(std::move(routing)), temperature_(temperature) {
  register_buffer("move_routes", routing_.move_routes);
  register_buffer("move_sentinel_route", routing_.move_sentinel_route);
  register_buffer("control_routes", routing_.control_routes);
  register_buffer("halt_route", routing_.halt_route);
  register_buffer("claim_now_route", routing_.claim_now_route);
  register_buffer("accept_column", routing_.accept_column);
  contract_.certified_accumulator_abs_bound =
      contract_.qk_score_upper + contract_.eligibility_margin;
  operation_manifest_ = {true, true, true, true, true, true, true, true, 1};
}

const BoundedPolicyContract& ZeroPolicy::contract() const { return contract_; }
const PolicyOperationManifest& ZeroPolicy::operation_manifest() const {
  return operation_manifest_;
}

PolicyOutput ZeroPolicy::forward(const PolicyInput& input) {
  const auto candidate_logits = input.candidate_tokens.slice(-1, 0, 2) * 0.0;
  const auto control_logits = input.state_tokens.slice(1, 0, 3).slice(-1, 0, 2) * 0.0;
  return apply_policy_gate(input, routing_, candidate_logits, control_logits,
                           temperature_);
}

PolicyKvSlot ZeroPolicy::encode_move(const MoveSlot& slot) {
  return {slot.move * 0.0, slot.move * 0.0};
}

TransformerPolicy::TransformerPolicy(
    PolicyRouting routing, TransformerPolicyInitialization initialization,
    double temperature)
    : routing_(std::move(routing)),
      initialization_(std::move(initialization)),
      temperature_(temperature) {
  register_buffer("move_routes", routing_.move_routes);
  register_buffer("move_sentinel_route", routing_.move_sentinel_route);
  register_buffer("control_routes", routing_.control_routes);
  register_buffer("halt_route", routing_.halt_route);
  register_buffer("claim_now_route", routing_.claim_now_route);
  register_buffer("accept_column", routing_.accept_column);
  register_buffer("sentinel_key", initialization_.sentinel_key);
  register_buffer("sentinel_value", initialization_.sentinel_value);
  register_parameter("candidate_input_raw", initialization_.candidate_input_raw, true);
  register_parameter("q_raw", initialization_.q_raw, true);
  register_parameter("state_k_raw", initialization_.state_k_raw, true);
  register_parameter("state_v_raw", initialization_.state_v_raw, true);
  register_parameter("state_base_raw", initialization_.state_base_raw, true);
  register_parameter("state_attention_raw", initialization_.state_attention_raw, true);
  register_parameter("history_k_raw", initialization_.history_k_raw, true);
  register_parameter("history_v_raw", initialization_.history_v_raw, true);
  register_parameter("history_base_raw", initialization_.history_base_raw, true);
  register_parameter("history_attention_raw", initialization_.history_attention_raw, true);
  register_parameter("readout_raw", initialization_.readout_raw, true);
  register_parameter("control_query_raw", initialization_.control_query_raw, true);
  contract_.maximum_fan_in = std::max(
      {initialization_.candidate_input_raw.size(0), initialization_.q_raw.size(0),
       initialization_.state_k_raw.size(0), initialization_.state_v_raw.size(0),
       initialization_.state_base_raw.size(0) +
           initialization_.state_attention_raw.size(0),
       initialization_.history_k_raw.size(0),
       initialization_.history_v_raw.size(0),
       initialization_.history_base_raw.size(0) +
           initialization_.history_attention_raw.size(0),
       initialization_.readout_raw.size(0),
       initialization_.control_query_raw.size(0)});
  contract_.accumulator_dtype = initialization_.candidate_input_raw.scalar_type();
  contract_.certified_accumulator_abs_bound =
      std::max(static_cast<double>(contract_.maximum_fan_in),
               contract_.qk_score_upper + contract_.eligibility_margin);
  operation_manifest_ = {true, true, true, true, true, true, true, true,
                         contract_.maximum_fan_in};
}

const BoundedPolicyContract& TransformerPolicy::contract() const {
  return contract_;
}
const PolicyOperationManifest& TransformerPolicy::operation_manifest() const {
  return operation_manifest_;
}

PolicyKvSlot TransformerPolicy::encode_move(const MoveSlot& slot) {
  return {bounded_linear(slot.move, initialization_.history_k_raw),
          bounded_linear(slot.move, initialization_.history_v_raw)};
}

PolicyOutput TransformerPolicy::forward(const PolicyInput& input) {
  auto candidate_h = bounded_linear(input.candidate_tokens,
                                    initialization_.candidate_input_raw);
  const auto state_active = input.state_tokens.select(-1, 0) * 0.0 + 1.0;
  const auto candidate_state = bounded_attention(
      candidate_h, input.state_tokens, initialization_.q_raw,
      initialization_.state_k_raw, initialization_.state_v_raw, state_active,
      0.0, temperature_);
  candidate_h = bounded_parameter(
      torch::matmul(candidate_h, bounded_parameter(initialization_.state_base_raw)) +
      torch::matmul(candidate_state.output,
                    bounded_parameter(initialization_.state_attention_raw)));

  std::vector<torch::Tensor> history_keys;
  std::vector<torch::Tensor> history_values;
  std::vector<torch::Tensor> history_active;
  TORCH_CHECK(input.trajectory.policy_kv_slots.size() ==
                  input.trajectory.move_slots.size() &&
                  !input.trajectory.move_slots.empty(),
              "TransformerPolicy requires fixed matching history slots");
  for (std::size_t slot = 0; slot < input.trajectory.move_slots.size(); ++slot) {
    history_keys.push_back(input.trajectory.policy_kv_slots.at(slot).key);
    history_values.push_back(input.trajectory.policy_kv_slots.at(slot).value);
    history_active.push_back(input.trajectory.move_slots.at(slot).active.squeeze(-1));
  }
  const auto active_stack = torch::stack(history_active, 1);
  const auto any_history = std::get<0>(active_stack.max(-1, true));
  const auto batch = input.state_tokens.size(0);
  history_keys.push_back(initialization_.sentinel_key.view({1, 2}).expand({batch, 2}));
  history_values.push_back(
      initialization_.sentinel_value.view({1, -1}).expand({batch, -1}));
  history_active.push_back((1.0 - any_history).squeeze(-1));
  const auto history_k = torch::stack(history_keys, 1);
  const auto history_v = torch::stack(history_values, 1);
  const auto history_eligibility = torch::stack(history_active, 1);
  const auto candidate_history_output = torch::matmul(
      deterministic_st_select(
          torch::matmul(bounded_linear(candidate_h, initialization_.q_raw),
                        history_k.transpose(-2, -1)) +
              contract_.eligibility_margin * history_eligibility.unsqueeze(-2),
          history_eligibility.unsqueeze(-2), temperature_).straight_through,
      history_v);
  candidate_h = bounded_parameter(
      torch::matmul(candidate_h, bounded_parameter(initialization_.history_base_raw)) +
      torch::matmul(candidate_history_output,
                    bounded_parameter(initialization_.history_attention_raw)));

  const auto control_seed = bounded_parameter(initialization_.control_query_raw)
                                .unsqueeze(0).expand({batch, -1, -1});
  const auto control_state = bounded_attention(
      control_seed, input.state_tokens, initialization_.q_raw,
      initialization_.state_k_raw, initialization_.state_v_raw, state_active,
      0.0, temperature_);
  auto control_h = bounded_parameter(
      torch::matmul(control_seed, bounded_parameter(initialization_.state_base_raw)) +
      torch::matmul(control_state.output,
                    bounded_parameter(initialization_.state_attention_raw)));
  const auto control_history_output = torch::matmul(
      deterministic_st_select(
          torch::matmul(bounded_linear(control_h, initialization_.q_raw),
                        history_k.transpose(-2, -1)) +
              contract_.eligibility_margin * history_eligibility.unsqueeze(-2),
          history_eligibility.unsqueeze(-2), temperature_).straight_through,
      history_v);
  control_h = bounded_parameter(
      torch::matmul(control_h, bounded_parameter(initialization_.history_base_raw)) +
      torch::matmul(control_history_output,
                    bounded_parameter(initialization_.history_attention_raw)));
  const auto candidate_logits =
      torch::matmul(candidate_h, bounded_parameter(initialization_.readout_raw));
  const auto control_logits =
      torch::matmul(control_h, bounded_parameter(initialization_.readout_raw));
  return route_policy(input, routing_, candidate_logits, control_logits,
                      temperature_, candidate_state.q, candidate_state.k,
                      candidate_state.v, history_k, history_v);
}

PolicyPairOutput run_policy_pair(PolicyPair pair,
                                 const PolicyInput& input,
                                 const torch::Tensor& side_route,
                                 const MoveSlot& emitted_slot) {
  const auto first = pair.white.forward(input);
  const auto second = pair.black.forward(input);
  const auto first_kv = pair.white.encode_move(emitted_slot);
  const auto second_kv = pair.black.encode_move(emitted_slot);
  const auto w0_matrix = side_route.select(-1, 0).unsqueeze(-1);
  const auto w1_matrix = side_route.select(-1, 1).unsqueeze(-1);
  const auto w0_tensor = w0_matrix.unsqueeze(-1);
  const auto w1_tensor = w1_matrix.unsqueeze(-1);
  const auto mix_matrix = [&](const torch::Tensor& a, const torch::Tensor& b) {
    return w0_matrix * a + w1_matrix * b;
  };
  const auto mix_tensor = [&](const torch::Tensor& a, const torch::Tensor& b) {
    return w0_tensor * a + w1_tensor * b;
  };
  PolicyOutput selected{
      mix_tensor(first.candidate_logits, second.candidate_logits),
      mix_matrix(first.candidate_probability, second.candidate_probability),
      mix_tensor(first.control_logits, second.control_logits),
      mix_matrix(first.control_probability, second.control_probability),
      mix_matrix(first.move_selection, second.move_selection),
      mix_matrix(first.intended_move_selection, second.intended_move_selection),
      mix_matrix(first.control_selection, second.control_selection),
      mix_tensor(first.candidate_q, second.candidate_q),
      mix_tensor(first.state_k, second.state_k),
      mix_tensor(first.state_v, second.state_v),
      mix_tensor(first.history_k, second.history_k),
      mix_tensor(first.history_v, second.history_v)};
  return {selected,
          {mix_matrix(first_kv.key, second_kv.key),
           mix_matrix(first_kv.value, second_kv.value)}};
}

}  // namespace cmz::vm3
