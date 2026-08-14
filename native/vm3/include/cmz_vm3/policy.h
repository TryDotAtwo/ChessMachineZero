#pragma once

#include <cmz_vm3/chess_state.h>
#include <cmz_vm3/tensor_contract.h>

#include <torch/torch.h>

#include <cstdint>

namespace cmz::vm3 {

struct PolicyInput {
  torch::Tensor state_tokens;
  torch::Tensor candidate_tokens;
  torch::Tensor move_legal;
  torch::Tensor claim_now;
  torch::Tensor claim_after_eligible;
  torch::Tensor auto_claim;
  torch::Tensor halt_required;
  const TrajectoryTape& trajectory;
};

struct PolicyOutput {
  torch::Tensor candidate_logits;
  torch::Tensor candidate_probability;
  torch::Tensor control_logits;
  torch::Tensor control_probability;
  torch::Tensor move_selection;
  torch::Tensor intended_move_selection;
  torch::Tensor control_selection;
  torch::Tensor candidate_q;
  torch::Tensor state_k;
  torch::Tensor state_v;
  torch::Tensor history_k;
  torch::Tensor history_v;
};

struct BoundedPolicyContract {
  std::int64_t qk_head_dim = kLookupHeadDim;
  double parameter_lower = -1.0;
  double parameter_upper = 1.0;
  double activation_lower = -1.0;
  double activation_upper = 1.0;
  double qk_score_lower = -2.0;
  double qk_score_upper = 2.0;
  double eligibility_margin = 5.0;
  double maximum_fixed_margin = 6.0;
  std::int64_t maximum_fan_in = 1;
  c10::ScalarType accumulator_dtype = torch::kFloat64;
  double certified_accumulator_abs_bound = 1.0;
  bool coordinatewise_signed_binary_softmax = true;
  bool every_projection_rebounded = true;
  bool local_accept_reject_softmax = true;
};

struct PolicyOperationManifest {
  bool signed_binary_parameters = false;
  bool rebounded_q = false;
  bool rebounded_k = false;
  bool rebounded_v = false;
  bool rebounded_output = false;
  bool rebounded_readout_input = false;
  bool local_accept_reject = false;
  bool fixed_history_with_tensor_eligibility = false;
  std::int64_t maximum_accumulator_fan_in = 0;
};

struct PolicyRouting {
  torch::Tensor move_routes;
  torch::Tensor move_sentinel_route;
  torch::Tensor control_routes;
  torch::Tensor halt_route;
  torch::Tensor claim_now_route;
  torch::Tensor accept_column;
};

PolicyRouting policy_routing_from_program(const FrozenChessProgram& program);

PolicyOutput apply_policy_gate(const PolicyInput& input,
                               const PolicyRouting& routing,
                               const torch::Tensor& candidate_logits,
                               const torch::Tensor& control_logits,
                               double temperature);

class PolicyModule : public torch::nn::Module {
 public:
  virtual ~PolicyModule() = default;
  virtual const BoundedPolicyContract& contract() const = 0;
  virtual const PolicyOperationManifest& operation_manifest() const = 0;
  virtual PolicyOutput forward(const PolicyInput& input) = 0;
  virtual PolicyKvSlot encode_move(const MoveSlot& slot) = 0;
};

torch::Tensor bounded_parameter(const torch::Tensor& raw);
void validate_policy_preflight(const PolicyModule& policy);

class ZeroPolicy final : public PolicyModule {
 public:
  explicit ZeroPolicy(PolicyRouting routing, double temperature = 1.0);
  const BoundedPolicyContract& contract() const override;
  const PolicyOperationManifest& operation_manifest() const override;
  PolicyOutput forward(const PolicyInput& input) override;
  PolicyKvSlot encode_move(const MoveSlot& slot) override;

 private:
  PolicyRouting routing_;
  double temperature_;
  BoundedPolicyContract contract_;
  PolicyOperationManifest operation_manifest_;
};

struct TransformerPolicyInitialization {
  torch::Tensor candidate_input_raw;
  torch::Tensor q_raw;
  torch::Tensor state_k_raw;
  torch::Tensor state_v_raw;
  torch::Tensor state_base_raw;
  torch::Tensor state_attention_raw;
  torch::Tensor history_k_raw;
  torch::Tensor history_v_raw;
  torch::Tensor history_base_raw;
  torch::Tensor history_attention_raw;
  torch::Tensor readout_raw;
  torch::Tensor control_query_raw;
  torch::Tensor sentinel_key;
  torch::Tensor sentinel_value;
};

class TransformerPolicy final : public PolicyModule {
 public:
  TransformerPolicy(PolicyRouting routing,
                    TransformerPolicyInitialization initialization,
                    double temperature = 1.0);
  const BoundedPolicyContract& contract() const override;
  const PolicyOperationManifest& operation_manifest() const override;
  PolicyOutput forward(const PolicyInput& input) override;
  PolicyKvSlot encode_move(const MoveSlot& slot) override;

 private:
  PolicyRouting routing_;
  TransformerPolicyInitialization initialization_;
  double temperature_;
  BoundedPolicyContract contract_;
  PolicyOperationManifest operation_manifest_;
};

struct PolicyPair {
  PolicyModule& white;
  PolicyModule& black;
};

struct PolicyPairOutput {
  PolicyOutput selected;
};

PolicyPairOutput run_policy_pair(PolicyPair pair,
                                 const PolicyInput& input,
                                 const torch::Tensor& side_route);
PolicyKvSlot encode_policy_pair_move(PolicyPair pair,
                                     const torch::Tensor& side_route,
                                     const MoveSlot& emitted_slot);

}  // namespace cmz::vm3
