#include <cmz_vm3/policy.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace cmz::vm3 {

PolicyRouting policy_routing_from_program(const FrozenChessProgram& program) {
  return {program.tensors.at("policy_move_routes"),
          program.tensors.at("policy_move_sentinel_route"),
          program.tensors.at("policy_control_routes"),
          program.tensors.at("policy_halt_route"),
          program.tensors.at("policy_claim_now_route"),
          program.tensors.at("policy_accept_column")};
}

void validate_policy_preflight(const PolicyModule& policy) {
  const auto& contract = policy.contract();
  const auto& manifest = policy.operation_manifest();
  TORCH_CHECK(contract.qk_head_dim == kLookupHeadDim,
              "policy Q/K head width is not physically two");
  TORCH_CHECK(contract.coordinatewise_signed_binary_softmax,
              "policy lacks signed-binary parameter bounds");
  TORCH_CHECK(contract.every_projection_rebounded,
              "policy lacks mandatory projection rebounding");
  TORCH_CHECK(contract.local_accept_reject_softmax,
              "policy lacks local ACCEPT/REJECT scoring");
  TORCH_CHECK(contract.eligibility_margin > 4.0,
              "policy internal eligibility margin must exceed four");
  TORCH_CHECK(manifest.signed_binary_parameters && manifest.rebounded_q &&
                  manifest.rebounded_k && manifest.rebounded_v &&
                  manifest.rebounded_output && manifest.rebounded_readout_input &&
                  manifest.local_accept_reject &&
                  manifest.fixed_history_with_tensor_eligibility,
              "policy operation manifest is incomplete");
  TORCH_CHECK(contract.maximum_fixed_margin >= 6.0,
              "policy contract omits a fixed route margin");

  std::int64_t recomputed_fan_in = 1;
  for (const auto& entry : policy.named_parameters()) {
    const auto& parameter = entry.value();
    TORCH_CHECK(torch::isfinite(parameter).all().item<bool>(),
                "policy contains a non-finite raw parameter");
    TORCH_CHECK(parameter.scalar_type() == contract.accumulator_dtype,
                "policy parameter dtype differs from its numeric contract");
    if (parameter.dim() >= 2) {
      recomputed_fan_in = std::max(recomputed_fan_in, parameter.size(-2));
    }
    if (entry.key() == "q_raw" || entry.key() == "state_k_raw" ||
        entry.key() == "history_k_raw") {
      TORCH_CHECK(parameter.dim() == 2 && parameter.size(-1) == kLookupHeadDim,
                  "policy operation manifest contradicts physical Q/K shape");
    }
  }
  recomputed_fan_in =
      std::max(recomputed_fan_in, manifest.maximum_accumulator_fan_in);
  TORCH_CHECK(contract.maximum_fan_in == recomputed_fan_in,
              "policy descriptor fan-in differs from actual parameter shapes");
  TORCH_CHECK(contract.certified_accumulator_abs_bound >=
                  static_cast<double>(recomputed_fan_in),
              "policy accumulator certificate understates fan-in");
  TORCH_CHECK(contract.certified_accumulator_abs_bound >=
                  contract.qk_score_upper + contract.eligibility_margin &&
                  contract.certified_accumulator_abs_bound >=
                      contract.maximum_fixed_margin,
              "policy accumulator certificate omits fixed score margins");
  const auto dtype_limit = contract.accumulator_dtype == torch::kFloat32
                               ? static_cast<double>(std::numeric_limits<float>::max())
                               : std::numeric_limits<double>::max();
  TORCH_CHECK(std::isfinite(contract.certified_accumulator_abs_bound) &&
                  contract.certified_accumulator_abs_bound < dtype_limit / 4.0,
              "policy accumulator certificate is unsafe");
}

}  // namespace cmz::vm3
