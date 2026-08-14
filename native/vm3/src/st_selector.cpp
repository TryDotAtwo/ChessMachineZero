#include <cmz_vm3/st_selector.h>

#include <cmath>

namespace cmz::vm3 {
namespace {

// CMZ_VM3_REGISTERED_AUTOGRAD: ExactHardSoftBackward
class ExactHardSoftBackward final
    : public torch::autograd::Function<ExactHardSoftBackward> {
 public:
  static torch::Tensor forward(torch::autograd::AutogradContext*,
                               torch::Tensor hard_forward,
                               torch::Tensor soft_surrogate) {
    (void)soft_surrogate;
    return hard_forward.clone();
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext*,
      torch::autograd::variable_list upstream) {
    return {torch::Tensor(), upstream.at(0)};
  }
};

void validate_temperature(double temperature) {
  TORCH_CHECK(temperature > 0.0 && std::isfinite(temperature),
              "VM3 selector temperature must be finite and positive");
}

}  // namespace

torch::Tensor exact_hard_soft_selection(const torch::Tensor& hard,
                                        const torch::Tensor& soft) {
  TORCH_CHECK(hard.sizes() == soft.sizes(),
              "VM3 hard and soft selections must have identical shapes");
  return ExactHardSoftBackward::apply(hard, soft);
}

StSelection deterministic_st_select(const torch::Tensor& scores,
                                    double temperature) {
  return deterministic_st_select(scores, torch::ones_like(scores), temperature);
}

StSelection deterministic_st_select(const torch::Tensor& scores,
                                    const torch::Tensor& surrogate_eligibility,
                                    double temperature) {
  validate_temperature(temperature);
  const auto shifted = scores - std::get<0>(scores.max(-1, true));
  const auto unnormalized = torch::exp(shifted / temperature) * surrogate_eligibility;
  const auto probabilities = unnormalized / unnormalized.sum(-1, true);
  const auto hard_indices = std::get<1>(scores.max(-1, false));
  const auto hard = at::one_hot(hard_indices, scores.size(-1)).to(scores.scalar_type());
  const auto straight_through = exact_hard_soft_selection(hard, probabilities);
  return {probabilities, hard_indices, hard, straight_through};
}

}  // namespace cmz::vm3
