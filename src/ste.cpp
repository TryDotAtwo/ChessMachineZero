#include "cmz/ste.h"

#include <torch/autograd.h>
#include <ATen/ops/_assert_async.h>

#include <cmath>
#include <limits>

namespace cmz {
namespace {

class HardmaxSte final : public torch::autograd::Function<HardmaxSte> {
public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext* context,
        torch::Tensor logits,
        torch::Tensor mask,
        double temperature) {
        TORCH_CHECK(logits.is_floating_point() && logits.dim() > 0 && logits.size(-1) > 0,
                    "hardmax requires floating logits with a nonempty last dimension");
        TORCH_CHECK(mask.scalar_type() == torch::kBool && mask.sizes() == logits.sizes() &&
                    mask.device() == logits.device(), "hardmax mask shape/dtype/device mismatch");
        TORCH_CHECK(std::isfinite(temperature) && temperature > 0.0,
                    "hardmax temperature must be finite and positive");
        // CPU throws immediately; CUDA enqueues an assertion on the current stream.
        // No .item(), CPU transfer, or synchronization belongs in this hot path.
        at::_assert_async(mask.any(-1).all(), "hardmax row must have at least one eligible entry");
        const auto masked = logits.masked_fill(mask.logical_not(), -std::numeric_limits<float>::infinity());
        const auto probabilities = torch::softmax(masked / temperature, -1);
        const auto indices = std::get<1>(masked.max(-1, true));
        auto hard = torch::zeros_like(logits).scatter(-1, indices, 1.0);
        context->save_for_backward({probabilities, mask});
        context->saved_data["temperature"] = temperature;
        return hard;
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* context,
        torch::autograd::variable_list gradients) {
        const auto saved = context->get_saved_variables();
        const auto& probabilities = saved[0];
        const auto& mask = saved[1];
        const auto temperature = context->saved_data["temperature"].toDouble();
        const auto& upstream = gradients[0];
        auto gradient = probabilities *
                        (upstream - (upstream * probabilities).sum(-1, true)) / temperature;
        gradient.masked_fill_(mask.logical_not(), 0.0);
        return {gradient, torch::Tensor(), torch::Tensor()};
    }
};

class Fp4Ste final : public torch::autograd::Function<Fp4Ste> {
public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext*, torch::Tensor values, double scale) {
        TORCH_CHECK(values.is_floating_point(), "FP4 values must use a floating dtype");
        TORCH_CHECK(std::isfinite(scale) && scale > 0.0, "FP4 scale must be finite and positive");
        const auto lattice = torch::tensor(
            {-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0,
             0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0},
            values.options()) * scale;
        // Saturate BEFORE subtraction: finite extremes otherwise round all lattice
        // distances to the same float and argmin incorrectly chooses the first code.
        const auto bounded = values.clamp(-6.0 * scale, 6.0 * scale);
        const auto indices = (bounded.unsqueeze(-1) - lattice).abs().argmin(-1);
        return lattice.index({indices});
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext*, torch::autograd::variable_list gradients) {
        return {gradients[0], torch::Tensor()};
    }
};

}  // namespace

torch::Tensor masked_hardmax_ste(
    const torch::Tensor& logits, const torch::Tensor& mask, double temperature) {
    return HardmaxSte::apply(logits, mask, temperature);
}

torch::Tensor fp4_ste(const torch::Tensor& values, double scale) {
    return Fp4Ste::apply(values, scale);
}

}  // namespace cmz
