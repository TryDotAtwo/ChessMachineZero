#include "cmz/artifact.h"
#include "cmz/vm_module.h"

#include <torch/torch.h>

#include <cstdint>
#include <utility>
#include <vector>

namespace {

cmz::TensorRecord projection(bool identity) {
    constexpr std::size_t width = 128;
    std::vector<std::uint8_t> packed(width * width / 2, 0);
    if (identity) {
        for (std::size_t index = 0; index < width; ++index) {
            const auto flat = index * width + index;
            const auto shift = static_cast<unsigned>((flat & 1U) * 4U);
            packed[flat / 2] |= static_cast<std::uint8_t>(2U << shift);
        }
    }
    return {"projection", {128, 128}, std::move(packed), {1.0F}, 16384};
}

cmz::Artifact graph(bool identity) {
    return cmz::Artifact::from_records(
        {projection(identity)},
        {
            {cmz::OpCode::RowRoute, {0}, {1}, {3, 2048}},
            {cmz::OpCode::TokenProject, {1}, {2}, {0}},
        });
}

}  // namespace

int main() {
    TORCH_CHECK(torch::cuda::is_available(), "CUDA is required");
    const auto options = torch::TensorOptions()
                             .dtype(torch::kFloat32)
                             .device(torch::kCUDA)
                             .requires_grad(true);
    auto input = torch::randn({2, 2048, 128}, options);
    cmz::FrozenVm identity_vm(graph(true), input.device());
    cmz::FrozenVm zero_vm(graph(false), input.device());

    const auto identity_context = identity_vm.forward(input);
    const auto zero_context = zero_vm.forward(input);

    TORCH_CHECK(identity_context.sizes() == torch::IntArrayRef({2, 2045, 128}));
    TORCH_CHECK(torch::equal(identity_context, input.slice(1, 3, 2048)));
    TORCH_CHECK(torch::count_nonzero(zero_context).item<std::int64_t>() == 0);
    identity_context.sum().backward();
    TORCH_CHECK(input.grad().defined());
    TORCH_CHECK(input.grad().slice(1, 0, 3).count_nonzero().item<std::int64_t>() == 0);
    TORCH_CHECK(input.grad().slice(1, 3, 2048).eq(1).all().item<bool>());

    bool rejected_undefined_ssa = false;
    try {
        cmz::FrozenVm invalid(
            cmz::Artifact::from_records(
                {projection(true)},
                {{cmz::OpCode::TokenProject, {99}, {1}, {0}}}),
            input.device());
    } catch (const c10::Error&) {
        rejected_undefined_ssa = true;
    }
    TORCH_CHECK(rejected_undefined_ssa, "undefined SSA input must fail during VM loading");
    return 0;
}
