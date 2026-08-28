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

cmz::TensorRecord query_projection() {
    std::vector<std::uint8_t> packed(128, 0);
    packed[0] = 2U;
    return {"query", {128, 2}, std::move(packed), {1.0F}, 256};
}

cmz::TensorRecord attention_keys() {
    constexpr std::size_t rows = 2045;
    std::vector<std::uint8_t> packed(rows, 0);
    packed[0] = 2U;
    for (std::size_t row = 1; row < rows; ++row) {
        packed[row] = 10U;
    }
    return {"keys", {2045, 2}, std::move(packed), {1.0F}, 4090};
}

cmz::TensorRecord position_bias() {
    std::vector<std::uint8_t> packed(64, 0);
    packed[1] = 6U;
    return {"position", {1, 128}, std::move(packed), {2.0F}, 128};
}

cmz::TensorRecord zero_context_record() {
    constexpr std::size_t elements = 2045 * 128;
    return {"context_0", {2045, 128}, std::vector<std::uint8_t>(elements / 2, 0),
            std::vector<float>((elements + 4095) / 4096, 1.0F), 4096};
}

cmz::Artifact graph(bool identity) {
    return cmz::Artifact::from_records(
        {projection(identity)},
        {
            {cmz::OpCode::RowRoute, {0}, {1}, {3, 2048}},
            {cmz::OpCode::TokenProject, {1}, {2}, {0}},
        });
}

cmz::Artifact attention_graph() {
    std::vector<std::uint32_t> attributes = {1, 2, 500};
    for (std::uint32_t index = 0; index < 2045; ++index) {
        attributes.push_back(index);
    }
    return cmz::Artifact::from_records(
        {query_projection(), attention_keys(), position_bias(), projection(false),
         projection(false), projection(false), projection(true)},
        {
            {cmz::OpCode::RowRoute, {0}, {1}, {3, 2048}},
            {cmz::OpCode::PositionAdd, {1}, {2}, {2}},
            {cmz::OpCode::TokenProject, {2}, {3}, {0}},
            {cmz::OpCode::HullAttention2D, {3, 2}, {4}, std::move(attributes)},
            {cmz::OpCode::ResidualAdd, {2, 4}, {5}, {}},
            {cmz::OpCode::GatedFfn, {5}, {6}, {3, 4, 5}},
            {cmz::OpCode::ResidualAdd, {5, 6}, {7}, {}},
            {cmz::OpCode::OutputProject, {7}, {8}, {6}},
            {cmz::OpCode::HardmaxSte, {8}, {9}, {500}},
        });
}

cmz::Artifact dynamic_attention_graph() {
    std::vector<std::uint32_t> attributes = {2, 500};
    for (std::uint32_t index = 0; index < 2045; ++index) {
        attributes.push_back(index);
    }
    return cmz::Artifact::from_records(
        {query_projection()},
        {
            {cmz::OpCode::RowRoute, {0}, {1}, {3, 2048}},
            {cmz::OpCode::TokenProject, {1}, {2}, {0}},
            {cmz::OpCode::HullAttention2D, {2, 2, 1}, {3}, std::move(attributes)},
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

    auto initial_artifact = graph(true);
    initial_artifact = cmz::Artifact::from_records(
        {projection(true), zero_context_record()}, initial_artifact.operations());
    cmz::FrozenVm initial_vm(std::move(initial_artifact), input.device());
    const auto initial = initial_vm.initial_context(3);
    TORCH_CHECK(initial.sizes() == torch::IntArrayRef({3, 2045, 128}));
    TORCH_CHECK(initial.is_cuda());
    TORCH_CHECK(initial.count_nonzero().item<std::int64_t>() == 0);

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

    auto attention_input = torch::zeros({1, 2048, 128}, options.requires_grad(false));
    attention_input.index_put_({0, torch::indexing::Slice(3, 2048), 0}, 1.0F);
    attention_input.index_put_({0, 3, 1}, 7.0F);
    attention_input.set_requires_grad(true);
    cmz::FrozenVm attention_vm(attention_graph(), attention_input.device());
    const auto attended = attention_vm.forward(attention_input);
    auto expected = torch::zeros_like(attended);
    expected.index_put_({0, torch::indexing::Slice(), 2}, 1.0F);
    TORCH_CHECK(torch::equal(attended, expected));
    attended.index({0, torch::indexing::Slice(), 2}).sum().backward();
    TORCH_CHECK(attention_input.grad().defined());
    TORCH_CHECK(attention_input.grad().abs().sum().item<float>() > 0.0F);

    auto dynamic_input = attention_input.detach().clone().set_requires_grad(true);
    cmz::FrozenVm dynamic_vm(dynamic_attention_graph(), dynamic_input.device());
    const auto dynamic_output = dynamic_vm.forward(dynamic_input);
    const auto dynamic_expected = dynamic_input.index({0, 3})
                                      .reshape({1, 1, 128})
                                      .expand({1, 2045, 128});
    TORCH_CHECK(torch::equal(dynamic_output, dynamic_expected));
    dynamic_output.index({0, torch::indexing::Slice(), 1}).sum().backward();
    TORCH_CHECK(dynamic_input.grad().defined());
    TORCH_CHECK(dynamic_input.grad().abs().sum().item<float>() > 0.0F);
    return 0;
}
