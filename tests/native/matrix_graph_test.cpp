#include "cmz/vm_module.h"
#include <ATen/Context.h>
#include <torch/torch.h>
#include <iostream>

int main() {
    TORCH_CHECK(torch::cuda::is_available(), "CUDA is required");
    at::globalContext().setAllowTF32CuBLAS(false);
    using cmz::OpCode;
    std::vector<std::uint8_t> packed(128, 0);
    packed[0] = 2;
    packed[1] = 32;
    auto artifact = cmz::Artifact::from_records(
        {{"two_columns", {128, 2}, packed, {1.0F}, 256}}, {
            {OpCode::RowRoute, {0}, {1}, {0, 2}},
            {OpCode::TokenProject, {1}, {2}, {0}},
            {static_cast<OpCode>(11), {2}, {3}, {}},
            {static_cast<OpCode>(12), {3}, {4}, {1, 4}},
            {static_cast<OpCode>(12), {4}, {5}, {4, 1}},
            {static_cast<OpCode>(11), {5}, {6}, {}},
            {static_cast<OpCode>(13), {5, 6}, {7}, {}},
        });
    const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto input = torch::zeros({2, 2048, 128}, options);
    input.index_put_({0, torch::indexing::Slice(0, 2), torch::indexing::Slice(0, 2)},
                     torch::tensor({{1.F, 2.F}, {3.F, 4.F}}, options));
    input.index_put_({1, torch::indexing::Slice(0, 2), torch::indexing::Slice(0, 2)},
                     torch::tensor({{2.F, 4.F}, {6.F, 8.F}}, options));
    input.set_requires_grad(true);
    cmz::FrozenVm vm(std::move(artifact), input.device());
    const auto actual = vm.execute_graph(input);
    const auto expected = torch::tensor({
        {{1.F, 3.F, 2.F, 4.F}, {3.F, 9.F, 6.F, 12.F}, {2.F, 6.F, 4.F, 8.F}, {4.F, 12.F, 8.F, 16.F}},
        {{4.F, 12.F, 8.F, 16.F}, {12.F, 36.F, 24.F, 48.F}, {8.F, 24.F, 16.F, 32.F}, {16.F, 48.F, 32.F, 64.F}},
    }, options);
    TORCH_CHECK(torch::equal(actual, expected), "layout/GEMM coordinates or batch isolation");
    actual.sum().backward();
    auto expected_grad = torch::zeros_like(input);
    expected_grad.index_put_({0, torch::indexing::Slice(0, 2), torch::indexing::Slice(0, 2)}, 20.F);
    expected_grad.index_put_({1, torch::indexing::Slice(0, 2), torch::indexing::Slice(0, 2)}, 40.F);
    TORCH_CHECK(torch::equal(input.grad(), expected_grad), "layout/GEMM exact input gradients");
    std::cout << "PASS layout/GEMM literal forward and input gradients, two batches\n";
}
