#include "cmz/hullkv.h"

#include <torch/torch.h>

int main() {
    TORCH_CHECK(torch::cuda::is_available(), "CUDA is required");
    const auto floats = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    const auto integers = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    auto keys = torch::tensor(
        {{0.0F, 0.0F}, {1.0F, 0.0F}, {0.0F, 1.0F}, {-1.0F, 0.0F},
         {0.0F, -1.0F}, {1.0F, 0.0F}},
        floats).set_requires_grad(true);
    auto queries = torch::tensor(
        {{1.0F, 0.0F}, {1.0F, 1.0F}, {0.0F, 0.0F}}, floats).set_requires_grad(true);
    const auto hull = torch::tensor({0, 1, 2, 3, 4, 5}, integers);

    const auto winners = cmz::support_topk_2d_cuda(queries, keys, hull, 1);
    const auto competitors = cmz::support_topk_2d_cuda(queries, keys, hull, 2);

    TORCH_CHECK(torch::equal(winners, torch::tensor({{1}, {1}, {0}}, integers)));
    TORCH_CHECK(torch::equal(
        competitors, torch::tensor({{1, 5}, {1, 2}, {0, 1}}, integers)));

    auto values = torch::tensor(
        {{0.0F, 0.0F}, {10.0F, 1.0F}, {2.0F, 9.0F}, {-3.0F, 0.0F},
         {0.0F, -4.0F}, {99.0F, 99.0F}},
        floats).set_requires_grad(true);
    const auto routed = cmz::hull_attention_2d_ste(queries, keys, values, hull, 2, 0.5);
    TORCH_CHECK(torch::equal(
        routed, torch::tensor({{10.0F, 1.0F}, {10.0F, 1.0F}, {0.0F, 0.0F}}, floats)));
    routed.sum().backward();
    TORCH_CHECK(torch::isfinite(queries.grad()).all().item<bool>());
    TORCH_CHECK(torch::isfinite(keys.grad()).all().item<bool>());
    TORCH_CHECK(torch::isfinite(values.grad()).all().item<bool>());
    TORCH_CHECK(queries.grad().abs().sum().item<float>() > 0.0F);
    TORCH_CHECK(keys.grad().abs().sum().item<float>() > 0.0F);
    TORCH_CHECK(values.grad().abs().sum().item<float>() > 0.0F);

    auto batched_queries = queries.detach().reshape({1, 3, 2}).repeat({2, 1, 1})
                               .set_requires_grad(true);
    auto batched_values = values.detach().reshape({1, 6, 2}).repeat({2, 1, 1})
                              .set_requires_grad(true);
    const auto batched = cmz::hull_attention_2d_batched_ste(
        batched_queries, keys.detach(), batched_values, hull, 2, 0.5);
    TORCH_CHECK(batched.sizes() == torch::IntArrayRef({2, 3, 2}));
    TORCH_CHECK(torch::equal(
        batched[0],
        torch::tensor({{10.0F, 1.0F}, {10.0F, 1.0F}, {0.0F, 0.0F}}, floats)));
    batched.sum().backward();
    TORCH_CHECK(batched_queries.grad().defined());
    TORCH_CHECK(batched_values.grad().defined());
    TORCH_CHECK(batched_queries.grad().abs().sum().item<float>() > 0.0F);
    TORCH_CHECK(batched_values.grad().abs().sum().item<float>() > 0.0F);
    return 0;
}
