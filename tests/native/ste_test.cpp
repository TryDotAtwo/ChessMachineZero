#include "cmz/ste.h"

#include <torch/torch.h>

int main() {
    TORCH_CHECK(torch::cuda::is_available(), "CUDA is required");
    const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

    auto logits = torch::tensor({{0.0F, 0.0F, 7.0F}}, options).set_requires_grad(true);
    const auto mask = torch::tensor({{true, true, false}}, options.dtype(torch::kBool));
    const auto hard = cmz::masked_hardmax_ste(logits, mask, 0.5);
    hard.backward(torch::tensor({{2.0F, -1.0F, 99.0F}}, options));
    TORCH_CHECK(torch::equal(hard, torch::tensor({{1.0F, 0.0F, 0.0F}}, options)));
    TORCH_CHECK(torch::allclose(
        logits.grad(), torch::tensor({{1.5F, -1.5F, 0.0F}}, options), 0.0, 0.0));

    auto values = torch::tensor({0.49F, 0.76F, -2.4F}, options).set_requires_grad(true);
    const auto quantized = cmz::fp4_ste(values, 1.0);
    quantized.sum().backward();
    TORCH_CHECK(torch::equal(quantized, torch::tensor({0.5F, 1.0F, -2.0F}, options)));
    TORCH_CHECK(torch::equal(values.grad(), torch::ones_like(values)));
    return 0;
}
