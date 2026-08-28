#include "cmz/vm_module.h"

#include <torch/torch.h>

int main() {
    TORCH_CHECK(torch::cuda::is_available(), "CUDA is required");
    const auto options = torch::TensorOptions()
                             .dtype(torch::kFloat32)
                             .device(torch::kCUDA)
                             .requires_grad(true);
    auto input = torch::randn({2, 2048, 128}, options);
    cmz::FrozenVm module;

    auto context = module.forward(input);

    TORCH_CHECK(context.sizes() == torch::IntArrayRef({2, 2045, 128}));
    TORCH_CHECK(context.device() == input.device());
    TORCH_CHECK(context.requires_grad());
    TORCH_CHECK(torch::equal(context, input.slice(1, 3, 2048)));
    context.sum().backward();
    TORCH_CHECK(input.grad().defined());
    TORCH_CHECK(torch::isfinite(input.grad()).all().item<bool>());
    TORCH_CHECK(input.grad().slice(1, 0, 3).count_nonzero().item<std::int64_t>() == 0);
    TORCH_CHECK(input.grad().slice(1, 3, 2048).eq(1).all().item<bool>());
    return 0;
}
