#include "cmz/ste.h"

#include <torch/torch.h>
#include <iostream>
#include <limits>
#include <string>

int run_tests(int argc, char** argv) {
    if (argc == 2 && (std::string(argv[1]) == "invalid-temperature" || std::string(argv[1]) == "invalid-scale")) {
        const auto logits = torch::ones({2, 3}, torch::kFloat32);
        const auto mask = torch::ones_like(logits, torch::kBool);
        for (double parameter : {0.0, -1.0, std::numeric_limits<double>::quiet_NaN(),
                                 std::numeric_limits<double>::infinity()}) {
            bool rejected = false;
            try {
                if (std::string(argv[1]) == "invalid-temperature")
                    cmz::masked_hardmax_ste(logits, mask, parameter);
                else cmz::fp4_ste(logits, parameter);
            } catch (const c10::Error&) { rejected = true; }
            if (!rejected) {
                std::cerr << "FAIL " << argv[1] << " accepted " << parameter << '\n';
                return 1;
            }
        }
        std::cout << "PASS " << argv[1] << '\n';
        return 0;
    }
    if (argc == 2) {
        const bool cuda = std::string(argv[1]) == "empty-mask-cuda";
        const auto opts = torch::TensorOptions().dtype(torch::kFloat32)
                              .device(cuda ? torch::kCUDA : torch::kCPU);
        // A valid row followed by an empty row catches all-versus-any reduction bugs.
        const auto logits = torch::ones({2, 3}, opts);
        const auto mask = torch::tensor({{true, false, false}, {false, false, false}}, opts.dtype(torch::kBool));
        try {
            cmz::masked_hardmax_ste(logits, mask, 0.5);
            if (cuda) torch::cuda::synchronize(); // Test-only: observe the asynchronous error.
        } catch (const c10::Error& error) {
            const std::string message(error.what());
            TORCH_CHECK(message.find(cuda ? "device-side assert" : "eligible") != std::string::npos,
                        "unexpected failure: ", message);
            std::cout << "PASS empty mask rejected on " << (cuda ? "CUDA" : "CPU") << '\n';
            return 0;
        }
        std::cerr << "FAIL empty mask row was accepted\n";
        return 1;
    }
    TORCH_CHECK(torch::cuda::is_available(), "CUDA is required");
    const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

    auto logits = torch::tensor({{0.0F, 0.0F, 7.0F}}, options).set_requires_grad(true);
    const auto mask = torch::tensor({{true, true, false}}, options.dtype(torch::kBool));
    const auto hard = cmz::masked_hardmax_ste(logits, mask, 0.5);
    hard.backward(torch::tensor({{2.0F, -1.0F, 99.0F}}, options));
    TORCH_CHECK(torch::equal(hard, torch::tensor({{1.0F, 0.0F, 0.0F}}, options)));
    TORCH_CHECK(torch::allclose(
        logits.grad(), torch::tensor({{1.5F, -1.5F, 0.0F}}, options), 0.0, 0.0));

    auto values = torch::tensor({0.49F, 0.76F, -2.4F,
        std::numeric_limits<float>::max(), -std::numeric_limits<float>::max()}, options).set_requires_grad(true);
    const auto quantized = cmz::fp4_ste(values, 1.0);
    quantized.sum().backward();
    TORCH_CHECK(torch::equal(quantized, torch::tensor({0.5F, 1.0F, -2.0F, 6.0F, -6.0F}, options)),
                "FP4 finite extremes must saturate");
    TORCH_CHECK(torch::equal(values.grad(), torch::ones_like(values)));
    std::cout << "PASS hardmax exact masked tie/gradient; FP4 saturation/identity gradient\n";
    return 0;
}

int main(int argc, char** argv) {
    try { return run_tests(argc, argv); }
    catch (const c10::Error& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
