#include <iostream>
#include <limits>

#include <torch/torch.h>

#include "cmz_vm2/attention.h"

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Function>
void require_invalid_argument(Function&& function, const char* message) {
    try {
        function();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error(message);
}

}  // namespace

int main() {
    try {
        const auto options = torch::TensorOptions().dtype(torch::kFloat64);
        const auto x = torch::tensor({{1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}}, options);
        const auto eye = torch::eye(2, options);
        const auto mask = torch::zeros({3, 3}, options);

        const auto out = cmz::vm2::self_attention(x, eye, eye, eye, mask);
        require(torch::equal(out.q, x), "Q must equal X under identity Wq");
        require(torch::equal(out.k, x), "K must equal X under identity Wk");
        require(torch::equal(out.v, x), "V must equal X under identity Wv");
        require(
            torch::equal(out.scores, torch::tensor({{1.0, 0.0, 1.0}, {0.0, 1.0, 1.0}, {1.0, 1.0, 2.0}}, options)),
            "scores must be exact QK^T");
        require(
            torch::equal(out.winners, torch::tensor({0, 1, 2}, torch::kInt64)),
            "hardmax must use lowest-index tie break");
        require(torch::equal(out.output, x), "AV must route the hardmax-selected values");

        const auto float_options = torch::TensorOptions().dtype(torch::kFloat32);
        require_invalid_argument(
            [&] {
                const auto float_x = x.to(torch::kFloat32);
                const auto float_eye = torch::eye(2, float_options);
                const auto float_mask = torch::zeros({3, 3}, float_options);
                (void)cmz::vm2::self_attention(float_x, float_eye, float_eye, float_eye, float_mask);
            },
            "float32 inputs must be rejected explicitly");
        require_invalid_argument(
            [&] {
                auto invalid_mask = mask.clone();
                invalid_mask.index_put_({0, 0}, 1.0);
                (void)cmz::vm2::self_attention(x, eye, eye, eye, invalid_mask);
            },
            "positive mask entries must be rejected explicitly");
        require_invalid_argument(
            [&] {
                auto invalid_x = x.clone();
                invalid_x.index_put_({0, 0}, std::numeric_limits<double>::infinity());
                (void)cmz::vm2::self_attention(invalid_x, eye, eye, eye, mask);
            },
            "non-finite inputs must be rejected explicitly");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
