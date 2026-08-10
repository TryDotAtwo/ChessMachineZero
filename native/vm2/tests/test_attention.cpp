#include <iostream>
#include <torch/torch.h>

#include "cmz_vm2/attention.h"

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
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

        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
