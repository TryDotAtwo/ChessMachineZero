#include <iostream>
#include <limits>
#include <vector>
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

        const auto batched_x = torch::stack({x, x.flip(0)});
        const auto batched = cmz::vm2::self_attention(batched_x, eye, eye, eye, mask);
        require(torch::equal(batched.output.index({0}), out.output),
                "batched attention must equal independent inference for batch item zero");
        const auto flipped = cmz::vm2::self_attention(x.flip(0), eye, eye, eye, mask);
        require(torch::equal(batched.output.index({1}), flipped.output),
                "batched attention must equal independent inference for batch item one");

        cmz::vm2::AttentionBlock first{
            torch::tensor({{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}}, options),
            torch::tensor({{1.0, 0.0, 0.0}, {0.0, 0.0, 1.0}}, options),
            torch::zeros({2, 2}, options),
            torch::tensor({{0.0}, {2.0}}, options),
        };
        cmz::vm2::AttentionBlock second{
            torch::tensor({{0.0, 0.0, 1.0}}, options),
            torch::tensor({{0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}, options),
            torch::zeros({1, 2}, options),
            torch::tensor({{1.0}, {2.0}}, options),
        };
        const std::vector<cmz::vm2::AttentionBlock> blocks{first, second};
        auto block_mask = torch::full({3, 3}, -std::numeric_limits<double>::infinity(), options);
        block_mask.index_put_({0, 0}, 0.0);
        block_mask.index_put_({0, 2}, 0.0);
        block_mask.index_put_({1, 0}, 0.0);
        block_mask.index_put_({1, 2}, 0.0);
        block_mask.index_put_({2, 1}, 0.0);
        block_mask.index_put_({2, 2}, 0.0);
        const auto dense_block_oracle = cmz::vm2::self_attention(x, eye, eye, eye, block_mask);
        const auto sparse = cmz::vm2::block_self_attention(x, eye, eye, eye, blocks);
        require(torch::equal(sparse.output, dense_block_oracle.output),
                "frozen block attention output must equal dense attention byte-for-byte");
        require(torch::equal(sparse.winners, dense_block_oracle.winners),
                "frozen block attention must preserve global lowest-index hardmax ties");

        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
