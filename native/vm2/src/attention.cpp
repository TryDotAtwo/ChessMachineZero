#include "cmz_vm2/attention.h"

#include <ATen/ops/one_hot.h>

namespace cmz::vm2 {
AttentionResult self_attention(
    const torch::Tensor& x,
    const torch::Tensor& wq,
    const torch::Tensor& wk,
    const torch::Tensor& wv,
    const torch::Tensor& mask) {
    const auto q = torch::matmul(x, wq);
    const auto k = torch::matmul(x, wk);
    const auto v = torch::matmul(x, wv);
    const auto scores = torch::matmul(q, k.transpose(-2, -1)) + mask;
    const auto winners = std::get<1>(scores.max(-1, false));
    const auto one_hot = at::one_hot(winners, scores.size(-1)).to(x.scalar_type());
    const auto output = torch::matmul(one_hot, v);
    return AttentionResult{q, k, v, scores, winners, output};
}

BlockAttentionResult block_self_attention(
    const torch::Tensor& x,
    const torch::Tensor& wq,
    const torch::Tensor& wk,
    const torch::Tensor& wv,
    const std::vector<AttentionBlock>& blocks) {
    const auto q = torch::matmul(x, wq);
    const auto k = torch::matmul(x, wk);
    const auto v = torch::matmul(x, wv);

    const auto evaluate = [&](const AttentionBlock& block) {
        const auto block_q = torch::matmul(block.query_selection, q);
        const auto block_k = torch::matmul(block.key_selection, k);
        const auto block_v = torch::matmul(block.key_selection, v);
        const auto scores = torch::matmul(block_q, block_k.t()) + block.mask;
        const auto local_winners = std::get<1>(scores.max(1, false));
        const auto selection = at::one_hot(local_winners, scores.size(1)).to(x.scalar_type());
        const auto block_output = torch::matmul(selection, block_v);
        const auto global_winners = torch::matmul(selection, block.global_key_ids);
        return std::pair{
            torch::matmul(block.query_selection.t(), block_output),
            torch::matmul(block.query_selection.t(), global_winners)};
    };

    auto routed = evaluate(blocks.front());
    auto output = routed.first;
    auto global_winners = routed.second;
    for (std::size_t block = 1; block < blocks.size(); ++block) {
        routed = evaluate(blocks[block]);
        output = output + routed.first;
        global_winners = global_winners + routed.second;
    }
    return BlockAttentionResult{
        q, k, v, global_winners.squeeze(1).to(torch::kInt64), output};
}

CompactBlockAttentionResult compact_block_self_attention(
    const torch::Tensor& x,
    const CompactAttentionProjection& projection,
    const std::vector<AttentionBlock>& blocks) {
    const auto q = torch::matmul(x, projection.wq);
    const auto k = torch::matmul(x, projection.wk);
    const auto v = torch::matmul(x, projection.wv);

    const auto evaluate = [&](const AttentionBlock& block) {
        const auto block_q = torch::matmul(block.query_selection, q);
        const auto block_k = torch::matmul(block.key_selection, k);
        const auto block_v = torch::matmul(block.key_selection, v);
        const auto scores = torch::matmul(block_q, block_k.t()) + block.mask;
        const auto local_winners = std::get<1>(scores.max(1, false));
        const auto selection = at::one_hot(local_winners, scores.size(1)).to(x.scalar_type());
        return std::pair{
            torch::matmul(block.query_selection.t(), torch::matmul(selection, block_v)),
            torch::matmul(block.query_selection.t(),
                          torch::matmul(selection, block.global_key_ids))};
    };
    auto routed = evaluate(blocks.front());
    auto output = routed.first;
    auto winners = routed.second;
    for (std::size_t block = 1; block < blocks.size(); ++block) {
        routed = evaluate(blocks[block]);
        output = output + routed.first;
        winners = winners + routed.second;
    }
    return CompactBlockAttentionResult{winners.squeeze(1).to(torch::kInt64), output};
}

}  // namespace cmz::vm2
