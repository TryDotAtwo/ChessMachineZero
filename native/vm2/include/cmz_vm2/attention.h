#pragma once

#include <vector>

#include <torch/torch.h>

namespace cmz::vm2 {

struct AttentionResult {
    torch::Tensor q;
    torch::Tensor k;
    torch::Tensor v;
    torch::Tensor scores;
    torch::Tensor winners;
    torch::Tensor output;
};

struct AttentionBlock {
    torch::Tensor query_selection;
    torch::Tensor key_selection;
    torch::Tensor mask;
    torch::Tensor global_key_ids;
};

struct BlockAttentionResult {
    torch::Tensor q;
    torch::Tensor k;
    torch::Tensor v;
    torch::Tensor winners;
    torch::Tensor output;
};

struct CompactAttentionProjection {
    torch::Tensor wq;
    torch::Tensor wk;
    torch::Tensor wv;
    torch::Tensor wo;
    std::int64_t score_width;
};

struct CompactBlockAttentionResult {
    torch::Tensor winners;
    torch::Tensor output;
};

AttentionResult self_attention(
    const torch::Tensor& x,
    const torch::Tensor& wq,
    const torch::Tensor& wk,
    const torch::Tensor& wv,
    const torch::Tensor& mask);

BlockAttentionResult block_self_attention(
    const torch::Tensor& x,
    const torch::Tensor& wq,
    const torch::Tensor& wk,
    const torch::Tensor& wv,
    const std::vector<AttentionBlock>& blocks);

CompactBlockAttentionResult compact_block_self_attention(
    const torch::Tensor& x,
    const CompactAttentionProjection& projection,
    const std::vector<AttentionBlock>& blocks);

}  // namespace cmz::vm2
