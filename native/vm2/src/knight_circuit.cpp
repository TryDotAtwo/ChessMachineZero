#include "cmz_vm2/knight_circuit.h"

#include <limits>
#include <stdexcept>

namespace cmz::vm2 {
namespace {

constexpr std::int64_t kDim = 52;
constexpr std::int64_t kQuery = 0;
constexpr std::int64_t kFileRelations = 1;
constexpr std::int64_t kRankRelations = kFileRelations + 64;
constexpr std::int64_t kLegalDefault = kRankRelations + 64;
constexpr std::int64_t kLegal12 = kLegalDefault + 1;
constexpr std::int64_t kLegal21 = kLegal12 + 1;
constexpr std::int64_t kTokenCount = kLegal21 + 1;
constexpr std::int64_t kSourceFile = 0;
constexpr std::int64_t kTargetFile = 8;
constexpr std::int64_t kSourceRank = 16;
constexpr std::int64_t kTargetRank = 24;
constexpr std::int64_t kFileDelta = 32;
constexpr std::int64_t kRankDelta = 41;
constexpr std::int64_t kLegal = 50;
constexpr std::int64_t kConstant = 51;

torch::Tensor frozen(torch::Tensor tensor) {
    return tensor.detach().set_requires_grad(false);
}

torch::Tensor stage_mask(std::int64_t begin, std::int64_t count) {
    const auto options = torch::TensorOptions().dtype(torch::kFloat64);
    auto mask = torch::full(
        {kTokenCount, kTokenCount}, -std::numeric_limits<double>::infinity(), options);
    for (std::int64_t row = 1; row < kTokenCount; ++row) mask.index_put_({row, row}, 0.0);
    for (std::int64_t key = begin; key < begin + count; ++key) mask.index_put_({kQuery, key}, 0.0);
    return frozen(mask);
}

torch::Tensor row_write() {
    auto write = torch::zeros({kTokenCount, kTokenCount}, torch::kFloat64);
    write.index_put_({kQuery, kQuery}, 1.0);
    return frozen(write);
}

torch::Tensor feature_write(std::int64_t begin, std::int64_t count) {
    auto write = torch::zeros({kDim, kDim}, torch::kFloat64);
    for (std::int64_t feature = begin; feature < begin + count; ++feature) {
        write.index_put_({feature, feature}, 1.0);
    }
    return frozen(write);
}

}  // namespace

KnightCircuit compile_knight_circuit() {
    const auto options = torch::TensorOptions().dtype(torch::kFloat64);
    auto tokens = torch::zeros({kTokenCount, kDim}, options);
    tokens.index_put_({kQuery, kConstant}, 1.0);
    for (std::int64_t left = 0; left < 8; ++left) {
        for (std::int64_t right = 0; right < 8; ++right) {
            const auto file_row = kFileRelations + left * 8 + right;
            tokens.index_put_({file_row, kSourceFile + left}, 1.0);
            tokens.index_put_({file_row, kTargetFile + right}, 1.0);
            tokens.index_put_({file_row, kFileDelta + std::abs(left - right)}, 1.0);
            const auto rank_row = kRankRelations + left * 8 + right;
            tokens.index_put_({rank_row, kSourceRank + left}, 1.0);
            tokens.index_put_({rank_row, kTargetRank + right}, 1.0);
            tokens.index_put_({rank_row, kRankDelta + std::abs(left - right)}, 1.0);
        }
    }
    tokens.index_put_({kLegalDefault, kConstant}, 1.0);
    tokens.index_put_({kLegal12, kFileDelta + 1}, 1.0);
    tokens.index_put_({kLegal12, kRankDelta + 2}, 1.0);
    tokens.index_put_({kLegal12, kLegal}, 1.0);
    tokens.index_put_({kLegal21, kFileDelta + 2}, 1.0);
    tokens.index_put_({kLegal21, kRankDelta + 1}, 1.0);
    tokens.index_put_({kLegal21, kLegal}, 1.0);

    std::array<torch::Tensor, kKnightStages> wq;
    std::array<torch::Tensor, kKnightStages> wk;
    std::array<torch::Tensor, kKnightStages> wv;
    for (std::int64_t stage = 0; stage < kKnightStages; ++stage) {
        wq[stage] = torch::zeros({kDim, kDim}, options);
        wk[stage] = torch::zeros({kDim, kDim}, options);
        wv[stage] = torch::eye(kDim, options);
    }
    for (std::int64_t feature = kSourceFile; feature < kTargetFile + 8; ++feature) {
        wq[0].index_put_({feature, feature}, 1.0);
        wk[0].index_put_({feature, feature}, 1.0);
    }
    for (std::int64_t feature = kSourceRank; feature < kTargetRank + 8; ++feature) {
        wq[1].index_put_({feature, feature}, 1.0);
        wk[1].index_put_({feature, feature}, 1.0);
    }
    for (std::int64_t feature = kFileDelta; feature < kRankDelta + 9; ++feature) {
        wq[2].index_put_({feature, feature}, 1.0);
        wk[2].index_put_({feature, feature}, 1.0);
    }
    wq[2].index_put_({kConstant, kConstant}, 1.0);
    wk[2].index_put_({kConstant, kConstant}, 1.5);

    std::array<torch::Tensor, kKnightStages> masks{
        stage_mask(kFileRelations, 64),
        stage_mask(kRankRelations, 64),
        stage_mask(kLegalDefault, 3)};
    std::array<torch::Tensor, kKnightStages> rows{row_write(), row_write(), row_write()};
    std::array<torch::Tensor, kKnightStages> features{
        feature_write(kFileDelta, 9),
        feature_write(kRankDelta, 9),
        feature_write(kLegal, 1)};
    for (std::int64_t stage = 0; stage < kKnightStages; ++stage) {
        wq[stage] = frozen(wq[stage]);
        wk[stage] = frozen(wk[stage]);
        wv[stage] = frozen(wv[stage]);
    }
    auto output_row = torch::zeros({1, kTokenCount}, options);
    output_row.index_put_({0, kQuery}, 1.0);
    auto output_feature = torch::zeros({kDim, 1}, options);
    output_feature.index_put_({kLegal, 0}, 1.0);
    return KnightCircuit{
        frozen(tokens), wq, wk, wv, masks, rows, features,
        frozen(output_row), frozen(output_feature)};
}

torch::Tensor bind_knight_pair(
    const KnightCircuit& circuit, std::int64_t source, std::int64_t target) {
    if (source < 0 || source >= 64 || target < 0 || target >= 64) {
        throw std::out_of_range("knight square outside board");
    }
    auto input = circuit.tokens.clone();
    input.index_put_({kQuery, kSourceFile + source % 8}, 1.0);
    input.index_put_({kQuery, kTargetFile + target % 8}, 1.0);
    input.index_put_({kQuery, kSourceRank + source / 8}, 1.0);
    input.index_put_({kQuery, kTargetRank + target / 8}, 1.0);
    return input;
}

}  // namespace cmz::vm2
