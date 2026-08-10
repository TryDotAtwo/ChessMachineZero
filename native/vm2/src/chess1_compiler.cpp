#include "cmz_vm2/chess1_compiler.h"

#include <limits>
#include <stdexcept>

namespace cmz::vm2 {
namespace {

constexpr std::int64_t kSourcePieceOffset = 0;
constexpr std::int64_t kTargetPieceOffset = 3;
constexpr std::int64_t kSideOffset = 6;
constexpr std::int64_t kSquareOffset = 8;
constexpr std::int64_t kTargetOffset = 72;
constexpr std::int64_t kSourceValidOffset = 137;
constexpr std::int64_t kTargetOnboardOffset = 139;
constexpr std::int64_t kSideBaselineFeature = 205;
constexpr std::int64_t kSideQueryFeature = 206;

void one_hot(torch::Tensor& tokens, std::int64_t row, std::int64_t offset, std::int64_t value) {
    tokens.index_put_({row, offset + value}, 1.0);
}

void weight(torch::Tensor& matrix, std::int64_t input, std::int64_t output) {
    matrix.index_put_({input, output}, 1.0);
}

void copy_range(torch::Tensor& matrix,
                std::int64_t input_offset,
                std::int64_t output_offset,
                std::int64_t count) {
    for (std::int64_t value = 0; value < count; ++value) {
        weight(matrix, input_offset + value, output_offset + value);
    }
}

void enable_write(WriteProjections& projections,
                  std::int64_t stage,
                  std::int64_t row_begin,
                  std::int64_t row_end,
                  std::int64_t feature_begin,
                  std::int64_t feature_end) {
    for (std::int64_t row = row_begin; row < row_end; ++row) {
        projections.row[stage][0].index_put_({row, row}, 1.0);
    }
    for (std::int64_t feature = feature_begin; feature < feature_end; ++feature) {
        projections.feature[stage][0].index_put_({feature, feature}, 1.0);
    }
}

}  // namespace

ProgramImage compile_chess1(const Chess1Board& board, std::int64_t selected_source) {
    if (selected_source < 0 || selected_source >= chess1::kCandidateTokenCount) {
        throw std::invalid_argument("selected source is outside 0..63");
    }

    const auto options = torch::TensorOptions().dtype(torch::kFloat64);
    auto tokens = torch::zeros({chess1::kTokenCount, kModelDim}, options);
    FrozenWeights weights;
    WriteProjections projections;
    std::array<torch::Tensor, kStageCount> masks;
    const auto negative_infinity = -std::numeric_limits<double>::infinity();

    for (std::int64_t stage = 0; stage < kStageCount; ++stage) {
        weights.wq[stage] = torch::zeros({kModelDim, kModelDim}, options);
        weights.wk[stage] = torch::zeros({kModelDim, kModelDim}, options);
        weights.wv[stage] = torch::zeros({kModelDim, kModelDim}, options);
        weights.wo[stage] = torch::eye(kModelDim, options);
        masks[stage] = torch::full(
            {chess1::kTokenCount, chess1::kTokenCount}, negative_infinity, options);
        for (std::int64_t row = 0; row < chess1::kTokenCount; ++row) {
            masks[stage].index_put_({row, row}, 0.0);
        }
        for (std::int64_t component = 0; component < kWriteProjectionComponents; ++component) {
            projections.row[stage][component] =
                torch::zeros({chess1::kTokenCount, chess1::kTokenCount}, options);
            projections.feature[stage][component] = torch::zeros({kModelDim, kModelDim}, options);
        }
    }

    one_hot(tokens, chess1::kSideToken, kSideOffset, static_cast<std::int64_t>(board.side));
    for (std::int64_t square = 0; square < 64; ++square) {
        const auto square_row = chess1::kSquareTokenBegin + square;
        one_hot(tokens, square_row, kSourcePieceOffset,
                static_cast<std::int64_t>(board.squares[square]));
        one_hot(tokens, square_row, kSquareOffset, square);

        const auto candidate_row = chess1::kCandidateTokenBegin + square;
        one_hot(tokens, candidate_row, kSquareOffset, square);

        const auto geometry_row = chess1::kGeometryTokenBegin + square;
        const bool valid = square >= 8 && square < 56;
        one_hot(tokens, geometry_row, kSquareOffset, square);
        one_hot(tokens, geometry_row, kTargetOffset, valid ? square + 8 : 64);
        one_hot(tokens, geometry_row, kSourceValidOffset, valid ? 1 : 0);
        one_hot(tokens, geometry_row, kTargetOnboardOffset, valid ? 1 : 0);
    }
    one_hot(tokens, chess1::kOffboardToken, kSourcePieceOffset,
            static_cast<std::int64_t>(Chess1Piece::Other));
    one_hot(tokens, chess1::kOffboardToken, kTargetOffset, 64);
    one_hot(tokens, chess1::kSideToken, kSideBaselineFeature, 0);
    one_hot(tokens, chess1::kSelectedToken, kSideOffset,
            static_cast<std::int64_t>(Chess1Side::Black));
    one_hot(tokens, chess1::kSourceWriteToken, kSourcePieceOffset,
            static_cast<std::int64_t>(Chess1Piece::Empty));
    one_hot(tokens, chess1::kTargetWriteToken, kSourcePieceOffset,
            static_cast<std::int64_t>(Chess1Piece::WhitePawn));
    for (std::int64_t square = 0; square < 64; ++square) {
        one_hot(tokens, chess1::kOutputSquareTokenBegin + square, kSquareOffset, square);
        one_hot(tokens, chess1::kOutputSquareTokenBegin + square, chess1::kLegalOffset, 1);
    }
    one_hot(tokens, chess1::kOutputSideToken, kSideQueryFeature, 0);

    std::int64_t relation_row = chess1::kLegalityRelationTokenBegin;
    for (std::int64_t source_piece = 0; source_piece < 3; ++source_piece) {
        for (std::int64_t side = 0; side < 2; ++side) {
            for (std::int64_t valid = 0; valid < 2; ++valid) {
                for (std::int64_t onboard = 0; onboard < 2; ++onboard) {
                    for (std::int64_t target_piece = 0; target_piece < 3; ++target_piece) {
                        one_hot(tokens, relation_row, kSourcePieceOffset, source_piece);
                        one_hot(tokens, relation_row, kTargetPieceOffset, target_piece);
                        one_hot(tokens, relation_row, kSideOffset, side);
                        one_hot(tokens, relation_row, kSourceValidOffset, valid);
                        one_hot(tokens, relation_row, kTargetOnboardOffset, onboard);
                        const bool legal =
                            source_piece == static_cast<std::int64_t>(Chess1Piece::WhitePawn) &&
                            side == static_cast<std::int64_t>(Chess1Side::White) && valid == 1 &&
                            onboard == 1 &&
                            target_piece == static_cast<std::int64_t>(Chess1Piece::Empty);
                        one_hot(tokens, relation_row, chess1::kLegalOffset, legal ? 1 : 0);
                        ++relation_row;
                    }
                }
            }
        }
    }

    // Stages 0..3 route source piece, geometry, target piece and side into every candidate.
    copy_range(weights.wv[0], kSourcePieceOffset, kSourcePieceOffset, 3);
    copy_range(weights.wv[1], kTargetOffset, kTargetOffset, 65);
    copy_range(weights.wv[1], kSourceValidOffset, kSourceValidOffset, 2);
    copy_range(weights.wv[1], kTargetOnboardOffset, kTargetOnboardOffset, 2);
    copy_range(weights.wv[2], kSourcePieceOffset, kTargetPieceOffset, 3);
    copy_range(weights.wv[3], kSideOffset, kSideOffset, 2);
    for (std::int64_t square = 0; square < 64; ++square) {
        const auto candidate = chess1::kCandidateTokenBegin + square;
        masks[0].index_put_({candidate, candidate}, negative_infinity);
        masks[0].index_put_({candidate, chess1::kSquareTokenBegin + square}, 0.0);
        masks[1].index_put_({candidate, candidate}, negative_infinity);
        masks[1].index_put_({candidate, chess1::kGeometryTokenBegin + square}, 0.0);
        masks[2].index_put_({candidate, candidate}, negative_infinity);
        const auto target = square >= 8 && square < 56 ? chess1::kSquareTokenBegin + square + 8
                                                       : chess1::kOffboardToken;
        masks[2].index_put_({candidate, target}, 0.0);
        masks[3].index_put_({candidate, candidate}, negative_infinity);
        masks[3].index_put_({candidate, chess1::kSideToken}, 0.0);
    }
    enable_write(projections, 0, chess1::kCandidateTokenBegin,
                 chess1::kCandidateTokenBegin + 64, kSourcePieceOffset, kSourcePieceOffset + 3);
    enable_write(projections, 1, chess1::kCandidateTokenBegin,
                 chess1::kCandidateTokenBegin + 64, kTargetOffset, kTargetOffset + 65);
    enable_write(projections, 1, chess1::kCandidateTokenBegin,
                 chess1::kCandidateTokenBegin + 64, kSourceValidOffset, kTargetOnboardOffset + 2);
    enable_write(projections, 2, chess1::kCandidateTokenBegin,
                 chess1::kCandidateTokenBegin + 64, kTargetPieceOffset, kTargetPieceOffset + 3);
    enable_write(projections, 3, chess1::kCandidateTokenBegin,
                 chess1::kCandidateTokenBegin + 64, kSideOffset, kSideOffset + 2);

    // Stage 4 performs an exact five-field relation lookup with QK^T + hardmax.
    std::int64_t score_feature = 0;
    const auto score_category = [&](std::int64_t offset, std::int64_t count) {
        for (std::int64_t value = 0; value < count; ++value) {
            weight(weights.wq[4], offset + value, score_feature + value);
            weight(weights.wk[4], offset + value, score_feature + value);
        }
        score_feature += count;
    };
    score_category(kSourcePieceOffset, 3);
    score_category(kTargetPieceOffset, 3);
    score_category(kSideOffset, 2);
    score_category(kSourceValidOffset, 2);
    score_category(kTargetOnboardOffset, 2);
    copy_range(weights.wv[4], chess1::kLegalOffset, chess1::kLegalOffset, 2);
    for (std::int64_t square = 0; square < 64; ++square) {
        const auto candidate = chess1::kCandidateTokenBegin + square;
        masks[4].index_put_({candidate, candidate}, negative_infinity);
        for (std::int64_t relation = 0; relation < chess1::kLegalityRelationTokenCount;
             ++relation) {
            masks[4].index_put_({candidate, chess1::kLegalityRelationTokenBegin + relation}, 0.0);
        }
    }
    enable_write(projections, 4, chess1::kCandidateTokenBegin,
                 chess1::kCandidateTokenBegin + 64, chess1::kLegalOffset,
                 chess1::kLegalOffset + 2);

    // Stage 5 selects one candidate token; stages 6 and 7 materialize two write tokens.
    copy_range(weights.wv[5], kSquareOffset, kSquareOffset, 64);
    copy_range(weights.wv[5], kTargetOffset, kTargetOffset, 65);
    copy_range(weights.wv[5], chess1::kLegalOffset, chess1::kLegalOffset, 2);
    masks[5].index_put_({chess1::kSelectedToken, chess1::kSelectedToken}, negative_infinity);
    masks[5].index_put_({chess1::kSelectedToken,
                         chess1::kCandidateTokenBegin + selected_source}, 0.0);
    enable_write(projections, 5, chess1::kSelectedToken, chess1::kSelectedToken + 1,
                 kSquareOffset, kTargetOffset + 65);
    enable_write(projections, 5, chess1::kSelectedToken, chess1::kSelectedToken + 1,
                 chess1::kLegalOffset, chess1::kLegalOffset + 2);

    copy_range(weights.wv[6], kSquareOffset, kSquareOffset, 64);
    copy_range(weights.wv[6], chess1::kLegalOffset, chess1::kLegalOffset, 2);
    masks[6].index_put_({chess1::kSourceWriteToken, chess1::kSourceWriteToken},
                        negative_infinity);
    masks[6].index_put_({chess1::kSourceWriteToken, chess1::kSelectedToken}, 0.0);
    enable_write(projections, 6, chess1::kSourceWriteToken, chess1::kSourceWriteToken + 1,
                 kSquareOffset, kSquareOffset + 64);
    enable_write(projections, 6, chess1::kSourceWriteToken, chess1::kSourceWriteToken + 1,
                 chess1::kLegalOffset, chess1::kLegalOffset + 2);

    copy_range(weights.wv[7], kTargetOffset, kSquareOffset, 64);
    copy_range(weights.wv[7], chess1::kLegalOffset, chess1::kLegalOffset, 2);
    masks[7].index_put_({chess1::kTargetWriteToken, chess1::kTargetWriteToken},
                        negative_infinity);
    masks[7].index_put_({chess1::kTargetWriteToken, chess1::kSelectedToken}, 0.0);
    enable_write(projections, 7, chess1::kTargetWriteToken, chess1::kTargetWriteToken + 1,
                 kSquareOffset, kSquareOffset + 64);
    enable_write(projections, 7, chess1::kTargetWriteToken, chess1::kTargetWriteToken + 1,
                 chess1::kLegalOffset, chess1::kLegalOffset + 2);

    // Stage 8: every output square hardmax-selects unchanged input or a legal write token.
    for (std::int64_t square = 0; square < 64; ++square) {
        weight(weights.wq[8], kSquareOffset + square, square);
        weight(weights.wk[8], kSquareOffset + square, square);
    }
    weights.wq[8].index_put_({chess1::kLegalOffset + 1, 64}, 2.0);
    weights.wk[8].index_put_({chess1::kLegalOffset + 1, 64}, 2.0);
    copy_range(weights.wv[8], kSourcePieceOffset, chess1::kOutputPieceOffset, 3);
    for (std::int64_t square = 0; square < 64; ++square) {
        const auto output = chess1::kOutputSquareTokenBegin + square;
        masks[8].index_put_({output, output}, negative_infinity);
        masks[8].index_put_({output, chess1::kSquareTokenBegin + square}, 0.0);
        masks[8].index_put_({output, chess1::kSourceWriteToken}, 0.0);
        masks[8].index_put_({output, chess1::kTargetWriteToken}, 0.0);
    }
    enable_write(projections, 8, chess1::kOutputSquareTokenBegin,
                 chess1::kOutputSquareTokenBegin + 64, chess1::kOutputPieceOffset,
                 chess1::kOutputPieceOffset + 3);

    // Stage 9 flips side only when the selected candidate carries LEGAL=TRUE.
    weights.wq[9].index_put_({kSideQueryFeature, 0}, 1.0);
    weights.wk[9].index_put_({kSideBaselineFeature, 0}, 0.5);
    weights.wk[9].index_put_({chess1::kLegalOffset + 1, 0}, 1.0);
    copy_range(weights.wv[9], kSideOffset, chess1::kOutputSideOffset, 2);
    masks[9].index_put_({chess1::kOutputSideToken, chess1::kOutputSideToken}, negative_infinity);
    masks[9].index_put_({chess1::kOutputSideToken, chess1::kSideToken}, 0.0);
    masks[9].index_put_({chess1::kOutputSideToken, chess1::kSelectedToken}, 0.0);
    enable_write(projections, 9, chess1::kOutputSideToken, chess1::kOutputSideToken + 1,
                 chess1::kOutputSideOffset, chess1::kOutputSideOffset + 2);

    return ProgramImage{tokens, masks, projections, weights};
}

}  // namespace cmz::vm2
