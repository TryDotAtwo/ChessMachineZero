#include "cmz_vm2/chess1_compiler.h"

#include <limits>
#include <stdexcept>

namespace cmz::vm2 {
namespace {

constexpr std::int64_t kSourcePieceOffset = 0;
constexpr std::int64_t kTargetPieceOffset = 4;
constexpr std::int64_t kSideOffset = 8;
constexpr std::int64_t kSquareOffset = 10;
constexpr std::int64_t kTargetOffset = 74;
constexpr std::int64_t kSourceValidOffset = 139;
constexpr std::int64_t kTargetOnboardOffset = 141;
constexpr std::int64_t kSideBaselineFeature = 206;
constexpr std::int64_t kSideQueryFeature = 207;
constexpr std::int64_t kNextSideOffset = 208;

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
    if (selected_source < -1 || selected_source >= chess1::kCandidateTokenCount) {
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

        for (std::int64_t side = 0; side < 2; ++side) {
            const auto geometry_row = chess1::kGeometryTokenBegin + side * 64 + square;
            const bool valid = square >= 8 && square < 56;
            const auto target = side == static_cast<std::int64_t>(Chess1Side::White)
                                    ? square + 8
                                    : square - 8;
            one_hot(tokens, geometry_row, kSquareOffset, square);
            one_hot(tokens, geometry_row, kSideOffset, side);
            one_hot(tokens, geometry_row, kTargetOffset, valid ? target : 64);
            one_hot(tokens, geometry_row, kSourceValidOffset, valid ? 1 : 0);
            one_hot(tokens, geometry_row, kTargetOnboardOffset, valid ? 1 : 0);
        }
    }
    one_hot(tokens, chess1::kOffboardToken, kSourcePieceOffset,
            static_cast<std::int64_t>(Chess1Piece::Other));
    one_hot(tokens, chess1::kOffboardToken, kTargetOffset, 64);
    one_hot(tokens, chess1::kSideToken, kSideBaselineFeature, 0);
    if (selected_source == -1) {
        one_hot(tokens, chess1::kSelectedToken, chess1::kLegalOffset, 1);
    }
    one_hot(tokens, chess1::kSourceWriteToken, kSourcePieceOffset,
            static_cast<std::int64_t>(Chess1Piece::Empty));
    for (std::int64_t square = 0; square < 64; ++square) {
        one_hot(tokens, chess1::kOutputSquareTokenBegin + square, kSquareOffset, square);
        one_hot(tokens, chess1::kOutputSquareTokenBegin + square, chess1::kLegalOffset, 1);
    }
    one_hot(tokens, chess1::kOutputSideToken, kSideQueryFeature, 0);

    std::int64_t relation_row = chess1::kLegalityRelationTokenBegin;
    for (std::int64_t source_piece = 0; source_piece < 4; ++source_piece) {
        for (std::int64_t side = 0; side < 2; ++side) {
            for (std::int64_t valid = 0; valid < 2; ++valid) {
                for (std::int64_t onboard = 0; onboard < 2; ++onboard) {
                    for (std::int64_t target_piece = 0; target_piece < 4; ++target_piece) {
                        one_hot(tokens, relation_row, kSourcePieceOffset, source_piece);
                        one_hot(tokens, relation_row, kTargetPieceOffset, target_piece);
                        one_hot(tokens, relation_row, kSideOffset, side);
                        one_hot(tokens, relation_row, kSourceValidOffset, valid);
                        one_hot(tokens, relation_row, kTargetOnboardOffset, onboard);
                        const bool matching_pawn =
                            (source_piece == static_cast<std::int64_t>(Chess1Piece::WhitePawn) &&
                             side == static_cast<std::int64_t>(Chess1Side::White)) ||
                            (source_piece == static_cast<std::int64_t>(Chess1Piece::BlackPawn) &&
                             side == static_cast<std::int64_t>(Chess1Side::Black));
                        const bool legal = matching_pawn && valid == 1 && onboard == 1 &&
                            target_piece == static_cast<std::int64_t>(Chess1Piece::Empty);
                        one_hot(tokens, relation_row, chess1::kLegalOffset, legal ? 1 : 0);
                        one_hot(tokens, relation_row, kNextSideOffset, 1 - side);
                        ++relation_row;
                    }
                }
            }
        }
    }

    // Stages 0..3 route source piece, side-keyed geometry and target piece.
    copy_range(weights.wv[0], kSourcePieceOffset, kSourcePieceOffset, 4);
    copy_range(weights.wv[1], kSideOffset, kSideOffset, 2);
    for (std::int64_t square = 0; square < 64; ++square) {
        const auto candidate = chess1::kCandidateTokenBegin + square;
        masks[0].index_put_({candidate, candidate}, negative_infinity);
        masks[0].index_put_({candidate, chess1::kSquareTokenBegin + square}, 0.0);
        masks[1].index_put_({candidate, candidate}, negative_infinity);
        masks[1].index_put_({candidate, chess1::kSideToken}, 0.0);
    }
    enable_write(projections, 0, chess1::kCandidateTokenBegin,
                 chess1::kCandidateTokenBegin + 64, kSourcePieceOffset, kSourcePieceOffset + 4);
    enable_write(projections, 1, chess1::kCandidateTokenBegin,
                 chess1::kCandidateTokenBegin + 64, kSideOffset, kSideOffset + 2);

    for (std::int64_t square = 0; square < 64; ++square) {
        weight(weights.wq[2], kSquareOffset + square, square);
        weight(weights.wk[2], kSquareOffset + square, square);
    }
    for (std::int64_t side = 0; side < 2; ++side) {
        weight(weights.wq[2], kSideOffset + side, 64 + side);
        weight(weights.wk[2], kSideOffset + side, 64 + side);
    }
    copy_range(weights.wv[2], kTargetOffset, kTargetOffset, 65);
    copy_range(weights.wv[2], kSourceValidOffset, kSourceValidOffset, 2);
    copy_range(weights.wv[2], kTargetOnboardOffset, kTargetOnboardOffset, 2);
    for (std::int64_t square = 0; square < 64; ++square) {
        const auto candidate = chess1::kCandidateTokenBegin + square;
        masks[2].index_put_({candidate, candidate}, negative_infinity);
        for (std::int64_t relation = 0; relation < chess1::kGeometryTokenCount; ++relation) {
            masks[2].index_put_({candidate, chess1::kGeometryTokenBegin + relation}, 0.0);
        }
    }
    enable_write(projections, 2, chess1::kCandidateTokenBegin,
                 chess1::kCandidateTokenBegin + 64, kTargetOffset, kTargetOffset + 65);
    enable_write(projections, 2, chess1::kCandidateTokenBegin,
                 chess1::kCandidateTokenBegin + 64, kSourceValidOffset, kTargetOnboardOffset + 2);

    for (std::int64_t square = 0; square < 64; ++square) {
        weight(weights.wq[3], kTargetOffset + square, square);
        weight(weights.wk[3], kSquareOffset + square, square);
    }
    weight(weights.wq[3], kTargetOffset + 64, 64);
    weight(weights.wk[3], kTargetOffset + 64, 64);
    copy_range(weights.wv[3], kSourcePieceOffset, kTargetPieceOffset, 4);
    for (std::int64_t square = 0; square < 64; ++square) {
        const auto candidate = chess1::kCandidateTokenBegin + square;
        masks[3].index_put_({candidate, candidate}, negative_infinity);
        masks[3].index_put_({candidate, chess1::kOffboardToken}, 0.0);
        for (std::int64_t target = 0; target < 64; ++target) {
            masks[3].index_put_({candidate, chess1::kSquareTokenBegin + target}, 0.0);
        }
    }
    enable_write(projections, 3, chess1::kCandidateTokenBegin,
                 chess1::kCandidateTokenBegin + 64, kTargetPieceOffset, kTargetPieceOffset + 4);

    // Stage 4 performs an exact five-field relation lookup with QK^T + hardmax.
    std::int64_t score_feature = 0;
    const auto score_category = [&](std::int64_t offset, std::int64_t count) {
        for (std::int64_t value = 0; value < count; ++value) {
            weight(weights.wq[4], offset + value, score_feature + value);
            weight(weights.wk[4], offset + value, score_feature + value);
        }
        score_feature += count;
    };
    score_category(kSourcePieceOffset, 4);
    score_category(kTargetPieceOffset, 4);
    score_category(kSideOffset, 2);
    score_category(kSourceValidOffset, 2);
    score_category(kTargetOnboardOffset, 2);
    copy_range(weights.wv[4], chess1::kLegalOffset, chess1::kLegalOffset, 2);
    copy_range(weights.wv[4], kNextSideOffset, kNextSideOffset, 2);
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
    enable_write(projections, 4, chess1::kCandidateTokenBegin,
                 chess1::kCandidateTokenBegin + 64, kNextSideOffset, kNextSideOffset + 2);

    // Stage 5 selects one candidate token; stages 6 and 7 materialize two write tokens.
    copy_range(weights.wv[5], kSquareOffset, kSquareOffset, 64);
    copy_range(weights.wv[5], kTargetOffset, kTargetOffset, 65);
    copy_range(weights.wv[5], kSourcePieceOffset, kSourcePieceOffset, 4);
    copy_range(weights.wv[5], chess1::kLegalOffset, chess1::kLegalOffset, 2);
    copy_range(weights.wv[5], kNextSideOffset, kNextSideOffset, 2);
    masks[5].index_put_({chess1::kSelectedToken, chess1::kSelectedToken}, negative_infinity);
    if (selected_source == -1) {
        weights.wq[5].index_put_({chess1::kLegalOffset + 1, 0}, 1.0);
        weights.wk[5].index_put_({chess1::kLegalOffset + 1, 0}, 1.0);
        for (std::int64_t candidate = 0; candidate < 64; ++candidate) {
            masks[5].index_put_({chess1::kSelectedToken,
                                 chess1::kCandidateTokenBegin + candidate}, 0.0);
        }
    } else {
        masks[5].index_put_({chess1::kSelectedToken,
                             chess1::kCandidateTokenBegin + selected_source}, 0.0);
    }
    enable_write(projections, 5, chess1::kSelectedToken, chess1::kSelectedToken + 1,
                 kSquareOffset, kTargetOffset + 65);
    enable_write(projections, 5, chess1::kSelectedToken, chess1::kSelectedToken + 1,
                 chess1::kLegalOffset, chess1::kLegalOffset + 2);
    enable_write(projections, 5, chess1::kSelectedToken, chess1::kSelectedToken + 1,
                 kSourcePieceOffset, kSourcePieceOffset + 4);
    enable_write(projections, 5, chess1::kSelectedToken, chess1::kSelectedToken + 1,
                 kNextSideOffset, kNextSideOffset + 2);

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
    copy_range(weights.wv[7], kSourcePieceOffset, kSourcePieceOffset, 4);
    copy_range(weights.wv[7], chess1::kLegalOffset, chess1::kLegalOffset, 2);
    masks[7].index_put_({chess1::kTargetWriteToken, chess1::kTargetWriteToken},
                        negative_infinity);
    masks[7].index_put_({chess1::kTargetWriteToken, chess1::kSelectedToken}, 0.0);
    enable_write(projections, 7, chess1::kTargetWriteToken, chess1::kTargetWriteToken + 1,
                 kSquareOffset, kSquareOffset + 64);
    enable_write(projections, 7, chess1::kTargetWriteToken, chess1::kTargetWriteToken + 1,
                 chess1::kLegalOffset, chess1::kLegalOffset + 2);
    enable_write(projections, 7, chess1::kTargetWriteToken, chess1::kTargetWriteToken + 1,
                 kSourcePieceOffset, kSourcePieceOffset + 4);

    // Stage 8: every output square hardmax-selects unchanged input or a legal write token.
    for (std::int64_t square = 0; square < 64; ++square) {
        weight(weights.wq[8], kSquareOffset + square, square);
        weight(weights.wk[8], kSquareOffset + square, square);
    }
    // Square equality contributes 1.0. The legal gate contributes 0.25, so
    // exact square+legal writes beat the matching input (1.25 > 1.0), while a
    // legal write for another square cannot overwrite it (0.25 < 1.0).
    weights.wq[8].index_put_({chess1::kLegalOffset + 1, 64}, 0.5);
    weights.wk[8].index_put_({chess1::kLegalOffset + 1, 64}, 0.5);
    copy_range(weights.wv[8], kSourcePieceOffset, chess1::kOutputPieceOffset, 4);
    for (std::int64_t square = 0; square < 64; ++square) {
        const auto output = chess1::kOutputSquareTokenBegin + square;
        masks[8].index_put_({output, output}, negative_infinity);
        masks[8].index_put_({output, chess1::kSquareTokenBegin + square}, 0.0);
        masks[8].index_put_({output, chess1::kSourceWriteToken}, 0.0);
        masks[8].index_put_({output, chess1::kTargetWriteToken}, 0.0);
    }
    enable_write(projections, 8, chess1::kOutputSquareTokenBegin,
                 chess1::kOutputSquareTokenBegin + 64, chess1::kOutputPieceOffset,
                 chess1::kOutputPieceOffset + 4);

    // Stage 9 flips side only when the selected candidate carries LEGAL=TRUE.
    weights.wq[9].index_put_({kSideQueryFeature, 0}, 1.0);
    weights.wk[9].index_put_({kSideBaselineFeature, 0}, 0.5);
    weights.wk[9].index_put_({chess1::kLegalOffset + 1, 0}, 1.0);
    copy_range(weights.wv[9], kSideOffset, chess1::kOutputSideOffset, 2);
    copy_range(weights.wv[9], kNextSideOffset, chess1::kOutputSideOffset, 2);
    masks[9].index_put_({chess1::kOutputSideToken, chess1::kOutputSideToken}, negative_infinity);
    masks[9].index_put_({chess1::kOutputSideToken, chess1::kSideToken}, 0.0);
    masks[9].index_put_({chess1::kOutputSideToken, chess1::kSelectedToken}, 0.0);
    enable_write(projections, 9, chess1::kOutputSideToken, chess1::kOutputSideToken + 1,
                 chess1::kOutputSideOffset, chess1::kOutputSideOffset + 2);

    // Stage 10 closes the recurrent state: emitted board and side become the
    // next transition's input tokens using attention routing only.
    copy_range(weights.wv[10], chess1::kOutputPieceOffset, kSourcePieceOffset, 4);
    copy_range(weights.wv[10], chess1::kOutputSideOffset, kSideOffset, 2);
    for (std::int64_t square = 0; square < 64; ++square) {
        const auto input = chess1::kSquareTokenBegin + square;
        const auto output = chess1::kOutputSquareTokenBegin + square;
        masks[10].index_put_({input, input}, negative_infinity);
        masks[10].index_put_({input, output}, 0.0);
    }
    masks[10].index_put_({chess1::kSideToken, chess1::kSideToken}, negative_infinity);
    masks[10].index_put_({chess1::kSideToken, chess1::kOutputSideToken}, 0.0);
    enable_write(projections, 10, chess1::kSquareTokenBegin,
                 chess1::kSquareTokenBegin + 64, kSourcePieceOffset,
                 kSourcePieceOffset + 4);
    enable_write(projections, 10, chess1::kSideToken, chess1::kSideToken + 1,
                 kSideOffset, kSideOffset + 2);

    return ProgramImage{tokens, masks, projections, weights};
}

ProgramImage compile_chess1_auto(const Chess1Board& board) {
    return compile_chess1(board, -1);
}

}  // namespace cmz::vm2
