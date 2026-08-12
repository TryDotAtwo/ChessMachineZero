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
constexpr std::int64_t kMoveKindOffset = 210;
constexpr std::int64_t kIntermediatePieceOffset = 212;
constexpr std::int64_t kMatchingPawnOffset = 216;
constexpr std::int64_t kTargetEmptyOffset = 218;
constexpr std::int64_t kPathClearOffset = 220;
constexpr std::int64_t kGeometryValidOffset = 222;
constexpr std::int64_t kStartRankOffset = 224;
constexpr std::int64_t kIntermediateTargetOffset = 226;

constexpr std::int64_t kMatchRelationBegin = chess1::kLegalityRelationTokenBegin;
constexpr std::int64_t kTargetEmptyRelationBegin =
    kMatchRelationBegin + chess1::kMatchingRelationTokenCount;
constexpr std::int64_t kPathRelationBegin =
    kTargetEmptyRelationBegin + chess1::kTargetEmptyRelationTokenCount;
constexpr std::int64_t kConjunctionRelationBegin =
    kPathRelationBegin + chess1::kPathRelationTokenCount;

void one_hot(torch::Tensor& tokens, std::int64_t row, std::int64_t offset, std::int64_t value) {
    tokens.index_put_({row, offset + value}, 1.0);
}

void weight(torch::Tensor& matrix,
            std::int64_t input,
            std::int64_t output,
            double value = 1.0) {
    matrix.index_put_({input, output}, value);
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

void enable_candidate_write(WriteProjections& projections,
                            std::int64_t stage,
                            std::int64_t feature_begin,
                            std::int64_t feature_end) {
    enable_write(projections, stage, chess1::kCandidateTokenBegin,
                 chess1::kCandidateTokenBegin + chess1::kCandidateTokenCount,
                 feature_begin, feature_end);
}

void allow_relation_group(std::array<torch::Tensor, kStageCount>& masks,
                          std::int64_t stage,
                          std::int64_t begin,
                          std::int64_t count,
                          double negative_infinity) {
    for (std::int64_t candidate = 0; candidate < chess1::kCandidateTokenCount; ++candidate) {
        const auto row = chess1::kCandidateTokenBegin + candidate;
        masks[stage].index_put_({row, row}, negative_infinity);
        for (std::int64_t relation = 0; relation < count; ++relation) {
            masks[stage].index_put_({row, begin + relation}, 0.0);
        }
    }
}

void score_category(FrozenWeights& weights,
                    std::int64_t stage,
                    std::int64_t input_offset,
                    std::int64_t score_offset,
                    std::int64_t count) {
    for (std::int64_t value = 0; value < count; ++value) {
        weight(weights.wq[stage], input_offset + value, score_offset + value);
        weight(weights.wk[stage], input_offset + value, score_offset + value);
    }
}

}  // namespace

ProgramImage compile_chess1(const Chess1Board& board, std::int64_t selected_candidate) {
    if (selected_candidate < -1 || selected_candidate >= chess1::kCandidateTokenCount) {
        throw std::invalid_argument("selected candidate is outside 0..127");
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
    one_hot(tokens, chess1::kSideToken, kSideBaselineFeature, 0);
    for (std::int64_t square = 0; square < 64; ++square) {
        const auto square_row = chess1::kSquareTokenBegin + square;
        one_hot(tokens, square_row, kSourcePieceOffset,
                static_cast<std::int64_t>(board.squares[square]));
        one_hot(tokens, square_row, kSquareOffset, square);

        for (std::int64_t kind = 0; kind < 2; ++kind) {
            const auto candidate = kind * 64 + square;
            const auto candidate_row = chess1::kCandidateTokenBegin + candidate;
            one_hot(tokens, candidate_row, kSquareOffset, square);
            one_hot(tokens, candidate_row, kMoveKindOffset, kind);

            for (std::int64_t side = 0; side < 2; ++side) {
                const auto geometry_row = chess1::kGeometryTokenBegin +
                    kind * 128 + side * 64 + square;
                const auto direction = side == static_cast<std::int64_t>(Chess1Side::White)
                                           ? 1
                                           : -1;
                const auto distance = kind == 0 ? 8 : 16;
                const auto target = square + direction * distance;
                const bool onboard = target >= 0 && target < 64;
                const bool source_valid = square >= 8 && square < 56;
                const bool start_rank = kind == 0 ||
                    (side == static_cast<std::int64_t>(Chess1Side::White)
                         ? square >= 8 && square < 16
                         : square >= 48 && square < 56);
                one_hot(tokens, geometry_row, kSquareOffset, square);
                one_hot(tokens, geometry_row, kSideOffset, side);
                one_hot(tokens, geometry_row, kMoveKindOffset, kind);
                one_hot(tokens, geometry_row, kTargetOffset, onboard ? target : 64);
                const auto intermediate = square + direction * 8;
                const bool intermediate_onboard = intermediate >= 0 && intermediate < 64;
                one_hot(tokens, geometry_row, kIntermediateTargetOffset,
                        kind == 1 && intermediate_onboard ? intermediate : 64);
                one_hot(tokens, geometry_row, kSourceValidOffset, source_valid ? 1 : 0);
                one_hot(tokens, geometry_row, kTargetOnboardOffset, onboard ? 1 : 0);
                one_hot(tokens, geometry_row, kGeometryValidOffset,
                        source_valid && onboard ? 1 : 0);
                one_hot(tokens, geometry_row, kStartRankOffset, start_rank ? 1 : 0);
            }
        }
    }
    one_hot(tokens, chess1::kOffboardToken, kSourcePieceOffset,
            static_cast<std::int64_t>(Chess1Piece::Other));
    one_hot(tokens, chess1::kOffboardToken, kTargetOffset, 64);
    if (selected_candidate == -1) {
        one_hot(tokens, chess1::kSelectedToken, chess1::kLegalOffset, 1);
    }
    one_hot(tokens, chess1::kSourceWriteToken, kSourcePieceOffset,
            static_cast<std::int64_t>(Chess1Piece::Empty));
    for (std::int64_t square = 0; square < 64; ++square) {
        one_hot(tokens, chess1::kOutputSquareTokenBegin + square, kSquareOffset, square);
        one_hot(tokens, chess1::kOutputSquareTokenBegin + square, chess1::kLegalOffset, 1);
    }
    one_hot(tokens, chess1::kOutputSideToken, kSideQueryFeature, 0);

    std::int64_t row = kMatchRelationBegin;
    for (std::int64_t source_piece = 0; source_piece < 4; ++source_piece) {
        for (std::int64_t side = 0; side < 2; ++side) {
            one_hot(tokens, row, kSourcePieceOffset, source_piece);
            one_hot(tokens, row, kSideOffset, side);
            const bool matching =
                (source_piece == static_cast<std::int64_t>(Chess1Piece::WhitePawn) &&
                 side == static_cast<std::int64_t>(Chess1Side::White)) ||
                (source_piece == static_cast<std::int64_t>(Chess1Piece::BlackPawn) &&
                 side == static_cast<std::int64_t>(Chess1Side::Black));
            one_hot(tokens, row++, kMatchingPawnOffset, matching ? 1 : 0);
        }
    }
    for (std::int64_t target_piece = 0; target_piece < 4; ++target_piece) {
        one_hot(tokens, row, kTargetPieceOffset, target_piece);
        one_hot(tokens, row++, kTargetEmptyOffset,
                target_piece == static_cast<std::int64_t>(Chess1Piece::Empty) ? 1 : 0);
    }
    for (std::int64_t kind = 0; kind < 2; ++kind) {
        for (std::int64_t intermediate = 0; intermediate < 4; ++intermediate) {
            one_hot(tokens, row, kMoveKindOffset, kind);
            one_hot(tokens, row, kIntermediatePieceOffset, intermediate);
            const bool clear = kind == 0 ||
                intermediate == static_cast<std::int64_t>(Chess1Piece::Empty);
            one_hot(tokens, row++, kPathClearOffset, clear ? 1 : 0);
        }
    }
    for (std::int64_t bits = 0; bits < 32; ++bits) {
        const auto matching = (bits >> 0) & 1;
        const auto target_empty = (bits >> 1) & 1;
        const auto path_clear = (bits >> 2) & 1;
        const auto geometry_valid = (bits >> 3) & 1;
        const auto start_rank = (bits >> 4) & 1;
        one_hot(tokens, row, kMatchingPawnOffset, matching);
        one_hot(tokens, row, kTargetEmptyOffset, target_empty);
        one_hot(tokens, row, kPathClearOffset, path_clear);
        one_hot(tokens, row, kGeometryValidOffset, geometry_valid);
        one_hot(tokens, row, kStartRankOffset, start_rank);
        one_hot(tokens, row, chess1::kLegalOffset,
                matching && target_empty && path_clear && geometry_valid && start_rank ? 1 : 0);
        one_hot(tokens, row, kNextSideOffset, 1);
        ++row;
    }
    if (row != chess1::kSelectedToken) {
        throw std::logic_error("chess relation token count mismatch");
    }

    // 0: source piece; 1: side; 2: side/kind/source keyed geometry.
    copy_range(weights.wv[0], kSourcePieceOffset, kSourcePieceOffset, 4);
    copy_range(weights.wv[1], kSideOffset, kSideOffset, 2);
    for (std::int64_t candidate = 0; candidate < chess1::kCandidateTokenCount; ++candidate) {
        const auto source = candidate % 64;
        const auto candidate_row = chess1::kCandidateTokenBegin + candidate;
        masks[0].index_put_({candidate_row, candidate_row}, negative_infinity);
        masks[0].index_put_({candidate_row, chess1::kSquareTokenBegin + source}, 0.0);
        masks[1].index_put_({candidate_row, candidate_row}, negative_infinity);
        masks[1].index_put_({candidate_row, chess1::kSideToken}, 0.0);
    }
    enable_candidate_write(projections, 0, kSourcePieceOffset, kSourcePieceOffset + 4);
    enable_candidate_write(projections, 1, kSideOffset, kSideOffset + 2);

    score_category(weights, 2, kSquareOffset, 0, 64);
    score_category(weights, 2, kSideOffset, 64, 2);
    score_category(weights, 2, kMoveKindOffset, 66, 2);
    copy_range(weights.wv[2], kTargetOffset, kTargetOffset, 65);
    copy_range(weights.wv[2], kSourceValidOffset, kSourceValidOffset, 2);
    copy_range(weights.wv[2], kTargetOnboardOffset, kTargetOnboardOffset, 2);
    copy_range(weights.wv[2], kGeometryValidOffset, kGeometryValidOffset, 2);
    copy_range(weights.wv[2], kStartRankOffset, kStartRankOffset, 2);
    copy_range(weights.wv[2], kIntermediateTargetOffset, kIntermediateTargetOffset, 65);
    for (std::int64_t candidate = 0; candidate < chess1::kCandidateTokenCount; ++candidate) {
        const auto candidate_row = chess1::kCandidateTokenBegin + candidate;
        masks[2].index_put_({candidate_row, candidate_row}, negative_infinity);
        for (std::int64_t geometry = 0; geometry < chess1::kGeometryTokenCount; ++geometry) {
            masks[2].index_put_({candidate_row, chess1::kGeometryTokenBegin + geometry}, 0.0);
        }
    }
    enable_candidate_write(projections, 2, kTargetOffset, kTargetOffset + 65);
    enable_candidate_write(projections, 2, kSourceValidOffset, kTargetOnboardOffset + 2);
    enable_candidate_write(projections, 2, kGeometryValidOffset, kGeometryValidOffset + 2);
    enable_candidate_write(projections, 2, kStartRankOffset, kStartRankOffset + 2);
    enable_candidate_write(projections, 2, kIntermediateTargetOffset,
                           kIntermediateTargetOffset + 65);

    // 3: target occupancy; 4: intermediate occupancy for double pushes.
    for (std::int64_t square = 0; square < 64; ++square) {
        weight(weights.wq[3], kTargetOffset + square, square);
        weight(weights.wq[4], kIntermediateTargetOffset + square, square);
        weight(weights.wk[3], kSquareOffset + square, square);
        weight(weights.wk[4], kSquareOffset + square, square);
    }
    weight(weights.wq[3], kTargetOffset + 64, 64);
    weight(weights.wq[4], kIntermediateTargetOffset + 64, 64);
    weight(weights.wk[3], kTargetOffset + 64, 64);
    weight(weights.wk[4], kTargetOffset + 64, 64);
    copy_range(weights.wv[3], kSourcePieceOffset, kTargetPieceOffset, 4);
    copy_range(weights.wv[4], kSourcePieceOffset, kIntermediatePieceOffset, 4);
    for (std::int64_t candidate = 0; candidate < chess1::kCandidateTokenCount; ++candidate) {
        const auto candidate_row = chess1::kCandidateTokenBegin + candidate;
        masks[3].index_put_({candidate_row, candidate_row}, negative_infinity);
        masks[4].index_put_({candidate_row, candidate_row}, negative_infinity);
        masks[3].index_put_({candidate_row, chess1::kOffboardToken}, 0.0);
        masks[4].index_put_({candidate_row, chess1::kOffboardToken}, 0.0);
        for (std::int64_t square = 0; square < 64; ++square) {
            masks[3].index_put_({candidate_row, chess1::kSquareTokenBegin + square}, 0.0);
            masks[4].index_put_({candidate_row, chess1::kSquareTokenBegin + square}, 0.0);
        }
    }
    enable_candidate_write(projections, 3, kTargetPieceOffset, kTargetPieceOffset + 4);
    enable_candidate_write(projections, 4, kIntermediatePieceOffset,
                           kIntermediatePieceOffset + 4);

    // 5..8: predicate-token lookups and their exact conjunction.
    score_category(weights, 5, kSourcePieceOffset, 0, 4);
    score_category(weights, 5, kSideOffset, 4, 2);
    copy_range(weights.wv[5], kMatchingPawnOffset, kMatchingPawnOffset, 2);
    allow_relation_group(masks, 5, kMatchRelationBegin,
                         chess1::kMatchingRelationTokenCount, negative_infinity);
    enable_candidate_write(projections, 5, kMatchingPawnOffset, kMatchingPawnOffset + 2);

    score_category(weights, 6, kTargetPieceOffset, 0, 4);
    copy_range(weights.wv[6], kTargetEmptyOffset, kTargetEmptyOffset, 2);
    allow_relation_group(masks, 6, kTargetEmptyRelationBegin,
                         chess1::kTargetEmptyRelationTokenCount, negative_infinity);
    enable_candidate_write(projections, 6, kTargetEmptyOffset, kTargetEmptyOffset + 2);

    score_category(weights, 7, kMoveKindOffset, 0, 2);
    score_category(weights, 7, kIntermediatePieceOffset, 2, 4);
    copy_range(weights.wv[7], kPathClearOffset, kPathClearOffset, 2);
    allow_relation_group(masks, 7, kPathRelationBegin,
                         chess1::kPathRelationTokenCount, negative_infinity);
    enable_candidate_write(projections, 7, kPathClearOffset, kPathClearOffset + 2);

    score_category(weights, 8, kMatchingPawnOffset, 0, 2);
    score_category(weights, 8, kTargetEmptyOffset, 2, 2);
    score_category(weights, 8, kPathClearOffset, 4, 2);
    score_category(weights, 8, kGeometryValidOffset, 6, 2);
    score_category(weights, 8, kStartRankOffset, 8, 2);
    copy_range(weights.wv[8], chess1::kLegalOffset, chess1::kLegalOffset, 2);
    allow_relation_group(masks, 8, kConjunctionRelationBegin,
                         chess1::kConjunctionRelationTokenCount, negative_infinity);
    enable_candidate_write(projections, 8, chess1::kLegalOffset, chess1::kLegalOffset + 2);

    // 9: deterministic hardmax selection among legal trace candidates.
    copy_range(weights.wv[9], kSquareOffset, kSquareOffset, 64);
    copy_range(weights.wv[9], kTargetOffset, kTargetOffset, 65);
    copy_range(weights.wv[9], kSourcePieceOffset, kSourcePieceOffset, 4);
    copy_range(weights.wv[9], chess1::kLegalOffset, chess1::kLegalOffset, 2);
    weight(weights.wv[9], kSideOffset + 0, kNextSideOffset + 1);
    weight(weights.wv[9], kSideOffset + 1, kNextSideOffset + 0);
    masks[9].index_put_({chess1::kSelectedToken, chess1::kSelectedToken}, negative_infinity);
    if (selected_candidate == -1) {
        weight(weights.wq[9], chess1::kLegalOffset + 1, 0);
        weight(weights.wk[9], chess1::kLegalOffset + 1, 0);
        for (std::int64_t candidate = 0; candidate < chess1::kCandidateTokenCount; ++candidate) {
            masks[9].index_put_({chess1::kSelectedToken,
                                 chess1::kCandidateTokenBegin + candidate}, 0.0);
        }
    } else {
        masks[9].index_put_({chess1::kSelectedToken,
                             chess1::kCandidateTokenBegin + selected_candidate}, 0.0);
    }
    enable_write(projections, 9, chess1::kSelectedToken, chess1::kSelectedToken + 1,
                 kSquareOffset, kTargetOffset + 65);
    enable_write(projections, 9, chess1::kSelectedToken, chess1::kSelectedToken + 1,
                 chess1::kLegalOffset, chess1::kLegalOffset + 2);
    enable_write(projections, 9, chess1::kSelectedToken, chess1::kSelectedToken + 1,
                 kSourcePieceOffset, kSourcePieceOffset + 4);
    enable_write(projections, 9, chess1::kSelectedToken, chess1::kSelectedToken + 1,
                 kNextSideOffset, kNextSideOffset + 2);

    // 10/11: selected candidate becomes source/target write tokens.
    copy_range(weights.wv[10], kSquareOffset, kSquareOffset, 64);
    copy_range(weights.wv[10], chess1::kLegalOffset, chess1::kLegalOffset, 2);
    masks[10].index_put_({chess1::kSourceWriteToken, chess1::kSourceWriteToken},
                         negative_infinity);
    masks[10].index_put_({chess1::kSourceWriteToken, chess1::kSelectedToken}, 0.0);
    enable_write(projections, 10, chess1::kSourceWriteToken, chess1::kSourceWriteToken + 1,
                 kSquareOffset, kSquareOffset + 64);
    enable_write(projections, 10, chess1::kSourceWriteToken, chess1::kSourceWriteToken + 1,
                 chess1::kLegalOffset, chess1::kLegalOffset + 2);

    copy_range(weights.wv[11], kTargetOffset, kSquareOffset, 64);
    copy_range(weights.wv[11], kSourcePieceOffset, kSourcePieceOffset, 4);
    copy_range(weights.wv[11], chess1::kLegalOffset, chess1::kLegalOffset, 2);
    masks[11].index_put_({chess1::kTargetWriteToken, chess1::kTargetWriteToken},
                         negative_infinity);
    masks[11].index_put_({chess1::kTargetWriteToken, chess1::kSelectedToken}, 0.0);
    enable_write(projections, 11, chess1::kTargetWriteToken, chess1::kTargetWriteToken + 1,
                 kSquareOffset, kSquareOffset + 64);
    enable_write(projections, 11, chess1::kTargetWriteToken, chess1::kTargetWriteToken + 1,
                 chess1::kLegalOffset, chess1::kLegalOffset + 2);
    enable_write(projections, 11, chess1::kTargetWriteToken, chess1::kTargetWriteToken + 1,
                 kSourcePieceOffset, kSourcePieceOffset + 4);

    // 12: output squares choose unchanged input or an exact legal write.
    for (std::int64_t square = 0; square < 64; ++square) {
        weight(weights.wq[12], kSquareOffset + square, square);
        weight(weights.wk[12], kSquareOffset + square, square);
    }
    weight(weights.wq[12], chess1::kLegalOffset + 1, 64, 0.5);
    weight(weights.wk[12], chess1::kLegalOffset + 1, 64, 0.5);
    copy_range(weights.wv[12], kSourcePieceOffset, chess1::kOutputPieceOffset, 4);
    for (std::int64_t square = 0; square < 64; ++square) {
        const auto output = chess1::kOutputSquareTokenBegin + square;
        masks[12].index_put_({output, output}, negative_infinity);
        masks[12].index_put_({output, chess1::kSquareTokenBegin + square}, 0.0);
        masks[12].index_put_({output, chess1::kSourceWriteToken}, 0.0);
        masks[12].index_put_({output, chess1::kTargetWriteToken}, 0.0);
    }
    enable_write(projections, 12, chess1::kOutputSquareTokenBegin,
                 chess1::kOutputSquareTokenBegin + 64, chess1::kOutputPieceOffset,
                 chess1::kOutputPieceOffset + 4);

    // 13: side changes iff LEGAL=true; 14: close recurrent state.
    weight(weights.wq[13], kSideQueryFeature, 0);
    weight(weights.wk[13], kSideBaselineFeature, 0, 0.5);
    weight(weights.wk[13], chess1::kLegalOffset + 1, 0);
    copy_range(weights.wv[13], kSideOffset, chess1::kOutputSideOffset, 2);
    copy_range(weights.wv[13], kNextSideOffset, chess1::kOutputSideOffset, 2);
    masks[13].index_put_({chess1::kOutputSideToken, chess1::kOutputSideToken}, negative_infinity);
    masks[13].index_put_({chess1::kOutputSideToken, chess1::kSideToken}, 0.0);
    masks[13].index_put_({chess1::kOutputSideToken, chess1::kSelectedToken}, 0.0);
    enable_write(projections, 13, chess1::kOutputSideToken, chess1::kOutputSideToken + 1,
                 chess1::kOutputSideOffset, chess1::kOutputSideOffset + 2);

    copy_range(weights.wv[14], chess1::kOutputPieceOffset, kSourcePieceOffset, 4);
    copy_range(weights.wv[14], chess1::kOutputSideOffset, kSideOffset, 2);
    for (std::int64_t square = 0; square < 64; ++square) {
        const auto input = chess1::kSquareTokenBegin + square;
        const auto output = chess1::kOutputSquareTokenBegin + square;
        masks[14].index_put_({input, input}, negative_infinity);
        masks[14].index_put_({input, output}, 0.0);
    }
    masks[14].index_put_({chess1::kSideToken, chess1::kSideToken}, negative_infinity);
    masks[14].index_put_({chess1::kSideToken, chess1::kOutputSideToken}, 0.0);
    enable_write(projections, 14, chess1::kSquareTokenBegin,
                 chess1::kSquareTokenBegin + 64, kSourcePieceOffset, kSourcePieceOffset + 4);
    enable_write(projections, 14, chess1::kSideToken, chess1::kSideToken + 1,
                 kSideOffset, kSideOffset + 2);

    for (std::int64_t stage = 0; stage < kStageCount; ++stage) {
        weights.wq[stage] = weights.wq[stage].detach().set_requires_grad(false);
        weights.wk[stage] = weights.wk[stage].detach().set_requires_grad(false);
        weights.wv[stage] = weights.wv[stage].detach().set_requires_grad(false);
        weights.wo[stage] = weights.wo[stage].detach().set_requires_grad(false);
        masks[stage] = masks[stage].detach().set_requires_grad(false);
        for (std::int64_t component = 0; component < kWriteProjectionComponents; ++component) {
            projections.row[stage][component] =
                projections.row[stage][component].detach().set_requires_grad(false);
            projections.feature[stage][component] =
                projections.feature[stage][component].detach().set_requires_grad(false);
        }
    }
    tokens = tokens.detach().set_requires_grad(false);
    return ProgramImage{tokens, masks, projections, weights};
}

ProgramImage compile_chess1_auto(const Chess1Board& board) {
    return compile_chess1(board, -1);
}

}  // namespace cmz::vm2
