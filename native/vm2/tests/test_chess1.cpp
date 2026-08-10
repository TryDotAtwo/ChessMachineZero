#include <stdexcept>
#include <string>

#include "cmz_vm2/chess1_compiler.h"
#include "cmz_vm2/machine.h"

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

}  // namespace

int main() {
    using cmz::vm2::Chess1Piece;
    using cmz::vm2::Chess1Side;
    cmz::vm2::Chess1Board board{};
    const auto image = cmz::vm2::compile_chess1(board, 8);

    require(image.tokens.size(0) == cmz::vm2::chess1::kTokenCount,
            "chess image must contain the fixed token schema");
    require(cmz::vm2::chess1::kSquareTokenCount == 64,
            "chess image must contain 64 input squares");
    require(cmz::vm2::chess1::kCandidateTokenCount == 64,
            "chess image must contain 64 pawn candidates");
    require(cmz::vm2::chess1::kOutputSquareTokenCount == 64,
            "chess image must contain 64 output squares");

    const auto candidate_legal = [](const cmz::vm2::Chess1Board& position,
                                    std::int64_t source) {
        const auto compiled = cmz::vm2::compile_chess1(position, source);
        const auto state = cmz::vm2::transition(compiled, compiled.tokens);
        return state.index({cmz::vm2::chess1::kCandidateTokenBegin + source,
                            cmz::vm2::chess1::kLegalOffset + 1})
                   .item<double>() == 1.0;
    };

    for (std::int64_t source = 0; source < 64; ++source) {
        cmz::vm2::Chess1Board position{};
        position.squares[source] = Chess1Piece::WhitePawn;
        const bool expected = source >= 8 && source < 56;
        require(candidate_legal(position, source) == expected,
                ("geometry legality mismatch at source " + std::to_string(source)).c_str());

        position.side = Chess1Side::Black;
        require(!candidate_legal(position, source), "black side must reject white pawn push");

        position.side = Chess1Side::White;
        position.squares[source] = Chess1Piece::Other;
        require(!candidate_legal(position, source), "non-pawn source must be illegal");

        if (source >= 8 && source < 56) {
            position.squares[source] = Chess1Piece::WhitePawn;
            position.squares[source + 8] = Chess1Piece::Other;
            require(!candidate_legal(position, source), "occupied target must be illegal");
        }
    }

    for (std::int64_t source = 8; source < 56; ++source) {
        cmz::vm2::Chess1Board position{};
        position.squares[source] = Chess1Piece::WhitePawn;
        const auto compiled = cmz::vm2::compile_chess1(position, source);
        const auto state = cmz::vm2::transition(compiled, compiled.tokens);
        for (std::int64_t square = 0; square < 64; ++square) {
            const auto expected = square == source
                                      ? Chess1Piece::Empty
                                      : (square == source + 8 ? Chess1Piece::WhitePawn
                                                              : position.squares[square]);
            const auto actual = state
                                    .index({cmz::vm2::chess1::kOutputSquareTokenBegin + square})
                                    .slice(0, cmz::vm2::chess1::kOutputPieceOffset,
                                           cmz::vm2::chess1::kOutputPieceOffset + 3)
                                    .argmax()
                                    .item<std::int64_t>();
            require(actual == static_cast<std::int64_t>(expected),
                    "legal move must alter exactly source and target output tokens");
        }
        require(state.index({cmz::vm2::chess1::kOutputSideToken,
                             cmz::vm2::chess1::kOutputSideOffset +
                                 static_cast<std::int64_t>(Chess1Side::Black)})
                    .item<double>() == 1.0,
                "legal move must emit black side-to-move");
    }

    cmz::vm2::Chess1Board illegal{};
    illegal.squares[8] = Chess1Piece::Other;
    const auto illegal_image = cmz::vm2::compile_chess1(illegal, 8);
    const auto illegal_state = cmz::vm2::transition(illegal_image, illegal_image.tokens);
    require(illegal_state.index({cmz::vm2::chess1::kOutputSquareTokenBegin + 8,
                                 cmz::vm2::chess1::kOutputPieceOffset +
                                     static_cast<std::int64_t>(Chess1Piece::Other)})
                .item<double>() == 1.0,
            "illegal selection must preserve the board");
    require(illegal_state.index({cmz::vm2::chess1::kOutputSideToken,
                                 cmz::vm2::chess1::kOutputSideOffset +
                                     static_cast<std::int64_t>(Chess1Side::White)})
                .item<double>() == 1.0,
            "illegal selection must preserve side-to-move");
    return 0;
}
