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
    require(cmz::vm2::chess1::kCandidateTokenCount == 256,
            "chess image must contain pawn pushes and diagonal captures");
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

    const auto double_candidate_legal = [](const cmz::vm2::Chess1Board& position,
                                           std::int64_t source) {
        const auto candidate = 64 + source;
        const auto compiled = cmz::vm2::compile_chess1(position, candidate);
        const auto state = cmz::vm2::transition(compiled, compiled.tokens);
        return state.index({cmz::vm2::chess1::kCandidateTokenBegin + candidate,
                            cmz::vm2::chess1::kLegalOffset + 1})
                   .item<double>() == 1.0;
    };

    const auto capture_candidate_legal = [](const cmz::vm2::Chess1Board& position,
                                            std::int64_t source,
                                            bool right) {
        const auto candidate = (right ? 3 : 2) * 64 + source;
        const auto compiled = cmz::vm2::compile_chess1(position, candidate);
        const auto state = cmz::vm2::transition(compiled, compiled.tokens);
        return state.index({cmz::vm2::chess1::kCandidateTokenBegin + candidate,
                            cmz::vm2::chess1::kLegalOffset + 1})
                   .item<double>() == 1.0;
    };

    {
        cmz::vm2::Chess1Board position{};
        position.squares[27] = Chess1Piece::WhitePawn;
        position.squares[34] = Chess1Piece::BlackPawn;
        position.squares[36] = Chess1Piece::BlackOther;
        require(capture_candidate_legal(position, 27, false),
                "white pawn must capture black piece forward-left");
        require(capture_candidate_legal(position, 27, true),
                "white pawn must capture black piece forward-right");
        position.squares[34] = Chess1Piece::Empty;
        require(!capture_candidate_legal(position, 27, false),
                "white pawn must not capture an empty target");
        position.squares[34] = Chess1Piece::WhiteOther;
        require(!capture_candidate_legal(position, 27, false),
                "white pawn must not capture its own piece");
    }

    {
        cmz::vm2::Chess1Board position{};
        position.side = Chess1Side::Black;
        position.squares[36] = Chess1Piece::BlackPawn;
        position.squares[27] = Chess1Piece::WhitePawn;
        position.squares[29] = Chess1Piece::WhiteOther;
        require(capture_candidate_legal(position, 36, false),
                "black pawn must capture white piece forward-left");
        require(capture_candidate_legal(position, 36, true),
                "black pawn must capture white piece forward-right");
    }

    {
        cmz::vm2::Chess1Board position{};
        position.squares[24] = Chess1Piece::WhitePawn;
        position.squares[31] = Chess1Piece::BlackPawn;
        require(!capture_candidate_legal(position, 24, false),
                "capture must not wrap left from file a");
        require(!capture_candidate_legal(position, 31, true),
                "capture must not wrap right from file h");
    }

    {
        cmz::vm2::Chess1Board position{};
        position.squares[8] = Chess1Piece::WhitePawn;
        require(double_candidate_legal(position, 8),
                "white pawn double push must be legal from its start rank");
        const auto compiled = cmz::vm2::compile_chess1(position, 64 + 8);
        const auto moved = cmz::vm2::transition(compiled, compiled.tokens);
        require(moved.index({cmz::vm2::chess1::kSquareTokenBegin + 8,
                             static_cast<std::int64_t>(Chess1Piece::Empty)})
                    .item<double>() == 1.0,
                "white double push must clear its source through matrix writes");
        require(moved.index({cmz::vm2::chess1::kSquareTokenBegin + 24,
                             static_cast<std::int64_t>(Chess1Piece::WhitePawn)})
                    .item<double>() == 1.0,
                "white double push must write its two-rank target through matrix writes");
        position.squares[16] = Chess1Piece::Other;
        require(!double_candidate_legal(position, 8),
                "occupied intermediate square must reject white double push");
        position.squares[16] = Chess1Piece::Empty;
        position.squares[24] = Chess1Piece::Other;
        require(!double_candidate_legal(position, 8),
                "occupied target square must reject white double push");
        position.squares[8] = Chess1Piece::Empty;
        position.squares[16] = Chess1Piece::WhitePawn;
        position.squares[24] = Chess1Piece::Empty;
        require(!double_candidate_legal(position, 16),
                "white double push must be illegal outside its start rank");
    }

    {
        cmz::vm2::Chess1Board position{};
        position.side = Chess1Side::Black;
        position.squares[48] = Chess1Piece::BlackPawn;
        require(double_candidate_legal(position, 48),
                "black pawn double push must be legal from its start rank");
        const auto compiled = cmz::vm2::compile_chess1(position, 64 + 48);
        const auto moved = cmz::vm2::transition(compiled, compiled.tokens);
        require(moved.index({cmz::vm2::chess1::kSquareTokenBegin + 32,
                             static_cast<std::int64_t>(Chess1Piece::BlackPawn)})
                    .item<double>() == 1.0,
                "black double push must write its two-rank target through matrix writes");
        position.squares[40] = Chess1Piece::Other;
        require(!double_candidate_legal(position, 48),
                "occupied intermediate square must reject black double push");
        position.squares[40] = Chess1Piece::Empty;
        position.squares[32] = Chess1Piece::Other;
        require(!double_candidate_legal(position, 48),
                "occupied target square must reject black double push");
    }

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
                                           cmz::vm2::chess1::kOutputPieceOffset + 5)
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

    for (std::int64_t source = 8; source < 56; ++source) {
        cmz::vm2::Chess1Board position{};
        position.side = Chess1Side::Black;
        position.squares[source] = Chess1Piece::BlackPawn;
        require(candidate_legal(position, source),
                "black pawn single push must be legal on interior ranks");

        const auto compiled = cmz::vm2::compile_chess1(position, source);
        const auto state = cmz::vm2::transition(compiled, compiled.tokens);
        for (std::int64_t square = 0; square < 64; ++square) {
            const auto expected = square == source
                                      ? Chess1Piece::Empty
                                      : (square == source - 8 ? Chess1Piece::BlackPawn
                                                              : position.squares[square]);
            const auto actual = state
                                    .index({cmz::vm2::chess1::kOutputSquareTokenBegin + square})
                                    .slice(0, cmz::vm2::chess1::kOutputPieceOffset,
                                           cmz::vm2::chess1::kOutputPieceOffset + 5)
                                    .argmax()
                                    .item<std::int64_t>();
            require(actual == static_cast<std::int64_t>(expected),
                    "black move must alter exactly source and south target");
        }
        require(state.index({cmz::vm2::chess1::kOutputSideToken,
                             cmz::vm2::chess1::kOutputSideOffset +
                                 static_cast<std::int64_t>(Chess1Side::White)})
                    .item<double>() == 1.0,
                "black legal move must emit white side-to-move");
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

    cmz::vm2::Chess1Board recurrent{};
    recurrent.squares[8] = Chess1Piece::WhitePawn;
    const auto recurrent_image = cmz::vm2::compile_chess1(recurrent, 8);
    const auto first = cmz::vm2::transition(recurrent_image, recurrent_image.tokens);
    require(first.index({cmz::vm2::chess1::kSquareTokenBegin + 8,
                         static_cast<std::int64_t>(Chess1Piece::Empty)})
                .item<double>() == 1.0,
            "output board must feed the recurrent input source square");
    require(first.index({cmz::vm2::chess1::kSquareTokenBegin + 16,
                         static_cast<std::int64_t>(Chess1Piece::WhitePawn)})
                .item<double>() == 1.0,
            "output board must feed the recurrent input target square");
    const auto second = cmz::vm2::transition(recurrent_image, first);
    require(torch::equal(
                first.slice(0, cmz::vm2::chess1::kSquareTokenBegin,
                            cmz::vm2::chess1::kSquareTokenBegin + 64)
                    .slice(1, 0, 5),
                second.slice(0, cmz::vm2::chess1::kSquareTokenBegin,
                             cmz::vm2::chess1::kSquareTokenBegin + 64)
                    .slice(1, 0, 5)),
            "repeating an exhausted selected move must preserve recurrent board bytes");
    require(torch::equal(
                first.index({cmz::vm2::chess1::kSideToken})
                    .slice(0, cmz::vm2::chess1::kInputSideOffset,
                           cmz::vm2::chess1::kInputSideOffset + 2),
                second.index({cmz::vm2::chess1::kSideToken})
                    .slice(0, cmz::vm2::chess1::kInputSideOffset,
                           cmz::vm2::chess1::kInputSideOffset + 2)),
            "repeating an exhausted selected move must preserve recurrent side bytes");
    require(second.index({cmz::vm2::chess1::kCandidateTokenBegin + 8,
                          cmz::vm2::chess1::kLegalOffset})
                .item<double>() == 1.0,
            "second inference trace must expose that the exhausted move is illegal");

    cmz::vm2::Chess1Board self_play{};
    self_play.squares[8] = Chess1Piece::WhitePawn;
    self_play.squares[48] = Chess1Piece::BlackPawn;
    const auto auto_image = cmz::vm2::compile_chess1_auto(self_play);
    const auto auto_white = cmz::vm2::transition(auto_image, auto_image.tokens);
    require(auto_white.index({cmz::vm2::chess1::kSquareTokenBegin + 16,
                              static_cast<std::int64_t>(Chess1Piece::WhitePawn)})
                .item<double>() == 1.0,
            "auto hardmax player must choose the lowest-index legal white pawn");
    const auto auto_black = cmz::vm2::transition(auto_image, auto_white);
    require(auto_black.index({cmz::vm2::chess1::kCandidateTokenBegin + 48,
                              cmz::vm2::chess1::kLegalOffset + 1})
                .item<double>() == 1.0,
            "black recurrent candidate must be legal before auto selection");
    require(auto_black.index({cmz::vm2::chess1::kSquareTokenBegin + 40,
                              static_cast<std::int64_t>(Chess1Piece::BlackPawn)})
                .item<double>() == 1.0,
            "next inference must choose a legal black pawn from recurrent state");
    const auto auto_white_again = cmz::vm2::transition(auto_image, auto_black);
    require(auto_white_again.index({cmz::vm2::chess1::kSquareTokenBegin + 24,
                                    static_cast<std::int64_t>(Chess1Piece::WhitePawn)})
                .item<double>() == 1.0,
            "third inference must continue token-selected recurrent self-play");
    return 0;
}
