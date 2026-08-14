#include <stdexcept>

#include "cmz_vm2/chess1_compiler.h"
#include "cmz_vm2/machine.h"

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

bool legal(const cmz::vm2::Chess1Board& board, std::int64_t source, bool right) {
    const auto candidate = (right ? 3 : 2) * 64 + source;
    const auto image = cmz::vm2::compile_chess1(board, candidate);
    const auto state = cmz::vm2::transition(image, image.tokens);
    return state.index({cmz::vm2::chess1::kCandidateTokenBegin + candidate,
                        cmz::vm2::chess1::kLegalOffset + 1})
               .item<double>() == 1.0;
}

}  // namespace

int main() {
    using cmz::vm2::Chess1Board;
    using cmz::vm2::Chess1Piece;
    using cmz::vm2::Chess1Side;

    const auto circuit = cmz::vm2::compile_chess1_circuit();

    Chess1Board white{};
    white.squares[27] = Chess1Piece::WhitePawn;
    white.squares[34] = Chess1Piece::BlackPawn;
    white.squares[36] = Chess1Piece::BlackOther;
    const auto bound = cmz::vm2::bind_chess1_input(circuit, white, 2 * 64 + 27);
    const auto fresh = cmz::vm2::compile_chess1(white, 2 * 64 + 27);
    require(torch::equal(bound.tokens, fresh.tokens),
            "binding board tokens must match fresh compilation byte-for-byte");
    for (std::int64_t stage = 0; stage < cmz::vm2::kStageCount; ++stage) {
        require(bound.weights.wq[stage].is_same(circuit.weights.wq[stage]),
                "bound image must share immutable weight storage");
        require(bound.attention_masks[stage].is_same(circuit.attention_masks[stage]),
                "bound image must share immutable mask storage");
    }
    require(legal(white, 27, false), "white left capture must be legal");
    require(legal(white, 27, true), "white right capture must be legal");

    const auto selected = 2 * 64 + 27;
    const auto image = cmz::vm2::compile_chess1(white, selected);
    const auto state = cmz::vm2::transition(image, image.tokens);
    require(state.index({cmz::vm2::chess1::kSquareTokenBegin + 27,
                         static_cast<std::int64_t>(Chess1Piece::Empty)})
                .item<double>() == 1.0,
            "capture must clear source through matrix write");
    require(state.index({cmz::vm2::chess1::kSquareTokenBegin + 34,
                         static_cast<std::int64_t>(Chess1Piece::WhitePawn)})
                .item<double>() == 1.0,
            "capture must replace target through matrix write");

    white.squares[34] = Chess1Piece::Empty;
    require(!legal(white, 27, false), "empty capture target must be illegal");
    white.squares[34] = Chess1Piece::WhiteOther;
    require(!legal(white, 27, false), "friendly capture target must be illegal");

    Chess1Board black{};
    black.side = Chess1Side::Black;
    black.squares[36] = Chess1Piece::BlackPawn;
    black.squares[27] = Chess1Piece::WhitePawn;
    black.squares[29] = Chess1Piece::WhiteOther;
    require(legal(black, 36, false), "black left capture must be legal");
    require(legal(black, 36, true), "black right capture must be legal");

    Chess1Board edges{};
    edges.squares[24] = Chess1Piece::WhitePawn;
    edges.squares[31] = Chess1Piece::WhitePawn;
    edges.squares[33] = Chess1Piece::BlackPawn;
    edges.squares[38] = Chess1Piece::BlackPawn;
    require(!legal(edges, 24, false), "file-a left capture must not wrap");
    require(!legal(edges, 31, true), "file-h right capture must not wrap");
    return 0;
}
