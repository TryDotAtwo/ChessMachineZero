#include <stdexcept>
#include <fstream>
#include <sstream>

#include "cmz_vm2/chess1_compiler.h"
#include "cmz_vm2/trace_exporter.h"

int main() {
    cmz::vm2::Chess1Board board{};
    board.squares[12] = cmz::vm2::Chess1Piece::WhitePawn;
    const auto image = cmz::vm2::compile_chess1(board, 12);
    const auto trace = cmz::vm2::trace_transition(image, image.tokens);
    if (trace.stages.size() != cmz::vm2::kStageCount) {
        throw std::runtime_error("trace must expose all 15 inference stages");
    }
    auto state = image.tokens;
    for (std::int64_t stage = 0; stage < cmz::vm2::kStageCount; ++stage) {
        const auto& snapshot = trace.stages[stage];
        if (!torch::equal(snapshot.x, state)) {
            throw std::runtime_error("stage X must equal recurrent input state");
        }
        if (!torch::equal(snapshot.q, torch::matmul(snapshot.x, image.weights.wq[stage])) ||
            !torch::equal(snapshot.k, torch::matmul(snapshot.x, image.weights.wk[stage])) ||
            !torch::equal(snapshot.v, torch::matmul(snapshot.x, image.weights.wv[stage]))) {
            throw std::runtime_error("Q/K/V must be exact matrix products");
        }
        const auto expected_scores = torch::matmul(snapshot.q, snapshot.k.t()) + image.attention_masks[stage];
        if (!torch::equal(snapshot.scores, expected_scores)) {
            throw std::runtime_error("scores must equal QK^T + mask");
        }
        const auto expected_winners = std::get<1>(snapshot.scores.max(1, false));
        if (!torch::equal(snapshot.winners, expected_winners)) {
            throw std::runtime_error("winners must equal deterministic row argmax");
        }
        if (!torch::equal(snapshot.x_prime, snapshot.apply_write_reference(image, stage))) {
            throw std::runtime_error("XPrime must equal matrix write reference");
        }
        state = snapshot.x_prime;
    }
    if (!torch::equal(trace.final_state, state)) {
        throw std::runtime_error("trace final state must equal last XPrime");
    }
    const std::string path = "/tmp/cmz-matrix-trace.json";
    cmz::vm2::write_trace_json(path, image, image.tokens,
                               "0123456789abcdef0123456789abcdef01234567");
    std::ifstream stream(path);
    std::stringstream buffer;
    buffer << stream.rdbuf();
    const auto json = buffer.str();
    for (const auto* required : {"cmz.matrix-trace.v1", "\"Wq\"", "\"S\"",
                                 "\"XPrime\"", "\"writes\"", "\"changed\""}) {
        if (json.find(required) == std::string::npos) {
            throw std::runtime_error(std::string("exported JSON is missing ") + required);
        }
    }
    return 0;
}
