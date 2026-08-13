#include <stdexcept>
#include <iostream>

#include "cmz_vm2/attention_cost.h"
#include "cmz_vm2/chess1_compiler.h"

int main() {
    cmz::vm2::Chess1Board board{};
    board.squares[12] = cmz::vm2::Chess1Piece::WhitePawn;
    const auto image = cmz::vm2::compile_chess1(board, 12);
    const auto report = cmz::vm2::analyze_attention_cost(image);
    if (report.stages.size() != cmz::vm2::kStageCount) {
        throw std::runtime_error("cost report must cover all stages");
    }
    const auto dense_pairs = cmz::vm2::kStageCount * cmz::vm2::chess1::kTokenCount *
                             cmz::vm2::chess1::kTokenCount;
    if (report.dense_score_pairs != dense_pairs || report.useful_score_pairs <= 0 ||
        report.useful_score_pairs >= report.dense_score_pairs) {
        throw std::runtime_error("cost report must prove a strict score-pair reduction");
    }
    std::int64_t stage_sum = 0;
    for (const auto& stage : report.stages) {
        if (stage.block_count <= 0 || stage.useful_score_pairs <= 0) {
            throw std::runtime_error("every stage needs a non-empty frozen block plan");
        }
        stage_sum += stage.useful_score_pairs;
    }
    if (stage_sum != report.useful_score_pairs) {
        throw std::runtime_error("stage costs must sum exactly");
    }
    std::cout << "dense_score_pairs=" << report.dense_score_pairs
              << " useful_score_pairs=" << report.useful_score_pairs
              << " reduction="
              << static_cast<double>(report.dense_score_pairs) /
                     static_cast<double>(report.useful_score_pairs)
              << '\n';
    for (const auto& stage : report.stages) {
        std::cout << "stage=" << stage.stage << " blocks=" << stage.block_count
                  << " useful=" << stage.useful_score_pairs << '\n';
    }
    return 0;
}
