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
    const auto merged = cmz::vm2::optimize_attention_blocks(
        image.attention_masks[2], 16, 4096.0);
    if (merged.blocks.empty() || merged.blocks.size() > 16) {
        throw std::runtime_error("optimizer must respect the block budget");
    }
    if (!cmz::vm2::covers_attention_mask_exactly(image.attention_masks[2], merged)) {
        throw std::runtime_error("merged plan must preserve every allowed and forbidden pair");
    }
    if (merged.padded_score_pairs < report.stages[2].useful_score_pairs ||
        merged.padded_score_pairs >= report.stages[2].dense_score_pairs) {
        throw std::runtime_error("merged plan must stay between useful and dense work");
    }
    const auto one_block = cmz::vm2::optimize_attention_blocks(
        image.attention_masks[2], 1, 4096.0);
    if (one_block.blocks.size() != 1 || one_block.estimated_cost <= merged.estimated_cost) {
        throw std::runtime_error("16-block plan must beat one padded block under launch model");
    }
    const auto blocks = cmz::vm2::materialize_attention_blocks(
        merged, cmz::vm2::chess1::kTokenCount, image.tokens.options());
    const auto dense = cmz::vm2::self_attention(
        image.tokens, image.weights.wq[2], image.weights.wk[2], image.weights.wv[2],
        image.attention_masks[2]);
    const auto block = cmz::vm2::block_self_attention(
        image.tokens, image.weights.wq[2], image.weights.wk[2], image.weights.wv[2], blocks);
    if (!torch::equal(dense.output, block.output) ||
        !torch::equal(dense.winners, block.winners)) {
        throw std::runtime_error("padded block execution must equal dense attention exactly");
    }
    std::cout << "stage2 budget=1 padded=" << one_block.padded_score_pairs
              << " cost=" << one_block.estimated_cost << '\n';
    std::cout << "stage2 budget=16 padded=" << merged.padded_score_pairs
              << " cost=" << merged.estimated_cost << '\n';
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
