#pragma once

#include <array>
#include <cstdint>

#include <torch/torch.h>

namespace cmz::vm2 {

inline constexpr std::int64_t kKnightStages = 10;

enum class KnightPiece : std::int64_t {
    Empty = 0, WhiteKnight = 1, BlackKnight = 2, WhiteOther = 3, BlackOther = 4,
};
enum class KnightSide : std::int64_t { White = 0, Black = 1 };

struct KnightBoard {
    std::array<KnightPiece, 64> squares{};
    KnightSide side = KnightSide::White;
};

struct KnightCircuit {
    torch::Tensor tokens;
    std::array<torch::Tensor, kKnightStages> wq;
    std::array<torch::Tensor, kKnightStages> wk;
    std::array<torch::Tensor, kKnightStages> wv;
    std::array<torch::Tensor, kKnightStages> masks;
    std::array<torch::Tensor, kKnightStages> row_writes;
    std::array<torch::Tensor, kKnightStages> feature_writes;
    torch::Tensor legal_row;
    torch::Tensor legal_feature;
    torch::Tensor legality_features;
    torch::Tensor board_rows;
    torch::Tensor board_features;
    torch::Tensor side_row;
    torch::Tensor side_features;
};

KnightCircuit compile_knight_circuit();
torch::Tensor bind_knight_input(
    const KnightCircuit& circuit,
    const KnightBoard& board,
    std::int64_t source,
    std::int64_t target);
torch::Tensor run_knight_circuit(const KnightCircuit& circuit, const torch::Tensor& input_tokens);
torch::Tensor select_knight_legal(const KnightCircuit& circuit, const torch::Tensor& state);
torch::Tensor select_knight_legality_tokens(
    const KnightCircuit& circuit, const torch::Tensor& state);
torch::Tensor select_knight_board(const KnightCircuit& circuit, const torch::Tensor& state);
torch::Tensor select_knight_side(const KnightCircuit& circuit, const torch::Tensor& state);

}  // namespace cmz::vm2
