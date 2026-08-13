#pragma once

#include <array>
#include <cstdint>

#include <torch/torch.h>

namespace cmz::vm2 {

inline constexpr std::int64_t kKnightStages = 3;

struct KnightCircuit {
    torch::Tensor tokens;
    std::array<torch::Tensor, kKnightStages> wq;
    std::array<torch::Tensor, kKnightStages> wk;
    std::array<torch::Tensor, kKnightStages> wv;
    std::array<torch::Tensor, kKnightStages> masks;
    std::array<torch::Tensor, kKnightStages> row_writes;
    std::array<torch::Tensor, kKnightStages> feature_writes;
    torch::Tensor output_row;
    torch::Tensor output_feature;
};

KnightCircuit compile_knight_circuit();
torch::Tensor bind_knight_pair(const KnightCircuit& circuit, std::int64_t source, std::int64_t target);
torch::Tensor run_knight_circuit(const KnightCircuit& circuit, const torch::Tensor& input_tokens);

}  // namespace cmz::vm2
