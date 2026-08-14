#pragma once

#include <torch/torch.h>

namespace cmz::vm2 {

struct LegalSetAssembler {
    torch::Tensor pawn_rows;
    torch::Tensor knight_rows;
};

LegalSetAssembler compile_legal_set_assembler(
    std::int64_t pawn_candidates,
    std::int64_t knight_candidates,
    const torch::TensorOptions& options = torch::TensorOptions().dtype(torch::kFloat64));

torch::Tensor assemble_legal_set(
    const LegalSetAssembler& assembler,
    const torch::Tensor& pawn_predicates,
    const torch::Tensor& knight_predicates);

}  // namespace cmz::vm2
