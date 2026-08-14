#include "cmz_vm2/legal_set.h"

namespace cmz::vm2 {

LegalSetAssembler compile_legal_set_assembler(
    std::int64_t pawn_candidates,
    std::int64_t knight_candidates,
    const torch::TensorOptions& options) {
    const auto total = pawn_candidates + knight_candidates;
    auto pawn_rows = torch::zeros({total, pawn_candidates}, options);
    auto knight_rows = torch::zeros({total, knight_candidates}, options);
    for (std::int64_t row = 0; row < pawn_candidates; ++row) {
        pawn_rows.index_put_({row, row}, 1.0);
    }
    for (std::int64_t row = 0; row < knight_candidates; ++row) {
        knight_rows.index_put_({pawn_candidates + row, row}, 1.0);
    }
    return LegalSetAssembler{
        pawn_rows.detach().set_requires_grad(false),
        knight_rows.detach().set_requires_grad(false)};
}

torch::Tensor assemble_legal_set(
    const LegalSetAssembler& assembler,
    const torch::Tensor& pawn_predicates,
    const torch::Tensor& knight_predicates) {
    return torch::matmul(assembler.pawn_rows, pawn_predicates) +
           torch::matmul(assembler.knight_rows, knight_predicates);
}

}  // namespace cmz::vm2
