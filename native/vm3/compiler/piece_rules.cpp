#include <cmz_vm3/rule_compiler.h>

#include <cstdlib>

namespace cmz::vm3 {

torch::Tensor compile_knight_geometry(const CandidateBank& bank) {
  auto result = torch::zeros({kMoveCandidateCount, 1}, torch::kFloat64);
  for (const auto& move : bank.moves) {
    if (move.promotion != Promotion::None) continue;
    const auto file_delta = std::abs(move.target % 8 - move.source % 8);
    const auto rank_delta = std::abs(move.target / 8 - move.source / 8);
    if ((file_delta == 1 && rank_delta == 2) ||
        (file_delta == 2 && rank_delta == 1)) {
      result.index_put_({move.id, 0}, 1.0);
    }
  }
  return result;
}

}  // namespace cmz::vm3
