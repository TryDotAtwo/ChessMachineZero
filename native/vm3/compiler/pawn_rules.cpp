#include <cmz_vm3/rule_compiler.h>

#include <cstdlib>

namespace cmz::vm3 {

torch::Tensor compile_pawn_geometry(const CandidateBank& bank) {
  auto result = torch::zeros({kMoveCandidateCount, 6}, torch::kFloat64);
  for (const auto& move : bank.moves) {
    const auto source_file = move.source % 8;
    const auto source_rank = move.source / 8;
    const auto target_file = move.target % 8;
    const auto target_rank = move.target / 8;
    const auto file_delta = target_file - source_file;
    const auto rank_delta = target_rank - source_rank;
    const auto id = move.id;
    const auto ordinary = move.promotion == Promotion::None;
    const auto white_choice = target_rank == 7 && !ordinary;
    const auto black_choice = target_rank == 0 && !ordinary;
    if (((target_rank != 7 && ordinary) || white_choice) &&
        file_delta == 0 && rank_delta == 1)
      result.index_put_({id, 0}, 1.0);
    if (ordinary && source_rank == 1 && file_delta == 0 && rank_delta == 2)
      result.index_put_({id, 1}, 1.0);
    if (((target_rank != 7 && ordinary) || white_choice) &&
        std::abs(file_delta) == 1 && rank_delta == 1)
      result.index_put_({id, 2}, 1.0);
    if (((target_rank != 0 && ordinary) || black_choice) &&
        file_delta == 0 && rank_delta == -1)
      result.index_put_({id, 3}, 1.0);
    if (ordinary && source_rank == 6 && file_delta == 0 && rank_delta == -2)
      result.index_put_({id, 4}, 1.0);
    if (((target_rank != 0 && ordinary) || black_choice) &&
        std::abs(file_delta) == 1 && rank_delta == -1)
      result.index_put_({id, 5}, 1.0);
  }
  return result;
}

}  // namespace cmz::vm3
