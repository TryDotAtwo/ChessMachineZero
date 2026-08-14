#include <cmz_vm3/trajectory.h>

namespace cmz::vm3 {

MoveSlot make_training_move_slot(const FrozenChessProgram& program,
                                 const torch::Tensor& emitted_move,
                                 const torch::Tensor& commit_predicate) {
  const auto& pad = program.tensors.at("trajectory_pad_token");
  const auto move = commit_predicate * emitted_move +
                    (1.0 - commit_predicate) * pad;
  return {move, commit_predicate};
}

}  // namespace cmz::vm3
