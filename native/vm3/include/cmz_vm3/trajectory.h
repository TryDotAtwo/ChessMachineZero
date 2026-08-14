#pragma once

#include <cmz_vm3/chess_state.h>

namespace cmz::vm3 {

struct MoveTokenFields {
  torch::Tensor source;
  torch::Tensor target;
  torch::Tensor promotion;
  torch::Tensor derived_flags;
  torch::Tensor ply_address;
  torch::Tensor mover_side;
  torch::Tensor post_position_identity;
};

MoveSlot make_training_move_slot(const FrozenChessProgram& program,
                                 const torch::Tensor& emitted_move,
                                 const torch::Tensor& commit_predicate);

}  // namespace cmz::vm3
