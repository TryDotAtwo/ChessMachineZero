#pragma once

#include <cmz_vm3/chess_state.h>
#include <cmz_vm3/policy.h>
#include <cmz_vm3/program_image.h>

namespace cmz::vm3 {

struct StepResult {
  ChessRingState next;
  torch::Tensor legal;
  torch::Tensor selected_move;
  torch::Tensor terminal;
  PolicyOutput policy;
};

StepResult recurrent_step(const FrozenChessProgram& program,
                          PolicyPair policies,
                          const ChessRingState& state);

}  // namespace cmz::vm3
