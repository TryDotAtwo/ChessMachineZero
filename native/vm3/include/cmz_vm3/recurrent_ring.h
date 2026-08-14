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

torch::Tensor compute_rule_legal(const FrozenChessProgram& program,
                                 const torch::Tensor& state_tokens);
torch::Tensor compute_rule_legal_batch(const FrozenChessProgram& program,
                                       const torch::Tensor& state_batch);

StepResult recurrent_step(const FrozenChessProgram& program,
                          PolicyPair policies,
                          const ChessRingState& state);

}  // namespace cmz::vm3
