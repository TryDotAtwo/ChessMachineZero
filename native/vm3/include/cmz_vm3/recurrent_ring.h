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

struct TrialTransitionBatch {
  torch::Tensor pseudo_legal;
  torch::Tensor board;
  torch::Tensor castling;
  torch::Tensor raw_ep;
  torch::Tensor halfmove;
  torch::Tensor fullmove_digits;
};

TrialTransitionBatch compute_trial_transitions(
    const FrozenChessProgram& program, const torch::Tensor& state_tokens);

StepResult recurrent_step(const FrozenChessProgram& program,
                          PolicyPair policies,
                          const ChessRingState& state);

}  // namespace cmz::vm3
