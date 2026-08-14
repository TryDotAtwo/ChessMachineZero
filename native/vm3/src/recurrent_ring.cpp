#include <cmz_vm3/recurrent_ring.h>

#include <cmz_vm3/rule_layout.h>
#include <cmz_vm3/st_selector.h>
#include <cmz_vm3/tensor_contract.h>
#include <cmz_vm3/trajectory.h>

namespace cmz::vm3 {
namespace {

torch::Tensor execute_packet_stage(const torch::Tensor& query,
                                   const torch::Tensor& memory,
                                   const FrozenStage& stage,
                                   double temperature) {
  const auto& block = stage.attention_plan.blocks.front();
  const auto key_tokens = torch::index_select(memory, -2, block.global_key_ids);
  const auto q = torch::matmul(query.unsqueeze(-2), stage.wq);
  const auto k = torch::matmul(key_tokens, stage.wk);
  const auto v = torch::matmul(key_tokens, stage.wv);
  const auto scores = torch::matmul(q, k.transpose(-2, -1)) + block.fixed_mask;
  const auto selection = deterministic_st_select(
      scores, block.fixed_eligibility, temperature).straight_through;
  return query + torch::matmul(
      torch::matmul(selection, v).squeeze(-2), stage.feature_router);
}

torch::Tensor assemble_rule_memory_batch(const FrozenChessProgram& program,
                                         const torch::Tensor& tokens) {
  const auto active = tokens.select(-1, kStateActiveFeature);
  const auto board = active.narrow(-1, kBoardOffset, kBoardRows)
                         .view({tokens.size(0), 64, 13});
  const auto side = active.narrow(-1, kSideOffset, kSideRows);
  const auto& source_square = program.tensors.at("candidate_source_square");
  const auto& target_square = program.tensors.at("candidate_target_square");
  const auto& middle_square = program.tensors.at("candidate_middle_square");
  const auto source_active = torch::matmul(source_square, board);
  const auto target_active = torch::matmul(target_square, board);
  const auto middle_active = torch::matmul(middle_square, board);
  const auto side_active = torch::matmul(
      program.tensors.at("candidate_broadcast"), side.unsqueeze(1));
  const auto geometry_features = torch::matmul(
      program.tensors.at("candidate_geometry"),
      program.tensors.at("geometry_feature_router"));
  const auto geometry_packet = torch::matmul(
      program.tensors.at("geometry_packet_row"), geometry_features.unsqueeze(-2));
  auto memory =
      (program.tensors.at("packet_template") + geometry_packet).unsqueeze(0);
  const auto& active_feature = program.tensors.at("packet_active_feature");
  memory = memory +
      torch::matmul(source_active, program.tensors.at("source_packet_rows"))
              .unsqueeze(-1) * active_feature;
  memory = memory +
      torch::matmul(target_active, program.tensors.at("target_packet_rows"))
              .unsqueeze(-1) * active_feature;
  memory = memory +
      torch::matmul(middle_active, program.tensors.at("middle_packet_rows"))
              .unsqueeze(-1) * active_feature;
  memory = memory +
      torch::matmul(side_active, program.tensors.at("side_packet_rows"))
              .unsqueeze(-1) * active_feature;
  const auto& between_rows = program.tensors.at("between_packet_rows");
  for (std::int64_t slot = 0; slot < 6; ++slot) {
    const auto between_active = torch::matmul(
        program.tensors.at("candidate_between_" + std::to_string(slot)), board);
    memory = memory +
        between_active.select(-1, 0).unsqueeze(-1).unsqueeze(-1) *
        between_rows.select(0, slot).unsqueeze(-1) *
        program.tensors.at("packet_empty_feature");
  }
  return memory;
}

}  // namespace

torch::Tensor compute_rule_legal(const FrozenChessProgram& program,
                                 const torch::Tensor& state_tokens) {
  return compute_rule_legal_batch(program, state_tokens.unsqueeze(0));
}

torch::Tensor compute_rule_legal_batch(const FrozenChessProgram& program,
                                       const torch::Tensor& state_batch) {
  auto memory = assemble_rule_memory_batch(program, state_batch);
  auto query = memory.select(-2, rule::kQueryRow);
  for (const auto& stage : program.pre_policy_stages)
    query = execute_packet_stage(query, memory, stage, 1.0);
  return query.select(-1, rule::kLegal);
}

StepResult recurrent_step(const FrozenChessProgram& program,
                          PolicyPair policies,
                          const ChessRingState& state) {
  const auto legal = compute_rule_legal(program, state.tokens);
  const auto& zero = program.tensors.at("zero_bit");
  PolicyInput input{state.tokens.unsqueeze(0),
                    program.tensors.at("candidate_tokens").unsqueeze(0),
                    legal,
                    zero,
                    zero * legal,
                    zero,
                    zero,
                    state.trajectory};
  const auto side = side_view(state.tokens).unsqueeze(0);
  auto routed = run_policy_pair(policies, input, side);
  const auto real_selection =
      routed.selected.move_selection.narrow(-1, 0, kMoveCandidateCount);
  const auto selected_move = torch::matmul(
      real_selection, program.tensors.at("candidate_descriptors"));
  const auto commit = routed.selected.control_selection.narrow(-1, 0, 1);
  const auto move_slot = make_training_move_slot(program, selected_move, commit);
  const auto policy_kv = encode_policy_pair_move(policies, side, move_slot);

  const auto current_board = board_state_view(state.tokens).unsqueeze(0);
  const auto source = torch::matmul(
      real_selection, program.tensors.at("candidate_source_square"));
  const auto target = torch::matmul(
      real_selection, program.tensors.at("candidate_target_square"));
  const auto moving_piece =
      torch::matmul(source.unsqueeze(1), current_board).squeeze(1);
  const auto updated_board =
      (1.0 - source - target).unsqueeze(-1) * current_board +
      source.unsqueeze(-1) * program.tensors.at("empty_piece") +
      target.unsqueeze(-1) * moving_piece.unsqueeze(1);
  const auto next_board = commit.unsqueeze(-1) * updated_board +
                          (1.0 - commit).unsqueeze(-1) * current_board;
  const auto flipped_side = torch::matmul(side, program.tensors.at("side_flip"));
  const auto next_side = commit * flipped_side + (1.0 - commit) * side;

  const auto board_delta = torch::matmul(
      (next_board - current_board).flatten(1),
      program.tensors.at("board_state_rows"));
  const auto side_delta = torch::matmul(
      next_side - side, program.tensors.at("side_state_rows"));
  const auto next_tokens =
      state.tokens.unsqueeze(0) +
      (board_delta + side_delta).unsqueeze(-1) *
          program.tensors.at("state_active_feature");
  auto trajectory = state.trajectory;
  trajectory.move_slots.push_back(move_slot);
  trajectory.policy_kv_slots.push_back(policy_kv);
  ChessRingState next{next_tokens.squeeze(0), std::move(trajectory), state.cursor};
  const auto terminal = terminal_view(next.tokens);
  return {std::move(next), legal, selected_move, terminal,
          std::move(routed.selected)};
}

}  // namespace cmz::vm3
