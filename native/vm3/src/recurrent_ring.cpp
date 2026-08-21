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
  const auto castling = active.narrow(-1, kCastlingOffset, kCastlingRows);
  const auto raw_ep = active.narrow(-1, kRawEpOffset, kRawEpRows);
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
  const auto& special_rows = program.tensors.at("special_packet_rows");
  const auto ep_match = torch::matmul(
      program.tensors.at("candidate_ep_target"), raw_ep.unsqueeze(-1));
  memory = memory + ep_match.unsqueeze(-1) * special_rows.select(0, 0).unsqueeze(-1) *
                        program.tensors.at("ep_match_feature");
  const auto ep_captured = torch::matmul(
      program.tensors.at("candidate_ep_captured_square"), board);
  memory = memory +
      torch::matmul(ep_captured, program.tensors.at("piece_feature_router"))
              .unsqueeze(-2) * special_rows.select(0, 1).unsqueeze(-1);
  const auto rights = torch::matmul(castling, program.tensors.at("castling_decode"));
  const auto right_match = torch::matmul(
      program.tensors.at("candidate_castle_right"), rights.unsqueeze(-1));
  memory = memory + right_match.unsqueeze(-1) *
                        special_rows.select(0, 2).unsqueeze(-1) *
                        program.tensors.at("castle_right_feature");
  const auto castle_rook = torch::matmul(
      program.tensors.at("candidate_castle_rook_source"), board);
  memory = memory +
      torch::matmul(castle_rook, program.tensors.at("piece_feature_router"))
              .unsqueeze(-2) * special_rows.select(0, 3).unsqueeze(-1);
  for (std::int64_t slot = 0; slot < 3; ++slot) {
    const auto empty = torch::matmul(
        program.tensors.at("candidate_castle_empty_" + std::to_string(slot)), board)
                           .select(-1, 0);
    memory = memory + empty.unsqueeze(-1).unsqueeze(-1) *
                          special_rows.select(0, 4 + slot).unsqueeze(-1) *
                          program.tensors.at("packet_empty_feature");
  }
  return memory;
}

torch::Tensor execute_rule_query(const FrozenChessProgram& program,
                                 const torch::Tensor& state_batch) {
  auto memory = assemble_rule_memory_batch(program, state_batch);
  auto query = memory.select(-2, rule::kQueryRow);
  for (const auto& stage : program.pre_policy_stages)
    query = execute_packet_stage(query, memory, stage, 1.0);
  return query;
}

torch::Tensor attention_at_least(const FrozenChessProgram& program,
                                 const torch::Tensor& value,
                                 const char* frozen_keys) {
  const auto query = torch::stack({value, torch::ones_like(value)}, -1);
  const auto scores = torch::matmul(
      query.unsqueeze(-2), program.tensors.at(frozen_keys).transpose(0, 1));
  const auto selection = deterministic_st_select(
      scores, program.tensors.at("boolean_class_eligibility"), 1.0)
                             .straight_through;
  return torch::matmul(
             selection, program.tensors.at("boolean_class_value").unsqueeze(-1))
      .squeeze(-1).squeeze(-1);
}

torch::Tensor attention_not(const FrozenChessProgram& program,
                            const torch::Tensor& value) {
  return attention_at_least(program, 1.0 - value, "boolean_keys_half");
}

torch::Tensor finish_attack_reduction(const FrozenChessProgram& program,
                                      const torch::Tensor& board,
                                      const torch::Tensor& side,
                                      const torch::Tensor& king_square_selection,
                                      const torch::Tensor& blocker_count) {
  const auto white = side.select(-1, 0).unsqueeze(-1).unsqueeze(-1);
  const auto black = side.select(-1, 1).unsqueeze(-1).unsqueeze(-1);
  const auto opponent = [&](std::int64_t white_piece,
                            std::int64_t black_piece) {
    return white * board.select(-1, black_piece) +
           black * board.select(-1, white_piece);
  };
  const auto geometry = [&](const char* name) {
    return torch::matmul(
        king_square_selection,
        program.tensors.at(name).transpose(0, 1)).squeeze(-2);
  };
  const auto path_clear = attention_not(
      program, attention_at_least(
                   program, blocker_count, "boolean_keys_half"));
  const auto pawn_geometry = white * geometry("attack_pawn_black") +
                             black * geometry("attack_pawn_white");
  const auto pawn = attention_at_least(
      program, opponent(1, 7) + pawn_geometry, "boolean_keys_one_half");
  const auto knight = attention_at_least(program,
      opponent(2, 8) + geometry("attack_knight"), "boolean_keys_one_half");
  const auto king = attention_at_least(program,
      opponent(6, 12) + geometry("attack_king"), "boolean_keys_one_half");
  const auto rook = attention_at_least(program,
      opponent(4, 10) + opponent(5, 11) +
          geometry("attack_rook") + path_clear,
      "boolean_keys_two_half");
  const auto bishop = attention_at_least(program,
      opponent(3, 9) + opponent(5, 11) +
          geometry("attack_bishop") + path_clear,
      "boolean_keys_two_half");
  return attention_at_least(program,
      (pawn + knight + king + rook + bishop).sum(-1), "boolean_keys_half");
}

StSelection select_king_square(const FrozenChessProgram& program,
                               const torch::Tensor& board,
                               const torch::Tensor& side) {
  const auto white = side.select(-1, 0).unsqueeze(-1).unsqueeze(-1);
  const auto black = side.select(-1, 1).unsqueeze(-1).unsqueeze(-1);
  const auto own_king = white * board.select(-1, 6) +
                        black * board.select(-1, 12);
  const auto q = torch::matmul(
      own_king, program.tensors.at("attack_square_code")).unsqueeze(-2);
  const auto scores = torch::matmul(
      q, program.tensors.at("attack_square_code").transpose(0, 1));
  return deterministic_st_select(
      scores, program.tensors.at("attack_square_eligibility"), 1.0);
}

torch::Tensor own_king_attacked(const FrozenChessProgram& program,
                                const torch::Tensor& board,
                                const torch::Tensor& side,
                                const char*) {
  const auto selection = select_king_square(program, board, side);
  const auto occupied = 1.0 - board.select(-1, 0);
  const auto selected_between = torch::matmul(
      selection.straight_through,
      program.tensors.at("attack_between_all")
          .permute({1, 0, 2})
          .reshape({64, 64 * 64}))
                                    .view({board.size(0), board.size(1), 64, 64});
  const auto blocker_count =
      (selected_between * occupied.unsqueeze(-2)).sum(-1);
  return finish_attack_reduction(
      program, board, side, selection.straight_through, blocker_count);
}

torch::Tensor own_king_attacked_sparse_hard(const FrozenChessProgram& program,
                                             const torch::Tensor& board,
                                             const torch::Tensor& side,
                                             const char* offsets_name) {
  TORCH_CHECK(board.size(0) == 1,
              "sparse hard-forward attack inference requires one recurrent "
              "state");
  const auto selection = select_king_square(program, board, side);
  const auto target_ids = selection.hard_indices.squeeze(-1).flatten();
  const auto between_ids = torch::index_select(
      program.tensors.at("attack_between_ids"), 0, target_ids)
                               .view({board.size(1), 64, 6});
  const auto between_valid = torch::index_select(
      program.tensors.at("attack_between_valid"), 0, target_ids)
                                 .view({board.size(1), 64, 6});
  const auto flat_ids = between_ids + program.tensors.at(offsets_name);
  const auto occupied = 1.0 - board.select(-1, 0);
  const auto between_occupied = torch::index_select(
      occupied.flatten(), 0, flat_ids.flatten())
                                    .view({1, board.size(1), 64, 6});
  const auto blocker_count =
      (between_occupied * between_valid.unsqueeze(0)).sum(-1);
  return finish_attack_reduction(
      program, board, side, selection.hard, blocker_count);
}

using AttackEvaluator = torch::Tensor (*)(
    const FrozenChessProgram&, const torch::Tensor&, const torch::Tensor&,
    const char*);

TrialTransitionBatch compute_trial_transitions_batch_impl(
    const FrozenChessProgram& program, const torch::Tensor& batch,
    AttackEvaluator attack);

}  // namespace

torch::Tensor compute_rule_legal(const FrozenChessProgram& program,
                                 const torch::Tensor& state_tokens) {
  return compute_trial_transitions(program, state_tokens).legal;
}

torch::Tensor compute_rule_legal_batch(const FrozenChessProgram& program,
                                       const torch::Tensor& state_batch) {
  return compute_trial_transitions_batch(program, state_batch).legal;
}

TrialTransitionBatch compute_trial_transitions(
    const FrozenChessProgram& program, const torch::Tensor& state_tokens) {
  return compute_trial_transitions_batch(program, state_tokens.unsqueeze(0));
}

TrialTransitionBatch compute_trial_transitions_hard_forward(
    const FrozenChessProgram& program, const torch::Tensor& state_tokens) {
  return compute_trial_transitions_batch_impl(
      program, state_tokens.unsqueeze(0), own_king_attacked_sparse_hard);
}

TrialTransitionBatch compute_trial_transitions_batch(
    const FrozenChessProgram& program, const torch::Tensor& batch) {
  return compute_trial_transitions_batch_impl(program, batch, own_king_attacked);
}

namespace {
TrialTransitionBatch compute_trial_transitions_batch_impl(
    const FrozenChessProgram& program, const torch::Tensor& batch,
    AttackEvaluator attack) {
  const auto batch_size = batch.size(0);
  const auto active = batch.select(-1, kStateActiveFeature);
  const auto board = active.narrow(-1, kBoardOffset, kBoardRows)
                         .view({batch_size, 64, 13});
  const auto side = active.narrow(-1, kSideOffset, kSideRows);
  const auto query = execute_rule_query(program, batch);
  const auto pseudo = query.select(-1, rule::kPseudoLegal);
  const auto& source = program.tensors.at("candidate_source_square");
  const auto& target = program.tensors.at("candidate_target_square");
  const auto source_piece = torch::matmul(source, board);
  const auto target_piece = torch::matmul(target, board);
  const auto moving_piece = source_piece;
  const auto promotion_piece =
      (program.tensors.at("candidate_promotion_piece").unsqueeze(0) *
       side.unsqueeze(1).unsqueeze(-1)).sum(-2);
  const auto promotion_active =
      program.tensors.at("candidate_promotion_active").unsqueeze(0);
  const auto written_piece = (1.0 - promotion_active) * moving_piece +
                             promotion_active * promotion_piece;
  const auto source_mask = source.unsqueeze(0).unsqueeze(-1);
  const auto target_mask = target.unsqueeze(0).unsqueeze(-1);
  auto trial_board = (1.0 - source_mask - target_mask) * board.unsqueeze(1) +
                     source_mask * program.tensors.at("empty_piece") +
                     target_mask * written_piece.unsqueeze(-2);

  const auto ep_flag = query.select(-1, rule::kEpWLegal) +
                       query.select(-1, rule::kEpBLegal);
  const auto ep_mask = ep_flag.unsqueeze(-1).unsqueeze(-1) *
      program.tensors.at("candidate_ep_captured_square").unsqueeze(0).unsqueeze(-1);
  trial_board = (1.0 - ep_mask) * trial_board +
                ep_mask * program.tensors.at("empty_piece");

  const auto castle_flag = query.select(-1, rule::kCastleWLegal) +
                           query.select(-1, rule::kCastleBLegal);
  const auto rook_source =
      program.tensors.at("candidate_castle_rook_source").unsqueeze(0).unsqueeze(-1);
  const auto rook_target =
      program.tensors.at("candidate_castle_rook_target").unsqueeze(0).unsqueeze(-1);
  const auto rook_piece = torch::matmul(
      program.tensors.at("candidate_castle_rook_source"), board);
  const auto castle_source_mask = castle_flag.unsqueeze(-1).unsqueeze(-1) * rook_source;
  const auto castle_target_mask = castle_flag.unsqueeze(-1).unsqueeze(-1) * rook_target;
  trial_board = (1.0 - castle_source_mask - castle_target_mask) * trial_board +
                castle_source_mask * program.tensors.at("empty_piece") +
                castle_target_mask * rook_piece.unsqueeze(-2);

  const auto current_castling =
      active.narrow(-1, kCastlingOffset, kCastlingRows);
  const auto current_rights =
      torch::matmul(current_castling, program.tensors.at("castling_decode"));
  const auto source_clear =
      (source_piece.unsqueeze(-1) *
       program.tensors.at("candidate_source_right_clear").unsqueeze(0)).sum(-2);
  const auto target_clear =
      (target_piece.unsqueeze(-1) *
       program.tensors.at("candidate_target_right_clear").unsqueeze(0)).sum(-2);
  const auto next_rights = current_rights.unsqueeze(1) *
                           (1.0 - source_clear) * (1.0 - target_clear);
  const auto decode = program.tensors.at("castling_decode");
  const auto right_matches = next_rights.unsqueeze(-2) * decode +
      (1.0 - next_rights.unsqueeze(-2)) * (1.0 - decode);
  const auto next_castling = right_matches.prod(-1);

  const auto next_ep = program.tensors.at("candidate_next_ep").unsqueeze(0);
  const auto current_halfmove = active.narrow(-1, kHalfmoveOffset, kHalfmoveRows);
  const auto quiet_halfmove =
      torch::matmul(current_halfmove, program.tensors.at("halfmove_increment"));
  const auto pawn_move = source_piece.select(-1, 1) + source_piece.select(-1, 7);
  const auto capture = 1.0 - target_piece.select(-1, 0);
  const auto reset = 1.0 - (1.0 - pawn_move) * (1.0 - capture) * (1.0 - ep_flag);
  const auto next_halfmove = reset.unsqueeze(-1) *
          program.tensors.at("halfmove_reset") +
      (1.0 - reset).unsqueeze(-1) * quiet_halfmove.unsqueeze(1);

  const auto fullmove = active.narrow(-1, kFullmoveOffset, kFullmoveRows)
                            .view({batch_size, 2, 128});
  const auto black = side.select(-1, 1).unsqueeze(-1);
  const auto incremented_low =
      torch::matmul(fullmove.select(-2, 0), program.tensors.at("radix_increment"));
  const auto next_low = (1.0 - black) * fullmove.select(-2, 0) +
                        black * incremented_low;
  const auto carry = black * fullmove.select(-2, 0).select(-1, 127).unsqueeze(-1);
  const auto incremented_high =
      torch::matmul(fullmove.select(-2, 1), program.tensors.at("radix_increment"));
  const auto next_high = (1.0 - carry) * fullmove.select(-2, 1) +
                         carry * incremented_high;
  const auto next_fullmove = torch::stack({next_low, next_high}, -2)
                                 .unsqueeze(1)
                                 .expand({batch_size, kMoveCandidateCount, 2, 128});
  const auto final_attacked = attack(
      program, trial_board, side, "attack_offsets_candidates");
  const auto origin_attacked = attack(
      program, board.unsqueeze(1), side, "attack_offsets_single").squeeze(1);
  const auto castle_active =
      program.tensors.at("candidate_castle_active").transpose(0, 1);
  const auto& castle_ids = program.tensors.at("castle_candidate_ids");
  const auto castle_source = torch::index_select(source, 0, castle_ids);
  const auto castle_moving_piece = torch::index_select(source_piece, 1, castle_ids);
  const auto transit = torch::index_select(
      program.tensors.at("candidate_castle_transit"), 0, castle_ids)
                           .unsqueeze(0).unsqueeze(-1);
  const auto transit_source_mask = castle_source.unsqueeze(0).unsqueeze(-1);
  const auto transit_board =
      (1.0 - transit_source_mask - transit) * board.unsqueeze(1) +
      transit_source_mask * program.tensors.at("empty_piece") +
      transit * castle_moving_piece.unsqueeze(-2);
  const auto transit_attacked = torch::matmul(
      attack(program, transit_board, side, "attack_offsets_castles"),
      program.tensors.at("castle_candidate_route"));
  const auto castle_attacked = attention_at_least(
      program, origin_attacked.unsqueeze(-1) + transit_attacked,
      "boolean_keys_half");
  const auto castle_hazard = attention_at_least(
      program, castle_active + castle_attacked, "boolean_keys_one_half");
  const auto castle_safe = attention_not(program, castle_hazard);
  const auto legal = attention_at_least(
      program, pseudo + attention_not(program, final_attacked) + castle_safe,
      "boolean_keys_two_half");
  return {pseudo, legal, origin_attacked, final_attacked,
          trial_board, next_castling, next_ep,
          next_halfmove, next_fullmove};
}
}  // namespace

StepResult recurrent_step(const FrozenChessProgram& program,
                          PolicyPair policies,
                          const ChessRingState& state) {
  const auto trials = compute_trial_transitions(program, state.tokens);
  const auto legal = trials.legal;
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
  const auto selected_board =
      (real_selection.unsqueeze(-1).unsqueeze(-1) * trials.board).sum(1);
  const auto next_board = commit.unsqueeze(-1) * selected_board +
                          (1.0 - commit).unsqueeze(-1) * current_board;
  const auto select_trial = [&](const torch::Tensor& trial) {
    return (real_selection.unsqueeze(-1) * trial).sum(1);
  };
  const auto current_castling = castling_view(state.tokens).unsqueeze(0);
  const auto current_raw_ep = raw_ep_view(state.tokens).unsqueeze(0);
  const auto current_halfmove = halfmove_view(state.tokens).unsqueeze(0);
  const auto current_fullmove = fullmove_digits_view(state.tokens)
                                    .flatten()
                                    .unsqueeze(0);
  const auto selected_castling = select_trial(trials.castling);
  const auto selected_raw_ep = select_trial(trials.raw_ep);
  const auto selected_halfmove = select_trial(trials.halfmove);
  const auto selected_fullmove = select_trial(
      trials.fullmove_digits.flatten(-2, -1));
  const auto next_castling = commit * selected_castling +
                             (1.0 - commit) * current_castling;
  const auto next_raw_ep = commit * selected_raw_ep +
                           (1.0 - commit) * current_raw_ep;
  const auto next_halfmove = commit * selected_halfmove +
                             (1.0 - commit) * current_halfmove;
  const auto next_fullmove = commit * selected_fullmove +
                             (1.0 - commit) * current_fullmove;
  const auto flipped_side = torch::matmul(side, program.tensors.at("side_flip"));
  const auto next_side = commit * flipped_side + (1.0 - commit) * side;

  const auto board_delta = torch::matmul(
      (next_board - current_board).flatten(1),
      program.tensors.at("board_state_rows"));
  const auto side_delta = torch::matmul(
      next_side - side, program.tensors.at("side_state_rows"));
  const auto castling_delta = torch::matmul(
      next_castling - current_castling,
      program.tensors.at("castling_state_rows"));
  const auto raw_ep_delta = torch::matmul(
      next_raw_ep - current_raw_ep, program.tensors.at("raw_ep_state_rows"));
  const auto halfmove_delta = torch::matmul(
      next_halfmove - current_halfmove,
      program.tensors.at("halfmove_state_rows"));
  const auto fullmove_delta = torch::matmul(
      next_fullmove - current_fullmove,
      program.tensors.at("fullmove_state_rows"));
  const auto next_tokens =
      state.tokens.unsqueeze(0) +
      (board_delta + side_delta + castling_delta + raw_ep_delta +
       halfmove_delta + fullmove_delta).unsqueeze(-1) *
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
