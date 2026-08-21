#include <cmz_vm3/chess_state.h>
#include <cmz_vm3/policy.h>
#include <cmz_vm3/recurrent_ring.h>
#include <cmz_vm3/rule_compiler.h>

#include <torch/torch.h>

#include <array>
#include <cstdint>
#include <iostream>

namespace {
bool require(bool condition, const char* message) {
  if (!condition) std::cerr << message << '\n';
  return condition;
}

cmz::vm3::InitialPosition empty_position(cmz::vm3::Side side) {
  cmz::vm3::InitialPosition position{};
  position.board.fill(cmz::vm3::PieceState::Empty);
  position.side = side;
  position.castling = 0;
  position.ep_square = -1;
  position.halfmove = 0;
  position.fullmove = 1;
  position.claim_mode = cmz::vm3::ClaimMode::LiteralClaim;
  return position;
}

std::int64_t move_id(const cmz::vm3::CandidateBank& bank,
                     std::int64_t source,
                     std::int64_t target,
                     cmz::vm3::Promotion promotion = cmz::vm3::Promotion::None) {
  for (const auto& move : bank.moves)
    if (move.source == source && move.target == target &&
        move.promotion == promotion) return move.id;
  return -1;
}

std::int64_t selected(const torch::Tensor& one_hot) {
  return one_hot.argmax(-1).item<std::int64_t>();
}

cmz::vm3::PieceState trial_piece(const cmz::vm3::TrialTransitionBatch& trial,
                                 std::int64_t candidate,
                                 std::int64_t square) {
  return static_cast<cmz::vm3::PieceState>(
      selected(trial.board.index({0, candidate, square})));
}
}  // namespace

int main(int argc, char**) {
  bool ok = true;
  const auto use_cuda = argc > 1;
  TORCH_CHECK(!use_cuda || torch::cuda::is_available(),
              "CUDA special-move gate requested without GPU");
  const auto options = torch::TensorOptions().dtype(torch::kFloat64).device(
      use_cuda ? torch::kCUDA : torch::kCPU);
  const auto program = cmz::vm3::compile_minimal_rule_program(options);
  const auto bank = cmz::vm3::compile_candidate_bank();

  auto white_castle = empty_position(cmz::vm3::Side::White);
  white_castle.board[4] = cmz::vm3::PieceState::WK;
  white_castle.board[0] = cmz::vm3::PieceState::WR;
  white_castle.board[7] = cmz::vm3::PieceState::WR;
  white_castle.board[60] = cmz::vm3::PieceState::BK;
  white_castle.castling = 3;
  auto white_trial = cmz::vm3::compute_trial_transitions(
      program, cmz::vm3::bind_initial_state(program, white_castle).tokens);
  const auto e1g1 = move_id(bank, 4, 6);
  const auto e1c1 = move_id(bank, 4, 2);
  ok &= require(white_trial.pseudo_legal.index({0, e1g1}).item<double>() == 1.0 &&
                    white_trial.pseudo_legal.index({0, e1c1}).item<double>() == 1.0,
                "both white castling geometries are pseudo legal");
  ok &= require(trial_piece(white_trial, e1g1, 4) == cmz::vm3::PieceState::Empty &&
                    trial_piece(white_trial, e1g1, 7) == cmz::vm3::PieceState::Empty &&
                    trial_piece(white_trial, e1g1, 6) == cmz::vm3::PieceState::WK &&
                    trial_piece(white_trial, e1g1, 5) == cmz::vm3::PieceState::WR,
                "white king-side trial moves both king and rook");
  ok &= require(selected(white_trial.castling.index({0, e1g1})) == 0,
                "king move clears both white castling rights");

  auto black_castle = empty_position(cmz::vm3::Side::Black);
  black_castle.board[60] = cmz::vm3::PieceState::BK;
  black_castle.board[56] = cmz::vm3::PieceState::BR;
  black_castle.board[63] = cmz::vm3::PieceState::BR;
  black_castle.board[4] = cmz::vm3::PieceState::WK;
  black_castle.castling = 12;
  const auto black_trial = cmz::vm3::compute_trial_transitions(
      program, cmz::vm3::bind_initial_state(program, black_castle).tokens);
  const auto e8g8 = move_id(bank, 60, 62);
  const auto e8c8 = move_id(bank, 60, 58);
  ok &= require(black_trial.pseudo_legal.index({0, e8g8}).item<double>() == 1.0 &&
                    black_trial.pseudo_legal.index({0, e8c8}).item<double>() == 1.0,
                "both black castling geometries are pseudo legal");

  auto ep = empty_position(cmz::vm3::Side::White);
  ep.board[4] = cmz::vm3::PieceState::WK;
  ep.board[60] = cmz::vm3::PieceState::BK;
  ep.board[36] = cmz::vm3::PieceState::WP;  // e5
  ep.board[35] = cmz::vm3::PieceState::BP;  // d5
  ep.ep_square = 43;                         // d6
  ep.halfmove = 17;
  const auto ep_trial = cmz::vm3::compute_trial_transitions(
      program, cmz::vm3::bind_initial_state(program, ep).tokens);
  const auto e5d6 = move_id(bank, 36, 43);
  ok &= require(ep_trial.pseudo_legal.index({0, e5d6}).item<double>() == 1.0 &&
                    trial_piece(ep_trial, e5d6, 36) == cmz::vm3::PieceState::Empty &&
                    trial_piece(ep_trial, e5d6, 35) == cmz::vm3::PieceState::Empty &&
                    trial_piece(ep_trial, e5d6, 43) == cmz::vm3::PieceState::WP,
                "en-passant trial clears the off-target captured pawn");
  ok &= require(selected(ep_trial.halfmove.index({0, e5d6})) == 0,
                "en-passant resets halfmove clock");
  auto differentiable_ep =
      cmz::vm3::bind_initial_state(program, ep).tokens.clone().set_requires_grad(true);
  const auto differentiable_trial =
      cmz::vm3::compute_trial_transitions(program, differentiable_ep);
  const auto special_loss = differentiable_trial.pseudo_legal.index({0, e5d6}) +
                            differentiable_trial.board.index({0, e5d6, 35, 0});
  special_loss.backward();
  ok &= require(differentiable_ep.grad().defined() &&
                    differentiable_ep.grad().abs().sum().item<double>() > 0.0,
                "special trial backward reaches canonical input state");

  auto promotion = empty_position(cmz::vm3::Side::White);
  promotion.board[4] = cmz::vm3::PieceState::WK;
  promotion.board[60] = cmz::vm3::PieceState::BK;
  promotion.board[48] = cmz::vm3::PieceState::WP;
  const auto promotion_trial = cmz::vm3::compute_trial_transitions(
      program, cmz::vm3::bind_initial_state(program, promotion).tokens);
  const std::array<std::pair<cmz::vm3::Promotion, cmz::vm3::PieceState>, 4> choices{{
      {cmz::vm3::Promotion::Queen, cmz::vm3::PieceState::WQ},
      {cmz::vm3::Promotion::Rook, cmz::vm3::PieceState::WR},
      {cmz::vm3::Promotion::Bishop, cmz::vm3::PieceState::WB},
      {cmz::vm3::Promotion::Knight, cmz::vm3::PieceState::WN}}};
  for (const auto& [promotion_kind, expected_piece] : choices) {
    const auto id = move_id(bank, 48, 56, promotion_kind);
    ok &= require(promotion_trial.pseudo_legal.index({0, id}).item<double>() == 1.0 &&
                      trial_piece(promotion_trial, id, 56) == expected_piece,
                  "promotion trial writes the selected piece");
  }
  auto promotion_only = promotion;
  promotion_only.board[4] = cmz::vm3::PieceState::Empty;
  auto promotion_state = cmz::vm3::bind_initial_state(program, promotion_only);
  cmz::vm3::ZeroPolicy white(cmz::vm3::policy_routing_from_program(program));
  cmz::vm3::ZeroPolicy black(cmz::vm3::policy_routing_from_program(program));
  cmz::vm3::PolicyPair policies{white, black};
  const auto committed_promotion =
      cmz::vm3::recurrent_step(program, policies, promotion_state);
  const auto committed_board =
      cmz::vm3::board_state_view(committed_promotion.next.tokens).argmax(-1);
  ok &= require(committed_promotion.selected_move.index({0, 2}).item<double>() ==
                        static_cast<double>(cmz::vm3::Promotion::Queen) &&
                    committed_board.index({48}).item<std::int64_t>() == 0 &&
                    committed_board.index({56}).item<std::int64_t>() ==
                        static_cast<std::int64_t>(cmz::vm3::PieceState::WQ) &&
                    cmz::vm3::halfmove_view(committed_promotion.next.tokens)
                            .argmax(-1).item<std::int64_t>() == 0,
                "recurrent commit selects the complete promotion trial state");

  auto metadata = empty_position(cmz::vm3::Side::White);
  metadata.board[4] = cmz::vm3::PieceState::WK;
  metadata.board[60] = cmz::vm3::PieceState::BK;
  metadata.board[12] = cmz::vm3::PieceState::WP;  // e2
  metadata.board[0] = cmz::vm3::PieceState::WR;
  metadata.board[63] = cmz::vm3::PieceState::BR;
  metadata.castling = 11;
  metadata.halfmove = 41;
  metadata.fullmove = 77;
  const auto metadata_trial = cmz::vm3::compute_trial_transitions(
      program, cmz::vm3::bind_initial_state(program, metadata).tokens);
  const auto e2e4 = move_id(bank, 12, 28);
  const auto a1a2 = move_id(bank, 0, 8);
  ok &= require(selected(metadata_trial.raw_ep.index({0, e2e4})) == 21 &&
                    selected(metadata_trial.halfmove.index({0, e2e4})) == 0,
                "double pawn push exposes e3 and resets clock");
  ok &= require(selected(metadata_trial.castling.index({0, a1a2})) == 9,
                "original rook move clears only its own right");
  const auto materialized = cmz::vm3::materialize_selected_trial_state(
      program,
      cmz::vm3::bind_initial_state(program, metadata).tokens.unsqueeze(0),
      metadata_trial,
      torch::one_hot(torch::tensor({e2e4}, options.dtype(torch::kInt64)),
                     cmz::vm3::kMoveCandidateCount).to(options),
      torch::ones({1, 1}, options));
  const auto materialized_board = cmz::vm3::board_state_view(materialized.squeeze(0))
                                      .argmax(-1);
  ok &= require(materialized_board.index({12}).item<std::int64_t>() == 0 &&
                    materialized_board.index({28}).item<std::int64_t>() ==
                        static_cast<std::int64_t>(cmz::vm3::PieceState::WP) &&
                    selected(cmz::vm3::raw_ep_view(materialized.squeeze(0))) == 21 &&
                    selected(cmz::vm3::side_view(materialized.squeeze(0))) ==
                        static_cast<std::int64_t>(cmz::vm3::Side::Black),
                "tensor materializer commits the selected trial into same ABI state");

  auto capture_rook = empty_position(cmz::vm3::Side::White);
  capture_rook.board[4] = cmz::vm3::PieceState::WK;
  capture_rook.board[60] = cmz::vm3::PieceState::BK;
  capture_rook.board[7] = cmz::vm3::PieceState::WR;
  capture_rook.board[63] = cmz::vm3::PieceState::BR;
  capture_rook.castling = 5;
  capture_rook.halfmove = 9;
  const auto capture_trial = cmz::vm3::compute_trial_transitions(
      program, cmz::vm3::bind_initial_state(program, capture_rook).tokens);
  const auto h1h8 = move_id(bank, 7, 63);
  ok &= require(selected(capture_trial.castling.index({0, h1h8})) == 0 &&
                    selected(capture_trial.halfmove.index({0, h1h8})) == 0,
                "original-rook capture clears both affected rights and clock");

  auto black_quiet = empty_position(cmz::vm3::Side::Black);
  black_quiet.board[4] = cmz::vm3::PieceState::WK;
  black_quiet.board[60] = cmz::vm3::PieceState::BK;
  black_quiet.board[57] = cmz::vm3::PieceState::BN;
  black_quiet.halfmove = 12;
  black_quiet.fullmove = 77;
  const auto quiet_trial = cmz::vm3::compute_trial_transitions(
      program, cmz::vm3::bind_initial_state(program, black_quiet).tokens);
  const auto b8a6 = move_id(bank, 57, 40);
  ok &= require(selected(quiet_trial.halfmove.index({0, b8a6})) == 13 &&
                    selected(quiet_trial.fullmove_digits.index({0, b8a6, 0})) == 78 % 128 &&
                    selected(quiet_trial.fullmove_digits.index({0, b8a6, 1})) == 78 / 128,
                "black quiet trial increments both clocks exactly");

  return ok ? 0 : 1;
}
