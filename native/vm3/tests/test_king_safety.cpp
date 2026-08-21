#include <cmz_vm3/chess_state.h>
#include <cmz_vm3/recurrent_ring.h>
#include <cmz_vm3/rule_compiler.h>

#include <torch/torch.h>

#include <iostream>

namespace {
bool require(bool condition, const char* message) {
  if (!condition) std::cerr << message << '\n';
  return condition;
}

cmz::vm3::InitialPosition empty_position() {
  cmz::vm3::InitialPosition position{};
  position.board.fill(cmz::vm3::PieceState::Empty);
  position.side = cmz::vm3::Side::White;
  position.ep_square = -1;
  position.fullmove = 1;
  position.claim_mode = cmz::vm3::ClaimMode::LiteralClaim;
  return position;
}

std::int64_t move_id(const cmz::vm3::CandidateBank& bank,
                     std::int64_t source, std::int64_t target) {
  for (const auto& move : bank.moves)
    if (move.source == source && move.target == target &&
        move.promotion == cmz::vm3::Promotion::None) return move.id;
  return -1;
}

bool bit(const torch::Tensor& tensor, std::int64_t candidate) {
  return tensor.index({0, candidate}).item<double>() == 1.0;
}
}  // namespace

int main(int argc, char**) {
  bool ok = true;
  const auto use_cuda = argc > 1;
  TORCH_CHECK(!use_cuda || torch::cuda::is_available(),
              "CUDA king-safety gate requested without GPU");
  const auto options = torch::TensorOptions().dtype(torch::kFloat64).device(
      use_cuda ? torch::kCUDA : torch::kCPU);
  const auto program = cmz::vm3::compile_minimal_rule_program(options);
  const auto bank = cmz::vm3::compile_candidate_bank();

  auto pin = empty_position();
  pin.board[4] = cmz::vm3::PieceState::WK;
  pin.board[12] = cmz::vm3::PieceState::WR;
  pin.board[60] = cmz::vm3::PieceState::BR;
  pin.board[56] = cmz::vm3::PieceState::BK;
  const auto pin_trial = cmz::vm3::compute_trial_transitions(
      program, cmz::vm3::bind_initial_state(program, pin).tokens);
  const auto pin_sparse = cmz::vm3::compute_trial_transitions_hard_forward(
      program, cmz::vm3::bind_initial_state(program, pin).tokens);
  ok &= require(torch::equal(pin_sparse.pseudo_legal, pin_trial.pseudo_legal) &&
                    torch::equal(pin_sparse.legal, pin_trial.legal) &&
                    torch::equal(pin_sparse.side_in_check,
                                 pin_trial.side_in_check) &&
                    torch::equal(pin_sparse.own_king_attacked,
                                 pin_trial.own_king_attacked),
                "sparse hard-forward attack path is exact with dense attention");

  auto batch_peer = empty_position();
  batch_peer.board[4] = cmz::vm3::PieceState::WK;
  batch_peer.board[20] = cmz::vm3::PieceState::BK;
  const auto batch_tokens = torch::stack(
      {cmz::vm3::bind_initial_state(program, pin).tokens,
       cmz::vm3::bind_initial_state(program, batch_peer).tokens});
  const auto dense_batch =
      cmz::vm3::compute_trial_transitions_batch(program, batch_tokens);
  const auto sparse_batch =
      cmz::vm3::compute_trial_transitions_hard_forward_batch(program,
                                                             batch_tokens);
  ok &= require(torch::equal(sparse_batch.legal, dense_batch.legal) &&
                    torch::equal(sparse_batch.board, dense_batch.board) &&
                    torch::equal(sparse_batch.castling,
                                 dense_batch.castling) &&
                    torch::equal(sparse_batch.raw_ep, dense_batch.raw_ep) &&
                    torch::equal(sparse_batch.halfmove,
                                 dense_batch.halfmove) &&
                    torch::equal(sparse_batch.fullmove_digits,
                                 dense_batch.fullmove_digits),
                "batched sparse hard-forward exactly matches dense trials");
  const auto e2f2 = move_id(bank, 12, 13);
  const auto e2e3 = move_id(bank, 12, 20);
  ok &= require(bit(pin_trial.pseudo_legal, e2f2) &&
                    !bit(pin_trial.legal, e2f2) && bit(pin_trial.legal, e2e3),
                "pin filters only moves that expose the own king");
  auto differentiable_pin = cmz::vm3::bind_initial_state(program, pin).tokens
                                .clone().set_requires_grad(true);
  const auto differentiable_trial = cmz::vm3::compute_trial_transitions(
      program, differentiable_pin);
  differentiable_trial.legal.index({0, e2e3}).backward();
  ok &= require(differentiable_pin.grad().defined() &&
                    differentiable_pin.grad().abs().sum().item<double>() > 0.0,
                "final legal attention selector backpropagates to canonical state");

  auto adjacent = empty_position();
  adjacent.board[4] = cmz::vm3::PieceState::WK;
  adjacent.board[20] = cmz::vm3::PieceState::BK;
  const auto adjacent_trial = cmz::vm3::compute_trial_transitions(
      program, cmz::vm3::bind_initial_state(program, adjacent).tokens);
  ok &= require(!bit(adjacent_trial.legal, move_id(bank, 4, 12)),
                "king adjacency is an opponent attack");

  auto ep = empty_position();
  ep.board[39] = cmz::vm3::PieceState::WK;  // h5
  ep.board[38] = cmz::vm3::PieceState::WP;  // g5
  ep.board[37] = cmz::vm3::PieceState::BP;  // f5
  ep.board[32] = cmz::vm3::PieceState::BR;  // a5
  ep.board[56] = cmz::vm3::PieceState::BK;
  ep.ep_square = 45;                         // f6
  const auto ep_trial = cmz::vm3::compute_trial_transitions(
      program, cmz::vm3::bind_initial_state(program, ep).tokens);
  const auto g5f6 = move_id(bank, 38, 45);
  ok &= require(bit(ep_trial.pseudo_legal, g5f6) && !bit(ep_trial.legal, g5f6),
                "en-passant discovered rook check is rejected from trial board");

  const auto castle_case = [&](std::int64_t attacking_rook,
                               const char* message) {
    auto castle = empty_position();
    castle.board[4] = cmz::vm3::PieceState::WK;
    castle.board[7] = cmz::vm3::PieceState::WR;
    castle.board[56] = cmz::vm3::PieceState::BK;
    castle.board[attacking_rook] = cmz::vm3::PieceState::BR;
    castle.castling = 1;
    const auto trial = cmz::vm3::compute_trial_transitions(
        program, cmz::vm3::bind_initial_state(program, castle).tokens);
    const auto id = move_id(bank, 4, 6);
    return require(bit(trial.pseudo_legal, id) && !bit(trial.legal, id), message);
  };
  ok &= castle_case(60, "castling from check is rejected");
  ok &= castle_case(61, "castling through check is rejected");
  ok &= castle_case(62, "castling into check is rejected");

  auto safe_castle = empty_position();
  safe_castle.board[4] = cmz::vm3::PieceState::WK;
  safe_castle.board[7] = cmz::vm3::PieceState::WR;
  safe_castle.board[56] = cmz::vm3::PieceState::BK;
  safe_castle.castling = 1;
  const auto safe_trial = cmz::vm3::compute_trial_transitions(
      program, cmz::vm3::bind_initial_state(program, safe_castle).tokens);
  ok &= require(bit(safe_trial.legal, move_id(bank, 4, 6)),
                "safe castling remains legal");

  return ok ? 0 : 1;
}
