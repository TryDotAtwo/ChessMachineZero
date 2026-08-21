#include <cmz_vm3/chess_state.h>
#include <cmz_vm3/policy.h>
#include <cmz_vm3/recurrent_ring.h>
#include <cmz_vm3/rule_compiler.h>

#include <torch/torch.h>

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

std::int64_t selected(const torch::Tensor& one_hot) {
  return one_hot.argmax(-1).item<std::int64_t>();
}
}  // namespace

int main(int argc, char**) {
  bool ok = true;
  const auto use_cuda = argc > 1;
  TORCH_CHECK(!use_cuda || torch::cuda::is_available(),
              "CUDA terminal gate requested without GPU");
  const auto options = torch::TensorOptions().dtype(torch::kFloat64).device(
      use_cuda ? torch::kCUDA : torch::kCPU);
  const auto program = cmz::vm3::compile_minimal_rule_program(options);
  cmz::vm3::ZeroPolicy white(cmz::vm3::policy_routing_from_program(program));
  cmz::vm3::ZeroPolicy black(cmz::vm3::policy_routing_from_program(program));
  cmz::vm3::PolicyPair policies{white, black};

  auto black_mated = empty_position(cmz::vm3::Side::Black);
  black_mated.board[63] = cmz::vm3::PieceState::BK;  // h8
  black_mated.board[54] = cmz::vm3::PieceState::WQ;  // g7
  black_mated.board[45] = cmz::vm3::PieceState::WK;  // f6
  const auto mate = cmz::vm3::recurrent_step(
      program, policies, cmz::vm3::bind_initial_state(program, black_mated));
  ok &= require(selected(mate.terminal) ==
                    static_cast<std::int64_t>(
                        cmz::vm3::TerminalKind::BlackCheckmated) &&
                    selected(cmz::vm3::result_view(mate.next.tokens)) ==
                    static_cast<std::int64_t>(cmz::vm3::GameResult::WhiteWin),
                "no legal moves while black is checked routes to black checkmate");
  ok &= require(mate.next.trajectory.move_slots.size() == 1 &&
                    mate.next.trajectory.move_slots[0].active.item<double>() == 0.0,
                "terminal routing appends no active MOVE token");

  auto stalemated = empty_position(cmz::vm3::Side::Black);
  stalemated.board[63] = cmz::vm3::PieceState::BK;  // h8
  stalemated.board[46] = cmz::vm3::PieceState::WQ;  // g6
  stalemated.board[45] = cmz::vm3::PieceState::WK;  // f6
  const auto stale = cmz::vm3::recurrent_step(
      program, policies, cmz::vm3::bind_initial_state(program, stalemated));
  ok &= require(selected(stale.terminal) ==
                    static_cast<std::int64_t>(
                        cmz::vm3::TerminalKind::Stalemate) &&
                    selected(cmz::vm3::result_view(stale.next.tokens)) ==
                    static_cast<std::int64_t>(cmz::vm3::GameResult::Draw),
                "no legal moves without check routes to stalemate draw");

  auto clock_draw = empty_position(cmz::vm3::Side::White);
  clock_draw.board[4] = cmz::vm3::PieceState::WK;
  clock_draw.board[60] = cmz::vm3::PieceState::BK;
  clock_draw.halfmove = cmz::vm3::kHalfmoveAutomaticLimit;
  const auto seventy_five = cmz::vm3::recurrent_step(
      program, policies, cmz::vm3::bind_initial_state(program, clock_draw));
  ok &= require(selected(seventy_five.terminal) ==
                    static_cast<std::int64_t>(
                        cmz::vm3::TerminalKind::SeventyFiveMove) &&
                    selected(cmz::vm3::result_view(seventy_five.next.tokens)) ==
                    static_cast<std::int64_t>(cmz::vm3::GameResult::Draw),
                "halfmove 150 routes to automatic seventy-five-move draw");

  auto absorbed = seventy_five.next;
  const auto again = cmz::vm3::recurrent_step(program, policies, absorbed);
  ok &= require(torch::equal(again.next.tokens, absorbed.tokens) &&
                    again.next.trajectory.move_slots.back().active.item<double>() ==
                    0.0,
                "terminal state is absorbing and appends only inactive padding");

  return ok ? 0 : 1;
}
