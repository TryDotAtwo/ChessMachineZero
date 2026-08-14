#include <cmz_vm3/chess_state.h>
#include <cmz_vm3/recurrent_ring.h>
#include <cmz_vm3/rule_compiler.h>

#include <torch/torch.h>

#include <cstdlib>
#include <iostream>
#include <set>

namespace {
bool require(bool condition, const char* message) {
  if (!condition) std::cerr << message << '\n';
  return condition;
}

cmz::vm3::InitialPosition single_piece(cmz::vm3::PieceState piece,
                                       std::int64_t square,
                                       cmz::vm3::Side side) {
  cmz::vm3::InitialPosition position{};
  position.board.fill(cmz::vm3::PieceState::Empty);
  position.board[static_cast<std::size_t>(square)] = piece;
  position.side = side;
  position.castling = 0;
  position.ep_square = -1;
  position.fullmove = 1;
  position.claim_mode = cmz::vm3::ClaimMode::LiteralClaim;
  return position;
}

std::set<std::int64_t> targets(const torch::Tensor& legal, std::int64_t source) {
  std::set<std::int64_t> result;
  const auto cpu = legal.to(torch::kCPU).squeeze(0);
  for (std::int64_t target = 0; target < 64; ++target) {
    if (cpu.index({source * 64 + target}).item<double>() == 1.0)
      result.insert(target);
  }
  return result;
}

std::set<std::int64_t> compiled_geometry_targets(
    const cmz::vm3::FrozenChessProgram& program,
    std::int64_t source,
    cmz::vm3::PieceState piece) {
  std::set<std::int64_t> result;
  const auto geometry = program.tensors.at("candidate_geometry").to(torch::kCPU);
  const auto piece_id = static_cast<std::int64_t>(piece);
  const auto column = piece_id == 2 || piece_id == 8
                          ? 0
                          : (piece_id >= 3 && piece_id <= 6 ? piece_id + 4
                                                            : piece_id + 2);
  for (std::int64_t target = 0; target < 64; ++target) {
    if (geometry.index({source * 64 + target, column}).item<double>() == 1.0)
      result.insert(target);
  }
  return result;
}

std::set<std::int64_t> leaper_targets(std::int64_t source,
                                      const std::set<std::pair<int, int>>& deltas) {
  std::set<std::int64_t> result;
  const auto file = source % 8;
  const auto rank = source / 8;
  for (const auto [df, dr] : deltas) {
    const auto f = file + df;
    const auto r = rank + dr;
    if (f >= 0 && f < 8 && r >= 0 && r < 8) result.insert(f + 8 * r);
  }
  return result;
}

std::set<std::int64_t> ray_targets(
    std::int64_t source, const std::set<std::pair<int, int>>& directions) {
  std::set<std::int64_t> result;
  for (const auto [df, dr] : directions) {
    auto file = source % 8 + df;
    auto rank = source / 8 + dr;
    while (file >= 0 && file < 8 && rank >= 0 && rank < 8) {
      result.insert(file + 8 * rank);
      file += df;
      rank += dr;
    }
  }
  return result;
}
}  // namespace

int main() {
  bool ok = true;
  const auto program = cmz::vm3::compile_minimal_rule_program();
  const std::set<std::pair<int, int>> knight_deltas{
      {-2, -1}, {-2, 1}, {-1, -2}, {-1, 2},
      {1, -2}, {1, 2}, {2, -1}, {2, 1}};
  const std::set<std::pair<int, int>> king_deltas{
      {-1, -1}, {-1, 0}, {-1, 1}, {0, -1},
      {0, 1}, {1, -1}, {1, 0}, {1, 1}};
  const std::set<std::pair<int, int>> bishop_directions{
      {-1, -1}, {-1, 1}, {1, -1}, {1, 1}};
  const std::set<std::pair<int, int>> rook_directions{
      {-1, 0}, {1, 0}, {0, -1}, {0, 1}};
  auto queen_directions = bishop_directions;
  queen_directions.insert(rook_directions.begin(), rook_directions.end());

  for (std::int64_t source = 0; source < 64; ++source) {
    const auto check = [&](cmz::vm3::PieceState piece,
                           const std::set<std::int64_t>& expected,
                           const char* message) {
      ok &= require(compiled_geometry_targets(program, source, piece) == expected,
                    message);
    };
    check(cmz::vm3::PieceState::WN,
          leaper_targets(source, knight_deltas), "exhaustive knight geometry");
    check(cmz::vm3::PieceState::WK,
          leaper_targets(source, king_deltas), "exhaustive king geometry");
    check(cmz::vm3::PieceState::WB,
          ray_targets(source, bishop_directions), "exhaustive bishop geometry");
    check(cmz::vm3::PieceState::WR,
          ray_targets(source, rook_directions), "exhaustive rook geometry");
    check(cmz::vm3::PieceState::WQ,
          ray_targets(source, queen_directions), "exhaustive queen geometry");
  }

  auto blocked = single_piece(cmz::vm3::PieceState::WR, 27,
                              cmz::vm3::Side::White);  // d4
  blocked.board[43] = cmz::vm3::PieceState::WB;        // d6 own blocker
  blocked.board[29] = cmz::vm3::PieceState::BN;        // f4 capturable blocker
  const auto blocked_state = cmz::vm3::bind_initial_state(program, blocked);
  const std::set<std::int64_t> blocked_expected{
      3, 11, 19, 24, 25, 26, 28, 29, 35};
  ok &= require(targets(cmz::vm3::compute_rule_legal(program, blocked_state.tokens),
                        27) == blocked_expected,
                "slider stops before own blocker and on enemy capture");

  auto promotion = single_piece(cmz::vm3::PieceState::WP, 48,
                                cmz::vm3::Side::White);  // a7
  const auto promotion_state = cmz::vm3::bind_initial_state(program, promotion);
  const auto promotion_legal =
      cmz::vm3::compute_rule_legal(program, promotion_state.tokens)
          .to(torch::kCPU).squeeze(0);
  const auto bank = cmz::vm3::compile_candidate_bank();
  std::set<std::int64_t> promotion_choices;
  for (const auto& candidate : bank.moves) {
    if (candidate.source == 48 && candidate.target == 56 &&
        candidate.promotion != cmz::vm3::Promotion::None &&
        promotion_legal.index({candidate.id}).item<double>() == 1.0)
      promotion_choices.insert(static_cast<std::int64_t>(candidate.promotion));
  }
  ok &= require(promotion_choices == std::set<std::int64_t>({1, 2, 3, 4}) &&
                    promotion_legal.index({48 * 64 + 56}).item<double>() == 0.0,
                "promotion requires exactly Q/R/B/N explicit candidates");

  return ok ? 0 : 1;
}
