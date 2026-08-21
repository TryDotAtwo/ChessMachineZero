#include <cmz_vm3/rule_compiler.h>
#include <cmz_vm3/tensor_contract.h>

#include <array>
#include <cmath>

namespace cmz::vm3 {
namespace {
torch::Tensor zeros(std::initializer_list<std::int64_t> shape) {
  return torch::zeros(shape, torch::kFloat64);
}
}  // namespace

FrozenTensorMap compile_attack_tensors(const CandidateBank& bank) {
  FrozenTensorMap tensors;
  auto square_code = zeros({64, 2});
  const auto pi = std::acos(-1.0);
  for (std::int64_t square = 0; square < 64; ++square) {
    const auto angle = 2.0 * pi * static_cast<double>(square) / 64.0;
    square_code.index_put_({square, 0}, std::cos(angle));
    square_code.index_put_({square, 1}, std::sin(angle));
  }
  auto pawn_white = zeros({64, 64});
  auto pawn_black = zeros({64, 64});
  auto knight = zeros({64, 64});
  auto king = zeros({64, 64});
  auto rook = zeros({64, 64});
  auto bishop = zeros({64, 64});
  std::array<torch::Tensor, 6> between{
      zeros({64, 64, 64}), zeros({64, 64, 64}),
      zeros({64, 64, 64}), zeros({64, 64, 64}),
      zeros({64, 64, 64}), zeros({64, 64, 64})};
  auto between_all = zeros({64, 64, 64});
  auto between_ids = torch::zeros({64, 64, 6}, torch::kInt64);
  auto between_valid = zeros({64, 64, 6});
  for (std::int64_t attacker = 0; attacker < 64; ++attacker) {
    const auto af = attacker % 8;
    const auto ar = attacker / 8;
    for (std::int64_t target = 0; target < 64; ++target) {
      const auto tf = target % 8;
      const auto tr = target / 8;
      const auto df = tf - af;
      const auto dr = tr - ar;
      pawn_white.index_put_({attacker, target}, dr == 1 && std::abs(df) == 1);
      pawn_black.index_put_({attacker, target}, dr == -1 && std::abs(df) == 1);
      knight.index_put_({attacker, target},
          (std::abs(df) == 1 && std::abs(dr) == 2) ||
          (std::abs(df) == 2 && std::abs(dr) == 1));
      king.index_put_({attacker, target},
          std::max(std::abs(df), std::abs(dr)) == 1);
      const auto rook_line = (df == 0) != (dr == 0);
      const auto bishop_line = std::abs(df) == std::abs(dr) && df != 0;
      rook.index_put_({attacker, target}, rook_line);
      bishop.index_put_({attacker, target}, bishop_line);
      if (rook_line || bishop_line) {
        const auto step_f = (df > 0) - (df < 0);
        const auto step_r = (dr > 0) - (dr < 0);
        const auto distance = std::max(std::abs(df), std::abs(dr));
        for (std::int64_t slot = 0; slot < distance - 1; ++slot) {
          const auto square = (af + (slot + 1) * step_f) +
                              8 * (ar + (slot + 1) * step_r);
          between[slot].index_put_({attacker, target, square}, 1.0);
          between_all.index_put_({attacker, target, square}, 1.0);
          between_ids.index_put_({target, attacker, slot}, square);
          between_valid.index_put_({target, attacker, slot}, 1.0);
        }
      }
    }
  }
  auto castle_active = zeros({kMoveCandidateCount, 1});
  auto castle_transit = zeros({kMoveCandidateCount, 64});
  std::vector<std::int64_t> castle_ids;
  for (const auto& move : bank.moves) {
    if (move.promotion != Promotion::None) continue;
    if ((move.source == 4 && (move.target == 2 || move.target == 6)) ||
        (move.source == 60 && (move.target == 58 || move.target == 62))) {
      castle_active.index_put_({move.id, 0}, 1.0);
      castle_transit.index_put_({move.id, (move.source + move.target) / 2}, 1.0);
      castle_ids.push_back(move.id);
    }
  }
  tensors.emplace("attack_pawn_white", pawn_white);
  tensors.emplace("attack_pawn_black", pawn_black);
  tensors.emplace("attack_knight", knight);
  tensors.emplace("attack_king", king);
  tensors.emplace("attack_rook", rook);
  tensors.emplace("attack_bishop", bishop);
  for (std::int64_t slot = 0; slot < 6; ++slot)
    tensors.emplace("attack_between_" + std::to_string(slot), between[slot]);
  tensors.emplace("attack_between_all", between_all);
  tensors.emplace("attack_between_ids", between_ids);
  tensors.emplace("attack_between_valid", between_valid);
  const auto batch_ids =
      torch::arange(kSparseHardForwardMaxBatch, torch::kInt64)
          .view({kSparseHardForwardMaxBatch, 1, 1, 1});
  const auto batched_offsets = [&](std::int64_t count) {
    const auto item_ids =
        torch::arange(count, torch::kInt64).view({1, count, 1, 1});
    return (batch_ids * count + item_ids) * 64;
  };
  tensors.emplace("attack_offsets_candidates",
                  batched_offsets(kMoveCandidateCount));
  tensors.emplace("attack_offsets_castles", batched_offsets(4));
  tensors.emplace("attack_offsets_single", batched_offsets(1));
  tensors.emplace("candidate_castle_active", castle_active);
  tensors.emplace("candidate_castle_transit", castle_transit);
  const auto castle_id_tensor = torch::tensor(castle_ids, torch::kInt64);
  auto castle_route = zeros({static_cast<std::int64_t>(castle_ids.size()),
                             kMoveCandidateCount});
  for (std::int64_t row = 0;
       row < static_cast<std::int64_t>(castle_ids.size()); ++row)
    castle_route.index_put_({row, castle_ids[row]}, 1.0);
  tensors.emplace("castle_candidate_ids", castle_id_tensor);
  tensors.emplace("castle_candidate_route", castle_route);
  tensors.emplace("attack_square_code", square_code);
  tensors.emplace("attack_square_eligibility",
                  torch::ones({1, 64}, torch::kFloat64));
  tensors.emplace("boolean_class_eligibility",
                  torch::ones({1, 2}, torch::kFloat64));
  tensors.emplace("boolean_class_value",
                  torch::tensor({0.0, 1.0}, torch::kFloat64));
  tensors.emplace("boolean_keys_half",
                  torch::tensor({{0.0, 0.0}, {1.0, -0.5}}, torch::kFloat64));
  tensors.emplace("boolean_keys_one_half",
                  torch::tensor({{0.0, 0.0}, {1.0, -1.5}}, torch::kFloat64));
  tensors.emplace("boolean_keys_two_half",
                  torch::tensor({{0.0, 0.0}, {1.0, -2.5}}, torch::kFloat64));
  return tensors;
}
}  // namespace cmz::vm3
