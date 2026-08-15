#include <cmz_vm3/rule_compiler.h>
#include <cmz_vm3/tensor_contract.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <string>

namespace cmz::vm3 {
namespace {
torch::Tensor zeros(std::initializer_list<std::int64_t> shape) {
  return torch::zeros(shape, torch::kFloat64);
}
}  // namespace

FrozenTensorMap compile_special_move_tensors(const CandidateBank& bank) {
  FrozenTensorMap tensors;
  auto ep_captured = zeros({kMoveCandidateCount, 64});
  auto ep_target = zeros({kMoveCandidateCount, 65});
  auto castle_geometry = zeros({kMoveCandidateCount, 1});
  auto castle_right = zeros({kMoveCandidateCount, 4});
  auto castle_rook_source = zeros({kMoveCandidateCount, 64});
  auto castle_rook_target = zeros({kMoveCandidateCount, 64});
  std::array<torch::Tensor, 3> castle_empty{
      zeros({kMoveCandidateCount, 64}), zeros({kMoveCandidateCount, 64}),
      zeros({kMoveCandidateCount, 64})};
  auto castle_empty_absent = torch::ones({kMoveCandidateCount, 3}, torch::kFloat64);
  auto promotion_piece = zeros({kMoveCandidateCount, 2, 13});
  auto promotion_active = zeros({kMoveCandidateCount, 1});
  auto next_ep = zeros({kMoveCandidateCount, 65});
  next_ep.index_put_({torch::indexing::Slice(), 0}, 1.0);
  auto source_right_clear = zeros({kMoveCandidateCount, 13, 4});
  auto target_right_clear = zeros({kMoveCandidateCount, 13, 4});

  auto castling_decode = zeros({16, 4});
  for (std::int64_t code = 0; code < 16; ++code)
    for (std::int64_t bit = 0; bit < 4; ++bit)
      castling_decode.index_put_({code, bit}, (code >> bit) & 1);
  auto halfmove_increment = zeros({151, 151});
  for (std::int64_t value = 0; value < 151; ++value)
    halfmove_increment.index_put_({value, std::min<std::int64_t>(value + 1, 150)}, 1.0);
  auto radix_increment = zeros({128, 128});
  for (std::int64_t value = 0; value < 128; ++value)
    radix_increment.index_put_({value, (value + 1) % 128}, 1.0);

  for (const auto& move : bank.moves) {
    const auto sf = move.source % 8;
    const auto sr = move.source / 8;
    const auto tf = move.target % 8;
    const auto tr = move.target / 8;
    const auto df = tf - sf;
    const auto dr = tr - sr;
    if (move.promotion == Promotion::None && std::abs(df) == 1 &&
        std::abs(dr) == 1) {
      ep_captured.index_put_({move.id, tf + 8 * sr}, 1.0);
      ep_target.index_put_({move.id, move.target + 1}, 1.0);
    }
    const auto set_castle = [&](std::int64_t right, std::int64_t rook_source,
                                std::int64_t rook_target,
                                std::initializer_list<std::int64_t> empty) {
      castle_geometry.index_put_({move.id, 0}, 1.0);
      castle_right.index_put_({move.id, right}, 1.0);
      castle_rook_source.index_put_({move.id, rook_source}, 1.0);
      castle_rook_target.index_put_({move.id, rook_target}, 1.0);
      std::int64_t slot = 0;
      for (const auto square : empty) {
        castle_empty[slot].index_put_({move.id, square}, 1.0);
        castle_empty_absent.index_put_({move.id, slot}, 0.0);
        ++slot;
      }
    };
    if (move.promotion == Promotion::None && move.source == 4 && move.target == 6)
      set_castle(0, 7, 5, {5, 6});
    if (move.promotion == Promotion::None && move.source == 4 && move.target == 2)
      set_castle(1, 0, 3, {1, 2, 3});
    if (move.promotion == Promotion::None && move.source == 60 && move.target == 62)
      set_castle(2, 63, 61, {61, 62});
    if (move.promotion == Promotion::None && move.source == 60 && move.target == 58)
      set_castle(3, 56, 59, {57, 58, 59});

    if (move.promotion != Promotion::None) {
      promotion_active.index_put_({move.id, 0}, 1.0);
      const auto offset = static_cast<std::int64_t>(move.promotion) == 1 ? 5
                          : static_cast<std::int64_t>(move.promotion) == 2 ? 4
                          : static_cast<std::int64_t>(move.promotion) == 3 ? 3 : 2;
      promotion_piece.index_put_({move.id, 0, offset}, 1.0);
      promotion_piece.index_put_({move.id, 1, offset + 6}, 1.0);
    }
    if (move.promotion == Promotion::None && sf == tf && sr == 1 && dr == 2) {
      next_ep.index_put_({move.id, 0}, 0.0);
      next_ep.index_put_({move.id, move.source + 8 + 1}, 1.0);
    }
    if (move.promotion == Promotion::None && sf == tf && sr == 6 && dr == -2) {
      next_ep.index_put_({move.id, 0}, 0.0);
      next_ep.index_put_({move.id, move.source - 8 + 1}, 1.0);
    }

    for (std::int64_t bit : {0, 1})
      source_right_clear.index_put_({move.id, 6, bit}, move.source == 4 ? 1.0 : 0.0);
    for (std::int64_t bit : {2, 3})
      source_right_clear.index_put_({move.id, 12, bit}, move.source == 60 ? 1.0 : 0.0);
    const std::array<std::pair<std::int64_t, std::int64_t>, 4> rook_rights{{
        {7, 0}, {0, 1}, {63, 2}, {56, 3}}};
    for (const auto [square, bit] : rook_rights) {
      const auto piece = bit < 2 ? 4 : 10;
      source_right_clear.index_put_({move.id, piece, bit},
                                    move.source == square ? 1.0 : 0.0);
      target_right_clear.index_put_({move.id, piece, bit},
                                    move.target == square ? 1.0 : 0.0);
    }
  }
  tensors.emplace("castling_decode", castling_decode);
  tensors.emplace("halfmove_increment", halfmove_increment);
  auto halfmove_reset = zeros({151});
  halfmove_reset.index_put_({0}, 1.0);
  tensors.emplace("halfmove_reset", halfmove_reset);
  tensors.emplace("radix_increment", radix_increment);
  tensors.emplace("candidate_ep_captured_square", ep_captured);
  tensors.emplace("candidate_ep_target", ep_target);
  tensors.emplace("candidate_castle_geometry", castle_geometry);
  tensors.emplace("candidate_castle_right", castle_right);
  tensors.emplace("candidate_castle_rook_source", castle_rook_source);
  tensors.emplace("candidate_castle_rook_target", castle_rook_target);
  tensors.emplace("candidate_castle_empty_absent", castle_empty_absent);
  for (std::int64_t slot = 0; slot < 3; ++slot)
    tensors.emplace("candidate_castle_empty_" + std::to_string(slot),
                    castle_empty[slot]);
  tensors.emplace("candidate_promotion_piece", promotion_piece);
  tensors.emplace("candidate_promotion_active", promotion_active);
  tensors.emplace("candidate_next_ep", next_ep);
  tensors.emplace("candidate_source_right_clear", source_right_clear);
  tensors.emplace("candidate_target_right_clear", target_right_clear);
  return tensors;
}

}  // namespace cmz::vm3
