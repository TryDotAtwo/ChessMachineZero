#include <cmz_vm3/candidate_bank.h>
#include <cmz_vm3/tensor_contract.h>

#include <torch/torch.h>

#include <iostream>
#include <set>
#include <tuple>

namespace {
bool require(bool condition, const char* message) {
  if (!condition) std::cerr << message << '\n';
  return condition;
}
bool row_one_hot(const torch::Tensor& tensor, std::int64_t row, std::int64_t column) {
  const auto selected = tensor.index({row});
  return selected.sum().item<double>() == 1.0 &&
         selected.index({column}).item<double>() == 1.0;
}
}

int main() {
  bool ok = true;
  const auto bank = cmz::vm3::compile_candidate_bank();
  ok &= require(bank.moves.size() == cmz::vm3::kMoveCandidateCount,
                "exact 4272 move descriptors");
  ok &= require(bank.selector_row_count == cmz::vm3::kCandidateSelectorRouteCount,
                "exact 4273 selector rows including sentinel");
  ok &= require(bank.sentinel_id == cmz::vm3::kMoveCandidateCount,
                "sentinel is the final selector row");

  std::set<std::int64_t> ids;
  for (std::size_t index = 0; index < bank.moves.size(); ++index) {
    const auto& move = bank.moves[index];
    ids.insert(move.id);
    ok &= require(move.id == static_cast<std::int64_t>(index),
                  "candidate IDs are deterministic and contiguous");
  }
  ok &= require(ids.size() == bank.moves.size(), "candidate IDs are unique");

  for (std::int64_t source = 0; source < 64; ++source) {
    for (std::int64_t target = 0; target < 64; ++target) {
      const auto id = source * 64 + target;
      const auto& move = bank.moves.at(static_cast<std::size_t>(id));
      ok &= require(move.source == source && move.target == target &&
                        move.promotion == cmz::vm3::Promotion::None,
                    "ordinary source-major target-minor order");
    }
  }

  std::set<std::pair<std::int64_t, std::int64_t>> promotion_pairs;
  std::set<std::tuple<std::int64_t, std::int64_t, std::int64_t>> promotions;
  for (std::size_t index = 4096; index < bank.moves.size(); ++index) {
    const auto& move = bank.moves[index];
    promotion_pairs.emplace(move.source, move.target);
    promotions.emplace(move.source, move.target,
                       static_cast<std::int64_t>(move.promotion));
  }
  ok &= require(promotion_pairs.size() == 44, "exact 44 promotion square pairs");
  ok &= require(promotions.size() == 176, "four choices for every promotion pair");
  for (const auto& pair : promotion_pairs) {
    for (auto promotion : {cmz::vm3::Promotion::Queen,
                           cmz::vm3::Promotion::Rook,
                           cmz::vm3::Promotion::Bishop,
                           cmz::vm3::Promotion::Knight}) {
      ok &= require(promotions.count({pair.first, pair.second,
                                      static_cast<std::int64_t>(promotion)}) == 1,
                    "Q/R/B/N promotion completeness");
    }
  }

  for (std::int64_t id = 0; id < cmz::vm3::kMoveCandidateCount; ++id) {
    const auto& move = bank.moves.at(static_cast<std::size_t>(id));
    ok &= require(row_one_hot(bank.source_square, id, move.source),
                  "raw source one-hot");
    ok &= require(row_one_hot(bank.target_square, id, move.target),
                  "raw target one-hot");
    ok &= require(row_one_hot(bank.promotion, id,
                              static_cast<std::int64_t>(move.promotion)),
                  "raw promotion one-hot");
    ok &= require(row_one_hot(bank.source_file, id, move.source % 8) &&
                      row_one_hot(bank.source_rank, id, move.source / 8) &&
                      row_one_hot(bank.target_file, id, move.target % 8) &&
                      row_one_hot(bank.target_rank, id, move.target / 8),
                  "raw coordinate one-hots");
  }
  const auto sentinel = bank.sentinel_id;
  ok &= require(bank.source_square.index({sentinel}).sum().item<double>() == 0.0 &&
                    bank.target_square.index({sentinel}).sum().item<double>() == 0.0 &&
                    bank.promotion.index({sentinel}).sum().item<double>() == 0.0 &&
                    bank.source_file.index({sentinel}).sum().item<double>() == 0.0 &&
                    bank.source_rank.index({sentinel}).sum().item<double>() == 0.0 &&
                    bank.target_file.index({sentinel}).sum().item<double>() == 0.0 &&
                    bank.target_rank.index({sentinel}).sum().item<double>() == 0.0,
                "sentinel contains no move identity");

  return ok ? 0 : 1;
}
