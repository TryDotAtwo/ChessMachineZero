#include <cmz_vm3/candidate_bank.h>
#include <cmz_vm3/tensor_contract.h>

#include <array>

namespace cmz::vm3 {
namespace {

void encode(CandidateBank& bank, const RawCandidate& candidate) {
  const auto row = candidate.id;
  const auto source_file = candidate.source % 8;
  const auto source_rank = candidate.source / 8;
  const auto target_file = candidate.target % 8;
  const auto target_rank = candidate.target / 8;
  bank.source_square.index_put_({row, candidate.source}, 1.0);
  bank.target_square.index_put_({row, candidate.target}, 1.0);
  bank.promotion.index_put_({row, static_cast<std::int64_t>(candidate.promotion)}, 1.0);
  bank.source_file.index_put_({row, source_file}, 1.0);
  bank.source_rank.index_put_({row, source_rank}, 1.0);
  bank.target_file.index_put_({row, target_file}, 1.0);
  bank.target_rank.index_put_({row, target_rank}, 1.0);
}

}  // namespace

CandidateBank compile_candidate_bank() {
  const auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);
  CandidateBank bank{
      {},
      kCandidateSelectorRouteCount,
      kMoveCandidateCount,
      torch::zeros({kCandidateSelectorRouteCount, 64}, options),
      torch::zeros({kCandidateSelectorRouteCount, 64}, options),
      torch::zeros({kCandidateSelectorRouteCount, 5}, options),
      torch::zeros({kCandidateSelectorRouteCount, 8}, options),
      torch::zeros({kCandidateSelectorRouteCount, 8}, options),
      torch::zeros({kCandidateSelectorRouteCount, 8}, options),
      torch::zeros({kCandidateSelectorRouteCount, 8}, options),
  };
  bank.moves.reserve(kMoveCandidateCount);

  for (std::int64_t source = 0; source < 64; ++source) {
    for (std::int64_t target = 0; target < 64; ++target) {
      RawCandidate candidate{static_cast<std::int64_t>(bank.moves.size()), source,
                             target, Promotion::None};
      bank.moves.push_back(candidate);
      encode(bank, candidate);
    }
  }

  constexpr std::array<Promotion, 4> promotions{
      Promotion::Queen, Promotion::Rook, Promotion::Bishop, Promotion::Knight};
  for (std::int64_t side = 0; side < 2; ++side) {
    const auto source_rank = side == 0 ? 6 : 1;
    const auto target_rank = side == 0 ? 7 : 0;
    for (std::int64_t source_file = 0; source_file < 8; ++source_file) {
      const auto first_target_file = source_file == 0 ? 0 : source_file - 1;
      const auto last_target_file = source_file == 7 ? 7 : source_file + 1;
      for (auto target_file = first_target_file; target_file <= last_target_file;
           ++target_file) {
        const auto source = source_file + 8 * source_rank;
        const auto target = target_file + 8 * target_rank;
        for (const auto promotion : promotions) {
          RawCandidate candidate{static_cast<std::int64_t>(bank.moves.size()), source,
                                 target, promotion};
          bank.moves.push_back(candidate);
          encode(bank, candidate);
        }
      }
    }
  }

  TORCH_CHECK(static_cast<std::int64_t>(bank.moves.size()) == kMoveCandidateCount,
              "candidate compiler emitted an invalid row count");
  return bank;
}

}  // namespace cmz::vm3
