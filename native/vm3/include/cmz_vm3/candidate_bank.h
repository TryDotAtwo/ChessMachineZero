#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <vector>

namespace cmz::vm3 {

enum class Promotion : std::int64_t {
  None = 0,
  Queen = 1,
  Rook = 2,
  Bishop = 3,
  Knight = 4,
};

struct RawCandidate {
  std::int64_t id;
  std::int64_t source;
  std::int64_t target;
  Promotion promotion;
};

struct CandidateBank {
  std::vector<RawCandidate> moves;
  std::int64_t selector_row_count;
  std::int64_t sentinel_id;
  torch::Tensor source_square;
  torch::Tensor target_square;
  torch::Tensor promotion;
  torch::Tensor source_file;
  torch::Tensor source_rank;
  torch::Tensor target_file;
  torch::Tensor target_rank;
};

CandidateBank compile_candidate_bank();

}  // namespace cmz::vm3
