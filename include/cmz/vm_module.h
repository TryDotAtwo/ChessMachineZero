#pragma once

#include <torch/torch.h>

namespace cmz {

inline constexpr std::int64_t kInputRows = 2048;
inline constexpr std::int64_t kContextRows = 2045;
inline constexpr std::int64_t kMoveRows = 3;
inline constexpr std::int64_t kVocabulary = 128;

class FrozenVm final {
public:
    torch::Tensor forward(const torch::Tensor& input) const;
};

}  // namespace cmz
