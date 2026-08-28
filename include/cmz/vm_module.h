#pragma once

#include "cmz/artifact.h"

#include <torch/torch.h>

#include <optional>
#include <vector>

namespace cmz {

inline constexpr std::int64_t kInputRows = 2048;
inline constexpr std::int64_t kContextRows = 2045;
inline constexpr std::int64_t kMoveRows = 3;
inline constexpr std::int64_t kVocabulary = 128;

class FrozenVm final {
public:
    FrozenVm(Artifact artifact, const torch::Device& device);
    torch::Tensor initial_context(std::int64_t batch_size) const;
    torch::Tensor forward(const torch::Tensor& input) const;

private:
    std::vector<Operation> operations_;
    std::vector<torch::Tensor> tensors_;
    std::vector<torch::Tensor> routing_indices_;
    std::optional<std::size_t> initial_context_index_;
};

}  // namespace cmz
