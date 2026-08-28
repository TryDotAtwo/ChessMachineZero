#include "cmz/vm_module.h"

namespace cmz {

torch::Tensor FrozenVm::forward(const torch::Tensor& input) const {
    TORCH_CHECK(input.is_cuda(), "VM input must remain on CUDA");
    TORCH_CHECK(input.is_floating_point(), "VM input must use a floating dtype");
    TORCH_CHECK(input.dim() == 3, "VM input must have shape [B,2048,128]");
    TORCH_CHECK(input.size(1) == kInputRows, "VM input row count must be 2048");
    TORCH_CHECK(input.size(2) == kVocabulary, "VM vocabulary width must be 128");
    return input.slice(1, kMoveRows, kInputRows);
}

}  // namespace cmz
