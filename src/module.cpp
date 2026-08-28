#include "cmz/vm_module.h"

#include "cmz/ste.h"

#include <c10/cuda/CUDAGuard.h>

#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace {

constexpr float kE2m1[16] = {
    0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F,
    -0.0F, -0.5F, -1.0F, -1.5F, -2.0F, -3.0F, -4.0F, -6.0F};

torch::Tensor materialize(const cmz::TensorRecord& record, const torch::Device& device) {
    std::int64_t element_count = 1;
    std::vector<std::int64_t> shape;
    shape.reserve(record.shape.size());
    for (const auto extent : record.shape) {
        element_count *= extent;
        shape.push_back(extent);
    }
    std::vector<float> decoded(static_cast<std::size_t>(element_count));
    for (std::int64_t index = 0; index < element_count; ++index) {
        const auto packed = record.packed[static_cast<std::size_t>(index) / 2U];
        const auto code = static_cast<std::uint8_t>(
            (packed >> ((static_cast<unsigned>(index) & 1U) * 4U)) & 0x0FU);
        const auto block = static_cast<std::size_t>(index) / record.block_size;
        decoded[static_cast<std::size_t>(index)] = kE2m1[code] * record.scales[block];
    }
    return torch::from_blob(decoded.data(), shape, torch::TensorOptions().dtype(torch::kFloat32))
        .clone()
        .to(device);
}

void require_schema(
    const cmz::Operation& operation,
    std::size_t inputs,
    std::size_t outputs,
    std::size_t attributes) {
    TORCH_CHECK(
        operation.inputs.size() == inputs && operation.outputs.size() == outputs &&
            operation.attributes.size() == attributes,
        "artifact operation schema mismatch");
}

void validate_graph(const cmz::Artifact& artifact) {
    std::unordered_set<std::uint32_t> defined = {0U};
    for (const auto& operation : artifact.operations()) {
        for (const auto input : operation.inputs) {
            TORCH_CHECK(defined.count(input) != 0U, "artifact graph reads an undefined SSA value");
        }
        switch (operation.opcode) {
            case cmz::OpCode::RowRoute:
                require_schema(operation, 1U, 1U, 2U);
                TORCH_CHECK(operation.attributes[0] < operation.attributes[1]);
                break;
            case cmz::OpCode::TokenProject:
            case cmz::OpCode::PositionAdd:
            case cmz::OpCode::OutputProject:
                require_schema(operation, 1U, 1U, 1U);
                TORCH_CHECK(operation.attributes[0] < artifact.tensors().size());
                break;
            case cmz::OpCode::ResidualAdd:
                require_schema(operation, 2U, 1U, 0U);
                break;
            case cmz::OpCode::GatedFfn:
                require_schema(operation, 1U, 1U, 3U);
                for (const auto tensor : operation.attributes) {
                    TORCH_CHECK(tensor < artifact.tensors().size());
                }
                break;
            case cmz::OpCode::HardmaxSte:
                require_schema(operation, 2U, 1U, 1U);
                TORCH_CHECK(operation.attributes[0] > 0U);
                break;
            case cmz::OpCode::HullAttention2D:
                TORCH_CHECK(false, "HullAttention2D graph binding is not implemented");
                break;
        }
        for (const auto output : operation.outputs) {
            TORCH_CHECK(output != 0U && defined.insert(output).second,
                        "artifact graph redefines an SSA value");
        }
    }
    TORCH_CHECK(!artifact.operations().empty(), "artifact graph is empty");
}

}  // namespace

namespace cmz {

FrozenVm::FrozenVm(Artifact artifact, const torch::Device& device)
    : operations_(artifact.operations()) {
    TORCH_CHECK(device.is_cuda(), "VM tensors must be materialized on CUDA");
    validate_graph(artifact);
    c10::cuda::CUDAGuard guard(device);
    tensors_.reserve(artifact.tensors().size());
    for (const auto& tensor : artifact.tensors()) {
        tensors_.push_back(materialize(tensor, device));
    }
}

torch::Tensor FrozenVm::forward(const torch::Tensor& input) const {
    TORCH_CHECK(input.is_cuda(), "VM input must remain on CUDA");
    TORCH_CHECK(input.is_floating_point(), "VM input must use a floating dtype");
    TORCH_CHECK(input.dim() == 3, "VM input must have shape [B,2048,128]");
    TORCH_CHECK(input.size(1) == kInputRows, "VM input row count must be 2048");
    TORCH_CHECK(input.size(2) == kVocabulary, "VM vocabulary width must be 128");
    std::unordered_map<std::uint32_t, torch::Tensor> values;
    values.emplace(0U, input);
    std::uint32_t final_output = 0U;
    for (const auto& operation : operations_) {
        TORCH_CHECK(!operation.outputs.empty(), "artifact operation has no output");
        torch::Tensor result;
        switch (operation.opcode) {
            case OpCode::RowRoute: {
                require_schema(operation, 1U, 1U, 2U);
                result = values.at(operation.inputs[0]).slice(
                    1, operation.attributes[0], operation.attributes[1]);
                break;
            }
            case OpCode::TokenProject:
            case OpCode::OutputProject: {
                require_schema(operation, 1U, 1U, 1U);
                result = torch::matmul(
                    values.at(operation.inputs[0]), tensors_.at(operation.attributes[0]));
                break;
            }
            case OpCode::PositionAdd: {
                require_schema(operation, 1U, 1U, 1U);
                result = values.at(operation.inputs[0]) + tensors_.at(operation.attributes[0]);
                break;
            }
            case OpCode::ResidualAdd: {
                require_schema(operation, 2U, 1U, 0U);
                result = values.at(operation.inputs[0]) + values.at(operation.inputs[1]);
                break;
            }
            case OpCode::GatedFfn: {
                require_schema(operation, 1U, 1U, 3U);
                const auto& source = values.at(operation.inputs[0]);
                const auto up = torch::matmul(source, tensors_.at(operation.attributes[0]));
                const auto gate = torch::silu(
                    torch::matmul(source, tensors_.at(operation.attributes[1])));
                result = torch::matmul(up * gate, tensors_.at(operation.attributes[2]));
                break;
            }
            case OpCode::HardmaxSte: {
                require_schema(operation, 2U, 1U, 1U);
                result = masked_hardmax_ste(
                    values.at(operation.inputs[0]),
                    values.at(operation.inputs[1]).to(torch::kBool),
                    static_cast<double>(operation.attributes[0]) / 1000.0);
                break;
            }
            case OpCode::HullAttention2D:
                TORCH_CHECK(false, "HullAttention2D graph binding is not implemented");
                break;
        }
        const auto inserted = values.emplace(operation.outputs[0], std::move(result));
        TORCH_CHECK(inserted.second, "artifact graph redefines an SSA value");
        final_output = operation.outputs[0];
    }
    TORCH_CHECK(final_output != 0U, "artifact graph is empty");
    const auto& output = values.at(final_output);
    TORCH_CHECK(
        output.dim() == 3 && output.size(1) == kContextRows && output.size(2) == kVocabulary,
        "artifact graph output must have shape [B,2045,128]");
    return output;
}

}  // namespace cmz
