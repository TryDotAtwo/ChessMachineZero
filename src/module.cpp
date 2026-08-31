#include "cmz/vm_module.h"

#include "cmz/hullkv.h"
#include "cmz/ste.h"

#include <c10/cuda/CUDAGuard.h>

#include <cstdint>
#include <algorithm>
#include <limits>
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

// -1 denotes the single symbolic batch dimension. Every other extent is static.
using Shape = std::vector<std::int64_t>;

Shape broadcast_shape(const Shape& left, const Shape& right) {
    Shape result(std::max(left.size(), right.size()), 1);
    for (std::size_t i = 0; i < result.size(); ++i) {
        const auto a = i < left.size() ? left[left.size() - 1 - i] : 1;
        const auto b = i < right.size() ? right[right.size() - 1 - i] : 1;
        TORCH_CHECK(a == b || a == 1 || b == 1, "artifact broadcast shape mismatch");
        result[result.size() - 1 - i] = a == 1 ? b : a;
    }
    TORCH_CHECK(result.front() == -1 &&
                std::find(result.begin() + 1, result.end(), -1) == result.end(),
                "artifact broadcast shape moves symbolic batch");
    return result;
}

Shape project_shape(const Shape& source, const Shape& weights) {
    TORCH_CHECK(source.size() >= 2 && weights.size() == 2 && source.back() == weights[0],
                "artifact projection shape mismatch");
    auto result = source;
    result.back() = weights[1];
    return result;
}

void validate_graph(const cmz::Artifact& artifact) {
    std::unordered_map<std::uint32_t, Shape> shapes = {{0U, {-1, 2048, 128}}};
    const auto frozen = [&](std::uint32_t index) -> Shape {
        TORCH_CHECK(index < artifact.tensors().size(), "artifact tensor index out of range");
        const auto& shape = artifact.tensors()[index].shape;
        return Shape(shape.begin(), shape.end());
    };
    bool has_context = false;
    for (const auto& tensor : artifact.tensors()) {
        if (tensor.name == "context_0") {
            TORCH_CHECK(!has_context, "artifact defines context_0 twice");
            TORCH_CHECK(tensor.shape == std::vector<std::uint32_t>({2045, 128}),
                        "context_0 must have shape [2045,128]");
            has_context = true;
        }
    }
    for (const auto& operation : artifact.operations()) {
        for (const auto input : operation.inputs) {
            TORCH_CHECK(shapes.count(input) != 0U, "artifact graph reads an undefined SSA value");
        }
        const auto input_shape = [&](std::size_t index) -> const Shape& {
            return shapes.at(operation.inputs.at(index));
        };
        Shape result;
        switch (operation.opcode) {
            case cmz::OpCode::MatrixTranspose:
                require_schema(operation, 1U, 1U, 0U);
                result = input_shape(0);
                TORCH_CHECK(result.size() == 3, "artifact transpose rank must be three");
                std::swap(result[1], result[2]);
                break;
            case cmz::OpCode::MatrixReshape: {
                require_schema(operation, 1U, 1U, 2U);
                const auto& source = input_shape(0);
                const std::int64_t rows = operation.attributes[0];
                const std::int64_t columns = operation.attributes[1];
                const auto limit = std::numeric_limits<std::int64_t>::max();
                TORCH_CHECK(source.size() == 3, "artifact reshape rank must be three");
                TORCH_CHECK(rows > 0 && columns > 0 && rows <= limit / columns,
                            "artifact reshape capacity overflow or zero extent");
                TORCH_CHECK(source[1] > 0 && source[2] > 0 && source[1] <= limit / source[2],
                            "artifact reshape source capacity overflow");
                TORCH_CHECK(rows * columns == source[1] * source[2],
                            "artifact reshape element count mismatch");
                result = {source[0], rows, columns};
                break;
            }
            case cmz::OpCode::MatrixMatmul: {
                require_schema(operation, 2U, 1U, 0U);
                const auto& left = input_shape(0);
                const auto& right = input_shape(1);
                TORCH_CHECK(left.size() == 3 && right.size() == 3,
                            "artifact matmul rank must be three");
                TORCH_CHECK(left[0] == right[0] && left[2] == right[1],
                            "artifact matmul shape mismatch");
                result = {left[0], left[1], right[2]};
                break;
            }
            case cmz::OpCode::RowRoute: {
                TORCH_CHECK(
                    operation.inputs.size() == 1U && operation.outputs.size() == 1U &&
                        (operation.attributes.size() == 2U || operation.attributes.size() == 3U),
                    "artifact RowRoute schema mismatch");
                const auto stride = operation.attributes.size() == 3U ? operation.attributes[2] : 1U;
                TORCH_CHECK(stride > 0U, "RowRoute stride must be positive");
                result = input_shape(0);
                TORCH_CHECK(result.size() >= 2 && operation.attributes[0] < operation.attributes[1] &&
                            operation.attributes[1] <= result[1], "artifact RowRoute bounds mismatch");
                result[1] = (std::int64_t(operation.attributes[1]) - operation.attributes[0] + stride - 1) / stride;
                break;
            }
            case cmz::OpCode::TokenProject:
            case cmz::OpCode::OutputProject:
                require_schema(operation, 1U, 1U, 1U);
                result = project_shape(input_shape(0), frozen(operation.attributes[0]));
                break;
            case cmz::OpCode::PositionAdd:
                require_schema(operation, 1U, 1U, 1U);
                result = broadcast_shape(input_shape(0), frozen(operation.attributes[0]));
                break;
            case cmz::OpCode::ResidualAdd:
                require_schema(operation, 2U, 1U, 0U);
                result = broadcast_shape(input_shape(0), input_shape(1));
                break;
            case cmz::OpCode::GatedFfn: {
                require_schema(operation, 1U, 1U, 3U);
                const auto up = project_shape(input_shape(0), frozen(operation.attributes[0]));
                const auto gate = project_shape(input_shape(0), frozen(operation.attributes[1]));
                result = project_shape(broadcast_shape(up, gate), frozen(operation.attributes[2]));
                break;
            }
            case cmz::OpCode::HardmaxSte:
                TORCH_CHECK((operation.inputs.size() == 1U || operation.inputs.size() == 2U) &&
                            operation.outputs.size() == 1U && operation.attributes.size() == 1U,
                            "artifact HardmaxSte schema mismatch");
                TORCH_CHECK(operation.attributes[0] > 0U);
                result = input_shape(0);
                TORCH_CHECK(result.size() >= 2, "artifact hardmax shape mismatch");
                if (operation.inputs.size() == 2U)
                    TORCH_CHECK(input_shape(1) == result, "artifact hardmax mask shape mismatch");
                break;
            case cmz::OpCode::HullAttention2D: {
                TORCH_CHECK((operation.inputs.size() == 2U || operation.inputs.size() == 3U) &&
                            operation.outputs.size() == 1U,
                            "artifact HullAttention2D schema mismatch");
                const bool shared = operation.inputs.size() == 2U;
                const std::size_t metadata = shared ? 3U : 2U;
                TORCH_CHECK(operation.attributes.size() > metadata, "attention candidate list is empty");
                const auto topk = operation.attributes[shared ? 1 : 0];
                TORCH_CHECK(topk > 0U && topk <= 32U && topk <= operation.attributes.size() - metadata,
                            "attention competitor capacity mismatch");
                TORCH_CHECK(operation.attributes[shared ? 2 : 1] > 0U, "attention temperature must be positive");
                const auto& q = input_shape(0);
                const auto k = shared ? frozen(operation.attributes[0]) : input_shape(1);
                const auto& v = input_shape(shared ? 1 : 2);
                TORCH_CHECK(q.size() == 3 && q[0] == -1 && q[2] == 2, "artifact attention query shape mismatch");
                TORCH_CHECK((shared && k.size() == 2 && k[1] == 2) ||
                            (!shared && k.size() == 3 && k[0] == -1 && k[2] == 2),
                            "artifact attention key shape mismatch");
                const auto rows = k[shared ? 0 : 1];
                TORCH_CHECK(v.size() == 3 && v[0] == -1 && v[1] == rows,
                            "artifact attention value shape mismatch");
                std::unordered_set<std::uint32_t> candidates;
                for (auto index = operation.attributes.begin() + metadata; index != operation.attributes.end(); ++index) {
                    TORCH_CHECK(*index < rows, "artifact attention candidate out of range");
                    TORCH_CHECK(candidates.insert(*index).second, "artifact attention duplicate candidate");
                }
                result = {-1, q[1], v[2]};
                break;
            }
            case cmz::OpCode::FrozenExpand:
                require_schema(operation, 1U, 1U, 1U);
                TORCH_CHECK(operation.inputs[0] == 0U, "FrozenExpand batch source must be VM input");
                result = frozen(operation.attributes[0]);
                result.insert(result.begin(), -1);
                break;
            case cmz::OpCode::RowConcat:
                TORCH_CHECK(!operation.inputs.empty() && operation.outputs.size() == 1U &&
                                operation.attributes.empty(),
                            "artifact RowConcat schema mismatch");
                result = input_shape(0);
                TORCH_CHECK(result.size() >= 2, "artifact RowConcat shape mismatch");
                for (std::size_t i = 1; i < operation.inputs.size(); ++i) {
                    const auto& next = input_shape(i);
                    TORCH_CHECK(next.size() == result.size(), "artifact RowConcat shape mismatch");
                    for (std::size_t dim = 0; dim < result.size(); ++dim)
                        TORCH_CHECK(dim == 1 || next[dim] == result[dim], "artifact RowConcat shape mismatch");
                    TORCH_CHECK(result[1] <= std::numeric_limits<std::int64_t>::max() - next[1], "RowConcat capacity overflow");
                    result[1] += next[1];
                }
                break;
            default:
                TORCH_CHECK(false, "unknown artifact opcode");
        }
        for (const auto output : operation.outputs) {
            TORCH_CHECK(output != 0U && shapes.emplace(output, result).second,
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
    for (std::size_t index = 0; index < artifact.tensors().size(); ++index) {
        const auto& tensor = artifact.tensors()[index];
        if (tensor.name == "context_0") {
            TORCH_CHECK(!initial_context_index_.has_value(), "artifact defines context_0 twice");
            TORCH_CHECK(tensor.shape == std::vector<std::uint32_t>({2045, 128}),
                        "context_0 must have shape [2045,128]");
            initial_context_index_ = index;
        }
        tensors_.push_back(materialize(tensor, device));
    }
    routing_indices_.resize(operations_.size());
    for (std::size_t index = 0; index < operations_.size(); ++index) {
        const auto& operation = operations_[index];
        if (operation.opcode == OpCode::HullAttention2D) {
            const auto metadata_count = operation.inputs.size() == 2U ? 3U : 2U;
            std::vector<std::int64_t> indices(
                operation.attributes.begin() + metadata_count, operation.attributes.end());
            routing_indices_[index] = torch::tensor(
                indices, torch::TensorOptions().dtype(torch::kInt64).device(device));
        }
    }
}

torch::Tensor FrozenVm::initial_context(std::int64_t batch_size) const {
    TORCH_CHECK(initial_context_index_.has_value(), "artifact does not define context_0");
    TORCH_CHECK(batch_size > 0, "initial context batch size must be positive");
    return tensors_[*initial_context_index_]
        .unsqueeze(0)
        .expand({batch_size, kContextRows, kVocabulary})
        .clone();
}

torch::Tensor FrozenVm::execute_graph(const torch::Tensor& input) const {
    TORCH_CHECK(input.is_cuda(), "VM input must remain on CUDA");
    TORCH_CHECK(input.is_floating_point(), "VM input must use a floating dtype");
    TORCH_CHECK(input.dim() == 3, "VM input must have shape [B,2048,128]");
    TORCH_CHECK(input.size(1) == kInputRows, "VM input row count must be 2048");
    TORCH_CHECK(input.size(2) == kVocabulary, "VM vocabulary width must be 128");
    std::unordered_map<std::uint32_t, torch::Tensor> values;
    values.emplace(0U, input);
    std::uint32_t final_output = 0U;
    for (std::size_t operation_index = 0; operation_index < operations_.size(); ++operation_index) {
        const auto& operation = operations_[operation_index];
        TORCH_CHECK(!operation.outputs.empty(), "artifact operation has no output");
        torch::Tensor result;
        switch (operation.opcode) {
            case OpCode::MatrixTranspose:
                result = values.at(operation.inputs[0]).transpose(1, 2);
                break;
            case OpCode::MatrixReshape: {
                const auto& source = values.at(operation.inputs[0]);
                result = source.reshape({source.size(0), operation.attributes[0], operation.attributes[1]});
                break;
            }
            case OpCode::MatrixMatmul:
                result = torch::matmul(values.at(operation.inputs[0]), values.at(operation.inputs[1]));
                break;
            case OpCode::RowRoute: {
                const auto stride = operation.attributes.size() == 3U
                    ? operation.attributes[2]
                    : 1U;
                result = values.at(operation.inputs[0]).slice(
                    1, operation.attributes[0], operation.attributes[1], stride);
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
                const auto& logits = values.at(operation.inputs[0]);
                const auto mask = operation.inputs.size() == 2U
                    ? values.at(operation.inputs[1]).to(torch::kBool)
                    : torch::ones_like(logits, torch::TensorOptions().dtype(torch::kBool));
                result = masked_hardmax_ste(
                    logits,
                    mask,
                    static_cast<double>(operation.attributes[0]) / 1000.0);
                break;
            }
            case OpCode::HullAttention2D: {
                if (operation.inputs.size() == 2U) {
                    result = hull_attention_2d_batched_ste(
                        values.at(operation.inputs[0]), tensors_.at(operation.attributes[0]),
                        values.at(operation.inputs[1]), routing_indices_[operation_index],
                        operation.attributes[1],
                        static_cast<double>(operation.attributes[2]) / 1000.0);
                } else {
                    result = hull_attention_2d_dynamic_batched_ste(
                        values.at(operation.inputs[0]), values.at(operation.inputs[1]),
                        values.at(operation.inputs[2]), routing_indices_[operation_index],
                        operation.attributes[0],
                        static_cast<double>(operation.attributes[1]) / 1000.0);
                }
                break;
            }
            case OpCode::FrozenExpand: {
                const auto batch_size = values.at(operation.inputs[0]).size(0);
                const auto& frozen = tensors_.at(operation.attributes[0]);
                std::vector<std::int64_t> shape = {batch_size};
                shape.insert(shape.end(), frozen.sizes().begin(), frozen.sizes().end());
                result = frozen.unsqueeze(0).expand(shape);
                break;
            }
            case OpCode::RowConcat: {
                std::vector<torch::Tensor> rows;
                rows.reserve(operation.inputs.size());
                for (const auto input_id : operation.inputs) {
                    rows.push_back(values.at(input_id));
                }
                result = torch::cat(rows, 1);
                break;
            }
        }
        const auto inserted = values.emplace(operation.outputs[0], std::move(result));
        TORCH_CHECK(inserted.second, "artifact graph redefines an SSA value");
        final_output = operation.outputs[0];
    }
    TORCH_CHECK(final_output != 0U, "artifact graph is empty");
    return values.at(final_output);
}

torch::Tensor FrozenVm::forward(const torch::Tensor& input) const {
    const auto output = execute_graph(input);
    TORCH_CHECK(
        output.dim() == 3 && output.size(1) == kContextRows && output.size(2) == kVocabulary,
        "artifact graph output must have shape [B,2045,128]");
    return output;
}

}  // namespace cmz
