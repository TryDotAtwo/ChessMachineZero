#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace cmz {

enum class OpCode : std::uint16_t {
    TokenProject = 1,
    PositionAdd = 2,
    HullAttention2D = 3,
    ResidualAdd = 4,
    GatedFfn = 5,
    OutputProject = 6,
    RowRoute = 7,
    HardmaxSte = 8,
    FrozenExpand = 9,
    RowConcat = 10,
};

struct TensorRecord {
    std::string name;
    std::vector<std::uint32_t> shape;
    std::vector<std::uint8_t> packed;
    std::vector<float> scales;
    std::uint16_t block_size;
};

struct Operation {
    OpCode opcode;
    std::vector<std::uint32_t> inputs;
    std::vector<std::uint32_t> outputs;
    std::vector<std::uint32_t> attributes;
};

class Artifact final {
public:
    static Artifact from_bytes(const std::uint8_t* data, std::size_t size);
    static Artifact from_records(
        std::vector<TensorRecord> tensors, std::vector<Operation> operations);

    template <std::size_t Size>
    static Artifact from_bytes(const std::array<std::uint8_t, Size>& data) {
        return from_bytes(data.data(), data.size());
    }

    const std::vector<TensorRecord>& tensors() const noexcept { return tensors_; }
    const std::vector<Operation>& operations() const noexcept { return operations_; }

private:
    std::vector<TensorRecord> tensors_;
    std::vector<Operation> operations_;
};

}  // namespace cmz
