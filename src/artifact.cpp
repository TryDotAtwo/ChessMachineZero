#include "cmz/artifact.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace cmz {
namespace {

constexpr std::size_t kHeaderSize = 56;
constexpr std::array<std::uint8_t, 8> kMagic = {'C', 'M', 'Z', 'V', 'M', '0', '0', '1'};

std::uint32_t read_u32(const std::uint8_t* data) {
    return static_cast<std::uint32_t>(data[0]) |
           (static_cast<std::uint32_t>(data[1]) << 8U) |
           (static_cast<std::uint32_t>(data[2]) << 16U) |
           (static_cast<std::uint32_t>(data[3]) << 24U);
}

std::uint16_t read_u16(const std::uint8_t* data) {
    return static_cast<std::uint16_t>(data[0]) |
           static_cast<std::uint16_t>(data[1] << 8U);
}

std::uint32_t rotate_right(std::uint32_t value, std::uint32_t amount) {
    return (value >> amount) | (value << (32U - amount));
}

std::array<std::uint8_t, 32> sha256(const std::uint8_t* data, std::size_t size) {
    constexpr std::array<std::uint32_t, 64> constants = {
        0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U,
        0x923f82a4U, 0xab1c5ed5U, 0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
        0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U, 0xe49b69c1U, 0xefbe4786U,
        0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
        0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U,
        0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
        0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U, 0xa2bfe8a1U, 0xa81a664bU,
        0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
        0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU,
        0x5b9cca4fU, 0x682e6ff3U, 0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
        0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};
    std::vector<std::uint8_t> padded(data, data + size);
    padded.push_back(0x80U);
    while (padded.size() % 64U != 56U) {
        padded.push_back(0U);
    }
    const auto bit_size = static_cast<std::uint64_t>(size) * 8U;
    for (int shift = 56; shift >= 0; shift -= 8) {
        padded.push_back(static_cast<std::uint8_t>(bit_size >> shift));
    }
    std::array<std::uint32_t, 8> state = {
        0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
        0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
    for (std::size_t chunk = 0; chunk < padded.size(); chunk += 64U) {
        std::array<std::uint32_t, 64> words{};
        for (std::size_t index = 0; index < 16U; ++index) {
            const auto offset = chunk + index * 4U;
            words[index] = (static_cast<std::uint32_t>(padded[offset]) << 24U) |
                           (static_cast<std::uint32_t>(padded[offset + 1U]) << 16U) |
                           (static_cast<std::uint32_t>(padded[offset + 2U]) << 8U) |
                           static_cast<std::uint32_t>(padded[offset + 3U]);
        }
        for (std::size_t index = 16U; index < words.size(); ++index) {
            const auto s0 = rotate_right(words[index - 15U], 7U) ^
                            rotate_right(words[index - 15U], 18U) ^ (words[index - 15U] >> 3U);
            const auto s1 = rotate_right(words[index - 2U], 17U) ^
                            rotate_right(words[index - 2U], 19U) ^ (words[index - 2U] >> 10U);
            words[index] = words[index - 16U] + s0 + words[index - 7U] + s1;
        }
        auto a = state[0]; auto b = state[1]; auto c = state[2]; auto d = state[3];
        auto e = state[4]; auto f = state[5]; auto g = state[6]; auto h = state[7];
        for (std::size_t index = 0; index < words.size(); ++index) {
            const auto sum1 = rotate_right(e, 6U) ^ rotate_right(e, 11U) ^ rotate_right(e, 25U);
            const auto choose = (e & f) ^ ((~e) & g);
            const auto temp1 = h + sum1 + choose + constants[index] + words[index];
            const auto sum0 = rotate_right(a, 2U) ^ rotate_right(a, 13U) ^ rotate_right(a, 22U);
            const auto majority = (a & b) ^ (a & c) ^ (b & c);
            const auto temp2 = sum0 + majority;
            h = g; g = f; f = e; e = d + temp1; d = c; c = b; b = a; a = temp1 + temp2;
        }
        state[0] += a; state[1] += b; state[2] += c; state[3] += d;
        state[4] += e; state[5] += f; state[6] += g; state[7] += h;
    }
    std::array<std::uint8_t, 32> result{};
    for (std::size_t index = 0; index < state.size(); ++index) {
        result[index * 4U] = static_cast<std::uint8_t>(state[index] >> 24U);
        result[index * 4U + 1U] = static_cast<std::uint8_t>(state[index] >> 16U);
        result[index * 4U + 2U] = static_cast<std::uint8_t>(state[index] >> 8U);
        result[index * 4U + 3U] = static_cast<std::uint8_t>(state[index]);
    }
    return result;
}

class Reader final {
public:
    Reader(const std::uint8_t* data, std::size_t size) : cursor_(data), remaining_(size) {}

    const std::uint8_t* take(std::size_t count) {
        if (count > remaining_) {
            throw std::runtime_error("artifact record exceeds body");
        }
        const auto* result = cursor_;
        cursor_ += count;
        remaining_ -= count;
        return result;
    }

    std::uint8_t u8() { return *take(1); }
    std::uint16_t u16() { return read_u16(take(2)); }
    std::uint32_t u32() { return read_u32(take(4)); }
    std::size_t remaining() const noexcept { return remaining_; }

private:
    const std::uint8_t* cursor_;
    std::size_t remaining_;
};

OpCode checked_opcode(std::uint16_t value) {
    if (value < static_cast<std::uint16_t>(OpCode::TokenProject) ||
        value > static_cast<std::uint16_t>(OpCode::RowConcat)) {
        throw std::runtime_error("unknown artifact opcode " + std::to_string(value));
    }
    return static_cast<OpCode>(value);
}

void validate_tensor(const TensorRecord& tensor) {
    if (tensor.name.empty() || tensor.shape.empty() || tensor.shape.size() > 8U ||
        tensor.block_size == 0U || tensor.scales.empty()) {
        throw std::runtime_error("invalid tensor metadata");
    }
    std::size_t elements = 1U;
    for (const auto extent : tensor.shape) {
        if (extent == 0U || elements > std::numeric_limits<std::size_t>::max() / extent) {
            throw std::runtime_error("invalid tensor shape");
        }
        elements *= extent;
    }
    if (std::any_of(tensor.scales.begin(), tensor.scales.end(), [](float scale) {
            return !std::isfinite(scale);
        })) {
        throw std::runtime_error("tensor scale is non-finite");
    }
    if (tensor.packed.size() != (elements + 1U) / 2U ||
        tensor.scales.size() != (elements + tensor.block_size - 1U) / tensor.block_size) {
        throw std::runtime_error("tensor FP4 payload layout mismatch");
    }
}

}  // namespace

Artifact Artifact::from_bytes(const std::uint8_t* data, std::size_t size) {
    if (data == nullptr || size < kHeaderSize) {
        throw std::runtime_error("artifact is shorter than fixed header");
    }
    if (!std::equal(kMagic.begin(), kMagic.end(), data) || read_u32(data + 8U) != 1U) {
        throw std::runtime_error("artifact magic or version mismatch");
    }
    const auto tensor_count = read_u32(data + 12U);
    const auto operation_count = read_u32(data + 16U);
    const auto body_size = read_u32(data + 20U);
    if (body_size != size - kHeaderSize) {
        throw std::runtime_error("artifact body length mismatch");
    }
    const auto digest = sha256(data + kHeaderSize, body_size);
    if (!std::equal(digest.begin(), digest.end(), data + 24U)) {
        throw std::runtime_error("artifact digest mismatch");
    }

    Artifact artifact;
    Reader reader(data + kHeaderSize, body_size);
    artifact.tensors_.reserve(tensor_count);
    for (std::uint32_t index = 0; index < tensor_count; ++index) {
        const auto name_size = reader.u16();
        const auto rank = reader.u8();
        const auto reserved = reader.u8();
        const auto block_size = reader.u16();
        const auto scale_count = reader.u16();
        const auto packed_size = reader.u32();
        if (name_size == 0U || rank == 0U || rank > 8U || reserved != 0U ||
            block_size == 0U || scale_count == 0U) {
            throw std::runtime_error("invalid tensor metadata");
        }
        TensorRecord tensor;
        tensor.block_size = block_size;
        std::size_t elements = 1U;
        for (std::uint8_t dimension = 0; dimension < rank; ++dimension) {
            const auto extent = reader.u32();
            if (extent == 0U || elements > std::numeric_limits<std::size_t>::max() / extent) {
                throw std::runtime_error("invalid tensor shape");
            }
            elements *= extent;
            tensor.shape.push_back(extent);
        }
        const auto* name = reader.take(name_size);
        tensor.name.assign(reinterpret_cast<const char*>(name), name_size);
        for (std::uint16_t scale = 0; scale < scale_count; ++scale) {
            const auto bits = reader.u32();
            float value = 0.0F;
            std::memcpy(&value, &bits, sizeof(value));
            if (!std::isfinite(value)) {
                throw std::runtime_error("tensor scale is non-finite");
            }
            tensor.scales.push_back(value);
        }
        const auto* packed = reader.take(packed_size);
        tensor.packed.assign(packed, packed + packed_size);
        const auto expected_packed = (elements + 1U) / 2U;
        const auto expected_scales = (elements + block_size - 1U) / block_size;
        if (packed_size != expected_packed || scale_count != expected_scales) {
            throw std::runtime_error("tensor FP4 payload layout mismatch");
        }
        artifact.tensors_.push_back(std::move(tensor));
    }
    artifact.operations_.reserve(operation_count);
    for (std::uint32_t index = 0; index < operation_count; ++index) {
        Operation operation;
        operation.opcode = checked_opcode(reader.u16());
        const auto input_count = reader.u8();
        const auto output_count = reader.u8();
        const auto attribute_count = reader.u16();
        if (reader.u16() != 0U) {
            throw std::runtime_error("operation reserved field is nonzero");
        }
        for (std::uint8_t input = 0; input < input_count; ++input) operation.inputs.push_back(reader.u32());
        for (std::uint8_t output = 0; output < output_count; ++output) operation.outputs.push_back(reader.u32());
        for (std::uint16_t attribute = 0; attribute < attribute_count; ++attribute) operation.attributes.push_back(reader.u32());
        artifact.operations_.push_back(std::move(operation));
    }
    if (reader.remaining() != 0U) {
        throw std::runtime_error("artifact contains trailing bytes");
    }
    return artifact;
}

Artifact Artifact::from_records(
    std::vector<TensorRecord> tensors, std::vector<Operation> operations) {
    for (const auto& tensor : tensors) {
        validate_tensor(tensor);
    }
    Artifact artifact;
    artifact.tensors_ = std::move(tensors);
    artifact.operations_ = std::move(operations);
    return artifact;
}

}  // namespace cmz
