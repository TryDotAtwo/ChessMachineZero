#pragma once

#include <array>
#include <cstdint>

#include <torch/torch.h>

namespace cmz::vm2 {

inline constexpr std::int64_t kRegisterCount = 4;
inline constexpr std::int64_t kValueCount = 16;
inline constexpr std::int64_t kProgramSlots = 4;
inline constexpr std::int64_t kStageCount = 6;
inline constexpr std::int64_t kModelDim = 128;
inline constexpr std::int64_t kControlToken = 0;
inline constexpr std::int64_t kExecutionToken = 1;
inline constexpr std::int64_t kProgramTokenBegin = 2;
inline constexpr std::int64_t kRegisterTokenBegin = kProgramTokenBegin + kProgramSlots;
inline constexpr std::int64_t kConstantTokenBegin = kRegisterTokenBegin + kRegisterCount;
inline constexpr std::int64_t kRelationTokenBegin = kConstantTokenBegin + kValueCount;
inline constexpr std::int64_t kLoadRelationCount = 16;
inline constexpr std::int64_t kMoveRelationCount = 16;
inline constexpr std::int64_t kAddRelationCount = 136;
inline constexpr std::int64_t kHaltRelationCount = 1;
inline constexpr std::int64_t kRelationCount =
    kLoadRelationCount + kMoveRelationCount + kAddRelationCount + kHaltRelationCount;
inline constexpr std::int64_t kTokenCount = kRelationTokenBegin + kRelationCount;

inline constexpr std::int64_t kRoleOffset = 0;
inline constexpr std::int64_t kOpcodeOffset = 8;
inline constexpr std::int64_t kDestinationOffset = 12;
inline constexpr std::int64_t kLeftRegisterOffset = 16;
inline constexpr std::int64_t kRightRegisterOffset = 20;
inline constexpr std::int64_t kValueOffset = 24;
inline constexpr std::int64_t kLeftValueOffset = 40;
inline constexpr std::int64_t kRightValueOffset = 56;
inline constexpr std::int64_t kResultValueOffset = 72;
inline constexpr std::int64_t kProgramCounterOffset = 88;
inline constexpr std::int64_t kNextProgramCounterOffset = 92;
inline constexpr std::int64_t kHaltOffset = 96;
inline constexpr std::int64_t kRegisterValueOffset = 97;

enum class Opcode : std::int64_t {
    Halt = 0,
    LoadConst = 1,
    Move = 2,
    Add = 3,
};

enum class TokenRole : std::int64_t {
    Control = 0,
    Instruction = 1,
    Register = 2,
    Constant = 3,
    Relation = 4,
    Execution = 5,
};

struct Instruction {
    Opcode opcode;
    std::int64_t dst;
    std::int64_t lhs;
    std::int64_t rhs;
};

struct FrozenWeights {
    std::array<torch::Tensor, kStageCount> wq;
    std::array<torch::Tensor, kStageCount> wk;
    std::array<torch::Tensor, kStageCount> wv;
    std::array<torch::Tensor, kStageCount> wo;
};

struct ProgramImage {
    torch::Tensor tokens;
    std::array<torch::Tensor, kStageCount> attention_masks;
    std::array<torch::Tensor, kStageCount> write_masks;
    FrozenWeights weights;
};

struct VmState {
    torch::Tensor tokens;
};

}  // namespace cmz::vm2
