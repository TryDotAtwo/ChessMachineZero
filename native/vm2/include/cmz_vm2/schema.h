#pragma once

#include <array>
#include <cstdint>

#include <torch/torch.h>

namespace cmz::vm2 {

inline constexpr std::int64_t kRegisterCount = 4;
inline constexpr std::int64_t kPredicateCount = 4;
inline constexpr std::int64_t kValueCount = 16;
inline constexpr std::int64_t kProgramSlots = 16;
inline constexpr std::int64_t kStageCount = 15;
inline constexpr std::int64_t kWriteProjectionComponents = 3;
inline constexpr std::int64_t kModelDim = 320;
inline constexpr std::int64_t kControlToken = 0;
inline constexpr std::int64_t kExecutionToken = 1;
inline constexpr std::int64_t kProgramTokenBegin = 2;
inline constexpr std::int64_t kRegisterTokenBegin = kProgramTokenBegin + kProgramSlots;
inline constexpr std::int64_t kPredicateTokenBegin = kRegisterTokenBegin + kRegisterCount;
inline constexpr std::int64_t kPredicateConstantTokenBegin = kPredicateTokenBegin + kPredicateCount;
inline constexpr std::int64_t kBranchTargetToken = kPredicateConstantTokenBegin + 2;
inline constexpr std::int64_t kBranchFallthroughToken = kBranchTargetToken + 1;
inline constexpr std::int64_t kConstantTokenBegin = kBranchFallthroughToken + 1;
inline constexpr std::int64_t kRelationTokenBegin = kConstantTokenBegin + kValueCount;
inline constexpr std::int64_t kLoadRelationCount = 16;
inline constexpr std::int64_t kMoveRelationCount = 16;
inline constexpr std::int64_t kAddRelationCount = 136;
inline constexpr std::int64_t kHaltRelationCount = 1;
inline constexpr std::int64_t kCmpEqRelationCount = 256;
inline constexpr std::int64_t kJumpRelationCount = 1;
inline constexpr std::int64_t kRelationCount =
    kLoadRelationCount + kMoveRelationCount + kAddRelationCount + kHaltRelationCount +
    kCmpEqRelationCount + kJumpRelationCount;
inline constexpr std::int64_t kTokenCount = kRelationTokenBegin + kRelationCount;

inline constexpr std::int64_t kRoleOffset = 0;
inline constexpr std::int64_t kOpcodeOffset = 8;
inline constexpr std::int64_t kDestinationOffset = 14;
inline constexpr std::int64_t kLeftRegisterOffset = 18;
inline constexpr std::int64_t kRightRegisterOffset = 22;
inline constexpr std::int64_t kValueOffset = 26;
inline constexpr std::int64_t kLeftValueOffset = 42;
inline constexpr std::int64_t kRightValueOffset = 58;
inline constexpr std::int64_t kResultValueOffset = 74;
inline constexpr std::int64_t kProgramCounterOffset = 90;
inline constexpr std::int64_t kNextProgramCounterOffset = 106;
inline constexpr std::int64_t kTargetProgramCounterOffset = 122;
inline constexpr std::int64_t kHaltOffset = 138;
inline constexpr std::int64_t kRegisterValueOffset = 139;
inline constexpr std::int64_t kPredicateDestinationOffset = 155;
inline constexpr std::int64_t kPredicateSourceOffset = 159;
inline constexpr std::int64_t kPredicateValueOffset = 164;
inline constexpr std::int64_t kResultPredicateOffset = 166;
inline constexpr std::int64_t kCandidatePredicateOffset = 168;
inline constexpr std::int64_t kCandidateProgramCounterOffset = 170;

enum class Opcode : std::int64_t {
    Halt = 0,
    LoadConst = 1,
    Move = 2,
    Add = 3,
    CmpEq = 4,
    JumpIf = 5,
};

enum class TokenRole : std::int64_t {
    Control = 0,
    Instruction = 1,
    Register = 2,
    Constant = 3,
    Relation = 4,
    Execution = 5,
    Predicate = 6,
    PredicateConstant = 7,
    BranchTarget = 8,
    BranchFallthrough = 9,
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

struct WriteProjections {
    std::array<std::array<torch::Tensor, kWriteProjectionComponents>, kStageCount> row;
    std::array<std::array<torch::Tensor, kWriteProjectionComponents>, kStageCount> feature;
};

struct ProgramImage {
    torch::Tensor tokens;
    std::array<torch::Tensor, kStageCount> attention_masks;
    WriteProjections write_projections;
    FrozenWeights weights;
};

struct VmState {
    torch::Tensor tokens;
};

}  // namespace cmz::vm2
