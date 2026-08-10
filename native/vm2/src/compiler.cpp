#include "cmz_vm2/compiler.h"

#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace cmz::vm2 {
namespace {

using TensorArray = std::array<torch::Tensor, kStageCount>;

void set_one_hot(torch::Tensor& tokens, std::int64_t row, std::int64_t offset, std::int64_t value) {
    tokens.index_put_({row, offset + value}, 1.0);
}

void set_weight(torch::Tensor& weight, std::int64_t input, std::int64_t output, double value = 1.0) {
    weight.index_put_({input, output}, value);
}

void allow(torch::Tensor& mask, std::int64_t query, std::int64_t key) {
    mask.index_put_({query, key}, 0.0);
}

void require_register(std::int64_t value, const char* field) {
    if (value < 0 || value >= kRegisterCount) {
        throw std::invalid_argument(std::string(field) + " register is outside 0..3");
    }
}

void require_value(std::int64_t value) {
    if (value < 0 || value >= kValueCount) {
        throw std::invalid_argument("constant is outside 0..15");
    }
}

void require_predicate(std::int64_t value, bool allow_true_literal = false) {
    const auto limit = kPredicateCount + (allow_true_literal ? 1 : 0);
    if (value < 0 || value >= limit) {
        throw std::invalid_argument("predicate is outside the compiled domain");
    }
}

void require_target(std::int64_t value) {
    if (value < 0 || value >= kProgramSlots) {
        throw std::invalid_argument("branch target is outside the program");
    }
}

void validate_instruction(const Instruction& instruction, std::int64_t slot) {
    switch (instruction.opcode) {
        case Opcode::LoadConst:
            require_register(instruction.dst, "destination");
            require_value(instruction.lhs);
            break;
        case Opcode::Move:
            require_register(instruction.dst, "destination");
            require_register(instruction.lhs, "source");
            break;
        case Opcode::Add:
            require_register(instruction.dst, "destination");
            require_register(instruction.lhs, "left");
            require_register(instruction.rhs, "right");
            break;
        case Opcode::CmpEq:
            require_predicate(instruction.dst);
            require_register(instruction.lhs, "left");
            require_register(instruction.rhs, "right");
            break;
        case Opcode::JumpIf:
            require_predicate(instruction.lhs, true);
            require_target(instruction.rhs);
            break;
        case Opcode::Halt:
            break;
        default:
            throw std::invalid_argument("unknown opcode");
    }
}

FrozenWeights make_weights(const torch::TensorOptions& options) {
    FrozenWeights weights;
    for (std::int64_t stage = 0; stage < kStageCount; ++stage) {
        weights.wq[stage] = torch::zeros({kModelDim, kModelDim}, options);
        weights.wk[stage] = torch::zeros({kModelDim, kModelDim}, options);
        weights.wv[stage] = torch::zeros({kModelDim, kModelDim}, options);
        weights.wo[stage] = torch::eye(kModelDim, options);
    }

    // Stage 0: the execution token selects the instruction matching its PC.
    for (std::int64_t pc = 0; pc < kProgramSlots; ++pc) {
        set_weight(weights.wq[0], kProgramCounterOffset + pc, pc);
        set_weight(weights.wk[0], kProgramCounterOffset + pc, pc);
    }
    weights.wv[0] = torch::eye(kModelDim, options);

    // Stage 1: route LOAD constants or MOVE/ADD left registers.
    set_weight(weights.wq[1], kOpcodeOffset + static_cast<std::int64_t>(Opcode::LoadConst), 0, 2.0);
    set_weight(weights.wq[1], kOpcodeOffset + static_cast<std::int64_t>(Opcode::Move), 1, 2.0);
    set_weight(weights.wq[1], kOpcodeOffset + static_cast<std::int64_t>(Opcode::Add), 1, 2.0);
    set_weight(weights.wq[1], kOpcodeOffset + static_cast<std::int64_t>(Opcode::CmpEq), 1, 2.0);
    set_weight(weights.wk[1], kRoleOffset + static_cast<std::int64_t>(TokenRole::Constant), 0, 2.0);
    set_weight(weights.wk[1], kRoleOffset + static_cast<std::int64_t>(TokenRole::Register), 1, 2.0);
    for (std::int64_t value = 0; value < kValueCount; ++value) {
        set_weight(weights.wq[1], kValueOffset + value, 2 + value);
        set_weight(weights.wk[1], kValueOffset + value, 2 + value);
        set_weight(weights.wv[1], kValueOffset + value, kLeftValueOffset + value);
        set_weight(weights.wv[1], kRegisterValueOffset + value, kLeftValueOffset + value);
    }
    for (std::int64_t reg = 0; reg < kRegisterCount; ++reg) {
        set_weight(weights.wq[1], kLeftRegisterOffset + reg, 2 + reg);
        set_weight(weights.wk[1], kDestinationOffset + reg, 2 + reg);
    }

    // Stage 2: route the ADD right register.
    set_weight(weights.wq[2], kOpcodeOffset + static_cast<std::int64_t>(Opcode::Add), 0, 2.0);
    set_weight(weights.wq[2], kOpcodeOffset + static_cast<std::int64_t>(Opcode::CmpEq), 0, 2.0);
    set_weight(weights.wk[2], kRoleOffset + static_cast<std::int64_t>(TokenRole::Register), 0, 2.0);
    for (std::int64_t reg = 0; reg < kRegisterCount; ++reg) {
        set_weight(weights.wq[2], kRightRegisterOffset + reg, 1 + reg);
        set_weight(weights.wk[2], kDestinationOffset + reg, 1 + reg);
    }
    for (std::int64_t value = 0; value < kValueCount; ++value) {
        set_weight(weights.wv[2], kRegisterValueOffset + value, kRightValueOffset + value);
    }

    // Stage 3: select the universal ALU relation by opcode and operand values.
    for (std::int64_t opcode = 0; opcode < 6; ++opcode) {
        set_weight(weights.wq[3], kOpcodeOffset + opcode, opcode, 4.0);
        set_weight(weights.wk[3], kOpcodeOffset + opcode, opcode, 4.0);
    }
    for (std::int64_t value = 0; value < kValueCount; ++value) {
        set_weight(weights.wq[3], kLeftValueOffset + value, 6 + value);
        set_weight(weights.wk[3], kLeftValueOffset + value, 6 + value);
        set_weight(weights.wq[3], kRightValueOffset + value, 22 + value);
        set_weight(weights.wk[3], kRightValueOffset + value, 22 + value);
        set_weight(weights.wv[3], kResultValueOffset + value, kResultValueOffset + value);
    }
    for (std::int64_t predicate = 0; predicate < 2; ++predicate) {
        set_weight(weights.wv[3], kResultPredicateOffset + predicate, kResultPredicateOffset + predicate);
    }

    // Stage 4: each register selects itself, except the destination selects execution.
    set_weight(weights.wq[4], kRoleOffset + static_cast<std::int64_t>(TokenRole::Register), 0);
    set_weight(weights.wk[4], kRoleOffset + static_cast<std::int64_t>(TokenRole::Execution), 0);
    for (std::int64_t reg = 0; reg < kRegisterCount; ++reg) {
        set_weight(weights.wq[4], kDestinationOffset + reg, 1 + reg, 2.0);
        set_weight(weights.wk[4], kDestinationOffset + reg, 1 + reg, 2.0);
    }
    for (std::int64_t value = 0; value < kValueCount; ++value) {
        set_weight(weights.wv[4], kRegisterValueOffset + value, kRegisterValueOffset + value);
        set_weight(weights.wv[4], kResultValueOffset + value, kRegisterValueOffset + value);
    }

    // Stage 5: predicate slots preserve themselves or accept CMP_EQ output.
    set_weight(weights.wq[5], kRoleOffset + static_cast<std::int64_t>(TokenRole::Predicate), 0);
    set_weight(weights.wk[5], kRoleOffset + static_cast<std::int64_t>(TokenRole::Predicate), 0);
    set_weight(weights.wq[5], kRoleOffset + static_cast<std::int64_t>(TokenRole::Predicate), 10, 2.0);
    set_weight(weights.wk[5], kOpcodeOffset + static_cast<std::int64_t>(Opcode::CmpEq), 10, 2.0);
    for (std::int64_t predicate = 0; predicate < kPredicateCount; ++predicate) {
        set_weight(weights.wq[5], kPredicateDestinationOffset + predicate, 1 + predicate, 2.0);
        set_weight(weights.wk[5], kPredicateDestinationOffset + predicate, 1 + predicate, 2.0);
    }
    for (std::int64_t value = 0; value < 2; ++value) {
        set_weight(weights.wv[5], kPredicateValueOffset + value, kPredicateValueOffset + value);
        set_weight(weights.wv[5], kResultPredicateOffset + value, kPredicateValueOffset + value);
    }

    // Stage 6: JUMP_IF reads a predicate slot or the immutable TRUE token.
    set_weight(weights.wq[6], kOpcodeOffset + static_cast<std::int64_t>(Opcode::JumpIf), 0, 2.0);
    set_weight(weights.wk[6], kRoleOffset + static_cast<std::int64_t>(TokenRole::Predicate), 0, 2.0);
    set_weight(weights.wk[6], kRoleOffset + static_cast<std::int64_t>(TokenRole::PredicateConstant), 0, 2.0);
    for (std::int64_t predicate = 0; predicate < kPredicateCount + 1; ++predicate) {
        set_weight(weights.wq[6], kPredicateSourceOffset + predicate, 1 + predicate);
        set_weight(weights.wk[6], kPredicateSourceOffset + predicate, 1 + predicate, 2.0);
    }
    for (std::int64_t value = 0; value < 2; ++value) {
        set_weight(weights.wv[6], kPredicateValueOffset + value, kResultPredicateOffset + value);
    }

    // Stages 7 and 8 build target and fallthrough branch candidates.
    set_weight(weights.wq[7], kRoleOffset + static_cast<std::int64_t>(TokenRole::BranchTarget), 0);
    set_weight(weights.wk[7], kOpcodeOffset + static_cast<std::int64_t>(Opcode::JumpIf), 0);
    set_weight(weights.wv[7], kOpcodeOffset + static_cast<std::int64_t>(Opcode::JumpIf),
               kCandidatePredicateOffset + 1);
    set_weight(weights.wq[8], kRoleOffset + static_cast<std::int64_t>(TokenRole::BranchFallthrough), 0);
    set_weight(weights.wk[8], kOpcodeOffset + static_cast<std::int64_t>(Opcode::JumpIf), 0);
    set_weight(weights.wv[8], kOpcodeOffset + static_cast<std::int64_t>(Opcode::JumpIf),
               kCandidatePredicateOffset);
    for (std::int64_t pc = 0; pc < kProgramSlots; ++pc) {
        set_weight(weights.wv[7], kTargetProgramCounterOffset + pc, kCandidateProgramCounterOffset + pc);
        set_weight(weights.wv[8], kNextProgramCounterOffset + pc, kCandidateProgramCounterOffset + pc);
    }

    // Stage 9 selects target/fallthrough through predicate-keyed attention.
    set_weight(weights.wq[9], kOpcodeOffset + static_cast<std::int64_t>(Opcode::JumpIf), 0, 2.0);
    set_weight(weights.wk[9], kRoleOffset + static_cast<std::int64_t>(TokenRole::BranchTarget), 0, 2.0);
    set_weight(weights.wk[9], kRoleOffset + static_cast<std::int64_t>(TokenRole::BranchFallthrough), 0, 2.0);
    for (std::int64_t value = 0; value < 2; ++value) {
        set_weight(weights.wq[9], kResultPredicateOffset + value, 1 + value);
        set_weight(weights.wk[9], kCandidatePredicateOffset + value, 1 + value);
    }
    for (std::int64_t pc = 0; pc < kProgramSlots; ++pc) {
        set_weight(weights.wv[9], kNextProgramCounterOffset + pc, kNextProgramCounterOffset + pc);
        set_weight(weights.wv[9], kCandidateProgramCounterOffset + pc, kNextProgramCounterOffset + pc);
    }

    // Stage 10: control and execution tokens copy next-PC/HALT from execution.
    set_weight(weights.wq[10], kRoleOffset + static_cast<std::int64_t>(TokenRole::Control), 0);
    set_weight(weights.wq[10], kRoleOffset + static_cast<std::int64_t>(TokenRole::Execution), 0);
    set_weight(weights.wk[10], kRoleOffset + static_cast<std::int64_t>(TokenRole::Execution), 0);
    for (std::int64_t pc = 0; pc < kProgramSlots; ++pc) {
        set_weight(weights.wv[10], kNextProgramCounterOffset + pc, kProgramCounterOffset + pc);
    }
    set_weight(weights.wv[10], kHaltOffset, kHaltOffset);

    for (std::int64_t stage = 0; stage < kStageCount; ++stage) {
        weights.wq[stage] = weights.wq[stage].detach().set_requires_grad(false);
        weights.wk[stage] = weights.wk[stage].detach().set_requires_grad(false);
        weights.wv[stage] = weights.wv[stage].detach().set_requires_grad(false);
        weights.wo[stage] = weights.wo[stage].detach().set_requires_grad(false);
    }
    return weights;
}

TensorArray make_attention_masks(const torch::TensorOptions& options) {
    TensorArray masks;
    const auto negative_infinity = -std::numeric_limits<double>::infinity();
    for (auto& mask : masks) {
        mask = torch::full({kTokenCount, kTokenCount}, negative_infinity, options);
        for (std::int64_t row = 0; row < kTokenCount; ++row) {
            allow(mask, row, kControlToken);
        }
    }
    masks[0].index_put_({kExecutionToken, kControlToken}, negative_infinity);
    for (std::int64_t row = kProgramTokenBegin; row < kProgramTokenBegin + kProgramSlots; ++row) {
        allow(masks[0], kExecutionToken, row);
    }
    for (std::int64_t row = kRegisterTokenBegin; row < kConstantTokenBegin + kValueCount; ++row) {
        allow(masks[1], kExecutionToken, row);
    }
    for (std::int64_t row = kRegisterTokenBegin; row < kRegisterTokenBegin + kRegisterCount; ++row) {
        allow(masks[2], kExecutionToken, row);
    }
    for (std::int64_t row = kRelationTokenBegin; row < kTokenCount; ++row) {
        allow(masks[3], kExecutionToken, row);
    }
    for (std::int64_t reg = 0; reg < kRegisterCount; ++reg) {
        const auto row = kRegisterTokenBegin + reg;
        allow(masks[4], row, row);
        allow(masks[4], row, kExecutionToken);
    }
    for (std::int64_t predicate = 0; predicate < kPredicateCount; ++predicate) {
        const auto row = kPredicateTokenBegin + predicate;
        allow(masks[5], row, row);
        allow(masks[5], row, kExecutionToken);
        allow(masks[6], kExecutionToken, row);
    }
    allow(masks[6], kExecutionToken, kPredicateConstantTokenBegin);
    allow(masks[6], kExecutionToken, kPredicateConstantTokenBegin + 1);
    allow(masks[7], kBranchTargetToken, kExecutionToken);
    allow(masks[8], kBranchFallthroughToken, kExecutionToken);
    masks[9].index_put_({kExecutionToken, kControlToken}, negative_infinity);
    allow(masks[9], kExecutionToken, kExecutionToken);
    allow(masks[9], kExecutionToken, kBranchTargetToken);
    allow(masks[9], kExecutionToken, kBranchFallthroughToken);
    allow(masks[10], kControlToken, kExecutionToken);
    allow(masks[10], kExecutionToken, kExecutionToken);
    for (auto& mask : masks) {
        mask = mask.detach().set_requires_grad(false);
    }
    return masks;
}

WriteProjections make_write_projections(const torch::TensorOptions& options) {
    WriteProjections projections;
    for (std::int64_t stage = 0; stage < kStageCount; ++stage) {
        for (std::int64_t component = 0; component < kWriteProjectionComponents; ++component) {
            projections.row[stage][component] = torch::zeros({kTokenCount, kTokenCount}, options);
            projections.feature[stage][component] = torch::zeros({kModelDim, kModelDim}, options);
        }
    }
    const auto enable = [&](std::int64_t stage,
                            std::int64_t component,
                            std::int64_t row_begin,
                            std::int64_t row_end,
                            std::int64_t feature_begin,
                            std::int64_t feature_end) {
        for (std::int64_t row = row_begin; row < row_end; ++row) {
            projections.row[stage][component].index_put_({row, row}, 1.0);
        }
        for (std::int64_t feature = feature_begin; feature < feature_end; ++feature) {
            projections.feature[stage][component].index_put_({feature, feature}, 1.0);
        }
    };
    enable(0, 0, kExecutionToken, kExecutionToken + 1, kOpcodeOffset, kHaltOffset);
    enable(0, 1, kExecutionToken, kExecutionToken + 1, kHaltOffset, kHaltOffset + 1);
    enable(0, 2, kExecutionToken, kExecutionToken + 1, kPredicateDestinationOffset,
           kPredicateSourceOffset + kPredicateCount + 1);
    enable(1, 0, kExecutionToken, kExecutionToken + 1, kLeftValueOffset, kLeftValueOffset + kValueCount);
    enable(2, 0, kExecutionToken, kExecutionToken + 1, kRightValueOffset, kRightValueOffset + kValueCount);
    enable(3, 0, kExecutionToken, kExecutionToken + 1, kResultValueOffset, kResultValueOffset + kValueCount);
    enable(3, 1, kExecutionToken, kExecutionToken + 1, kResultPredicateOffset, kResultPredicateOffset + 2);
    enable(4, 0, kRegisterTokenBegin, kRegisterTokenBegin + kRegisterCount, kRegisterValueOffset,
           kRegisterValueOffset + kValueCount);
    enable(5, 0, kPredicateTokenBegin, kPredicateTokenBegin + kPredicateCount, kPredicateValueOffset,
           kPredicateValueOffset + 2);
    enable(6, 0, kExecutionToken, kExecutionToken + 1, kResultPredicateOffset, kResultPredicateOffset + 2);
    enable(7, 0, kBranchTargetToken, kBranchTargetToken + 1, kCandidatePredicateOffset,
           kCandidateProgramCounterOffset + kProgramSlots);
    enable(8, 0, kBranchFallthroughToken, kBranchFallthroughToken + 1, kCandidatePredicateOffset,
           kCandidateProgramCounterOffset + kProgramSlots);
    enable(9, 0, kExecutionToken, kExecutionToken + 1, kNextProgramCounterOffset,
           kNextProgramCounterOffset + kProgramSlots);
    enable(10, 0, kControlToken, kControlToken + 1, kProgramCounterOffset,
           kProgramCounterOffset + kProgramSlots);
    enable(10, 1, kControlToken, kControlToken + 1, kHaltOffset, kHaltOffset + 1);
    enable(10, 2, kExecutionToken, kExecutionToken + 1, kOpcodeOffset,
           kCandidateProgramCounterOffset + kProgramSlots);
    for (std::int64_t stage = 0; stage < kStageCount; ++stage) {
        for (std::int64_t component = 0; component < kWriteProjectionComponents; ++component) {
            projections.row[stage][component] =
                projections.row[stage][component].detach().set_requires_grad(false);
            projections.feature[stage][component] =
                projections.feature[stage][component].detach().set_requires_grad(false);
        }
    }
    return projections;
}

void emit_relation(
    torch::Tensor& tokens,
    std::int64_t row,
    Opcode opcode,
    std::int64_t left,
    std::int64_t right,
    std::int64_t result,
    bool use_right) {
    set_one_hot(tokens, row, kRoleOffset, static_cast<std::int64_t>(TokenRole::Relation));
    set_one_hot(tokens, row, kOpcodeOffset, static_cast<std::int64_t>(opcode));
    set_one_hot(tokens, row, kLeftValueOffset, left);
    if (use_right) {
        set_one_hot(tokens, row, kRightValueOffset, right);
    }
    set_one_hot(tokens, row, kResultValueOffset, result);
}

void emit_predicate_relation(
    torch::Tensor& tokens, std::int64_t row, std::int64_t left, std::int64_t right) {
    set_one_hot(tokens, row, kRoleOffset, static_cast<std::int64_t>(TokenRole::Relation));
    set_one_hot(tokens, row, kOpcodeOffset, static_cast<std::int64_t>(Opcode::CmpEq));
    set_one_hot(tokens, row, kLeftValueOffset, left);
    set_one_hot(tokens, row, kRightValueOffset, right);
    set_one_hot(tokens, row, kResultPredicateOffset, left == right ? 1 : 0);
}

}  // namespace

ProgramImage compile_program(const std::array<Instruction, kProgramSlots>& program) {
    for (std::int64_t slot = 0; slot < kProgramSlots; ++slot) {
        validate_instruction(program[slot], slot);
    }
    if (program.back().opcode != Opcode::Halt) {
        throw std::invalid_argument("program must end with HALT");
    }

    const auto options = torch::TensorOptions().dtype(torch::kFloat64);
    auto tokens = torch::zeros({kTokenCount, kModelDim}, options);
    set_one_hot(tokens, kControlToken, kRoleOffset, static_cast<std::int64_t>(TokenRole::Control));
    set_one_hot(tokens, kControlToken, kProgramCounterOffset, 0);
    set_one_hot(tokens, kExecutionToken, kRoleOffset, static_cast<std::int64_t>(TokenRole::Execution));
    set_one_hot(tokens, kExecutionToken, kProgramCounterOffset, 0);

    for (std::int64_t slot = 0; slot < kProgramSlots; ++slot) {
        const auto row = kProgramTokenBegin + slot;
        const auto& instruction = program[slot];
        set_one_hot(tokens, row, kRoleOffset, static_cast<std::int64_t>(TokenRole::Instruction));
        set_one_hot(tokens, row, kOpcodeOffset, static_cast<std::int64_t>(instruction.opcode));
        set_one_hot(tokens, row, kProgramCounterOffset, slot);
        set_one_hot(
            tokens,
            row,
            kNextProgramCounterOffset,
            instruction.opcode == Opcode::Halt || slot == kProgramSlots - 1 ? slot : slot + 1);
        if (instruction.opcode == Opcode::Halt) {
            tokens.index_put_({row, kHaltOffset}, 1.0);
        } else if (instruction.opcode == Opcode::LoadConst || instruction.opcode == Opcode::Move ||
                   instruction.opcode == Opcode::Add) {
            set_one_hot(tokens, row, kDestinationOffset, instruction.dst);
        }
        if (instruction.opcode == Opcode::LoadConst) {
            set_one_hot(tokens, row, kValueOffset, instruction.lhs);
        } else if (instruction.opcode == Opcode::Move || instruction.opcode == Opcode::Add ||
                   instruction.opcode == Opcode::CmpEq) {
            set_one_hot(tokens, row, kLeftRegisterOffset, instruction.lhs);
            if (instruction.opcode == Opcode::Add || instruction.opcode == Opcode::CmpEq) {
                set_one_hot(tokens, row, kRightRegisterOffset, instruction.rhs);
            }
        }
        if (instruction.opcode == Opcode::CmpEq) {
            set_one_hot(tokens, row, kPredicateDestinationOffset, instruction.dst);
        } else if (instruction.opcode == Opcode::JumpIf) {
            set_one_hot(tokens, row, kPredicateSourceOffset, instruction.lhs);
            set_one_hot(tokens, row, kTargetProgramCounterOffset, instruction.rhs);
        }
    }

    for (std::int64_t reg = 0; reg < kRegisterCount; ++reg) {
        const auto row = kRegisterTokenBegin + reg;
        set_one_hot(tokens, row, kRoleOffset, static_cast<std::int64_t>(TokenRole::Register));
        set_one_hot(tokens, row, kDestinationOffset, reg);
        set_one_hot(tokens, row, kRegisterValueOffset, 0);
    }
    for (std::int64_t predicate = 0; predicate < kPredicateCount; ++predicate) {
        const auto row = kPredicateTokenBegin + predicate;
        set_one_hot(tokens, row, kRoleOffset, static_cast<std::int64_t>(TokenRole::Predicate));
        set_one_hot(tokens, row, kPredicateDestinationOffset, predicate);
        set_one_hot(tokens, row, kPredicateSourceOffset, predicate);
        set_one_hot(tokens, row, kPredicateValueOffset, 0);
    }
    for (std::int64_t value = 0; value < 2; ++value) {
        const auto row = kPredicateConstantTokenBegin + value;
        set_one_hot(tokens, row, kRoleOffset, static_cast<std::int64_t>(TokenRole::PredicateConstant));
        set_one_hot(tokens, row, kPredicateValueOffset, value);
        if (value == 1) {
            set_one_hot(tokens, row, kPredicateSourceOffset, kPredicateCount);
        }
    }
    set_one_hot(tokens, kBranchTargetToken, kRoleOffset, static_cast<std::int64_t>(TokenRole::BranchTarget));
    set_one_hot(tokens, kBranchFallthroughToken, kRoleOffset,
                static_cast<std::int64_t>(TokenRole::BranchFallthrough));
    for (std::int64_t value = 0; value < kValueCount; ++value) {
        const auto row = kConstantTokenBegin + value;
        set_one_hot(tokens, row, kRoleOffset, static_cast<std::int64_t>(TokenRole::Constant));
        set_one_hot(tokens, row, kValueOffset, value);
    }

    std::int64_t row = kRelationTokenBegin;
    for (std::int64_t value = 0; value < kValueCount; ++value) {
        emit_relation(tokens, row++, Opcode::LoadConst, value, 0, value, false);
    }
    for (std::int64_t value = 0; value < kValueCount; ++value) {
        emit_relation(tokens, row++, Opcode::Move, value, 0, value, false);
    }
    for (std::int64_t left = 0; left < kValueCount; ++left) {
        for (std::int64_t right = 0; left + right < kValueCount; ++right) {
            emit_relation(tokens, row++, Opcode::Add, left, right, left + right, true);
        }
    }
    emit_relation(tokens, row++, Opcode::Halt, 0, 0, 0, false);
    for (std::int64_t left = 0; left < kValueCount; ++left) {
        for (std::int64_t right = 0; right < kValueCount; ++right) {
            emit_predicate_relation(tokens, row++, left, right);
        }
    }
    emit_relation(tokens, row++, Opcode::JumpIf, 0, 0, 0, false);
    if (row != kTokenCount) {
        throw std::logic_error("relation token count mismatch");
    }

    tokens = tokens.detach().set_requires_grad(false);
    return ProgramImage{
        tokens,
        make_attention_masks(options),
        make_write_projections(options),
        make_weights(options),
    };
}

}  // namespace cmz::vm2
