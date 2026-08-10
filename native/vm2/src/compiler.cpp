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

void enable_write(torch::Tensor& mask, std::int64_t row, std::int64_t offset, std::int64_t width) {
    mask.index_put_({row, torch::indexing::Slice(offset, offset + width)}, 1.0);
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
        case Opcode::Halt:
            if (slot != kProgramSlots - 1) {
                throw std::invalid_argument("HALT must occupy the final program slot");
            }
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
    set_weight(weights.wk[2], kRoleOffset + static_cast<std::int64_t>(TokenRole::Register), 0, 2.0);
    for (std::int64_t reg = 0; reg < kRegisterCount; ++reg) {
        set_weight(weights.wq[2], kRightRegisterOffset + reg, 1 + reg);
        set_weight(weights.wk[2], kDestinationOffset + reg, 1 + reg);
    }
    for (std::int64_t value = 0; value < kValueCount; ++value) {
        set_weight(weights.wv[2], kRegisterValueOffset + value, kRightValueOffset + value);
    }

    // Stage 3: select the universal ALU relation by opcode and operand values.
    for (std::int64_t opcode = 0; opcode < 4; ++opcode) {
        set_weight(weights.wq[3], kOpcodeOffset + opcode, opcode, 4.0);
        set_weight(weights.wk[3], kOpcodeOffset + opcode, opcode, 4.0);
    }
    for (std::int64_t value = 0; value < kValueCount; ++value) {
        set_weight(weights.wq[3], kLeftValueOffset + value, 4 + value);
        set_weight(weights.wk[3], kLeftValueOffset + value, 4 + value);
        set_weight(weights.wq[3], kRightValueOffset + value, 20 + value);
        set_weight(weights.wk[3], kRightValueOffset + value, 20 + value);
        set_weight(weights.wv[3], kResultValueOffset + value, kResultValueOffset + value);
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

    // Stage 5: control and execution tokens copy next-PC/HALT from execution.
    set_weight(weights.wq[5], kRoleOffset + static_cast<std::int64_t>(TokenRole::Control), 0);
    set_weight(weights.wq[5], kRoleOffset + static_cast<std::int64_t>(TokenRole::Execution), 0);
    set_weight(weights.wk[5], kRoleOffset + static_cast<std::int64_t>(TokenRole::Execution), 0);
    for (std::int64_t pc = 0; pc < kProgramSlots; ++pc) {
        set_weight(weights.wv[5], kNextProgramCounterOffset + pc, kProgramCounterOffset + pc);
    }
    set_weight(weights.wv[5], kHaltOffset, kHaltOffset);

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
    allow(masks[5], kControlToken, kExecutionToken);
    allow(masks[5], kExecutionToken, kExecutionToken);
    for (auto& mask : masks) {
        mask = mask.detach().set_requires_grad(false);
    }
    return masks;
}

TensorArray make_write_masks(const torch::TensorOptions& options) {
    TensorArray masks;
    for (auto& mask : masks) {
        mask = torch::zeros({kTokenCount, kModelDim}, options);
    }
    enable_write(masks[0], kExecutionToken, kOpcodeOffset, 4);
    enable_write(masks[0], kExecutionToken, kDestinationOffset, 4);
    enable_write(masks[0], kExecutionToken, kLeftRegisterOffset, 4);
    enable_write(masks[0], kExecutionToken, kRightRegisterOffset, 4);
    enable_write(masks[0], kExecutionToken, kValueOffset, kValueCount);
    enable_write(masks[0], kExecutionToken, kNextProgramCounterOffset, kProgramSlots);
    enable_write(masks[0], kExecutionToken, kHaltOffset, 1);
    enable_write(masks[1], kExecutionToken, kLeftValueOffset, kValueCount);
    enable_write(masks[2], kExecutionToken, kRightValueOffset, kValueCount);
    enable_write(masks[3], kExecutionToken, kResultValueOffset, kValueCount);
    for (std::int64_t reg = 0; reg < kRegisterCount; ++reg) {
        enable_write(masks[4], kRegisterTokenBegin + reg, kRegisterValueOffset, kValueCount);
    }
    enable_write(masks[5], kControlToken, kProgramCounterOffset, kProgramSlots);
    enable_write(masks[5], kControlToken, kHaltOffset, 1);
    enable_write(masks[5], kExecutionToken, kProgramCounterOffset, kProgramSlots);
    enable_write(masks[5], kExecutionToken, kOpcodeOffset, 4);
    enable_write(masks[5], kExecutionToken, kDestinationOffset, 12);
    enable_write(masks[5], kExecutionToken, kValueOffset, 64);
    enable_write(masks[5], kExecutionToken, kNextProgramCounterOffset, kProgramSlots);
    enable_write(masks[5], kExecutionToken, kHaltOffset, 1);
    for (auto& mask : masks) {
        mask = mask.detach().set_requires_grad(false);
    }
    return masks;
}

std::pair<TensorArray, TensorArray> make_state_projections(const torch::TensorOptions& options) {
    const auto write_masks = make_write_masks(options);
    TensorArray keep;
    TensorArray take;
    for (std::int64_t stage = 0; stage < kStageCount; ++stage) {
        take[stage] = torch::diag_embed(write_masks[stage]).detach().set_requires_grad(false);
        keep[stage] = torch::diag_embed(1.0 - write_masks[stage]).detach().set_requires_grad(false);
    }
    return {keep, take};
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
        set_one_hot(tokens, row, kNextProgramCounterOffset, slot == kProgramSlots - 1 ? slot : slot + 1);
        if (instruction.opcode == Opcode::Halt) {
            tokens.index_put_({row, kHaltOffset}, 1.0);
        } else {
            set_one_hot(tokens, row, kDestinationOffset, instruction.dst);
        }
        if (instruction.opcode == Opcode::LoadConst) {
            set_one_hot(tokens, row, kValueOffset, instruction.lhs);
        } else if (instruction.opcode == Opcode::Move || instruction.opcode == Opcode::Add) {
            set_one_hot(tokens, row, kLeftRegisterOffset, instruction.lhs);
            if (instruction.opcode == Opcode::Add) {
                set_one_hot(tokens, row, kRightRegisterOffset, instruction.rhs);
            }
        }
    }

    for (std::int64_t reg = 0; reg < kRegisterCount; ++reg) {
        const auto row = kRegisterTokenBegin + reg;
        set_one_hot(tokens, row, kRoleOffset, static_cast<std::int64_t>(TokenRole::Register));
        set_one_hot(tokens, row, kDestinationOffset, reg);
        set_one_hot(tokens, row, kRegisterValueOffset, 0);
    }
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
    if (row != kTokenCount) {
        throw std::logic_error("relation token count mismatch");
    }

    tokens = tokens.detach().set_requires_grad(false);
    auto [keep_projections, take_projections] = make_state_projections(options);
    return ProgramImage{
        tokens,
        make_attention_masks(options),
        std::move(keep_projections),
        std::move(take_projections),
        make_weights(options),
    };
}

}  // namespace cmz::vm2
