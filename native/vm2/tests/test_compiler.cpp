#include <array>
#include <stdexcept>

#include <torch/torch.h>

#include "cmz_vm2/compiler.h"

namespace {

template <typename Function>
void require_invalid_argument(Function&& function, const char* message) {
    try {
        function();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error(message);
}

}  // namespace

int main() {
    static_assert(cmz::vm2::kRegisterCount == 4);
    static_assert(cmz::vm2::kValueCount == 16);
    static_assert(cmz::vm2::kProgramSlots == 16);
    static_assert(cmz::vm2::kStageCount == 11);
    if (static_cast<int>(cmz::vm2::Opcode::Add) != 3) {
        return 1;
    }

    using cmz::vm2::Instruction;
    using cmz::vm2::Opcode;
    const std::array program{
        Instruction{Opcode::LoadConst, 0, 2, 0},
        Instruction{Opcode::LoadConst, 1, 3, 0},
        Instruction{Opcode::Add, 2, 0, 1},
        Instruction{Opcode::Halt, 0, 0, 0},
    };
    const auto image = cmz::vm2::compile_program(program);
    if (image.tokens.dim() != 2 || image.tokens.scalar_type() != torch::kFloat64) {
        throw std::runtime_error("compiler must emit a float64 token matrix");
    }
    for (std::int64_t stage = 0; stage < cmz::vm2::kStageCount; ++stage) {
        for (std::int64_t component = 0; component < cmz::vm2::kWriteProjectionComponents; ++component) {
            const auto& row = image.write_projections.row[stage][component];
            const auto& feature = image.write_projections.feature[stage][component];
            if (row.sizes() != torch::IntArrayRef({cmz::vm2::kTokenCount, cmz::vm2::kTokenCount}) ||
                feature.sizes() != torch::IntArrayRef({cmz::vm2::kModelDim, cmz::vm2::kModelDim})) {
                throw std::runtime_error("write routing must use compact row/feature matrices");
            }
        }
    }
    for (const auto& weight : image.weights.wq) {
        if (weight.requires_grad()) {
            throw std::runtime_error("compiled VM weights must be frozen");
        }
    }

    auto changed_program = program;
    changed_program[0].lhs = 4;
    const auto changed_image = cmz::vm2::compile_program(changed_program);
    if (torch::equal(image.tokens, changed_image.tokens)) {
        throw std::runtime_error("program bytecode must change program tokens");
    }
    for (std::int64_t stage = 0; stage < cmz::vm2::kStageCount; ++stage) {
        if (!torch::equal(image.weights.wq[stage], changed_image.weights.wq[stage]) ||
            !torch::equal(image.weights.wk[stage], changed_image.weights.wk[stage]) ||
            !torch::equal(image.weights.wv[stage], changed_image.weights.wv[stage]) ||
            !torch::equal(image.weights.wo[stage], changed_image.weights.wo[stage])) {
            throw std::runtime_error("universal VM weights must not depend on bytecode");
        }
    }

    auto invalid_register = program;
    invalid_register[0].dst = 4;
    require_invalid_argument(
        [&] { (void)cmz::vm2::compile_program(invalid_register); },
        "invalid destination register must be rejected");
    auto invalid_constant = program;
    invalid_constant[0].lhs = 16;
    require_invalid_argument(
        [&] { (void)cmz::vm2::compile_program(invalid_constant); },
        "constant outside 0..15 must be rejected");
    std::array<Instruction, cmz::vm2::kProgramSlots> missing_halt;
    missing_halt.fill(Instruction{Opcode::Move, 0, 1, 0});
    require_invalid_argument(
        [&] { (void)cmz::vm2::compile_program(missing_halt); },
        "program without HALT must be rejected");
    const std::array predicate_program{
        Instruction{Opcode::CmpEq, 0, 0, 1},
        Instruction{Opcode::JumpIf, 0, 0, 3},
        Instruction{Opcode::JumpIf, 0, cmz::vm2::kPredicateCount, 0},
        Instruction{Opcode::Halt, 0, 0, 0},
    };
    (void)cmz::vm2::compile_program(predicate_program);
    auto invalid_predicate = predicate_program;
    invalid_predicate[0].dst = cmz::vm2::kPredicateCount;
    require_invalid_argument(
        [&] { (void)cmz::vm2::compile_program(invalid_predicate); },
        "invalid predicate destination must be rejected");
    auto invalid_target = predicate_program;
    invalid_target[1].rhs = cmz::vm2::kProgramSlots;
    require_invalid_argument(
        [&] { (void)cmz::vm2::compile_program(invalid_target); },
        "invalid branch target must be rejected");
    return 0;
}
