#pragma once

#include <array>

#include "cmz_vm2/schema.h"

namespace cmz::vm2 {

ProgramImage compile_program(const std::array<Instruction, kProgramSlots>& program);

template <std::size_t N>
ProgramImage compile_program(const std::array<Instruction, N>& source) {
    static_assert(N <= static_cast<std::size_t>(kProgramSlots));
    std::array<Instruction, kProgramSlots> program{};
    program.fill(Instruction{Opcode::Halt, 0, 0, 0});
    for (std::size_t index = 0; index < N; ++index) {
        program[index] = source[index];
    }
    return compile_program(program);
}

}  // namespace cmz::vm2
