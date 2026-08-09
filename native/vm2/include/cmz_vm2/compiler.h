#pragma once

#include <array>

#include "cmz_vm2/schema.h"

namespace cmz::vm2 {

ProgramImage compile_program(const std::array<Instruction, kProgramSlots>& program);

}  // namespace cmz::vm2
