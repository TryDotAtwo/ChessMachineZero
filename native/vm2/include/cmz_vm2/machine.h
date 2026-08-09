#pragma once

#include <cstdint>

#include "cmz_vm2/schema.h"

namespace cmz::vm2 {

struct RunResult {
    torch::Tensor final_state;
    torch::Tensor state_trace;
    std::int64_t steps;
};

torch::Tensor transition(const ProgramImage& image, const torch::Tensor& state);
RunResult run(const ProgramImage& image, std::int64_t max_steps);

}  // namespace cmz::vm2
