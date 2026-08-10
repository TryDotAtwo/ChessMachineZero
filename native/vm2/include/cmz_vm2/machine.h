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
RunResult run_fixed(const ProgramImage& image, std::int64_t inference_steps);

}  // namespace cmz::vm2
