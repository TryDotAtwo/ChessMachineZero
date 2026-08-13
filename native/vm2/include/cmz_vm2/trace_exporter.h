#pragma once

#include <vector>
#include <string>

#include "cmz_vm2/schema.h"

namespace cmz::vm2 {

struct StageSnapshot {
    torch::Tensor x;
    torch::Tensor q;
    torch::Tensor k;
    torch::Tensor v;
    torch::Tensor scores;
    torch::Tensor winners;
    torch::Tensor attention;
    torch::Tensor y;
    torch::Tensor x_prime;

    torch::Tensor apply_write_reference(const ProgramImage& image, std::int64_t stage) const;
};

struct TransitionTrace {
    std::vector<StageSnapshot> stages;
    torch::Tensor final_state;
};

StageSnapshot trace_stage(
    const ProgramImage& image, const torch::Tensor& state, std::int64_t stage);
TransitionTrace trace_transition(const ProgramImage& image, const torch::Tensor& initial_state);
void write_trace_json(
    const std::string& path,
    const ProgramImage& image,
    const torch::Tensor& initial_state,
    const std::string& commit);

}  // namespace cmz::vm2
