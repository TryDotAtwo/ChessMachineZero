#include "cmz_vm2/knight_circuit.h"

#include "cmz_vm2/attention.h"

namespace cmz::vm2 {

torch::Tensor run_knight_circuit(
    const KnightCircuit& circuit, const torch::Tensor& input_tokens) {
    auto state = input_tokens;
    for (std::int64_t stage = 0; stage < kKnightStages; ++stage) {
        const auto attention = self_attention(
            state, circuit.wq[stage], circuit.wk[stage], circuit.wv[stage], circuit.masks[stage]);
        state = state + torch::matmul(
            torch::matmul(circuit.row_writes[stage], attention.output - state),
            circuit.feature_writes[stage]);
    }
    return torch::matmul(
        torch::matmul(circuit.output_row, state), circuit.output_feature).to(torch::kInt64);
}

}  // namespace cmz::vm2
