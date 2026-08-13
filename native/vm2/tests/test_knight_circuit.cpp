#include <cmath>
#include <stdexcept>

#include "cmz_vm2/knight_circuit.h"

int main() {
    const auto circuit = cmz::vm2::compile_knight_circuit();
    for (std::int64_t source = 0; source < 64; ++source) {
        for (std::int64_t target = 0; target < 64; ++target) {
            const auto input = cmz::vm2::bind_knight_pair(circuit, source, target);
            const auto output = cmz::vm2::run_knight_circuit(circuit, input);
            const auto df = std::abs(source % 8 - target % 8);
            const auto dr = std::abs(source / 8 - target / 8);
            const bool expected = (df == 1 && dr == 2) || (df == 2 && dr == 1);
            if (output.item<std::int64_t>() != static_cast<std::int64_t>(expected)) {
                throw std::runtime_error("parametric knight circuit mismatch");
            }
        }
    }

    const auto first = cmz::vm2::bind_knight_pair(circuit, 1, 18);
    const auto second = cmz::vm2::bind_knight_pair(circuit, 1, 17);
    const auto changed = first.ne(second).nonzero();
    if (changed.size(0) != 2 || changed.select(1, 0).ne(0).any().item<bool>()) {
        throw std::runtime_error("binding may change only target coordinate features on token zero");
    }
    if (circuit.tokens.requires_grad()) throw std::runtime_error("tokens must be frozen");
    for (std::int64_t stage = 0; stage < cmz::vm2::kKnightStages; ++stage) {
        if (circuit.wq[stage].requires_grad() || circuit.wk[stage].requires_grad() ||
            circuit.wv[stage].requires_grad() || circuit.masks[stage].requires_grad() ||
            circuit.row_writes[stage].requires_grad() || circuit.feature_writes[stage].requires_grad()) {
            throw std::runtime_error("all knight rule tensors must be frozen");
        }
    }
    if (circuit.output_row.requires_grad() || circuit.output_feature.requires_grad()) {
        throw std::runtime_error("output selectors must be frozen");
    }
    return 0;
}
