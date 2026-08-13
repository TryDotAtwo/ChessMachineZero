#include <stdexcept>

#include "cmz_vm2/knight_geometry.h"

int main() {
    const auto circuit = cmz::vm2::compile_knight_geometry();
    const auto result = cmz::vm2::evaluate_knight_geometry(circuit);
    if (result.sizes() != std::vector<std::int64_t>{64, 64}) {
        throw std::runtime_error("knight geometry must return a 64x64 token relation");
    }
    for (std::int64_t source = 0; source < 64; ++source) {
        for (std::int64_t target = 0; target < 64; ++target) {
            const auto file_delta = std::abs(source % 8 - target % 8);
            const auto rank_delta = std::abs(source / 8 - target / 8);
            const bool expected =
                (file_delta == 1 && rank_delta == 2) ||
                (file_delta == 2 && rank_delta == 1);
            if (result.index({source, target}).item<std::int64_t>() != expected) {
                throw std::runtime_error("fixed attention knight geometry mismatch");
            }
        }
    }
    if (circuit.tokens.requires_grad() || circuit.wq.requires_grad() ||
        circuit.wk.requires_grad() || circuit.wv.requires_grad()) {
        throw std::runtime_error("knight circuit must be frozen");
    }
    return 0;
}
