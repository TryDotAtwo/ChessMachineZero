#include "cmz_vm2/knight_geometry.h"

#include <ATen/ops/one_hot.h>

namespace cmz::vm2 {

KnightGeometryCircuit compile_knight_geometry() {
    const auto options = torch::TensorOptions().dtype(torch::kFloat64);
    auto tokens = torch::zeros({64 * 64, 3}, options);
    for (std::int64_t source = 0; source < 64; ++source) {
        for (std::int64_t target = 0; target < 64; ++target) {
            const auto file_delta = std::abs(source % 8 - target % 8);
            const auto rank_delta = std::abs(source / 8 - target / 8);
            const bool legal = (file_delta == 1 && rank_delta == 2) ||
                               (file_delta == 2 && rank_delta == 1);
            tokens.index_put_({source * 64 + target, 0}, 1.0);
            tokens.index_put_({source * 64 + target, legal ? 2 : 1}, 1.0);
        }
    }
    auto wq = torch::zeros({3, 2}, options);
    wq.index_put_({1, 0}, 1.0);
    wq.index_put_({2, 1}, 1.0);
    auto wk = torch::eye(2, options);
    auto wv = torch::eye(2, options);
    auto class_keys = torch::eye(2, options);
    return KnightGeometryCircuit{
        tokens.detach().set_requires_grad(false),
        wq.detach().set_requires_grad(false),
        wk.detach().set_requires_grad(false),
        wv.detach().set_requires_grad(false),
        class_keys.detach().set_requires_grad(false)};
}

torch::Tensor evaluate_knight_geometry(const KnightGeometryCircuit& circuit) {
    const auto q = torch::matmul(circuit.tokens, circuit.wq);
    const auto k = torch::matmul(circuit.class_keys, circuit.wk);
    const auto v = torch::matmul(circuit.class_keys, circuit.wv);
    const auto scores = torch::matmul(q, k.t());
    const auto winners = std::get<1>(scores.max(1, false));
    const auto hardmax = at::one_hot(winners, 2).to(circuit.tokens.scalar_type());
    return torch::matmul(hardmax, v).select(1, 1).reshape({64, 64}).to(torch::kInt64);
}

}  // namespace cmz::vm2
