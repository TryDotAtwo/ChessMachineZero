#include <limits>
#include <stdexcept>

#include "cmz_vm2/move_policy.h"

using namespace cmz::vm2;

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

int main() {
    const auto options = torch::TensorOptions().dtype(torch::kFloat64);
    const auto candidates = torch::tensor({{1., 0.}, {0., 1.}, {1., 1.}}, options);
    const auto legality_tokens = torch::tensor({{1., 0.}, {0., 1.}, {0., 1.}}, options);
    const auto fallback = compile_first_legal_policy(2, options);
    const auto result = run_move_policy(fallback, candidates, legality_tokens);

    require(result.scores.index({0}).item<double>() < -1e8 &&
                result.scores.index({1}).item<double>() == 0. &&
                result.scores.index({2}).item<double>() == 0.,
            "fixed predicate projection must suppress NOT_LEGAL candidates");
    require(torch::equal(result.selection, torch::tensor({0., 1., 0.}, options)),
            "fallback must hardmax-select the first trace-legal candidate");
    require(!fallback.candidate_projection.requires_grad(),
            "temporary policy weights must be frozen");

    auto learned = fallback;
    learned.candidate_projection = torch::tensor({{2.}, {0.}}, options);
    const auto learned_result = run_move_policy(learned, candidates, legality_tokens);
    require(torch::equal(learned_result.selection, torch::tensor({0., 0., 1.}, options)),
            "the same policy contract must accept a future learned scorer");
    return 0;
}
