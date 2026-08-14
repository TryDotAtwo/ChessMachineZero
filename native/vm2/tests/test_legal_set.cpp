#include <stdexcept>

#include "cmz_vm2/legal_set.h"
#include "cmz_vm2/move_policy.h"

using namespace cmz::vm2;

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

int main() {
    const auto options = torch::TensorOptions().dtype(torch::kFloat64);
    const auto pawn = torch::tensor({{1., 0.}, {0., 1.}}, options);
    const auto knight = torch::tensor({{0., 1.}, {1., 0.}}, options);
    const auto assembler = compile_legal_set_assembler(2, 2, options);
    const auto legal_set = assemble_legal_set(assembler, pawn, knight);
    require(torch::equal(
                legal_set,
                torch::tensor({{1., 0.}, {0., 1.}, {0., 1.}, {1., 0.}}, options)),
            "frozen row matrices must preserve pawn then knight predicate tokens");

    const auto policy = compile_first_legal_policy(1, options);
    const auto choice = run_move_policy(
        policy, torch::zeros({4, 1}, options), legal_set);
    require(torch::equal(choice.selection, torch::tensor({0., 1., 0., 0.}, options)),
            "one policy hardmax must select across the combined rule legal set");
    return 0;
}
