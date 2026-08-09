#include <array>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "cmz_vm2/attention.h"
#include "cmz_vm2/compiler.h"
#include "cmz_vm2/machine.h"

namespace {

std::int64_t one_hot_index(const torch::Tensor& row, std::int64_t offset, std::int64_t width) {
    return row.slice(0, offset, offset + width).argmax().item<std::int64_t>();
}

std::int64_t register_value(const torch::Tensor& state, std::int64_t reg) {
    return one_hot_index(
        state[cmz::vm2::kRegisterTokenBegin + reg],
        cmz::vm2::kRegisterValueOffset,
        cmz::vm2::kValueCount);
}

torch::Tensor apply_stage_for_test(
    const cmz::vm2::ProgramImage& image,
    torch::Tensor state,
    std::int64_t stage) {
    const auto attention = cmz::vm2::self_attention(
        state,
        image.weights.wq[stage],
        image.weights.wk[stage],
        image.weights.wv[stage],
        image.attention_masks[stage]);
    const auto projected = torch::matmul(attention.output, image.weights.wo[stage]);
    return state * (1.0 - image.write_masks[stage]) + projected * image.write_masks[stage];
}

std::vector<std::int64_t> pc_trace(const torch::Tensor& states) {
    std::vector<std::int64_t> result;
    for (std::int64_t step = 0; step < states.size(0); ++step) {
        result.push_back(one_hot_index(
            states[step][cmz::vm2::kControlToken],
            cmz::vm2::kProgramCounterOffset,
            cmz::vm2::kProgramSlots));
    }
    return result;
}

void require_add(std::int64_t left, std::int64_t right) {
    using cmz::vm2::Instruction;
    using cmz::vm2::Opcode;
    const std::array program{
        Instruction{Opcode::LoadConst, 0, left, 0},
        Instruction{Opcode::LoadConst, 1, right, 0},
        Instruction{Opcode::Add, 2, 0, 1},
        Instruction{Opcode::Halt, 0, 0, 0},
    };
    const auto result = cmz::vm2::run(cmz::vm2::compile_program(program), 4);
    const auto actual = register_value(result.final_state, 2);
    if (actual != left + right) {
        const auto after_left = register_value(result.state_trace[1], 0);
        const auto after_right = register_value(result.state_trace[2], 1);
        auto add_state = result.state_trace[2];
        const auto add_image = cmz::vm2::compile_program(program);
        add_state = apply_stage_for_test(add_image, add_state, 0);
        const auto decoded_opcode = one_hot_index(
            add_state[cmz::vm2::kExecutionToken], cmz::vm2::kOpcodeOffset, 4);
        const auto decoded_left_register = one_hot_index(
            add_state[cmz::vm2::kExecutionToken],
            cmz::vm2::kLeftRegisterOffset,
            cmz::vm2::kRegisterCount);
        const auto left_attention = cmz::vm2::self_attention(
            add_state,
            add_image.weights.wq[1],
            add_image.weights.wk[1],
            add_image.weights.wv[1],
            add_image.attention_masks[1]);
        const auto left_winner = left_attention.winners[cmz::vm2::kExecutionToken].item<std::int64_t>();
        const auto query_id0 = left_attention.q[cmz::vm2::kExecutionToken][2].item<double>();
        const auto query_id1 = left_attention.q[cmz::vm2::kExecutionToken][3].item<double>();
        const auto left_score_r0 = left_attention.scores[cmz::vm2::kExecutionToken]
                                       [cmz::vm2::kRegisterTokenBegin]
                                           .item<double>();
        const auto left_score_r1 = left_attention.scores[cmz::vm2::kExecutionToken]
                                       [cmz::vm2::kRegisterTokenBegin + 1]
                                           .item<double>();
        add_state = apply_stage_for_test(add_image, add_state, 1);
        add_state = apply_stage_for_test(add_image, add_state, 2);
        const auto routed_left = one_hot_index(
            add_state[cmz::vm2::kExecutionToken],
            cmz::vm2::kLeftValueOffset,
            cmz::vm2::kValueCount);
        const auto routed_right = one_hot_index(
            add_state[cmz::vm2::kExecutionToken],
            cmz::vm2::kRightValueOffset,
            cmz::vm2::kValueCount);
        const auto alu_attention = cmz::vm2::self_attention(
            add_state,
            add_image.weights.wq[3],
            add_image.weights.wk[3],
            add_image.weights.wv[3],
            add_image.attention_masks[3]);
        const auto relation_winner =
            alu_attention.winners[cmz::vm2::kExecutionToken].item<std::int64_t>();
        add_state = apply_stage_for_test(add_image, add_state, 3);
        const auto alu_result = one_hot_index(
            add_state[cmz::vm2::kExecutionToken],
            cmz::vm2::kResultValueOffset,
            cmz::vm2::kValueCount);
        throw std::runtime_error(
            "attention ALU returned " + std::to_string(actual) + " for " + std::to_string(left) + "+" +
            std::to_string(right) + " (r0=" + std::to_string(after_left) + ", r1=" +
            std::to_string(after_right) + ", opcode=" + std::to_string(decoded_opcode) + ", lhs-reg=" +
            std::to_string(decoded_left_register) + ", left-winner=" + std::to_string(left_winner) +
            ", q=" + std::to_string(query_id0) + "/" + std::to_string(query_id1) + ", scores=" +
            std::to_string(left_score_r0) + "/" + std::to_string(left_score_r1) +
            ", left=" + std::to_string(routed_left) + ", right=" + std::to_string(routed_right) +
            ", relation=" + std::to_string(relation_winner) + ", alu=" + std::to_string(alu_result) + ")");
    }
}

void require_move(std::int64_t value) {
    using cmz::vm2::Instruction;
    using cmz::vm2::Opcode;
    const std::array program{
        Instruction{Opcode::LoadConst, 0, value, 0},
        Instruction{Opcode::Move, 3, 0, 0},
        Instruction{Opcode::Move, 2, 3, 0},
        Instruction{Opcode::Halt, 0, 0, 0},
    };
    const auto result = cmz::vm2::run(cmz::vm2::compile_program(program), 4);
    if (register_value(result.final_state, 2) != value) {
        throw std::runtime_error("attention MOVE returned an incorrect value");
    }
}

}  // namespace

int main() {
    try {
        using cmz::vm2::Instruction;
        using cmz::vm2::Opcode;
        const std::array program{
            Instruction{Opcode::LoadConst, 0, 2, 0},
            Instruction{Opcode::LoadConst, 1, 3, 0},
            Instruction{Opcode::Add, 2, 0, 1},
            Instruction{Opcode::Halt, 0, 0, 0},
        };
        const auto image = cmz::vm2::compile_program(program);
        const auto decode = cmz::vm2::self_attention(
            image.tokens,
            image.weights.wq[0],
            image.weights.wk[0],
            image.weights.wv[0],
            image.attention_masks[0]);
        if (decode.winners[cmz::vm2::kExecutionToken].item<std::int64_t>() !=
            cmz::vm2::kProgramTokenBegin) {
            throw std::runtime_error("decode hardmax must select instruction token at current PC");
        }
        const auto first_state = cmz::vm2::transition(image, image.tokens);
        if (one_hot_index(
                first_state[cmz::vm2::kControlToken],
                cmz::vm2::kProgramCounterOffset,
                cmz::vm2::kProgramSlots) != 1) {
            throw std::runtime_error("first attention transition must advance PC to 1");
        }
        if (register_value(first_state, 0) != 2) {
            throw std::runtime_error("first attention transition must load r0=2");
        }
        const auto result = cmz::vm2::run(image, 8);
        if (result.steps != 4) {
            throw std::runtime_error("program must halt after four VM steps");
        }
        if (register_value(result.final_state, 2) != 5) {
            throw std::runtime_error("ADD must produce r2=5");
        }
        if (result.final_state[cmz::vm2::kControlToken][cmz::vm2::kHaltOffset].item<double>() != 1.0) {
            throw std::runtime_error("HALT must set the halt token");
        }
        if (pc_trace(result.state_trace) != std::vector<std::int64_t>({0, 1, 2, 3, 3})) {
            throw std::runtime_error("PC trace must be exactly 0,1,2,3,3");
        }
        const auto repeated = cmz::vm2::run(image, 8);
        if (!torch::equal(result.state_trace, repeated.state_trace)) {
            throw std::runtime_error("repeated VM runs must be byte-identical");
        }
        for (std::int64_t left = 0; left < cmz::vm2::kValueCount; ++left) {
            for (std::int64_t right = 0; left + right < cmz::vm2::kValueCount; ++right) {
                require_add(left, right);
            }
        }
        for (std::int64_t value = 0; value < cmz::vm2::kValueCount; ++value) {
            require_move(value);
        }
        try {
            (void)cmz::vm2::run(image, 3);
            throw std::runtime_error("step limit must fail before HALT");
        } catch (const std::runtime_error& error) {
            if (std::string(error.what()) != "VM step limit reached before HALT") {
                throw;
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
