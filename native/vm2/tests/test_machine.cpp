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
    for (std::int64_t component = 0; component < cmz::vm2::kWriteProjectionComponents; ++component) {
        const auto delta = projected - state;
        state = state + torch::matmul(
                            torch::matmul(image.write_projections.row[stage][component], delta),
                            image.write_projections.feature[stage][component]);
    }
    return state;
}

std::int64_t predicate_value(const torch::Tensor& state, std::int64_t predicate) {
    return one_hot_index(
        state[cmz::vm2::kPredicateTokenBegin + predicate],
        cmz::vm2::kPredicateValueOffset,
        2);
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
    const auto result = cmz::vm2::run_fixed(cmz::vm2::compile_program(program), 4);
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
    const auto result = cmz::vm2::run_fixed(cmz::vm2::compile_program(program), 4);
    if (register_value(result.final_state, 2) != value) {
        throw std::runtime_error("attention MOVE returned an incorrect value");
    }
}

void require_cmp_eq(std::int64_t left, std::int64_t right) {
    using cmz::vm2::Instruction;
    using cmz::vm2::Opcode;
    const std::array program{
        Instruction{Opcode::LoadConst, 0, left, 0},
        Instruction{Opcode::LoadConst, 1, right, 0},
        Instruction{Opcode::CmpEq, 0, 0, 1},
        Instruction{Opcode::Halt, 0, 0, 0},
    };
    const auto result = cmz::vm2::run_fixed(cmz::vm2::compile_program(program), 4);
    const auto actual = predicate_value(result.final_state, 0);
    if (actual != (left == right ? 1 : 0)) {
        throw std::runtime_error(
            "CMP_EQ attention result mismatch for " + std::to_string(left) + "," +
            std::to_string(right) + ": " + std::to_string(actual) + ", regs=" +
            std::to_string(register_value(result.state_trace[2], 0)) + "," +
            std::to_string(register_value(result.state_trace[2], 1)));
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
        const auto result = cmz::vm2::run_fixed(image, 8);
        if (result.steps != 8) {
            throw std::runtime_error("runtime must execute exactly eight fixed inference steps");
        }
        if (register_value(result.final_state, 2) != 5) {
            throw std::runtime_error("ADD must produce r2=5");
        }
        if (result.final_state[cmz::vm2::kControlToken][cmz::vm2::kHaltOffset].item<double>() != 1.0) {
            throw std::runtime_error("HALT must set the halt token");
        }
        const auto actual_pc_trace = pc_trace(result.state_trace);
        if (actual_pc_trace != std::vector<std::int64_t>({0, 1, 2, 3, 3, 3, 3, 3, 3})) {
            std::string trace;
            for (const auto pc : actual_pc_trace) {
                trace += std::to_string(pc) + ",";
            }
            throw std::runtime_error("PC trace must remain at HALT for the full fixed unroll: " + trace);
        }
        for (std::int64_t step = 5; step < result.state_trace.size(0); ++step) {
            if (!torch::equal(result.state_trace[4], result.state_trace[step])) {
                throw std::runtime_error("HALT state must be absorbing after step four");
            }
        }
        const auto repeated = cmz::vm2::run_fixed(image, 8);
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
        for (std::int64_t left = 0; left < cmz::vm2::kValueCount; ++left) {
            for (std::int64_t right = 0; right < cmz::vm2::kValueCount; ++right) {
                require_cmp_eq(left, right);
            }
        }
        for (std::int64_t truth = 0; truth < 2; ++truth) {
            for (std::int64_t target = 0; target < cmz::vm2::kProgramSlots; ++target) {
                const std::array branch_program{
                    Instruction{Opcode::LoadConst, 0, 0, 0},
                    Instruction{Opcode::LoadConst, 1, truth == 1 ? 0 : 1, 0},
                    Instruction{Opcode::CmpEq, 0, 0, 1},
                    Instruction{Opcode::JumpIf, 0, 0, target},
                    Instruction{Opcode::Halt, 0, 0, 0},
                };
                const auto branch = cmz::vm2::run_fixed(cmz::vm2::compile_program(branch_program), 4);
                const auto actual_pc = one_hot_index(
                    branch.final_state[cmz::vm2::kControlToken],
                    cmz::vm2::kProgramCounterOffset,
                    cmz::vm2::kProgramSlots);
                const auto expected_pc = truth == 1 ? target : 4;
                if (actual_pc != expected_pc) {
                    throw std::runtime_error("JUMP_IF attention result mismatch");
                }
            }
        }
        const std::array loop_program{
            Instruction{Opcode::LoadConst, 0, 0, 0},
            Instruction{Opcode::LoadConst, 1, 3, 0},
            Instruction{Opcode::LoadConst, 2, 1, 0},
            Instruction{Opcode::Add, 0, 0, 2},
            Instruction{Opcode::CmpEq, 0, 0, 1},
            Instruction{Opcode::JumpIf, 0, 0, 7},
            Instruction{Opcode::JumpIf, 0, cmz::vm2::kPredicateCount, 3},
            Instruction{Opcode::Halt, 0, 0, 0},
        };
        const auto loop = cmz::vm2::run_fixed(cmz::vm2::compile_program(loop_program), 20);
        if (register_value(loop.final_state, 0) != 3 || predicate_value(loop.final_state, 0) != 1) {
            throw std::runtime_error("predicate-token loop must finish with r0=3 and p0=TRUE");
        }
        for (std::int64_t step = 16; step < loop.state_trace.size(0); ++step) {
            if (!torch::equal(loop.state_trace[15], loop.state_trace[step])) {
                throw std::runtime_error("loop HALT state must remain absorbing");
            }
        }
        const auto prefix = cmz::vm2::run_fixed(image, 3);
        if (prefix.steps != 3 || prefix.state_trace.size(0) != 4) {
            throw std::runtime_error("short fixed unroll must return exactly its requested prefix");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
