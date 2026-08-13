#include "cmz_vm2/trace_exporter.h"

#include <ATen/ops/one_hot.h>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace cmz::vm2 {
namespace {

torch::Tensor apply_writes(
    const ProgramImage& image,
    torch::Tensor state,
    const torch::Tensor& projected,
    std::int64_t stage) {
    for (std::int64_t component = 0; component < kWriteProjectionComponents; ++component) {
        const auto delta = projected - state;
        state = state + torch::matmul(
                            torch::matmul(image.write_projections.row[stage][component], delta),
                            image.write_projections.feature[stage][component]);
    }
    return state;
}

void write_number(std::ostream& output, double value) {
    if (std::isinf(value) && value < 0) {
        output << "\"-inf\"";
    } else {
        output << std::setprecision(17) << value;
    }
}

void write_matrix(
    std::ostream& output,
    const char* name,
    const torch::Tensor& source,
    bool negative_infinity_default = false) {
    const auto tensor = source.to(torch::kCPU).to(torch::kFloat64).contiguous();
    output << "{\"name\":\"" << name << "\",\"rows\":" << tensor.size(0)
           << ",\"cols\":" << tensor.size(1) << ",\"storage\":\"coo\",\"default\":";
    if (negative_infinity_default) {
        output << "\"-inf\"";
    } else {
        output << '0';
    }
    output << ",\"entries\":[";
    bool first = true;
    const auto accessor = tensor.accessor<double, 2>();
    for (std::int64_t row = 0; row < tensor.size(0); ++row) {
        for (std::int64_t col = 0; col < tensor.size(1); ++col) {
            const auto value = accessor[row][col];
            const auto is_default = negative_infinity_default
                                        ? (std::isinf(value) && value < 0)
                                        : value == 0.0;
            if (is_default) {
                continue;
            }
            output << (first ? "" : ",") << '[' << row << ',' << col << ',';
            write_number(output, value);
            output << ']';
            first = false;
        }
    }
    output << "]}";
}

void write_matrix_field(
    std::ostream& output,
    const char* field,
    const char* name,
    const torch::Tensor& tensor,
    bool negative_infinity_default = false) {
    output << '\"' << field << "\":";
    write_matrix(output, name, tensor, negative_infinity_default);
}

}  // namespace

torch::Tensor StageSnapshot::apply_write_reference(
    const ProgramImage& image, std::int64_t stage) const {
    return apply_writes(image, x, y, stage);
}

StageSnapshot trace_stage(
    const ProgramImage& image, const torch::Tensor& state, std::int64_t stage) {
    const auto q = torch::matmul(state, image.weights.wq[stage]);
    const auto k = torch::matmul(state, image.weights.wk[stage]);
    const auto v = torch::matmul(state, image.weights.wv[stage]);
    const auto scores = torch::matmul(q, k.t()) + image.attention_masks[stage];
    const auto winners = std::get<1>(scores.max(1, false));
    const auto attention = at::one_hot(winners, scores.size(1)).to(state.scalar_type());
    const auto output = torch::matmul(attention, v);
    const auto y = torch::matmul(output, image.weights.wo[stage]);
    const auto x_prime = apply_writes(image, state, y, stage);
    return StageSnapshot{state, q, k, v, scores, winners, attention, y, x_prime};
}

TransitionTrace trace_transition(
    const ProgramImage& image, const torch::Tensor& initial_state) {
    auto state = initial_state;
    std::vector<StageSnapshot> stages;
    stages.reserve(kStageCount);
    for (std::int64_t stage = 0; stage < kStageCount; ++stage) {
        auto snapshot = trace_stage(image, state, stage);
        state = snapshot.x_prime;
        stages.push_back(std::move(snapshot));
    }
    return TransitionTrace{std::move(stages), state};
}

void write_trace_json(
    const std::string& path,
    const ProgramImage& image,
    const torch::Tensor& initial_state,
    const std::string& commit) {
    if (commit.size() != 40) {
        throw std::invalid_argument("trace provenance requires a full 40-character commit");
    }
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("cannot open trace output path");
    }
    const auto trace = trace_transition(image, initial_state);
    output << "{\"schema\":\"cmz.matrix-trace.v1\",\"provenance\":{\"commit\":\""
           << commit << "\",\"generator\":\"cmz_vm2_export_chess1_trace\",\"dtype\":\"float64\"},"
           << "\"claim\":{\"scope\":\"pawn-rule-slice\",\"fullChess\":false},\"tokenLabels\":[";
    for (std::int64_t token = 0; token < initial_state.size(0); ++token) {
        output << (token == 0 ? "" : ",") << "\"TOKEN:" << token << "\"";
    }
    output << "],\"stages\":[";
    for (std::int64_t stage = 0; stage < kStageCount; ++stage) {
        const auto& snapshot = trace.stages[stage];
        output << (stage == 0 ? "" : ",") << "{\"index\":" << stage
               << ",\"name\":\"stage-" << stage << "\",\"matrices\":{";
        write_matrix_field(output, "X", "X", snapshot.x); output << ',';
        write_matrix_field(output, "Wq", "Wq", image.weights.wq[stage]); output << ',';
        write_matrix_field(output, "Wk", "Wk", image.weights.wk[stage]); output << ',';
        write_matrix_field(output, "Wv", "Wv", image.weights.wv[stage]); output << ',';
        write_matrix_field(output, "Q", "Q", snapshot.q); output << ',';
        write_matrix_field(output, "K", "K", snapshot.k); output << ',';
        write_matrix_field(output, "V", "V", snapshot.v); output << ',';
        write_matrix_field(output, "M", "M", image.attention_masks[stage], true); output << ',';
        write_matrix_field(output, "S", "S", snapshot.scores, true); output << ',';
        write_matrix_field(output, "A", "A", snapshot.attention); output << ',';
        write_matrix_field(output, "Y", "Y", snapshot.y); output << ',';
        write_matrix_field(output, "XPrime", "XPrime", snapshot.x_prime);
        output << "},\"writes\":[";
        auto before = snapshot.x;
        for (std::int64_t component = 0; component < kWriteProjectionComponents; ++component) {
            const auto delta = snapshot.y - before;
            const auto after = before + torch::matmul(
                torch::matmul(image.write_projections.row[stage][component], delta),
                image.write_projections.feature[stage][component]);
            output << (component == 0 ? "" : ",") << "{\"component\":" << component << ',';
            write_matrix_field(output, "R", "R", image.write_projections.row[stage][component]); output << ',';
            write_matrix_field(output, "C", "C", image.write_projections.feature[stage][component]); output << ',';
            write_matrix_field(output, "before", "before", before); output << ',';
            write_matrix_field(output, "after", "after", after); output << '}';
            before = after;
        }
        output << "],\"winners\":[";
        const auto winners = snapshot.winners.to(torch::kCPU).contiguous();
        for (std::int64_t row = 0; row < winners.size(0); ++row) {
            output << (row == 0 ? "" : ",") << winners[row].item<std::int64_t>();
        }
        output << "],\"changed\":[";
        const auto before_cpu = snapshot.x.to(torch::kCPU).to(torch::kFloat64).contiguous();
        const auto after_cpu = snapshot.x_prime.to(torch::kCPU).to(torch::kFloat64).contiguous();
        const auto before_values = before_cpu.accessor<double, 2>();
        const auto after_values = after_cpu.accessor<double, 2>();
        bool first_change = true;
        for (std::int64_t row = 0; row < before_cpu.size(0); ++row) {
            for (std::int64_t col = 0; col < before_cpu.size(1); ++col) {
                if (before_values[row][col] == after_values[row][col]) continue;
                output << (first_change ? "" : ",") << "{\"row\":" << row << ",\"col\":" << col << ",\"before\":";
                write_number(output, before_values[row][col]);
                output << ",\"after\":";
                write_number(output, after_values[row][col]);
                output << '}';
                first_change = false;
            }
        }
        output << "]}";
    }
    output << "]}";
}

}  // namespace cmz::vm2
