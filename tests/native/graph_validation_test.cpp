#include "cmz/vm_module.h"
#include <iostream>
#include <string>
#include <vector>

namespace {
using cmz::OpCode;
cmz::TensorRecord zeros(const std::vector<std::uint32_t>& shape) {
    std::size_t count = 1;
    for (auto extent : shape) count *= extent;
    return {"fixture", shape, std::vector<std::uint8_t>((count + 1) / 2),
            std::vector<float>((count + 4095) / 4096, 1.0F), 4096};
}

int failures = 0;
void rejects(const char* name, const std::vector<cmz::TensorRecord>& tensors,
             const std::vector<cmz::Operation>& operations, const char* reason) {
    try {
        // An impossible device proves shape validation precedes CUDA initialization,
        // materialization or kernel launch. A CUDA device error is NOT acceptance.
        cmz::FrozenVm vm(cmz::Artifact::from_records(tensors, operations), torch::Device("cuda:127"));
    } catch (const c10::Error& error) {
        if (std::string(error.what()).find(reason) != std::string::npos) {
            std::cout << "PASS " << name << '\n';
            return;
        }
    }
    std::cerr << "FAIL " << name << ": expected pre-device rejection: " << reason << '\n';
    ++failures;
}
}

int main() {
    const std::vector<cmz::Operation> prefix = {
        {OpCode::RowRoute, {0}, {1}, {0, 2}},
        {OpCode::TokenProject, {1}, {2}, {0}}};
    const auto attention = [&](const char* name, std::vector<std::uint32_t> attrs,
                               std::vector<std::uint32_t> inputs, const char* reason) {
        auto ops = prefix;
        ops.push_back({OpCode::HullAttention2D, std::move(inputs), {3}, std::move(attrs)});
        rejects(name, {zeros({128, 2})}, ops, reason);
    };
    attention("dynamic K two rows candidate 2", {1, 500, 2}, {2, 2, 1}, "candidate out of range");
    attention("duplicate candidates", {2, 500, 0, 0}, {2, 2, 1}, "duplicate candidate");
    attention("wrong query width", {1, 500, 0}, {1, 2, 1}, "attention query shape");
    attention("wrong key width", {1, 500, 0}, {2, 1, 1}, "attention key shape");
    attention("wrong value rows", {1, 500, 0}, {2, 2, 0}, "attention value shape");
    rejects("route over input boundary", {}, {{OpCode::RowRoute, {0}, {1}, {0, 2049}}}, "RowRoute bounds");
    rejects("route stride zero", {}, {{OpCode::RowRoute, {0}, {1}, {0, 2, 0}}}, "stride");
    rejects("projection width", {zeros({127, 2})}, {{OpCode::TokenProject, {0}, {1}, {0}}}, "projection shape");
    rejects("position broadcast", {zeros({3, 128})}, {{OpCode::PositionAdd, {0}, {1}, {0}}}, "broadcast shape");
    rejects("batch cannot broadcast to static extent", {zeros({2, 2048, 128})},
        {{OpCode::PositionAdd, {0}, {1}, {0}}}, "broadcast shape");
    rejects("residual broadcast", {}, {{OpCode::RowRoute, {0}, {1}, {0, 2}},
        {OpCode::ResidualAdd, {0, 1}, {2}, {}}}, "broadcast shape");
    rejects("FFN gated width", {zeros({128, 2}), zeros({128, 3}), zeros({2, 128})},
        {{OpCode::GatedFfn, {0}, {1}, {0, 1, 2}}}, "broadcast shape");
    rejects("FFN down width", {zeros({128, 2}), zeros({128, 2}), zeros({3, 128})},
        {{OpCode::GatedFfn, {0}, {1}, {0, 1, 2}}}, "projection shape");
    rejects("hardmax mask shape", {}, {{OpCode::RowRoute, {0}, {1}, {0, 2}},
        {OpCode::HardmaxSte, {0, 1}, {2}, {500}}}, "hardmax mask shape");
    rejects("concat width", {zeros({128, 2})}, {{OpCode::TokenProject, {0}, {1}, {0}},
        {OpCode::RowConcat, {0, 1}, {2}, {}}}, "RowConcat shape");
    rejects("expanded route bound", {zeros({2, 128})}, {{OpCode::FrozenExpand, {0}, {1}, {0}},
        {OpCode::RowRoute, {1}, {2}, {0, 3}}}, "RowRoute bounds");
    rejects("static key value count", {zeros({128, 2}), zeros({2, 2})},
        {{OpCode::TokenProject, {0}, {1}, {0}},
         {OpCode::HullAttention2D, {1, 0}, {2}, {1, 1, 500, 0}}}, "attention value shape");
    rejects("strided shape propagated", {zeros({128, 2})},
        {{OpCode::RowRoute, {0}, {1}, {0, 5, 3}},
         {OpCode::OutputProject, {1}, {2}, {0}},
         {OpCode::HullAttention2D, {2, 2, 1}, {3}, {1, 500, 2}}}, "candidate out of range");
    std::cout << "graph validation failures=" << failures << '\n';
    return failures ? 1 : 0;
}
