#pragma once

#include <torch/torch.h>

namespace cmz::vm2 {

struct KnightGeometryCircuit {
    torch::Tensor tokens;
    torch::Tensor wq;
    torch::Tensor wk;
    torch::Tensor wv;
    torch::Tensor class_keys;
};

KnightGeometryCircuit compile_knight_geometry();
torch::Tensor evaluate_knight_geometry(const KnightGeometryCircuit& circuit);

}  // namespace cmz::vm2
