#pragma once

#include <cmz_vm3/candidate_bank.h>
#include <cmz_vm3/policy.h>
#include <cmz_vm3/program_image.h>

namespace cmz::vm3 {

using FrozenTensorMap = std::unordered_map<std::string, torch::Tensor>;

torch::Tensor compile_pawn_geometry(const CandidateBank& bank);
torch::Tensor compile_knight_geometry(const CandidateBank& bank);
FrozenTensorMap compile_special_move_tensors(const CandidateBank& bank);
FrozenTensorMap compile_attack_tensors(const CandidateBank& bank);
FrozenChessProgram compile_minimal_rule_program(
    const torch::TensorOptions& options = torch::TensorOptions().dtype(torch::kFloat64));
void save_frozen_program(const FrozenChessProgram& program,
                         const std::filesystem::path& path);

}  // namespace cmz::vm3
