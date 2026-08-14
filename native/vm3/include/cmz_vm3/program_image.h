#pragma once

#include <torch/torch.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace cmz::vm3 {

struct ProgramTensorManifest {
  std::string name;
  std::vector<std::int64_t> shape;
  c10::ScalarType dtype;
  std::array<std::uint8_t, 32> sha256;
};

enum class IntervalOp : std::uint8_t {
  CanonicalInput = 0,
  BoundedPolicyProbability = 1,
  FrozenMatmul = 2,
  Add = 3,
  Multiply = 4,
  EligibilityMargin = 5,
  ReduceMax = 6,
};

struct IntervalProofNode {
  IntervalOp op;
  std::vector<std::uint32_t> inputs;
  std::string tensor_name;
  double claimed_lower;
  double claimed_upper;
};

struct SelectorScoreCertificate {
  std::string selector_name;
  std::uint32_t proof_root;
  double enabled_score_lower;
  double disabled_score_upper;
  std::int64_t sentinel_row;
};

struct Attention2dBlock {
  torch::Tensor query_router;
  torch::Tensor key_router;
  torch::Tensor output_router;
  torch::Tensor fixed_mask;
  torch::Tensor fixed_eligibility;
};

struct Attention2dPlan { std::vector<Attention2dBlock> blocks; };

struct FrozenStage {
  torch::Tensor wq;
  torch::Tensor wk;
  torch::Tensor wv;
  torch::Tensor fixed_mask;
  torch::Tensor row_router;
  torch::Tensor feature_router;
  Attention2dPlan attention_plan;
};

struct FrozenChessProgram {
  std::uint32_t schema_version{};
  std::vector<ProgramTensorManifest> manifest;
  std::unordered_map<std::string, torch::Tensor> tensors;
  std::vector<IntervalProofNode> interval_proof;
  std::vector<SelectorScoreCertificate> selector_certificates;
  std::vector<FrozenStage> pre_policy_stages;
  std::vector<FrozenStage> post_policy_stages;
};

FrozenChessProgram load_frozen_program(const std::filesystem::path& path,
                                       const torch::Device& device);

}  // namespace cmz::vm3
