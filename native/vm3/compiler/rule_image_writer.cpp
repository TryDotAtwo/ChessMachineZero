#include <cmz_vm3/rule_compiler.h>

#include <array>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>

namespace cmz::vm3 {
namespace {

template <typename T>
void write_value(std::ofstream& output, const T& value) {
  output.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

void write_string(std::ofstream& output, const std::string& value) {
  write_value(output, static_cast<std::uint32_t>(value.size()));
  output.write(value.data(), static_cast<std::streamsize>(value.size()));
}

void write_stage(std::ofstream& output,
                 const std::string& prefix,
                 const FrozenStage& stage) {
  write_string(output, prefix + "wq");
  write_string(output, prefix + "wk");
  write_string(output, prefix + "wv");
  write_string(output, prefix + "fixed_mask");
  write_string(output, prefix + "row_router");
  write_string(output, prefix + "feature_router");
  write_value(output,
              static_cast<std::uint32_t>(stage.attention_plan.blocks.size()));
  for (std::size_t block = 0; block < stage.attention_plan.blocks.size(); ++block) {
    const auto block_prefix = prefix;
    write_string(output, block_prefix + "query_router");
    write_string(output, block_prefix + "key_router");
    write_string(output, block_prefix + "output_router");
    write_string(output, block_prefix + "mask");
    write_string(output, block_prefix + "eligibility");
    write_string(output, block_prefix + "global_key_ids");
  }
  write_value(output, static_cast<std::uint32_t>(
                          stage.attention_plan.query_group_offsets.size()));
  for (const auto offset : stage.attention_plan.query_group_offsets)
    write_value(output, offset);
}

}  // namespace

void save_frozen_program(const FrozenChessProgram& program,
                         const std::filesystem::path& path) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) throw std::runtime_error("cannot create VM3 rule image");
  const std::array<char, 8> magic{'C', 'M', 'Z', 'V', 'M', '3', '\0', '\0'};
  output.write(magic.data(), static_cast<std::streamsize>(magic.size()));
  write_value(output, program.schema_version);
  write_value(output, static_cast<std::uint32_t>(program.manifest.size()));
  write_value(output, static_cast<std::uint32_t>(program.interval_proof.size()));
  write_value(output,
              static_cast<std::uint32_t>(program.selector_certificates.size()));
  write_value(output,
              static_cast<std::uint32_t>(program.pre_policy_stages.size()));
  write_value(output,
              static_cast<std::uint32_t>(program.post_policy_stages.size()));

  for (const auto& entry : program.manifest) {
    const auto tensor = program.tensors.at(entry.name).to(torch::kCPU).contiguous();
    write_string(output, entry.name);
    const auto dtype_code = entry.dtype == torch::kFloat64 ? 1u : 2u;
    write_value(output, dtype_code);
    write_value(output, static_cast<std::uint32_t>(entry.shape.size()));
    for (const auto dimension : entry.shape) write_value(output, dimension);
    write_value(output, static_cast<std::uint64_t>(tensor.nbytes()));
    output.write(reinterpret_cast<const char*>(entry.sha256.data()),
                 static_cast<std::streamsize>(entry.sha256.size()));
    output.write(reinterpret_cast<const char*>(tensor.const_data_ptr()),
                 static_cast<std::streamsize>(tensor.nbytes()));
  }
  for (const auto& node : program.interval_proof) {
    write_value(output, static_cast<std::uint32_t>(node.op));
    write_value(output, static_cast<std::uint32_t>(node.inputs.size()));
    for (const auto input : node.inputs) write_value(output, input);
    write_string(output, node.tensor_name);
    write_value(output, node.claimed_lower);
    write_value(output, node.claimed_upper);
  }
  for (const auto& certificate : program.selector_certificates) {
    write_string(output, certificate.selector_name);
    write_value(output, certificate.proof_root);
    write_value(output, certificate.enabled_score_lower);
    write_value(output, certificate.disabled_score_upper);
    write_value(output, certificate.sentinel_row);
  }
  for (std::size_t index = 0; index < program.pre_policy_stages.size(); ++index)
    write_stage(output, "pre_stage_" + std::to_string(index) + "_",
                program.pre_policy_stages[index]);
  for (std::size_t index = 0; index < program.post_policy_stages.size(); ++index)
    write_stage(output, "post_stage_" + std::to_string(index) + "_",
                program.post_policy_stages[index]);
  if (!output) throw std::runtime_error("failed to write VM3 rule image");
}

}  // namespace cmz::vm3
