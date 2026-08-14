#include <cmz_vm3/program_image.h>

#include <torch/torch.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::array<char, 8> kMagic{'C', 'M', 'Z', 'V', 'M', '3', '\0', '\0'};

template <typename T>
void write_value(std::ofstream& out, const T& value) {
  out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

void write_string(std::ofstream& out, const std::string& value) {
  const auto size = static_cast<std::uint32_t>(value.size());
  write_value(out, size);
  out.write(value.data(), static_cast<std::streamsize>(value.size()));
}

constexpr std::array<std::uint8_t, 32> kDoubleTwoSha256{
    0x3f, 0x71, 0x0a, 0xc0, 0x88, 0xdb, 0x33, 0x36,
    0x30, 0x87, 0xde, 0x2b, 0x9a, 0x65, 0x75, 0x41,
    0xfe, 0x54, 0x47, 0x82, 0x1d, 0xeb, 0xaa, 0x9f,
    0xe5, 0xcb, 0xd5, 0x38, 0xeb, 0x1a, 0x5f, 0x29};

struct ArtifactOptions {
  double certificate_enabled_lower = 2.0;
  double certificate_disabled_upper = 1.0;
  double proof_claimed_lower = 0.0;
  double proof_claimed_upper = 3.0;
  std::int64_t sentinel_row = 4272;
  bool corrupt_payload_after_hash = false;
  bool cyclic_proof = false;
};

std::filesystem::path write_artifact(const std::string& name,
                                     const ArtifactOptions& options = {}) {
  const auto path = std::filesystem::temp_directory_path() / name;
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    throw std::runtime_error("cannot create test artifact");
  }

  out.write(kMagic.data(), static_cast<std::streamsize>(kMagic.size()));
  write_value(out, std::uint32_t{1});  // schema
  write_value(out, std::uint32_t{2});  // tensors
  write_value(out, std::uint32_t{3});  // proof nodes
  write_value(out, std::uint32_t{1});  // selector certificates
  write_value(out, std::uint32_t{1});  // pre-policy stages
  write_value(out, std::uint32_t{0});  // post-policy stages

  write_string(out, "margin");
  write_value(out, std::uint32_t{1});  // serialized float64
  write_value(out, std::uint32_t{0});  // scalar rank
  std::vector<std::uint8_t> payload(sizeof(double));
  const double margin = 2.0;
  std::memcpy(payload.data(), &margin, sizeof(double));
  write_value(out, static_cast<std::uint64_t>(payload.size()));
  out.write(reinterpret_cast<const char*>(kDoubleTwoSha256.data()),
            kDoubleTwoSha256.size());
  if (options.corrupt_payload_after_hash) {
    payload.front() ^= 0x01;
  }
  out.write(reinterpret_cast<const char*>(payload.data()), payload.size());

  write_string(out, "scale");
  write_value(out, std::uint32_t{1});  // serialized float64
  write_value(out, std::uint32_t{2});  // matrix rank
  write_value(out, std::int64_t{1});
  write_value(out, std::int64_t{1});
  write_value(out, static_cast<std::uint64_t>(payload.size()));
  out.write(reinterpret_cast<const char*>(kDoubleTwoSha256.data()),
            kDoubleTwoSha256.size());
  out.write(reinterpret_cast<const char*>(payload.data()), payload.size());

  // Node 0: BoundedPolicyProbability -> [0,1].
  write_value(out, std::uint32_t{1});
  write_value(out, std::uint32_t{options.cyclic_proof ? 1u : 0u});
  if (options.cyclic_proof) {
    write_value(out, std::uint32_t{0});
  }
  write_string(out, "");
  write_value(out, 0.0);
  write_value(out, 1.0);

  // Node 1: frozen [2] matmul maps [0,1] to [0,2].
  write_value(out, std::uint32_t{2});
  write_value(out, std::uint32_t{1});
  write_value(out, std::uint32_t{0});
  write_string(out, "scale");
  write_value(out, 0.0);
  write_value(out, 2.0);

  // Node 2: base + margin * eligibility. Disabled=[0,1], enabled=[2,3].
  write_value(out, std::uint32_t{5});
  write_value(out, std::uint32_t{1});
  write_value(out, std::uint32_t{0});
  write_string(out, "margin");
  write_value(out, options.proof_claimed_lower);
  write_value(out, options.proof_claimed_upper);

  write_string(out, "candidate_policy");
  write_value(out, std::uint32_t{2});
  write_value(out, options.certificate_enabled_lower);
  write_value(out, options.certificate_disabled_upper);
  write_value(out, options.sentinel_row);

  // One ordered stage and one block, all referencing verified payloads.
  for (int i = 0; i < 6; ++i) write_string(out, "scale");
  write_value(out, std::uint32_t{1});
  for (int i = 0; i < 6; ++i) write_string(out, "scale");
  write_value(out, std::uint32_t{2});
  write_value(out, std::int64_t{0});
  write_value(out, std::int64_t{1});

  out.close();
  return path;
}

bool expect_failure(const std::filesystem::path& path,
                    const std::string& expected_message) {
  try {
    (void)cmz::vm3::load_frozen_program(path, torch::kCPU);
  } catch (const std::exception& error) {
    if (std::string(error.what()).find(expected_message) != std::string::npos) {
      return true;
    }
    std::cerr << "wrong failure: " << error.what() << '\n';
    return false;
  }
  std::cerr << "artifact unexpectedly loaded: " << path << '\n';
  return false;
}

bool require(bool condition, const char* message) {
  if (!condition) {
    std::cerr << message << '\n';
  }
  return condition;
}

}  // namespace

int main() {
  bool ok = true;

  const auto valid_path = write_artifact("cmz-vm3-valid-program.bin");
  const auto program = cmz::vm3::load_frozen_program(valid_path, torch::kCPU);
  ok &= require(program.schema_version == 1, "schema version");
  ok &= require(program.manifest.size() == 2, "manifest size");
  ok &= require(program.tensors.size() == 2, "tensor count");
  ok &= require(program.interval_proof.size() == 3, "proof count");
  ok &= require(program.selector_certificates.size() == 1, "certificate count");
  ok &= require(program.pre_policy_stages.size() == 1, "pre stage order");
  ok &= require(program.pre_policy_stages.front().attention_plan.blocks.size() == 1,
                "attention block order");
  ok &= require(program.pre_policy_stages.front().attention_plan.query_group_offsets ==
                    std::vector<std::int64_t>({0, 1}),
                "query group offsets");
  ok &= require(program.post_policy_stages.empty(), "post stages empty");
  const auto margin = program.tensors.at("margin");
  ok &= require(margin.device().is_cpu(), "requested device");
  ok &= require(margin.scalar_type() == torch::kFloat64, "tensor dtype");
  ok &= require(margin.dim() == 0, "scalar tensor rank");
  ok &= require(margin.item<double>() == 2.0, "tensor payload");
  ok &= require(!margin.requires_grad(), "frozen tensor");

  ArtifactOptions forged;
  forged.certificate_enabled_lower = 1.5;
  ok &= expect_failure(write_artifact("cmz-vm3-forged-bound.bin", forged),
                       "selector certificate");

  ArtifactOptions overlap;
  overlap.certificate_enabled_lower = 1.0;
  ok &= expect_failure(write_artifact("cmz-vm3-overlap.bin", overlap),
                       "enabled score must dominate");

  ArtifactOptions bad_claim;
  bad_claim.proof_claimed_upper = 2.5;
  ok &= expect_failure(write_artifact("cmz-vm3-bad-proof-claim.bin", bad_claim),
                       "interval proof claim");

  ArtifactOptions corrupt;
  corrupt.corrupt_payload_after_hash = true;
  ok &= expect_failure(write_artifact("cmz-vm3-corrupt-payload.bin", corrupt),
                       "SHA-256");

  ArtifactOptions cycle;
  cycle.cyclic_proof = true;
  ok &= expect_failure(write_artifact("cmz-vm3-cycle.bin", cycle),
                       "interval proof cycle");

  ArtifactOptions bad_sentinel;
  bad_sentinel.sentinel_row = -1;
  ok &= expect_failure(write_artifact("cmz-vm3-bad-sentinel.bin", bad_sentinel),
                       "sentinel row");

  return ok ? 0 : 1;
}
