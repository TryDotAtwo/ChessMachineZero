#include <cmz_vm3/chess_state.h>
#include <cmz_vm3/policy.h>
#include <cmz_vm3/recurrent_ring.h>
#include <cmz_vm3/rule_compiler.h>

#include <torch/torch.h>

#include <iostream>
#include <filesystem>
#include <set>

namespace {
bool require(bool condition, const char* message) {
  if (!condition) std::cerr << message << '\n';
  return condition;
}

std::int64_t ordinary(std::int64_t source, std::int64_t target) {
  return source * 64 + target;
}

cmz::vm3::InitialPosition minimal_position() {
  cmz::vm3::InitialPosition position{};
  position.board.fill(cmz::vm3::PieceState::Empty);
  position.board[1] = cmz::vm3::PieceState::WN;   // b1
  position.board[8] = cmz::vm3::PieceState::WP;   // a2
  position.board[55] = cmz::vm3::PieceState::BP;  // h7
  position.board[62] = cmz::vm3::PieceState::BN;  // g8
  position.side = cmz::vm3::Side::White;
  position.castling = 0;
  position.ep_square = -1;
  position.halfmove = 0;
  position.fullmove = 1;
  position.claim_mode = cmz::vm3::ClaimMode::LiteralClaim;
  return position;
}

std::set<std::int64_t> active_ids(const torch::Tensor& legal) {
  std::set<std::int64_t> result;
  const auto cpu = legal.to(torch::kCPU).squeeze(0);
  for (std::int64_t id = 0; id < cpu.size(0); ++id) {
    if (cpu.index({id}).item<double>() == 1.0) result.insert(id);
  }
  return result;
}
}  // namespace

int main(int argc, char**) {
  bool ok = true;
  const bool use_cuda = argc > 1;
  TORCH_CHECK(!use_cuda || torch::cuda::is_available(),
              "CUDA minimal-ring gate requested without GPU");
  const auto device = torch::Device(use_cuda ? torch::kCUDA : torch::kCPU);
  const auto options = torch::TensorOptions().dtype(torch::kFloat64).device(device);

  const auto compiled = cmz::vm3::compile_minimal_rule_program(options);
  const auto image_path = std::filesystem::temp_directory_path() /
                          "cmz_vm3_minimal_ring_image.bin";
  cmz::vm3::save_frozen_program(compiled, image_path);
  const auto program = cmz::vm3::load_frozen_program(image_path, device);
  std::filesystem::remove(image_path);
  ok &= require(!program.manifest.empty() && program.pre_policy_stages.size() == 34,
                "compiler emits one hashed immutable 34-stage rule image");
  if (!use_cuda) {
    for (const auto& entry : program.manifest) {
      ok &= require(cmz::vm3::tensor_sha256(program.tensors.at(entry.name)) ==
                        entry.sha256,
                    "every immutable tensor matches its manifest SHA-256");
    }
  }
  for (const auto& stage : program.pre_policy_stages) {
    ok &= require(stage.wq.size(-1) == 2 && stage.wk.size(-1) == 2 &&
                      !stage.wq.requires_grad() && !stage.wk.requires_grad() &&
                      !stage.wv.requires_grad(),
                  "every frozen rule lookup has physical d_head=2");
  }
  auto state = cmz::vm3::bind_initial_state(program, minimal_position());
  cmz::vm3::ZeroPolicy white(cmz::vm3::policy_routing_from_program(program));
  cmz::vm3::ZeroPolicy black(cmz::vm3::policy_routing_from_program(program));
  cmz::vm3::PolicyPair policies{white, black};

  const auto first = cmz::vm3::recurrent_step(program, policies, state);
  const std::set<std::int64_t> expected_white{
      ordinary(1, 11), ordinary(1, 16), ordinary(1, 18),
      ordinary(8, 16), ordinary(8, 24)};
  ok &= require(active_ids(first.legal) == expected_white,
                "one shared legal tensor contains exact white pawn+knight moves");
  ok &= require(first.selected_move.index({0, 0}).item<std::int64_t>() == 1 &&
                    first.selected_move.index({0, 1}).item<std::int64_t>() == 11 &&
                    first.selected_move.index({0, 2}).item<std::int64_t>() == 0,
                "first legal policy selects b1d2");
  const auto first_board = cmz::vm3::board_state_view(first.next.tokens).argmax(-1);
  ok &= require(first_board.index({1}).item<std::int64_t>() ==
                    static_cast<std::int64_t>(cmz::vm3::PieceState::Empty) &&
                    first_board.index({11}).item<std::int64_t>() ==
                    static_cast<std::int64_t>(cmz::vm3::PieceState::WN),
                "selected knight move is written exactly once");
  ok &= require(cmz::vm3::side_view(first.next.tokens).argmax(-1).item<std::int64_t>() ==
                    static_cast<std::int64_t>(cmz::vm3::Side::Black),
                "side flips through the recurrent tensor write");
  ok &= require(first.next.tokens.sizes() == state.tokens.sizes() &&
                    first.next.trajectory.move_slots.size() == 1 &&
                    first.next.trajectory.move_slots[0].active.item<double>() == 1.0,
                "same ABI output appends exactly one active MOVE slot");

  auto gradient_state = cmz::vm3::bind_initial_state(program, minimal_position());
  gradient_state.tokens = gradient_state.tokens.clone().set_requires_grad(true);
  auto gradient_step = cmz::vm3::recurrent_step(program, policies, gradient_state);
  gradient_step.legal.retain_grad();
  const auto state_reward = torch::linspace(
      -1.0, 1.0, gradient_step.next.tokens.numel(), options)
                                .view_as(gradient_step.next.tokens);
  (gradient_step.next.tokens * state_reward).sum().backward();
  ok &= require(gradient_step.legal.grad().defined() &&
                    gradient_step.legal.grad().abs().sum().item<double>() > 0.0 &&
                    gradient_state.tokens.grad().defined() &&
                    gradient_state.tokens.grad().abs().sum().item<double>() > 0.0,
                "board loss backpropagates through policy eligibility and frozen rules");

  const auto second = cmz::vm3::recurrent_step(program, policies, first.next);
  const std::set<std::int64_t> expected_black{
      ordinary(55, 39), ordinary(55, 47),
      ordinary(62, 45), ordinary(62, 47), ordinary(62, 52)};
  ok &= require(active_ids(second.legal) == expected_black,
                "second inference consumes the first output board directly");
  ok &= require(second.selected_move.index({0, 0}).item<std::int64_t>() == 55 &&
                    second.selected_move.index({0, 1}).item<std::int64_t>() == 39,
                "second first-legal policy selects h7h5");
  const auto second_board = cmz::vm3::board_state_view(second.next.tokens).argmax(-1);
  ok &= require(second_board.index({55}).item<std::int64_t>() ==
                    static_cast<std::int64_t>(cmz::vm3::PieceState::Empty) &&
                    second_board.index({39}).item<std::int64_t>() ==
                    static_cast<std::int64_t>(cmz::vm3::PieceState::BP) &&
                    second.next.trajectory.move_slots.size() == 2,
                "second output is recurrent and appends one more MOVE");

  auto capture_position = minimal_position();
  capture_position.board.fill(cmz::vm3::PieceState::Empty);
  capture_position.board[27] = cmz::vm3::PieceState::WP;  // d4
  capture_position.board[34] = cmz::vm3::PieceState::BN;  // c5
  capture_position.board[36] = cmz::vm3::PieceState::BN;  // e5
  const auto capture_state = cmz::vm3::bind_initial_state(program, capture_position);
  const auto captures = cmz::vm3::recurrent_step(program, policies, capture_state);
  const std::set<std::int64_t> expected_captures{
      ordinary(27, 34), ordinary(27, 35), ordinary(27, 36)};
  ok &= require(active_ids(captures.legal) == expected_captures,
                "pawn pushes and enemy diagonal captures share the same rule graph");

  auto occupied_target = minimal_position();
  occupied_target.board.fill(cmz::vm3::PieceState::Empty);
  occupied_target.board[1] = cmz::vm3::PieceState::WN;
  occupied_target.board[11] = cmz::vm3::PieceState::WP;
  const auto occupied_state = cmz::vm3::bind_initial_state(program, occupied_target);
  const auto occupied = cmz::vm3::recurrent_step(program, policies, occupied_state);
  ok &= require(active_ids(occupied.legal).count(ordinary(1, 11)) == 0,
                "knight cannot land on a same-side occupied square");

  auto blocked_double = minimal_position();
  blocked_double.board.fill(cmz::vm3::PieceState::Empty);
  blocked_double.board[8] = cmz::vm3::PieceState::WP;
  blocked_double.board[16] = cmz::vm3::PieceState::WB;
  const auto blocked_state = cmz::vm3::bind_initial_state(program, blocked_double);
  const auto blocked = cmz::vm3::recurrent_step(program, policies, blocked_state);
  ok &= require(active_ids(blocked.legal).count(ordinary(8, 24)) == 0,
                "occupied intermediate square rejects a pawn double push");

  return ok ? 0 : 1;
}
