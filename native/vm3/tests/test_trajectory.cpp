#include <cmz_vm3/trajectory.h>

#include <torch/torch.h>

#include <iostream>
#include <vector>

namespace {
bool require(bool condition, const char* message) {
  if (!condition) std::cerr << message << '\n';
  return condition;
}
}

int main(int argc, char**) {
  bool ok = true;
  const bool require_cuda = argc > 1;
  TORCH_CHECK(!require_cuda || torch::cuda::is_available(),
              "CUDA trajectory gate requested without an available GPU");
  const auto device = torch::Device(require_cuda ? torch::kCUDA : torch::kCPU);
  const auto options = torch::TensorOptions().dtype(torch::kFloat64).device(device);

  cmz::vm3::FrozenChessProgram program;
  auto pad = torch::tensor({-1.0, -1.0, -1.0, -1.0}, options);
  program.tensors.emplace("trajectory_pad_token", pad);

  auto emitted0 = torch::tensor({1.0, 2.0, 3.0, 4.0}, options.requires_grad(true));
  auto emitted1 = torch::tensor({5.0, 6.0, 7.0, 8.0}, options.requires_grad(true));
  auto commit = torch::ones({}, options);
  std::vector<cmz::vm3::MoveSlot> tape;
  tape.push_back(cmz::vm3::make_training_move_slot(program, emitted0, commit));
  const auto first_pointer = tape.front().move.data_ptr();
  const auto first_snapshot = tape.front().move.clone();
  tape.push_back(cmz::vm3::make_training_move_slot(program, emitted1, commit));

  ok &= require(tape.size() == 2, "one structural slot per ply");
  ok &= require(torch::equal(tape[0].move, emitted0) &&
                    torch::equal(tape[1].move, emitted1),
                "committed slots contain exact MOVE tokens");
  ok &= require(tape.front().move.data_ptr() == first_pointer &&
                    torch::equal(tape.front().move, first_snapshot),
                "appending must not mutate or replace prior slots");
  ok &= require(tape[0].move.data_ptr() != tape[1].move.data_ptr(),
                "each ply owns a distinct O(d_model) tensor");
  (tape[0].move.sum() + 2.0 * tape[1].move.sum()).backward();
  ok &= require(torch::equal(emitted0.grad(), torch::ones_like(emitted0)),
                "gradient reaches first active MOVE token");
  ok &= require(torch::equal(emitted1.grad(), 2.0 * torch::ones_like(emitted1)),
                "gradient reaches second active MOVE token");

  auto inactive_move = torch::tensor({9.0, 8.0, 7.0, 6.0}, options.requires_grad(true));
  auto inactive = cmz::vm3::make_training_move_slot(
      program, inactive_move, torch::zeros({}, options));
  ok &= require(torch::equal(inactive.move, pad),
                "claim/terminal structural slots contain PAD");
  ok &= require(inactive.active.item<double>() == 0.0,
                "inactive slot predicate remains zero");
  inactive.move.sum().backward();
  ok &= require(torch::equal(inactive_move.grad(), torch::zeros_like(inactive_move)),
                "inactive padding does not leak move gradient");

  auto differentiable_commit = torch::tensor(0.25, options.requires_grad(true));
  auto mixed_move = torch::tensor({2.0, 4.0, 6.0, 8.0}, options.requires_grad(true));
  auto mixed = cmz::vm3::make_training_move_slot(
      program, mixed_move, differentiable_commit);
  const auto expected = differentiable_commit * mixed_move +
                        (1.0 - differentiable_commit) * pad;
  ok &= require(torch::equal(mixed.move, expected),
                "slot is the declared functional tensor formula");
  mixed.move.sum().backward();
  ok &= require(differentiable_commit.grad().abs().item<double>() > 0.0,
                "commit predicate retains gradient");

  return ok ? 0 : 1;
}
