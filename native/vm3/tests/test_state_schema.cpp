#include <cmz_vm3/chess_state.h>

#include <torch/torch.h>

#include <iostream>

namespace {
bool require(bool condition, const char* message) {
  if (!condition) std::cerr << message << '\n';
  return condition;
}

cmz::vm3::InitialPosition base_position() {
  cmz::vm3::InitialPosition input{};
  input.board.fill(cmz::vm3::PieceState::Empty);
  input.side = cmz::vm3::Side::White;
  input.castling = 0;
  input.ep_square = -1;
  input.halfmove = 0;
  input.fullmove = 1;
  input.claim_mode = cmz::vm3::ClaimMode::LiteralClaim;
  return input;
}

bool is_one_hot(const torch::Tensor& tensor, std::int64_t index) {
  return tensor.sum().item<double>() == 1.0 &&
         tensor.index({index}).item<double>() == 1.0;
}
}

int main(int argc, char**) {
  bool ok = true;
  const bool require_cuda = argc > 1;
  TORCH_CHECK(!require_cuda || torch::cuda::is_available(),
              "CUDA state ABI gate requested without an available GPU");
  const auto device = torch::Device(require_cuda ? torch::kCUDA : torch::kCPU);
  const auto options = torch::TensorOptions().dtype(torch::kFloat64).device(device);
  cmz::vm3::FrozenChessProgram program;
  auto state_template = torch::zeros({cmz::vm3::kStateCoreRows + 7, 4}, options);
  state_template.index_put_({torch::indexing::Slice(), 1}, 0.25);
  program.tensors.emplace("state_template", state_template);
  program.tensors.emplace("state_layout", cmz::vm3::canonical_state_layout(options));

  auto input = base_position();
  for (std::int64_t square = 0; square < 64; ++square) {
    input.board[static_cast<std::size_t>(square)] =
        static_cast<cmz::vm3::PieceState>(square % 13);
  }
  auto state = cmz::vm3::bind_initial_state(program, input);
  const auto board = cmz::vm3::board_state_view(state.tokens);
  ok &= require(board.sizes() == torch::IntArrayRef({64, 13}), "board view shape");
  for (std::int64_t square = 0; square < 64; ++square) {
    ok &= require(is_one_hot(board.index({square}), square % 13),
                  "all 13 square states must bind exactly");
  }
  ok &= require(state.tokens.sizes() == state_template.sizes(),
                "state output ABI must equal input template ABI");
  ok &= require(torch::equal(
                    state.tokens.index({torch::indexing::Slice(), 1}),
                    state_template.index({torch::indexing::Slice(), 1})),
                "identity features and workspace must be preserved");

  for (std::int64_t rights = 0; rights < 16; ++rights) {
    input.castling = static_cast<std::uint8_t>(rights);
    auto bound = cmz::vm3::bind_initial_state(program, input);
    ok &= require(is_one_hot(cmz::vm3::castling_view(bound.tokens), rights),
                  "all 16 castling states");
  }
  input.castling = 0;

  for (std::int64_t ep = -1; ep < 64; ++ep) {
    input.ep_square = ep;
    auto bound = cmz::vm3::bind_initial_state(program, input);
    ok &= require(is_one_hot(cmz::vm3::raw_ep_view(bound.tokens), ep + 1),
                  "NONE plus every raw EP square");
  }
  input.ep_square = -1;

  for (auto side : {cmz::vm3::Side::White, cmz::vm3::Side::Black}) {
    input.side = side;
    auto bound = cmz::vm3::bind_initial_state(program, input);
    ok &= require(is_one_hot(cmz::vm3::side_view(bound.tokens),
                          static_cast<std::int64_t>(side)), "side one-hot");
  }
  for (auto mode : {cmz::vm3::ClaimMode::LiteralClaim,
                    cmz::vm3::ClaimMode::AutoClaimSelfPlay}) {
    input.claim_mode = mode;
    auto bound = cmz::vm3::bind_initial_state(program, input);
    ok &= require(is_one_hot(cmz::vm3::claim_mode_view(bound.tokens),
                          static_cast<std::int64_t>(mode)), "claim mode one-hot");
  }

  input = base_position();
  input.halfmove = 150;
  input.fullmove = 9526;
  state = cmz::vm3::bind_initial_state(program, input);
  ok &= require(is_one_hot(cmz::vm3::halfmove_view(state.tokens), 150),
                "automatic halfmove boundary");
  const auto fullmove = cmz::vm3::fullmove_digits_view(state.tokens);
  ok &= require(is_one_hot(fullmove.index({0}), 9526 % 128) &&
                    is_one_hot(fullmove.index({1}), 9526 / 128),
                "two base-128 fullmove digits");
  const auto cursor_digits = cmz::vm3::cursor_digits_view(state.tokens);
  ok &= require(is_one_hot(cursor_digits.index({0}), 0) &&
                    is_one_hot(cursor_digits.index({1}), 0) &&
                    is_one_hot(cursor_digits.index({2}), 0),
                "zero three-digit trajectory cursor");
  ok &= require(cmz::vm3::claim_availability_view(state.tokens).sum().item<double>() == 0.0,
                "no initial claim availability");
  ok &= require(is_one_hot(cmz::vm3::terminal_view(state.tokens), 0),
                "initial terminal kind is Running");
  ok &= require(is_one_hot(cmz::vm3::result_view(state.tokens), 0),
                "initial result is Running");
  ok &= require(state.cursor.dim() == 0 && state.cursor.item<double>() == 0.0,
                "device cursor scalar starts at zero");
  ok &= require(state.trajectory.move_slots.empty() &&
                    state.trajectory.policy_kv_slots.empty(),
                "initial logical trajectory is empty");

  const auto must_fail = [&](cmz::vm3::InitialPosition bad) {
    try {
      (void)cmz::vm3::bind_initial_state(program, bad);
      return false;
    } catch (const std::exception&) {
      return true;
    }
  };
  auto bad = base_position(); bad.castling = 16;
  ok &= require(must_fail(bad), "invalid castling fails closed");
  bad = base_position(); bad.ep_square = 64;
  ok &= require(must_fail(bad), "invalid EP fails closed");
  bad = base_position(); bad.halfmove = 151;
  ok &= require(must_fail(bad), "invalid halfmove fails closed");
  bad = base_position(); bad.fullmove = 0;
  ok &= require(must_fail(bad), "invalid fullmove zero fails closed");
  bad = base_position(); bad.fullmove = 9527;
  ok &= require(must_fail(bad), "invalid fullmove maximum fails closed");
  auto forged_program = program;
  forged_program.tensors["state_layout"] =
      program.tensors.at("state_layout").clone();
  forged_program.tensors["state_layout"].index_put_({0, 0}, 1);
  try {
    (void)cmz::vm3::bind_initial_state(forged_program, base_position());
    ok &= require(false, "forged state layout must fail closed");
  } catch (const std::exception&) {
  }

  return ok ? 0 : 1;
}
