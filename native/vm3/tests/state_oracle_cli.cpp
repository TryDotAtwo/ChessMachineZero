#include <cmz_vm3/chess_state.h>

#include <torch/torch.h>

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {
std::int64_t selected(const torch::Tensor& one_hot) {
  return one_hot.argmax().item<std::int64_t>();
}
}

int main(int argc, char** argv) {
  if (argc != 8) return 2;
  cmz::vm3::InitialPosition input{};
  std::stringstream board_stream(argv[1]);
  std::string field;
  std::vector<std::int64_t> pieces;
  while (std::getline(board_stream, field, ',')) pieces.push_back(std::stoll(field));
  if (pieces.size() != 64) return 3;
  for (std::size_t square = 0; square < pieces.size(); ++square) {
    input.board[square] = static_cast<cmz::vm3::PieceState>(pieces[square]);
  }
  input.side = static_cast<cmz::vm3::Side>(std::stoll(argv[2]));
  input.castling = static_cast<std::uint8_t>(std::stoll(argv[3]));
  input.ep_square = std::stoll(argv[4]);
  input.halfmove = std::stoll(argv[5]);
  input.fullmove = std::stoll(argv[6]);
  input.claim_mode = static_cast<cmz::vm3::ClaimMode>(std::stoll(argv[7]));

  cmz::vm3::FrozenChessProgram program;
  program.tensors.emplace(
      "state_template",
      torch::zeros({cmz::vm3::kStateCoreRows + 3, 2}, torch::kFloat64));
  program.tensors.emplace(
      "state_layout",
      cmz::vm3::canonical_state_layout(torch::TensorOptions().device(torch::kCPU)));
  const auto state = cmz::vm3::bind_initial_state(program, input);
  const auto board = cmz::vm3::board_state_view(state.tokens);
  for (std::int64_t square = 0; square < 64; ++square) {
    if (square) std::cout << ',';
    std::cout << selected(board.index({square}));
  }
  const auto fullmove = cmz::vm3::fullmove_digits_view(state.tokens);
  std::cout << '\n' << selected(cmz::vm3::side_view(state.tokens))
            << ',' << selected(cmz::vm3::castling_view(state.tokens))
            << ',' << selected(cmz::vm3::raw_ep_view(state.tokens))
            << ',' << selected(cmz::vm3::halfmove_view(state.tokens))
            << ',' << selected(fullmove.index({0}))
            << ',' << selected(fullmove.index({1}))
            << ',' << selected(cmz::vm3::claim_mode_view(state.tokens))
            << ',' << selected(cmz::vm3::terminal_view(state.tokens))
            << ',' << selected(cmz::vm3::result_view(state.tokens)) << '\n';
  return 0;
}
