#include <cmz_vm3/chess_state.h>
#include <cmz_vm3/recurrent_ring.h>
#include <cmz_vm3/rule_compiler.h>

#include <torch/torch.h>

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {
char promotion_letter(cmz::vm3::Promotion promotion) {
  constexpr char letters[] = {'\0', 'q', 'r', 'b', 'n'};
  return letters[static_cast<std::int64_t>(promotion)];
}

std::string square_name(std::int64_t square) {
  std::string result;
  result.push_back(static_cast<char>('a' + square % 8));
  result.push_back(static_cast<char>('1' + square / 8));
  return result;
}
}  // namespace

int main(int argc, char**) {
  const auto use_cuda = argc > 1;
  TORCH_CHECK(!use_cuda || torch::cuda::is_available(),
              "CUDA pseudo-legal oracle requested without GPU");
  const auto options = torch::TensorOptions().dtype(torch::kFloat64).device(
      use_cuda ? torch::kCUDA : torch::kCPU);
  const auto program = cmz::vm3::compile_minimal_rule_program(options);
  const auto bank = cmz::vm3::compile_candidate_bank();
  std::vector<torch::Tensor> states;
  std::string line;
  while (std::getline(std::cin, line)) {
    std::stringstream record(line);
    std::string board_field;
    std::string side_field;
    if (!std::getline(record, board_field, '|') ||
        !std::getline(record, side_field, '|')) return 2;
    cmz::vm3::InitialPosition input{};
    input.board.fill(cmz::vm3::PieceState::Empty);
    std::stringstream board_stream(board_field);
    std::string piece;
    std::int64_t square = 0;
    while (std::getline(board_stream, piece, ',')) {
      if (square >= 64) return 3;
      input.board[square++] =
          static_cast<cmz::vm3::PieceState>(std::stoll(piece));
    }
    if (square != 64) return 4;
    input.side = static_cast<cmz::vm3::Side>(std::stoll(side_field));
    input.castling = 0;
    input.ep_square = -1;
    input.fullmove = 1;
    input.claim_mode = cmz::vm3::ClaimMode::LiteralClaim;
    states.push_back(cmz::vm3::bind_initial_state(program, input).tokens);
  }
  constexpr std::int64_t batch_size = 4;
  for (std::int64_t begin = 0; begin < static_cast<std::int64_t>(states.size());
       begin += batch_size) {
    const auto count = std::min<std::int64_t>(batch_size, states.size() - begin);
    std::vector<torch::Tensor> chunk(states.begin() + begin,
                                     states.begin() + begin + count);
    const auto legal_batch = cmz::vm3::compute_rule_legal_batch(
        program, torch::stack(chunk)).to(torch::kCPU);
    for (std::int64_t item = 0; item < count; ++item) {
      const auto legal = legal_batch.select(0, item);
      std::cout << "CMZ|";
      bool first = true;
      for (const auto& candidate : bank.moves) {
        if (legal.index({candidate.id}).item<double>() != 1.0) continue;
        if (!first) std::cout << ',';
        first = false;
        std::cout << square_name(candidate.source) << square_name(candidate.target);
        if (candidate.promotion != cmz::vm3::Promotion::None)
          std::cout << promotion_letter(candidate.promotion);
      }
      std::cout << '\n';
    }
  }
  return 0;
}
