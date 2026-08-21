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

int main(int argc, char** argv) {
  bool use_cuda = false;
  bool final_legal = false;
  bool sparse_hard = false;
  for (int index = 1; index < argc; ++index) {
    use_cuda = use_cuda || std::string(argv[index]) == "--cuda";
    final_legal = final_legal || std::string(argv[index]) == "--legal";
    sparse_hard = sparse_hard || std::string(argv[index]) == "--sparse-hard";
  }
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
    std::string castling_field;
    std::string ep_field;
    if (!std::getline(record, board_field, '|') ||
        !std::getline(record, side_field, '|') ||
        !std::getline(record, castling_field, '|') ||
        !std::getline(record, ep_field, '|')) return 2;
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
    input.castling = static_cast<std::uint8_t>(std::stoll(castling_field));
    input.ep_square = std::stoll(ep_field);
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
    cmz::vm3::TrialTransitionBatch trials;
    if (sparse_hard) {
      std::vector<torch::Tensor> pseudo_legal;
      std::vector<torch::Tensor> legal;
      std::vector<torch::Tensor> side_in_check;
      std::vector<torch::Tensor> own_king_attacked;
      for (const auto& state : chunk) {
        const auto item = cmz::vm3::compute_trial_transitions_hard_forward(
            program, state);
        pseudo_legal.push_back(item.pseudo_legal);
        legal.push_back(item.legal);
        side_in_check.push_back(item.side_in_check);
        own_king_attacked.push_back(item.own_king_attacked);
      }
      trials = {torch::cat(pseudo_legal), torch::cat(legal),
                torch::cat(side_in_check), torch::cat(own_king_attacked)};
    } else {
      trials = cmz::vm3::compute_trial_transitions_batch(program,
                                                         torch::stack(chunk));
    }
    const auto legal_batch =
        (final_legal ? trials.legal : trials.pseudo_legal).to(torch::kCPU);
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
