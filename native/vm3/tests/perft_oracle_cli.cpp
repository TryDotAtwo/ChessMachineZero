#include <cmz_vm3/chess_state.h>
#include <cmz_vm3/recurrent_ring.h>
#include <cmz_vm3/rule_compiler.h>

#include <torch/torch.h>

#include <cstdint>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct PerftRunner {
  const cmz::vm3::FrozenChessProgram& program;
  const cmz::vm3::CandidateBank& bank;
  torch::TensorOptions options;

  std::uint64_t count(const torch::Tensor& state, std::int64_t depth) const {
    if (depth == 0) return 1;
    const auto trials = cmz::vm3::compute_trial_transitions_hard_forward_batch(
        program, state.unsqueeze(0));
    std::vector<std::int64_t> legal_ids;
    const auto legal = trials.legal.select(0, 0).to(torch::kCPU);
    for (const auto& candidate : bank.moves)
      if (legal.index({candidate.id}).item<double>() == 1.0)
        legal_ids.push_back(candidate.id);
    if (depth == 1) return legal_ids.size();

    std::uint64_t total = 0;
    const auto commit = torch::ones({1, 1}, options);
    for (const auto move_id : legal_ids) {
      const auto selection = torch::one_hot(
          torch::tensor({move_id}, options.dtype(torch::kInt64)),
          cmz::vm3::kMoveCandidateCount).to(options);
      const auto next = cmz::vm3::materialize_selected_trial_state(
          program, state.unsqueeze(0), trials, selection, commit).squeeze(0);
      total += count(next, depth - 1);
    }
    return total;
  }
};

cmz::vm3::InitialPosition parse_position(const std::string& line,
                                         std::int64_t& depth) {
  std::stringstream record(line);
  std::string board_field, side_field, castling_field, ep_field;
  std::string halfmove_field, fullmove_field, depth_field;
  if (!std::getline(record, board_field, '|') ||
      !std::getline(record, side_field, '|') ||
      !std::getline(record, castling_field, '|') ||
      !std::getline(record, ep_field, '|') ||
      !std::getline(record, halfmove_field, '|') ||
      !std::getline(record, fullmove_field, '|') ||
      !std::getline(record, depth_field, '|'))
    throw std::runtime_error("bad perft record");

  cmz::vm3::InitialPosition input{};
  input.board.fill(cmz::vm3::PieceState::Empty);
  std::stringstream board_stream(board_field);
  std::string piece;
  std::int64_t square = 0;
  while (std::getline(board_stream, piece, ',')) {
    if (square >= 64) throw std::runtime_error("too many board cells");
    input.board[square++] =
        static_cast<cmz::vm3::PieceState>(std::stoll(piece));
  }
  if (square != 64) throw std::runtime_error("too few board cells");
  input.side = static_cast<cmz::vm3::Side>(std::stoll(side_field));
  input.castling = static_cast<std::uint8_t>(std::stoll(castling_field));
  input.ep_square = std::stoll(ep_field);
  input.halfmove = std::stoll(halfmove_field);
  input.fullmove = std::stoll(fullmove_field);
  input.claim_mode = cmz::vm3::ClaimMode::LiteralClaim;
  depth = std::stoll(depth_field);
  return input;
}
}  // namespace

int main(int argc, char** argv) {
  bool use_cuda = false;
  for (int index = 1; index < argc; ++index)
    use_cuda = use_cuda || std::string(argv[index]) == "--cuda";
  TORCH_CHECK(!use_cuda || torch::cuda::is_available(),
              "CUDA perft oracle requested without GPU");
  const auto options = torch::TensorOptions().dtype(torch::kFloat64).device(
      use_cuda ? torch::kCUDA : torch::kCPU);
  const auto program = cmz::vm3::compile_minimal_rule_program(options);
  const auto bank = cmz::vm3::compile_candidate_bank();
  const PerftRunner runner{program, bank, options};

  std::string line;
  while (std::getline(std::cin, line)) {
    if (line.empty()) continue;
    std::int64_t depth = 0;
    const auto input = parse_position(line, depth);
    const auto state = cmz::vm3::bind_initial_state(program, input).tokens;
    std::cout << "CMZ_PERFT|" << runner.count(state, depth) << '\n';
  }
  return 0;
}
