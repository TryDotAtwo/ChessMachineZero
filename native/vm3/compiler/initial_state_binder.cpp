#include <cmz_vm3/chess_state.h>

#include <stdexcept>

namespace cmz::vm3 {
namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::invalid_argument(message);
}

void activate(torch::Tensor& active, std::int64_t row) {
  active.index_put_({row}, 1.0);
}
}

torch::Tensor canonical_state_layout(const torch::TensorOptions& options) {
  return torch::tensor({
      {kBoardOffset, kBoardRows},
      {kSideOffset, kSideRows},
      {kCastlingOffset, kCastlingRows},
      {kRawEpOffset, kRawEpRows},
      {kHalfmoveOffset, kHalfmoveRows},
      {kFullmoveOffset, kFullmoveRows},
      {kCursorDigitsOffset, kCursorDigitsRows},
      {kClaimModeOffset, kClaimModeRows},
      {kClaimAvailabilityOffset, kClaimAvailabilityRows},
      {kTerminalOffset, kTerminalRows},
      {kResultOffset, kResultRows}},
      options.dtype(torch::kInt64));
}

ChessRingState bind_initial_state(const FrozenChessProgram& program,
                                  const InitialPosition& input) {
  const auto found = program.tensors.find("state_template");
  const auto layout = program.tensors.find("state_layout");
  require(found != program.tensors.end(), "VM3 state_template is missing");
  require(layout != program.tensors.end(), "VM3 state_layout is missing");
  require(torch::equal(
              layout->second.to(torch::kCPU),
              canonical_state_layout(torch::TensorOptions().device(torch::kCPU))),
          "VM3 state_layout does not match the canonical ABI");
  require(found->second.dim() == 2 && found->second.size(0) >= kStateCoreRows &&
              found->second.size(1) > kStateActiveFeature,
          "VM3 state_template has an incompatible ABI");
  require(static_cast<std::int64_t>(input.side) >= 0 &&
              static_cast<std::int64_t>(input.side) < kSideRows,
          "VM3 initial side is invalid");
  require(input.castling < kCastlingRows, "VM3 initial castling is invalid");
  require(input.ep_square >= -1 && input.ep_square < 64,
          "VM3 initial raw EP is invalid");
  require(input.halfmove >= 0 && input.halfmove < kHalfmoveRows,
          "VM3 initial halfmove is invalid");
  require(input.fullmove >= 1 && input.fullmove <= kMaximumFullmove,
          "VM3 initial fullmove is invalid");
  require(static_cast<std::int64_t>(input.claim_mode) >= 0 &&
              static_cast<std::int64_t>(input.claim_mode) < kClaimModeRows,
          "VM3 initial claim mode is invalid");

  auto tokens = found->second.clone();
  auto active = tokens.select(-1, kStateActiveFeature);
  active.narrow(0, 0, kStateCoreRows).zero_();
  for (std::int64_t square = 0; square < 64; ++square) {
    const auto piece = static_cast<std::int64_t>(
        input.board.at(static_cast<std::size_t>(square)));
    require(piece >= 0 && piece < 13, "VM3 initial piece state is invalid");
    activate(active, kBoardOffset + square * 13 + piece);
  }
  activate(active, kSideOffset + static_cast<std::int64_t>(input.side));
  activate(active, kCastlingOffset + input.castling);
  activate(active, kRawEpOffset + input.ep_square + 1);
  activate(active, kHalfmoveOffset + input.halfmove);
  activate(active, kFullmoveOffset + input.fullmove % 128);
  activate(active, kFullmoveOffset + 128 + input.fullmove / 128);
  activate(active, kCursorDigitsOffset);
  activate(active, kCursorDigitsOffset + 128);
  activate(active, kCursorDigitsOffset + 256);
  activate(active, kClaimModeOffset + static_cast<std::int64_t>(input.claim_mode));
  activate(active, kTerminalOffset + static_cast<std::int64_t>(TerminalKind::Running));
  activate(active, kResultOffset + static_cast<std::int64_t>(GameResult::Running));

  return {tokens, TrajectoryTape{}, torch::zeros({}, tokens.options())};
}

}  // namespace cmz::vm3
