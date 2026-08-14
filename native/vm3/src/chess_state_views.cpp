#include <cmz_vm3/chess_state.h>

namespace cmz::vm3 {
namespace {
torch::Tensor active_rows(const torch::Tensor& tokens,
                          std::int64_t offset,
                          std::int64_t width) {
  return tokens.select(-1, kStateActiveFeature).narrow(0, offset, width);
}
}

torch::Tensor board_state_view(const torch::Tensor& tokens) {
  return active_rows(tokens, kBoardOffset, kBoardRows).view({64, 13});
}
torch::Tensor side_view(const torch::Tensor& tokens) {
  return active_rows(tokens, kSideOffset, kSideRows);
}
torch::Tensor castling_view(const torch::Tensor& tokens) {
  return active_rows(tokens, kCastlingOffset, kCastlingRows);
}
torch::Tensor raw_ep_view(const torch::Tensor& tokens) {
  return active_rows(tokens, kRawEpOffset, kRawEpRows);
}
torch::Tensor halfmove_view(const torch::Tensor& tokens) {
  return active_rows(tokens, kHalfmoveOffset, kHalfmoveRows);
}
torch::Tensor fullmove_digits_view(const torch::Tensor& tokens) {
  return active_rows(tokens, kFullmoveOffset, kFullmoveRows).view({2, 128});
}
torch::Tensor cursor_digits_view(const torch::Tensor& tokens) {
  return active_rows(tokens, kCursorDigitsOffset, kCursorDigitsRows).view({3, 128});
}
torch::Tensor claim_mode_view(const torch::Tensor& tokens) {
  return active_rows(tokens, kClaimModeOffset, kClaimModeRows);
}
torch::Tensor claim_availability_view(const torch::Tensor& tokens) {
  return active_rows(tokens, kClaimAvailabilityOffset, kClaimAvailabilityRows);
}
torch::Tensor terminal_view(const torch::Tensor& tokens) {
  return active_rows(tokens, kTerminalOffset, kTerminalRows);
}
torch::Tensor result_view(const torch::Tensor& tokens) {
  return active_rows(tokens, kResultOffset, kResultRows);
}

}  // namespace cmz::vm3
