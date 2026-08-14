#pragma once

#include <cmz_vm3/program_image.h>

#include <array>
#include <cstdint>
#include <vector>

namespace cmz::vm3 {

enum class PieceState : std::int64_t {
  Empty, WP, WN, WB, WR, WQ, WK, BP, BN, BB, BR, BQ, BK
};
enum class Side : std::int64_t { White, Black };
enum class ClaimMode : std::int64_t { LiteralClaim, AutoClaimSelfPlay };
enum class ClaimAvailability : std::int64_t { ClaimNow, ClaimAfter };
enum class TerminalKind : std::int64_t {
  Running, WhiteCheckmated, BlackCheckmated, Stalemate,
  MaterialDeadSubset, Fivefold, SeventyFiveMove,
  ClaimedThreefold, ClaimedFiftyMove, CapacityExhausted
};
enum class GameResult : std::int64_t { Running, WhiteWin, BlackWin, Draw };

struct MoveSlot { torch::Tensor move, active; };
struct PolicyKvSlot { torch::Tensor key, value; };
struct TrajectoryTape {
  std::vector<MoveSlot> move_slots;
  std::vector<PolicyKvSlot> policy_kv_slots;
};
struct ChessRingState {
  torch::Tensor tokens;
  TrajectoryTape trajectory;
  torch::Tensor cursor;
};
struct InitialPosition {
  std::array<PieceState, 64> board;
  Side side;
  std::uint8_t castling;
  std::int64_t ep_square;
  std::int64_t halfmove;
  std::int64_t fullmove;
  ClaimMode claim_mode;
};

inline constexpr std::int64_t kStateActiveFeature = 0;
inline constexpr std::int64_t kBoardOffset = 0;
inline constexpr std::int64_t kBoardRows = 64 * 13;
inline constexpr std::int64_t kSideOffset = kBoardOffset + kBoardRows;
inline constexpr std::int64_t kSideRows = 2;
inline constexpr std::int64_t kCastlingOffset = kSideOffset + kSideRows;
inline constexpr std::int64_t kCastlingRows = 16;
inline constexpr std::int64_t kRawEpOffset = kCastlingOffset + kCastlingRows;
inline constexpr std::int64_t kRawEpRows = 65;
inline constexpr std::int64_t kHalfmoveOffset = kRawEpOffset + kRawEpRows;
inline constexpr std::int64_t kHalfmoveRows = 151;
inline constexpr std::int64_t kFullmoveOffset = kHalfmoveOffset + kHalfmoveRows;
inline constexpr std::int64_t kFullmoveRows = 2 * 128;
inline constexpr std::int64_t kCursorDigitsOffset = kFullmoveOffset + kFullmoveRows;
inline constexpr std::int64_t kCursorDigitsRows = 3 * 128;
inline constexpr std::int64_t kClaimModeOffset = kCursorDigitsOffset + kCursorDigitsRows;
inline constexpr std::int64_t kClaimModeRows = 2;
inline constexpr std::int64_t kClaimAvailabilityOffset = kClaimModeOffset + kClaimModeRows;
inline constexpr std::int64_t kClaimAvailabilityRows = 2;
inline constexpr std::int64_t kTerminalOffset = kClaimAvailabilityOffset + kClaimAvailabilityRows;
inline constexpr std::int64_t kTerminalRows = 10;
inline constexpr std::int64_t kResultOffset = kTerminalOffset + kTerminalRows;
inline constexpr std::int64_t kResultRows = 4;
inline constexpr std::int64_t kStateCoreRows = kResultOffset + kResultRows;
inline constexpr std::int64_t kMaximumFullmove = 9526;
inline constexpr std::int64_t kStateLayoutFieldCount = 11;

torch::Tensor canonical_state_layout(const torch::TensorOptions& options);

torch::Tensor board_state_view(const torch::Tensor& tokens);
torch::Tensor side_view(const torch::Tensor& tokens);
torch::Tensor castling_view(const torch::Tensor& tokens);
torch::Tensor raw_ep_view(const torch::Tensor& tokens);
torch::Tensor halfmove_view(const torch::Tensor& tokens);
torch::Tensor fullmove_digits_view(const torch::Tensor& tokens);
torch::Tensor cursor_digits_view(const torch::Tensor& tokens);
torch::Tensor claim_mode_view(const torch::Tensor& tokens);
torch::Tensor claim_availability_view(const torch::Tensor& tokens);
torch::Tensor terminal_view(const torch::Tensor& tokens);
torch::Tensor result_view(const torch::Tensor& tokens);

ChessRingState bind_initial_state(const FrozenChessProgram& program,
                                  const InitialPosition& input);

}  // namespace cmz::vm3
