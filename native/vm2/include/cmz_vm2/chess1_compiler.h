#pragma once

#include <array>
#include <cstdint>

#include "cmz_vm2/schema.h"

namespace cmz::vm2 {

enum class Chess1Piece : std::int64_t {
    Empty = 0,
    WhitePawn = 1,
    BlackPawn = 2,
    Other = 3,
};

enum class Chess1Side : std::int64_t {
    White = 0,
    Black = 1,
};

struct Chess1Board {
    std::array<Chess1Piece, 64> squares{};
    Chess1Side side = Chess1Side::White;
};

namespace chess1 {

inline constexpr std::int64_t kSquareTokenCount = 64;
inline constexpr std::int64_t kCandidateTokenCount = 128;
inline constexpr std::int64_t kOutputSquareTokenCount = 64;
inline constexpr std::int64_t kGeometryTokenCount = 256;
inline constexpr std::int64_t kMatchingRelationTokenCount = 8;
inline constexpr std::int64_t kTargetEmptyRelationTokenCount = 4;
inline constexpr std::int64_t kPathRelationTokenCount = 8;
inline constexpr std::int64_t kConjunctionRelationTokenCount = 32;
inline constexpr std::int64_t kLegalityRelationTokenCount =
    kMatchingRelationTokenCount + kTargetEmptyRelationTokenCount +
    kPathRelationTokenCount + kConjunctionRelationTokenCount;
inline constexpr std::int64_t kSideToken = 0;
inline constexpr std::int64_t kSquareTokenBegin = 1;
inline constexpr std::int64_t kOffboardToken = kSquareTokenBegin + kSquareTokenCount;
inline constexpr std::int64_t kCandidateTokenBegin = kOffboardToken + 1;
inline constexpr std::int64_t kGeometryTokenBegin = kCandidateTokenBegin + kCandidateTokenCount;
inline constexpr std::int64_t kLegalityRelationTokenBegin =
    kGeometryTokenBegin + kGeometryTokenCount;
inline constexpr std::int64_t kSelectedToken =
    kLegalityRelationTokenBegin + kLegalityRelationTokenCount;
inline constexpr std::int64_t kSourceWriteToken = kSelectedToken + 1;
inline constexpr std::int64_t kTargetWriteToken = kSourceWriteToken + 1;
inline constexpr std::int64_t kOutputSquareTokenBegin =
    kTargetWriteToken + 1;
inline constexpr std::int64_t kOutputSideToken =
    kOutputSquareTokenBegin + kOutputSquareTokenCount;
inline constexpr std::int64_t kTokenCount =
    kOutputSideToken + 1;

inline constexpr std::int64_t kLegalOffset = 310;
inline constexpr std::int64_t kInputPieceOffset = 0;
inline constexpr std::int64_t kInputSideOffset = 8;
inline constexpr std::int64_t kOutputPieceOffset = 200;
inline constexpr std::int64_t kOutputSideOffset = 204;

}  // namespace chess1

ProgramImage compile_chess1(const Chess1Board& board, std::int64_t selected_source);
ProgramImage compile_chess1_auto(const Chess1Board& board);

}  // namespace cmz::vm2
