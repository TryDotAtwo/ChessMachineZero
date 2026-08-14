#pragma once

#include <cstdint>

namespace cmz::vm3::rule {

inline constexpr std::int64_t kPacketRows = 47;
inline constexpr std::int64_t kFeatureCount = 64;
inline constexpr std::int64_t kQueryRow = 0;
inline constexpr std::int64_t kSourceBegin = 1;
inline constexpr std::int64_t kTargetBegin = 14;
inline constexpr std::int64_t kMiddleBegin = 27;
inline constexpr std::int64_t kSideBegin = 40;
inline constexpr std::int64_t kGeometryRow = 42;
inline constexpr std::int64_t kTruthBegin = 43;

inline constexpr std::int64_t kConst = 0;
inline constexpr std::int64_t kActive = 1;
inline constexpr std::int64_t kPieceWp = 2;
inline constexpr std::int64_t kPieceBp = 3;
inline constexpr std::int64_t kPieceWn = 4;
inline constexpr std::int64_t kPieceBn = 5;
inline constexpr std::int64_t kPieceEmpty = 6;
inline constexpr std::int64_t kPieceWhite = 7;
inline constexpr std::int64_t kPieceBlack = 8;
inline constexpr std::int64_t kPieceNotWhite = 9;
inline constexpr std::int64_t kPieceNotBlack = 10;
inline constexpr std::int64_t kSideWhite = 11;
inline constexpr std::int64_t kSideBlack = 12;
inline constexpr std::int64_t kGeomKnight = 13;
inline constexpr std::int64_t kGeomWhitePush1 = 14;
inline constexpr std::int64_t kGeomWhitePush2 = 15;
inline constexpr std::int64_t kGeomWhiteCapture = 16;
inline constexpr std::int64_t kGeomBlackPush1 = 17;
inline constexpr std::int64_t kGeomBlackPush2 = 18;
inline constexpr std::int64_t kGeomBlackCapture = 19;
inline constexpr std::int64_t kSourceWp = 20;
inline constexpr std::int64_t kSourceBp = 21;
inline constexpr std::int64_t kSourceWn = 22;
inline constexpr std::int64_t kSourceBn = 23;
inline constexpr std::int64_t kTargetEmpty = 24;
inline constexpr std::int64_t kTargetWhite = 25;
inline constexpr std::int64_t kTargetBlack = 26;
inline constexpr std::int64_t kTargetNotWhite = 27;
inline constexpr std::int64_t kTargetNotBlack = 28;
inline constexpr std::int64_t kMiddleEmpty = 29;
inline constexpr std::int64_t kActorWp = 30;
inline constexpr std::int64_t kActorBp = 31;
inline constexpr std::int64_t kActorWn = 32;
inline constexpr std::int64_t kActorBn = 33;
inline constexpr std::int64_t kWnGeom = 34;
inline constexpr std::int64_t kBnGeom = 35;
inline constexpr std::int64_t kWp1Geom = 36;
inline constexpr std::int64_t kWp2Geom = 37;
inline constexpr std::int64_t kWpcGeom = 38;
inline constexpr std::int64_t kBp1Geom = 39;
inline constexpr std::int64_t kBp2Geom = 40;
inline constexpr std::int64_t kBpcGeom = 41;
inline constexpr std::int64_t kWnLegal = 42;
inline constexpr std::int64_t kBnLegal = 43;
inline constexpr std::int64_t kWp1Legal = 44;
inline constexpr std::int64_t kWp2Target = 45;
inline constexpr std::int64_t kWpcLegal = 46;
inline constexpr std::int64_t kBp1Legal = 47;
inline constexpr std::int64_t kBp2Target = 48;
inline constexpr std::int64_t kBpcLegal = 49;
inline constexpr std::int64_t kWp2Legal = 50;
inline constexpr std::int64_t kBp2Legal = 51;
inline constexpr std::int64_t kOr0 = 52;
inline constexpr std::int64_t kOr1 = 53;
inline constexpr std::int64_t kOr2 = 54;
inline constexpr std::int64_t kOr3 = 55;
inline constexpr std::int64_t kOr4 = 56;
inline constexpr std::int64_t kOr5 = 57;
inline constexpr std::int64_t kLegal = 58;
inline constexpr std::int64_t kTruthX = 59;
inline constexpr std::int64_t kTruthY = 60;
inline constexpr std::int64_t kTruthAnd = 61;
inline constexpr std::int64_t kTruthOr = 62;

}  // namespace cmz::vm3::rule
