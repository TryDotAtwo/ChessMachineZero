#pragma once

#include <cstdint>

namespace cmz::vm3 {

inline constexpr std::int64_t kLookupHeadDim = 2;
inline constexpr std::int64_t kMoveCandidateCount = 4096 + 176;
inline constexpr std::int64_t kCandidateSelectorRouteCount =
    kMoveCandidateCount + 1;
inline constexpr std::int64_t kPolicyControlCount = 3;
inline constexpr std::int64_t kControlRouteCount = kPolicyControlCount + 1;
inline constexpr std::int64_t kPieceStateCount = 13;
inline constexpr std::int64_t kHalfmoveClaimLimit = 100;
inline constexpr std::int64_t kHalfmoveAutomaticLimit = 150;
inline constexpr std::int64_t kMaxOrthodoxGamePlies = 19050;
inline constexpr std::int64_t kMaxOrthodoxRingSteps =
    kMaxOrthodoxGamePlies + 1;

}  // namespace cmz::vm3
