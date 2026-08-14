#include <cmz_vm3/tensor_contract.h>

#include <cstdint>
#include <iostream>

namespace {

bool require(bool condition, const char* message) {
  if (!condition) {
    std::cerr << message << '\n';
  }
  return condition;
}

}  // namespace

int main() {
  bool ok = true;
  ok &= require(cmz::vm3::kLookupHeadDim == 2, "lookup head must be physical 2D");
  ok &= require(cmz::vm3::kMoveCandidateCount == 4272, "move candidate count");
  ok &= require(cmz::vm3::kCandidateSelectorRouteCount == 4273,
                "candidate selector includes one sentinel");
  ok &= require(cmz::vm3::kPolicyControlCount == 3, "policy control count");
  ok &= require(cmz::vm3::kControlRouteCount == 4, "control routes include halt");
  ok &= require(cmz::vm3::kPieceStateCount == 13, "piece state count");
  ok &= require(cmz::vm3::kHalfmoveClaimLimit == 100, "50-move claim limit");
  ok &= require(cmz::vm3::kHalfmoveAutomaticLimit == 150,
                "75-move automatic limit");
  ok &= require(cmz::vm3::kMaxOrthodoxGamePlies == 19050,
                "conservative maximum committed plies");
  ok &= require(cmz::vm3::kMaxOrthodoxRingSteps == 19051,
                "terminal classification tick");
  return ok ? 0 : 1;
}
