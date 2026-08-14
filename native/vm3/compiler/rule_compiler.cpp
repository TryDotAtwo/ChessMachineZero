#include <cmz_vm3/chess_state.h>
#include <cmz_vm3/rule_compiler.h>
#include <cmz_vm3/rule_layout.h>
#include <cmz_vm3/tensor_contract.h>

#include <ATen/ops/one_hot.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <utility>

namespace cmz::vm3 {
namespace {
using namespace rule;

torch::Tensor zeros(std::initializer_list<std::int64_t> shape) {
  return torch::zeros(shape, torch::kFloat64);
}

FrozenStage make_stage(std::int64_t key_begin,
                       std::int64_t key_count,
                       torch::Tensor wq,
                       torch::Tensor wk,
                       torch::Tensor wv) {
  auto query_router = zeros({1, kPacketRows});
  query_router.index_put_({0, kQueryRow}, 1.0);
  auto key_router = zeros({key_count, kPacketRows});
  auto global_ids = zeros({key_count});
  for (std::int64_t key = 0; key < key_count; ++key) {
    key_router.index_put_({key, key_begin + key}, 1.0);
    global_ids.index_put_({key}, static_cast<double>(key_begin + key));
  }
  auto output_router = zeros({kPacketRows, 1});
  output_router.index_put_({kQueryRow, 0}, 1.0);
  Attention2dPlan plan;
  plan.query_group_offsets = {0, 1};
  plan.blocks.push_back({query_router, key_router, output_router,
                         zeros({1, key_count}),
                         torch::ones({1, key_count}, torch::kFloat64),
                         global_ids});
  return {std::move(wq), std::move(wk), std::move(wv),
          zeros({1, 1}), zeros({1, 1}), zeros({1, 1}),
          std::move(plan)};
}

FrozenStage active_lookup(std::int64_t key_begin,
                          std::int64_t key_count,
                          std::initializer_list<std::pair<std::int64_t,
                                                          std::int64_t>> copies) {
  auto wq = zeros({kFeatureCount, 2});
  auto wk = zeros({kFeatureCount, 2});
  auto wv = zeros({kFeatureCount, kFeatureCount});
  wq.index_put_({kConst, 0}, 1.0);
  wk.index_put_({kActive, 0}, 1.0);
  for (const auto& [source, target] : copies) wv.index_put_({source, target}, 1.0);
  return make_stage(key_begin, key_count, wq, wk, wv);
}

FrozenStage single_copy(std::int64_t key_row,
                        std::initializer_list<std::pair<std::int64_t,
                                                        std::int64_t>> copies) {
  auto wq = zeros({kFeatureCount, 2});
  auto wk = zeros({kFeatureCount, 2});
  auto wv = zeros({kFeatureCount, kFeatureCount});
  wq.index_put_({kConst, 0}, 1.0);
  wk.index_put_({kConst, 0}, 1.0);
  for (const auto& [source, target] : copies) wv.index_put_({source, target}, 1.0);
  return make_stage(key_row, 1, wq, wk, wv);
}

FrozenStage truth_gate(std::int64_t left,
                       std::int64_t right,
                       std::int64_t destination,
                       bool conjunction) {
  auto wq = zeros({kFeatureCount, 2});
  auto wk = zeros({kFeatureCount, 2});
  auto wv = zeros({kFeatureCount, kFeatureCount});
  wq.index_put_({kConst, 0}, -1.0);
  wq.index_put_({kConst, 1}, -1.0);
  wq.index_put_({left, 0}, 2.0);
  wq.index_put_({right, 1}, 2.0);
  wk.index_put_({kTruthX, 0}, 1.0);
  wk.index_put_({kTruthY, 1}, 1.0);
  wv.index_put_({conjunction ? kTruthAnd : kTruthOr, destination}, 1.0);
  return make_stage(kTruthBegin, 4, wq, wk, wv);
}

void move_stage(FrozenStage& stage, const torch::Device& device) {
  stage.wq = stage.wq.to(device);
  stage.wk = stage.wk.to(device);
  stage.wv = stage.wv.to(device);
  for (auto& block : stage.attention_plan.blocks) {
    block.query_router = block.query_router.to(device);
    block.key_router = block.key_router.to(device);
    block.output_router = block.output_router.to(device);
    block.fixed_mask = block.fixed_mask.to(device);
    block.fixed_eligibility = block.fixed_eligibility.to(device);
    block.global_key_ids = block.global_key_ids.to(device);
  }
}

void add_policy_routing(FrozenChessProgram& program) {
  auto move_routes = zeros({kMoveCandidateCount, kCandidateSelectorRouteCount});
  move_routes.index_put_({torch::indexing::Slice(),
                          torch::indexing::Slice(0, kMoveCandidateCount)},
                         torch::eye(kMoveCandidateCount, torch::kFloat64));
  auto move_sentinel = zeros({1, kCandidateSelectorRouteCount});
  move_sentinel.index_put_({0, kMoveCandidateCount}, 1.0);
  auto control_routes = zeros({kPolicyControlCount, kControlRouteCount});
  control_routes.index_put_({torch::indexing::Slice(),
                             torch::indexing::Slice(0, kPolicyControlCount)},
                            torch::eye(kPolicyControlCount, torch::kFloat64));
  auto halt_route = zeros({1, kControlRouteCount});
  halt_route.index_put_({0, 3}, 1.0);
  auto claim_route = zeros({1, kControlRouteCount});
  claim_route.index_put_({0, 1}, 1.0);
  program.tensors.emplace("policy_move_routes", move_routes);
  program.tensors.emplace("policy_move_sentinel_route", move_sentinel);
  program.tensors.emplace("policy_control_routes", control_routes);
  program.tensors.emplace("policy_halt_route", halt_route);
  program.tensors.emplace("policy_claim_now_route", claim_route);
  program.tensors.emplace("policy_accept_column",
                          torch::tensor({{0.0}, {1.0}}, torch::kFloat64));
}

void register_stage_tensors(FrozenChessProgram& program) {
  for (std::size_t index = 0; index < program.pre_policy_stages.size(); ++index) {
    auto& stage = program.pre_policy_stages[index];
    const auto prefix = "pre_stage_" + std::to_string(index) + "_";
    program.tensors.emplace(prefix + "wq", stage.wq);
    program.tensors.emplace(prefix + "wk", stage.wk);
    program.tensors.emplace(prefix + "wv", stage.wv);
    program.tensors.emplace(prefix + "fixed_mask", stage.fixed_mask);
    program.tensors.emplace(prefix + "row_router", stage.row_router);
    program.tensors.emplace(prefix + "feature_router", stage.feature_router);
    const auto& block = stage.attention_plan.blocks.front();
    program.tensors.emplace(prefix + "query_router", block.query_router);
    program.tensors.emplace(prefix + "key_router", block.key_router);
    program.tensors.emplace(prefix + "output_router", block.output_router);
    program.tensors.emplace(prefix + "mask", block.fixed_mask);
    program.tensors.emplace(prefix + "eligibility", block.fixed_eligibility);
    program.tensors.emplace(prefix + "global_key_ids", block.global_key_ids);
  }
}

void build_manifest(FrozenChessProgram& program) {
  std::vector<std::string> names;
  names.reserve(program.tensors.size());
  for (const auto& [name, tensor] : program.tensors) names.push_back(name);
  std::sort(names.begin(), names.end());
  for (const auto& name : names) {
    const auto& tensor = program.tensors.at(name);
    program.manifest.push_back(
        {name, tensor.sizes().vec(), tensor.scalar_type(), tensor_sha256(tensor)});
  }
}

}  // namespace

FrozenChessProgram compile_minimal_rule_program(const torch::TensorOptions& options) {
  FrozenChessProgram program;
  program.schema_version = 1;
  const auto bank = compile_candidate_bank();
  const auto pawn = compile_pawn_geometry(bank);
  const auto knight = compile_knight_geometry(bank);
  const auto geometry = torch::cat({knight, pawn}, -1);

  auto packet = zeros({kPacketRows, kFeatureCount});
  packet.index_put_({kQueryRow, kConst}, 1.0);
  for (std::int64_t group : {kSourceBegin, kTargetBegin, kMiddleBegin}) {
    for (std::int64_t piece = 0; piece < kPieceStateCount; ++piece) {
      const auto row = group + piece;
      packet.index_put_({row, kConst}, 1.0);
      packet.index_put_({row, kPieceEmpty}, piece == 0 ? 1.0 : 0.0);
      packet.index_put_({row, kPieceWp}, piece == 1 ? 1.0 : 0.0);
      packet.index_put_({row, kPieceWn}, piece == 2 ? 1.0 : 0.0);
      packet.index_put_({row, kPieceBp}, piece == 7 ? 1.0 : 0.0);
      packet.index_put_({row, kPieceBn}, piece == 8 ? 1.0 : 0.0);
      const auto white = piece >= 1 && piece <= 6;
      const auto black = piece >= 7 && piece <= 12;
      packet.index_put_({row, kPieceWhite}, white ? 1.0 : 0.0);
      packet.index_put_({row, kPieceBlack}, black ? 1.0 : 0.0);
      packet.index_put_({row, kPieceNotWhite}, white ? 0.0 : 1.0);
      packet.index_put_({row, kPieceNotBlack}, black ? 0.0 : 1.0);
    }
  }
  for (std::int64_t side = 0; side < 2; ++side) {
    packet.index_put_({kSideBegin + side, kConst}, 1.0);
    packet.index_put_({kSideBegin + side, kSideWhite + side}, 1.0);
  }
  packet.index_put_({kGeometryRow, kConst}, 1.0);
  const std::array<std::array<double, 2>, 4> corners{{
      {{-1.0, -1.0}}, {{-1.0, 1.0}}, {{1.0, -1.0}}, {{1.0, 1.0}}}};
  for (std::int64_t index = 0; index < 4; ++index) {
    const auto row = kTruthBegin + index;
    packet.index_put_({row, kConst}, 1.0);
    packet.index_put_({row, kTruthX}, corners[index][0]);
    packet.index_put_({row, kTruthY}, corners[index][1]);
    const auto a = index >= 2;
    const auto b = index % 2 == 1;
    packet.index_put_({row, kTruthAnd}, a && b ? 1.0 : 0.0);
    packet.index_put_({row, kTruthOr}, a || b ? 1.0 : 0.0);
  }

  program.pre_policy_stages.push_back(active_lookup(
      kSourceBegin, 13, {{kPieceWp, kSourceWp}, {kPieceBp, kSourceBp},
                         {kPieceWn, kSourceWn}, {kPieceBn, kSourceBn}}));
  program.pre_policy_stages.push_back(active_lookup(
      kTargetBegin, 13,
      {{kPieceEmpty, kTargetEmpty}, {kPieceWhite, kTargetWhite},
       {kPieceBlack, kTargetBlack}, {kPieceNotWhite, kTargetNotWhite},
       {kPieceNotBlack, kTargetNotBlack}}));
  program.pre_policy_stages.push_back(
      active_lookup(kMiddleBegin, 13, {{kPieceEmpty, kMiddleEmpty}}));
  program.pre_policy_stages.push_back(active_lookup(
      kSideBegin, 2, {{kSideWhite, kSideWhite}, {kSideBlack, kSideBlack}}));
  program.pre_policy_stages.push_back(single_copy(
      kGeometryRow,
      {{kGeomKnight, kGeomKnight}, {kGeomWhitePush1, kGeomWhitePush1},
       {kGeomWhitePush2, kGeomWhitePush2},
       {kGeomWhiteCapture, kGeomWhiteCapture},
       {kGeomBlackPush1, kGeomBlackPush1},
       {kGeomBlackPush2, kGeomBlackPush2},
       {kGeomBlackCapture, kGeomBlackCapture}}));

  const std::array<std::array<std::int64_t, 3>, 4> actor_gates{{
      {{kSideWhite, kSourceWp, kActorWp}},
      {{kSideBlack, kSourceBp, kActorBp}},
      {{kSideWhite, kSourceWn, kActorWn}},
      {{kSideBlack, kSourceBn, kActorBn}}}};
  for (const auto& gate : actor_gates)
    program.pre_policy_stages.push_back(truth_gate(gate[0], gate[1], gate[2], true));
  const std::array<std::array<std::int64_t, 3>, 8> geometry_gates{{
      {{kActorWn, kGeomKnight, kWnGeom}}, {{kActorBn, kGeomKnight, kBnGeom}},
      {{kActorWp, kGeomWhitePush1, kWp1Geom}},
      {{kActorWp, kGeomWhitePush2, kWp2Geom}},
      {{kActorWp, kGeomWhiteCapture, kWpcGeom}},
      {{kActorBp, kGeomBlackPush1, kBp1Geom}},
      {{kActorBp, kGeomBlackPush2, kBp2Geom}},
      {{kActorBp, kGeomBlackCapture, kBpcGeom}}}};
  for (const auto& gate : geometry_gates)
    program.pre_policy_stages.push_back(truth_gate(gate[0], gate[1], gate[2], true));
  const std::array<std::array<std::int64_t, 3>, 8> target_gates{{
      {{kWnGeom, kTargetNotWhite, kWnLegal}},
      {{kBnGeom, kTargetNotBlack, kBnLegal}},
      {{kWp1Geom, kTargetEmpty, kWp1Legal}},
      {{kWp2Geom, kTargetEmpty, kWp2Target}},
      {{kWpcGeom, kTargetBlack, kWpcLegal}},
      {{kBp1Geom, kTargetEmpty, kBp1Legal}},
      {{kBp2Geom, kTargetEmpty, kBp2Target}},
      {{kBpcGeom, kTargetWhite, kBpcLegal}}}};
  for (const auto& gate : target_gates)
    program.pre_policy_stages.push_back(truth_gate(gate[0], gate[1], gate[2], true));
  program.pre_policy_stages.push_back(
      truth_gate(kWp2Target, kMiddleEmpty, kWp2Legal, true));
  program.pre_policy_stages.push_back(
      truth_gate(kBp2Target, kMiddleEmpty, kBp2Legal, true));
  const std::array<std::int64_t, 8> branches{{
      kWnLegal, kBnLegal, kWp1Legal, kWp2Legal,
      kWpcLegal, kBp1Legal, kBp2Legal, kBpcLegal}};
  const std::array<std::int64_t, 4> or_level1{{kOr0, kOr1, kOr2, kOr3}};
  for (std::int64_t index = 0; index < 4; ++index)
    program.pre_policy_stages.push_back(truth_gate(
        branches[index * 2], branches[index * 2 + 1], or_level1[index], false));
  program.pre_policy_stages.push_back(truth_gate(kOr0, kOr1, kOr4, false));
  program.pre_policy_stages.push_back(truth_gate(kOr2, kOr3, kOr5, false));
  program.pre_policy_stages.push_back(truth_gate(kOr4, kOr5, kLegal, false));

  auto middle = zeros({kMoveCandidateCount, 64});
  auto descriptors = zeros({kMoveCandidateCount, 3});
  for (const auto& move : bank.moves) {
    const auto source_rank = move.source / 8;
    const auto target_rank = move.target / 8;
    const auto same_file = move.source % 8 == move.target % 8;
    const auto square = same_file && std::abs(target_rank - source_rank) == 2
                            ? move.source % 8 + 8 * ((source_rank + target_rank) / 2)
                            : move.source;
    middle.index_put_({move.id, square}, 1.0);
    descriptors.index_put_({move.id, 0}, static_cast<double>(move.source));
    descriptors.index_put_({move.id, 1}, static_cast<double>(move.target));
    descriptors.index_put_({move.id, 2}, static_cast<double>(move.promotion));
  }
  const auto candidate_tokens = torch::cat(
      {bank.source_square.narrow(0, 0, kMoveCandidateCount),
       bank.target_square.narrow(0, 0, kMoveCandidateCount),
       bank.promotion.narrow(0, 0, kMoveCandidateCount),
       bank.source_file.narrow(0, 0, kMoveCandidateCount),
       bank.source_rank.narrow(0, 0, kMoveCandidateCount),
       bank.target_file.narrow(0, 0, kMoveCandidateCount),
       bank.target_rank.narrow(0, 0, kMoveCandidateCount)}, -1);

  auto source_rows = zeros({13, kPacketRows});
  auto target_rows = zeros({13, kPacketRows});
  auto middle_rows = zeros({13, kPacketRows});
  for (std::int64_t piece = 0; piece < 13; ++piece) {
    source_rows.index_put_({piece, kSourceBegin + piece}, 1.0);
    target_rows.index_put_({piece, kTargetBegin + piece}, 1.0);
    middle_rows.index_put_({piece, kMiddleBegin + piece}, 1.0);
  }
  auto side_rows = zeros({2, kPacketRows});
  side_rows.index_put_({0, kSideBegin}, 1.0);
  side_rows.index_put_({1, kSideBegin + 1}, 1.0);
  auto geometry_row = zeros({kPacketRows, 1});
  geometry_row.index_put_({kGeometryRow, 0}, 1.0);
  auto geometry_features = zeros({7, kFeatureCount});
  for (std::int64_t feature = 0; feature < 7; ++feature)
    geometry_features.index_put_({feature, kGeomKnight + feature}, 1.0);
  auto active_feature = zeros({kFeatureCount});
  active_feature.index_put_({kActive}, 1.0);

  program.tensors.emplace("state_template", zeros({kStateCoreRows, kFeatureCount}));
  program.tensors.emplace("state_layout", canonical_state_layout({}));
  program.tensors.emplace("packet_template", packet);
  program.tensors.emplace("candidate_tokens", candidate_tokens);
  program.tensors.emplace("candidate_source_square",
                          bank.source_square.narrow(0, 0, kMoveCandidateCount));
  program.tensors.emplace("candidate_target_square",
                          bank.target_square.narrow(0, 0, kMoveCandidateCount));
  program.tensors.emplace("candidate_middle_square", middle);
  program.tensors.emplace("candidate_geometry", geometry);
  program.tensors.emplace("candidate_descriptors", descriptors);
  program.tensors.emplace("source_packet_rows", source_rows);
  program.tensors.emplace("target_packet_rows", target_rows);
  program.tensors.emplace("middle_packet_rows", middle_rows);
  program.tensors.emplace("side_packet_rows", side_rows);
  program.tensors.emplace("geometry_packet_row", geometry_row);
  program.tensors.emplace("geometry_feature_router", geometry_features);
  program.tensors.emplace("packet_active_feature", active_feature);
  program.tensors.emplace("candidate_broadcast", torch::ones({kMoveCandidateCount, 1}, torch::kFloat64));
  program.tensors.emplace("empty_piece", at::one_hot(torch::tensor(0), 13).to(torch::kFloat64));
  program.tensors.emplace("side_flip", torch::tensor({{0.0, 1.0}, {1.0, 0.0}}, torch::kFloat64));
  auto board_rows = zeros({kBoardRows, kStateCoreRows});
  board_rows.index_put_({torch::indexing::Slice(),
                         torch::indexing::Slice(0, kBoardRows)},
                        torch::eye(kBoardRows, torch::kFloat64));
  auto side_state_rows = zeros({2, kStateCoreRows});
  side_state_rows.index_put_({0, kSideOffset}, 1.0);
  side_state_rows.index_put_({1, kSideOffset + 1}, 1.0);
  auto state_active_feature = zeros({kFeatureCount});
  state_active_feature.index_put_({kStateActiveFeature}, 1.0);
  program.tensors.emplace("board_state_rows", board_rows);
  program.tensors.emplace("side_state_rows", side_state_rows);
  program.tensors.emplace("state_active_feature", state_active_feature);
  program.tensors.emplace("trajectory_pad_token", zeros({1, 3}));
  program.tensors.emplace("zero_bit", zeros({1, 1}));
  add_policy_routing(program);
  register_stage_tensors(program);
  build_manifest(program);

  const auto device = options.device();
  for (auto& [name, tensor] : program.tensors) tensor = tensor.to(device);
  for (auto& stage : program.pre_policy_stages) move_stage(stage, device);
  return program;
}

}  // namespace cmz::vm3
