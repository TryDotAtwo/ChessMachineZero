#include "cmz_engine.h"

#include <cuda_runtime.h>
#include <torch/torch.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

extern "C" int cmz_cuda_double_values(const uint32_t* host_input, size_t len, uint32_t* host_output);
extern "C" int cmz_cuda_hardmax2d_values(
    const float* host_keys_xy,
    size_t len,
    float query_x,
    float query_y,
    uint32_t* host_selected,
    float* host_score);
extern "C" int cmz_cutlass_hardmax2d_values(
    const float* host_keys_xy,
    size_t len,
    float query_x,
    float query_y,
    uint32_t* host_selected,
    float* host_score);
extern "C" int cmz_cutlass_topk2d_values(
    const float* host_keys_xy,
    size_t len,
    size_t k,
    float query_x,
    float query_y,
    uint32_t* host_selected);
extern "C" int cmz_cuda_select_trace_packet(
    const uint32_t* host_tokens,
    size_t packet_count,
    size_t query_index,
    uint32_t* host_output);
extern "C" int cmz_cuda_emit_trace_packet_attention(
    uint32_t op,
    uint32_t a0,
    uint32_t a1,
    uint32_t a2,
    uint32_t a3,
    uint32_t tag,
    uint32_t commit,
    uint32_t* host_output);
extern "C" int cmz_cuda_project_board_latest_writes(
    const uint32_t* host_tokens,
    size_t packet_count,
    uint32_t* host_square_piece_tokens,
    size_t square_capacity,
    uint32_t* host_side_to_move);
extern "C" int cmz_cuda_attack_table_lookup_attention(
    const uint32_t* host_keys,
    const uint64_t* host_values,
    size_t value_count,
    uint32_t query_key,
    uint64_t* host_output);
extern "C" int cmz_cuda_candidate_target_attention(
    uint32_t token,
    uint32_t from_square,
    uint64_t friendly_mask,
    uint64_t enemy_mask,
    uint64_t occupancy_mask,
    uint32_t ep_square,
    uint64_t* host_output);
extern "C" int cmz_cuda_candidate_moves_attention(
    const uint32_t* host_board_tokens,
    uint32_t white_to_move,
    uint32_t castling_rights,
    uint32_t ep_square,
    uint32_t* host_move_records,
    size_t record_capacity,
    uint32_t* host_record_count,
    uint32_t* host_layer_count);
extern "C" int cmz_cuda_ray_scan_attention(
    uint32_t from_square,
    int32_t delta_file,
    int32_t delta_rank,
    uint64_t occupancy_mask,
    uint64_t* host_output);
extern "C" int cmz_cuda_legal_filter_v2_attention(
    const uint32_t* host_board_tokens,
    uint32_t white_to_move,
    uint32_t from_square,
    uint32_t to_square,
    uint32_t promotion_token,
    uint32_t en_passant,
    uint32_t castle,
    uint32_t* host_legal,
    uint32_t* host_layer_count);
extern "C" int cmz_cuda_legal_filter_v2_batch_attention(
    const uint32_t* host_board_tokens,
    uint32_t white_to_move,
    const uint32_t* host_from_squares,
    const uint32_t* host_to_squares,
    const uint32_t* host_promotion_tokens,
    const uint32_t* host_en_passant,
    const uint32_t* host_castle,
    size_t move_count,
    uint32_t* host_legal_outputs,
    uint32_t* host_layer_count);
extern "C" int cmz_cuda_make_move_board_attention(
    const uint32_t* host_board_tokens,
    uint32_t white_to_move,
    uint32_t from_square,
    uint32_t to_square,
    uint32_t promotion_token,
    uint32_t en_passant,
    uint32_t castle,
    uint32_t* host_next_board_tokens,
    uint32_t* host_layer_count);
extern "C" int cmz_cuda_make_move_metadata_attention(
    uint32_t white_to_move,
    uint32_t castling_rights,
    uint32_t halfmove_clock,
    uint32_t fullmove_number,
    uint32_t moving_piece_token,
    uint32_t target_piece_token,
    uint32_t from_square,
    uint32_t to_square,
    uint32_t en_passant,
    uint32_t* host_next_metadata,
    uint32_t* host_layer_count);
extern "C" int cmz_cuda_terminal_status_attention(
    const uint32_t* host_board_tokens,
    uint32_t white_to_move,
    uint32_t castling_rights,
    uint32_t ep_square,
    uint32_t halfmove_clock,
    uint32_t repetition_count,
    uint32_t adjudication_cap_reached,
    uint32_t* host_result_reason,
    uint32_t* host_layer_count);
extern "C" int cmz_cuda_resolve_move_attention(
    const uint32_t* host_move_records,
    const uint32_t* host_legal_bits,
    size_t record_count,
    uint32_t requested_from,
    uint32_t requested_to,
    uint32_t requested_promotion,
    uint32_t* host_selected_record,
    uint32_t* host_found);
extern "C" int cmz_cuda_castle_target_attention(
    const uint32_t* host_board_tokens,
    uint32_t castling_rights,
    uint32_t white,
    uint64_t* host_output);

struct CmzEngine {
    std::string last_error;
    std::string legal_trace_stream_fen;
    size_t legal_trace_stream_packet_count = 0;
    size_t legal_trace_stream_cursor = 0;
    size_t attention_decode_count = 0;
    size_t cuda_trace_emit_attention_count = 0;
    size_t cuda_trace_select_count = 0;
    size_t cutlass_hardmax2d_count = 0;
    size_t cuda_board_projection_count = 0;
    size_t cuda_attack_table_attention_count = 0;
    size_t cuda_candidate_table_attention_count = 0;
    size_t cuda_candidate_move_attention_count = 0;
    size_t cuda_candidate_move_layer_count = 0;
    size_t cuda_resolve_move_attention_count = 0;
    size_t cuda_ray_scan_attention_count = 0;
    size_t cuda_legal_filter_attention_count = 0;
    size_t cuda_legal_filter_batch_attention_count = 0;
    size_t cuda_legal_filter_v2_attention_count = 0;
    size_t cuda_legal_filter_v2_layer_count = 0;
    size_t cuda_legal_filter_v2_batch_attention_count = 0;
    size_t cuda_legal_filter_v2_batch_layer_count = 0;
    size_t cuda_castle_attention_count = 0;
    size_t cuda_make_move_board_attention_count = 0;
    size_t cuda_make_move_metadata_attention_count = 0;
    size_t cuda_terminal_status_attention_count = 0;
    size_t cuda_terminal_status_layer_count = 0;
    size_t frozen_layer_step_count = 0;
    bool decoder_initialized = false;
    torch::Tensor decoder_command_weight;
    torch::Tensor decoder_command_bias;
};

namespace {

constexpr char kEmpty = '\0';
constexpr int kCastleWhiteKing = 1;
constexpr int kCastleWhiteQueen = 2;
constexpr int kCastleBlackKing = 4;
constexpr int kCastleBlackQueen = 8;
constexpr size_t kTracePacketWidth = 7;
constexpr uint32_t kTraceCandidate = 10;
constexpr uint32_t kTraceLegalSet = 11;
constexpr uint32_t kTraceCommitMove = 14;
constexpr uint32_t kTraceTerminalSet = 15;
constexpr uint32_t kTraceProgramHalt = 17;
constexpr uint32_t kTraceWriteSq = 1;
constexpr uint32_t kTraceWriteReg = 3;
constexpr uint32_t kTraceWriteCastle = 6;
constexpr uint32_t kTraceWriteEp = 7;
constexpr uint32_t kTraceWriteClock = 8;
constexpr uint32_t kTraceTagBoard = 1;
constexpr uint32_t kTraceTagState = 2;
constexpr uint32_t kTraceTagMove = 3;
constexpr uint32_t kTraceTagLegal = 4;
constexpr uint32_t kTraceTagTerminal = 5;
constexpr uint32_t kMoveFlagCapture = 1;
constexpr uint32_t kMoveFlagEp = 2;
constexpr uint32_t kMoveFlagCastle = 4;
constexpr uint32_t kMoveFlagPromotion = 8;
constexpr uint32_t kPromoBase = 5;
constexpr uint32_t kSquareCount = 64;
constexpr uint32_t kRegSideToMove = 1;
constexpr uint32_t kNoEp = 64;
constexpr uint32_t kResultOngoing = 0;
constexpr uint32_t kResultWhiteWin = 1;
constexpr uint32_t kResultBlackWin = 2;
constexpr uint32_t kResultDraw = 3;
constexpr uint32_t kReasonNone = 0;
constexpr uint32_t kReasonCheckmate = 1;
constexpr uint32_t kReasonStalemate = 2;
constexpr uint32_t kReasonFiftyMove = 3;
constexpr uint32_t kReasonThreefold = 4;
constexpr uint32_t kReasonInsufficientMaterial = 5;
constexpr uint32_t kReasonAdjudicationCap = 6;
constexpr size_t kDecoderCommandCount = 6;
constexpr size_t kCandidateMoveRecordWidth = 5;
constexpr size_t kCandidateMoveRecordCapacity = 512;
constexpr const char* kRuntimeMode = "native_cuda_trace_select_decoder";
constexpr const char* kPerceptaContractJson =
    "{\"executor_head_dim\":2,"
    "\"rule_attention_backend\":\"hull_hardmax_2d\","
    "\"hull_score_backend\":\"cutlass_gemm_2d\","
    "\"hull_select_backend\":\"cuda_hardmax_select\","
    "\"hull_host_argmax\":false,"
    "\"topk_backend\":\"nested_hull_topk_2d\","
    "\"long_context_cache\":\"HullKVCache\","
    "\"trace_streaming\":true,"
    "\"simple_kv_cache\":false,"
    "\"python_hot_path\":false,"
    "\"fallback_allowed\":false,"
    "\"decoder_shared_white_black\":true,"
    "\"decoder_attention\":\"2d_heads\","
    "\"decoder_backend\":\"libtorch_cuda_policy_only_v1\","
    "\"learning_method\":\"self_play_policy_gradient\","
    "\"actor_critic\":false,"
    "\"critic_head_enabled\":false,"
    "\"value_head_enabled\":false,"
    "\"externally_prescribed_critic\":false,"
    "\"soft_surrogate_available\":false,"
    "\"tracepacket_backprop\":false}";
constexpr const char* kFrozenRuleGraphJson =
    "{\"graph_type\":\"frozen_attention_layer_stack\","
    "\"board_projection\":\"latest_write_hardmax_2d\","
    "\"board_projection_backend\":\"cuda_latest_write_projection\","
    "\"trace_select\":\"cursor_hardmax_2d\","
    "\"trace_select_backend\":\"cuda_trace_select_packet\","
    "\"trace_select_long_context_cache\":\"HullKVCache\","
    "\"trace_append_backend\":\"cuda_trace_packet_emit_attention\","
    "\"trace_streaming_backend\":\"incremental_packet_attention\","
    "\"trace_streaming_buffered\":false,"
    "\"trace_streaming_full_trace_precompute\":false,"
    "\"trace_append_cpp_loop_remaining\":false,"
    "\"hull_lookup_backend\":\"cutlass_gemm_2d\","
    "\"hullkv_rule_hot_path\":true,"
    "\"hull_hardmax_select_backend\":\"cuda_hardmax_select\","
    "\"hull_hardmax_host_argmax\":false,"
    "\"nested_hull_topk_backend\":\"cutlass_qk_cuda_topk_select\","
    "\"nested_hull_topk_cpu\":false,"
    "\"dashboard_policy_decoder\":true,"
    "\"dashboard_policy_selection_backend\":\"native_libtorch_policy_decoder\","
    "\"piece_dispatch\":\"frozen_table_attention\","
    "\"attack_masks\":\"static_attack_mask_table_attention\","
    "\"attack_masks_backend\":\"cuda_qk_hardmax_v_table_lookup\","
    "\"table_lookup_semantics\":\"qk_hardmax_v\","
    "\"attack_path_lowered\":\"pawn_knight_king_slider_ray_scan\","
    "\"ray_scan\":\"blocker_aware_ray_scan_attention\","
    "\"ray_scan_backend\":\"cuda_nearest_blocker_attention\","
    "\"ray_scan_semantics\":\"qk_hardmax_v_nearest_blocker\","
    "\"candidate_targets\":\"target_mask_attention\","
    "\"candidate_targets_backend\":\"cuda_qk_hardmax_v_target_lookup\","
    "\"candidate_target_dispatch_backend\":\"qk_hardmax_piece_family_select\","
    "\"candidate_offset_targets_backend\":\"qk_explicit_offset_slot_writes\","
    "\"candidate_filter_backend\":\"cuda_dynamic_mask_attention\","
    "\"pseudo_legal_moves_backend\":\"cuda_candidate_moves_layered_attention\","
    "\"pseudo_legal_cpp_control_flow_remaining\":false,"
    "\"candidate_moves_layers\":\"context_select,piece_dispatch,target_mask_select,castle_merge,promotion_expand,record_emit,prefix_rank_select,record_order_select\","
    "\"resolve_move_backend\":\"cuda_resolve_move_qk_hardmax_legal_set_attention\","
    "\"resolve_move_cpp_loop_remaining\":false,"
    "\"resolve_move_scan\":false,"
    "\"resolve_move_qk_hardmax_2d\":true,"
    "\"castling_targets\":\"castle_path_attention\","
    "\"castling_targets_backend\":\"cuda_castle_path_attention\","
    "\"legal_filter\":\"king_safety_attention\","
    "\"legal_filter_backend\":\"cuda_legal_filter_v2_layered_self_attention\","
    "\"legal_filter_v2_current_backend\":\"cuda_qk_hardmax_v_write_layers\","
    "\"legal_filter_v2_layers_started\":\"move_type_select,board_write_select,en_passant_capture_select,castle_rook_write_select,promotion_select,king_square_select,attack_source_select,ray_blocker_select,final_legal_select\","
    "\"legal_filter_v2_layers_complete\":\"move_type_select,board_write_select,en_passant_capture_select,castle_rook_write_select,promotion_select,king_square_select,attack_source_select,ray_blocker_select,final_legal_select\","
    "\"legal_filter_v2_inner_select\":\"qk_hardmax_2d_helpers\","
    "\"legal_filter_v2_inner_select_plain_cuda_loops_remaining\":false,"
    "\"legal_filter_v1_single_kernel_remaining\":false,"
    "\"legal_filter_batch_backend\":\"cuda_legal_filter_v2_batched_layered_self_attention\","
    "\"legal_filter_batch_v1_kernel_remaining\":false,"
    "\"legacy_legal_filter_cuda_symbols_present\":false,"
    "\"small_launch_fusion\":\"legal_filter_batch_v2\","
    "\"make_move\":\"board_write_attention\","
    "\"make_move_backend\":\"cuda_board_write_attention\","
    "\"make_move_board_squares_backend\":\"cuda_make_move_board_attention\","
    "\"make_move_board_square_layers\":\"move_type_select,board_write_select,en_passant_capture_select,castle_rook_write_select,promotion_select\","
    "\"make_move_board_metadata_backend\":\"cuda_make_move_metadata_attention\","
    "\"make_move_metadata_layers\":\"side_toggle_select,castling_rights_select,ep_square_select,halfmove_clock_select,fullmove_number_select\","
    "\"terminal_predicates\":\"terminal_status_attention\","
    "\"terminal_predicates_backend\":\"cuda_terminal_status_layered_attention\","
    "\"terminal_cpp_logic_remaining\":false,"
    "\"terminal_status_layers\":\"draw_rule_select,legal_presence_select,check_state_select,material_class_select,material_status_select,final_status_select\","
    "\"terminal_material_backend\":\"qk_material_class_bitmask_attention\","
    "\"move_record_expansion\":\"move_record_attention\","
    "\"promotion_expansion\":\"promotion_attention\","
    "\"trace_emission\":\"trace_packet_attention\","
    "\"make_move_trace_emission\":\"trace_packet_attention\","
    "\"attention_only_rule_substrate\":true,"
    "\"tensor_layer_substrate\":false,"
    "\"all_rules_must_lower_to_frozen_2d_self_attention\":true,"
    "\"target_full_frozen_attention_only\":true,"
    "\"current_full_frozen_2d_self_attention_only\":false,"
    "\"full_frozen_attention_only\":false,"
    "\"full_rule_lowering_complete\":false,"
    "\"semantic_attention_purity\":false,"
    "\"contract_overclaim_fixed\":true,"
    "\"cpp_control_flow_rule_vm_remaining\":false,"
    "\"semantic_source_audit\":\"rust_cuda_body_scan_v1\","
    "\"metadata_only_tests_remaining\":false,"
    "\"candidate_pawn_targets_backend\":\"qk_explicit_pawn_slot_writes\","
    "\"candidate_single_offset_backend\":\"qk_bounds_slot_friendly_filter\","
    "\"candidate_single_offset_coordinate_backend\":\"qk_coordinate_slot_lookup\","
    "\"candidate_single_offset_coordinate_table_backend\":\"qk_coordinate_table_slots\","
    "\"candidate_slider_targets_backend\":\"qk_explicit_slider_ray_slot_writes\","
    "\"candidate_slider_ray_backend\":\"qk_explicit_7_step_ray_slot_writes\","
    "\"candidate_record_emit_backend\":\"qk_candidate_slot_write_attention\","
    "\"candidate_record_compaction_backend\":\"qk_prefix_rank_slot_write_attention\","
    "\"strict_qk_layer_split_remaining\":\"terminal_check_state_king_scan,castle_target_chess_control_flow,legal_filter_batch_attack_chess_control_flow,legal_filter_batch_ray_scan_control_flow\","
    "\"remaining_non_attention_paths\":\"terminal_check_state_king_scan,castle_target_chess_control_flow,legal_filter_batch_attack_chess_control_flow,legal_filter_batch_ray_scan_control_flow\","
    "\"monolithic_custom_cuda_rule_kernels_allowed\":false,"
    "\"monolithic_custom_cuda_rule_kernels_remaining\":true,"
    "\"legal_filter_v1_monolithic_cuda_kernel_deprecated\":true,"
    "\"legal_filter_v2_target\":\"stack_of_frozen_2d_self_attention_layers\","
    "\"legal_filter_v2_required_backend\":\"cutlass_qk_scores_hardmax_v_write\","
    "\"legal_filter_v2_required_layers\":\"move_type_select,board_write_select,en_passant_capture_select,castle_rook_write_select,promotion_select,king_square_select,attack_source_select,ray_blocker_select,final_legal_select\"}";

struct Board {
    std::array<char, 64> squares{};
    bool white_to_move = true;
    int castling = 0;
    int ep_square = -1;
    int halfmove_clock = 0;
    int fullmove_number = 1;
};

struct Move {
    int from = -1;
    int to = -1;
    char promotion = kEmpty;
    bool en_passant = false;
    bool castle = false;
};

bool is_attacked(const Board& board, int target, bool by_white);

struct Point2D {
    float x = 0.0F;
    float y = 0.0F;
    uint32_t original_index = 0;
};

struct HullSupport {
    uint32_t original_index = 0;
    float score = 0.0F;
};

struct PolicyMoveSelectionNative {
    Move move;
    uint32_t legal_count = 0;
    uint32_t selected_index = 0;
    float score = 0.0F;
};

HullSupport hull_support_2d(CmzEngine* engine, const float* keys_xy, size_t key_count, float query_x, float query_y);
torch::Device required_cuda_device();
void ensure_decoder_initialized(CmzEngine* engine);

bool is_white_piece(char piece) {
    return piece >= 'A' && piece <= 'Z';
}

bool is_black_piece(char piece) {
    return piece >= 'a' && piece <= 'z';
}

bool is_side_piece(char piece, bool white) {
    return white ? is_white_piece(piece) : is_black_piece(piece);
}

bool is_enemy_piece(char piece, bool white) {
    return white ? is_black_piece(piece) : is_white_piece(piece);
}

int file_of(int square) {
    return square & 7;
}

int rank_of(int square) {
    return square >> 3;
}

int square_index(int file, int rank) {
    return rank * 8 + file;
}

bool on_board(int file, int rank) {
    return file >= 0 && file < 8 && rank >= 0 && rank < 8;
}

std::string square_name(int square) {
    std::string out;
    out.push_back(static_cast<char>('a' + file_of(square)));
    out.push_back(static_cast<char>('1' + rank_of(square)));
    return out;
}

int parse_square(const std::string& text) {
    if (text.size() != 2 || text[0] < 'a' || text[0] > 'h' || text[1] < '1' || text[1] > '8') {
        throw std::runtime_error("invalid FEN en-passant square");
    }
    return square_index(text[0] - 'a', text[1] - '1');
}

std::string move_to_uci(const Move& move) {
    std::string out = square_name(move.from) + square_name(move.to);
    if (move.promotion != kEmpty) {
        out.push_back(move.promotion);
    }
    return out;
}

uint32_t promotion_id(char promotion) {
    if (promotion == kEmpty) {
        return 0;
    }
    if (promotion == 'n') {
        return 1;
    }
    if (promotion == 'b') {
        return 2;
    }
    if (promotion == 'r') {
        return 3;
    }
    if (promotion == 'q') {
        return 4;
    }
    throw std::runtime_error("invalid native promotion code");
}

uint32_t move_id(const Move& move) {
    return static_cast<uint32_t>(move.from) * kSquareCount * kPromoBase +
           static_cast<uint32_t>(move.to) * kPromoBase + promotion_id(move.promotion);
}

char promotion_from_id(uint32_t promotion) {
    if (promotion == 0) {
        return kEmpty;
    }
    if (promotion == 1) {
        return 'n';
    }
    if (promotion == 2) {
        return 'b';
    }
    if (promotion == 3) {
        return 'r';
    }
    if (promotion == 4) {
        return 'q';
    }
    throw std::runtime_error("invalid UCI promotion id");
}

uint32_t piece_token(char piece) {
    switch (piece) {
        case kEmpty:
            return 0;
        case 'P':
            return 1;
        case 'N':
            return 2;
        case 'B':
            return 3;
        case 'R':
            return 4;
        case 'Q':
            return 5;
        case 'K':
            return 6;
        case 'p':
            return 7;
        case 'n':
            return 8;
        case 'b':
            return 9;
        case 'r':
            return 10;
        case 'q':
            return 11;
        case 'k':
            return 12;
        default:
            throw std::runtime_error("invalid piece token request");
    }
}

char piece_from_token(uint32_t token) {
    switch (token) {
        case 0:
            return kEmpty;
        case 1:
            return 'P';
        case 2:
            return 'N';
        case 3:
            return 'B';
        case 4:
            return 'R';
        case 5:
            return 'Q';
        case 6:
            return 'K';
        case 7:
            return 'p';
        case 8:
            return 'n';
        case 9:
            return 'b';
        case 10:
            return 'r';
        case 11:
            return 'q';
        case 12:
            return 'k';
        default:
            throw std::runtime_error("invalid piece token decode request");
    }
}

bool frozen_piece_token_matches_side(uint32_t token, bool white) {
    return white ? (token >= 1 && token <= 6) : (token >= 7 && token <= 12);
}

bool frozen_piece_token_is_leaper_or_pawn(uint32_t token) {
    return token == 1 || token == 2 || token == 6 || token == 7 || token == 8 || token == 12;
}

void add_mask_square(uint64_t& mask, int file, int rank) {
    if (on_board(file, rank)) {
        mask |= 1ULL << static_cast<uint32_t>(square_index(file, rank));
    }
}

void add_mask_ray(uint64_t& mask, int file, int rank, int df, int dr) {
    int next_file = file + df;
    int next_rank = rank + dr;
    while (on_board(next_file, next_rank)) {
        mask |= 1ULL << static_cast<uint32_t>(square_index(next_file, next_rank));
        next_file += df;
        next_rank += dr;
    }
}

uint64_t frozen_piece_attack_mask_table(uint32_t token, uint32_t from_square) {
    if (from_square >= 64) {
        throw std::runtime_error("frozen attack mask square out of range");
    }
    const int file = file_of(static_cast<int>(from_square));
    const int rank = rank_of(static_cast<int>(from_square));
    uint64_t mask = 0;
    switch (token) {
        case 1:
            add_mask_square(mask, file - 1, rank + 1);
            add_mask_square(mask, file + 1, rank + 1);
            break;
        case 7:
            add_mask_square(mask, file - 1, rank - 1);
            add_mask_square(mask, file + 1, rank - 1);
            break;
        case 2:
        case 8:
            for (const auto& offset : std::array<std::array<int, 2>, 8>{
                     {{{1, 2}}, {{2, 1}}, {{2, -1}}, {{1, -2}}, {{-1, -2}}, {{-2, -1}}, {{-2, 1}}, {{-1, 2}}}}) {
                add_mask_square(mask, file + offset[0], rank + offset[1]);
            }
            break;
        case 3:
        case 9:
            for (const auto& direction : std::array<std::array<int, 2>, 4>{
                     {{{1, 1}}, {{1, -1}}, {{-1, 1}}, {{-1, -1}}}}) {
                add_mask_ray(mask, file, rank, direction[0], direction[1]);
            }
            break;
        case 4:
        case 10:
            for (const auto& direction : std::array<std::array<int, 2>, 4>{
                     {{{1, 0}}, {{-1, 0}}, {{0, 1}}, {{0, -1}}}}) {
                add_mask_ray(mask, file, rank, direction[0], direction[1]);
            }
            break;
        case 5:
        case 11:
            for (const auto& direction : std::array<std::array<int, 2>, 8>{
                     {{{1, 1}}, {{1, -1}}, {{-1, 1}}, {{-1, -1}}, {{1, 0}}, {{-1, 0}}, {{0, 1}}, {{0, -1}}}}) {
                add_mask_ray(mask, file, rank, direction[0], direction[1]);
            }
            break;
        case 6:
        case 12:
            for (int df = -1; df <= 1; ++df) {
                for (int dr = -1; dr <= 1; ++dr) {
                    if (df != 0 || dr != 0) {
                        add_mask_square(mask, file + df, rank + dr);
                    }
                }
            }
            break;
        default:
            throw std::runtime_error("frozen attack mask piece token out of range");
    }
    return mask;
}

constexpr size_t kAttackTableTokenCount = 13;
constexpr size_t kAttackTableEntryCount = kAttackTableTokenCount * kSquareCount;

std::array<uint32_t, kAttackTableEntryCount> frozen_attack_table_attention_keys() {
    std::array<uint32_t, kAttackTableEntryCount> keys{};
    for (uint32_t token = 0; token < kAttackTableTokenCount; ++token) {
        for (uint32_t square = 0; square < kSquareCount; ++square) {
            keys[token * kSquareCount + square] = token * kSquareCount + square;
        }
    }
    return keys;
}

std::array<uint64_t, kAttackTableEntryCount> frozen_attack_table_attention_values() {
    std::array<uint64_t, kAttackTableEntryCount> values{};
    values.fill(0ULL);
    for (uint32_t token = 1; token < kAttackTableTokenCount; ++token) {
        for (uint32_t square = 0; square < kSquareCount; ++square) {
            values[token * kSquareCount + square] = frozen_piece_attack_mask_table(token, square);
        }
    }
    return values;
}

uint64_t frozen_attack_table_attention_lookup(CmzEngine* engine, uint32_t token, uint32_t from_square) {
    if (token == 0 || token >= kAttackTableTokenCount) {
        throw std::runtime_error("frozen attack mask piece token out of range");
    }
    if (from_square >= kSquareCount) {
        throw std::runtime_error("frozen attack mask square out of range");
    }
    static const std::array<uint32_t, kAttackTableEntryCount> keys = frozen_attack_table_attention_keys();
    static const std::array<uint64_t, kAttackTableEntryCount> values = frozen_attack_table_attention_values();
    uint64_t selected = 0ULL;
    const uint32_t query_key = token * kSquareCount + from_square;
    const int status = cmz_cuda_attack_table_lookup_attention(
        keys.data(),
        values.data(),
        values.size(),
        query_key,
        &selected);
    if (status != 0) {
        throw std::runtime_error(std::string("CUDA attack table attention lookup failed: ") +
                                 cudaGetErrorString(static_cast<cudaError_t>(status)));
    }
    if (engine != nullptr) {
        ++engine->cuda_attack_table_attention_count;
    }
    return selected;
}

bool valid_ray_delta(int delta_file, int delta_rank) {
    if (delta_file == 0 && delta_rank == 0) {
        return false;
    }
    return delta_file >= -1 && delta_file <= 1 && delta_rank >= -1 && delta_rank <= 1;
}

uint64_t frozen_ray_scan_mask_attention(
    CmzEngine* engine,
    uint32_t from_square,
    int delta_file,
    int delta_rank,
    uint64_t occupancy_mask) {
    if (from_square >= 64) {
        throw std::runtime_error("frozen ray scan square out of range");
    }
    if (!valid_ray_delta(delta_file, delta_rank)) {
        throw std::runtime_error("frozen ray scan direction must be one king-step direction");
    }

    uint64_t mask = 0;
    const int status = cmz_cuda_ray_scan_attention(
        from_square, static_cast<int32_t>(delta_file), static_cast<int32_t>(delta_rank), occupancy_mask, &mask);
    if (status != 0) {
        throw std::runtime_error(std::string("CUDA ray-scan nearest-blocker attention failed: ") +
                                 cudaGetErrorString(static_cast<cudaError_t>(status)));
    }
    if (engine != nullptr) {
        ++engine->cuda_ray_scan_attention_count;
    }
    return mask;
}

uint64_t frozen_ray_scan_mask_layer(uint32_t from_square, int delta_file, int delta_rank, uint64_t occupancy_mask) {
    return frozen_ray_scan_mask_attention(nullptr, from_square, delta_file, delta_rank, occupancy_mask);
}

uint64_t board_occupancy_mask(const Board& board) {
    uint64_t mask = 0;
    for (uint32_t square = 0; square < 64; ++square) {
        if (board.squares[square] != kEmpty) {
            mask |= 1ULL << square;
        }
    }
    return mask;
}

uint64_t side_occupancy_mask(const Board& board, bool white) {
    uint64_t mask = 0;
    for (uint32_t square = 0; square < 64; ++square) {
        if (is_side_piece(board.squares[square], white)) {
            mask |= 1ULL << square;
        }
    }
    return mask;
}

std::array<uint32_t, 64> board_piece_tokens(const Board& board) {
    std::array<uint32_t, 64> tokens{};
    for (size_t square = 0; square < tokens.size(); ++square) {
        tokens[square] = piece_token(board.squares[square]);
    }
    return tokens;
}

uint32_t promotion_piece_token(bool white, char promotion) {
    if (promotion == kEmpty) {
        return 0;
    }
    const char piece = white ? static_cast<char>(std::toupper(static_cast<unsigned char>(promotion))) : promotion;
    return piece_token(piece);
}

int single_bit_square(uint64_t mask) {
    for (int square = 0; square < 64; ++square) {
        if ((mask & (1ULL << static_cast<uint32_t>(square))) != 0) {
            return square;
        }
    }
    return -1;
}

uint64_t slider_candidate_mask(uint32_t token, uint32_t from_square, uint64_t friendly_mask, uint64_t occupancy_mask) {
    constexpr int bishop_dirs[4][2] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
    constexpr int rook_dirs[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    constexpr int queen_dirs[8][2] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}, {1, 0}, {-1, 0}, {0, 1}, {0, -1}};

    const int(*directions)[2] = nullptr;
    int direction_count = 0;
    if (token == 3 || token == 9) {
        directions = bishop_dirs;
        direction_count = 4;
    } else if (token == 4 || token == 10) {
        directions = rook_dirs;
        direction_count = 4;
    } else if (token == 5 || token == 11) {
        directions = queen_dirs;
        direction_count = 8;
    } else {
        throw std::runtime_error("slider candidate mask requires bishop, rook, or queen token");
    }

    uint64_t mask = 0;
    for (int index = 0; index < direction_count; ++index) {
        mask |= frozen_ray_scan_mask_layer(
            from_square, directions[index][0], directions[index][1], occupancy_mask) &
                ~friendly_mask;
    }
    return mask;
}

uint64_t pawn_candidate_mask(
    uint32_t token,
    uint32_t from_square,
    uint64_t enemy_mask,
    uint64_t occupancy_mask,
    uint32_t ep_square) {
    const bool white = token == 1;
    if (token != 1 && token != 7) {
        throw std::runtime_error("pawn candidate mask requires pawn token");
    }
    const int file = file_of(static_cast<int>(from_square));
    const int rank = rank_of(static_cast<int>(from_square));
    const int direction = white ? 1 : -1;
    const int start_rank = white ? 1 : 6;
    uint64_t mask = 0;

    const int one_rank = rank + direction;
    if (on_board(file, one_rank)) {
        const uint32_t one = static_cast<uint32_t>(square_index(file, one_rank));
        const uint64_t one_mask = 1ULL << one;
        if ((occupancy_mask & one_mask) == 0) {
            mask |= one_mask;
            const int two_rank = rank + 2 * direction;
            if (rank == start_rank && on_board(file, two_rank)) {
                const uint32_t two = static_cast<uint32_t>(square_index(file, two_rank));
                const uint64_t two_mask = 1ULL << two;
                if ((occupancy_mask & two_mask) == 0) {
                    mask |= two_mask;
                }
            }
        }
    }

    const uint64_t attack_mask = frozen_piece_attack_mask_table(token, from_square);
    mask |= attack_mask & enemy_mask;
    if (ep_square < 64 && (attack_mask & (1ULL << ep_square)) != 0) {
        const int captured = white ? static_cast<int>(ep_square) - 8 : static_cast<int>(ep_square) + 8;
        if (captured >= 0 && captured < 64 && (enemy_mask & (1ULL << static_cast<uint32_t>(captured))) != 0) {
            mask |= 1ULL << ep_square;
        }
    }
    return mask;
}

uint64_t frozen_candidate_target_mask_attention(
    CmzEngine* engine,
    uint32_t token,
    uint32_t from_square,
    uint64_t friendly_mask,
    uint64_t enemy_mask,
    uint64_t occupancy_mask,
    uint32_t ep_square) {
    if (from_square >= 64) {
        throw std::runtime_error("frozen candidate target square out of range");
    }
    if (token < 1 || token > 12) {
        throw std::runtime_error("frozen candidate target piece token out of range");
    }
    if ((friendly_mask & enemy_mask) != 0) {
        throw std::runtime_error("frozen candidate target friendly and enemy masks overlap");
    }
    uint64_t mask = 0;
    const int status = cmz_cuda_candidate_target_attention(
        token, from_square, friendly_mask, enemy_mask, occupancy_mask, ep_square, &mask);
    if (status != 0) {
        throw std::runtime_error(std::string("CUDA candidate-target attention failed: ") +
                                 cudaGetErrorString(static_cast<cudaError_t>(status)));
    }
    if (engine != nullptr) {
        ++engine->cuda_candidate_table_attention_count;
    }
    return mask;
}

uint64_t frozen_candidate_target_mask_layer(
    uint32_t token,
    uint32_t from_square,
    uint64_t friendly_mask,
    uint64_t enemy_mask,
    uint64_t occupancy_mask,
    uint32_t ep_square) {
    return frozen_candidate_target_mask_attention(
        nullptr, token, from_square, friendly_mask, enemy_mask, occupancy_mask, ep_square);
}

uint64_t frozen_castle_target_mask_attention(CmzEngine* engine, const Board& board, bool white) {
    const auto tokens = board_piece_tokens(board);
    uint64_t mask = 0;
    const int status =
        cmz_cuda_castle_target_attention(tokens.data(), static_cast<uint32_t>(board.castling), white ? 1U : 0U, &mask);
    if (status != 0) {
        throw std::runtime_error(std::string("CUDA castle-path attention failed: ") +
                                 cudaGetErrorString(static_cast<cudaError_t>(status)));
    }
    if (engine != nullptr) {
        ++engine->cuda_castle_attention_count;
    }
    return mask;
}

uint64_t frozen_castle_target_mask_layer(const Board& board, bool white) {
    return frozen_castle_target_mask_attention(nullptr, board, white);
}

Move parse_uci_move(const std::string& uci) {
    if (uci.size() != 4 && uci.size() != 5) {
        throw std::runtime_error("invalid UCI move length");
    }
    Move move;
    move.from = parse_square(uci.substr(0, 2));
    move.to = parse_square(uci.substr(2, 2));
    move.promotion = uci.size() == 5 ? promotion_from_id(promotion_id(uci[4])) : kEmpty;
    return move;
}

std::vector<std::string> split_spaces(const char* text) {
    std::vector<std::string> parts;
    std::string current;
    for (const char* cursor = text; *cursor != '\0'; ++cursor) {
        if (*cursor == ' ') {
            if (!current.empty()) {
                parts.push_back(current);
                current.clear();
            }
        } else {
            current.push_back(*cursor);
        }
    }
    if (!current.empty()) {
        parts.push_back(current);
    }
    return parts;
}

Board parse_fen(const char* fen) {
    if (fen == nullptr) {
        throw std::runtime_error("null FEN pointer");
    }
    const std::vector<std::string> parts = split_spaces(fen);
    if (parts.size() < 4) {
        throw std::runtime_error("FEN requires at least four fields");
    }

    Board board;
    board.squares.fill(kEmpty);
    int rank = 7;
    int file = 0;
    for (char token : parts[0]) {
        if (token == '/') {
            if (file != 8) {
                throw std::runtime_error("invalid FEN board row width");
            }
            --rank;
            file = 0;
            continue;
        }
        if (token >= '1' && token <= '8') {
            file += token - '0';
            if (file > 8) {
                throw std::runtime_error("invalid FEN board row overflow");
            }
            continue;
        }
        const std::string pieces = "PNBRQKpnbrqk";
        if (pieces.find(token) == std::string::npos || !on_board(file, rank)) {
            throw std::runtime_error("invalid FEN board piece");
        }
        board.squares[square_index(file, rank)] = token;
        ++file;
    }
    if (rank != 0 || file != 8) {
        throw std::runtime_error("invalid FEN board shape");
    }

    if (parts[1] == "w") {
        board.white_to_move = true;
    } else if (parts[1] == "b") {
        board.white_to_move = false;
    } else {
        throw std::runtime_error("invalid FEN side to move");
    }

    if (parts[2] != "-") {
        for (char right : parts[2]) {
            if (right == 'K') {
                board.castling |= kCastleWhiteKing;
            } else if (right == 'Q') {
                board.castling |= kCastleWhiteQueen;
            } else if (right == 'k') {
                board.castling |= kCastleBlackKing;
            } else if (right == 'q') {
                board.castling |= kCastleBlackQueen;
            } else {
                throw std::runtime_error("invalid FEN castling rights");
            }
        }
    }
    board.ep_square = parts[3] == "-" ? -1 : parse_square(parts[3]);
    if (parts.size() >= 5) {
        board.halfmove_clock = std::stoi(parts[4]);
    }
    if (parts.size() >= 6) {
        board.fullmove_number = std::stoi(parts[5]);
    }
    return board;
}

bool ray_attacks(const Board& board, int target, bool by_white, const int directions[][2], int direction_count, char slider_a, char slider_b) {
    const uint64_t occupancy = board_occupancy_mask(board);
    for (int index = 0; index < direction_count; ++index) {
        const uint64_t scan_mask = frozen_ray_scan_mask_layer(
            static_cast<uint32_t>(target), directions[index][0], directions[index][1], occupancy);
        const int blocker = single_bit_square(scan_mask & occupancy);
        if (blocker >= 0) {
            const char piece = board.squares[blocker];
            if (is_side_piece(piece, by_white)) {
                const char lower = static_cast<char>(std::tolower(static_cast<unsigned char>(piece)));
                if (lower == slider_a || lower == slider_b) {
                    return true;
                }
            }
        }
    }
    return false;
}

bool is_attacked(const Board& board, int target, bool by_white) {
    const uint64_t target_mask = 1ULL << static_cast<uint32_t>(target);
    for (int square = 0; square < 64; ++square) {
        const char piece = board.squares[square];
        if (piece == kEmpty || !is_side_piece(piece, by_white)) {
            continue;
        }
        const uint32_t token = piece_token(piece);
        if (frozen_piece_token_matches_side(token, by_white) && frozen_piece_token_is_leaper_or_pawn(token) &&
            (frozen_piece_attack_mask_table(token, static_cast<uint32_t>(square)) & target_mask) != 0) {
            return true;
        }
    }

    constexpr int diagonal_dirs[4][2] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
    constexpr int straight_dirs[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    if (ray_attacks(board, target, by_white, diagonal_dirs, 4, 'b', 'q')) {
        return true;
    }
    if (ray_attacks(board, target, by_white, straight_dirs, 4, 'r', 'q')) {
        return true;
    }
    return false;
}

int king_square(const Board& board, bool white) {
    const char king = white ? 'K' : 'k';
    for (int square = 0; square < 64; ++square) {
        if (board.squares[square] == king) {
            return square;
        }
    }
    return -1;
}

void add_promotion_moves(std::vector<Move>& moves, int from, int to, bool en_passant = false) {
    for (char promotion : {'q', 'r', 'b', 'n'}) {
        moves.push_back(Move{from, to, promotion, en_passant, false});
    }
}

void append_mask_targets(std::vector<Move>& moves, int from, uint64_t target_mask) {
    for (int target = 0; target < 64; ++target) {
        if ((target_mask & (1ULL << static_cast<uint32_t>(target))) != 0) {
            moves.push_back(Move{from, target});
        }
    }
}

void add_pawn_moves(const Board& board, int from, bool white, std::vector<Move>& moves) {
    const int rank = rank_of(from);
    const int promotion_from_rank = white ? 6 : 1;
    const uint32_t token = white ? 1 : 7;
    const uint64_t friendly = side_occupancy_mask(board, white);
    const uint64_t enemy = side_occupancy_mask(board, !white);
    const uint64_t occupancy = friendly | enemy;
    const uint64_t target_mask = frozen_candidate_target_mask_layer(
        token,
        static_cast<uint32_t>(from),
        friendly,
        enemy,
        occupancy,
        board.ep_square >= 0 ? static_cast<uint32_t>(board.ep_square) : 64U);

    for (int target = 0; target < 64; ++target) {
        if ((target_mask & (1ULL << static_cast<uint32_t>(target))) == 0) {
            continue;
        }
        if (target == board.ep_square && board.squares[target] == kEmpty) {
            moves.push_back(Move{from, target, kEmpty, true, false});
        } else if (rank == promotion_from_rank) {
            add_promotion_moves(moves, from, target);
        } else {
            moves.push_back(Move{from, target});
        }
    }
}

void add_knight_moves(const Board& board, int from, bool white, std::vector<Move>& moves) {
    const uint64_t friendly = side_occupancy_mask(board, white);
    const uint64_t enemy = side_occupancy_mask(board, !white);
    const uint64_t occupancy = friendly | enemy;
    const uint32_t token = white ? 2 : 8;
    append_mask_targets(
        moves,
        from,
        frozen_candidate_target_mask_layer(token, static_cast<uint32_t>(from), friendly, enemy, occupancy, 64));
}

void add_slider_moves(const Board& board, int from, bool white, uint32_t token, std::vector<Move>& moves) {
    const uint64_t friendly = side_occupancy_mask(board, white);
    const uint64_t enemy = side_occupancy_mask(board, !white);
    const uint64_t occupancy = friendly | enemy;
    append_mask_targets(
        moves,
        from,
        frozen_candidate_target_mask_layer(token, static_cast<uint32_t>(from), friendly, enemy, occupancy, 64));
}

void add_castles(const Board& board, bool white, std::vector<Move>& moves) {
    const uint64_t castle_mask = frozen_castle_target_mask_layer(board, white);
    if ((castle_mask & (1ULL << 6U)) != 0) {
        moves.push_back(Move{4, 6, kEmpty, false, true});
    }
    if ((castle_mask & (1ULL << 2U)) != 0) {
        moves.push_back(Move{4, 2, kEmpty, false, true});
    }
    if ((castle_mask & (1ULL << 62U)) != 0) {
        moves.push_back(Move{60, 62, kEmpty, false, true});
    }
    if ((castle_mask & (1ULL << 58U)) != 0) {
        moves.push_back(Move{60, 58, kEmpty, false, true});
    }
}

void add_king_moves(const Board& board, int from, bool white, std::vector<Move>& moves) {
    const uint64_t friendly = side_occupancy_mask(board, white);
    const uint64_t enemy = side_occupancy_mask(board, !white);
    const uint64_t occupancy = friendly | enemy;
    const uint32_t token = white ? 6 : 12;
    append_mask_targets(
        moves,
        from,
        frozen_candidate_target_mask_layer(token, static_cast<uint32_t>(from), friendly, enemy, occupancy, 64));
    add_castles(board, white, moves);
}

std::vector<Move> pseudo_legal_moves(CmzEngine* engine, const Board& board) {
    const auto board_tokens = board_piece_tokens(board);
    std::array<uint32_t, kCandidateMoveRecordCapacity * kCandidateMoveRecordWidth> records{};
    uint32_t record_count = 0;
    uint32_t layer_count = 0;
    const int status = cmz_cuda_candidate_moves_attention(
        board_tokens.data(),
        board.white_to_move ? 1U : 0U,
        static_cast<uint32_t>(board.castling),
        board.ep_square >= 0 ? static_cast<uint32_t>(board.ep_square) : kNoEp,
        records.data(),
        kCandidateMoveRecordCapacity,
        &record_count,
        &layer_count);
    if (status != 0) {
        throw std::runtime_error(std::string("CUDA candidate-moves attention failed: ") +
                                 cudaGetErrorString(static_cast<cudaError_t>(status)));
    }
    if (record_count > kCandidateMoveRecordCapacity) {
        throw std::runtime_error("CUDA candidate-moves attention exceeded record capacity");
    }
    if (engine != nullptr) {
        ++engine->cuda_candidate_move_attention_count;
        engine->cuda_candidate_move_layer_count += layer_count;
        engine->frozen_layer_step_count += layer_count;
    }
    std::vector<Move> moves;
    moves.reserve(record_count);
    for (uint32_t index = 0; index < record_count; ++index) {
        const size_t base = static_cast<size_t>(index) * kCandidateMoveRecordWidth;
        moves.push_back(Move{
            static_cast<int>(records[base]),
            static_cast<int>(records[base + 1]),
            promotion_from_id(records[base + 2]),
            records[base + 3] != 0U,
            records[base + 4] != 0U});
    }
    return moves;
}

std::vector<Move> pseudo_legal_moves(const Board& board) {
    return pseudo_legal_moves(nullptr, board);
}

Board frozen_make_move_board_layer(CmzEngine* engine, const Board& board, const Move& move) {
    Board next = board;
    const bool white = board.white_to_move;
    const char piece = board.squares[move.from];
    const char target_piece = board.squares[move.to];
    const auto board_tokens = board_piece_tokens(board);
    std::array<uint32_t, 64> next_tokens{};
    uint32_t layer_count = 0;
    const int status = cmz_cuda_make_move_board_attention(
        board_tokens.data(),
        white ? 1U : 0U,
        static_cast<uint32_t>(move.from),
        static_cast<uint32_t>(move.to),
        promotion_piece_token(white, move.promotion),
        move.en_passant ? 1U : 0U,
        move.castle ? 1U : 0U,
        next_tokens.data(),
        &layer_count);
    if (status != 0) {
        throw std::runtime_error(std::string("CUDA make-move board self-attention failed: ") +
                                 cudaGetErrorString(static_cast<cudaError_t>(status)));
    }
    if (engine != nullptr) {
        ++engine->cuda_make_move_board_attention_count;
        engine->frozen_layer_step_count += layer_count;
    }
    for (size_t square = 0; square < next.squares.size(); ++square) {
        next.squares[square] = piece_from_token(next_tokens[square]);
    }

    std::array<uint32_t, 5> next_metadata{};
    layer_count = 0;
    const int metadata_status = cmz_cuda_make_move_metadata_attention(
        white ? 1U : 0U,
        static_cast<uint32_t>(board.castling),
        static_cast<uint32_t>(board.halfmove_clock),
        static_cast<uint32_t>(board.fullmove_number),
        piece_token(piece),
        piece_token(target_piece),
        static_cast<uint32_t>(move.from),
        static_cast<uint32_t>(move.to),
        move.en_passant ? 1U : 0U,
        next_metadata.data(),
        &layer_count);
    if (metadata_status != 0) {
        throw std::runtime_error(std::string("CUDA make-move metadata self-attention failed: ") +
                                 cudaGetErrorString(static_cast<cudaError_t>(metadata_status)));
    }
    if (engine != nullptr) {
        ++engine->cuda_make_move_metadata_attention_count;
        engine->frozen_layer_step_count += layer_count;
    }
    next.white_to_move = next_metadata[0] != 0U;
    next.castling = static_cast<int>(next_metadata[1]);
    next.ep_square = next_metadata[2] == kNoEp ? -1 : static_cast<int>(next_metadata[2]);
    next.halfmove_clock = static_cast<int>(next_metadata[3]);
    next.fullmove_number = static_cast<int>(next_metadata[4]);
    return next;
}

bool frozen_legal_filter_attention(CmzEngine* engine, const Board& board, const Move& move) {
    const bool white = board.white_to_move;
    const auto tokens = board_piece_tokens(board);
    uint32_t legal = 0;
    uint32_t layer_count = 0;
    const int status = cmz_cuda_legal_filter_v2_attention(
        tokens.data(),
        white ? 1U : 0U,
        static_cast<uint32_t>(move.from),
        static_cast<uint32_t>(move.to),
        promotion_piece_token(white, move.promotion),
        move.en_passant ? 1U : 0U,
        move.castle ? 1U : 0U,
        &legal,
        &layer_count);
    if (status != 0) {
        throw std::runtime_error(std::string("CUDA legal-filter v2 layered self-attention failed: ") +
                                 cudaGetErrorString(static_cast<cudaError_t>(status)));
    }
    if (engine != nullptr) {
        ++engine->cuda_legal_filter_v2_attention_count;
        engine->cuda_legal_filter_v2_layer_count += layer_count;
    }
    return legal != 0;
}

std::vector<uint32_t> frozen_legal_filter_batch_attention(
    CmzEngine* engine,
    const Board& board,
    const std::vector<Move>& moves) {
    std::vector<uint32_t> legal_outputs(moves.size(), 0U);
    if (moves.empty()) {
        return legal_outputs;
    }

    const bool white = board.white_to_move;
    const auto board_tokens = board_piece_tokens(board);
    std::vector<uint32_t> from_squares;
    std::vector<uint32_t> to_squares;
    std::vector<uint32_t> promotion_tokens;
    std::vector<uint32_t> en_passant;
    std::vector<uint32_t> castle;
    from_squares.reserve(moves.size());
    to_squares.reserve(moves.size());
    promotion_tokens.reserve(moves.size());
    en_passant.reserve(moves.size());
    castle.reserve(moves.size());

    for (const Move& move : moves) {
        from_squares.push_back(static_cast<uint32_t>(move.from));
        to_squares.push_back(static_cast<uint32_t>(move.to));
        promotion_tokens.push_back(promotion_piece_token(white, move.promotion));
        en_passant.push_back(move.en_passant ? 1U : 0U);
        castle.push_back(move.castle ? 1U : 0U);
    }

    uint32_t layer_count = 0;
    const int status = cmz_cuda_legal_filter_v2_batch_attention(
        board_tokens.data(),
        white ? 1U : 0U,
        from_squares.data(),
        to_squares.data(),
        promotion_tokens.data(),
        en_passant.data(),
        castle.data(),
        moves.size(),
        legal_outputs.data(),
        &layer_count);
    if (status != 0) {
        throw std::runtime_error(std::string("CUDA batched legal-filter v2 layered self-attention failed: ") +
                                 cudaGetErrorString(static_cast<cudaError_t>(status)));
    }
    if (engine != nullptr) {
        ++engine->cuda_legal_filter_v2_batch_attention_count;
        engine->cuda_legal_filter_v2_batch_layer_count += layer_count;
    }
    return legal_outputs;
}

bool frozen_legal_filter_layer(const Board& board, const Move& move) {
    return frozen_legal_filter_attention(nullptr, board, move);
}

bool legal_after_king_filter(const Board& board, const Move& move) {
    return frozen_legal_filter_layer(board, move);
}

bool frozen_candidate_move_from_request(const Board& board, const Move& requested, Move* out_move) {
    if (requested.from < 0 || requested.from >= 64 || requested.to < 0 || requested.to >= 64) {
        return false;
    }
    const char piece = board.squares[requested.from];
    const bool white = board.white_to_move;
    if (!is_side_piece(piece, white)) {
        return false;
    }

    const char lower = static_cast<char>(std::tolower(static_cast<unsigned char>(piece)));
    Move move = requested;
    move.en_passant = false;
    move.castle = false;

    if (lower == 'k' && std::abs(move.to - move.from) == 2) {
        if (move.promotion != kEmpty) {
            return false;
        }
        const uint64_t castle_mask = frozen_castle_target_mask_layer(board, white);
        if ((castle_mask & (1ULL << static_cast<uint32_t>(move.to))) == 0) {
            return false;
        }
        move.castle = true;
        *out_move = move;
        return true;
    }

    const uint32_t token = piece_token(piece);
    const uint64_t friendly = side_occupancy_mask(board, white);
    const uint64_t enemy = side_occupancy_mask(board, !white);
    const uint64_t occupancy = friendly | enemy;
    const uint64_t target_mask = frozen_candidate_target_mask_layer(
        token,
        static_cast<uint32_t>(move.from),
        friendly,
        enemy,
        occupancy,
        board.ep_square >= 0 ? static_cast<uint32_t>(board.ep_square) : 64U);
    if ((target_mask & (1ULL << static_cast<uint32_t>(move.to))) == 0) {
        return false;
    }

    if (lower == 'p') {
        const int promotion_rank = white ? 7 : 0;
        if (rank_of(move.to) == promotion_rank) {
            if (move.promotion == kEmpty) {
                return false;
            }
        } else if (move.promotion != kEmpty) {
            return false;
        }
        if (move.to == board.ep_square && board.squares[move.to] == kEmpty && file_of(move.to) != file_of(move.from)) {
            move.en_passant = true;
        }
    } else if (move.promotion != kEmpty) {
        return false;
    }

    *out_move = move;
    return true;
}

bool frozen_move_legal_attention(CmzEngine* engine, const Board& board, const std::string& uci) {
    Move move;
    if (!frozen_candidate_move_from_request(board, parse_uci_move(uci), &move)) {
        return false;
    }
    return frozen_legal_filter_attention(engine, board, move);
}

bool frozen_move_legal_layer(const Board& board, const std::string& uci) {
    return frozen_move_legal_attention(nullptr, board, uci);
}

std::vector<std::string> legal_moves_uci(CmzEngine* engine, const Board& board) {
    std::vector<std::string> out;
    const std::vector<Move> moves = pseudo_legal_moves(engine, board);
    const std::vector<uint32_t> legal_bits = frozen_legal_filter_batch_attention(engine, board, moves);
    for (size_t index = 0; index < moves.size(); ++index) {
        const Move& move = moves[index];
        if (legal_bits[index] != 0U) {
            out.push_back(move_to_uci(move));
        }
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

std::vector<std::string> legal_moves_uci(const Board& board) {
    return legal_moves_uci(nullptr, board);
}

std::vector<Move> legal_moves_native(CmzEngine* engine, const Board& board) {
    std::vector<Move> out;
    const std::vector<Move> moves = pseudo_legal_moves(engine, board);
    const std::vector<uint32_t> legal_bits = frozen_legal_filter_batch_attention(engine, board, moves);
    for (size_t index = 0; index < moves.size(); ++index) {
        if (legal_bits[index] != 0U) {
            out.push_back(moves[index]);
        }
    }
    std::sort(out.begin(), out.end(), [](const Move& left, const Move& right) {
        return move_to_uci(left) < move_to_uci(right);
    });
    out.erase(
        std::unique(out.begin(), out.end(), [](const Move& left, const Move& right) {
            return move_to_uci(left) == move_to_uci(right);
        }),
        out.end());
    return out;
}

PolicyMoveSelectionNative policy_select_move_native(CmzEngine* engine, const Board& board) {
    if (engine == nullptr) {
        throw std::runtime_error("policy move selection requires CmzEngine");
    }
    const std::vector<Move> legal_moves = legal_moves_native(engine, board);
    if (legal_moves.empty()) {
        throw std::runtime_error("policy move selection requires at least one legal move");
    }
    ensure_decoder_initialized(engine);
    const torch::Device device = required_cuda_device();
    std::vector<float> key_values;
    key_values.reserve(legal_moves.size() * 2U);
    for (const Move& move : legal_moves) {
        key_values.push_back(static_cast<float>(move.from) / 63.0F);
        key_values.push_back(
            static_cast<float>(move.to) / 63.0F + static_cast<float>(promotion_id(move.promotion)) / 256.0F);
    }
    const auto host_float = torch::TensorOptions().dtype(torch::kFloat32);
    const auto keys = torch::from_blob(
                          key_values.data(),
                          {static_cast<int64_t>(legal_moves.size()), 2},
                          host_float)
                          .clone()
                          .to(device);
    const auto query = engine->decoder_command_weight.select(0, 4);
    const auto scores = torch::matmul(keys, query);
    const auto selected_tensor = torch::argmax(scores, 0);
    const uint32_t selected_index = static_cast<uint32_t>(selected_tensor.item<int64_t>());
    if (selected_index >= legal_moves.size()) {
        throw std::runtime_error("policy decoder selected out-of-range legal move index");
    }
    const float selected_score = scores.index({static_cast<int64_t>(selected_index)}).item<float>();
    return PolicyMoveSelectionNative{
        legal_moves[selected_index],
        static_cast<uint32_t>(legal_moves.size()),
        selected_index,
        selected_score};
}

uint32_t move_flags(const Board& board, const Move& move) {
    uint32_t flags = 0;
    if (is_enemy_piece(board.squares[move.to], board.white_to_move) || move.en_passant) {
        flags |= kMoveFlagCapture;
    }
    if (move.en_passant) {
        flags |= kMoveFlagEp;
    }
    if (move.castle) {
        flags |= kMoveFlagCastle;
    }
    if (move.promotion != kEmpty) {
        flags |= kMoveFlagPromotion;
    }
    return flags;
}

std::vector<Move> sorted_pseudo_legal_moves(CmzEngine* engine, const Board& board) {
    return pseudo_legal_moves(engine, board);
}

std::vector<Move> sorted_pseudo_legal_moves(const Board& board) {
    return sorted_pseudo_legal_moves(nullptr, board);
}

void frozen_trace_packet_emit_attention_layer(
    CmzEngine* engine,
    std::vector<uint32_t>& out,
    uint32_t op,
    uint32_t a0,
    uint32_t a1,
    uint32_t a2,
    uint32_t a3,
    uint32_t tag,
    uint32_t commit) {
    std::array<uint32_t, kTracePacketWidth> packet{};
    const int status = cmz_cuda_emit_trace_packet_attention(op, a0, a1, a2, a3, tag, commit, packet.data());
    if (status != 0) {
        throw std::runtime_error(std::string("CUDA trace packet emit attention failed: ") +
                                 cudaGetErrorString(static_cast<cudaError_t>(status)));
    }
    out.insert(out.end(), packet.begin(), packet.end());
    if (engine != nullptr) {
        ++engine->cuda_trace_emit_attention_count;
        ++engine->frozen_layer_step_count;
    }
}

void frozen_attention_select_trace_packet(
    CmzEngine* engine,
    const std::vector<uint32_t>& memory_tokens,
    size_t query_index,
    uint32_t* out_packet_tokens) {
    const size_t packet_count = memory_tokens.size() / kTracePacketWidth;
    if (packet_count == 0) {
        throw std::runtime_error("trace-select memory is empty");
    }
    std::vector<float> hull_keys;
    hull_keys.reserve(packet_count * 2U);
    for (size_t index = 0; index < packet_count; ++index) {
        const float x = static_cast<float>(index);
        hull_keys.push_back(x);
        hull_keys.push_back(-(x * x));
    }
    const float query = static_cast<float>(query_index);
    const HullSupport support = hull_support_2d(engine, hull_keys.data(), packet_count, 2.0F * query, 1.0F);
    const int status =
        cmz_cuda_select_trace_packet(memory_tokens.data(), packet_count, support.original_index, out_packet_tokens);
    if (status != 0) {
        throw std::runtime_error("CUDA trace-select packet kernel failed with code " + std::to_string(status));
    }
    if (engine != nullptr) {
        ++engine->cuda_trace_select_count;
        ++engine->frozen_layer_step_count;
    }
}

std::vector<uint32_t> frozen_legal_trace_attention_tokens(CmzEngine* engine, const Board& board) {
    std::vector<uint32_t> tokens;
    const std::vector<Move> moves = sorted_pseudo_legal_moves(engine, board);
    const std::vector<uint32_t> legal_bits = frozen_legal_filter_batch_attention(engine, board, moves);
    tokens.reserve((moves.size() * 2 + 1) * kTracePacketWidth);
    for (size_t index = 0; index < moves.size(); ++index) {
        const Move& move = moves[index];
        const uint32_t id = move_id(move);
        const uint32_t promo = promotion_id(move.promotion);
        const uint32_t flags = move_flags(board, move);
        frozen_trace_packet_emit_attention_layer(
            engine,
            tokens,
            kTraceCandidate,
            id,
            static_cast<uint32_t>(move.from),
            static_cast<uint32_t>(move.to),
            promo,
            kTraceTagMove,
            flags);
        frozen_trace_packet_emit_attention_layer(
            engine,
            tokens,
            kTraceLegalSet,
            id,
            legal_bits[index] != 0U ? 1U : 0U,
            0,
            0,
            kTraceTagLegal,
            0);
    }
    frozen_trace_packet_emit_attention_layer(engine, tokens, kTraceProgramHalt, 0, 0, 0, 0, kTraceTagLegal, 1);
    return tokens;
}

std::vector<uint32_t> legal_trace_tokens(const Board& board) {
    return frozen_legal_trace_attention_tokens(nullptr, board);
}

void frozen_legal_trace_attention_packet(
    CmzEngine* engine,
    const Board& board,
    size_t query_index,
    uint32_t* out_packet_tokens) {
    if (out_packet_tokens == nullptr) {
        throw std::runtime_error("frozen legal trace packet output pointer is null");
    }
    const std::vector<Move> moves = sorted_pseudo_legal_moves(engine, board);
    const size_t packet_count = moves.size() * 2 + 1;
    if (query_index >= packet_count) {
        throw std::runtime_error("frozen legal trace packet cursor out of range");
    }

    std::vector<uint32_t> packet_memory;
    packet_memory.reserve(kTracePacketWidth);
    if (query_index == packet_count - 1) {
        frozen_trace_packet_emit_attention_layer(
            engine, packet_memory, kTraceProgramHalt, 0, 0, 0, 0, kTraceTagLegal, 1);
    } else {
        const size_t move_index = query_index / 2;
        const Move& move = moves[move_index];
        const uint32_t id = move_id(move);
        if ((query_index % 2) == 0) {
            frozen_trace_packet_emit_attention_layer(
                engine,
                packet_memory,
                kTraceCandidate,
                id,
                static_cast<uint32_t>(move.from),
                static_cast<uint32_t>(move.to),
                promotion_id(move.promotion),
                kTraceTagMove,
                move_flags(board, move));
        } else {
            const bool legal = frozen_legal_filter_attention(engine, board, move);
            frozen_trace_packet_emit_attention_layer(
                engine,
                packet_memory,
                kTraceLegalSet,
                id,
                legal ? 1U : 0U,
                0,
                0,
                kTraceTagLegal,
                0);
        }
    }
    frozen_attention_select_trace_packet(engine, packet_memory, 0, out_packet_tokens);
}

Move resolve_legal_move(CmzEngine* engine, const Board& board, const std::string& uci) {
    const Move requested = parse_uci_move(uci);
    const std::vector<Move> moves = sorted_pseudo_legal_moves(engine, board);
    const std::vector<uint32_t> legal_bits = frozen_legal_filter_batch_attention(engine, board, moves);
    std::vector<uint32_t> records(moves.size() * kCandidateMoveRecordWidth, 0U);
    for (size_t index = 0; index < moves.size(); ++index) {
        const size_t base = index * kCandidateMoveRecordWidth;
        records[base] = static_cast<uint32_t>(moves[index].from);
        records[base + 1] = static_cast<uint32_t>(moves[index].to);
        records[base + 2] = promotion_id(moves[index].promotion);
        records[base + 3] = moves[index].en_passant ? 1U : 0U;
        records[base + 4] = moves[index].castle ? 1U : 0U;
    }
    std::array<uint32_t, kCandidateMoveRecordWidth> selected{};
    uint32_t found = 0;
    const int status = cmz_cuda_resolve_move_attention(
        records.empty() ? nullptr : records.data(),
        legal_bits.empty() ? nullptr : legal_bits.data(),
        moves.size(),
        static_cast<uint32_t>(requested.from),
        static_cast<uint32_t>(requested.to),
        promotion_id(requested.promotion),
        selected.data(),
        &found);
    if (status != 0) {
        throw std::runtime_error(std::string("CUDA resolve-move attention failed: ") +
                                 cudaGetErrorString(static_cast<cudaError_t>(status)));
    }
    if (engine != nullptr) {
        ++engine->cuda_resolve_move_attention_count;
        ++engine->frozen_layer_step_count;
    }
    if (found != 0U) {
        return Move{
            static_cast<int>(selected[0]),
            static_cast<int>(selected[1]),
            promotion_from_id(selected[2]),
            selected[3] != 0U,
            selected[4] != 0U};
    }
    throw std::runtime_error("illegal native make-move request");
}

Move resolve_legal_move(const Board& board, const std::string& uci) {
    return resolve_legal_move(nullptr, board, uci);
}

bool insufficient_material(const Board& board) {
    int non_king_count = 0;
    int minor_count = 0;
    int bishop_count = 0;
    int knight_count = 0;
    int bishop_color_mask = 0;
    for (int square = 0; square < 64; ++square) {
        const char piece = board.squares[square];
        if (piece == kEmpty) {
            continue;
        }
        const char lower = static_cast<char>(std::tolower(static_cast<unsigned char>(piece)));
        if (lower == 'k') {
            continue;
        }
        ++non_king_count;
        if (lower == 'b') {
            ++minor_count;
            ++bishop_count;
            bishop_color_mask |= 1 << ((file_of(square) + rank_of(square)) & 1);
        } else if (lower == 'n') {
            ++minor_count;
            ++knight_count;
        } else {
            return false;
        }
    }
    if (non_king_count == 0) {
        return true;
    }
    if (non_king_count == 1 && minor_count == 1) {
        return true;
    }
    if (knight_count == 0 && bishop_count == non_king_count && (bishop_color_mask == 1 || bishop_color_mask == 2)) {
        return true;
    }
    return false;
}

std::array<uint32_t, 2> frozen_terminal_predicate_layer(
    CmzEngine* engine,
    const Board& board,
    int repetition_count,
    bool adjudication_cap_reached) {
    const auto board_tokens = board_piece_tokens(board);
    std::array<uint32_t, 2> result_reason{};
    uint32_t layer_count = 0;
    const int status = cmz_cuda_terminal_status_attention(
        board_tokens.data(),
        board.white_to_move ? 1U : 0U,
        static_cast<uint32_t>(board.castling),
        board.ep_square >= 0 ? static_cast<uint32_t>(board.ep_square) : kNoEp,
        static_cast<uint32_t>(board.halfmove_clock),
        static_cast<uint32_t>(repetition_count),
        adjudication_cap_reached ? 1U : 0U,
        result_reason.data(),
        &layer_count);
    if (status != 0) {
        throw std::runtime_error(std::string("CUDA terminal status attention failed: ") +
                                 cudaGetErrorString(static_cast<cudaError_t>(status)));
    }
    if (engine != nullptr) {
        ++engine->cuda_terminal_status_attention_count;
        engine->cuda_terminal_status_layer_count += layer_count;
        engine->frozen_layer_step_count += layer_count;
    }
    return result_reason;
}

std::array<uint32_t, 2> frozen_terminal_predicate_layer(
    const Board& board,
    int repetition_count,
    bool adjudication_cap_reached) {
    return frozen_terminal_predicate_layer(nullptr, board, repetition_count, adjudication_cap_reached);
}

void frozen_board_transition_emit_attention_layer(
    CmzEngine* engine,
    std::vector<uint32_t>& tokens,
    const Board& before,
    const Board& after,
    uint32_t ply) {
    for (int square = 0; square < 64; ++square) {
        if (before.squares[square] != after.squares[square]) {
            frozen_trace_packet_emit_attention_layer(
                engine,
                tokens,
                kTraceWriteSq,
                static_cast<uint32_t>(square),
                piece_token(after.squares[square]),
                ply,
                0,
                kTraceTagBoard,
                1);
        }
    }
    frozen_trace_packet_emit_attention_layer(
        engine, tokens, kTraceWriteReg, kRegSideToMove, after.white_to_move ? 0U : 1U, ply, 0, kTraceTagState, 1);
    frozen_trace_packet_emit_attention_layer(
        engine, tokens, kTraceWriteCastle, static_cast<uint32_t>(after.castling), ply, 0, 0, kTraceTagState, 1);
    frozen_trace_packet_emit_attention_layer(
        engine,
        tokens,
        kTraceWriteEp,
        after.ep_square < 0 ? kNoEp : static_cast<uint32_t>(after.ep_square),
        ply,
        0,
        0,
        kTraceTagState,
        1);
    frozen_trace_packet_emit_attention_layer(
        engine,
        tokens,
        kTraceWriteClock,
        static_cast<uint32_t>(after.halfmove_clock),
        static_cast<uint32_t>(after.fullmove_number),
        ply,
        0,
        kTraceTagState,
        1);
}

std::vector<uint32_t> frozen_make_move_trace_attention_tokens(
    CmzEngine* engine,
    const Board& board,
    const std::string& uci,
    uint32_t ply,
    int repetition_count,
    bool adjudication_cap_reached) {
    const Move move = resolve_legal_move(engine, board, uci);
    const Board after = frozen_make_move_board_layer(engine, board, move);
    const uint32_t next_ply = ply + 1;
    std::vector<uint32_t> tokens;
    tokens.reserve(16 * kTracePacketWidth);
    frozen_trace_packet_emit_attention_layer(
        engine,
        tokens,
        kTraceCommitMove,
        move_id(move),
        static_cast<uint32_t>(move.from),
        static_cast<uint32_t>(move.to),
        promotion_id(move.promotion),
        kTraceTagMove,
        move_flags(board, move));
    frozen_board_transition_emit_attention_layer(engine, tokens, board, after, next_ply);
    const auto terminal = frozen_terminal_predicate_layer(engine, after, repetition_count, adjudication_cap_reached);
    frozen_trace_packet_emit_attention_layer(
        engine,
        tokens,
        kTraceTerminalSet,
        terminal[0],
        terminal[1],
        next_ply,
        0,
        kTraceTagTerminal,
        terminal[0] == kResultOngoing ? 0U : 1U);
    frozen_trace_packet_emit_attention_layer(engine, tokens, kTraceProgramHalt, 0, 0, 0, 0, kTraceTagTerminal, 1);
    return tokens;
}

std::vector<uint32_t> make_move_trace_tokens(
    CmzEngine* engine,
    const Board& board,
    const std::string& uci,
    uint32_t ply,
    int repetition_count,
    bool adjudication_cap_reached) {
    return frozen_make_move_trace_attention_tokens(engine, board, uci, ply, repetition_count, adjudication_cap_reached);
}

std::string join_lines(const std::vector<std::string>& lines) {
    std::string out;
    for (const std::string& line : lines) {
        out += line;
        out.push_back('\n');
    }
    return out;
}

void write_output(const std::string& text, char* out, size_t out_len, size_t* written) {
    if (written != nullptr) {
        *written = text.size() + 1;
    }
    if (out == nullptr || out_len == 0) {
        return;
    }
    const size_t copy_len = std::min(out_len - 1, text.size());
    std::memcpy(out, text.data(), copy_len);
    out[copy_len] = '\0';
}

int set_error(CmzEngine* engine, const std::string& message) {
    if (engine != nullptr) {
        engine->last_error = message;
    }
    return 1;
}

float cross(const Point2D& origin, const Point2D& left, const Point2D& right) {
    return (left.x - origin.x) * (right.y - origin.y) - (left.y - origin.y) * (right.x - origin.x);
}

std::vector<Point2D> convex_hull_2d(const float* keys_xy, size_t key_count) {
    if (keys_xy == nullptr) {
        throw std::runtime_error("HullKV keys pointer is null");
    }
    if (key_count == 0) {
        throw std::runtime_error("HullKV requires at least one 2D key");
    }

    std::vector<Point2D> points;
    points.reserve(key_count);
    for (size_t index = 0; index < key_count; ++index) {
        points.push_back(Point2D{
            keys_xy[index * 2],
            keys_xy[index * 2 + 1],
            static_cast<uint32_t>(index),
        });
    }
    std::sort(points.begin(), points.end(), [](const Point2D& left, const Point2D& right) {
        if (left.x != right.x) {
            return left.x < right.x;
        }
        if (left.y != right.y) {
            return left.y < right.y;
        }
        return left.original_index < right.original_index;
    });

    std::vector<Point2D> unique_points;
    unique_points.reserve(points.size());
    for (const Point2D& point : points) {
        if (!unique_points.empty() && unique_points.back().x == point.x && unique_points.back().y == point.y) {
            continue;
        }
        unique_points.push_back(point);
    }
    if (unique_points.size() <= 2) {
        return unique_points;
    }

    std::vector<Point2D> lower;
    for (const Point2D& point : unique_points) {
        while (lower.size() >= 2 && cross(lower[lower.size() - 2], lower.back(), point) <= 0.0F) {
            lower.pop_back();
        }
        lower.push_back(point);
    }

    std::vector<Point2D> upper;
    for (auto cursor = unique_points.rbegin(); cursor != unique_points.rend(); ++cursor) {
        while (upper.size() >= 2 && cross(upper[upper.size() - 2], upper.back(), *cursor) <= 0.0F) {
            upper.pop_back();
        }
        upper.push_back(*cursor);
    }

    lower.pop_back();
    upper.pop_back();
    lower.insert(lower.end(), upper.begin(), upper.end());
    if (lower.empty()) {
        return unique_points;
    }
    return lower;
}

HullSupport hull_support_2d(CmzEngine* engine, const float* keys_xy, size_t key_count, float query_x, float query_y) {
    std::vector<Point2D> hull = convex_hull_2d(keys_xy, key_count);
    std::sort(hull.begin(), hull.end(), [](const Point2D& left, const Point2D& right) {
        return left.original_index < right.original_index;
    });

    std::vector<float> hull_keys;
    hull_keys.reserve(hull.size() * 2);
    for (const Point2D& point : hull) {
        hull_keys.push_back(point.x);
        hull_keys.push_back(point.y);
    }

    uint32_t selected_local = 0;
    float selected_score = 0.0F;
    const int status = cmz_cutlass_hardmax2d_values(
        hull_keys.data(),
        hull.size(),
        query_x,
        query_y,
        &selected_local,
        &selected_score);
    if (status != 0) {
        throw std::runtime_error(std::string("HullKV CUTLASS hardmax failed: ") +
                                 cudaGetErrorString(static_cast<cudaError_t>(status)));
    }
    if (selected_local >= hull.size()) {
        throw std::runtime_error("HullKV CUTLASS hardmax returned out-of-range index");
    }
    if (engine != nullptr) {
        ++engine->cutlass_hardmax2d_count;
    }
    return HullSupport{hull[selected_local].original_index, selected_score};
}

torch::Device required_cuda_device() {
    if (!torch::cuda::is_available()) {
        throw std::runtime_error("CUDA is required for native LibTorch decoder hot path");
    }
    return torch::Device(torch::kCUDA, 0);
}

void ensure_decoder_initialized(CmzEngine* engine) {
    if (engine == nullptr) {
        throw std::runtime_error("null CmzEngine handle");
    }
    if (engine->decoder_initialized) {
        return;
    }
    const torch::Device device = required_cuda_device();
    torch::manual_seed(20260527);
    const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device).requires_grad(true);
    engine->decoder_command_weight = torch::tensor(
                                         {
                                             {0.07F, -0.03F},
                                             {-0.05F, 0.04F},
                                             {0.03F, 0.08F},
                                             {-0.02F, -0.06F},
                                             {0.09F, 0.02F},
                                             {-0.04F, 0.01F},
                                         },
                                         options)
                                         .clone()
                                         .detach()
                                         .set_requires_grad(true);
    engine->decoder_command_bias = torch::zeros({static_cast<int64_t>(kDecoderCommandCount)}, options)
                                       .clone()
                                       .detach()
                                       .set_requires_grad(true);
    engine->decoder_initialized = true;
}

torch::Tensor decoder_hidden_2d(CmzEngine* engine, const uint32_t* square_piece_tokens, size_t square_count, uint32_t side_to_move) {
    if (square_piece_tokens == nullptr || square_count != 64) {
        throw std::runtime_error("decoder forward requires exactly 64 square piece tokens");
    }
    ensure_decoder_initialized(engine);
    const torch::Device device = required_cuda_device();
    std::vector<float> piece_values;
    std::vector<float> key_values;
    piece_values.reserve(64);
    key_values.reserve(128);
    for (size_t square = 0; square < 64; ++square) {
        piece_values.push_back(static_cast<float>(square_piece_tokens[square]) / 12.0F);
        key_values.push_back(static_cast<float>(square & 7U) / 7.0F);
        key_values.push_back(static_cast<float>(square >> 3U) / 7.0F);
    }
    const auto host_float = torch::TensorOptions().dtype(torch::kFloat32);
    const auto pieces = torch::from_blob(piece_values.data(), {64}, host_float).clone().to(device);
    const auto occupancy = pieces.ne(0.0F).to(torch::kFloat32);
    const auto values = torch::stack({pieces, occupancy}, 1);
    const auto keys = torch::from_blob(key_values.data(), {64, 2}, host_float).clone().to(device);
    const auto query = torch::tensor(
        {1.0F, side_to_move == 0 ? 0.0F : 1.0F},
        torch::TensorOptions().dtype(torch::kFloat32).device(device));
    const auto scores = torch::matmul(keys, query);
    const auto weights = torch::softmax(scores, 0);
    return torch::sum(weights.unsqueeze(1) * values, 0);
}

torch::Tensor decoder_policy_logits(
    CmzEngine* engine,
    const uint32_t* square_piece_tokens,
    size_t square_count,
    uint32_t side_to_move) {
    const auto hidden = decoder_hidden_2d(engine, square_piece_tokens, square_count, side_to_move);
    return torch::matmul(engine->decoder_command_weight, hidden) + engine->decoder_command_bias;
}

void zero_decoder_grads(CmzEngine* engine) {
    for (torch::Tensor* tensor : {
             &engine->decoder_command_weight,
             &engine->decoder_command_bias,
         }) {
        if (tensor->grad().defined()) {
            tensor->mutable_grad().zero_();
        }
    }
}

void apply_decoder_gradients(CmzEngine* engine, float learning_rate) {
    torch::NoGradGuard guard;
    for (torch::Tensor* tensor : {
             &engine->decoder_command_weight,
             &engine->decoder_command_bias,
         }) {
        if (!tensor->grad().defined()) {
            throw std::runtime_error("decoder trainable tensor has no gradient");
        }
        *tensor = (tensor->detach() - learning_rate * tensor->grad()).set_requires_grad(true);
    }
}

std::array<uint32_t, 65> frozen_attention_project_board_state(
    CmzEngine* engine,
    const uint32_t* trace_tokens,
    size_t packet_count) {
    std::array<uint32_t, 65> projected{};
    projected.fill(0U);
    const int status = cmz_cuda_project_board_latest_writes(
        trace_tokens,
        packet_count,
        projected.data(),
        64,
        projected.data() + 64);
    if (status != 0) {
        throw std::runtime_error(std::string("CUDA latest-write board projection failed: ") +
                                 cudaGetErrorString(static_cast<cudaError_t>(status)));
    }
    if (engine != nullptr) {
        ++engine->cuda_board_projection_count;
        engine->frozen_layer_step_count += 65;
    }
    return projected;
}

std::vector<uint32_t> nested_hull_topk_2d(
    CmzEngine* engine,
    const float* keys_xy,
    size_t key_count,
    size_t k,
    float query_x,
    float query_y) {
    if (k > key_count) {
        throw std::runtime_error("NestedHullTopK2D k exceeds key count");
    }
    std::vector<uint32_t> selected(k, 0U);
    const int status = cmz_cutlass_topk2d_values(keys_xy, key_count, k, query_x, query_y, selected.data());
    if (status != 0) {
        throw std::runtime_error(std::string("NestedHullTopK2D CUTLASS top-k failed: ") +
                                 cudaGetErrorString(static_cast<cudaError_t>(status)));
    }
    if (engine != nullptr) {
        ++engine->cutlass_hardmax2d_count;
    }
    return selected;
}

}  // namespace

extern "C" int cmz_engine_create(CmzEngine** out) {
    if (out == nullptr) {
        return 1;
    }
    try {
        *out = new CmzEngine();
        return 0;
    } catch (const std::exception&) {
        *out = nullptr;
        return 1;
    }
}

extern "C" void cmz_engine_destroy(CmzEngine* engine) {
    delete engine;
}

extern "C" const char* cmz_engine_last_error(const CmzEngine* engine) {
    if (engine == nullptr) {
        return "null CmzEngine handle";
    }
    return engine->last_error.c_str();
}

extern "C" int cmz_engine_runtime_mode(const CmzEngine* engine, char* out, size_t out_len) {
    if (engine == nullptr) {
        return 1;
    }
    if (out == nullptr || out_len == 0) {
        return set_error(const_cast<CmzEngine*>(engine), "runtime mode output buffer is empty");
    }
    write_output(kRuntimeMode, out, out_len, nullptr);
    return 0;
}

extern "C" int cmz_engine_percepta_contract_json(const CmzEngine* engine, char* out, size_t out_len, size_t* written) {
    if (engine == nullptr) {
        return 1;
    }
    write_output(kPerceptaContractJson, out, out_len, written);
    if (out != nullptr && out_len > 0 && std::strlen(kPerceptaContractJson) + 1 > out_len) {
        return set_error(const_cast<CmzEngine*>(engine), "Percepta contract output buffer too small");
    }
    const_cast<CmzEngine*>(engine)->last_error.clear();
    return 0;
}

extern "C" int cmz_engine_frozen_rule_graph_json(const CmzEngine* engine, char* out, size_t out_len, size_t* written) {
    if (engine == nullptr) {
        return 1;
    }
    write_output(kFrozenRuleGraphJson, out, out_len, written);
    if (out != nullptr && out_len > 0 && std::strlen(kFrozenRuleGraphJson) + 1 > out_len) {
        return set_error(const_cast<CmzEngine*>(engine), "frozen rule graph output buffer too small");
    }
    const_cast<CmzEngine*>(engine)->last_error.clear();
    return 0;
}

extern "C" size_t cmz_engine_frozen_layer_step_count(const CmzEngine* engine) {
    return engine == nullptr ? 0 : engine->frozen_layer_step_count;
}

extern "C" int cmz_engine_frozen_attack_mask(
    CmzEngine* engine,
    uint32_t piece_token,
    uint32_t from_square,
    uint64_t* out_mask) {
    if (out_mask == nullptr) {
        return set_error(engine, "frozen attack mask output pointer is null");
    }
    try {
        *out_mask = frozen_attack_table_attention_lookup(engine, piece_token, from_square);
        if (engine != nullptr) {
            ++engine->frozen_layer_step_count;
            engine->last_error.clear();
        }
        return 0;
    } catch (const std::exception& ex) {
        return set_error(engine, ex.what());
    }
}

extern "C" int cmz_engine_frozen_ray_scan_mask(
    CmzEngine* engine,
    uint32_t from_square,
    int32_t delta_file,
    int32_t delta_rank,
    uint64_t occupancy_mask,
    uint64_t* out_mask) {
    if (out_mask == nullptr) {
        return set_error(engine, "frozen ray scan output pointer is null");
    }
    try {
        *out_mask = frozen_ray_scan_mask_attention(engine, from_square, delta_file, delta_rank, occupancy_mask);
        if (engine != nullptr) {
            ++engine->frozen_layer_step_count;
            engine->last_error.clear();
        }
        return 0;
    } catch (const std::exception& ex) {
        return set_error(engine, ex.what());
    }
}

extern "C" int cmz_engine_frozen_candidate_target_mask(
    CmzEngine* engine,
    uint32_t piece_token,
    uint32_t from_square,
    uint64_t friendly_mask,
    uint64_t enemy_mask,
    uint64_t occupancy_mask,
    uint32_t ep_square,
    uint64_t* out_mask) {
    if (out_mask == nullptr) {
        return set_error(engine, "frozen candidate target output pointer is null");
    }
    try {
        *out_mask = frozen_candidate_target_mask_attention(
            engine,
            piece_token, from_square, friendly_mask, enemy_mask, occupancy_mask, ep_square);
        if (engine != nullptr) {
            ++engine->frozen_layer_step_count;
            engine->last_error.clear();
        }
        return 0;
    } catch (const std::exception& ex) {
        return set_error(engine, ex.what());
    }
}

extern "C" int cmz_engine_frozen_castle_target_mask(
    CmzEngine* engine,
    const char* fen,
    int32_t white,
    uint64_t* out_mask) {
    if (out_mask == nullptr) {
        return set_error(engine, "frozen castle target output pointer is null");
    }
    try {
        const Board board = parse_fen(fen);
        *out_mask = frozen_castle_target_mask_attention(engine, board, white != 0);
        if (engine != nullptr) {
            ++engine->frozen_layer_step_count;
            engine->last_error.clear();
        }
        return 0;
    } catch (const std::exception& ex) {
        return set_error(engine, ex.what());
    }
}

extern "C" int cmz_engine_frozen_move_legal(
    CmzEngine* engine,
    const char* fen,
    const char* uci,
    int32_t* out_legal) {
    if (out_legal == nullptr) {
        return set_error(engine, "frozen move legal output pointer is null");
    }
    if (uci == nullptr) {
        return set_error(engine, "frozen move legal UCI pointer is null");
    }
    try {
        const Board board = parse_fen(fen);
        *out_legal = frozen_move_legal_attention(engine, board, uci) ? 1 : 0;
        if (engine != nullptr) {
            ++engine->frozen_layer_step_count;
            engine->last_error.clear();
        }
        return 0;
    } catch (const std::exception& ex) {
        return set_error(engine, ex.what());
    }
}

extern "C" int cmz_engine_frozen_terminal_status(
    CmzEngine* engine,
    const char* fen,
    uint32_t repetition_count,
    int32_t adjudication_cap_reached,
    uint32_t* out_result,
    uint32_t* out_reason) {
    if (out_result == nullptr || out_reason == nullptr) {
        return set_error(engine, "frozen terminal status output pointer is null");
    }
    try {
        const Board board = parse_fen(fen);
        const auto terminal =
            frozen_terminal_predicate_layer(engine, board, static_cast<int>(repetition_count), adjudication_cap_reached != 0);
        *out_result = terminal[0];
        *out_reason = terminal[1];
        if (engine != nullptr) {
            engine->last_error.clear();
        }
        return 0;
    } catch (const std::exception& ex) {
        return set_error(engine, ex.what());
    }
}

extern "C" size_t cmz_engine_attention_decode_count(const CmzEngine* engine) {
    return engine == nullptr ? 0 : engine->attention_decode_count;
}

extern "C" size_t cmz_engine_cuda_trace_emit_attention_count(const CmzEngine* engine) {
    return engine == nullptr ? 0 : engine->cuda_trace_emit_attention_count;
}

extern "C" size_t cmz_engine_cuda_trace_select_count(const CmzEngine* engine) {
    return engine == nullptr ? 0 : engine->cuda_trace_select_count;
}

extern "C" size_t cmz_engine_cutlass_hardmax2d_count(const CmzEngine* engine) {
    return engine == nullptr ? 0 : engine->cutlass_hardmax2d_count;
}

extern "C" size_t cmz_engine_cuda_board_projection_count(const CmzEngine* engine) {
    return engine == nullptr ? 0 : engine->cuda_board_projection_count;
}

extern "C" size_t cmz_engine_cuda_attack_table_attention_count(const CmzEngine* engine) {
    return engine == nullptr ? 0 : engine->cuda_attack_table_attention_count;
}

extern "C" size_t cmz_engine_cuda_candidate_table_attention_count(const CmzEngine* engine) {
    return engine == nullptr ? 0 : engine->cuda_candidate_table_attention_count;
}

extern "C" size_t cmz_engine_cuda_candidate_move_attention_count(const CmzEngine* engine) {
    return engine == nullptr ? 0 : engine->cuda_candidate_move_attention_count;
}

extern "C" size_t cmz_engine_cuda_candidate_move_layer_count(const CmzEngine* engine) {
    return engine == nullptr ? 0 : engine->cuda_candidate_move_layer_count;
}

extern "C" size_t cmz_engine_cuda_resolve_move_attention_count(const CmzEngine* engine) {
    return engine == nullptr ? 0 : engine->cuda_resolve_move_attention_count;
}

extern "C" size_t cmz_engine_cuda_ray_scan_attention_count(const CmzEngine* engine) {
    return engine == nullptr ? 0 : engine->cuda_ray_scan_attention_count;
}

extern "C" size_t cmz_engine_cuda_legal_filter_attention_count(const CmzEngine* engine) {
    return engine == nullptr ? 0 : engine->cuda_legal_filter_attention_count;
}

extern "C" size_t cmz_engine_cuda_legal_filter_batch_attention_count(const CmzEngine* engine) {
    return engine == nullptr ? 0 : engine->cuda_legal_filter_batch_attention_count;
}

extern "C" size_t cmz_engine_cuda_legal_filter_v2_attention_count(const CmzEngine* engine) {
    return engine == nullptr ? 0 : engine->cuda_legal_filter_v2_attention_count;
}

extern "C" size_t cmz_engine_cuda_legal_filter_v2_layer_count(const CmzEngine* engine) {
    return engine == nullptr ? 0 : engine->cuda_legal_filter_v2_layer_count;
}

extern "C" size_t cmz_engine_cuda_legal_filter_v2_batch_attention_count(const CmzEngine* engine) {
    return engine == nullptr ? 0 : engine->cuda_legal_filter_v2_batch_attention_count;
}

extern "C" size_t cmz_engine_cuda_legal_filter_v2_batch_layer_count(const CmzEngine* engine) {
    return engine == nullptr ? 0 : engine->cuda_legal_filter_v2_batch_layer_count;
}

extern "C" size_t cmz_engine_cuda_castle_attention_count(const CmzEngine* engine) {
    return engine == nullptr ? 0 : engine->cuda_castle_attention_count;
}

extern "C" size_t cmz_engine_cuda_make_move_board_attention_count(const CmzEngine* engine) {
    return engine == nullptr ? 0 : engine->cuda_make_move_board_attention_count;
}

extern "C" size_t cmz_engine_cuda_make_move_metadata_attention_count(const CmzEngine* engine) {
    return engine == nullptr ? 0 : engine->cuda_make_move_metadata_attention_count;
}

extern "C" size_t cmz_engine_cuda_terminal_status_attention_count(const CmzEngine* engine) {
    return engine == nullptr ? 0 : engine->cuda_terminal_status_attention_count;
}

extern "C" size_t cmz_engine_cuda_terminal_status_layer_count(const CmzEngine* engine) {
    return engine == nullptr ? 0 : engine->cuda_terminal_status_layer_count;
}

extern "C" int cmz_engine_cuda_available(const CmzEngine*) {
    int count = 0;
    const cudaError_t status = cudaGetDeviceCount(&count);
    return status == cudaSuccess && count > 0 ? 1 : 0;
}

extern "C" int cmz_engine_cuda_device_name(const CmzEngine* engine, char* out, size_t out_len) {
    if (out == nullptr || out_len == 0) {
        return set_error(const_cast<CmzEngine*>(engine), "device name output buffer is empty");
    }
    int device = 0;
    cudaDeviceProp prop{};
    const cudaError_t status = cudaGetDeviceProperties(&prop, device);
    if (status != cudaSuccess) {
        return set_error(const_cast<CmzEngine*>(engine), cudaGetErrorString(status));
    }
    const std::string name(prop.name);
    write_output(name, out, out_len, nullptr);
    return 0;
}

extern "C" int cmz_engine_hull_hardmax_2d(
    CmzEngine* engine,
    const float* keys_xy,
    size_t key_count,
    float query_x,
    float query_y,
    uint32_t* out_index,
    float* out_score) {
    if (out_index == nullptr || out_score == nullptr) {
        return set_error(engine, "HullHardmax2D output pointer is null");
    }
    try {
        const HullSupport support = hull_support_2d(engine, keys_xy, key_count, query_x, query_y);
        *out_index = support.original_index;
        *out_score = support.score;
        if (engine != nullptr) {
            engine->last_error.clear();
        }
        return 0;
    } catch (const std::exception& ex) {
        return set_error(engine, ex.what());
    }
}

extern "C" int cmz_engine_nested_hull_topk_2d(
    CmzEngine* engine,
    const float* keys_xy,
    size_t key_count,
    size_t k,
    float query_x,
    float query_y,
    uint32_t* out_indices) {
    if (out_indices == nullptr && k > 0) {
        return set_error(engine, "NestedHullTopK2D output pointer is null");
    }
    try {
        const std::vector<uint32_t> indices = nested_hull_topk_2d(engine, keys_xy, key_count, k, query_x, query_y);
        if (!indices.empty()) {
            std::memcpy(out_indices, indices.data(), indices.size() * sizeof(uint32_t));
        }
        if (engine != nullptr) {
            engine->last_error.clear();
        }
        return 0;
    } catch (const std::exception& ex) {
        return set_error(engine, ex.what());
    }
}

extern "C" int cmz_engine_legal_moves_uci(CmzEngine* engine, const char* fen, char* out, size_t out_len, size_t* written) {
    try {
        const Board board = parse_fen(fen);
        const std::string text = join_lines(legal_moves_uci(engine, board));
        write_output(text, out, out_len, written);
        if (out != nullptr && out_len > 0 && text.size() + 1 > out_len) {
            return set_error(engine, "legal move output buffer too small");
        }
        if (engine != nullptr) {
            engine->last_error.clear();
        }
        return 0;
    } catch (const std::exception& ex) {
        return set_error(engine, ex.what());
    }
}

extern "C" int cmz_engine_legal_trace_packets(
    CmzEngine* engine,
    const char* fen,
    uint32_t* out_tokens,
    size_t packet_capacity,
    size_t* packet_count) {
    try {
        const Board board = parse_fen(fen);
        if (out_tokens == nullptr || packet_capacity == 0) {
            if (packet_count != nullptr) {
                *packet_count = sorted_pseudo_legal_moves(board).size() * 2 + 1;
            }
            if (engine != nullptr) {
                engine->last_error.clear();
            }
            return 0;
        }
        const std::vector<uint32_t> tokens = legal_trace_tokens(board);
        const size_t required_packets = tokens.size() / kTracePacketWidth;
        if (packet_count != nullptr) {
            *packet_count = required_packets;
        }
        if (packet_capacity < required_packets) {
            return set_error(engine, "legal trace packet output buffer too small");
        }
        std::memcpy(out_tokens, tokens.data(), tokens.size() * sizeof(uint32_t));
        if (engine != nullptr) {
            engine->last_error.clear();
        }
        return 0;
    } catch (const std::exception& ex) {
        return set_error(engine, ex.what());
    }
}

extern "C" int cmz_engine_frozen_legal_trace_attention_packets(
    CmzEngine* engine,
    const char* fen,
    uint32_t* out_tokens,
    size_t packet_capacity,
    size_t* packet_count) {
    try {
        const Board board = parse_fen(fen);
        if (out_tokens == nullptr || packet_capacity == 0) {
            if (packet_count != nullptr) {
                *packet_count = sorted_pseudo_legal_moves(board).size() * 2 + 1;
            }
            if (engine != nullptr) {
                engine->last_error.clear();
            }
            return 0;
        }
        const std::vector<uint32_t> tokens = frozen_legal_trace_attention_tokens(engine, board);
        const size_t required_packets = tokens.size() / kTracePacketWidth;
        if (packet_count != nullptr) {
            *packet_count = required_packets;
        }
        if (packet_capacity < required_packets) {
            return set_error(engine, "frozen legal trace attention output buffer too small");
        }
        std::memcpy(out_tokens, tokens.data(), tokens.size() * sizeof(uint32_t));
        if (engine != nullptr) {
            engine->last_error.clear();
        }
        return 0;
    } catch (const std::exception& ex) {
        return set_error(engine, ex.what());
    }
}

extern "C" int cmz_engine_legal_trace_begin(CmzEngine* engine, const char* fen, size_t* packet_count) {
    if (engine == nullptr) {
        return 1;
    }
    try {
        const Board board = parse_fen(fen);
        engine->legal_trace_stream_fen = fen;
        engine->legal_trace_stream_packet_count = sorted_pseudo_legal_moves(board).size() * 2 + 1;
        engine->legal_trace_stream_cursor = 0;
        engine->attention_decode_count = 0;
        if (packet_count != nullptr) {
            *packet_count = engine->legal_trace_stream_packet_count;
        }
        engine->last_error.clear();
        return 0;
    } catch (const std::exception& ex) {
        return set_error(engine, ex.what());
    }
}

extern "C" int cmz_engine_legal_trace_next(CmzEngine* engine, uint32_t* out_packet_tokens) {
    if (engine == nullptr) {
        return 1;
    }
    if (out_packet_tokens == nullptr) {
        return set_error(engine, "legal trace next output pointer is null");
    }
    const size_t packet_count = engine->legal_trace_stream_packet_count;
    if (engine->legal_trace_stream_cursor >= packet_count) {
        return set_error(engine, "legal trace stream exhausted");
    }
    try {
        const Board board = parse_fen(engine->legal_trace_stream_fen.c_str());
        frozen_legal_trace_attention_packet(engine, board, engine->legal_trace_stream_cursor, out_packet_tokens);
        ++engine->legal_trace_stream_cursor;
        ++engine->attention_decode_count;
        engine->last_error.clear();
        return 0;
    } catch (const std::exception& ex) {
        return set_error(engine, ex.what());
    }
}

extern "C" int cmz_engine_make_move_trace_packets(
    CmzEngine* engine,
    const char* fen,
    const char* uci,
    uint32_t ply,
    uint32_t repetition_count,
    int adjudication_cap_reached,
    uint32_t* out_tokens,
    size_t packet_capacity,
    size_t* packet_count) {
    try {
        if (uci == nullptr) {
            return set_error(engine, "null UCI pointer");
        }
        const Board board = parse_fen(fen);
        CmzEngine* counter_engine = (out_tokens == nullptr || packet_capacity == 0) ? nullptr : engine;
        const std::vector<uint32_t> tokens = make_move_trace_tokens(
            counter_engine,
            board,
            std::string(uci),
            ply,
            static_cast<int>(repetition_count),
            adjudication_cap_reached != 0);
        const size_t required_packets = tokens.size() / kTracePacketWidth;
        if (packet_count != nullptr) {
            *packet_count = required_packets;
        }
        if (out_tokens == nullptr || packet_capacity == 0) {
            if (engine != nullptr) {
                engine->last_error.clear();
            }
            return 0;
        }
        if (packet_capacity < required_packets) {
            return set_error(engine, "make-move trace packet output buffer too small");
        }
        std::memcpy(out_tokens, tokens.data(), tokens.size() * sizeof(uint32_t));
        if (engine != nullptr) {
            engine->last_error.clear();
        }
        return 0;
    } catch (const std::exception& ex) {
        return set_error(engine, ex.what());
    }
}

extern "C" int cmz_engine_frozen_make_move_trace_attention_packets(
    CmzEngine* engine,
    const char* fen,
    const char* uci,
    uint32_t ply,
    uint32_t repetition_count,
    int adjudication_cap_reached,
    uint32_t* out_tokens,
    size_t packet_capacity,
    size_t* packet_count) {
    try {
        if (uci == nullptr) {
            return set_error(engine, "null UCI pointer");
        }
        const Board board = parse_fen(fen);
        CmzEngine* counter_engine = (out_tokens == nullptr || packet_capacity == 0) ? nullptr : engine;
        const std::vector<uint32_t> tokens = frozen_make_move_trace_attention_tokens(
            counter_engine,
            board,
            std::string(uci),
            ply,
            static_cast<int>(repetition_count),
            adjudication_cap_reached != 0);
        const size_t required_packets = tokens.size() / kTracePacketWidth;
        if (packet_count != nullptr) {
            *packet_count = required_packets;
        }
        if (out_tokens == nullptr || packet_capacity == 0) {
            if (engine != nullptr) {
                engine->last_error.clear();
            }
            return 0;
        }
        if (packet_capacity < required_packets) {
            return set_error(engine, "frozen make-move trace packet output buffer too small");
        }
        std::memcpy(out_tokens, tokens.data(), tokens.size() * sizeof(uint32_t));
        if (engine != nullptr) {
            engine->last_error.clear();
        }
        return 0;
    } catch (const std::exception& ex) {
        return set_error(engine, ex.what());
    }
}

extern "C" int cmz_engine_project_board_trace(
    CmzEngine* engine,
    const uint32_t* trace_tokens,
    size_t packet_count,
    uint32_t* out_square_piece_tokens,
    size_t square_capacity,
    uint32_t* out_side_to_move) {
    if (trace_tokens == nullptr && packet_count > 0) {
        return set_error(engine, "board trace projection input pointer is null");
    }
    if (out_square_piece_tokens == nullptr || square_capacity < 64) {
        return set_error(engine, "board trace projection square output buffer too small");
    }
    if (out_side_to_move == nullptr) {
        return set_error(engine, "board trace projection side output pointer is null");
    }
    try {
        const auto projected = frozen_attention_project_board_state(engine, trace_tokens, packet_count);
        std::copy(projected.begin(), projected.begin() + 64, out_square_piece_tokens);
        *out_side_to_move = projected[64];
    } catch (const std::exception& ex) {
        return set_error(engine, ex.what());
    }
    if (engine != nullptr) {
        engine->last_error.clear();
    }
    return 0;
}

extern "C" int cmz_engine_decoder_forward(
    CmzEngine* engine,
    const uint32_t* square_piece_tokens,
    size_t square_count,
    uint32_t side_to_move,
    float* out_command_logits,
    size_t command_capacity) {
    if (out_command_logits == nullptr || command_capacity < kDecoderCommandCount) {
        return set_error(engine, "decoder forward command output buffer too small");
    }
    try {
        const auto output = decoder_policy_logits(engine, square_piece_tokens, square_count, side_to_move);
        const auto logits_cpu = output.detach().to(torch::kCPU);
        const float* logits = logits_cpu.data_ptr<float>();
        std::memcpy(out_command_logits, logits, kDecoderCommandCount * sizeof(float));
        if (engine != nullptr) {
            engine->last_error.clear();
        }
        return 0;
    } catch (const std::exception& ex) {
        return set_error(engine, ex.what());
    }
}

extern "C" int cmz_engine_decoder_policy_gradient_step(
    CmzEngine* engine,
    const uint32_t* square_piece_tokens,
    size_t square_count,
    uint32_t side_to_move,
    uint32_t selected_command,
    float reward,
    float learning_rate,
    float* out_loss) {
    if (selected_command >= kDecoderCommandCount) {
        return set_error(engine, "selected decoder command out of range");
    }
    if (learning_rate <= 0.0F) {
        return set_error(engine, "decoder learning rate must be positive");
    }
    if (out_loss == nullptr) {
        return set_error(engine, "decoder loss output pointer is null");
    }
    try {
        ensure_decoder_initialized(engine);
        zero_decoder_grads(engine);
        const auto logits = decoder_policy_logits(engine, square_piece_tokens, square_count, side_to_move);
        const auto reward_tensor = torch::tensor(
            reward,
            torch::TensorOptions().dtype(torch::kFloat32).device(required_cuda_device()));
        const auto log_probs = torch::log_softmax(logits, 0);
        const auto selected = torch::tensor(
            static_cast<int64_t>(selected_command),
            torch::TensorOptions().dtype(torch::kInt64).device(required_cuda_device()));
        const auto log_prob = log_probs.index({selected});
        const auto loss = -log_prob * reward_tensor;
        loss.backward();
        *out_loss = loss.detach().to(torch::kCPU).item<float>();
        apply_decoder_gradients(engine, learning_rate);
        if (engine != nullptr) {
            engine->last_error.clear();
        }
        return 0;
    } catch (const std::exception& ex) {
        return set_error(engine, ex.what());
    }
}

extern "C" int cmz_engine_policy_select_move(
    CmzEngine* engine,
    const char* fen,
    uint32_t* out_move_id,
    uint32_t* out_legal_count,
    uint32_t* out_selected_index,
    float* out_score) {
    if (engine == nullptr) {
        return 1;
    }
    if (fen == nullptr) {
        return set_error(engine, "policy select move FEN pointer is null");
    }
    if (out_move_id == nullptr || out_legal_count == nullptr || out_selected_index == nullptr || out_score == nullptr) {
        return set_error(engine, "policy select move output pointer is null");
    }
    try {
        const Board board = parse_fen(fen);
        const PolicyMoveSelectionNative selection = policy_select_move_native(engine, board);
        *out_move_id = move_id(selection.move);
        *out_legal_count = selection.legal_count;
        *out_selected_index = selection.selected_index;
        *out_score = selection.score;
        engine->last_error.clear();
        return 0;
    } catch (const std::exception& ex) {
        return set_error(engine, ex.what());
    }
}

extern "C" int cmz_engine_cuda_probe_double(CmzEngine* engine, const uint32_t* input, size_t len, uint32_t* output) {
    if ((input == nullptr || output == nullptr) && len > 0) {
        return set_error(engine, "CUDA probe input/output pointer is null");
    }
    const int status = cmz_cuda_double_values(input, len, output);
    if (status != 0) {
        return set_error(engine, std::string("CUDA probe failed: ") + cudaGetErrorString(static_cast<cudaError_t>(status)));
    }
    if (engine != nullptr) {
        engine->last_error.clear();
    }
    return 0;
}
