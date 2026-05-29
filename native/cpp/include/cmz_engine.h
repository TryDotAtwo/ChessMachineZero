#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CmzEngine CmzEngine;

int cmz_engine_create(CmzEngine** out);
void cmz_engine_destroy(CmzEngine* engine);
const char* cmz_engine_last_error(const CmzEngine* engine);
int cmz_engine_runtime_mode(const CmzEngine* engine, char* out, size_t out_len);
int cmz_engine_percepta_contract_json(const CmzEngine* engine, char* out, size_t out_len, size_t* written);
int cmz_engine_frozen_rule_graph_json(const CmzEngine* engine, char* out, size_t out_len, size_t* written);
size_t cmz_engine_frozen_layer_step_count(const CmzEngine* engine);
size_t cmz_engine_cuda_trace_emit_attention_count(const CmzEngine* engine);
int cmz_engine_frozen_attack_mask(
    CmzEngine* engine,
    uint32_t piece_token,
    uint32_t from_square,
    uint64_t* out_mask);
int cmz_engine_frozen_ray_scan_mask(
    CmzEngine* engine,
    uint32_t from_square,
    int32_t delta_file,
    int32_t delta_rank,
    uint64_t occupancy_mask,
    uint64_t* out_mask);
int cmz_engine_frozen_candidate_target_mask(
    CmzEngine* engine,
    uint32_t piece_token,
    uint32_t from_square,
    uint64_t friendly_mask,
    uint64_t enemy_mask,
    uint64_t occupancy_mask,
    uint32_t ep_square,
    uint64_t* out_mask);
int cmz_engine_frozen_castle_target_mask(CmzEngine* engine, const char* fen, int32_t white, uint64_t* out_mask);
int cmz_engine_frozen_move_legal(CmzEngine* engine, const char* fen, const char* uci, int32_t* out_legal);
int cmz_engine_frozen_terminal_status(
    CmzEngine* engine,
    const char* fen,
    uint32_t repetition_count,
    int32_t adjudication_cap_reached,
    uint32_t* out_result,
    uint32_t* out_reason);
size_t cmz_engine_attention_decode_count(const CmzEngine* engine);
size_t cmz_engine_cuda_trace_select_count(const CmzEngine* engine);
size_t cmz_engine_cutlass_hardmax2d_count(const CmzEngine* engine);
size_t cmz_engine_cuda_board_projection_count(const CmzEngine* engine);
size_t cmz_engine_cuda_attack_table_attention_count(const CmzEngine* engine);
size_t cmz_engine_cuda_candidate_table_attention_count(const CmzEngine* engine);
size_t cmz_engine_cuda_candidate_move_attention_count(const CmzEngine* engine);
size_t cmz_engine_cuda_candidate_move_layer_count(const CmzEngine* engine);
size_t cmz_engine_cuda_resolve_move_attention_count(const CmzEngine* engine);
size_t cmz_engine_cuda_ray_scan_attention_count(const CmzEngine* engine);
size_t cmz_engine_cuda_legal_filter_attention_count(const CmzEngine* engine);
size_t cmz_engine_cuda_legal_filter_batch_attention_count(const CmzEngine* engine);
size_t cmz_engine_cuda_legal_filter_v2_attention_count(const CmzEngine* engine);
size_t cmz_engine_cuda_legal_filter_v2_layer_count(const CmzEngine* engine);
size_t cmz_engine_cuda_legal_filter_v2_batch_attention_count(const CmzEngine* engine);
size_t cmz_engine_cuda_legal_filter_v2_batch_layer_count(const CmzEngine* engine);
size_t cmz_engine_cuda_castle_attention_count(const CmzEngine* engine);
size_t cmz_engine_cuda_make_move_board_attention_count(const CmzEngine* engine);
size_t cmz_engine_cuda_make_move_metadata_attention_count(const CmzEngine* engine);
size_t cmz_engine_cuda_terminal_status_attention_count(const CmzEngine* engine);
size_t cmz_engine_cuda_terminal_status_layer_count(const CmzEngine* engine);
int cmz_engine_cuda_available(const CmzEngine* engine);
int cmz_engine_cuda_device_name(const CmzEngine* engine, char* out, size_t out_len);
int cmz_engine_hull_hardmax_2d(
    CmzEngine* engine,
    const float* keys_xy,
    size_t key_count,
    float query_x,
    float query_y,
    uint32_t* out_index,
    float* out_score);
int cmz_engine_nested_hull_topk_2d(
    CmzEngine* engine,
    const float* keys_xy,
    size_t key_count,
    size_t k,
    float query_x,
    float query_y,
    uint32_t* out_indices);
int cmz_engine_legal_moves_uci(CmzEngine* engine, const char* fen, char* out, size_t out_len, size_t* written);
int cmz_engine_legal_trace_packets(
    CmzEngine* engine,
    const char* fen,
    uint32_t* out_tokens,
    size_t packet_capacity,
    size_t* packet_count);
int cmz_engine_frozen_legal_trace_attention_packets(
    CmzEngine* engine,
    const char* fen,
    uint32_t* out_tokens,
    size_t packet_capacity,
    size_t* packet_count);
int cmz_engine_legal_trace_begin(CmzEngine* engine, const char* fen, size_t* packet_count);
int cmz_engine_legal_trace_next(CmzEngine* engine, uint32_t* out_packet_tokens);
int cmz_engine_make_move_trace_packets(
    CmzEngine* engine,
    const char* fen,
    const char* uci,
    uint32_t ply,
    uint32_t repetition_count,
    int adjudication_cap_reached,
    uint32_t* out_tokens,
    size_t packet_capacity,
    size_t* packet_count);
int cmz_engine_frozen_make_move_trace_attention_packets(
    CmzEngine* engine,
    const char* fen,
    const char* uci,
    uint32_t ply,
    uint32_t repetition_count,
    int adjudication_cap_reached,
    uint32_t* out_tokens,
    size_t packet_capacity,
    size_t* packet_count);
int cmz_engine_project_board_trace(
    CmzEngine* engine,
    const uint32_t* trace_tokens,
    size_t packet_count,
    uint32_t* out_square_piece_tokens,
    size_t square_capacity,
    uint32_t* out_side_to_move);
int cmz_engine_decoder_forward(
    CmzEngine* engine,
    const uint32_t* square_piece_tokens,
    size_t square_count,
    uint32_t side_to_move,
    float* out_command_logits,
    size_t command_capacity);
int cmz_engine_decoder_policy_gradient_step(
    CmzEngine* engine,
    const uint32_t* square_piece_tokens,
    size_t square_count,
    uint32_t side_to_move,
    uint32_t selected_command,
    float reward,
    float learning_rate,
    float* out_loss);
int cmz_engine_policy_select_move(
    CmzEngine* engine,
    const char* fen,
    uint32_t* out_move_id,
    uint32_t* out_legal_count,
    uint32_t* out_selected_index,
    float* out_score);
int cmz_engine_cuda_probe_double(CmzEngine* engine, const uint32_t* input, size_t len, uint32_t* output);

#ifdef __cplusplus
}
#endif
