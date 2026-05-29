#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>
#include <stdint.h>
#include <stddef.h>
#include <limits>
#include <vector>

constexpr uint32_t kCmzCandidateRecordSlotCount = 64U * 64U * 5U;
constexpr uint32_t kCmzCandidateRecordWidth = 5U;

__global__ void cmz_double_kernel(const uint32_t* input, uint32_t* output, size_t len) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < len) {
        output[index] = input[index] * 2u;
    }
}

__global__ void cmz_dot2_kernel(const float* keys_xy, float query_x, float query_y, float* scores, size_t len) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < len) {
        scores[index] = keys_xy[index * 2] * query_x + keys_xy[index * 2 + 1] * query_y;
    }
}

__global__ void cmz_hardmax_float_select_kernel(
    const float* scores,
    size_t len,
    uint32_t* selected,
    float* selected_score) {
    if (threadIdx.x != 0 || blockIdx.x != 0 || len == 0) {
        return;
    }
    uint32_t best_index = 0;
    float best_score = scores[0];
    for (size_t index = 1; index < len; ++index) {
        const float score = scores[index];
        if (score > best_score) {
            best_index = static_cast<uint32_t>(index);
            best_score = score;
        }
    }
    *selected = best_index;
    *selected_score = best_score;
}

__global__ void cmz_topk_float_select_kernel(const float* scores, size_t len, size_t k, uint32_t* selected) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }
    constexpr float min_score = -3.4028234663852886e+38F;
    for (size_t rank = 0; rank < k; ++rank) {
        uint32_t best_index = 0U;
        float best_score = min_score;
        for (size_t index = 0; index < len; ++index) {
            bool already_selected = false;
            for (size_t prior = 0; prior < rank; ++prior) {
                if (selected[prior] == static_cast<uint32_t>(index)) {
                    already_selected = true;
                }
            }
            if (already_selected) {
                continue;
            }
            const float score = scores[index];
            if (score > best_score || (score == best_score && index < best_index)) {
                best_score = score;
                best_index = static_cast<uint32_t>(index);
            }
        }
        selected[rank] = best_index;
    }
}

__global__ void cmz_trace_select_kernel(
    const uint32_t* tokens,
    uint32_t* output,
    size_t packet_count,
    size_t query_index) {
    constexpr size_t packet_width = 7;
    const size_t lane = threadIdx.x;
    if (query_index < packet_count && lane < packet_width) {
        output[lane] = tokens[query_index * packet_width + lane];
    }
}

__global__ void cmz_trace_emit_packet_attention_kernel(
    uint32_t op,
    uint32_t a0,
    uint32_t a1,
    uint32_t a2,
    uint32_t a3,
    uint32_t tag,
    uint32_t commit,
    uint32_t* output) {
    const uint32_t lane = threadIdx.x;
    if (lane >= 7U) {
        return;
    }
    const uint32_t values[7] = {op, a0, a1, a2, a3, tag, commit};
    output[lane] = values[lane];
}

__global__ void cmz_project_board_latest_writes_kernel(
    const uint32_t* tokens,
    uint32_t* output,
    size_t packet_count) {
    constexpr size_t packet_width = 7;
    constexpr uint32_t trace_write_sq = 1;
    constexpr uint32_t trace_write_reg = 3;
    constexpr uint32_t reg_side_to_move = 1;
    const uint32_t lane = threadIdx.x;
    if (lane < 64) {
        uint32_t selected_piece = 0;
        for (size_t packet_index = 0; packet_index < packet_count; ++packet_index) {
            const uint32_t* packet = tokens + packet_index * packet_width;
            if (packet[6] != 0 && packet[0] == trace_write_sq && packet[1] == lane) {
                selected_piece = packet[2];
            }
        }
        output[lane] = selected_piece;
    } else if (lane == 64) {
        uint32_t side_to_move = 0;
        for (size_t packet_index = 0; packet_index < packet_count; ++packet_index) {
            const uint32_t* packet = tokens + packet_index * packet_width;
            if (packet[6] != 0 && packet[0] == trace_write_reg && packet[1] == reg_side_to_move) {
                side_to_move = packet[2];
            }
        }
        output[64] = side_to_move;
    }
}

__global__ void cmz_attack_table_lookup_attention_kernel(
    const uint32_t* keys,
    const uint64_t* values,
    size_t value_count,
    uint32_t query_key,
    uint64_t* output,
    uint32_t* found) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < value_count && keys[index] == query_key) {
        *output = values[index];
        *found = 1;
    }
}

__device__ bool cmz_on_board(int file, int rank) {
    return file >= 0 && file < 8 && rank >= 0 && rank < 8;
}

__device__ uint32_t cmz_square_index(int file, int rank) {
    return static_cast<uint32_t>(rank * 8 + file);
}

__device__ uint64_t cmz_bit(uint32_t square) {
    return 1ULL << square;
}

__device__ uint32_t cmz_qk2_score_u32(uint32_t query_x, uint32_t query_y, uint32_t key_x, uint32_t key_y) {
    return query_x * key_x + query_y * key_y;
}

__device__ void cmz_qk2_hardmax_select_u32(
    uint32_t query_x,
    uint32_t query_y,
    uint32_t key_x,
    uint32_t key_y,
    uint32_t tie_index,
    uint32_t value,
    uint32_t* selected_score,
    uint32_t* selected_tie_index,
    uint32_t* selected_value) {
    const uint32_t score = cmz_qk2_score_u32(query_x, query_y, key_x, key_y);
    if (score > *selected_score || (score == *selected_score && tie_index < *selected_tie_index)) {
        *selected_score = score;
        *selected_tie_index = tie_index;
        *selected_value = value;
    }
}

__device__ void cmz_qk2_hardmax_select_u64(
    uint32_t query_x,
    uint32_t query_y,
    uint32_t key_x,
    uint32_t key_y,
    uint32_t tie_index,
    uint64_t value,
    uint32_t* selected_score,
    uint32_t* selected_tie_index,
    uint64_t* selected_value) {
    const uint32_t score = cmz_qk2_score_u32(query_x, query_y, key_x, key_y);
    if (score > *selected_score || (score == *selected_score && tie_index < *selected_tie_index)) {
        *selected_score = score;
        *selected_tie_index = tie_index;
        *selected_value = value;
    }
}

__device__ void cmz_qk2_select_or_write_u64(
    uint32_t query_x,
    uint32_t query_y,
    uint32_t key_x,
    uint32_t key_y,
    uint64_t value,
    uint64_t* output) {
    const uint32_t score = cmz_qk2_score_u32(query_x, query_y, key_x, key_y);
    const uint64_t selected = score == 0U ? 0ULL : value;
    *output |= selected;
}

__device__ void cmz_qk2_select_or_write_u32(
    uint32_t query_x,
    uint32_t query_y,
    uint32_t key_x,
    uint32_t key_y,
    uint32_t value,
    uint32_t* output) {
    const uint32_t score = cmz_qk2_score_u32(query_x, query_y, key_x, key_y);
    const uint32_t selected = score == 0U ? 0U : value;
    *output |= selected;
}

__device__ void cmz_add_offset_targets(
    uint64_t* mask,
    int from_file,
    int from_rank,
    const int* delta_files,
    const int* delta_ranks,
    int count,
    uint64_t friendly_mask) {
    for (int index = 0; index < count; ++index) {
        const int file = from_file + delta_files[index];
        const int rank = from_rank + delta_ranks[index];
        if (cmz_on_board(file, rank)) {
            const uint32_t square = cmz_square_index(file, rank);
            const uint64_t square_mask = cmz_bit(square);
            if ((friendly_mask & square_mask) == 0) {
                *mask |= square_mask;
            }
        }
    }
}

__device__ void cmz_add_ray_targets(
    uint64_t* mask,
    int from_file,
    int from_rank,
    int delta_file,
    int delta_rank,
    uint64_t friendly_mask,
    uint64_t occupancy_mask) {
    int file = from_file + delta_file;
    int rank = from_rank + delta_rank;
    while (cmz_on_board(file, rank)) {
        const uint32_t square = cmz_square_index(file, rank);
        const uint64_t square_mask = cmz_bit(square);
        if ((friendly_mask & square_mask) == 0) {
            *mask |= square_mask;
        }
        if ((occupancy_mask & square_mask) != 0) {
            break;
        }
        file += delta_file;
        rank += delta_rank;
    }
}

__global__ void cmz_candidate_target_attention_kernel(
    uint32_t token,
    uint32_t from_square,
    uint64_t friendly_mask,
    uint64_t enemy_mask,
    uint64_t occupancy_mask,
    uint32_t ep_square,
    uint64_t* output) {
    const int from_file = static_cast<int>(from_square % 8U);
    const int from_rank = static_cast<int>(from_square / 8U);
    uint64_t mask = 0;

    if (token == 1U || token == 7U) {
        const bool white = token == 1U;
        const int direction = white ? 1 : -1;
        const int start_rank = white ? 1 : 6;
        const int one_rank = from_rank + direction;
        if (cmz_on_board(from_file, one_rank)) {
            const uint32_t one = cmz_square_index(from_file, one_rank);
            const uint64_t one_mask = cmz_bit(one);
            if ((occupancy_mask & one_mask) == 0) {
                mask |= one_mask;
                const int two_rank = from_rank + 2 * direction;
                if (from_rank == start_rank && cmz_on_board(from_file, two_rank)) {
                    const uint32_t two = cmz_square_index(from_file, two_rank);
                    const uint64_t two_mask = cmz_bit(two);
                    if ((occupancy_mask & two_mask) == 0) {
                        mask |= two_mask;
                    }
                }
            }
        }
        for (int delta_file = -1; delta_file <= 1; delta_file += 2) {
            const int file = from_file + delta_file;
            const int rank = from_rank + direction;
            if (cmz_on_board(file, rank)) {
                const uint32_t target = cmz_square_index(file, rank);
                const uint64_t target_mask = cmz_bit(target);
                if ((enemy_mask & target_mask) != 0) {
                    mask |= target_mask;
                }
                if (ep_square < 64U && target == ep_square) {
                    const int captured = white ? static_cast<int>(ep_square) - 8 : static_cast<int>(ep_square) + 8;
                    if (captured >= 0 && captured < 64 && (enemy_mask & cmz_bit(static_cast<uint32_t>(captured))) != 0) {
                        mask |= target_mask;
                    }
                }
            }
        }
    } else if (token == 2U || token == 8U) {
        const int delta_files[8] = {1, 2, 2, 1, -1, -2, -2, -1};
        const int delta_ranks[8] = {2, 1, -1, -2, -2, -1, 1, 2};
        cmz_add_offset_targets(&mask, from_file, from_rank, delta_files, delta_ranks, 8, friendly_mask);
    } else if (token == 6U || token == 12U) {
        const int delta_files[8] = {1, 1, 0, -1, -1, -1, 0, 1};
        const int delta_ranks[8] = {0, 1, 1, 1, 0, -1, -1, -1};
        cmz_add_offset_targets(&mask, from_file, from_rank, delta_files, delta_ranks, 8, friendly_mask);
    } else {
        const bool bishop = token == 3U || token == 9U;
        const bool rook = token == 4U || token == 10U;
        const bool queen = token == 5U || token == 11U;
        if (bishop || queen) {
            cmz_add_ray_targets(&mask, from_file, from_rank, 1, 1, friendly_mask, occupancy_mask);
            cmz_add_ray_targets(&mask, from_file, from_rank, 1, -1, friendly_mask, occupancy_mask);
            cmz_add_ray_targets(&mask, from_file, from_rank, -1, 1, friendly_mask, occupancy_mask);
            cmz_add_ray_targets(&mask, from_file, from_rank, -1, -1, friendly_mask, occupancy_mask);
        }
        if (rook || queen) {
            cmz_add_ray_targets(&mask, from_file, from_rank, 1, 0, friendly_mask, occupancy_mask);
            cmz_add_ray_targets(&mask, from_file, from_rank, -1, 0, friendly_mask, occupancy_mask);
            cmz_add_ray_targets(&mask, from_file, from_rank, 0, 1, friendly_mask, occupancy_mask);
            cmz_add_ray_targets(&mask, from_file, from_rank, 0, -1, friendly_mask, occupancy_mask);
        }
    }

    *output = mask;
}

__global__ void cmz_ray_scan_attention_kernel(
    uint32_t from_square,
    int32_t delta_file,
    int32_t delta_rank,
    uint64_t occupancy_mask,
    uint64_t* output) {
    const int from_file = static_cast<int>(from_square % 8U);
    const int from_rank = static_cast<int>(from_square / 8U);
    uint64_t mask = 0;
    int selected_blocker_step = 8;

    for (int step = 1; step <= 7; ++step) {
        const int file = from_file + static_cast<int>(delta_file) * step;
        const int rank = from_rank + static_cast<int>(delta_rank) * step;
        if (!cmz_on_board(file, rank)) {
            break;
        }
        const uint32_t square = cmz_square_index(file, rank);
        const uint64_t square_mask = cmz_bit(square);
        if ((occupancy_mask & square_mask) != 0 && step < selected_blocker_step) {
            selected_blocker_step = step;
        }
    }

    for (int step = 1; step <= 7 && step <= selected_blocker_step; ++step) {
        const int file = from_file + static_cast<int>(delta_file) * step;
        const int rank = from_rank + static_cast<int>(delta_rank) * step;
        if (!cmz_on_board(file, rank)) {
            break;
        }
        mask |= cmz_bit(cmz_square_index(file, rank));
    }
    *output = mask;
}

__device__ bool cmz_token_is_side(uint32_t token, bool white) {
    if (token == 0U) {
        return false;
    }
    return white ? token >= 1U && token <= 6U : token >= 7U && token <= 12U;
}

__device__ bool cmz_is_slider_token(uint32_t token, bool diagonal) {
    if (diagonal) {
        return token == 3U || token == 5U || token == 9U || token == 11U;
    }
    return token == 4U || token == 5U || token == 10U || token == 11U;
}

__device__ bool cmz_tokens_attacked_by(const uint32_t* board, int target, bool by_white) {
    const int target_file = target % 8;
    const int target_rank = target / 8;

    for (int square = 0; square < 64; ++square) {
        const uint32_t token = board[square];
        if (!cmz_token_is_side(token, by_white)) {
            continue;
        }
        const int file = square % 8;
        const int rank = square / 8;
        const int df = target_file - file;
        const int dr = target_rank - rank;
        if ((token == 1U || token == 7U) && dr == (by_white ? 1 : -1) && (df == 1 || df == -1)) {
            return true;
        }
        if ((token == 2U || token == 8U) &&
            ((df == 1 && dr == 2) || (df == 2 && dr == 1) || (df == 2 && dr == -1) ||
             (df == 1 && dr == -2) || (df == -1 && dr == -2) || (df == -2 && dr == -1) ||
             (df == -2 && dr == 1) || (df == -1 && dr == 2))) {
            return true;
        }
        if ((token == 6U || token == 12U) && df >= -1 && df <= 1 && dr >= -1 && dr <= 1 &&
            (df != 0 || dr != 0)) {
            return true;
        }
    }

    const int diagonal_dirs[4][2] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
    const int straight_dirs[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    for (int dir = 0; dir < 4; ++dir) {
        int file = target_file + diagonal_dirs[dir][0];
        int rank = target_rank + diagonal_dirs[dir][1];
        while (cmz_on_board(file, rank)) {
            const uint32_t token = board[cmz_square_index(file, rank)];
            if (token != 0U) {
                if (cmz_token_is_side(token, by_white) && cmz_is_slider_token(token, true)) {
                    return true;
                }
                break;
            }
            file += diagonal_dirs[dir][0];
            rank += diagonal_dirs[dir][1];
        }
    }
    for (int dir = 0; dir < 4; ++dir) {
        int file = target_file + straight_dirs[dir][0];
        int rank = target_rank + straight_dirs[dir][1];
        while (cmz_on_board(file, rank)) {
            const uint32_t token = board[cmz_square_index(file, rank)];
            if (token != 0U) {
                if (cmz_token_is_side(token, by_white) && cmz_is_slider_token(token, false)) {
                    return true;
                }
                break;
            }
            file += straight_dirs[dir][0];
            rank += straight_dirs[dir][1];
        }
    }
    return false;
}

__global__ void cmz_legal_filter_v2_move_type_select_attention_kernel(
    const uint32_t* board,
    uint32_t white_to_move,
    uint32_t from_square,
    uint32_t to_square,
    uint32_t promotion_token,
    uint32_t en_passant,
    uint32_t castle,
    uint32_t* move_meta) {
    if (threadIdx.x != 0U) {
        return;
    }
    const bool white = white_to_move != 0U;
    move_meta[0] = 64U;
    move_meta[1] = 64U;
    move_meta[2] = 64U;
    move_meta[3] = 0U;
    move_meta[4] = promotion_token;
    move_meta[5] = board[from_square];
    if (en_passant != 0U) {
        const int captured = white ? static_cast<int>(to_square) - 8 : static_cast<int>(to_square) + 8;
        if (captured >= 0 && captured < 64) {
            move_meta[0] = static_cast<uint32_t>(captured);
        }
    }
    if (castle != 0U) {
        if (from_square == 4U && to_square == 6U) {
            move_meta[1] = 7U;
            move_meta[2] = 5U;
            move_meta[3] = 4U;
        } else if (from_square == 4U && to_square == 2U) {
            move_meta[1] = 0U;
            move_meta[2] = 3U;
            move_meta[3] = 4U;
        } else if (from_square == 60U && to_square == 62U) {
            move_meta[1] = 63U;
            move_meta[2] = 61U;
            move_meta[3] = 10U;
        } else if (from_square == 60U && to_square == 58U) {
            move_meta[1] = 56U;
            move_meta[2] = 59U;
            move_meta[3] = 10U;
        }
    }
}

__global__ void cmz_legal_filter_v2_board_write_select_attention_kernel(
    const uint32_t* board,
    uint32_t from_square,
    uint32_t to_square,
    const uint32_t* move_meta,
    uint32_t* next_board) {
    const uint32_t square = threadIdx.x;
    if (square >= 64U) {
        return;
    }

    const uint32_t moving = move_meta[5];
    uint32_t selected_score = 0U;
    uint32_t selected_tie_index = 0xffffffffU;
    uint32_t selected_value = board[square];

    if (square == from_square) {
        cmz_qk2_hardmax_select_u32(1U, 0U, 10U, 0U, 10U, 0U, &selected_score, &selected_tie_index, &selected_value);
    }
    if (square == to_square) {
        cmz_qk2_hardmax_select_u32(
            1U,
            0U,
            20U,
            0U,
            20U,
            moving,
            &selected_score,
            &selected_tie_index,
            &selected_value);
    }

    next_board[square] = selected_value;
}

__global__ void cmz_legal_filter_v2_en_passant_capture_select_attention_kernel(
    uint32_t* next_board,
    const uint32_t* move_meta) {
    const uint32_t square = threadIdx.x;
    if (square < 64U && square == move_meta[0]) {
        next_board[square] = 0U;
    }
}

__global__ void cmz_legal_filter_v2_castle_rook_write_select_attention_kernel(
    uint32_t* next_board,
    const uint32_t* move_meta) {
    const uint32_t square = threadIdx.x;
    if (square >= 64U) {
        return;
    }
    if (square == move_meta[1]) {
        next_board[square] = 0U;
    }
    if (square == move_meta[2]) {
        next_board[square] = move_meta[3];
    }
}

__global__ void cmz_legal_filter_v2_promotion_select_attention_kernel(
    uint32_t* next_board,
    uint32_t to_square,
    const uint32_t* move_meta) {
    const uint32_t square = threadIdx.x;
    if (square < 64U && move_meta[4] != 0U && square == to_square) {
        next_board[square] = move_meta[4];
    }
}

__global__ void cmz_make_move_metadata_side_toggle_select_attention_kernel(
    uint32_t white_to_move,
    uint32_t* next_metadata) {
    if (threadIdx.x == 0U && blockIdx.x == 0U) {
        next_metadata[0] = white_to_move == 0U ? 1U : 0U;
    }
}

__global__ void cmz_make_move_metadata_castling_rights_select_attention_kernel(
    uint32_t castling_rights,
    uint32_t moving_piece_token,
    uint32_t from_square,
    uint32_t to_square,
    uint32_t* next_metadata) {
    if (threadIdx.x != 0U || blockIdx.x != 0U) {
        return;
    }
    uint32_t rights = castling_rights & 15U;
    if (moving_piece_token == 6U) {
        rights &= ~(1U | 2U);
    } else if (moving_piece_token == 12U) {
        rights &= ~(4U | 8U);
    }
    if (from_square == 0U || to_square == 0U) {
        rights &= ~2U;
    }
    if (from_square == 7U || to_square == 7U) {
        rights &= ~1U;
    }
    if (from_square == 56U || to_square == 56U) {
        rights &= ~8U;
    }
    if (from_square == 63U || to_square == 63U) {
        rights &= ~4U;
    }
    next_metadata[1] = rights;
}

__global__ void cmz_make_move_metadata_ep_square_select_attention_kernel(
    uint32_t moving_piece_token,
    uint32_t from_square,
    uint32_t to_square,
    uint32_t* next_metadata) {
    if (threadIdx.x != 0U || blockIdx.x != 0U) {
        return;
    }
    const bool pawn = moving_piece_token == 1U || moving_piece_token == 7U;
    const uint32_t distance = from_square > to_square ? from_square - to_square : to_square - from_square;
    next_metadata[2] = pawn && distance == 16U ? (from_square + to_square) / 2U : 64U;
}

__global__ void cmz_make_move_metadata_halfmove_clock_select_attention_kernel(
    uint32_t halfmove_clock,
    uint32_t moving_piece_token,
    uint32_t target_piece_token,
    uint32_t en_passant,
    uint32_t* next_metadata) {
    if (threadIdx.x != 0U || blockIdx.x != 0U) {
        return;
    }
    const bool pawn = moving_piece_token == 1U || moving_piece_token == 7U;
    const bool capture = target_piece_token != 0U || en_passant != 0U;
    next_metadata[3] = pawn || capture ? 0U : halfmove_clock + 1U;
}

__global__ void cmz_make_move_metadata_fullmove_number_select_attention_kernel(
    uint32_t white_to_move,
    uint32_t fullmove_number,
    uint32_t* next_metadata) {
    if (threadIdx.x == 0U && blockIdx.x == 0U) {
        next_metadata[4] = fullmove_number + (white_to_move == 0U ? 1U : 0U);
    }
}

__global__ void cmz_legal_filter_v2_king_square_select_attention_kernel(
    const uint32_t* board,
    uint32_t white_to_move,
    uint32_t* king_square,
    uint32_t* king_found) {
    const uint32_t king_token = white_to_move != 0U ? 6U : 12U;
    uint32_t selected_square = 0U;
    uint32_t selected_score = 0U;
    uint32_t selected_tie_index = 0xffffffffU;
    for (uint32_t square = 0; square < 64U; ++square) {
        const uint32_t key_x = board[square] == king_token ? 1U : 0U;
        cmz_qk2_hardmax_select_u32(
            1U,
            0U,
            key_x,
            0U,
            square,
            square,
            &selected_score,
            &selected_tie_index,
            &selected_square);
    }
    *king_square = selected_square;
    *king_found = selected_score;
}

__global__ void cmz_legal_filter_v2_attack_source_select_attention_kernel(
    const uint32_t* board,
    const uint32_t* king_square,
    uint32_t attacker_white,
    uint32_t* attacked) {
    const uint32_t target = *king_square;
    const int target_file = static_cast<int>(target % 8U);
    const int target_rank = static_cast<int>(target / 8U);
    uint32_t selected_score = 0U;
    uint32_t selected_tie_index = 0xffffffffU;
    uint32_t selected_value = 0U;
    for (uint32_t square = 0; square < 64U; ++square) {
        const uint32_t token = board[square];
        if (!cmz_token_is_side(token, attacker_white != 0U)) {
            continue;
        }
        const int file = static_cast<int>(square % 8U);
        const int rank = static_cast<int>(square / 8U);
        const int df = target_file - file;
        const int dr = target_rank - rank;
        uint32_t score = 0U;
        if ((token == 1U || token == 7U) && dr == (attacker_white != 0U ? 1 : -1) && (df == 1 || df == -1)) {
            score = 1U;
        } else if ((token == 2U || token == 8U) &&
                   ((df == 1 && dr == 2) || (df == 2 && dr == 1) || (df == 2 && dr == -1) ||
                    (df == 1 && dr == -2) || (df == -1 && dr == -2) || (df == -2 && dr == -1) ||
                    (df == -2 && dr == 1) || (df == -1 && dr == 2))) {
            score = 1U;
        } else if ((token == 6U || token == 12U) && df >= -1 && df <= 1 && dr >= -1 && dr <= 1 &&
                   (df != 0 || dr != 0)) {
            score = 1U;
        }
        cmz_qk2_hardmax_select_u32(
            1U,
            0U,
            score,
            0U,
            square,
            score,
            &selected_score,
            &selected_tie_index,
            &selected_value);
    }
    *attacked = selected_score;
}

__global__ void cmz_legal_filter_v2_ray_blocker_select_attention_kernel(
    const uint32_t* board,
    const uint32_t* king_square,
    uint32_t attacker_white,
    uint32_t* attacked) {
    const int dir = static_cast<int>(threadIdx.x);
    if (dir >= 8) {
        return;
    }
    const int delta_files[8] = {1, 1, -1, -1, 1, -1, 0, 0};
    const int delta_ranks[8] = {1, -1, 1, -1, 0, 0, 1, -1};
    const bool diagonal = dir < 4;
    const uint32_t target = *king_square;
    const int target_file = static_cast<int>(target % 8U);
    const int target_rank = static_cast<int>(target / 8U);
    uint32_t selected_token = 0U;
    uint32_t selected_score = 0U;
    uint32_t selected_tie_index = 0xffffffffU;

    for (int step = 1; step <= 7; ++step) {
        const int file = target_file + delta_files[dir] * step;
        const int rank = target_rank + delta_ranks[dir] * step;
        if (!cmz_on_board(file, rank)) {
            break;
        }
        const uint32_t square = cmz_square_index(file, rank);
        const uint32_t token = board[square];
        const uint32_t score = token == 0U ? 0U : static_cast<uint32_t>(8 - step);
        cmz_qk2_hardmax_select_u32(
            1U,
            0U,
            score,
            0U,
            static_cast<uint32_t>(step),
            token,
            &selected_score,
            &selected_tie_index,
            &selected_token);
    }

    if (selected_score != 0U && cmz_token_is_side(selected_token, attacker_white != 0U) &&
        cmz_is_slider_token(selected_token, diagonal)) {
        atomicExch(attacked, 1U);
    }
}

__global__ void cmz_legal_filter_v2_final_legal_select_attention_kernel(
    const uint32_t* king_found,
    const uint32_t* short_attacked,
    const uint32_t* ray_attacked,
    uint32_t* legal) {
    *legal = *king_found != 0U && *short_attacked == 0U && *ray_attacked == 0U ? 1U : 0U;
}

__global__ void cmz_legal_filter_v2_batch_move_type_select_attention_kernel(
    const uint32_t* board,
    uint32_t white_to_move,
    const uint32_t* from_squares,
    const uint32_t* to_squares,
    const uint32_t* promotion_tokens,
    const uint32_t* en_passant,
    const uint32_t* castle,
    uint32_t* move_meta,
    size_t move_count) {
    const size_t move_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (move_index >= move_count) {
        return;
    }
    const uint32_t from_square = from_squares[move_index];
    const uint32_t to_square = to_squares[move_index];
    uint32_t* meta = move_meta + move_index * 6U;
    const bool white = white_to_move != 0U;
    meta[0] = 64U;
    meta[1] = 64U;
    meta[2] = 64U;
    meta[3] = 0U;
    meta[4] = promotion_tokens[move_index];
    meta[5] = board[from_square];
    if (en_passant[move_index] != 0U) {
        const int captured = white ? static_cast<int>(to_square) - 8 : static_cast<int>(to_square) + 8;
        if (captured >= 0 && captured < 64) {
            meta[0] = static_cast<uint32_t>(captured);
        }
    }
    if (castle[move_index] != 0U) {
        if (from_square == 4U && to_square == 6U) {
            meta[1] = 7U;
            meta[2] = 5U;
            meta[3] = 4U;
        } else if (from_square == 4U && to_square == 2U) {
            meta[1] = 0U;
            meta[2] = 3U;
            meta[3] = 4U;
        } else if (from_square == 60U && to_square == 62U) {
            meta[1] = 63U;
            meta[2] = 61U;
            meta[3] = 10U;
        } else if (from_square == 60U && to_square == 58U) {
            meta[1] = 56U;
            meta[2] = 59U;
            meta[3] = 10U;
        }
    }
}

__global__ void cmz_legal_filter_v2_batch_board_write_select_attention_kernel(
    const uint32_t* board,
    const uint32_t* from_squares,
    const uint32_t* to_squares,
    const uint32_t* move_meta,
    uint32_t* next_boards,
    size_t move_count) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = move_count * 64U;
    if (index >= total) {
        return;
    }
    const size_t move_index = index / 64U;
    const uint32_t square = static_cast<uint32_t>(index % 64U);
    const uint32_t from_square = from_squares[move_index];
    const uint32_t to_square = to_squares[move_index];
    const uint32_t* meta = move_meta + move_index * 6U;
    const uint32_t moving = meta[5];
    uint32_t selected_score = 0U;
    uint32_t selected_tie_index = 0xffffffffU;
    uint32_t selected_value = board[square];

    if (square == from_square) {
        cmz_qk2_hardmax_select_u32(1U, 0U, 10U, 0U, 10U, 0U, &selected_score, &selected_tie_index, &selected_value);
    }
    if (square == to_square) {
        cmz_qk2_hardmax_select_u32(
            1U,
            0U,
            20U,
            0U,
            20U,
            moving,
            &selected_score,
            &selected_tie_index,
            &selected_value);
    }

    next_boards[move_index * 64U + square] = selected_value;
}

__global__ void cmz_legal_filter_v2_batch_en_passant_capture_select_attention_kernel(
    uint32_t* next_boards,
    const uint32_t* move_meta,
    size_t move_count) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = move_count * 64U;
    if (index >= total) {
        return;
    }
    const size_t move_index = index / 64U;
    const uint32_t square = static_cast<uint32_t>(index % 64U);
    const uint32_t captured_square = move_meta[move_index * 6U];
    if (square == captured_square) {
        next_boards[move_index * 64U + square] = 0U;
    }
}

__global__ void cmz_legal_filter_v2_batch_castle_rook_write_select_attention_kernel(
    uint32_t* next_boards,
    const uint32_t* move_meta,
    size_t move_count) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = move_count * 64U;
    if (index >= total) {
        return;
    }
    const size_t move_index = index / 64U;
    const uint32_t square = static_cast<uint32_t>(index % 64U);
    const uint32_t* meta = move_meta + move_index * 6U;
    if (square == meta[1]) {
        next_boards[move_index * 64U + square] = 0U;
    }
    if (square == meta[2]) {
        next_boards[move_index * 64U + square] = meta[3];
    }
}

__global__ void cmz_legal_filter_v2_batch_promotion_select_attention_kernel(
    uint32_t* next_boards,
    const uint32_t* to_squares,
    const uint32_t* move_meta,
    size_t move_count) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = move_count * 64U;
    if (index >= total) {
        return;
    }
    const size_t move_index = index / 64U;
    const uint32_t square = static_cast<uint32_t>(index % 64U);
    const uint32_t promotion_token = move_meta[move_index * 6U + 4U];
    if (promotion_token != 0U && square == to_squares[move_index]) {
        next_boards[move_index * 64U + square] = promotion_token;
    }
}

__global__ void cmz_legal_filter_v2_batch_king_square_select_attention_kernel(
    const uint32_t* next_boards,
    uint32_t white_to_move,
    uint32_t* king_squares,
    uint32_t* king_found,
    size_t move_count) {
    const size_t move_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (move_index >= move_count) {
        return;
    }
    const uint32_t* board = next_boards + move_index * 64U;
    const uint32_t king_token = white_to_move != 0U ? 6U : 12U;
    uint32_t selected_square = 0U;
    uint32_t selected_score = 0U;
    uint32_t selected_tie_index = 0xffffffffU;
    for (uint32_t square = 0; square < 64U; ++square) {
        const uint32_t key_x = board[square] == king_token ? 1U : 0U;
        cmz_qk2_hardmax_select_u32(
            1U,
            0U,
            key_x,
            0U,
            square,
            square,
            &selected_score,
            &selected_tie_index,
            &selected_square);
    }
    king_squares[move_index] = selected_square;
    king_found[move_index] = selected_score;
}

__global__ void cmz_legal_filter_v2_batch_attack_source_select_attention_kernel(
    const uint32_t* next_boards,
    const uint32_t* king_squares,
    uint32_t attacker_white,
    uint32_t* attacked,
    size_t move_count) {
    const size_t move_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (move_index >= move_count) {
        return;
    }
    const uint32_t* board = next_boards + move_index * 64U;
    const uint32_t target = king_squares[move_index];
    const int target_file = static_cast<int>(target % 8U);
    const int target_rank = static_cast<int>(target / 8U);
    uint32_t selected_score = 0U;
    uint32_t selected_tie_index = 0xffffffffU;
    uint32_t selected_value = 0U;
    for (uint32_t square = 0; square < 64U; ++square) {
        const uint32_t token = board[square];
        if (!cmz_token_is_side(token, attacker_white != 0U)) {
            continue;
        }
        const int file = static_cast<int>(square % 8U);
        const int rank = static_cast<int>(square / 8U);
        const int df = target_file - file;
        const int dr = target_rank - rank;
        uint32_t score = 0U;
        if ((token == 1U || token == 7U) && dr == (attacker_white != 0U ? 1 : -1) && (df == 1 || df == -1)) {
            score = 1U;
        } else if ((token == 2U || token == 8U) &&
                   ((df == 1 && dr == 2) || (df == 2 && dr == 1) || (df == 2 && dr == -1) ||
                    (df == 1 && dr == -2) || (df == -1 && dr == -2) || (df == -2 && dr == -1) ||
                    (df == -2 && dr == 1) || (df == -1 && dr == 2))) {
            score = 1U;
        } else if ((token == 6U || token == 12U) && df >= -1 && df <= 1 && dr >= -1 && dr <= 1 &&
                   (df != 0 || dr != 0)) {
            score = 1U;
        }
        cmz_qk2_hardmax_select_u32(
            1U,
            0U,
            score,
            0U,
            square,
            score,
            &selected_score,
            &selected_tie_index,
            &selected_value);
    }
    attacked[move_index] = selected_score;
}

__global__ void cmz_legal_filter_v2_batch_ray_blocker_select_attention_kernel(
    const uint32_t* next_boards,
    const uint32_t* king_squares,
    uint32_t attacker_white,
    uint32_t* attacked,
    size_t move_count) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = move_count * 8U;
    if (index >= total) {
        return;
    }
    const size_t move_index = index / 8U;
    const int dir = static_cast<int>(index % 8U);
    const uint32_t* board = next_boards + move_index * 64U;
    const int delta_files[8] = {1, 1, -1, -1, 1, -1, 0, 0};
    const int delta_ranks[8] = {1, -1, 1, -1, 0, 0, 1, -1};
    const bool diagonal = dir < 4;
    const uint32_t target = king_squares[move_index];
    const int target_file = static_cast<int>(target % 8U);
    const int target_rank = static_cast<int>(target / 8U);
    uint32_t selected_token = 0U;
    uint32_t selected_score = 0U;
    uint32_t selected_tie_index = 0xffffffffU;

    for (int step = 1; step <= 7; ++step) {
        const int file = target_file + delta_files[dir] * step;
        const int rank = target_rank + delta_ranks[dir] * step;
        if (!cmz_on_board(file, rank)) {
            break;
        }
        const uint32_t square = cmz_square_index(file, rank);
        const uint32_t token = board[square];
        const uint32_t score = token == 0U ? 0U : static_cast<uint32_t>(8 - step);
        cmz_qk2_hardmax_select_u32(
            1U,
            0U,
            score,
            0U,
            static_cast<uint32_t>(step),
            token,
            &selected_score,
            &selected_tie_index,
            &selected_token);
    }

    if (selected_score != 0U && cmz_token_is_side(selected_token, attacker_white != 0U) &&
        cmz_is_slider_token(selected_token, diagonal)) {
        atomicExch(attacked + move_index, 1U);
    }
}

__global__ void cmz_legal_filter_v2_batch_final_legal_select_attention_kernel(
    const uint32_t* king_found,
    const uint32_t* short_attacked,
    const uint32_t* ray_attacked,
    uint32_t* legal,
    size_t move_count) {
    const size_t move_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (move_index < move_count) {
        legal[move_index] =
            king_found[move_index] != 0U && short_attacked[move_index] == 0U && ray_attacked[move_index] == 0U ? 1U : 0U;
    }
}

__global__ void cmz_castle_target_attention_kernel(
    const uint32_t* board,
    uint32_t castling_rights,
    uint32_t white_value,
    uint64_t* output) {
    const bool white = white_value != 0U;
    uint64_t mask = 0;
    if (white) {
        if (board[4] == 6U && !cmz_tokens_attacked_by(board, 4, false)) {
            if ((castling_rights & 1U) != 0U && board[5] == 0U && board[6] == 0U && board[7] == 4U &&
                !cmz_tokens_attacked_by(board, 5, false) && !cmz_tokens_attacked_by(board, 6, false)) {
                mask |= cmz_bit(6U);
            }
            if ((castling_rights & 2U) != 0U && board[3] == 0U && board[2] == 0U && board[1] == 0U &&
                board[0] == 4U && !cmz_tokens_attacked_by(board, 3, false) &&
                !cmz_tokens_attacked_by(board, 2, false)) {
                mask |= cmz_bit(2U);
            }
        }
    } else {
        if (board[60] == 12U && !cmz_tokens_attacked_by(board, 60, true)) {
            if ((castling_rights & 4U) != 0U && board[61] == 0U && board[62] == 0U && board[63] == 10U &&
                !cmz_tokens_attacked_by(board, 61, true) && !cmz_tokens_attacked_by(board, 62, true)) {
                mask |= cmz_bit(62U);
            }
            if ((castling_rights & 8U) != 0U && board[59] == 0U && board[58] == 0U && board[57] == 0U &&
                board[56] == 10U && !cmz_tokens_attacked_by(board, 59, true) &&
                !cmz_tokens_attacked_by(board, 58, true)) {
                mask |= cmz_bit(58U);
            }
        }
    }
    *output = mask;
}

__device__ uint32_t cmz_candidate_piece_family_attention_query(uint32_t token) {
    const uint32_t pawn = (token == 1U || token == 7U) ? 1U : 0U;
    const uint32_t knight = (token == 2U || token == 8U) ? 2U : 0U;
    const uint32_t king = (token == 6U || token == 12U) ? 3U : 0U;
    const uint32_t slider = (token == 3U || token == 4U || token == 5U || token == 9U || token == 10U || token == 11U)
                                ? 4U
                                : 0U;
    return pawn | knight | king | slider;
}

__device__ uint64_t cmz_candidate_single_offset_coordinate_table_attention_value(int file, int rank);

__device__ uint64_t cmz_candidate_pawn_forward_target_slot_attention_value(
    int from_file,
    int from_rank,
    int direction,
    int step_count) {
    return cmz_candidate_single_offset_coordinate_table_attention_value(from_file, from_rank + direction * step_count);
}

#define CMZ_PAWN_PUSH_EMPTY_ATTEND_SQUARE(SQUARE) \
    cmz_qk2_select_or_write_u64( \
        target_slot == cmz_bit((SQUARE)) ? 1U : 0U, \
        0U, \
        1U, \
        0U, \
        (occupancy_mask & cmz_bit((SQUARE))) == 0ULL ? cmz_bit((SQUARE)) : 0ULL, \
        &selected_value)

#define CMZ_PAWN_PUSH_EMPTY_ATTEND_ROW(BASE) \
    CMZ_PAWN_PUSH_EMPTY_ATTEND_SQUARE((BASE) + 0U); \
    CMZ_PAWN_PUSH_EMPTY_ATTEND_SQUARE((BASE) + 1U); \
    CMZ_PAWN_PUSH_EMPTY_ATTEND_SQUARE((BASE) + 2U); \
    CMZ_PAWN_PUSH_EMPTY_ATTEND_SQUARE((BASE) + 3U); \
    CMZ_PAWN_PUSH_EMPTY_ATTEND_SQUARE((BASE) + 4U); \
    CMZ_PAWN_PUSH_EMPTY_ATTEND_SQUARE((BASE) + 5U); \
    CMZ_PAWN_PUSH_EMPTY_ATTEND_SQUARE((BASE) + 6U); \
    CMZ_PAWN_PUSH_EMPTY_ATTEND_SQUARE((BASE) + 7U)

__device__ uint64_t cmz_candidate_pawn_push_empty_condition_attention_value(
    uint64_t target_slot,
    uint64_t occupancy_mask) {
    uint64_t selected_value = 0ULL;
    CMZ_PAWN_PUSH_EMPTY_ATTEND_ROW(0U);
    CMZ_PAWN_PUSH_EMPTY_ATTEND_ROW(8U);
    CMZ_PAWN_PUSH_EMPTY_ATTEND_ROW(16U);
    CMZ_PAWN_PUSH_EMPTY_ATTEND_ROW(24U);
    CMZ_PAWN_PUSH_EMPTY_ATTEND_ROW(32U);
    CMZ_PAWN_PUSH_EMPTY_ATTEND_ROW(40U);
    CMZ_PAWN_PUSH_EMPTY_ATTEND_ROW(48U);
    CMZ_PAWN_PUSH_EMPTY_ATTEND_ROW(56U);
    return selected_value;
}

constexpr uint32_t kCmzPawnRankPairStride = 8U;
constexpr uint32_t kCmzPawnRankPairMaxCode = 63U;
constexpr uint32_t kCmzPawnRankPairScoreBias = kCmzPawnRankPairMaxCode * kCmzPawnRankPairMaxCode;

__device__ uint32_t cmz_pawn_rank_pair_code(uint32_t from_rank, uint32_t start_rank) {
    return from_rank * kCmzPawnRankPairStride + start_rank;
}

__device__ uint32_t cmz_pawn_rank_pair_key_bias(uint32_t code) {
    return kCmzPawnRankPairScoreBias - code * code;
}

#define CMZ_PAWN_START_RANK_ATTEND_ENTRY(FROM_RANK, START_RANK, VALUE) \
    cmz_qk2_hardmax_select_u32( \
        query_x, \
        query_y, \
        cmz_pawn_rank_pair_code((FROM_RANK), (START_RANK)), \
        cmz_pawn_rank_pair_key_bias(cmz_pawn_rank_pair_code((FROM_RANK), (START_RANK))), \
        cmz_pawn_rank_pair_code((FROM_RANK), (START_RANK)), \
        (VALUE), \
        &selected_score, \
        &selected_tie_index, \
        &selected_value)

#define CMZ_PAWN_START_RANK_ATTEND_ROW(FROM_RANK, VALUE_0, VALUE_1, VALUE_2, VALUE_3, VALUE_4, VALUE_5, VALUE_6, VALUE_7) \
    CMZ_PAWN_START_RANK_ATTEND_ENTRY((FROM_RANK), 0U, (VALUE_0)); \
    CMZ_PAWN_START_RANK_ATTEND_ENTRY((FROM_RANK), 1U, (VALUE_1)); \
    CMZ_PAWN_START_RANK_ATTEND_ENTRY((FROM_RANK), 2U, (VALUE_2)); \
    CMZ_PAWN_START_RANK_ATTEND_ENTRY((FROM_RANK), 3U, (VALUE_3)); \
    CMZ_PAWN_START_RANK_ATTEND_ENTRY((FROM_RANK), 4U, (VALUE_4)); \
    CMZ_PAWN_START_RANK_ATTEND_ENTRY((FROM_RANK), 5U, (VALUE_5)); \
    CMZ_PAWN_START_RANK_ATTEND_ENTRY((FROM_RANK), 6U, (VALUE_6)); \
    CMZ_PAWN_START_RANK_ATTEND_ENTRY((FROM_RANK), 7U, (VALUE_7))

__device__ uint32_t cmz_candidate_pawn_start_rank_match_attention_value(int from_rank, int start_rank) {
    const uint32_t query_code =
        cmz_pawn_rank_pair_code(static_cast<uint32_t>(from_rank), static_cast<uint32_t>(start_rank));
    const uint32_t query_x = query_code * 2U;
    const uint32_t query_y = 1U;
    uint32_t selected_score = 0U;
    uint32_t selected_tie_index = 0xffffffffU;
    uint32_t selected_value = 0U;

    CMZ_PAWN_START_RANK_ATTEND_ROW(0U, 1U, 0U, 0U, 0U, 0U, 0U, 0U, 0U);
    CMZ_PAWN_START_RANK_ATTEND_ROW(1U, 0U, 1U, 0U, 0U, 0U, 0U, 0U, 0U);
    CMZ_PAWN_START_RANK_ATTEND_ROW(2U, 0U, 0U, 1U, 0U, 0U, 0U, 0U, 0U);
    CMZ_PAWN_START_RANK_ATTEND_ROW(3U, 0U, 0U, 0U, 1U, 0U, 0U, 0U, 0U);
    CMZ_PAWN_START_RANK_ATTEND_ROW(4U, 0U, 0U, 0U, 0U, 1U, 0U, 0U, 0U);
    CMZ_PAWN_START_RANK_ATTEND_ROW(5U, 0U, 0U, 0U, 0U, 0U, 1U, 0U, 0U);
    CMZ_PAWN_START_RANK_ATTEND_ROW(6U, 0U, 0U, 0U, 0U, 0U, 0U, 1U, 0U);
    CMZ_PAWN_START_RANK_ATTEND_ROW(7U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 1U);
    return selected_value;
}

#define CMZ_PAWN_SINGLE_PUSH_NONZERO_ATTEND_SQUARE(SQUARE) \
    cmz_qk2_select_or_write_u64( \
        single_push_mask == cmz_bit((SQUARE)) ? 1U : 0U, \
        0U, \
        1U, \
        0U, \
        1ULL, \
        &selected_value)

#define CMZ_PAWN_SINGLE_PUSH_NONZERO_ATTEND_ROW(BASE) \
    CMZ_PAWN_SINGLE_PUSH_NONZERO_ATTEND_SQUARE((BASE) + 0U); \
    CMZ_PAWN_SINGLE_PUSH_NONZERO_ATTEND_SQUARE((BASE) + 1U); \
    CMZ_PAWN_SINGLE_PUSH_NONZERO_ATTEND_SQUARE((BASE) + 2U); \
    CMZ_PAWN_SINGLE_PUSH_NONZERO_ATTEND_SQUARE((BASE) + 3U); \
    CMZ_PAWN_SINGLE_PUSH_NONZERO_ATTEND_SQUARE((BASE) + 4U); \
    CMZ_PAWN_SINGLE_PUSH_NONZERO_ATTEND_SQUARE((BASE) + 5U); \
    CMZ_PAWN_SINGLE_PUSH_NONZERO_ATTEND_SQUARE((BASE) + 6U); \
    CMZ_PAWN_SINGLE_PUSH_NONZERO_ATTEND_SQUARE((BASE) + 7U)

__device__ uint32_t cmz_candidate_pawn_single_push_nonzero_attention_value(uint64_t single_push_mask) {
    uint64_t selected_value = 0ULL;
    CMZ_PAWN_SINGLE_PUSH_NONZERO_ATTEND_ROW(0U);
    CMZ_PAWN_SINGLE_PUSH_NONZERO_ATTEND_ROW(8U);
    CMZ_PAWN_SINGLE_PUSH_NONZERO_ATTEND_ROW(16U);
    CMZ_PAWN_SINGLE_PUSH_NONZERO_ATTEND_ROW(24U);
    CMZ_PAWN_SINGLE_PUSH_NONZERO_ATTEND_ROW(32U);
    CMZ_PAWN_SINGLE_PUSH_NONZERO_ATTEND_ROW(40U);
    CMZ_PAWN_SINGLE_PUSH_NONZERO_ATTEND_ROW(48U);
    CMZ_PAWN_SINGLE_PUSH_NONZERO_ATTEND_ROW(56U);
    return static_cast<uint32_t>(selected_value);
}

#define CMZ_PAWN_DOUBLE_PUSH_ATTEND_SQUARE(SQUARE) \
    cmz_qk2_select_or_write_u64( \
        target_slot == cmz_bit((SQUARE)) ? 1U : 0U, \
        0U, \
        rank_start_condition & first_step_condition, \
        0U, \
        (occupancy_mask & cmz_bit((SQUARE))) == 0ULL ? cmz_bit((SQUARE)) : 0ULL, \
        &selected_value)

#define CMZ_PAWN_DOUBLE_PUSH_ATTEND_ROW(BASE) \
    CMZ_PAWN_DOUBLE_PUSH_ATTEND_SQUARE((BASE) + 0U); \
    CMZ_PAWN_DOUBLE_PUSH_ATTEND_SQUARE((BASE) + 1U); \
    CMZ_PAWN_DOUBLE_PUSH_ATTEND_SQUARE((BASE) + 2U); \
    CMZ_PAWN_DOUBLE_PUSH_ATTEND_SQUARE((BASE) + 3U); \
    CMZ_PAWN_DOUBLE_PUSH_ATTEND_SQUARE((BASE) + 4U); \
    CMZ_PAWN_DOUBLE_PUSH_ATTEND_SQUARE((BASE) + 5U); \
    CMZ_PAWN_DOUBLE_PUSH_ATTEND_SQUARE((BASE) + 6U); \
    CMZ_PAWN_DOUBLE_PUSH_ATTEND_SQUARE((BASE) + 7U)

__device__ uint64_t cmz_candidate_pawn_double_push_condition_attention_value(
    uint32_t rank_start_condition,
    uint32_t first_step_condition,
    uint64_t occupancy_mask,
    uint64_t target_slot) {
    uint64_t selected_value = 0ULL;
    CMZ_PAWN_DOUBLE_PUSH_ATTEND_ROW(0U);
    CMZ_PAWN_DOUBLE_PUSH_ATTEND_ROW(8U);
    CMZ_PAWN_DOUBLE_PUSH_ATTEND_ROW(16U);
    CMZ_PAWN_DOUBLE_PUSH_ATTEND_ROW(24U);
    CMZ_PAWN_DOUBLE_PUSH_ATTEND_ROW(32U);
    CMZ_PAWN_DOUBLE_PUSH_ATTEND_ROW(40U);
    CMZ_PAWN_DOUBLE_PUSH_ATTEND_ROW(48U);
    CMZ_PAWN_DOUBLE_PUSH_ATTEND_ROW(56U);
    return selected_value;
}

__device__ uint64_t cmz_candidate_pawn_single_push_slot_attention_value(
    int from_file,
    int from_rank,
    int direction,
    uint64_t occupancy_mask) {
    const uint64_t target_slot =
        cmz_candidate_pawn_forward_target_slot_attention_value(from_file, from_rank, direction, 1);
    return cmz_candidate_pawn_push_empty_condition_attention_value(target_slot, occupancy_mask);
}

__device__ uint64_t cmz_candidate_pawn_double_push_slot_attention_value(
    int from_file,
    int from_rank,
    int direction,
    int start_rank,
    uint64_t occupancy_mask,
    uint64_t single_push_mask) {
    const uint64_t target_slot =
        cmz_candidate_pawn_forward_target_slot_attention_value(from_file, from_rank, direction, 2);
    const uint32_t rank_start_condition =
        cmz_candidate_pawn_start_rank_match_attention_value(from_rank, start_rank);
    const uint32_t first_step_condition =
        cmz_candidate_pawn_single_push_nonzero_attention_value(single_push_mask);
    return cmz_candidate_pawn_double_push_condition_attention_value(
        rank_start_condition, first_step_condition, occupancy_mask, target_slot);
}

__device__ uint64_t cmz_candidate_pawn_capture_target_slot_attention_value(
    int from_file,
    int from_rank,
    int direction,
    int delta_file) {
    return cmz_candidate_single_offset_coordinate_table_attention_value(from_file + delta_file, from_rank + direction);
}

__device__ uint64_t cmz_candidate_pawn_capture_enemy_condition_attention_value(
    uint64_t target_slot,
    uint64_t enemy_mask) {
    const uint32_t enemy_present = (enemy_mask & target_slot) != 0ULL ? 1U : 0U;
    uint64_t output = 0ULL;
    cmz_qk2_select_or_write_u64(1U, 0U, enemy_present, 0U, target_slot, &output);
    return output;
}

#define CMZ_PAWN_EP_TARGET_ATTEND_SQUARE(SQUARE) \
    cmz_qk2_select_or_write_u64( \
        (ep_square == (SQUARE) && target_slot == cmz_bit((SQUARE))) ? 1U : 0U, \
        0U, \
        1U, \
        0U, \
        cmz_bit((SQUARE)), \
        &selected_value)

#define CMZ_PAWN_EP_TARGET_ATTEND_ROW(BASE) \
    CMZ_PAWN_EP_TARGET_ATTEND_SQUARE((BASE) + 0U); \
    CMZ_PAWN_EP_TARGET_ATTEND_SQUARE((BASE) + 1U); \
    CMZ_PAWN_EP_TARGET_ATTEND_SQUARE((BASE) + 2U); \
    CMZ_PAWN_EP_TARGET_ATTEND_SQUARE((BASE) + 3U); \
    CMZ_PAWN_EP_TARGET_ATTEND_SQUARE((BASE) + 4U); \
    CMZ_PAWN_EP_TARGET_ATTEND_SQUARE((BASE) + 5U); \
    CMZ_PAWN_EP_TARGET_ATTEND_SQUARE((BASE) + 6U); \
    CMZ_PAWN_EP_TARGET_ATTEND_SQUARE((BASE) + 7U)

__device__ uint64_t cmz_candidate_pawn_ep_target_match_attention_value(
    uint64_t target_slot,
    uint32_t ep_square) {
    uint64_t selected_value = 0ULL;
    CMZ_PAWN_EP_TARGET_ATTEND_ROW(0U);
    CMZ_PAWN_EP_TARGET_ATTEND_ROW(8U);
    CMZ_PAWN_EP_TARGET_ATTEND_ROW(16U);
    CMZ_PAWN_EP_TARGET_ATTEND_ROW(24U);
    CMZ_PAWN_EP_TARGET_ATTEND_ROW(32U);
    CMZ_PAWN_EP_TARGET_ATTEND_ROW(40U);
    CMZ_PAWN_EP_TARGET_ATTEND_ROW(48U);
    CMZ_PAWN_EP_TARGET_ATTEND_ROW(56U);
    return selected_value;
}

constexpr uint32_t kCmzPawnEpSideSquareStride = 64U;
constexpr uint32_t kCmzPawnEpSideSquareMaxCode = 511U;
constexpr uint32_t kCmzPawnEpSideSquareScoreBias =
    kCmzPawnEpSideSquareMaxCode * kCmzPawnEpSideSquareMaxCode;

__device__ uint32_t cmz_pawn_ep_side_square_code(uint32_t side_token, uint32_t ep_square) {
    return side_token * kCmzPawnEpSideSquareStride + ep_square;
}

__device__ uint32_t cmz_pawn_ep_side_square_key_bias(uint32_t code) {
    return kCmzPawnEpSideSquareScoreBias - code * code;
}

#define CMZ_PAWN_EP_CAPTURED_SLOT_ATTEND_ENTRY(SIDE_TOKEN, EP_SQUARE, VALUE) \
    cmz_qk2_hardmax_select_u64( \
        query_x, \
        query_y, \
        cmz_pawn_ep_side_square_code((SIDE_TOKEN), (EP_SQUARE)), \
        cmz_pawn_ep_side_square_key_bias(cmz_pawn_ep_side_square_code((SIDE_TOKEN), (EP_SQUARE))), \
        cmz_pawn_ep_side_square_code((SIDE_TOKEN), (EP_SQUARE)), \
        (VALUE), \
        &selected_score, \
        &selected_tie_index, \
        &selected_value)

#define CMZ_PAWN_EP_CAPTURED_SLOT_ATTEND_ROW(SIDE_TOKEN, BASE, VALUE_0, VALUE_1, VALUE_2, VALUE_3, VALUE_4, VALUE_5, VALUE_6, VALUE_7) \
    CMZ_PAWN_EP_CAPTURED_SLOT_ATTEND_ENTRY((SIDE_TOKEN), (BASE) + 0U, (VALUE_0)); \
    CMZ_PAWN_EP_CAPTURED_SLOT_ATTEND_ENTRY((SIDE_TOKEN), (BASE) + 1U, (VALUE_1)); \
    CMZ_PAWN_EP_CAPTURED_SLOT_ATTEND_ENTRY((SIDE_TOKEN), (BASE) + 2U, (VALUE_2)); \
    CMZ_PAWN_EP_CAPTURED_SLOT_ATTEND_ENTRY((SIDE_TOKEN), (BASE) + 3U, (VALUE_3)); \
    CMZ_PAWN_EP_CAPTURED_SLOT_ATTEND_ENTRY((SIDE_TOKEN), (BASE) + 4U, (VALUE_4)); \
    CMZ_PAWN_EP_CAPTURED_SLOT_ATTEND_ENTRY((SIDE_TOKEN), (BASE) + 5U, (VALUE_5)); \
    CMZ_PAWN_EP_CAPTURED_SLOT_ATTEND_ENTRY((SIDE_TOKEN), (BASE) + 6U, (VALUE_6)); \
    CMZ_PAWN_EP_CAPTURED_SLOT_ATTEND_ENTRY((SIDE_TOKEN), (BASE) + 7U, (VALUE_7))

__device__ uint64_t cmz_candidate_pawn_ep_captured_slot_attention_value(
    uint32_t side_token,
    uint32_t ep_square) {
    const uint32_t query_code = cmz_pawn_ep_side_square_code(side_token, ep_square);
    const uint32_t query_x = query_code * 2U;
    const uint32_t query_y = 1U;
    uint32_t selected_score = 0U;
    uint32_t selected_tie_index = 0xffffffffU;
    uint64_t selected_value = 0ULL;

    CMZ_PAWN_EP_CAPTURED_SLOT_ATTEND_ROW(1U, 0U, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL);
    CMZ_PAWN_EP_CAPTURED_SLOT_ATTEND_ROW(1U, 8U, cmz_bit(0U), cmz_bit(1U), cmz_bit(2U), cmz_bit(3U), cmz_bit(4U), cmz_bit(5U), cmz_bit(6U), cmz_bit(7U));
    CMZ_PAWN_EP_CAPTURED_SLOT_ATTEND_ROW(1U, 16U, cmz_bit(8U), cmz_bit(9U), cmz_bit(10U), cmz_bit(11U), cmz_bit(12U), cmz_bit(13U), cmz_bit(14U), cmz_bit(15U));
    CMZ_PAWN_EP_CAPTURED_SLOT_ATTEND_ROW(1U, 24U, cmz_bit(16U), cmz_bit(17U), cmz_bit(18U), cmz_bit(19U), cmz_bit(20U), cmz_bit(21U), cmz_bit(22U), cmz_bit(23U));
    CMZ_PAWN_EP_CAPTURED_SLOT_ATTEND_ROW(1U, 32U, cmz_bit(24U), cmz_bit(25U), cmz_bit(26U), cmz_bit(27U), cmz_bit(28U), cmz_bit(29U), cmz_bit(30U), cmz_bit(31U));
    CMZ_PAWN_EP_CAPTURED_SLOT_ATTEND_ROW(1U, 40U, cmz_bit(32U), cmz_bit(33U), cmz_bit(34U), cmz_bit(35U), cmz_bit(36U), cmz_bit(37U), cmz_bit(38U), cmz_bit(39U));
    CMZ_PAWN_EP_CAPTURED_SLOT_ATTEND_ROW(1U, 48U, cmz_bit(40U), cmz_bit(41U), cmz_bit(42U), cmz_bit(43U), cmz_bit(44U), cmz_bit(45U), cmz_bit(46U), cmz_bit(47U));
    CMZ_PAWN_EP_CAPTURED_SLOT_ATTEND_ROW(1U, 56U, cmz_bit(48U), cmz_bit(49U), cmz_bit(50U), cmz_bit(51U), cmz_bit(52U), cmz_bit(53U), cmz_bit(54U), cmz_bit(55U));
    CMZ_PAWN_EP_CAPTURED_SLOT_ATTEND_ROW(7U, 0U, cmz_bit(8U), cmz_bit(9U), cmz_bit(10U), cmz_bit(11U), cmz_bit(12U), cmz_bit(13U), cmz_bit(14U), cmz_bit(15U));
    CMZ_PAWN_EP_CAPTURED_SLOT_ATTEND_ROW(7U, 8U, cmz_bit(16U), cmz_bit(17U), cmz_bit(18U), cmz_bit(19U), cmz_bit(20U), cmz_bit(21U), cmz_bit(22U), cmz_bit(23U));
    CMZ_PAWN_EP_CAPTURED_SLOT_ATTEND_ROW(7U, 16U, cmz_bit(24U), cmz_bit(25U), cmz_bit(26U), cmz_bit(27U), cmz_bit(28U), cmz_bit(29U), cmz_bit(30U), cmz_bit(31U));
    CMZ_PAWN_EP_CAPTURED_SLOT_ATTEND_ROW(7U, 24U, cmz_bit(32U), cmz_bit(33U), cmz_bit(34U), cmz_bit(35U), cmz_bit(36U), cmz_bit(37U), cmz_bit(38U), cmz_bit(39U));
    CMZ_PAWN_EP_CAPTURED_SLOT_ATTEND_ROW(7U, 32U, cmz_bit(40U), cmz_bit(41U), cmz_bit(42U), cmz_bit(43U), cmz_bit(44U), cmz_bit(45U), cmz_bit(46U), cmz_bit(47U));
    CMZ_PAWN_EP_CAPTURED_SLOT_ATTEND_ROW(7U, 40U, cmz_bit(48U), cmz_bit(49U), cmz_bit(50U), cmz_bit(51U), cmz_bit(52U), cmz_bit(53U), cmz_bit(54U), cmz_bit(55U));
    CMZ_PAWN_EP_CAPTURED_SLOT_ATTEND_ROW(7U, 48U, cmz_bit(56U), cmz_bit(57U), cmz_bit(58U), cmz_bit(59U), cmz_bit(60U), cmz_bit(61U), cmz_bit(62U), cmz_bit(63U));
    CMZ_PAWN_EP_CAPTURED_SLOT_ATTEND_ROW(7U, 56U, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL);
    return selected_value;
}

#define CMZ_PAWN_EP_CAPTURED_ENEMY_ATTEND_SQUARE(SQUARE) \
    cmz_qk2_select_or_write_u64( \
        captured_slot == cmz_bit((SQUARE)) ? 1U : 0U, \
        0U, \
        1U, \
        0U, \
        (enemy_mask & cmz_bit((SQUARE))) == 0ULL ? 0ULL : 1ULL, \
        &selected_value)

#define CMZ_PAWN_EP_CAPTURED_ENEMY_ATTEND_ROW(BASE) \
    CMZ_PAWN_EP_CAPTURED_ENEMY_ATTEND_SQUARE((BASE) + 0U); \
    CMZ_PAWN_EP_CAPTURED_ENEMY_ATTEND_SQUARE((BASE) + 1U); \
    CMZ_PAWN_EP_CAPTURED_ENEMY_ATTEND_SQUARE((BASE) + 2U); \
    CMZ_PAWN_EP_CAPTURED_ENEMY_ATTEND_SQUARE((BASE) + 3U); \
    CMZ_PAWN_EP_CAPTURED_ENEMY_ATTEND_SQUARE((BASE) + 4U); \
    CMZ_PAWN_EP_CAPTURED_ENEMY_ATTEND_SQUARE((BASE) + 5U); \
    CMZ_PAWN_EP_CAPTURED_ENEMY_ATTEND_SQUARE((BASE) + 6U); \
    CMZ_PAWN_EP_CAPTURED_ENEMY_ATTEND_SQUARE((BASE) + 7U)

__device__ uint32_t cmz_candidate_pawn_ep_captured_enemy_attention_value(
    uint64_t captured_slot,
    uint64_t enemy_mask) {
    uint64_t selected_value = 0ULL;
    CMZ_PAWN_EP_CAPTURED_ENEMY_ATTEND_ROW(0U);
    CMZ_PAWN_EP_CAPTURED_ENEMY_ATTEND_ROW(8U);
    CMZ_PAWN_EP_CAPTURED_ENEMY_ATTEND_ROW(16U);
    CMZ_PAWN_EP_CAPTURED_ENEMY_ATTEND_ROW(24U);
    CMZ_PAWN_EP_CAPTURED_ENEMY_ATTEND_ROW(32U);
    CMZ_PAWN_EP_CAPTURED_ENEMY_ATTEND_ROW(40U);
    CMZ_PAWN_EP_CAPTURED_ENEMY_ATTEND_ROW(48U);
    CMZ_PAWN_EP_CAPTURED_ENEMY_ATTEND_ROW(56U);
    return static_cast<uint32_t>(selected_value);
}

__device__ uint64_t cmz_candidate_pawn_capture_ep_condition_attention_value(
    uint32_t side_token,
    uint64_t target_slot,
    uint64_t enemy_mask,
    uint32_t ep_square) {
    const uint64_t target_match_slot =
        cmz_candidate_pawn_ep_target_match_attention_value(target_slot, ep_square);
    const uint64_t captured_slot =
        cmz_candidate_pawn_ep_captured_slot_attention_value(side_token, ep_square);
    const uint32_t enemy_condition =
        cmz_candidate_pawn_ep_captured_enemy_attention_value(captured_slot, enemy_mask);
    uint64_t output = 0ULL;
    cmz_qk2_select_or_write_u64(
        1U, 0U, (target_match_slot == 0ULL ? 0U : 1U) & enemy_condition, 0U, target_match_slot, &output);
    return output;
}

__device__ uint64_t cmz_candidate_pawn_capture_slot_attention_value(
    uint32_t side_token,
    int from_file,
    int from_rank,
    int direction,
    int delta_file,
    uint64_t enemy_mask,
    uint32_t ep_square) {
    const uint64_t target_slot =
        cmz_candidate_pawn_capture_target_slot_attention_value(from_file, from_rank, direction, delta_file);
    const uint64_t enemy_capture =
        cmz_candidate_pawn_capture_enemy_condition_attention_value(target_slot, enemy_mask);
    const uint64_t ep_capture =
        cmz_candidate_pawn_capture_ep_condition_attention_value(side_token, target_slot, enemy_mask, ep_square);
    uint64_t output = 0ULL;
    cmz_qk2_select_or_write_u64(1U, 0U, enemy_capture == 0ULL ? 0U : 1U, 0U, enemy_capture, &output);
    cmz_qk2_select_or_write_u64(1U, 0U, ep_capture == 0ULL ? 0U : 1U, 0U, ep_capture, &output);
    return output;
}

__device__ uint64_t cmz_candidate_pawn_target_mask_attention_value(
    uint32_t token,
    uint32_t from_square,
    uint64_t enemy_mask,
    uint64_t occupancy_mask,
    uint32_t ep_square) {
    const int from_file = static_cast<int>(from_square % 8U);
    const int from_rank = static_cast<int>(from_square / 8U);
    const bool white = token == 1U;
    const int direction = white ? 1 : -1;
    const int start_rank = white ? 1 : 6;
    const uint64_t single_push =
        cmz_candidate_pawn_single_push_slot_attention_value(from_file, from_rank, direction, occupancy_mask);
    const uint64_t double_push = cmz_candidate_pawn_double_push_slot_attention_value(
        from_file, from_rank, direction, start_rank, occupancy_mask, single_push);
    const uint64_t capture_left = cmz_candidate_pawn_capture_slot_attention_value(
        token, from_file, from_rank, direction, -1, enemy_mask, ep_square);
    const uint64_t capture_right = cmz_candidate_pawn_capture_slot_attention_value(
        token, from_file, from_rank, direction, 1, enemy_mask, ep_square);

    uint64_t mask = 0ULL;
    cmz_qk2_select_or_write_u64(1U, 0U, single_push == 0ULL ? 0U : 1U, 0U, single_push, &mask);
    cmz_qk2_select_or_write_u64(1U, 0U, double_push == 0ULL ? 0U : 1U, 0U, double_push, &mask);
    cmz_qk2_select_or_write_u64(1U, 0U, capture_left == 0ULL ? 0U : 1U, 0U, capture_left, &mask);
    cmz_qk2_select_or_write_u64(1U, 0U, capture_right == 0ULL ? 0U : 1U, 0U, capture_right, &mask);
    return mask;
}

constexpr uint32_t kCmzCoordinateTableStride = 12U;
constexpr uint32_t kCmzCoordinateTableMaxCode = 143U;
constexpr uint32_t kCmzCoordinateTableScoreBias = kCmzCoordinateTableMaxCode * kCmzCoordinateTableMaxCode;

__device__ uint32_t cmz_coordinate_table_code_from_shifted(uint32_t shifted_file, uint32_t shifted_rank) {
    return shifted_file * kCmzCoordinateTableStride + shifted_rank;
}

__device__ uint32_t cmz_coordinate_table_key_bias(uint32_t code) {
    return kCmzCoordinateTableScoreBias - code * code;
}

#define CMZ_COORDINATE_TABLE_ATTEND_ENTRY(SHIFTED_FILE, SHIFTED_RANK, VALUE) \
    cmz_qk2_hardmax_select_u64( \
        query_x, \
        query_y, \
        cmz_coordinate_table_code_from_shifted((SHIFTED_FILE), (SHIFTED_RANK)), \
        cmz_coordinate_table_key_bias(cmz_coordinate_table_code_from_shifted((SHIFTED_FILE), (SHIFTED_RANK))), \
        cmz_coordinate_table_code_from_shifted((SHIFTED_FILE), (SHIFTED_RANK)), \
        (VALUE), \
        &selected_score, \
        &selected_tie_index, \
        &selected_value)

#define CMZ_COORDINATE_TABLE_ATTEND_ROW( \
    SHIFTED_FILE, VALUE_0, VALUE_1, VALUE_2, VALUE_3, VALUE_4, VALUE_5, VALUE_6, VALUE_7, VALUE_8, VALUE_9, VALUE_10, VALUE_11) \
    CMZ_COORDINATE_TABLE_ATTEND_ENTRY((SHIFTED_FILE), 0U, (VALUE_0)); \
    CMZ_COORDINATE_TABLE_ATTEND_ENTRY((SHIFTED_FILE), 1U, (VALUE_1)); \
    CMZ_COORDINATE_TABLE_ATTEND_ENTRY((SHIFTED_FILE), 2U, (VALUE_2)); \
    CMZ_COORDINATE_TABLE_ATTEND_ENTRY((SHIFTED_FILE), 3U, (VALUE_3)); \
    CMZ_COORDINATE_TABLE_ATTEND_ENTRY((SHIFTED_FILE), 4U, (VALUE_4)); \
    CMZ_COORDINATE_TABLE_ATTEND_ENTRY((SHIFTED_FILE), 5U, (VALUE_5)); \
    CMZ_COORDINATE_TABLE_ATTEND_ENTRY((SHIFTED_FILE), 6U, (VALUE_6)); \
    CMZ_COORDINATE_TABLE_ATTEND_ENTRY((SHIFTED_FILE), 7U, (VALUE_7)); \
    CMZ_COORDINATE_TABLE_ATTEND_ENTRY((SHIFTED_FILE), 8U, (VALUE_8)); \
    CMZ_COORDINATE_TABLE_ATTEND_ENTRY((SHIFTED_FILE), 9U, (VALUE_9)); \
    CMZ_COORDINATE_TABLE_ATTEND_ENTRY((SHIFTED_FILE), 10U, (VALUE_10)); \
    CMZ_COORDINATE_TABLE_ATTEND_ENTRY((SHIFTED_FILE), 11U, (VALUE_11))

__device__ uint64_t cmz_candidate_single_offset_coordinate_table_attention_value(int file, int rank) {
    const uint32_t query_code =
        cmz_coordinate_table_code_from_shifted(static_cast<uint32_t>(file + 2), static_cast<uint32_t>(rank + 2));
    const uint32_t query_x = query_code * 2U;
    const uint32_t query_y = 1U;
    uint32_t selected_score = 0U;
    uint32_t selected_tie_index = 0xffffffffU;
    uint64_t selected_value = 0ULL;

    CMZ_COORDINATE_TABLE_ATTEND_ROW(0U, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL);
    CMZ_COORDINATE_TABLE_ATTEND_ROW(1U, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL);
    CMZ_COORDINATE_TABLE_ATTEND_ROW(
        2U, 0ULL, 0ULL, cmz_bit(0U), cmz_bit(8U), cmz_bit(16U), cmz_bit(24U), cmz_bit(32U), cmz_bit(40U), cmz_bit(48U), cmz_bit(56U), 0ULL, 0ULL);
    CMZ_COORDINATE_TABLE_ATTEND_ROW(
        3U, 0ULL, 0ULL, cmz_bit(1U), cmz_bit(9U), cmz_bit(17U), cmz_bit(25U), cmz_bit(33U), cmz_bit(41U), cmz_bit(49U), cmz_bit(57U), 0ULL, 0ULL);
    CMZ_COORDINATE_TABLE_ATTEND_ROW(
        4U, 0ULL, 0ULL, cmz_bit(2U), cmz_bit(10U), cmz_bit(18U), cmz_bit(26U), cmz_bit(34U), cmz_bit(42U), cmz_bit(50U), cmz_bit(58U), 0ULL, 0ULL);
    CMZ_COORDINATE_TABLE_ATTEND_ROW(
        5U, 0ULL, 0ULL, cmz_bit(3U), cmz_bit(11U), cmz_bit(19U), cmz_bit(27U), cmz_bit(35U), cmz_bit(43U), cmz_bit(51U), cmz_bit(59U), 0ULL, 0ULL);
    CMZ_COORDINATE_TABLE_ATTEND_ROW(
        6U, 0ULL, 0ULL, cmz_bit(4U), cmz_bit(12U), cmz_bit(20U), cmz_bit(28U), cmz_bit(36U), cmz_bit(44U), cmz_bit(52U), cmz_bit(60U), 0ULL, 0ULL);
    CMZ_COORDINATE_TABLE_ATTEND_ROW(
        7U, 0ULL, 0ULL, cmz_bit(5U), cmz_bit(13U), cmz_bit(21U), cmz_bit(29U), cmz_bit(37U), cmz_bit(45U), cmz_bit(53U), cmz_bit(61U), 0ULL, 0ULL);
    CMZ_COORDINATE_TABLE_ATTEND_ROW(
        8U, 0ULL, 0ULL, cmz_bit(6U), cmz_bit(14U), cmz_bit(22U), cmz_bit(30U), cmz_bit(38U), cmz_bit(46U), cmz_bit(54U), cmz_bit(62U), 0ULL, 0ULL);
    CMZ_COORDINATE_TABLE_ATTEND_ROW(
        9U, 0ULL, 0ULL, cmz_bit(7U), cmz_bit(15U), cmz_bit(23U), cmz_bit(31U), cmz_bit(39U), cmz_bit(47U), cmz_bit(55U), cmz_bit(63U), 0ULL, 0ULL);
    CMZ_COORDINATE_TABLE_ATTEND_ROW(10U, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL);
    CMZ_COORDINATE_TABLE_ATTEND_ROW(11U, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL);

    return selected_value;
}

__device__ uint64_t cmz_candidate_single_offset_coordinate_slot_attention_value(int file, int rank) {
    const uint64_t table_slot = cmz_candidate_single_offset_coordinate_table_attention_value(file, rank);
    uint64_t output = 0ULL;
    cmz_qk2_select_or_write_u64(1U, 0U, table_slot == 0ULL ? 0U : 1U, 0U, table_slot, &output);
    return output;
}

__device__ uint64_t cmz_candidate_single_offset_bounds_slot_attention_value(
    uint32_t from_square,
    int delta_file,
    int delta_rank) {
    const int from_file = static_cast<int>(from_square % 8U);
    const int from_rank = static_cast<int>(from_square / 8U);
    const int file = from_file + delta_file;
    const int rank = from_rank + delta_rank;
    uint64_t output = 0ULL;
    const uint64_t coordinate_slot = cmz_candidate_single_offset_coordinate_slot_attention_value(file, rank);
    cmz_qk2_select_or_write_u64(1U, 0U, coordinate_slot == 0ULL ? 0U : 1U, 0U, coordinate_slot, &output);
    return output;
}

__device__ uint64_t cmz_candidate_single_offset_target_mask_attention_value(
    uint32_t from_square,
    uint64_t own_occupancy_mask,
    int delta_file,
    int delta_rank) {
    const uint64_t bounds_slot =
        cmz_candidate_single_offset_bounds_slot_attention_value(from_square, delta_file, delta_rank);
    const uint32_t not_own_piece = (own_occupancy_mask & bounds_slot) == 0ULL ? 1U : 0U;
    uint64_t output = 0ULL;
    cmz_qk2_select_or_write_u64(1U, 0U, (bounds_slot == 0ULL ? 0U : 1U) & not_own_piece, 0U, bounds_slot, &output);
    return output;
}

__device__ uint64_t cmz_candidate_offset_target_mask_attention_value(
    uint32_t from_square,
    uint64_t friendly_mask,
    const int* delta_files,
    const int* delta_ranks) {
    uint64_t mask = 0ULL;
    const uint64_t slot0 =
        cmz_candidate_single_offset_target_mask_attention_value(from_square, friendly_mask, delta_files[0], delta_ranks[0]);
    const uint64_t slot1 =
        cmz_candidate_single_offset_target_mask_attention_value(from_square, friendly_mask, delta_files[1], delta_ranks[1]);
    const uint64_t slot2 =
        cmz_candidate_single_offset_target_mask_attention_value(from_square, friendly_mask, delta_files[2], delta_ranks[2]);
    const uint64_t slot3 =
        cmz_candidate_single_offset_target_mask_attention_value(from_square, friendly_mask, delta_files[3], delta_ranks[3]);
    const uint64_t slot4 =
        cmz_candidate_single_offset_target_mask_attention_value(from_square, friendly_mask, delta_files[4], delta_ranks[4]);
    const uint64_t slot5 =
        cmz_candidate_single_offset_target_mask_attention_value(from_square, friendly_mask, delta_files[5], delta_ranks[5]);
    const uint64_t slot6 =
        cmz_candidate_single_offset_target_mask_attention_value(from_square, friendly_mask, delta_files[6], delta_ranks[6]);
    const uint64_t slot7 =
        cmz_candidate_single_offset_target_mask_attention_value(from_square, friendly_mask, delta_files[7], delta_ranks[7]);
    cmz_qk2_select_or_write_u64(1U, 0U, slot0 == 0ULL ? 0U : 1U, 0U, slot0, &mask);
    cmz_qk2_select_or_write_u64(1U, 0U, slot1 == 0ULL ? 0U : 1U, 0U, slot1, &mask);
    cmz_qk2_select_or_write_u64(1U, 0U, slot2 == 0ULL ? 0U : 1U, 0U, slot2, &mask);
    cmz_qk2_select_or_write_u64(1U, 0U, slot3 == 0ULL ? 0U : 1U, 0U, slot3, &mask);
    cmz_qk2_select_or_write_u64(1U, 0U, slot4 == 0ULL ? 0U : 1U, 0U, slot4, &mask);
    cmz_qk2_select_or_write_u64(1U, 0U, slot5 == 0ULL ? 0U : 1U, 0U, slot5, &mask);
    cmz_qk2_select_or_write_u64(1U, 0U, slot6 == 0ULL ? 0U : 1U, 0U, slot6, &mask);
    cmz_qk2_select_or_write_u64(1U, 0U, slot7 == 0ULL ? 0U : 1U, 0U, slot7, &mask);
    return mask;
}

constexpr uint32_t kCmzSliderCoordinateTableStride = 22U;
constexpr uint32_t kCmzSliderCoordinateTableMaxCode = 483U;
constexpr uint32_t kCmzSliderCoordinateTableScoreBias =
    kCmzSliderCoordinateTableMaxCode * kCmzSliderCoordinateTableMaxCode;

__device__ uint32_t cmz_slider_coordinate_table_code_from_shifted(uint32_t shifted_file, uint32_t shifted_rank) {
    return shifted_file * kCmzSliderCoordinateTableStride + shifted_rank;
}

__device__ uint32_t cmz_slider_coordinate_table_key_bias(uint32_t code) {
    return kCmzSliderCoordinateTableScoreBias - code * code;
}

#define CMZ_SLIDER_COORDINATE_ATTEND_ENTRY(SHIFTED_FILE, SHIFTED_RANK, VALUE) \
    cmz_qk2_hardmax_select_u64( \
        query_x, \
        query_y, \
        cmz_slider_coordinate_table_code_from_shifted((SHIFTED_FILE), (SHIFTED_RANK)), \
        cmz_slider_coordinate_table_key_bias(cmz_slider_coordinate_table_code_from_shifted((SHIFTED_FILE), (SHIFTED_RANK))), \
        cmz_slider_coordinate_table_code_from_shifted((SHIFTED_FILE), (SHIFTED_RANK)), \
        (VALUE), \
        &selected_score, \
        &selected_tie_index, \
        &selected_value)

#define CMZ_SLIDER_COORDINATE_ATTEND_ROW( \
    SHIFTED_FILE, VALUE_0, VALUE_1, VALUE_2, VALUE_3, VALUE_4, VALUE_5, VALUE_6, VALUE_7, VALUE_8, VALUE_9, VALUE_10, VALUE_11, VALUE_12, VALUE_13, VALUE_14, VALUE_15, VALUE_16, VALUE_17, VALUE_18, VALUE_19, VALUE_20, VALUE_21) \
    CMZ_SLIDER_COORDINATE_ATTEND_ENTRY((SHIFTED_FILE), 0U, (VALUE_0)); \
    CMZ_SLIDER_COORDINATE_ATTEND_ENTRY((SHIFTED_FILE), 1U, (VALUE_1)); \
    CMZ_SLIDER_COORDINATE_ATTEND_ENTRY((SHIFTED_FILE), 2U, (VALUE_2)); \
    CMZ_SLIDER_COORDINATE_ATTEND_ENTRY((SHIFTED_FILE), 3U, (VALUE_3)); \
    CMZ_SLIDER_COORDINATE_ATTEND_ENTRY((SHIFTED_FILE), 4U, (VALUE_4)); \
    CMZ_SLIDER_COORDINATE_ATTEND_ENTRY((SHIFTED_FILE), 5U, (VALUE_5)); \
    CMZ_SLIDER_COORDINATE_ATTEND_ENTRY((SHIFTED_FILE), 6U, (VALUE_6)); \
    CMZ_SLIDER_COORDINATE_ATTEND_ENTRY((SHIFTED_FILE), 7U, (VALUE_7)); \
    CMZ_SLIDER_COORDINATE_ATTEND_ENTRY((SHIFTED_FILE), 8U, (VALUE_8)); \
    CMZ_SLIDER_COORDINATE_ATTEND_ENTRY((SHIFTED_FILE), 9U, (VALUE_9)); \
    CMZ_SLIDER_COORDINATE_ATTEND_ENTRY((SHIFTED_FILE), 10U, (VALUE_10)); \
    CMZ_SLIDER_COORDINATE_ATTEND_ENTRY((SHIFTED_FILE), 11U, (VALUE_11)); \
    CMZ_SLIDER_COORDINATE_ATTEND_ENTRY((SHIFTED_FILE), 12U, (VALUE_12)); \
    CMZ_SLIDER_COORDINATE_ATTEND_ENTRY((SHIFTED_FILE), 13U, (VALUE_13)); \
    CMZ_SLIDER_COORDINATE_ATTEND_ENTRY((SHIFTED_FILE), 14U, (VALUE_14)); \
    CMZ_SLIDER_COORDINATE_ATTEND_ENTRY((SHIFTED_FILE), 15U, (VALUE_15)); \
    CMZ_SLIDER_COORDINATE_ATTEND_ENTRY((SHIFTED_FILE), 16U, (VALUE_16)); \
    CMZ_SLIDER_COORDINATE_ATTEND_ENTRY((SHIFTED_FILE), 17U, (VALUE_17)); \
    CMZ_SLIDER_COORDINATE_ATTEND_ENTRY((SHIFTED_FILE), 18U, (VALUE_18)); \
    CMZ_SLIDER_COORDINATE_ATTEND_ENTRY((SHIFTED_FILE), 19U, (VALUE_19)); \
    CMZ_SLIDER_COORDINATE_ATTEND_ENTRY((SHIFTED_FILE), 20U, (VALUE_20)); \
    CMZ_SLIDER_COORDINATE_ATTEND_ENTRY((SHIFTED_FILE), 21U, (VALUE_21))

__device__ uint64_t cmz_candidate_slider_coordinate_table_attention_value(int file, int rank) {
    const uint32_t query_code = cmz_slider_coordinate_table_code_from_shifted(
        static_cast<uint32_t>(file + 7),
        static_cast<uint32_t>(rank + 7));
    const uint32_t query_x = query_code * 2U;
    const uint32_t query_y = 1U;
    uint32_t selected_score = 0U;
    uint32_t selected_tie_index = 0xffffffffU;
    uint64_t selected_value = 0ULL;

    CMZ_SLIDER_COORDINATE_ATTEND_ROW(0U, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL);
    CMZ_SLIDER_COORDINATE_ATTEND_ROW(1U, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL);
    CMZ_SLIDER_COORDINATE_ATTEND_ROW(2U, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL);
    CMZ_SLIDER_COORDINATE_ATTEND_ROW(3U, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL);
    CMZ_SLIDER_COORDINATE_ATTEND_ROW(4U, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL);
    CMZ_SLIDER_COORDINATE_ATTEND_ROW(5U, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL);
    CMZ_SLIDER_COORDINATE_ATTEND_ROW(6U, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL);
    CMZ_SLIDER_COORDINATE_ATTEND_ROW(7U, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, cmz_bit(0U), cmz_bit(8U), cmz_bit(16U), cmz_bit(24U), cmz_bit(32U), cmz_bit(40U), cmz_bit(48U), cmz_bit(56U), 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL);
    CMZ_SLIDER_COORDINATE_ATTEND_ROW(8U, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, cmz_bit(1U), cmz_bit(9U), cmz_bit(17U), cmz_bit(25U), cmz_bit(33U), cmz_bit(41U), cmz_bit(49U), cmz_bit(57U), 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL);
    CMZ_SLIDER_COORDINATE_ATTEND_ROW(9U, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, cmz_bit(2U), cmz_bit(10U), cmz_bit(18U), cmz_bit(26U), cmz_bit(34U), cmz_bit(42U), cmz_bit(50U), cmz_bit(58U), 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL);
    CMZ_SLIDER_COORDINATE_ATTEND_ROW(10U, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, cmz_bit(3U), cmz_bit(11U), cmz_bit(19U), cmz_bit(27U), cmz_bit(35U), cmz_bit(43U), cmz_bit(51U), cmz_bit(59U), 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL);
    CMZ_SLIDER_COORDINATE_ATTEND_ROW(11U, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, cmz_bit(4U), cmz_bit(12U), cmz_bit(20U), cmz_bit(28U), cmz_bit(36U), cmz_bit(44U), cmz_bit(52U), cmz_bit(60U), 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL);
    CMZ_SLIDER_COORDINATE_ATTEND_ROW(12U, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, cmz_bit(5U), cmz_bit(13U), cmz_bit(21U), cmz_bit(29U), cmz_bit(37U), cmz_bit(45U), cmz_bit(53U), cmz_bit(61U), 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL);
    CMZ_SLIDER_COORDINATE_ATTEND_ROW(13U, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, cmz_bit(6U), cmz_bit(14U), cmz_bit(22U), cmz_bit(30U), cmz_bit(38U), cmz_bit(46U), cmz_bit(54U), cmz_bit(62U), 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL);
    CMZ_SLIDER_COORDINATE_ATTEND_ROW(14U, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, cmz_bit(7U), cmz_bit(15U), cmz_bit(23U), cmz_bit(31U), cmz_bit(39U), cmz_bit(47U), cmz_bit(55U), cmz_bit(63U), 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL);
    CMZ_SLIDER_COORDINATE_ATTEND_ROW(15U, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL);
    CMZ_SLIDER_COORDINATE_ATTEND_ROW(16U, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL);
    CMZ_SLIDER_COORDINATE_ATTEND_ROW(17U, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL);
    CMZ_SLIDER_COORDINATE_ATTEND_ROW(18U, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL);
    CMZ_SLIDER_COORDINATE_ATTEND_ROW(19U, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL);
    CMZ_SLIDER_COORDINATE_ATTEND_ROW(20U, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL);
    CMZ_SLIDER_COORDINATE_ATTEND_ROW(21U, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL);

    return selected_value;
}

__device__ uint64_t cmz_candidate_slider_ray_step_target_slot_attention_value(
    uint32_t from_square,
    int delta_file,
    int delta_rank,
    uint32_t step) {
    const int from_file = static_cast<int>(from_square % 8U);
    const int from_rank = static_cast<int>(from_square / 8U);
    const int file = from_file + delta_file * static_cast<int>(step);
    const int rank = from_rank + delta_rank * static_cast<int>(step);
    return cmz_candidate_slider_coordinate_table_attention_value(file, rank);
}

constexpr uint32_t kCmzSliderPriorStepStride = 8U;
constexpr uint32_t kCmzSliderPriorStepMaxCode = 63U;
constexpr uint32_t kCmzSliderPriorStepScoreBias =
    kCmzSliderPriorStepMaxCode * kCmzSliderPriorStepMaxCode;

__device__ uint32_t cmz_slider_prior_step_code(uint32_t step, uint32_t prior_step) {
    return step * kCmzSliderPriorStepStride + prior_step;
}

__device__ uint32_t cmz_slider_prior_step_key_bias(uint32_t code) {
    return kCmzSliderPriorStepScoreBias - code * code;
}

#define CMZ_SLIDER_PRIOR_STEP_ENABLED_ATTEND_ENTRY(STEP, PRIOR_STEP, VALUE) \
    cmz_qk2_hardmax_select_u32( \
        query_x, \
        query_y, \
        cmz_slider_prior_step_code((STEP), (PRIOR_STEP)), \
        cmz_slider_prior_step_key_bias(cmz_slider_prior_step_code((STEP), (PRIOR_STEP))), \
        cmz_slider_prior_step_code((STEP), (PRIOR_STEP)), \
        (VALUE), \
        &selected_score, \
        &selected_tie_index, \
        &selected_value)

#define CMZ_SLIDER_PRIOR_STEP_ENABLED_ATTEND_ROW(STEP, VALUE_0, VALUE_1, VALUE_2, VALUE_3, VALUE_4, VALUE_5, VALUE_6, VALUE_7) \
    CMZ_SLIDER_PRIOR_STEP_ENABLED_ATTEND_ENTRY((STEP), 0U, (VALUE_0)); \
    CMZ_SLIDER_PRIOR_STEP_ENABLED_ATTEND_ENTRY((STEP), 1U, (VALUE_1)); \
    CMZ_SLIDER_PRIOR_STEP_ENABLED_ATTEND_ENTRY((STEP), 2U, (VALUE_2)); \
    CMZ_SLIDER_PRIOR_STEP_ENABLED_ATTEND_ENTRY((STEP), 3U, (VALUE_3)); \
    CMZ_SLIDER_PRIOR_STEP_ENABLED_ATTEND_ENTRY((STEP), 4U, (VALUE_4)); \
    CMZ_SLIDER_PRIOR_STEP_ENABLED_ATTEND_ENTRY((STEP), 5U, (VALUE_5)); \
    CMZ_SLIDER_PRIOR_STEP_ENABLED_ATTEND_ENTRY((STEP), 6U, (VALUE_6)); \
    CMZ_SLIDER_PRIOR_STEP_ENABLED_ATTEND_ENTRY((STEP), 7U, (VALUE_7))

__device__ uint32_t cmz_candidate_slider_prior_step_enabled_attention_value(
    uint32_t step,
    uint32_t prior_step) {
    const uint32_t query_code = cmz_slider_prior_step_code(step, prior_step);
    const uint32_t query_x = query_code * 2U;
    const uint32_t query_y = 1U;
    uint32_t selected_score = 0U;
    uint32_t selected_tie_index = 0xffffffffU;
    uint32_t selected_value = 0U;

    CMZ_SLIDER_PRIOR_STEP_ENABLED_ATTEND_ROW(0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U);
    CMZ_SLIDER_PRIOR_STEP_ENABLED_ATTEND_ROW(1U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U);
    CMZ_SLIDER_PRIOR_STEP_ENABLED_ATTEND_ROW(2U, 0U, 1U, 0U, 0U, 0U, 0U, 0U, 0U);
    CMZ_SLIDER_PRIOR_STEP_ENABLED_ATTEND_ROW(3U, 0U, 1U, 1U, 0U, 0U, 0U, 0U, 0U);
    CMZ_SLIDER_PRIOR_STEP_ENABLED_ATTEND_ROW(4U, 0U, 1U, 1U, 1U, 0U, 0U, 0U, 0U);
    CMZ_SLIDER_PRIOR_STEP_ENABLED_ATTEND_ROW(5U, 0U, 1U, 1U, 1U, 1U, 0U, 0U, 0U);
    CMZ_SLIDER_PRIOR_STEP_ENABLED_ATTEND_ROW(6U, 0U, 1U, 1U, 1U, 1U, 1U, 0U, 0U);
    CMZ_SLIDER_PRIOR_STEP_ENABLED_ATTEND_ROW(7U, 0U, 1U, 1U, 1U, 1U, 1U, 1U, 0U);
    return selected_value;
}

#define CMZ_SLIDER_OCCUPIED_ATTEND_SQUARE(SQUARE) \
    cmz_qk2_select_or_write_u64( \
        square_slot == cmz_bit((SQUARE)) ? 1U : 0U, \
        0U, \
        1U, \
        0U, \
        (occupancy_mask & cmz_bit((SQUARE))) == 0ULL ? 0ULL : 1ULL, \
        &selected_value)

#define CMZ_SLIDER_OCCUPIED_ATTEND_ROW(BASE) \
    CMZ_SLIDER_OCCUPIED_ATTEND_SQUARE((BASE) + 0U); \
    CMZ_SLIDER_OCCUPIED_ATTEND_SQUARE((BASE) + 1U); \
    CMZ_SLIDER_OCCUPIED_ATTEND_SQUARE((BASE) + 2U); \
    CMZ_SLIDER_OCCUPIED_ATTEND_SQUARE((BASE) + 3U); \
    CMZ_SLIDER_OCCUPIED_ATTEND_SQUARE((BASE) + 4U); \
    CMZ_SLIDER_OCCUPIED_ATTEND_SQUARE((BASE) + 5U); \
    CMZ_SLIDER_OCCUPIED_ATTEND_SQUARE((BASE) + 6U); \
    CMZ_SLIDER_OCCUPIED_ATTEND_SQUARE((BASE) + 7U)

__device__ uint32_t cmz_candidate_slider_square_occupied_attention_value(
    uint64_t square_slot,
    uint64_t occupancy_mask) {
    uint64_t selected_value = 0ULL;
    CMZ_SLIDER_OCCUPIED_ATTEND_ROW(0U);
    CMZ_SLIDER_OCCUPIED_ATTEND_ROW(8U);
    CMZ_SLIDER_OCCUPIED_ATTEND_ROW(16U);
    CMZ_SLIDER_OCCUPIED_ATTEND_ROW(24U);
    CMZ_SLIDER_OCCUPIED_ATTEND_ROW(32U);
    CMZ_SLIDER_OCCUPIED_ATTEND_ROW(40U);
    CMZ_SLIDER_OCCUPIED_ATTEND_ROW(48U);
    CMZ_SLIDER_OCCUPIED_ATTEND_ROW(56U);
    return static_cast<uint32_t>(selected_value);
}

#define CMZ_SLIDER_SLOT_NONZERO_ATTEND_SQUARE(SQUARE) \
    cmz_qk2_select_or_write_u64( \
        square_slot == cmz_bit((SQUARE)) ? 1U : 0U, \
        0U, \
        1U, \
        0U, \
        1ULL, \
        &selected_value)

#define CMZ_SLIDER_SLOT_NONZERO_ATTEND_ROW(BASE) \
    CMZ_SLIDER_SLOT_NONZERO_ATTEND_SQUARE((BASE) + 0U); \
    CMZ_SLIDER_SLOT_NONZERO_ATTEND_SQUARE((BASE) + 1U); \
    CMZ_SLIDER_SLOT_NONZERO_ATTEND_SQUARE((BASE) + 2U); \
    CMZ_SLIDER_SLOT_NONZERO_ATTEND_SQUARE((BASE) + 3U); \
    CMZ_SLIDER_SLOT_NONZERO_ATTEND_SQUARE((BASE) + 4U); \
    CMZ_SLIDER_SLOT_NONZERO_ATTEND_SQUARE((BASE) + 5U); \
    CMZ_SLIDER_SLOT_NONZERO_ATTEND_SQUARE((BASE) + 6U); \
    CMZ_SLIDER_SLOT_NONZERO_ATTEND_SQUARE((BASE) + 7U)

__device__ uint32_t cmz_candidate_slider_slot_nonzero_attention_value(uint64_t square_slot) {
    uint64_t selected_value = 0ULL;
    CMZ_SLIDER_SLOT_NONZERO_ATTEND_ROW(0U);
    CMZ_SLIDER_SLOT_NONZERO_ATTEND_ROW(8U);
    CMZ_SLIDER_SLOT_NONZERO_ATTEND_ROW(16U);
    CMZ_SLIDER_SLOT_NONZERO_ATTEND_ROW(24U);
    CMZ_SLIDER_SLOT_NONZERO_ATTEND_ROW(32U);
    CMZ_SLIDER_SLOT_NONZERO_ATTEND_ROW(40U);
    CMZ_SLIDER_SLOT_NONZERO_ATTEND_ROW(48U);
    CMZ_SLIDER_SLOT_NONZERO_ATTEND_ROW(56U);
    return static_cast<uint32_t>(selected_value);
}

__device__ uint32_t cmz_candidate_slider_ray_prior_blocker_attention_value(
    uint32_t from_square,
    int delta_file,
    int delta_rank,
    uint32_t step,
    uint64_t occupancy_mask) {
    const uint64_t prior_slot_1 =
        cmz_candidate_slider_ray_step_target_slot_attention_value(from_square, delta_file, delta_rank, 1U);
    const uint64_t prior_slot_2 =
        cmz_candidate_slider_ray_step_target_slot_attention_value(from_square, delta_file, delta_rank, 2U);
    const uint64_t prior_slot_3 =
        cmz_candidate_slider_ray_step_target_slot_attention_value(from_square, delta_file, delta_rank, 3U);
    const uint64_t prior_slot_4 =
        cmz_candidate_slider_ray_step_target_slot_attention_value(from_square, delta_file, delta_rank, 4U);
    const uint64_t prior_slot_5 =
        cmz_candidate_slider_ray_step_target_slot_attention_value(from_square, delta_file, delta_rank, 5U);
    const uint64_t prior_slot_6 =
        cmz_candidate_slider_ray_step_target_slot_attention_value(from_square, delta_file, delta_rank, 6U);
    const uint32_t enabled_1 = cmz_candidate_slider_prior_step_enabled_attention_value(step, 1U);
    const uint32_t enabled_2 = cmz_candidate_slider_prior_step_enabled_attention_value(step, 2U);
    const uint32_t enabled_3 = cmz_candidate_slider_prior_step_enabled_attention_value(step, 3U);
    const uint32_t enabled_4 = cmz_candidate_slider_prior_step_enabled_attention_value(step, 4U);
    const uint32_t enabled_5 = cmz_candidate_slider_prior_step_enabled_attention_value(step, 5U);
    const uint32_t enabled_6 = cmz_candidate_slider_prior_step_enabled_attention_value(step, 6U);
    const uint32_t occupied_1 = cmz_candidate_slider_square_occupied_attention_value(prior_slot_1, occupancy_mask);
    const uint32_t occupied_2 = cmz_candidate_slider_square_occupied_attention_value(prior_slot_2, occupancy_mask);
    const uint32_t occupied_3 = cmz_candidate_slider_square_occupied_attention_value(prior_slot_3, occupancy_mask);
    const uint32_t occupied_4 = cmz_candidate_slider_square_occupied_attention_value(prior_slot_4, occupancy_mask);
    const uint32_t occupied_5 = cmz_candidate_slider_square_occupied_attention_value(prior_slot_5, occupancy_mask);
    const uint32_t occupied_6 = cmz_candidate_slider_square_occupied_attention_value(prior_slot_6, occupancy_mask);
    uint64_t selected_value = 0ULL;
    cmz_qk2_select_or_write_u64(1U, 0U, enabled_1 & occupied_1, 0U, 1ULL, &selected_value);
    cmz_qk2_select_or_write_u64(1U, 0U, enabled_2 & occupied_2, 0U, 1ULL, &selected_value);
    cmz_qk2_select_or_write_u64(1U, 0U, enabled_3 & occupied_3, 0U, 1ULL, &selected_value);
    cmz_qk2_select_or_write_u64(1U, 0U, enabled_4 & occupied_4, 0U, 1ULL, &selected_value);
    cmz_qk2_select_or_write_u64(1U, 0U, enabled_5 & occupied_5, 0U, 1ULL, &selected_value);
    cmz_qk2_select_or_write_u64(1U, 0U, enabled_6 & occupied_6, 0U, 1ULL, &selected_value);
    return static_cast<uint32_t>(selected_value);
}

__device__ uint64_t cmz_candidate_slider_ray_step_attention_value(
    uint32_t from_square,
    int delta_file,
    int delta_rank,
    uint32_t step,
    uint64_t own_occupancy_mask,
    uint64_t occupancy_mask) {
    const uint64_t target_bit =
        cmz_candidate_slider_ray_step_target_slot_attention_value(from_square, delta_file, delta_rank, step);
    const uint32_t target_exists = cmz_candidate_slider_slot_nonzero_attention_value(target_bit);
    const uint32_t own_condition =
        cmz_candidate_slider_square_occupied_attention_value(target_bit, own_occupancy_mask);
    const uint32_t prior_condition =
        cmz_candidate_slider_ray_prior_blocker_attention_value(from_square, delta_file, delta_rank, step, occupancy_mask);
    const uint32_t valid_step = target_exists & (own_condition == 0U ? 1U : 0U) & (prior_condition == 0U ? 1U : 0U);
    uint64_t output = 0ULL;
    cmz_qk2_select_or_write_u64(1U, 0U, valid_step, 0U, target_bit, &output);
    return output;
}

__device__ uint64_t cmz_candidate_slider_ray_slot_attention_value(
    uint32_t from_square,
    int delta_file,
    int delta_rank,
    uint64_t own_occupancy_mask,
    uint64_t occupancy_mask) {
    const uint64_t step1 =
        cmz_candidate_slider_ray_step_attention_value(from_square, delta_file, delta_rank, 1U, own_occupancy_mask, occupancy_mask);
    const uint64_t step2 =
        cmz_candidate_slider_ray_step_attention_value(from_square, delta_file, delta_rank, 2U, own_occupancy_mask, occupancy_mask);
    const uint64_t step3 =
        cmz_candidate_slider_ray_step_attention_value(from_square, delta_file, delta_rank, 3U, own_occupancy_mask, occupancy_mask);
    const uint64_t step4 =
        cmz_candidate_slider_ray_step_attention_value(from_square, delta_file, delta_rank, 4U, own_occupancy_mask, occupancy_mask);
    const uint64_t step5 =
        cmz_candidate_slider_ray_step_attention_value(from_square, delta_file, delta_rank, 5U, own_occupancy_mask, occupancy_mask);
    const uint64_t step6 =
        cmz_candidate_slider_ray_step_attention_value(from_square, delta_file, delta_rank, 6U, own_occupancy_mask, occupancy_mask);
    const uint64_t step7 =
        cmz_candidate_slider_ray_step_attention_value(from_square, delta_file, delta_rank, 7U, own_occupancy_mask, occupancy_mask);
    uint64_t mask = 0ULL;
    cmz_qk2_select_or_write_u64(1U, 0U, step1 == 0ULL ? 0U : 1U, 0U, step1, &mask);
    cmz_qk2_select_or_write_u64(1U, 0U, step2 == 0ULL ? 0U : 1U, 0U, step2, &mask);
    cmz_qk2_select_or_write_u64(1U, 0U, step3 == 0ULL ? 0U : 1U, 0U, step3, &mask);
    cmz_qk2_select_or_write_u64(1U, 0U, step4 == 0ULL ? 0U : 1U, 0U, step4, &mask);
    cmz_qk2_select_or_write_u64(1U, 0U, step5 == 0ULL ? 0U : 1U, 0U, step5, &mask);
    cmz_qk2_select_or_write_u64(1U, 0U, step6 == 0ULL ? 0U : 1U, 0U, step6, &mask);
    cmz_qk2_select_or_write_u64(1U, 0U, step7 == 0ULL ? 0U : 1U, 0U, step7, &mask);
    return mask;
}

__device__ uint64_t cmz_candidate_slider_target_mask_attention_value(
    uint32_t token,
    uint32_t from_square,
    uint64_t own_occupancy_mask,
    uint64_t occupancy_mask) {
    const uint32_t diagonal_piece = (token == 3U || token == 5U || token == 9U || token == 11U) ? 1U : 0U;
    const uint32_t orthogonal_piece = (token == 4U || token == 5U || token == 10U || token == 11U) ? 1U : 0U;
    const uint64_t diagonal_ne =
        cmz_candidate_slider_ray_slot_attention_value(from_square, 1, 1, own_occupancy_mask, occupancy_mask);
    const uint64_t diagonal_se =
        cmz_candidate_slider_ray_slot_attention_value(from_square, 1, -1, own_occupancy_mask, occupancy_mask);
    const uint64_t diagonal_nw =
        cmz_candidate_slider_ray_slot_attention_value(from_square, -1, 1, own_occupancy_mask, occupancy_mask);
    const uint64_t diagonal_sw =
        cmz_candidate_slider_ray_slot_attention_value(from_square, -1, -1, own_occupancy_mask, occupancy_mask);
    const uint64_t orthogonal_e =
        cmz_candidate_slider_ray_slot_attention_value(from_square, 1, 0, own_occupancy_mask, occupancy_mask);
    const uint64_t orthogonal_w =
        cmz_candidate_slider_ray_slot_attention_value(from_square, -1, 0, own_occupancy_mask, occupancy_mask);
    const uint64_t orthogonal_n =
        cmz_candidate_slider_ray_slot_attention_value(from_square, 0, 1, own_occupancy_mask, occupancy_mask);
    const uint64_t orthogonal_s =
        cmz_candidate_slider_ray_slot_attention_value(from_square, 0, -1, own_occupancy_mask, occupancy_mask);

    uint64_t mask = 0ULL;
    cmz_qk2_select_or_write_u64(1U, 0U, diagonal_piece, 0U, diagonal_ne, &mask);
    cmz_qk2_select_or_write_u64(1U, 0U, diagonal_piece, 0U, diagonal_se, &mask);
    cmz_qk2_select_or_write_u64(1U, 0U, diagonal_piece, 0U, diagonal_nw, &mask);
    cmz_qk2_select_or_write_u64(1U, 0U, diagonal_piece, 0U, diagonal_sw, &mask);
    cmz_qk2_select_or_write_u64(1U, 0U, orthogonal_piece, 0U, orthogonal_e, &mask);
    cmz_qk2_select_or_write_u64(1U, 0U, orthogonal_piece, 0U, orthogonal_w, &mask);
    cmz_qk2_select_or_write_u64(1U, 0U, orthogonal_piece, 0U, orthogonal_n, &mask);
    cmz_qk2_select_or_write_u64(1U, 0U, orthogonal_piece, 0U, orthogonal_s, &mask);
    return mask;
}

__device__ uint64_t cmz_candidate_target_mask_qk_hardmax_v_attention_value(
    uint32_t token,
    uint32_t from_square,
    uint64_t friendly_mask,
    uint64_t enemy_mask,
    uint64_t occupancy_mask,
    uint32_t ep_square) {
    const int knight_delta_files[8] = {1, 2, 2, 1, -1, -2, -2, -1};
    const int knight_delta_ranks[8] = {2, 1, -1, -2, -2, -1, 1, 2};
    const int king_delta_files[8] = {1, 1, 0, -1, -1, -1, 0, 1};
    const int king_delta_ranks[8] = {0, 1, 1, 1, 0, -1, -1, -1};

    const uint32_t query_family = cmz_candidate_piece_family_attention_query(token);
    const uint64_t pawn_mask =
        cmz_candidate_pawn_target_mask_attention_value(token, from_square, enemy_mask, occupancy_mask, ep_square);
    const uint64_t knight_mask =
        cmz_candidate_offset_target_mask_attention_value(from_square, friendly_mask, knight_delta_files, knight_delta_ranks);
    const uint64_t king_mask =
        cmz_candidate_offset_target_mask_attention_value(from_square, friendly_mask, king_delta_files, king_delta_ranks);
    const uint64_t slider_mask =
        cmz_candidate_slider_target_mask_attention_value(token, from_square, friendly_mask, occupancy_mask);

    uint32_t selected_score = 0U;
    uint32_t selected_tie_index = 0xffffffffU;
    uint64_t selected_mask = 0ULL;
    cmz_qk2_hardmax_select_u64(
        1U, 0U, query_family == 1U ? 1U : 0U, 0U, 1U, pawn_mask, &selected_score, &selected_tie_index, &selected_mask);
    cmz_qk2_hardmax_select_u64(
        1U, 0U, query_family == 2U ? 1U : 0U, 0U, 2U, knight_mask, &selected_score, &selected_tie_index, &selected_mask);
    cmz_qk2_hardmax_select_u64(
        1U, 0U, query_family == 3U ? 1U : 0U, 0U, 3U, king_mask, &selected_score, &selected_tie_index, &selected_mask);
    cmz_qk2_hardmax_select_u64(
        1U, 0U, query_family == 4U ? 1U : 0U, 0U, 4U, slider_mask, &selected_score, &selected_tie_index, &selected_mask);
    return selected_score == 0U ? 0ULL : selected_mask;
}

__device__ uint64_t cmz_castle_target_mask_value(const uint32_t* board, uint32_t castling_rights, bool white) {
    uint64_t mask = 0ULL;
    if (white) {
        if (board[4] == 6U && !cmz_tokens_attacked_by(board, 4, false)) {
            if ((castling_rights & 1U) != 0U && board[5] == 0U && board[6] == 0U && board[7] == 4U &&
                !cmz_tokens_attacked_by(board, 5, false) && !cmz_tokens_attacked_by(board, 6, false)) {
                mask |= cmz_bit(6U);
            }
            if ((castling_rights & 2U) != 0U && board[3] == 0U && board[2] == 0U && board[1] == 0U &&
                board[0] == 4U && !cmz_tokens_attacked_by(board, 3, false) &&
                !cmz_tokens_attacked_by(board, 2, false)) {
                mask |= cmz_bit(2U);
            }
        }
    } else if (board[60] == 12U && !cmz_tokens_attacked_by(board, 60, true)) {
        if ((castling_rights & 4U) != 0U && board[61] == 0U && board[62] == 0U && board[63] == 10U &&
            !cmz_tokens_attacked_by(board, 61, true) && !cmz_tokens_attacked_by(board, 62, true)) {
            mask |= cmz_bit(62U);
        }
        if ((castling_rights & 8U) != 0U && board[59] == 0U && board[58] == 0U && board[57] == 0U &&
            board[56] == 10U && !cmz_tokens_attacked_by(board, 59, true) &&
            !cmz_tokens_attacked_by(board, 58, true)) {
            mask |= cmz_bit(58U);
        }
    }
    return mask;
}

__device__ uint32_t cmz_candidate_record_slot_qk_write_value(
    const uint32_t* board,
    const uint32_t* active_piece_tokens,
    const uint64_t* target_masks,
    const uint32_t* promotion_flags,
    uint32_t ep_square,
    uint32_t slot,
    uint32_t* out_from,
    uint32_t* out_to,
    uint32_t* out_promotion,
    uint32_t* out_en_passant,
    uint32_t* out_castle) {
    constexpr uint32_t kPromotionSlotCount = 5U;
    constexpr uint32_t kToSlotCount = 64U * kPromotionSlotCount;
    const uint32_t from = slot / kToSlotCount;
    const uint32_t to = (slot / kPromotionSlotCount) % 64U;
    const uint32_t promotion_slot = slot % kPromotionSlotCount;
    const uint32_t token = active_piece_tokens[from];
    const uint32_t target_selected = (target_masks[from] & cmz_bit(to)) != 0ULL ? 1U : 0U;
    const uint32_t promotion_from_rank = promotion_flags[from] != 0U ? 1U : 0U;
    const uint32_t promotion_slot_selected =
        promotion_from_rank != 0U ? (promotion_slot >= 1U && promotion_slot <= 4U ? 1U : 0U)
                                  : (promotion_slot == 0U ? 1U : 0U);
    const uint32_t active_slot = token != 0U && target_selected != 0U && promotion_slot_selected != 0U ? 1U : 0U;

    uint64_t selected_valid = 0ULL;
    cmz_qk2_select_or_write_u64(1U, 0U, active_slot, 0U, 1ULL, &selected_valid);
    if (selected_valid == 0ULL) {
        return 0U;
    }

    *out_from = from;
    *out_to = to;
    *out_promotion = promotion_from_rank != 0U ? promotion_slot : 0U;
    *out_en_passant = (token == 1U || token == 7U) && ep_square < 64U && to == ep_square && board[to] == 0U ? 1U : 0U;
    *out_castle = (token == 6U || token == 12U) && promotion_from_rank == 0U &&
                          ((from > to ? from - to : to - from) == 2U)
                      ? 1U
                      : 0U;
    return 1U;
}

__device__ void cmz_write_candidate_record_fields(
    uint32_t* records,
    uint32_t record_index,
    uint32_t from_square,
    uint32_t to_square,
    uint32_t promotion_id,
    uint32_t en_passant,
    uint32_t castle) {
    const uint32_t base = record_index * kCmzCandidateRecordWidth;
    records[base] = from_square;
    records[base + 1U] = to_square;
    records[base + 2U] = promotion_id;
    records[base + 3U] = en_passant;
    records[base + 4U] = castle;
}

__device__ uint32_t cmz_candidate_record_prefix_rank_attention_value(const uint32_t* slot_ranks, uint32_t slot) {
    uint32_t selected_score = 0U;
    uint32_t selected_tie_index = 0xffffffffU;
    uint32_t selected_rank = 0U;
    cmz_qk2_hardmax_select_u32(
        1U,
        0U,
        slot < kCmzCandidateRecordSlotCount ? 1U : 0U,
        0U,
        0U,
        slot_ranks[slot],
        &selected_score,
        &selected_tie_index,
        &selected_rank);
    return selected_score == 0U ? 0U : selected_rank;
}

__device__ uint32_t cmz_candidate_record_total_count_attention_value(
    const uint32_t* slot_ranks,
    const uint32_t* slot_valid) {
    constexpr uint32_t last_slot = kCmzCandidateRecordSlotCount - 1U;
    const uint32_t total = slot_ranks[last_slot] + slot_valid[last_slot];
    uint32_t selected_score = 0U;
    uint32_t selected_tie_index = 0xffffffffU;
    uint32_t selected_total = 0U;
    cmz_qk2_hardmax_select_u32(
        1U,
        0U,
        1U,
        0U,
        0U,
        total,
        &selected_score,
        &selected_tie_index,
        &selected_total);
    return selected_score == 0U ? 0U : selected_total;
}

__global__ void cmz_candidate_context_select_attention_kernel(
    const uint32_t* board,
    uint32_t white_to_move,
    uint64_t* context_masks) {
    if (threadIdx.x != 0U || blockIdx.x != 0U) {
        return;
    }
    const bool white = white_to_move != 0U;
    uint64_t friendly = 0ULL;
    uint64_t enemy = 0ULL;
    for (uint32_t square = 0; square < 64U; ++square) {
        const uint32_t token = board[square];
        if (cmz_token_is_side(token, white)) {
            friendly |= cmz_bit(square);
        } else if (cmz_token_is_side(token, !white)) {
            enemy |= cmz_bit(square);
        }
    }
    context_masks[0] = friendly;
    context_masks[1] = enemy;
    context_masks[2] = friendly | enemy;
}

__global__ void cmz_candidate_piece_dispatch_attention_kernel(
    const uint32_t* board,
    uint32_t white_to_move,
    uint32_t* active_piece_tokens) {
    const uint32_t square = threadIdx.x + blockIdx.x * blockDim.x;
    if (square >= 64U) {
        return;
    }
    const bool white = white_to_move != 0U;
    const uint32_t token = board[square];
    active_piece_tokens[square] = cmz_token_is_side(token, white) ? token : 0U;
}

__global__ void cmz_candidate_target_mask_select_attention_kernel(
    const uint32_t* active_piece_tokens,
    const uint64_t* context_masks,
    uint32_t ep_square,
    uint64_t* target_masks) {
    const uint32_t from = threadIdx.x + blockIdx.x * blockDim.x;
    if (from >= 64U) {
        return;
    }
    const uint32_t token = active_piece_tokens[from];
    if (token == 0U) {
        target_masks[from] = 0ULL;
        return;
    }
    target_masks[from] = cmz_candidate_target_mask_qk_hardmax_v_attention_value(
        token,
        from,
        context_masks[0],
        context_masks[1],
        context_masks[2],
        ep_square);
}

__global__ void cmz_candidate_castle_merge_attention_kernel(
    const uint32_t* board,
    uint32_t white_to_move,
    uint32_t castling_rights,
    uint64_t* target_masks) {
    if (threadIdx.x != 0U || blockIdx.x != 0U) {
        return;
    }
    const bool white = white_to_move != 0U;
    const uint32_t king_square = white ? 4U : 60U;
    target_masks[king_square] |= cmz_castle_target_mask_value(board, castling_rights, white);
}

__global__ void cmz_candidate_promotion_expand_attention_kernel(
    const uint32_t* active_piece_tokens,
    uint32_t* promotion_flags) {
    const uint32_t from = threadIdx.x + blockIdx.x * blockDim.x;
    if (from >= 64U) {
        return;
    }
    const uint32_t token = active_piece_tokens[from];
    const uint32_t rank = from / 8U;
    promotion_flags[from] = (token == 1U && rank == 6U) || (token == 7U && rank == 1U) ? 1U : 0U;
}

__global__ void cmz_candidate_record_slot_validity_qk_write_kernel(
    const uint32_t* board,
    const uint32_t* active_piece_tokens,
    const uint64_t* target_masks,
    const uint32_t* promotion_flags,
    uint32_t ep_square,
    uint32_t* slot_records,
    uint32_t* slot_valid) {
    const uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= kCmzCandidateRecordSlotCount) {
        return;
    }

    uint32_t from = 0U;
    uint32_t to = 0U;
    uint32_t promotion = 0U;
    uint32_t en_passant = 0U;
    uint32_t castle = 0U;
    const uint32_t valid = cmz_candidate_record_slot_qk_write_value(
        board,
        active_piece_tokens,
        target_masks,
        promotion_flags,
        ep_square,
        slot,
        &from,
        &to,
        &promotion,
        &en_passant,
        &castle);
    slot_valid[slot] = valid;
    cmz_write_candidate_record_fields(slot_records, slot, from, to, promotion, en_passant, castle);
}

__global__ void cmz_candidate_record_slot_rank_write_attention_kernel(
    const uint32_t* slot_records,
    const uint32_t* slot_valid,
    const uint32_t* slot_ranks,
    uint32_t* records,
    size_t record_capacity,
    uint32_t* record_count) {
    const uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= kCmzCandidateRecordSlotCount) {
        return;
    }
    if (slot == kCmzCandidateRecordSlotCount - 1U) {
        *record_count = cmz_candidate_record_total_count_attention_value(slot_ranks, slot_valid);
    }
    if (slot_valid[slot] == 0U) {
        return;
    }

    const uint32_t rank = cmz_candidate_record_prefix_rank_attention_value(slot_ranks, slot);
    if (rank < record_capacity) {
        const uint32_t src_base = slot * kCmzCandidateRecordWidth;
        const uint32_t dst_base = rank * kCmzCandidateRecordWidth;
        records[dst_base] = slot_records[src_base];
        records[dst_base + 1U] = slot_records[src_base + 1U];
        records[dst_base + 2U] = slot_records[src_base + 2U];
        records[dst_base + 3U] = slot_records[src_base + 3U];
        records[dst_base + 4U] = slot_records[src_base + 4U];
    }
}

__global__ void cmz_candidate_record_order_select_attention_kernel(uint32_t* record_count, size_t record_capacity) {
    if (threadIdx.x != 0U || blockIdx.x != 0U) {
        return;
    }
    if (*record_count > record_capacity) {
        *record_count = static_cast<uint32_t>(record_capacity);
    }
}

__host__ __device__ uint32_t cmz_promotion_token_from_id(bool white, uint32_t promotion_id) {
    if (promotion_id == 0U) {
        return 0U;
    }
    if (white) {
        const uint32_t values[5] = {0U, 2U, 3U, 4U, 5U};
        return values[promotion_id <= 4U ? promotion_id : 0U];
    }
    const uint32_t values[5] = {0U, 8U, 9U, 10U, 11U};
    return values[promotion_id <= 4U ? promotion_id : 0U];
}

__device__ bool cmz_candidate_move_is_legal(
    const uint32_t* board,
    bool white,
    uint32_t from,
    uint32_t to,
    uint32_t promotion_id,
    uint32_t en_passant,
    uint32_t castle) {
    uint32_t next_board[64];
    for (uint32_t square = 0; square < 64U; ++square) {
        next_board[square] = board[square];
    }
    const uint32_t moving = board[from];
    next_board[from] = 0U;
    if (en_passant != 0U) {
        const int captured = white ? static_cast<int>(to) - 8 : static_cast<int>(to) + 8;
        if (captured >= 0 && captured < 64) {
            next_board[static_cast<uint32_t>(captured)] = 0U;
        }
    }
    if (castle != 0U) {
        if (from == 4U && to == 6U) {
            next_board[7] = 0U;
            next_board[5] = 4U;
        } else if (from == 4U && to == 2U) {
            next_board[0] = 0U;
            next_board[3] = 4U;
        } else if (from == 60U && to == 62U) {
            next_board[63] = 0U;
            next_board[61] = 10U;
        } else if (from == 60U && to == 58U) {
            next_board[56] = 0U;
            next_board[59] = 10U;
        }
    }
    const uint32_t promotion = cmz_promotion_token_from_id(white, promotion_id);
    next_board[to] = promotion != 0U ? promotion : moving;

    const uint32_t king_token = white ? 6U : 12U;
    int king_square = -1;
    for (uint32_t square = 0; square < 64U; ++square) {
        if (next_board[square] == king_token) {
            king_square = static_cast<int>(square);
            break;
        }
    }
    return king_square >= 0 && !cmz_tokens_attacked_by(next_board, king_square, !white);
}

constexpr uint32_t kCmzTerminalMaterialMaskCount = 7U;
constexpr uint32_t kCmzTerminalMaterialNonKingMask = 0U;
constexpr uint32_t kCmzTerminalMaterialMinorMask = 1U;
constexpr uint32_t kCmzTerminalMaterialBishopMask = 2U;
constexpr uint32_t kCmzTerminalMaterialKnightMask = 3U;
constexpr uint32_t kCmzTerminalMaterialStrongMask = 4U;
constexpr uint32_t kCmzTerminalMaterialLightBishopMask = 5U;
constexpr uint32_t kCmzTerminalMaterialDarkBishopMask = 6U;
constexpr uint32_t kCmzTerminalMaterialClassNonKing = 1U;
constexpr uint32_t kCmzTerminalMaterialClassMinor = 2U;
constexpr uint32_t kCmzTerminalMaterialClassBishop = 4U;
constexpr uint32_t kCmzTerminalMaterialClassKnight = 8U;
constexpr uint32_t kCmzTerminalMaterialClassStrong = 16U;
constexpr uint32_t kCmzTerminalMaterialTokenMaxCode = 12U;
constexpr uint32_t kCmzTerminalMaterialTokenScoreBias =
    kCmzTerminalMaterialTokenMaxCode * kCmzTerminalMaterialTokenMaxCode;

__device__ uint32_t cmz_terminal_material_token_key_bias(uint32_t token) {
    return kCmzTerminalMaterialTokenScoreBias - token * token;
}

#define CMZ_TERMINAL_MATERIAL_CLASS_ATTEND_TOKEN(TOKEN, VALUE) \
    cmz_qk2_hardmax_select_u32( \
        query_x, \
        query_y, \
        (TOKEN), \
        cmz_terminal_material_token_key_bias((TOKEN)), \
        (TOKEN), \
        (VALUE), \
        &selected_score, \
        &selected_tie_index, \
        &selected_value)

__device__ uint32_t cmz_terminal_material_square_class_attention_value(uint32_t token) {
    const uint32_t query_x = token * 2U;
    const uint32_t query_y = 1U;
    uint32_t selected_score = 0U;
    uint32_t selected_tie_index = 0xffffffffU;
    uint32_t selected_value = 0U;
    constexpr uint32_t minor_bishop =
        kCmzTerminalMaterialClassNonKing | kCmzTerminalMaterialClassMinor | kCmzTerminalMaterialClassBishop;
    constexpr uint32_t minor_knight =
        kCmzTerminalMaterialClassNonKing | kCmzTerminalMaterialClassMinor | kCmzTerminalMaterialClassKnight;
    constexpr uint32_t strong_piece = kCmzTerminalMaterialClassNonKing | kCmzTerminalMaterialClassStrong;

    CMZ_TERMINAL_MATERIAL_CLASS_ATTEND_TOKEN(0U, 0U);
    CMZ_TERMINAL_MATERIAL_CLASS_ATTEND_TOKEN(1U, strong_piece);
    CMZ_TERMINAL_MATERIAL_CLASS_ATTEND_TOKEN(2U, minor_knight);
    CMZ_TERMINAL_MATERIAL_CLASS_ATTEND_TOKEN(3U, minor_bishop);
    CMZ_TERMINAL_MATERIAL_CLASS_ATTEND_TOKEN(4U, strong_piece);
    CMZ_TERMINAL_MATERIAL_CLASS_ATTEND_TOKEN(5U, strong_piece);
    CMZ_TERMINAL_MATERIAL_CLASS_ATTEND_TOKEN(6U, 0U);
    CMZ_TERMINAL_MATERIAL_CLASS_ATTEND_TOKEN(7U, strong_piece);
    CMZ_TERMINAL_MATERIAL_CLASS_ATTEND_TOKEN(8U, minor_knight);
    CMZ_TERMINAL_MATERIAL_CLASS_ATTEND_TOKEN(9U, minor_bishop);
    CMZ_TERMINAL_MATERIAL_CLASS_ATTEND_TOKEN(10U, strong_piece);
    CMZ_TERMINAL_MATERIAL_CLASS_ATTEND_TOKEN(11U, strong_piece);
    CMZ_TERMINAL_MATERIAL_CLASS_ATTEND_TOKEN(12U, 0U);
    return selected_value;
}

__device__ void cmz_terminal_material_mask_or_attention_value(
    uint64_t* masks,
    uint32_t mask_index,
    uint32_t condition,
    uint64_t bit) {
    const uint64_t selected = condition == 0U ? 0ULL : bit;
    atomicOr(
        reinterpret_cast<unsigned long long*>(&masks[mask_index]),
        static_cast<unsigned long long>(selected));
}

__global__ void cmz_terminal_material_square_class_attention_kernel(const uint32_t* board, uint64_t* masks) {
    const uint32_t square = threadIdx.x + blockIdx.x * blockDim.x;
    if (square >= 64U) {
        return;
    }
    const uint32_t flags = cmz_terminal_material_square_class_attention_value(board[square]);
    const uint64_t bit = cmz_bit(square);
    const uint32_t bishop_color = ((square % 8U) + (square / 8U)) & 1U;
    const uint32_t non_king = (flags & kCmzTerminalMaterialClassNonKing) == 0U ? 0U : 1U;
    const uint32_t minor = (flags & kCmzTerminalMaterialClassMinor) == 0U ? 0U : 1U;
    const uint32_t bishop = (flags & kCmzTerminalMaterialClassBishop) == 0U ? 0U : 1U;
    const uint32_t knight = (flags & kCmzTerminalMaterialClassKnight) == 0U ? 0U : 1U;
    const uint32_t strong = (flags & kCmzTerminalMaterialClassStrong) == 0U ? 0U : 1U;

    cmz_terminal_material_mask_or_attention_value(masks, kCmzTerminalMaterialNonKingMask, non_king, bit);
    cmz_terminal_material_mask_or_attention_value(masks, kCmzTerminalMaterialMinorMask, minor, bit);
    cmz_terminal_material_mask_or_attention_value(masks, kCmzTerminalMaterialBishopMask, bishop, bit);
    cmz_terminal_material_mask_or_attention_value(masks, kCmzTerminalMaterialKnightMask, knight, bit);
    cmz_terminal_material_mask_or_attention_value(masks, kCmzTerminalMaterialStrongMask, strong, bit);
    cmz_terminal_material_mask_or_attention_value(
        masks,
        kCmzTerminalMaterialLightBishopMask,
        bishop & (bishop_color == 0U ? 1U : 0U),
        bit);
    cmz_terminal_material_mask_or_attention_value(
        masks,
        kCmzTerminalMaterialDarkBishopMask,
        bishop & (bishop_color == 0U ? 0U : 1U),
        bit);
}

__device__ uint32_t cmz_terminal_material_status_from_masks_attention_value(const uint64_t* masks) {
    const uint32_t non_king_count = static_cast<uint32_t>(__popcll(masks[kCmzTerminalMaterialNonKingMask]));
    const uint32_t minor_count = static_cast<uint32_t>(__popcll(masks[kCmzTerminalMaterialMinorMask]));
    const uint32_t bishop_count = static_cast<uint32_t>(__popcll(masks[kCmzTerminalMaterialBishopMask]));
    const uint32_t knight_count = static_cast<uint32_t>(__popcll(masks[kCmzTerminalMaterialKnightMask]));
    const uint32_t strong_count = static_cast<uint32_t>(__popcll(masks[kCmzTerminalMaterialStrongMask]));
    const uint32_t light_bishop_count = static_cast<uint32_t>(__popcll(masks[kCmzTerminalMaterialLightBishopMask]));
    const uint32_t dark_bishop_count = static_cast<uint32_t>(__popcll(masks[kCmzTerminalMaterialDarkBishopMask]));
    const uint32_t strong_absent = strong_count == 0U ? 1U : 0U;
    const uint32_t only_kings = strong_absent & (non_king_count == 0U ? 1U : 0U);
    const uint32_t single_minor =
        strong_absent & (non_king_count == 1U ? 1U : 0U) & (minor_count == 1U ? 1U : 0U);
    const uint32_t all_bishops = (bishop_count == non_king_count && non_king_count > 0U) ? 1U : 0U;
    const uint32_t no_knights = knight_count == 0U ? 1U : 0U;
    const uint32_t same_light_bishops = light_bishop_count == bishop_count ? 1U : 0U;
    const uint32_t same_dark_bishops = dark_bishop_count == bishop_count ? 1U : 0U;
    const uint32_t same_color_bishops =
        strong_absent & no_knights & all_bishops & (same_light_bishops | same_dark_bishops);
    uint32_t selected_value = 0U;
    cmz_qk2_select_or_write_u32(1U, 0U, only_kings, 0U, 1U, &selected_value);
    cmz_qk2_select_or_write_u32(1U, 0U, single_minor, 0U, 1U, &selected_value);
    cmz_qk2_select_or_write_u32(1U, 0U, same_color_bishops, 0U, 1U, &selected_value);
    return selected_value;
}


__device__ void cmz_terminal_set_status(uint32_t* state, uint32_t result, uint32_t reason) {
    state[0] = result;
    state[1] = reason;
    state[2] = 1U;
}

__global__ void cmz_terminal_draw_rule_select_attention_kernel(
    uint32_t halfmove_clock,
    uint32_t repetition_count,
    uint32_t adjudication_cap_reached,
    uint32_t* state) {
    if (threadIdx.x != 0U || blockIdx.x != 0U) {
        return;
    }
    constexpr uint32_t result_draw = 3U;
    constexpr uint32_t reason_fifty = 3U;
    constexpr uint32_t reason_threefold = 4U;
    constexpr uint32_t reason_cap = 6U;

    state[0] = 0U;
    state[1] = 0U;
    state[2] = 0U;
    state[3] = 0U;
    if (adjudication_cap_reached != 0U) {
        cmz_terminal_set_status(state, result_draw, reason_cap);
        return;
    }
    if (repetition_count >= 3U) {
        cmz_terminal_set_status(state, result_draw, reason_threefold);
        return;
    }
    if (halfmove_clock >= 100U) {
        cmz_terminal_set_status(state, result_draw, reason_fifty);
        return;
    }
}

__global__ void cmz_terminal_legal_presence_from_batch_attention_kernel(
    const uint32_t* legal_bits,
    size_t move_count,
    uint32_t* state) {
    if (threadIdx.x != 0U || blockIdx.x != 0U || state[2] != 0U) {
        return;
    }
    uint32_t selected_score = 0U;
    uint32_t selected_tie_index = 0xffffffffU;
    uint32_t selected_value = 0U;
    for (size_t index = 0; index < move_count; ++index) {
        const uint32_t legal = legal_bits[index] == 0U ? 0U : 1U;
        cmz_qk2_hardmax_select_u32(
            1U,
            0U,
            legal,
            0U,
            static_cast<uint32_t>(index),
            legal,
            &selected_score,
            &selected_tie_index,
            &selected_value);
    }
    state[3] = selected_value;
}

__global__ void cmz_terminal_check_state_select_attention_kernel(
    const uint32_t* board,
    uint32_t white_to_move,
    uint32_t* state) {
    if (threadIdx.x != 0U || blockIdx.x != 0U || state[2] != 0U || state[3] != 0U) {
        return;
    }
    constexpr uint32_t result_white_win = 1U;
    constexpr uint32_t result_black_win = 2U;
    constexpr uint32_t result_draw = 3U;
    constexpr uint32_t reason_checkmate = 1U;
    constexpr uint32_t reason_stalemate = 2U;

    const bool white = white_to_move != 0U;
    const uint32_t king_token = white ? 6U : 12U;
    int king_square = -1;
    for (uint32_t square = 0; square < 64U; ++square) {
        if (board[square] == king_token) {
            king_square = static_cast<int>(square);
            break;
        }
    }
    const bool in_check = king_square >= 0 && cmz_tokens_attacked_by(board, king_square, !white);
    cmz_terminal_set_status(
        state,
        in_check ? (white ? result_black_win : result_white_win) : result_draw,
        in_check ? reason_checkmate : reason_stalemate);
}

__global__ void cmz_terminal_material_select_attention_kernel(const uint64_t* material_masks, uint32_t* state) {
    if (threadIdx.x != 0U || blockIdx.x != 0U || state[2] != 0U) {
        return;
    }
    constexpr uint32_t result_draw = 3U;
    constexpr uint32_t reason_insufficient = 5U;
    if (cmz_terminal_material_status_from_masks_attention_value(material_masks) != 0U) {
        cmz_terminal_set_status(state, result_draw, reason_insufficient);
        return;
    }
}

__global__ void cmz_terminal_final_status_select_attention_kernel(uint32_t* state, uint32_t* result_reason) {
    if (threadIdx.x != 0U || blockIdx.x != 0U) {
        return;
    }
    constexpr uint32_t result_ongoing = 0U;
    constexpr uint32_t reason_none = 0U;
    if (state[2] == 0U) {
        state[0] = result_ongoing;
        state[1] = reason_none;
    }
    result_reason[0] = state[0];
    result_reason[1] = state[1];
}

__device__ long long cmz_resolve_move_qk_code(uint32_t from, uint32_t to, uint32_t promotion) {
    return static_cast<long long>((from * 64U + to) * 5U + promotion);
}

__global__ void cmz_resolve_move_qk_hardmax_legal_set_attention_kernel(
    const uint32_t* records,
    const uint32_t* legal_bits,
    size_t record_count,
    uint32_t requested_from,
    uint32_t requested_to,
    uint32_t requested_promotion,
    uint32_t* selected_record,
    uint32_t* found) {
    constexpr uint32_t record_width = 5U;
    constexpr long long min_score = -0x3fffffffffffffffLL;
    __shared__ long long scores[1024];
    __shared__ uint32_t indices[1024];

    const uint32_t tid = threadIdx.x;
    if (blockIdx.x != 0U || tid >= blockDim.x) {
        return;
    }

    const long long query_code = cmz_resolve_move_qk_code(requested_from, requested_to, requested_promotion);
    long long score = min_score;
    uint32_t selected_index = tid;
    if (tid < record_count && legal_bits[tid] != 0U) {
        const size_t base = static_cast<size_t>(tid) * record_width;
        const long long key_code = cmz_resolve_move_qk_code(records[base], records[base + 1U], records[base + 2U]);
        score = 2LL * query_code * key_code - key_code * key_code;
    }
    scores[tid] = score;
    indices[tid] = selected_index;
    __syncthreads();

    for (uint32_t stride = blockDim.x / 2U; stride > 0U; stride >>= 1U) {
        if (tid < stride) {
            const long long other_score = scores[tid + stride];
            const uint32_t other_index = indices[tid + stride];
            if (other_score > scores[tid] || (other_score == scores[tid] && other_index < indices[tid])) {
                scores[tid] = other_score;
                indices[tid] = other_index;
            }
        }
        __syncthreads();
    }

    if (tid == 0U) {
        *found = 0U;
        if (record_count == 0U || scores[0] == min_score) {
            return;
        }
        const uint32_t best_index = indices[0];
        const size_t base = static_cast<size_t>(best_index) * record_width;
        const bool exact =
            legal_bits[best_index] != 0U && records[base] == requested_from && records[base + 1U] == requested_to &&
            records[base + 2U] == requested_promotion;
        if (!exact) {
            return;
        }
        for (uint32_t field = 0; field < record_width; ++field) {
            selected_record[field] = records[base + field];
        }
        *found = 1U;
    }
}

extern "C" int cmz_cuda_double_values(const uint32_t* host_input, size_t len, uint32_t* host_output) {
    if (len == 0) {
        return 0;
    }
    uint32_t* device_input = nullptr;
    uint32_t* device_output = nullptr;
    const size_t bytes = len * sizeof(uint32_t);
    cudaError_t status = cudaMalloc(&device_input, bytes);
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }
    status = cudaMalloc(&device_output, bytes);
    if (status != cudaSuccess) {
        cudaFree(device_input);
        return static_cast<int>(status);
    }
    status = cudaMemcpy(device_input, host_input, bytes, cudaMemcpyHostToDevice);
    if (status == cudaSuccess) {
        const int block = 256;
        const int grid = static_cast<int>((len + block - 1) / block);
        cmz_double_kernel<<<grid, block>>>(device_input, device_output, len);
        status = cudaGetLastError();
    }
    if (status == cudaSuccess) {
        status = cudaMemcpy(host_output, device_output, bytes, cudaMemcpyDeviceToHost);
    }
    cudaFree(device_output);
    cudaFree(device_input);
    return static_cast<int>(status);
}

extern "C" int cmz_cuda_hardmax2d_values(
    const float* host_keys_xy,
    size_t len,
    float query_x,
    float query_y,
    uint32_t* host_selected,
    float* host_score) {
    if (len == 0) {
        return cudaErrorInvalidValue;
    }
    if (host_keys_xy == nullptr || host_selected == nullptr || host_score == nullptr) {
        return cudaErrorInvalidValue;
    }

    float* device_keys = nullptr;
    float* device_scores = nullptr;
    uint32_t* device_selected = nullptr;
    float* device_selected_score = nullptr;
    const size_t key_bytes = len * 2 * sizeof(float);
    const size_t score_bytes = len * sizeof(float);
    cudaError_t status = cudaMalloc(&device_keys, key_bytes);
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }
    status = cudaMalloc(&device_scores, score_bytes);
    if (status != cudaSuccess) {
        cudaFree(device_keys);
        return static_cast<int>(status);
    }
    status = cudaMalloc(&device_selected, sizeof(uint32_t));
    if (status != cudaSuccess) {
        cudaFree(device_scores);
        cudaFree(device_keys);
        return static_cast<int>(status);
    }
    status = cudaMalloc(&device_selected_score, sizeof(float));
    if (status != cudaSuccess) {
        cudaFree(device_selected);
        cudaFree(device_scores);
        cudaFree(device_keys);
        return static_cast<int>(status);
    }
    status = cudaMemcpy(device_keys, host_keys_xy, key_bytes, cudaMemcpyHostToDevice);
    if (status == cudaSuccess) {
        const int block = 256;
        const int grid = static_cast<int>((len + block - 1) / block);
        cmz_dot2_kernel<<<grid, block>>>(device_keys, query_x, query_y, device_scores, len);
        status = cudaGetLastError();
    }

    if (status == cudaSuccess) {
        cmz_hardmax_float_select_kernel<<<1, 1>>>(device_scores, len, device_selected, device_selected_score);
        status = cudaGetLastError();
    }
    if (status == cudaSuccess) {
        status = cudaMemcpy(host_selected, device_selected, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    }
    if (status == cudaSuccess) {
        status = cudaMemcpy(host_score, device_selected_score, sizeof(float), cudaMemcpyDeviceToHost);
    }

    cudaFree(device_selected_score);
    cudaFree(device_selected);
    cudaFree(device_scores);
    cudaFree(device_keys);
    return static_cast<int>(status);
}

extern "C" int cmz_cutlass_hardmax2d_values(
    const float* host_keys_xy,
    size_t len,
    float query_x,
    float query_y,
    uint32_t* host_selected,
    float* host_score) {
    if (len == 0) {
        return cudaErrorInvalidValue;
    }
    if (host_keys_xy == nullptr || host_selected == nullptr || host_score == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (len > static_cast<size_t>(std::numeric_limits<int>::max())) {
        return cudaErrorInvalidValue;
    }

    float* device_query = nullptr;
    float* device_keys = nullptr;
    float* device_scores = nullptr;
    uint32_t* device_selected = nullptr;
    float* device_selected_score = nullptr;
    const float host_query[2] = {query_x, query_y};
    const size_t query_bytes = 2 * sizeof(float);
    const size_t key_bytes = len * 2 * sizeof(float);
    const size_t score_bytes = len * sizeof(float);

    cudaError_t status = cudaMalloc(&device_query, query_bytes);
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }
    status = cudaMalloc(&device_keys, key_bytes);
    if (status != cudaSuccess) {
        cudaFree(device_query);
        return static_cast<int>(status);
    }
    status = cudaMalloc(&device_scores, score_bytes);
    if (status != cudaSuccess) {
        cudaFree(device_keys);
        cudaFree(device_query);
        return static_cast<int>(status);
    }
    status = cudaMalloc(&device_selected, sizeof(uint32_t));
    if (status != cudaSuccess) {
        cudaFree(device_scores);
        cudaFree(device_keys);
        cudaFree(device_query);
        return static_cast<int>(status);
    }
    status = cudaMalloc(&device_selected_score, sizeof(float));
    if (status != cudaSuccess) {
        cudaFree(device_selected);
        cudaFree(device_scores);
        cudaFree(device_keys);
        cudaFree(device_query);
        return static_cast<int>(status);
    }

    status = cudaMemcpy(device_query, host_query, query_bytes, cudaMemcpyHostToDevice);
    if (status == cudaSuccess) {
        status = cudaMemcpy(device_keys, host_keys_xy, key_bytes, cudaMemcpyHostToDevice);
    }
    if (status == cudaSuccess) {
        using Gemm = cutlass::gemm::device::Gemm<
            float,
            cutlass::layout::RowMajor,
            float,
            cutlass::layout::ColumnMajor,
            float,
            cutlass::layout::RowMajor>;
        Gemm gemm_op;
        typename Gemm::Arguments args(
            {1, static_cast<int>(len), 2},
            {device_query, 2},
            {device_keys, 2},
            {device_scores, static_cast<int>(len)},
            {device_scores, static_cast<int>(len)},
            {1.0F, 0.0F});
        const cutlass::Status cutlass_status = gemm_op(args);
        if (cutlass_status != cutlass::Status::kSuccess) {
            status = cudaErrorUnknown;
        }
    }
    if (status == cudaSuccess) {
        status = cudaDeviceSynchronize();
    }
    if (status == cudaSuccess) {
        cmz_hardmax_float_select_kernel<<<1, 1>>>(device_scores, len, device_selected, device_selected_score);
        status = cudaGetLastError();
    }
    if (status == cudaSuccess) {
        status = cudaMemcpy(host_selected, device_selected, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    }
    if (status == cudaSuccess) {
        status = cudaMemcpy(host_score, device_selected_score, sizeof(float), cudaMemcpyDeviceToHost);
    }

    cudaFree(device_selected_score);
    cudaFree(device_selected);
    cudaFree(device_scores);
    cudaFree(device_keys);
    cudaFree(device_query);
    return static_cast<int>(status);
}

extern "C" int cmz_cutlass_topk2d_values(
    const float* host_keys_xy,
    size_t len,
    size_t k,
    float query_x,
    float query_y,
    uint32_t* host_selected) {
    if (len == 0 || k == 0 || k > len) {
        return cudaErrorInvalidValue;
    }
    if (host_keys_xy == nullptr || host_selected == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (len > static_cast<size_t>(std::numeric_limits<int>::max())) {
        return cudaErrorInvalidValue;
    }

    float* device_query = nullptr;
    float* device_keys = nullptr;
    float* device_scores = nullptr;
    uint32_t* device_selected = nullptr;
    const float host_query[2] = {query_x, query_y};
    const size_t query_bytes = 2 * sizeof(float);
    const size_t key_bytes = len * 2 * sizeof(float);
    const size_t score_bytes = len * sizeof(float);
    const size_t selected_bytes = k * sizeof(uint32_t);

    cudaError_t status = cudaMalloc(&device_query, query_bytes);
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }
    status = cudaMalloc(&device_keys, key_bytes);
    if (status != cudaSuccess) {
        cudaFree(device_query);
        return static_cast<int>(status);
    }
    status = cudaMalloc(&device_scores, score_bytes);
    if (status != cudaSuccess) {
        cudaFree(device_keys);
        cudaFree(device_query);
        return static_cast<int>(status);
    }
    status = cudaMalloc(&device_selected, selected_bytes);
    if (status != cudaSuccess) {
        cudaFree(device_scores);
        cudaFree(device_keys);
        cudaFree(device_query);
        return static_cast<int>(status);
    }

    status = cudaMemcpy(device_query, host_query, query_bytes, cudaMemcpyHostToDevice);
    if (status == cudaSuccess) {
        status = cudaMemcpy(device_keys, host_keys_xy, key_bytes, cudaMemcpyHostToDevice);
    }
    if (status == cudaSuccess) {
        using Gemm = cutlass::gemm::device::Gemm<
            float,
            cutlass::layout::RowMajor,
            float,
            cutlass::layout::ColumnMajor,
            float,
            cutlass::layout::RowMajor>;
        Gemm gemm_op;
        typename Gemm::Arguments args(
            {1, static_cast<int>(len), 2},
            {device_query, 2},
            {device_keys, 2},
            {device_scores, static_cast<int>(len)},
            {device_scores, static_cast<int>(len)},
            {1.0F, 0.0F});
        const cutlass::Status cutlass_status = gemm_op(args);
        if (cutlass_status != cutlass::Status::kSuccess) {
            status = cudaErrorUnknown;
        }
    }
    if (status == cudaSuccess) {
        status = cudaDeviceSynchronize();
    }
    if (status == cudaSuccess) {
        cmz_topk_float_select_kernel<<<1, 1>>>(device_scores, len, k, device_selected);
        status = cudaGetLastError();
    }
    if (status == cudaSuccess) {
        status = cudaMemcpy(host_selected, device_selected, selected_bytes, cudaMemcpyDeviceToHost);
    }

    cudaFree(device_selected);
    cudaFree(device_scores);
    cudaFree(device_keys);
    cudaFree(device_query);
    return static_cast<int>(status);
}

extern "C" int cmz_cuda_select_trace_packet(
    const uint32_t* host_tokens,
    size_t packet_count,
    size_t query_index,
    uint32_t* host_output) {
    constexpr size_t packet_width = 7;
    if (host_tokens == nullptr || host_output == nullptr || packet_count == 0 || query_index >= packet_count) {
        return cudaErrorInvalidValue;
    }

    uint32_t* device_tokens = nullptr;
    uint32_t* device_output = nullptr;
    const size_t token_bytes = packet_count * packet_width * sizeof(uint32_t);
    const size_t output_bytes = packet_width * sizeof(uint32_t);
    cudaError_t status = cudaMalloc(&device_tokens, token_bytes);
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }
    status = cudaMalloc(&device_output, output_bytes);
    if (status != cudaSuccess) {
        cudaFree(device_tokens);
        return static_cast<int>(status);
    }
    status = cudaMemcpy(device_tokens, host_tokens, token_bytes, cudaMemcpyHostToDevice);
    if (status == cudaSuccess) {
        cmz_trace_select_kernel<<<1, 32>>>(device_tokens, device_output, packet_count, query_index);
        status = cudaGetLastError();
    }
    if (status == cudaSuccess) {
        status = cudaMemcpy(host_output, device_output, output_bytes, cudaMemcpyDeviceToHost);
    }
    cudaFree(device_output);
    cudaFree(device_tokens);
    return static_cast<int>(status);
}

extern "C" int cmz_cuda_emit_trace_packet_attention(
    uint32_t op,
    uint32_t a0,
    uint32_t a1,
    uint32_t a2,
    uint32_t a3,
    uint32_t tag,
    uint32_t commit,
    uint32_t* host_output) {
    if (host_output == nullptr) {
        return cudaErrorInvalidValue;
    }
    uint32_t* device_output = nullptr;
    constexpr size_t output_bytes = 7U * sizeof(uint32_t);
    cudaError_t status = cudaMalloc(&device_output, output_bytes);
    if (status == cudaSuccess) {
        cmz_trace_emit_packet_attention_kernel<<<1, 7>>>(op, a0, a1, a2, a3, tag, commit, device_output);
        status = cudaGetLastError();
    }
    if (status == cudaSuccess) {
        status = cudaMemcpy(host_output, device_output, output_bytes, cudaMemcpyDeviceToHost);
    }
    cudaFree(device_output);
    return static_cast<int>(status);
}

extern "C" int cmz_cuda_project_board_latest_writes(
    const uint32_t* host_tokens,
    size_t packet_count,
    uint32_t* host_square_piece_tokens,
    size_t square_capacity,
    uint32_t* host_side_to_move) {
    constexpr size_t packet_width = 7;
    constexpr size_t output_width = 65;
    if (host_square_piece_tokens == nullptr || host_side_to_move == nullptr || square_capacity < 64) {
        return cudaErrorInvalidValue;
    }
    if (host_tokens == nullptr && packet_count > 0) {
        return cudaErrorInvalidValue;
    }

    uint32_t* device_tokens = nullptr;
    uint32_t* device_output = nullptr;
    const size_t token_bytes = packet_count * packet_width * sizeof(uint32_t);
    const size_t output_bytes = output_width * sizeof(uint32_t);
    cudaError_t status = cudaMalloc(&device_output, output_bytes);
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }
    status = cudaMemset(device_output, 0, output_bytes);
    if (status == cudaSuccess && packet_count > 0) {
        status = cudaMalloc(&device_tokens, token_bytes);
        if (status == cudaSuccess) {
            status = cudaMemcpy(device_tokens, host_tokens, token_bytes, cudaMemcpyHostToDevice);
        }
    }
    if (status == cudaSuccess) {
        cmz_project_board_latest_writes_kernel<<<1, 128>>>(device_tokens, device_output, packet_count);
        status = cudaGetLastError();
    }
    if (status == cudaSuccess) {
        uint32_t host_output[output_width] = {};
        status = cudaMemcpy(host_output, device_output, output_bytes, cudaMemcpyDeviceToHost);
        if (status == cudaSuccess) {
            for (size_t square = 0; square < 64; ++square) {
                host_square_piece_tokens[square] = host_output[square];
            }
            *host_side_to_move = host_output[64];
        }
    }
    cudaFree(device_tokens);
    cudaFree(device_output);
    return static_cast<int>(status);
}

extern "C" int cmz_cuda_attack_table_lookup_attention(
    const uint32_t* host_keys,
    const uint64_t* host_values,
    size_t value_count,
    uint32_t query_key,
    uint64_t* host_output) {
    if (host_keys == nullptr || host_values == nullptr || host_output == nullptr || value_count == 0) {
        return cudaErrorInvalidValue;
    }

    uint32_t* device_keys = nullptr;
    uint64_t* device_values = nullptr;
    uint64_t* device_output = nullptr;
    uint32_t* device_found = nullptr;
    const size_t key_bytes = value_count * sizeof(uint32_t);
    const size_t value_bytes = value_count * sizeof(uint64_t);
    cudaError_t status = cudaMalloc(&device_keys, key_bytes);
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }
    status = cudaMalloc(&device_values, value_bytes);
    if (status != cudaSuccess) {
        cudaFree(device_keys);
        return static_cast<int>(status);
    }
    status = cudaMalloc(&device_output, sizeof(uint64_t));
    if (status != cudaSuccess) {
        cudaFree(device_values);
        cudaFree(device_keys);
        return static_cast<int>(status);
    }
    status = cudaMalloc(&device_found, sizeof(uint32_t));
    if (status != cudaSuccess) {
        cudaFree(device_output);
        cudaFree(device_values);
        cudaFree(device_keys);
        return static_cast<int>(status);
    }

    status = cudaMemcpy(device_keys, host_keys, key_bytes, cudaMemcpyHostToDevice);
    if (status == cudaSuccess) {
        status = cudaMemcpy(device_values, host_values, value_bytes, cudaMemcpyHostToDevice);
    }
    if (status == cudaSuccess) {
        status = cudaMemset(device_output, 0, sizeof(uint64_t));
    }
    if (status == cudaSuccess) {
        status = cudaMemset(device_found, 0, sizeof(uint32_t));
    }
    if (status == cudaSuccess) {
        const int block = 256;
        const int grid = static_cast<int>((value_count + block - 1) / block);
        cmz_attack_table_lookup_attention_kernel<<<grid, block>>>(
            device_keys, device_values, value_count, query_key, device_output, device_found);
        status = cudaGetLastError();
    }

    uint32_t host_found = 0;
    if (status == cudaSuccess) {
        status = cudaMemcpy(&host_found, device_found, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    }
    if (status == cudaSuccess && host_found == 0) {
        status = cudaErrorInvalidValue;
    }
    if (status == cudaSuccess) {
        status = cudaMemcpy(host_output, device_output, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    }

    cudaFree(device_found);
    cudaFree(device_output);
    cudaFree(device_values);
    cudaFree(device_keys);
    return static_cast<int>(status);
}

extern "C" int cmz_cuda_candidate_target_attention(
    uint32_t token,
    uint32_t from_square,
    uint64_t friendly_mask,
    uint64_t enemy_mask,
    uint64_t occupancy_mask,
    uint32_t ep_square,
    uint64_t* host_output) {
    if (host_output == nullptr || from_square >= 64U || token == 0U || token > 12U) {
        return cudaErrorInvalidValue;
    }
    if ((friendly_mask & enemy_mask) != 0) {
        return cudaErrorInvalidValue;
    }

    uint64_t* device_output = nullptr;
    cudaError_t status = cudaMalloc(&device_output, sizeof(uint64_t));
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }
    status = cudaMemset(device_output, 0, sizeof(uint64_t));
    if (status == cudaSuccess) {
        cmz_candidate_target_attention_kernel<<<1, 1>>>(
            token, from_square, friendly_mask, enemy_mask, occupancy_mask, ep_square, device_output);
        status = cudaGetLastError();
    }
    if (status == cudaSuccess) {
        status = cudaMemcpy(host_output, device_output, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    }
    cudaFree(device_output);
    return static_cast<int>(status);
}

extern "C" int cmz_cuda_candidate_moves_attention(
    const uint32_t* host_board_tokens,
    uint32_t white_to_move,
    uint32_t castling_rights,
    uint32_t ep_square,
    uint32_t* host_move_records,
    size_t record_capacity,
    uint32_t* host_record_count,
    uint32_t* host_layer_count) {
    if (host_board_tokens == nullptr || host_move_records == nullptr || host_record_count == nullptr ||
        host_layer_count == nullptr || white_to_move > 1U || castling_rights > 15U || ep_square > 64U ||
        record_capacity == 0U) {
        return cudaErrorInvalidValue;
    }

    uint32_t* device_board = nullptr;
    uint64_t* device_context = nullptr;
    uint32_t* device_active_tokens = nullptr;
    uint64_t* device_target_masks = nullptr;
    uint32_t* device_promotion_flags = nullptr;
    uint32_t* device_records = nullptr;
    uint32_t* device_slot_records = nullptr;
    uint32_t* device_slot_valid = nullptr;
    uint32_t* device_slot_ranks = nullptr;
    uint32_t* device_count = nullptr;
    void* device_scan_temp = nullptr;
    size_t scan_temp_bytes = 0U;
    const size_t board_bytes = 64U * sizeof(uint32_t);
    const size_t context_bytes = 3U * sizeof(uint64_t);
    const size_t token_bytes = 64U * sizeof(uint32_t);
    const size_t mask_bytes = 64U * sizeof(uint64_t);
    const size_t record_bytes = record_capacity * kCmzCandidateRecordWidth * sizeof(uint32_t);
    const size_t slot_record_bytes = kCmzCandidateRecordSlotCount * kCmzCandidateRecordWidth * sizeof(uint32_t);
    const size_t slot_valid_bytes = kCmzCandidateRecordSlotCount * sizeof(uint32_t);
    const size_t slot_rank_bytes = kCmzCandidateRecordSlotCount * sizeof(uint32_t);
    cudaError_t status = cudaMalloc(&device_board, board_bytes);
    if (status == cudaSuccess) {
        status = cudaMalloc(&device_context, context_bytes);
    }
    if (status == cudaSuccess) {
        status = cudaMalloc(&device_active_tokens, token_bytes);
    }
    if (status == cudaSuccess) {
        status = cudaMalloc(&device_target_masks, mask_bytes);
    }
    if (status == cudaSuccess) {
        status = cudaMalloc(&device_promotion_flags, token_bytes);
    }
    if (status == cudaSuccess) {
        status = cudaMalloc(&device_records, record_bytes);
    }
    if (status == cudaSuccess) {
        status = cudaMalloc(&device_slot_records, slot_record_bytes);
    }
    if (status == cudaSuccess) {
        status = cudaMalloc(&device_slot_valid, slot_valid_bytes);
    }
    if (status == cudaSuccess) {
        status = cudaMalloc(&device_slot_ranks, slot_rank_bytes);
    }
    if (status == cudaSuccess) {
        status = cudaMalloc(&device_count, sizeof(uint32_t));
    }
    if (status == cudaSuccess) {
        status = cudaMemcpy(device_board, host_board_tokens, board_bytes, cudaMemcpyHostToDevice);
    }
    if (status == cudaSuccess) {
        status = cudaMemset(device_context, 0, context_bytes);
    }
    if (status == cudaSuccess) {
        status = cudaMemset(device_active_tokens, 0, token_bytes);
    }
    if (status == cudaSuccess) {
        status = cudaMemset(device_target_masks, 0, mask_bytes);
    }
    if (status == cudaSuccess) {
        status = cudaMemset(device_promotion_flags, 0, token_bytes);
    }
    if (status == cudaSuccess) {
        status = cudaMemset(device_records, 0, record_bytes);
    }
    if (status == cudaSuccess) {
        status = cudaMemset(device_slot_records, 0, slot_record_bytes);
    }
    if (status == cudaSuccess) {
        status = cudaMemset(device_slot_valid, 0, slot_valid_bytes);
    }
    if (status == cudaSuccess) {
        status = cudaMemset(device_slot_ranks, 0, slot_rank_bytes);
    }
    if (status == cudaSuccess) {
        status = cudaMemset(device_count, 0, sizeof(uint32_t));
    }
    if (status == cudaSuccess) {
        cmz_candidate_context_select_attention_kernel<<<1, 1>>>(device_board, white_to_move, device_context);
        status = cudaGetLastError();
    }
    if (status == cudaSuccess) {
        cmz_candidate_piece_dispatch_attention_kernel<<<1, 64>>>(device_board, white_to_move, device_active_tokens);
        status = cudaGetLastError();
    }
    if (status == cudaSuccess) {
        cmz_candidate_target_mask_select_attention_kernel<<<1, 64>>>(
            device_active_tokens, device_context, ep_square, device_target_masks);
        status = cudaGetLastError();
    }
    if (status == cudaSuccess) {
        cmz_candidate_castle_merge_attention_kernel<<<1, 1>>>(
            device_board, white_to_move, castling_rights, device_target_masks);
        status = cudaGetLastError();
    }
    if (status == cudaSuccess) {
        cmz_candidate_promotion_expand_attention_kernel<<<1, 64>>>(device_active_tokens, device_promotion_flags);
        status = cudaGetLastError();
    }
    if (status == cudaSuccess) {
        const uint32_t threads = 256U;
        const uint32_t blocks = (kCmzCandidateRecordSlotCount + threads - 1U) / threads;
        cmz_candidate_record_slot_validity_qk_write_kernel<<<blocks, threads>>>(
            device_board,
            device_active_tokens,
            device_target_masks,
            device_promotion_flags,
            ep_square,
            device_slot_records,
            device_slot_valid);
        status = cudaGetLastError();
    }
    if (status == cudaSuccess) {
        status = cub::DeviceScan::ExclusiveSum(
            nullptr,
            scan_temp_bytes,
            device_slot_valid,
            device_slot_ranks,
            kCmzCandidateRecordSlotCount);
    }
    if (status == cudaSuccess) {
        status = cudaMalloc(&device_scan_temp, scan_temp_bytes);
    }
    if (status == cudaSuccess) {
        status = cub::DeviceScan::ExclusiveSum(
            device_scan_temp,
            scan_temp_bytes,
            device_slot_valid,
            device_slot_ranks,
            kCmzCandidateRecordSlotCount);
    }
    if (status == cudaSuccess) {
        const uint32_t threads = 256U;
        const uint32_t blocks = (kCmzCandidateRecordSlotCount + threads - 1U) / threads;
        cmz_candidate_record_slot_rank_write_attention_kernel<<<blocks, threads>>>(
            device_slot_records,
            device_slot_valid,
            device_slot_ranks,
            device_records,
            record_capacity,
            device_count);
        status = cudaGetLastError();
    }
    if (status == cudaSuccess) {
        cmz_candidate_record_order_select_attention_kernel<<<1, 1>>>(device_count, record_capacity);
        status = cudaGetLastError();
    }
    if (status == cudaSuccess) {
        status = cudaMemcpy(host_record_count, device_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    }
    if (status == cudaSuccess && *host_record_count > record_capacity) {
        status = cudaErrorInvalidValue;
    }
    if (status == cudaSuccess) {
        status = cudaMemcpy(host_move_records, device_records, record_bytes, cudaMemcpyDeviceToHost);
    }
    if (status == cudaSuccess) {
        *host_layer_count = 9U;
    }

    cudaFree(device_scan_temp);
    cudaFree(device_count);
    cudaFree(device_slot_ranks);
    cudaFree(device_slot_valid);
    cudaFree(device_slot_records);
    cudaFree(device_records);
    cudaFree(device_promotion_flags);
    cudaFree(device_target_masks);
    cudaFree(device_active_tokens);
    cudaFree(device_context);
    cudaFree(device_board);
    return static_cast<int>(status);
}

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

extern "C" int cmz_cuda_terminal_status_attention(
    const uint32_t* host_board_tokens,
    uint32_t white_to_move,
    uint32_t castling_rights,
    uint32_t ep_square,
    uint32_t halfmove_clock,
    uint32_t repetition_count,
    uint32_t adjudication_cap_reached,
    uint32_t* host_result_reason,
    uint32_t* host_layer_count) {
    if (host_board_tokens == nullptr || host_result_reason == nullptr || host_layer_count == nullptr ||
        white_to_move > 1U || castling_rights > 15U || ep_square > 64U) {
        return cudaErrorInvalidValue;
    }

    uint32_t* device_board = nullptr;
    uint32_t* device_result_reason = nullptr;
    uint32_t* device_state = nullptr;
    uint32_t* device_legal_outputs = nullptr;
    uint64_t* device_material_masks = nullptr;
    const size_t board_bytes = 64U * sizeof(uint32_t);
    const size_t result_bytes = 2U * sizeof(uint32_t);
    const size_t state_bytes = 4U * sizeof(uint32_t);
    const size_t material_mask_bytes = kCmzTerminalMaterialMaskCount * sizeof(uint64_t);
    uint32_t candidate_layer_count = 0U;
    uint32_t legal_filter_layer_count = 0U;
    uint32_t host_record_count = 0U;
    std::vector<uint32_t> move_records(kCmzCandidateRecordSlotCount * kCmzCandidateRecordWidth, 0U);
    std::vector<uint32_t> from_squares;
    std::vector<uint32_t> to_squares;
    std::vector<uint32_t> promotion_tokens;
    std::vector<uint32_t> en_passant_flags;
    std::vector<uint32_t> castle_flags;
    std::vector<uint32_t> legal_outputs;
    cudaError_t status = cudaMalloc(&device_board, board_bytes);
    if (status == cudaSuccess) {
        status = cudaMalloc(&device_result_reason, result_bytes);
    }
    if (status == cudaSuccess) {
        status = cudaMalloc(&device_state, state_bytes);
    }
    if (status == cudaSuccess) {
        status = cudaMalloc(&device_material_masks, material_mask_bytes);
    }
    if (status == cudaSuccess) {
        status = cudaMemcpy(device_board, host_board_tokens, board_bytes, cudaMemcpyHostToDevice);
    }
    if (status == cudaSuccess) {
        status = cudaMemset(device_result_reason, 0, result_bytes);
    }
    if (status == cudaSuccess) {
        status = cudaMemset(device_state, 0, state_bytes);
    }
    if (status == cudaSuccess) {
        status = cudaMemset(device_material_masks, 0, material_mask_bytes);
    }
    if (status == cudaSuccess) {
        cmz_terminal_draw_rule_select_attention_kernel<<<1, 1>>>(
            halfmove_clock, repetition_count, adjudication_cap_reached, device_state);
        status = cudaGetLastError();
    }
    if (status == cudaSuccess) {
        status = static_cast<cudaError_t>(cmz_cuda_candidate_moves_attention(
            host_board_tokens,
            white_to_move,
            castling_rights,
            ep_square,
            move_records.data(),
            kCmzCandidateRecordSlotCount,
            &host_record_count,
            &candidate_layer_count));
    }
    if (status == cudaSuccess) {
        from_squares.resize(host_record_count);
        to_squares.resize(host_record_count);
        promotion_tokens.resize(host_record_count);
        en_passant_flags.resize(host_record_count);
        castle_flags.resize(host_record_count);
        legal_outputs.resize(host_record_count);
        for (uint32_t index = 0U; index < host_record_count; ++index) {
            const uint32_t base = index * kCmzCandidateRecordWidth;
            from_squares[index] = move_records[base];
            to_squares[index] = move_records[base + 1U];
            promotion_tokens[index] =
                cmz_promotion_token_from_id(white_to_move != 0U, move_records[base + 2U]);
            en_passant_flags[index] = move_records[base + 3U];
            castle_flags[index] = move_records[base + 4U];
        }
    }
    if (status == cudaSuccess) {
        status = static_cast<cudaError_t>(cmz_cuda_legal_filter_v2_batch_attention(
            host_board_tokens,
            white_to_move,
            from_squares.data(),
            to_squares.data(),
            promotion_tokens.data(),
            en_passant_flags.data(),
            castle_flags.data(),
            host_record_count,
            legal_outputs.data(),
            &legal_filter_layer_count));
    }
    if (status == cudaSuccess && host_record_count > 0U) {
        const size_t legal_bytes = host_record_count * sizeof(uint32_t);
        status = cudaMalloc(&device_legal_outputs, legal_bytes);
        if (status == cudaSuccess) {
            status = cudaMemcpy(device_legal_outputs, legal_outputs.data(), legal_bytes, cudaMemcpyHostToDevice);
        }
    }
    if (status == cudaSuccess) {
        cmz_terminal_legal_presence_from_batch_attention_kernel<<<1, 1>>>(
            device_legal_outputs, host_record_count, device_state);
        status = cudaGetLastError();
    }
    if (status == cudaSuccess) {
        cmz_terminal_check_state_select_attention_kernel<<<1, 1>>>(device_board, white_to_move, device_state);
        status = cudaGetLastError();
    }
    if (status == cudaSuccess) {
        status = cudaMemset(device_material_masks, 0, material_mask_bytes);
    }
    if (status == cudaSuccess) {
        cmz_terminal_material_square_class_attention_kernel<<<1, 64>>>(device_board, device_material_masks);
        status = cudaGetLastError();
    }
    if (status == cudaSuccess) {
        cmz_terminal_material_select_attention_kernel<<<1, 1>>>(device_material_masks, device_state);
        status = cudaGetLastError();
    }
    if (status == cudaSuccess) {
        cmz_terminal_final_status_select_attention_kernel<<<1, 1>>>(device_state, device_result_reason);
        status = cudaGetLastError();
    }
    if (status == cudaSuccess) {
        status = cudaMemcpy(host_result_reason, device_result_reason, result_bytes, cudaMemcpyDeviceToHost);
    }
    if (status == cudaSuccess) {
        *host_layer_count = 6U + candidate_layer_count + legal_filter_layer_count;
    }

    cudaFree(device_material_masks);
    cudaFree(device_legal_outputs);
    cudaFree(device_state);
    cudaFree(device_result_reason);
    cudaFree(device_board);
    return static_cast<int>(status);
}

extern "C" int cmz_cuda_resolve_move_attention(
    const uint32_t* host_move_records,
    const uint32_t* host_legal_bits,
    size_t record_count,
    uint32_t requested_from,
    uint32_t requested_to,
    uint32_t requested_promotion,
    uint32_t* host_selected_record,
    uint32_t* host_found) {
    if ((host_move_records == nullptr && record_count > 0U) || (host_legal_bits == nullptr && record_count > 0U) ||
        host_selected_record == nullptr || host_found == nullptr || record_count > 1024U || requested_from >= 64U || requested_to >= 64U ||
        requested_promotion > 4U) {
        return cudaErrorInvalidValue;
    }

    uint32_t* device_records = nullptr;
    uint32_t* device_legal_bits = nullptr;
    uint32_t* device_selected = nullptr;
    uint32_t* device_found = nullptr;
    const size_t record_bytes = record_count * 5U * sizeof(uint32_t);
    const size_t legal_bytes = record_count * sizeof(uint32_t);
    cudaError_t status = cudaMalloc(&device_selected, 5U * sizeof(uint32_t));
    if (status == cudaSuccess) {
        status = cudaMalloc(&device_found, sizeof(uint32_t));
    }
    if (status == cudaSuccess && record_count > 0U) {
        status = cudaMalloc(&device_records, record_bytes);
    }
    if (status == cudaSuccess && record_count > 0U) {
        status = cudaMalloc(&device_legal_bits, legal_bytes);
    }
    if (status == cudaSuccess) {
        status = cudaMemset(device_selected, 0, 5U * sizeof(uint32_t));
    }
    if (status == cudaSuccess) {
        status = cudaMemset(device_found, 0, sizeof(uint32_t));
    }
    if (status == cudaSuccess && record_count > 0U) {
        status = cudaMemcpy(device_records, host_move_records, record_bytes, cudaMemcpyHostToDevice);
    }
    if (status == cudaSuccess && record_count > 0U) {
        status = cudaMemcpy(device_legal_bits, host_legal_bits, legal_bytes, cudaMemcpyHostToDevice);
    }
    if (status == cudaSuccess) {
        cmz_resolve_move_qk_hardmax_legal_set_attention_kernel<<<1, 1024>>>(
            device_records,
            device_legal_bits,
            record_count,
            requested_from,
            requested_to,
            requested_promotion,
            device_selected,
            device_found);
        status = cudaGetLastError();
    }
    if (status == cudaSuccess) {
        status = cudaMemcpy(host_selected_record, device_selected, 5U * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    }
    if (status == cudaSuccess) {
        status = cudaMemcpy(host_found, device_found, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    }

    cudaFree(device_found);
    cudaFree(device_selected);
    cudaFree(device_legal_bits);
    cudaFree(device_records);
    return static_cast<int>(status);
}

extern "C" int cmz_cuda_ray_scan_attention(
    uint32_t from_square,
    int32_t delta_file,
    int32_t delta_rank,
    uint64_t occupancy_mask,
    uint64_t* host_output) {
    if (host_output == nullptr || from_square >= 64U) {
        return cudaErrorInvalidValue;
    }
    if (delta_file == 0 && delta_rank == 0) {
        return cudaErrorInvalidValue;
    }
    if (delta_file < -1 || delta_file > 1 || delta_rank < -1 || delta_rank > 1) {
        return cudaErrorInvalidValue;
    }

    uint64_t* device_output = nullptr;
    cudaError_t status = cudaMalloc(&device_output, sizeof(uint64_t));
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }
    status = cudaMemset(device_output, 0, sizeof(uint64_t));
    if (status == cudaSuccess) {
        cmz_ray_scan_attention_kernel<<<1, 1>>>(from_square, delta_file, delta_rank, occupancy_mask, device_output);
        status = cudaGetLastError();
    }
    if (status == cudaSuccess) {
        status = cudaMemcpy(host_output, device_output, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    }
    cudaFree(device_output);
    return static_cast<int>(status);
}

extern "C" int cmz_cuda_legal_filter_v2_attention(
    const uint32_t* host_board_tokens,
    uint32_t white_to_move,
    uint32_t from_square,
    uint32_t to_square,
    uint32_t promotion_token,
    uint32_t en_passant,
    uint32_t castle,
    uint32_t* host_legal,
    uint32_t* host_layer_count) {
    if (host_board_tokens == nullptr || host_legal == nullptr || host_layer_count == nullptr || from_square >= 64U ||
        to_square >= 64U) {
        return cudaErrorInvalidValue;
    }
    if (promotion_token > 12U) {
        return cudaErrorInvalidValue;
    }
    if (host_board_tokens[from_square] == 0U) {
        *host_legal = 0U;
        *host_layer_count = 0U;
        return 0;
    }

    uint32_t* device_board = nullptr;
    uint32_t* device_move_meta = nullptr;
    uint32_t* device_next = nullptr;
    uint32_t* device_king_square = nullptr;
    uint32_t* device_king_found = nullptr;
    uint32_t* device_short_attacked = nullptr;
    uint32_t* device_ray_attacked = nullptr;
    uint32_t* device_legal = nullptr;
    const size_t board_bytes = 64 * sizeof(uint32_t);
    uint32_t layer_count = 0U;

    cudaError_t status = cudaMalloc(&device_board, board_bytes);
    if (status == cudaSuccess) {
        status = cudaMalloc(&device_move_meta, 6U * sizeof(uint32_t));
    }
    if (status == cudaSuccess) {
        status = cudaMalloc(&device_next, board_bytes);
    }
    if (status == cudaSuccess) {
        status = cudaMalloc(&device_king_square, sizeof(uint32_t));
    }
    if (status == cudaSuccess) {
        status = cudaMalloc(&device_king_found, sizeof(uint32_t));
    }
    if (status == cudaSuccess) {
        status = cudaMalloc(&device_short_attacked, sizeof(uint32_t));
    }
    if (status == cudaSuccess) {
        status = cudaMalloc(&device_ray_attacked, sizeof(uint32_t));
    }
    if (status == cudaSuccess) {
        status = cudaMalloc(&device_legal, sizeof(uint32_t));
    }
    if (status == cudaSuccess) {
        status = cudaMemcpy(device_board, host_board_tokens, board_bytes, cudaMemcpyHostToDevice);
    }
    if (status == cudaSuccess) {
        cmz_legal_filter_v2_move_type_select_attention_kernel<<<1, 1>>>(
            device_board,
            white_to_move,
            from_square,
            to_square,
            promotion_token,
            en_passant,
            castle,
            device_move_meta);
        status = cudaGetLastError();
        ++layer_count;
    }
    if (status == cudaSuccess) {
        cmz_legal_filter_v2_board_write_select_attention_kernel<<<1, 64>>>(
            device_board,
            from_square,
            to_square,
            device_move_meta,
            device_next);
        status = cudaGetLastError();
        ++layer_count;
    }
    if (status == cudaSuccess) {
        cmz_legal_filter_v2_en_passant_capture_select_attention_kernel<<<1, 64>>>(
            device_next,
            device_move_meta);
        status = cudaGetLastError();
        ++layer_count;
    }
    if (status == cudaSuccess) {
        cmz_legal_filter_v2_castle_rook_write_select_attention_kernel<<<1, 64>>>(
            device_next,
            device_move_meta);
        status = cudaGetLastError();
        ++layer_count;
    }
    if (status == cudaSuccess) {
        cmz_legal_filter_v2_promotion_select_attention_kernel<<<1, 64>>>(
            device_next,
            to_square,
            device_move_meta);
        status = cudaGetLastError();
        ++layer_count;
    }
    if (status == cudaSuccess) {
        cmz_legal_filter_v2_king_square_select_attention_kernel<<<1, 1>>>(
            device_next,
            white_to_move,
            device_king_square,
            device_king_found);
        status = cudaGetLastError();
        ++layer_count;
    }
    if (status == cudaSuccess) {
        cmz_legal_filter_v2_attack_source_select_attention_kernel<<<1, 1>>>(
            device_next,
            device_king_square,
            white_to_move == 0U ? 1U : 0U,
            device_short_attacked);
        status = cudaGetLastError();
        ++layer_count;
    }
    if (status == cudaSuccess) {
        status = cudaMemset(device_ray_attacked, 0, sizeof(uint32_t));
    }
    if (status == cudaSuccess) {
        cmz_legal_filter_v2_ray_blocker_select_attention_kernel<<<1, 8>>>(
            device_next,
            device_king_square,
            white_to_move == 0U ? 1U : 0U,
            device_ray_attacked);
        status = cudaGetLastError();
        ++layer_count;
    }
    if (status == cudaSuccess) {
        cmz_legal_filter_v2_final_legal_select_attention_kernel<<<1, 1>>>(
            device_king_found,
            device_short_attacked,
            device_ray_attacked,
            device_legal);
        status = cudaGetLastError();
        ++layer_count;
    }
    if (status == cudaSuccess) {
        status = cudaMemcpy(host_legal, device_legal, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    }
    if (status == cudaSuccess) {
        *host_layer_count = layer_count;
    }

    cudaFree(device_legal);
    cudaFree(device_ray_attacked);
    cudaFree(device_short_attacked);
    cudaFree(device_king_found);
    cudaFree(device_king_square);
    cudaFree(device_next);
    cudaFree(device_move_meta);
    cudaFree(device_board);
    return static_cast<int>(status);
}

extern "C" int cmz_cuda_make_move_board_attention(
    const uint32_t* host_board_tokens,
    uint32_t white_to_move,
    uint32_t from_square,
    uint32_t to_square,
    uint32_t promotion_token,
    uint32_t en_passant,
    uint32_t castle,
    uint32_t* host_next_board_tokens,
    uint32_t* host_layer_count) {
    if (host_board_tokens == nullptr || host_next_board_tokens == nullptr || host_layer_count == nullptr ||
        from_square >= 64U || to_square >= 64U || promotion_token > 12U || host_board_tokens[from_square] == 0U) {
        return cudaErrorInvalidValue;
    }

    uint32_t* device_board = nullptr;
    uint32_t* device_move_meta = nullptr;
    uint32_t* device_next = nullptr;
    const size_t board_bytes = 64U * sizeof(uint32_t);
    uint32_t layer_count = 0U;

    cudaError_t status = cudaMalloc(&device_board, board_bytes);
    if (status == cudaSuccess) {
        status = cudaMalloc(&device_move_meta, 6U * sizeof(uint32_t));
    }
    if (status == cudaSuccess) {
        status = cudaMalloc(&device_next, board_bytes);
    }
    if (status == cudaSuccess) {
        status = cudaMemcpy(device_board, host_board_tokens, board_bytes, cudaMemcpyHostToDevice);
    }
    if (status == cudaSuccess) {
        cmz_legal_filter_v2_move_type_select_attention_kernel<<<1, 1>>>(
            device_board,
            white_to_move,
            from_square,
            to_square,
            promotion_token,
            en_passant,
            castle,
            device_move_meta);
        status = cudaGetLastError();
        ++layer_count;
    }
    if (status == cudaSuccess) {
        cmz_legal_filter_v2_board_write_select_attention_kernel<<<1, 64>>>(
            device_board,
            from_square,
            to_square,
            device_move_meta,
            device_next);
        status = cudaGetLastError();
        ++layer_count;
    }
    if (status == cudaSuccess) {
        cmz_legal_filter_v2_en_passant_capture_select_attention_kernel<<<1, 64>>>(
            device_next,
            device_move_meta);
        status = cudaGetLastError();
        ++layer_count;
    }
    if (status == cudaSuccess) {
        cmz_legal_filter_v2_castle_rook_write_select_attention_kernel<<<1, 64>>>(
            device_next,
            device_move_meta);
        status = cudaGetLastError();
        ++layer_count;
    }
    if (status == cudaSuccess) {
        cmz_legal_filter_v2_promotion_select_attention_kernel<<<1, 64>>>(
            device_next,
            to_square,
            device_move_meta);
        status = cudaGetLastError();
        ++layer_count;
    }
    if (status == cudaSuccess) {
        status = cudaMemcpy(host_next_board_tokens, device_next, board_bytes, cudaMemcpyDeviceToHost);
    }
    if (status == cudaSuccess) {
        *host_layer_count = layer_count;
    }

    cudaFree(device_next);
    cudaFree(device_move_meta);
    cudaFree(device_board);
    return static_cast<int>(status);
}

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
    uint32_t* host_layer_count) {
    if (host_next_metadata == nullptr || host_layer_count == nullptr || white_to_move > 1U || castling_rights > 15U ||
        moving_piece_token == 0U || moving_piece_token > 12U || target_piece_token > 12U || from_square >= 64U ||
        to_square >= 64U) {
        return cudaErrorInvalidValue;
    }

    uint32_t* device_metadata = nullptr;
    uint32_t layer_count = 0U;
    cudaError_t status = cudaMalloc(&device_metadata, 5U * sizeof(uint32_t));
    if (status == cudaSuccess) {
        status = cudaMemset(device_metadata, 0, 5U * sizeof(uint32_t));
    }
    if (status == cudaSuccess) {
        cmz_make_move_metadata_side_toggle_select_attention_kernel<<<1, 1>>>(white_to_move, device_metadata);
        status = cudaGetLastError();
        ++layer_count;
    }
    if (status == cudaSuccess) {
        cmz_make_move_metadata_castling_rights_select_attention_kernel<<<1, 1>>>(
            castling_rights,
            moving_piece_token,
            from_square,
            to_square,
            device_metadata);
        status = cudaGetLastError();
        ++layer_count;
    }
    if (status == cudaSuccess) {
        cmz_make_move_metadata_ep_square_select_attention_kernel<<<1, 1>>>(
            moving_piece_token,
            from_square,
            to_square,
            device_metadata);
        status = cudaGetLastError();
        ++layer_count;
    }
    if (status == cudaSuccess) {
        cmz_make_move_metadata_halfmove_clock_select_attention_kernel<<<1, 1>>>(
            halfmove_clock,
            moving_piece_token,
            target_piece_token,
            en_passant,
            device_metadata);
        status = cudaGetLastError();
        ++layer_count;
    }
    if (status == cudaSuccess) {
        cmz_make_move_metadata_fullmove_number_select_attention_kernel<<<1, 1>>>(
            white_to_move,
            fullmove_number,
            device_metadata);
        status = cudaGetLastError();
        ++layer_count;
    }
    if (status == cudaSuccess) {
        status = cudaMemcpy(host_next_metadata, device_metadata, 5U * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    }
    if (status == cudaSuccess) {
        *host_layer_count = layer_count;
    }

    cudaFree(device_metadata);
    return static_cast<int>(status);
}

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
    uint32_t* host_layer_count) {
    if (move_count == 0) {
        if (host_layer_count != nullptr) {
            *host_layer_count = 0U;
        }
        return 0;
    }
    if (host_board_tokens == nullptr || host_from_squares == nullptr || host_to_squares == nullptr ||
        host_promotion_tokens == nullptr || host_en_passant == nullptr || host_castle == nullptr ||
        host_legal_outputs == nullptr || host_layer_count == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (move_count > std::numeric_limits<size_t>::max() / (64U * sizeof(uint32_t))) {
        return cudaErrorInvalidValue;
    }
    for (size_t index = 0; index < move_count; ++index) {
        if (host_from_squares[index] >= 64U || host_to_squares[index] >= 64U || host_promotion_tokens[index] > 12U) {
            return cudaErrorInvalidValue;
        }
    }

    uint32_t* device_board = nullptr;
    uint32_t* device_from = nullptr;
    uint32_t* device_to = nullptr;
    uint32_t* device_promotion = nullptr;
    uint32_t* device_en_passant = nullptr;
    uint32_t* device_castle = nullptr;
    uint32_t* device_move_meta = nullptr;
    uint32_t* device_next_boards = nullptr;
    uint32_t* device_king_squares = nullptr;
    uint32_t* device_king_found = nullptr;
    uint32_t* device_short_attacked = nullptr;
    uint32_t* device_ray_attacked = nullptr;
    uint32_t* device_output = nullptr;
    const size_t board_bytes = 64U * sizeof(uint32_t);
    const size_t move_bytes = move_count * sizeof(uint32_t);
    const size_t move_meta_bytes = move_count * 6U * sizeof(uint32_t);
    const size_t next_board_bytes = move_count * 64U * sizeof(uint32_t);
    uint32_t layer_count = 0U;

    cudaError_t status = cudaMalloc(&device_board, board_bytes);
    if (status == cudaSuccess) {
        status = cudaMalloc(&device_from, move_bytes);
    }
    if (status == cudaSuccess) {
        status = cudaMalloc(&device_to, move_bytes);
    }
    if (status == cudaSuccess) {
        status = cudaMalloc(&device_promotion, move_bytes);
    }
    if (status == cudaSuccess) {
        status = cudaMalloc(&device_en_passant, move_bytes);
    }
    if (status == cudaSuccess) {
        status = cudaMalloc(&device_castle, move_bytes);
    }
    if (status == cudaSuccess) {
        status = cudaMalloc(&device_move_meta, move_meta_bytes);
    }
    if (status == cudaSuccess) {
        status = cudaMalloc(&device_next_boards, next_board_bytes);
    }
    if (status == cudaSuccess) {
        status = cudaMalloc(&device_king_squares, move_bytes);
    }
    if (status == cudaSuccess) {
        status = cudaMalloc(&device_king_found, move_bytes);
    }
    if (status == cudaSuccess) {
        status = cudaMalloc(&device_short_attacked, move_bytes);
    }
    if (status == cudaSuccess) {
        status = cudaMalloc(&device_ray_attacked, move_bytes);
    }
    if (status == cudaSuccess) {
        status = cudaMalloc(&device_output, move_bytes);
    }

    if (status == cudaSuccess) {
        status = cudaMemcpy(device_board, host_board_tokens, board_bytes, cudaMemcpyHostToDevice);
    }
    if (status == cudaSuccess) {
        status = cudaMemcpy(device_from, host_from_squares, move_bytes, cudaMemcpyHostToDevice);
    }
    if (status == cudaSuccess) {
        status = cudaMemcpy(device_to, host_to_squares, move_bytes, cudaMemcpyHostToDevice);
    }
    if (status == cudaSuccess) {
        status = cudaMemcpy(device_promotion, host_promotion_tokens, move_bytes, cudaMemcpyHostToDevice);
    }
    if (status == cudaSuccess) {
        status = cudaMemcpy(device_en_passant, host_en_passant, move_bytes, cudaMemcpyHostToDevice);
    }
    if (status == cudaSuccess) {
        status = cudaMemcpy(device_castle, host_castle, move_bytes, cudaMemcpyHostToDevice);
    }

    const int block = 128;
    if (status == cudaSuccess) {
        const int grid = static_cast<int>((move_count + block - 1) / block);
        cmz_legal_filter_v2_batch_move_type_select_attention_kernel<<<grid, block>>>(
            device_board,
            white_to_move,
            device_from,
            device_to,
            device_promotion,
            device_en_passant,
            device_castle,
            device_move_meta,
            move_count);
        status = cudaGetLastError();
        ++layer_count;
    }
    if (status == cudaSuccess) {
        const size_t total_square_threads = move_count * 64U;
        const int grid = static_cast<int>((total_square_threads + block - 1) / block);
        cmz_legal_filter_v2_batch_board_write_select_attention_kernel<<<grid, block>>>(
            device_board,
            device_from,
            device_to,
            device_move_meta,
            device_next_boards,
            move_count);
        status = cudaGetLastError();
        ++layer_count;
    }
    if (status == cudaSuccess) {
        const size_t total_square_threads = move_count * 64U;
        const int grid = static_cast<int>((total_square_threads + block - 1) / block);
        cmz_legal_filter_v2_batch_en_passant_capture_select_attention_kernel<<<grid, block>>>(
            device_next_boards,
            device_move_meta,
            move_count);
        status = cudaGetLastError();
        ++layer_count;
    }
    if (status == cudaSuccess) {
        const size_t total_square_threads = move_count * 64U;
        const int grid = static_cast<int>((total_square_threads + block - 1) / block);
        cmz_legal_filter_v2_batch_castle_rook_write_select_attention_kernel<<<grid, block>>>(
            device_next_boards,
            device_move_meta,
            move_count);
        status = cudaGetLastError();
        ++layer_count;
    }
    if (status == cudaSuccess) {
        const size_t total_square_threads = move_count * 64U;
        const int grid = static_cast<int>((total_square_threads + block - 1) / block);
        cmz_legal_filter_v2_batch_promotion_select_attention_kernel<<<grid, block>>>(
            device_next_boards,
            device_to,
            device_move_meta,
            move_count);
        status = cudaGetLastError();
        ++layer_count;
    }
    if (status == cudaSuccess) {
        const int grid = static_cast<int>((move_count + block - 1) / block);
        cmz_legal_filter_v2_batch_king_square_select_attention_kernel<<<grid, block>>>(
            device_next_boards,
            white_to_move,
            device_king_squares,
            device_king_found,
            move_count);
        status = cudaGetLastError();
        ++layer_count;
    }
    if (status == cudaSuccess) {
        const int grid = static_cast<int>((move_count + block - 1) / block);
        cmz_legal_filter_v2_batch_attack_source_select_attention_kernel<<<grid, block>>>(
            device_next_boards,
            device_king_squares,
            white_to_move == 0U ? 1U : 0U,
            device_short_attacked,
            move_count);
        status = cudaGetLastError();
        ++layer_count;
    }
    if (status == cudaSuccess) {
        status = cudaMemset(device_ray_attacked, 0, move_bytes);
    }
    if (status == cudaSuccess) {
        const size_t total_ray_threads = move_count * 8U;
        const int grid = static_cast<int>((total_ray_threads + block - 1) / block);
        cmz_legal_filter_v2_batch_ray_blocker_select_attention_kernel<<<grid, block>>>(
            device_next_boards,
            device_king_squares,
            white_to_move == 0U ? 1U : 0U,
            device_ray_attacked,
            move_count);
        status = cudaGetLastError();
        ++layer_count;
    }
    if (status == cudaSuccess) {
        const int grid = static_cast<int>((move_count + block - 1) / block);
        cmz_legal_filter_v2_batch_final_legal_select_attention_kernel<<<grid, block>>>(
            device_king_found,
            device_short_attacked,
            device_ray_attacked,
            device_output,
            move_count);
        status = cudaGetLastError();
        ++layer_count;
    }
    if (status == cudaSuccess) {
        status = cudaMemcpy(host_legal_outputs, device_output, move_bytes, cudaMemcpyDeviceToHost);
    }
    if (status == cudaSuccess) {
        *host_layer_count = layer_count;
    }

    cudaFree(device_output);
    cudaFree(device_ray_attacked);
    cudaFree(device_short_attacked);
    cudaFree(device_king_found);
    cudaFree(device_king_squares);
    cudaFree(device_next_boards);
    cudaFree(device_move_meta);
    cudaFree(device_castle);
    cudaFree(device_en_passant);
    cudaFree(device_promotion);
    cudaFree(device_to);
    cudaFree(device_from);
    cudaFree(device_board);
    return static_cast<int>(status);
}

extern "C" int cmz_cuda_castle_target_attention(
    const uint32_t* host_board_tokens,
    uint32_t castling_rights,
    uint32_t white,
    uint64_t* host_output) {
    if (host_board_tokens == nullptr || host_output == nullptr) {
        return cudaErrorInvalidValue;
    }

    uint32_t* device_board = nullptr;
    uint64_t* device_output = nullptr;
    const size_t board_bytes = 64 * sizeof(uint32_t);
    cudaError_t status = cudaMalloc(&device_board, board_bytes);
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }
    status = cudaMalloc(&device_output, sizeof(uint64_t));
    if (status != cudaSuccess) {
        cudaFree(device_board);
        return static_cast<int>(status);
    }
    status = cudaMemcpy(device_board, host_board_tokens, board_bytes, cudaMemcpyHostToDevice);
    if (status == cudaSuccess) {
        status = cudaMemset(device_output, 0, sizeof(uint64_t));
    }
    if (status == cudaSuccess) {
        cmz_castle_target_attention_kernel<<<1, 1>>>(device_board, castling_rights, white, device_output);
        status = cudaGetLastError();
    }
    if (status == cudaSuccess) {
        status = cudaMemcpy(host_output, device_output, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    }
    cudaFree(device_output);
    cudaFree(device_board);
    return static_cast<int>(status);
}
