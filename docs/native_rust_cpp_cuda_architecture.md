# Native Rust + C++ + CUDA Architecture

## Scope

- milestone_id=native_rust_cpp_cuda_v1
- goal=remove Python from production runtime by replacing runtime layers with Rust orchestration plus C++/CUDA compute library
- strategy_policy=no hardcoded MCTS, no hardcoded beam search, no engine labels, no tablebase labels, no human-game data, no handcrafted chess evaluation
- search_policy=future model chooses what to inspect; native layer exposes deterministic chess-world operations and fast tensor/attention primitives

## Target Stack

```text
Rust
  CLI / config / orchestration / IO / logging / tests
        |
        | FFI: C ABI, opaque handles, coarse operations only
        v
C++ host layer
  owns CUDA context, streams, buffers, error handling, future CUTLASS/cuBLAS/NCCL integration
        |
        v
CUDA .cu kernels
  hot path: attention/select, permutation, hash, dedup, top-k, trace inference glue
```

## FFI Contract

- Rust calls coarse operations only.
- Rust must not launch individual kernels or manage low-level device buffers.
- C ABI uses opaque `CmzEngine*` handle.
- C++ owns CUDA runtime interactions.
- Error reporting uses explicit status code plus `cmz_engine_last_error`.

Current exported functions:

```c
int cmz_engine_create(CmzEngine** out);
void cmz_engine_destroy(CmzEngine* engine);
const char* cmz_engine_last_error(const CmzEngine* engine);
int cmz_engine_runtime_mode(const CmzEngine* engine, char* out, size_t out_len);
int cmz_engine_percepta_contract_json(const CmzEngine* engine, char* out, size_t out_len, size_t* written);
size_t cmz_engine_attention_decode_count(const CmzEngine* engine);
size_t cmz_engine_cuda_trace_select_count(const CmzEngine* engine);
size_t cmz_engine_cutlass_hardmax2d_count(const CmzEngine* engine);
size_t cmz_engine_cuda_board_projection_count(const CmzEngine* engine);
size_t cmz_engine_cuda_attack_table_attention_count(const CmzEngine* engine);
size_t cmz_engine_cuda_candidate_table_attention_count(const CmzEngine* engine);
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
int cmz_engine_cuda_available(const CmzEngine* engine);
int cmz_engine_cuda_device_name(const CmzEngine* engine, char* out, size_t out_len);
int cmz_engine_frozen_attack_mask(CmzEngine* engine, uint32_t piece_token, uint32_t from_square, uint64_t* out_mask);
int cmz_engine_frozen_ray_scan_mask(CmzEngine* engine, uint32_t from_square, int32_t delta_file, int32_t delta_rank, uint64_t occupancy_mask, uint64_t* out_mask);
int cmz_engine_frozen_candidate_target_mask(CmzEngine* engine, uint32_t piece_token, uint32_t from_square, uint64_t friendly_mask, uint64_t enemy_mask, uint64_t occupancy_mask, uint32_t ep_square, uint64_t* out_mask);
int cmz_engine_frozen_castle_target_mask(CmzEngine* engine, const char* fen, int32_t white, uint64_t* out_mask);
int cmz_engine_frozen_move_legal(CmzEngine* engine, const char* fen, const char* uci, int32_t* out_legal);
int cmz_engine_frozen_terminal_status(CmzEngine* engine, const char* fen, uint32_t repetition_count, int32_t adjudication_cap_reached, uint32_t* out_result, uint32_t* out_reason);
int cmz_engine_hull_hardmax_2d(CmzEngine* engine, const float* keys_xy, size_t key_count, float query_x, float query_y, uint32_t* out_index, float* out_score);
int cmz_engine_nested_hull_topk_2d(CmzEngine* engine, const float* keys_xy, size_t key_count, size_t k, float query_x, float query_y, uint32_t* out_indices);
int cmz_engine_legal_moves_uci(CmzEngine* engine, const char* fen, char* out, size_t out_len, size_t* written);
int cmz_engine_legal_trace_packets(CmzEngine* engine, const char* fen, uint32_t* out_tokens, size_t packet_capacity, size_t* packet_count);
int cmz_engine_frozen_legal_trace_attention_packets(CmzEngine* engine, const char* fen, uint32_t* out_tokens, size_t packet_capacity, size_t* packet_count);
int cmz_engine_legal_trace_begin(CmzEngine* engine, const char* fen, size_t* packet_count);
int cmz_engine_legal_trace_next(CmzEngine* engine, uint32_t* out_packet_tokens);
int cmz_engine_make_move_trace_packets(CmzEngine* engine, const char* fen, const char* uci, uint32_t ply, uint32_t repetition_count, int adjudication_cap_reached, uint32_t* out_tokens, size_t packet_capacity, size_t* packet_count);
int cmz_engine_frozen_make_move_trace_attention_packets(CmzEngine* engine, const char* fen, const char* uci, uint32_t ply, uint32_t repetition_count, int adjudication_cap_reached, uint32_t* out_tokens, size_t packet_capacity, size_t* packet_count);
int cmz_engine_project_board_trace(CmzEngine* engine, const uint32_t* trace_tokens, size_t packet_count, uint32_t* out_square_piece_tokens, size_t square_capacity, uint32_t* out_side_to_move);
int cmz_engine_cuda_probe_double(CmzEngine* engine, const uint32_t* input, size_t len, uint32_t* output);
```

## Current Implementation

- `docker/native/Dockerfile` extends `gpu-dev-cutlass-nsight:2026-05-24` with Rust stable, cargo, rustfmt, clippy, and pkg-config.
- `docker/native/run_native_container.ps1` builds `cmz-native-dev:2026-05-26` and starts persistent container `cmz-native-dev`.
- `docker/native/exec_native.ps1` runs commands inside `cmz-native-dev` and writes logs to `test_results/native_container_logs/`.
- `native/Cargo.toml` defines the Rust workspace.
- `native/cpp/CMakeLists.txt` requires CUTLASS headers at `/opt/cutlass/include`; missing CUTLASS is a hard build error because fallback is forbidden.
- `native/crates/cmz-engine-sys` provides safe Rust wrapper over the C ABI.
- `native/crates/cmz-engine-sys/src/packets.rs` implements native `MovePacket` and `TracePacket` codecs compatible with the Python prototype layout.
- `native/crates/cmz-cli` provides native CLI legal move enumeration and legal trace token output.
- `native/crates/cmz-dashboard` provides a Rust HTTP dashboard snapshot/server that consumes native trace streams, calls native policy-only decoder move selection, verifies selected moves against native legal traces, and performs no chess rule computation.
- `native/cpp` builds static `cmz_engine` through CMake/Ninja with C++17 and CUDA 17.
- `native/cpp/src/cmz_engine.cpp` implements FEN parsing, legal UCI move generation, legal trace emission, one-packet legal trace streaming, make-move trace emission, and terminal record emission in C++.
- Native trace stream currently uses a CUDA trace-select attention slice: packet memory is copied to device memory, a CUDA kernel selects the packet at the decode cursor, and failure returns an explicit CUDA error without CPU fallback. The path is exposed as `runtime_mode=native_cuda_trace_select_decoder`, `cmz_engine_attention_decode_count`, and `cmz_engine_cuda_trace_select_count`.
- `cmz_engine_percepta_contract_json` declares the current native Percepta contract: `executor_head_dim=2`, `rule_attention_backend=hull_hardmax_2d`, `hull_score_backend=cutlass_gemm_2d`, `hull_select_backend=cuda_hardmax_select`, `hull_host_argmax=false`, `topk_backend=nested_hull_topk_2d`, `long_context_cache=HullKVCache`, `trace_streaming=true`, `simple_kv_cache=false`, `python_hot_path=false`, `fallback_allowed=false`, `decoder_backend=libtorch_cuda_policy_only_v1`, `learning_method=self_play_policy_gradient`, `actor_critic=false`, `critic_head_enabled=false`, `value_head_enabled=false`, `externally_prescribed_critic=false`, and `soft_surrogate_available=false`.
- `native/cpp/src/cmz_cuda_kernels.cu` implements exact CUDA probe and 2D query/key dot-score kernels used by HullHardmax2D support lookup.
- Native HullKV path builds 2D convex hulls in C++ host code and uses CUTLASS GEMM scoring over hull vertices; dense hardmax remains a Rust test oracle, not the reported production contract.
- Native CUTLASS hardmax path uses `cmz_cutlass_hardmax2d_values` for the frozen QK score operation `query[1,2] * keys[2,N] -> scores[1,N]`, then uses `cmz_hardmax_float_select_kernel` for CUDA-side hardmax/select; host CPU argmax over score arrays is forbidden.
- Native CUDA trace-select path uses `cmz_cuda_select_trace_packet` for one-packet decode and increments `cmz_engine_cuda_trace_select_count`; CPU trace-select fallback is forbidden.
- Native CUDA board-projection path uses `cmz_cuda_project_board_latest_writes` for latest-write frozen attention over board trace packets and increments `cmz_engine_cuda_board_projection_count`; CPU board-projection fallback is forbidden.
- Native CUDA attack-table path uses `cmz_cuda_attack_table_lookup_attention` for frozen QK-hardmax-V table lookup over `(piece_token, square)` keys and attack-mask values; `cmz_engine_frozen_attack_mask` routes through this CUDA path and increments `cmz_engine_cuda_attack_table_attention_count`; CPU attack-table lookup fallback is forbidden.
- Native CUDA candidate-target path uses `cmz_cuda_candidate_target_attention` for target-mask frozen attention over `(piece_token, square, occupancy state)` and dynamic mask values; `cmz_engine_frozen_candidate_target_mask` routes through this CUDA path and increments `cmz_engine_cuda_candidate_table_attention_count`; CPU candidate-target fallback is forbidden.
- Native CUDA ray-scan path uses `cmz_cuda_ray_scan_attention` for nearest-blocker frozen attention over a ray; `cmz_engine_frozen_ray_scan_mask` routes through this CUDA path and increments `cmz_engine_cuda_ray_scan_attention_count`; CPU ray-scan fallback is forbidden.
- Native CUDA legal-filter v2 single-move path uses `cmz_cuda_legal_filter_v2_attention`; `cmz_engine_frozen_move_legal` routes through 9 explicit frozen self-attention layers and increments `cmz_engine_cuda_legal_filter_v2_attention_count` plus `cmz_engine_cuda_legal_filter_v2_layer_count`; internal priority/argmax score-select uses `cmz_qk2_score_u32` plus `cmz_qk2_hardmax_select_u32`; the old `cmz_cuda_legal_filter_attention` single kernel is no longer used by this path.
- Native CUDA legal-filter batch v2 path uses `cmz_cuda_legal_filter_v2_batch_attention`; frozen legal trace generation, legal move listing, and legal move resolution route through 9 explicit batched frozen self-attention layers and increment `cmz_engine_cuda_legal_filter_v2_batch_attention_count` plus `cmz_engine_cuda_legal_filter_v2_batch_layer_count`; internal priority/argmax score-select uses `cmz_qk2_score_u32` plus `cmz_qk2_hardmax_select_u32`; the old `cmz_cuda_legal_filter_batch_attention` route is no longer used by this path.
- Native CUDA castling path uses `cmz_cuda_castle_target_attention` for castle path emptiness and attack checks; `cmz_engine_frozen_castle_target_mask` routes through this CUDA path and increments `cmz_engine_cuda_castle_attention_count`; CPU castle-target fallback is forbidden.
- Native decoder scaffold exists in Rust as `PerceptaDecoderScaffold`; it enforces `head_dim=2`, shared white/black mode, generic command count, HullKV-compatible workspace slots, `value_head_enabled=false`, and `externally_prescribed_critic=false`.
- Native decoder v1 exists in C++/LibTorch/CUDA behind `cmz_engine_decoder_forward`, `cmz_engine_decoder_policy_gradient_step`, and `cmz_engine_policy_select_move`; forward computes command logits only from board-hidden projection through 2D attention over board square keys and piece/occupancy values; policy-gradient step updates decoder policy tensors only and keeps TracePacket detached; policy selection returns a trace-legal move record for the Rust dashboard and native orchestration. No value baseline, critic head, actor-critic module, handcrafted evaluation, MCTS, beam search, or prescribed search policy is present.
- `native/.cargo/config.toml` adds a Linux rpath for the LibTorch shared libraries inside the Docker image; `docker/native/exec_native.ps1` exports the same library path for test and command execution.
- `cmz_engine_frozen_rule_graph_json` declares the native frozen rule-layer lowering status. Current lowered layers are `board_projection=latest_write_hardmax_2d`, `board_projection_backend=cuda_latest_write_projection`, `trace_select=cursor_hardmax_2d`, `trace_select_backend=cuda_trace_select_packet`, `trace_select_long_context_cache=HullKVCache`, `trace_append_backend=cuda_trace_packet_emit_attention`, `trace_append_cpp_loop_remaining=false`, `hull_lookup_backend=cutlass_gemm_2d`, `hullkv_rule_hot_path=true`, `hull_hardmax_select_backend=cuda_hardmax_select`, `hull_hardmax_host_argmax=false`, `nested_hull_topk_backend=cutlass_qk_cuda_topk_select`, `nested_hull_topk_cpu=false`, `piece_dispatch=frozen_table_attention`, `attack_masks=static_attack_mask_table_attention`, `attack_masks_backend=cuda_qk_hardmax_v_table_lookup`, `table_lookup_semantics=qk_hardmax_v`, `ray_scan=blocker_aware_ray_scan_attention`, `ray_scan_backend=cuda_nearest_blocker_attention`, `ray_scan_semantics=qk_hardmax_v_nearest_blocker`, `candidate_targets=target_mask_attention`, `candidate_targets_backend=cuda_qk_hardmax_v_target_lookup`, `candidate_target_dispatch_backend=qk_hardmax_piece_family_select`, `candidate_offset_targets_backend=qk_explicit_offset_slot_writes`, `candidate_single_offset_coordinate_table_backend=qk_coordinate_table_slots`, `candidate_filter_backend=cuda_dynamic_mask_attention`, `pseudo_legal_moves_backend=cuda_candidate_moves_layered_attention`, `candidate_moves_layers=context_select,piece_dispatch,target_mask_select,castle_merge,promotion_expand,record_emit,prefix_rank_select,record_order_select`, `pseudo_legal_cpp_control_flow_remaining=false`, `resolve_move_backend=cuda_resolve_move_qk_hardmax_legal_set_attention`, `resolve_move_cpp_loop_remaining=false`, `resolve_move_scan=false`, `resolve_move_qk_hardmax_2d=true`, `castling_targets=castle_path_attention`, `castling_targets_backend=cuda_castle_path_attention`, `legal_filter=king_safety_attention`, `legal_filter_backend=cuda_legal_filter_v2_layered_self_attention`, `legal_filter_v2_current_backend=cuda_qk_hardmax_v_write_layers`, `legal_filter_v2_layers_complete=move_type_select,board_write_select,en_passant_capture_select,castle_rook_write_select,promotion_select,king_square_select,attack_source_select,ray_blocker_select,final_legal_select`, `legal_filter_v2_inner_select=qk_hardmax_2d_helpers`, `legal_filter_v2_inner_select_plain_cuda_loops_remaining=false`, `legal_filter_v1_single_kernel_remaining=false`, `legal_filter_batch_backend=cuda_legal_filter_v2_batched_layered_self_attention`, `legal_filter_batch_v1_kernel_remaining=false`, `legacy_legal_filter_cuda_symbols_present=false`, `small_launch_fusion=legal_filter_batch_v2`, `make_move=board_write_attention`, `make_move_backend=cuda_board_write_attention`, `make_move_board_squares_backend=cuda_make_move_board_attention`, `make_move_board_square_layers=move_type_select,board_write_select,en_passant_capture_select,castle_rook_write_select,promotion_select`, `make_move_board_metadata_backend=cuda_make_move_metadata_attention`, `terminal_predicates=terminal_status_attention`, `terminal_predicates_backend=cuda_terminal_status_layered_attention`, `terminal_status_layers=draw_rule_select,legal_presence_select,check_state_select,material_class_select,material_status_select,final_status_select`, `terminal_material_backend=qk_material_class_bitmask_attention`, `terminal_cpp_logic_remaining=false`, `move_record_expansion=move_record_attention`, `promotion_expansion=promotion_attention`, `trace_emission=trace_packet_attention`, and `make_move_trace_emission=trace_packet_attention`.
- Current authoritative contract is `target_full_frozen_attention_only=true`, `all_rules_must_lower_to_frozen_2d_self_attention=true`, `attention_only_rule_substrate=true`, `tensor_layer_substrate=false`, `current_full_frozen_2d_self_attention_only=false`, `full_frozen_attention_only=false`, `full_rule_lowering_complete=false`, `semantic_attention_purity=false`, `contract_overclaim_fixed=true`, `cpp_control_flow_rule_vm_remaining=false`, `trace_streaming_backend=incremental_packet_attention`, `trace_streaming_buffered=false`, `trace_streaming_full_trace_precompute=false`, `dashboard_policy_decoder=true`, `dashboard_policy_selection_backend=native_libtorch_policy_decoder`, `semantic_source_audit=rust_cuda_body_scan_v1`, `metadata_only_tests_remaining=false`, `candidate_pawn_targets_backend=qk_explicit_pawn_slot_writes`, `candidate_single_offset_backend=qk_bounds_slot_friendly_filter`, `candidate_single_offset_coordinate_backend=qk_coordinate_slot_lookup`, `candidate_single_offset_coordinate_table_backend=qk_coordinate_table_slots`, `candidate_slider_targets_backend=qk_explicit_slider_ray_slot_writes`, `candidate_slider_ray_backend=qk_explicit_7_step_ray_slot_writes`, `candidate_record_emit_backend=qk_candidate_slot_write_attention`, `candidate_record_compaction_backend=qk_prefix_rank_slot_write_attention`, `terminal_material_backend=qk_material_class_bitmask_attention`, `strict_qk_layer_split_remaining=terminal_check_state_king_scan,castle_target_chess_control_flow,legal_filter_batch_attack_chess_control_flow,legal_filter_batch_ray_scan_control_flow`, `monolithic_custom_cuda_rule_kernels_allowed=false`, and `monolithic_custom_cuda_rule_kernels_remaining=true`.
- Contract honesty correction: previous `full_frozen_attention_only=true` was a semantic overclaim. Current native rule path contains CUDA attention-labeled layers, and Rust tests now scan CUDA function bodies for concrete source-level chess-control-flow evidence before allowing gap names to disappear.
- Required legal-filter v2 target is `legal_filter_v2_target=stack_of_frozen_2d_self_attention_layers`, `legal_filter_v2_required_backend=cutlass_qk_scores_hardmax_v_write`, and `legal_filter_v2_required_layers=move_type_select,board_write_select,en_passant_capture_select,castle_rook_write_select,promotion_select,king_square_select,attack_source_select,ray_blocker_select,final_legal_select`.
- `cmz_engine_frozen_layer_step_count` counts execution through the lowered frozen-attention layer stack. Board trace projection, trace packet streaming, and frozen legal trace emission now increment this counter.
- `cmz_engine_frozen_attack_mask` exposes the frozen attack-mask layer for static chess geometry. The lowered attack path currently covers pawn, knight, and king attack tests inside the native attack checker; bishop, rook, and queen static empty-board masks are exposed and tested.
- `cmz_engine_frozen_ray_scan_mask` exposes the blocker-aware frozen ray-scan layer. Slider attack detection and slider target-mask generation now route through this ray-scan layer.
- `cmz_engine_frozen_candidate_target_mask` exposes a frozen candidate-target layer. Pawn, knight, bishop, rook, queen, and king target-square masks now route through this layer.
- `cmz_cuda_candidate_moves_attention` exposes pseudo-legal move-record generation as seven CUDA frozen-attention layers: context_select, piece_dispatch, target_mask_select, castle_merge, promotion_expand, record_emit, and record_order_select. Native legal move listing, legal trace generation, and make-move resolution consume CUDA-emitted candidate records.
- `cmz_cuda_resolve_move_attention` resolves requested UCI moves against CUDA candidate records plus CUDA legal-filter bits; the old C++ match loop is no longer the rule path.
- `cmz_engine_frozen_move_legal` exposes the frozen move-legality layer. Legal filtering now routes through candidate decode, castling target masks, board write transition, king lookup, and frozen attack tests.
- `cmz_cuda_make_move_board_attention` exposes reusable make-move board-square self-attention. The layer stack writes source-square emptying, destination moving piece, en-passant capture deletion, castling rook movement, and promotion replacement. `cmz_engine_make_move_trace_packets` and `cmz_engine_frozen_make_move_trace_attention_packets` now use this path for board square updates.
- `cmz_cuda_make_move_metadata_attention` exposes reusable make-move metadata self-attention. The layer stack writes side-to-move toggle, castling rights, en-passant square, halfmove clock, and fullmove number. `cmz_engine_make_move_trace_packets` and `cmz_engine_frozen_make_move_trace_attention_packets` now use this path for metadata updates.
- `cmz_engine_frozen_terminal_status` exposes terminal predicates through CUDA frozen-attention layers inside `cmz_cuda_terminal_status_attention`: draw_rule_select, candidate_moves_attention, legal_filter_v2_batch_attention, legal_presence_batch_reduce, check_state_select, material_select, and final_status_select. Ongoing, checkmate, stalemate, fifty-move, threefold, insufficient-material, and adjudication statuses route through this CUDA path.
- `cmz_engine_frozen_legal_trace_attention_packets` exposes frozen legal trace emission through `move_record_attention`, `promotion_attention`, and `trace_packet_attention`; output equivalence with the native legal trace packet stream is tested.
- `cmz_engine_frozen_make_move_trace_attention_packets` exposes frozen make-move trace emission through `trace_packet_attention`; output equivalence with the native make-move trace packet stream is tested.

## Chess Rule Coverage In Native V1

- side to move from FEN
- all piece movement
- pawn double pushes
- captures
- promotions to queen, rook, bishop, knight
- en passant
- pinned en-passant legality filter
- castling rights
- castling through check rejection
- king safety legal filter
- deterministic lexicographic UCI ordering

## Current Boundary

- Native v55 is a verified compute-library, trace-codec, legal-trace, stream-trace, make-move-trace, incremental CUDA trace-select stream-decode without full trace precompute buffer, CUDA trace-append packet emission, HullKV/2D attention contract with CUTLASS QK score and CUDA hardmax/select, board trace projection, CUDA QK-hardmax-V attack-table lookup, CUDA candidate-target top-level QK piece-family dispatch, explicit pawn-target QK slot writes, pawn single-push condition lowered into 64-square QK write entries, pawn double-push condition lowered into start-rank QK table plus single-push nonzero QK table plus 64-square target-empty QK write entries, pawn en-passant capture condition lowered into target-match QK table plus captured-slot side/ep QK table plus captured-enemy QK table, slider ray-step condition lowered into 22x22 slider coordinate QK table plus prior-step enabled QK table plus occupied-square QK table plus final QK write, pawn slot helper split into named QK condition layers, single-offset coordinate-slot QK table lookup, explicit 12x12 shifted-coordinate QK entries for single-offset coordinate lookup, single-offset bounds-slot QK filtering, explicit slider ray-slot QK writes, explicit seven-step slider-ray QK writes, explicit candidate offset QK slot writes, QK candidate-record slot decode/write, QK prefix-rank record compaction without per-slot prior-scan loop or atomicMax, terminal legal-presence routed through pseudo-legal candidate attention plus batched legal-filter attention plus GPU presence reduce, terminal material routed through QK material-class bitmask attention and status-select attention, nine-layer CUDA pseudo-legal candidate move attention, CUDA resolve-move QK-hardmax legal-set lookup, CUDA nearest-blocker ray-scan attention, CUDA castle-path attention, single-move legal_filter_v2 complete 9-layer self-attention with inner 2D QK hardmax helpers, batched legal_filter_v2 complete 9-layer self-attention with inner 2D QK hardmax helpers, reusable make-move board-square CUDA self-attention layers, reusable make-move metadata CUDA self-attention layers, CUDA terminal-status attention, removed legacy legal-filter v1 monolithic CUDA symbols, removed legacy Python ranker/baseline/search modules, removed Python/PyTorch Percepta attention runtime modules, Rust dashboard foundation with native policy decoder selection, policy-only LibTorch CUDA decoder slice, semantic CUDA source-body audit gate, and a truthful target/current split for the `full_frozen_attention_only` rule graph contract.
- Native v55 truth contract is `target_full_frozen_attention_only=true`, `current_full_frozen_2d_self_attention_only=false`, `full_frozen_attention_only=false`, `full_rule_lowering_complete=false`, `semantic_attention_purity=false`, `contract_overclaim_fixed=true`, `dashboard_policy_decoder=true`, `metadata_only_tests_remaining=false`, `candidate_pawn_targets_backend=qk_explicit_pawn_slot_writes`, `candidate_single_offset_backend=qk_bounds_slot_friendly_filter`, `candidate_single_offset_coordinate_backend=qk_coordinate_slot_lookup`, `candidate_single_offset_coordinate_table_backend=qk_coordinate_table_slots`, `candidate_slider_targets_backend=qk_explicit_slider_ray_slot_writes`, `candidate_slider_ray_backend=qk_explicit_7_step_ray_slot_writes`, `candidate_record_emit_backend=qk_candidate_slot_write_attention`, `candidate_record_compaction_backend=qk_prefix_rank_slot_write_attention`, `terminal_material_backend=qk_material_class_bitmask_attention`, and `monolithic_custom_cuda_rule_kernels_remaining=true` for the native hot rule path. Custom CUDA/CUTLASS is still the desired low-level substrate for frozen 2D self-attention score/select/value/write layers, but current source-body scans still find terminal check-state, castle, and batched legal-filter control-flow bodies that must be lowered before semantic purity can be claimed. The trainable decoder contract is policy-only: VM provides chess rules and state transitions; decoder receives no built-in critic, value baseline, evaluator, search algorithm, or pruning policy.

## Next Native Steps

1. Lower terminal check-state offender: `terminal_check_state_king_scan`.
2. Lower castle offender: `castle_target_chess_control_flow`.
3. Lower batched legal-filter residual offenders: `legal_filter_batch_attack_chess_control_flow` and `legal_filter_batch_ray_scan_control_flow`.
