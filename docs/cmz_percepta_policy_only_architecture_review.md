# ChessMachineZero Percepta-Like Policy-Only Architecture Review

## Review Target

```text
project=ChessMachineZero
review_doc=cmz_percepta_policy_only_architecture_review
date=2026-05-28
architecture_target=single transformer-computer with frozen rules VM plus trainable policy-only decoder
current_native_status=target_full_frozen_attention_only=true; current_full_frozen_2d_self_attention_only=false; full_frozen_attention_only=false
current_decoder_status=policy-only scaffold, not trained strategy
review_goal=validate conceptual architecture and identify implementation gaps without adding external search/eval
```

## Core Thesis

ChessMachineZero should be a transformer-hosted chess computer, not a conventional chess engine wrapped around a neural policy/value model.

The frozen VM is the chess world:

```text
VM = rules + legal move generation + apply move + terminal status + execution trace
VM != evaluator
VM != search algorithm
VM != move advisor
VM != handcrafted strategy
```

The trainable decoder is the free decision-making system:

```text
decoder = policy controller over generic VM commands
decoder learns strategy from self-play only
decoder decides when to inspect, plan, remember, branch, and commit
decoder receives no built-in critic/value head/evaluator/search policy
```

The system boundary is:

```text
board/workspace/trace state
  -> trainable policy-only decoder
  -> generic VM command
  -> frozen VM attention layers
  -> updated board/workspace/rule status/trace
  -> repeat until COMMIT_MOVE or terminal
```

## Non-Negotiable Constraints

```text
human_game_labels=false
engine_labels=false
tablebase_labels=false
opening_book=false
handcrafted_eval=false
external_MCTS=false
external_beam_search=false
external_pruning_policy=false
python_hot_path=false
silent_fallback=false
weaker_behavior_fallback=false
flat_external_policy_head_for_legal_moves=false
```

Allowed signals:

```text
chess rules
self-play terminal result
trace consistency checks
python-chess oracle in tests only
deterministic unit tests
CUDA/CUTLASS as low-level execution substrate for attention layers
```

## Mental Model

Use this mental model for review:

```text
VM = eyes/hands/legs
decoder = learned brain
trace = memory/audit trail
workspace = private differentiable scratchpad
host = I/O, streaming, display, test harness
```

VM tells the decoder:

```text
move_is_legal
next_board_after_move
terminal_status
command_status
trace_packets_for_audit
```

VM never tells the decoder:

```text
move_quality
material_score
positional_score
best_line
search_depth_to_use
branch_to_prune
which_move_human/engine/tablebase_prefers
```

## Current Native Implementation Snapshot

```text
runtime_stack=Rust orchestration + C++ host layer + CUDA/CUTLASS kernels
native_rule_vm_contract=target_full_frozen_attention_only=true; full_frozen_attention_only=false
current_full_frozen_2d_self_attention_only=false
full_rule_lowering_complete=false
semantic_attention_purity=false
contract_overclaim_fixed=true
trace_streaming_backend=incremental_packet_attention
trace_streaming_buffered=false
trace_streaming_full_trace_precompute=false
semantic_source_audit=rust_cuda_body_scan_v1
metadata_only_tests_remaining=false
candidate_pawn_targets_backend=qk_explicit_pawn_slot_writes
candidate_single_offset_backend=qk_bounds_slot_friendly_filter
candidate_single_offset_coordinate_backend=qk_coordinate_slot_lookup
candidate_single_offset_coordinate_table_backend=qk_coordinate_table_slots
candidate_slider_targets_backend=qk_explicit_slider_ray_slot_writes
candidate_slider_ray_backend=qk_explicit_7_step_ray_slot_writes
candidate_record_emit_backend=qk_candidate_slot_write_attention
candidate_record_compaction_backend=parallel_qk_slot_rank_write_attention
strict_qk_layer_split_remaining=candidate_record_prefix_rank_control_flow,terminal_legal_presence_chess_search,terminal_material_counting_control_flow,terminal_check_state_king_scan,castle_target_chess_control_flow,legal_filter_batch_attack_chess_control_flow,legal_filter_batch_ray_scan_control_flow
python_hot_path=false
fallback_allowed=false
```

Native VM rule execution target is named frozen attention layers. CUDA/CUTLASS is treated as implementation substrate for QK score, hardmax/select, V read, and residual/write behavior. Current source audit uses Rust CUDA body scanning and names concrete remaining chess-control-flow offenders instead of a broad metadata-test gap.

Implemented native rule-layer categories:

```text
board_projection=latest_write_hardmax_2d
trace_select=cursor_hardmax_2d
trace_append=cuda_trace_packet_emit_attention
attack_masks=cuda_qk_hardmax_v_table_lookup
candidate_targets=cuda_qk_hardmax_v_target_lookup
ray_scan=cuda_nearest_blocker_attention
castling_targets=cuda_castle_path_attention
pseudo_legal_moves=cuda_candidate_moves_layered_attention
legal_filter=cuda_legal_filter_v2_layered_self_attention
legal_filter_batch=cuda_legal_filter_v2_batched_layered_self_attention
make_move_board_squares=cuda_make_move_board_attention
make_move_metadata=cuda_make_move_metadata_attention
terminal_predicates=cuda_terminal_status_layered_attention
resolve_move=cuda_resolve_move_qk_hardmax_legal_set_attention
HullHardmax2D=CUTLASS_QK_scores + CUDA_hardmax_select
NestedHullTopK2D=CUTLASS_QK_scores + CUDA_topk_select
```

Implemented full-rule examples:

```text
pawn/knight/bishop/rook/queen/king movement
captures
promotion_to_NBRQ
en_passant
pinned_en_passant_legality
castling_rights
castling_through_check_rejection
king_safety_filter
checkmate
stalemate
fifty_move_rule
threefold_repetition_boundary
insufficient_material
deterministic_UCI_ordering
```

Current decoder contract:

```text
decoder_backend=libtorch_cuda_policy_only_v1
decoder_attention=2d_heads
head_dim=2
shared_white_black=true
learning_method=self_play_policy_gradient
actor_critic=false
critic_head_enabled=false
value_head_enabled=false
externally_prescribed_critic=false
tracepacket_backprop=false
```

## Composite Transformer-Computer View

VM and decoder are separate engineering/training boundaries, not separate conceptual agents.

```text
composite_model =
  frozen_VM_attention_layers
  + trainable_policy_decoder_attention_layers
  + shared trace/workspace interface
```

Engineering boundary:

```text
frozen_VM.trainable=false
policy_decoder.trainable=true
VM/decoder C++ APIs can be separate for testing and debugging
```

Conceptual boundary:

```text
single transformer-computer
same board/workspace/trace language
same command loop
same GPU runtime
```

Training boundary:

```text
VM gradients=none
TracePacket gradients=none
decoder gradients=policy loss only
self-play result provides reward
```

## Data Model

### Board State

Board state may appear in two forms:

```text
board_trace[packet_count,7] = append-only audit/replay trace
board_hidden[64 or 68, hidden_dim/token_fields] = internal VM/decoder state
```

TracePacket is side channel:

```text
used_for=UI,audit,replay,test_debug,optional_candidate_log,optional_legal_set_log
not_used_for=primary differentiable memory,planner_memory,backprop_path
```

Board hidden state is the main internal state:

```text
state_transfer=board_hidden + workspace_hidden + side_to_move_hidden
trace_transfer=optional/debug/audit
```

### TracePacket

Current packet width:

```text
TracePacket[7] = [OP,A0,A1,A2,A3,TAG,COMMIT]
```

Trace semantics:

```text
append_only=true
fixed_width=true
deterministic_encoding=true
board_reconstruction=latest_square_write_attention
legal_set=trace-emitted LEGAL_SET packets
commit=COMMIT_MOVE packet
terminal=TERMINAL_SET packet
```

### Workspace Hidden

Workspace is private scratchpad, not an externally prescribed search tree.

Target workspace state:

```text
workspace_hidden[workspace_slots, hidden_dim]
workspace_meta_hidden[workspace_slots, meta_dim]
active_slot_hidden
valid/free flags
parent/depth/status metadata if learned command usage needs storage
```

Important rule:

```text
workspace_slots are memory affordances, not a forced tree-search algorithm.
decoder may use slots as branches, notes, candidates, cache, or ignore slots.
```

## Frozen VM Contract

VM input:

```text
board_hidden
side_to_move_hidden
workspace_hidden
selected_command_hidden
optional_trace_context
```

VM output:

```text
updated_board_hidden
updated_workspace_hidden
rule_hidden_outputs
terminal_hidden
last_command_status_hidden
optional_trace_packets
```

Required VM operations:

```text
LEGAL_QUERY
APPLY_MOVE
TERMINAL_QUERY
MAKE_MOVE
WRITE_WORKSPACE
READ_WORKSPACE
SELECT_WORKSPACE
RESET_WORKSPACE
COMMIT_MOVE
NOOP
```

VM must reject invalid commands explicitly:

```text
illegal_move -> illegal_command_status
invalid_slot -> illegal_command_status
malformed_command -> illegal_command_status
CUDA_failure -> hard_error
fallback -> forbidden
```

Terminal hidden is always present:

```text
ongoing
white_win
black_win
draw
mate
stalemate
rule_draw
illegal_command_status
```

## Frozen Attention Interpretation

All chess rule primitives must lower to frozen attention semantics:

```text
QK score
hardmax/select
V read
write/residual/update
```

CUDA/CUTLASS is allowed only as implementation of these semantics:

```text
semantic_layer=frozen_2d_attention
technical_backend=CUDA/CUTLASS kernel
not_allowed=arbitrary CPU interpreter hot path
not_allowed=silent CPU substitute
```

Examples:

```text
latest square write:
  query=(square_id,current_time)
  keys=(write_square,write_time)
  hardmax selects latest matching write
  value=piece_token

ray scan:
  query=(ray_id,from_square)
  keys=(candidate_blocker_ray,candidate_blocker_distance)
  hardmax/select picks nearest blocker
  value=allowed_target_mask

legal filter:
  move_type_select
  board_write_select
  en_passant_capture_select
  castle_rook_write_select
  promotion_select
  king_square_select
  attack_source_select
  ray_blocker_select
  final_legal_select
```

## Policy-Only Decoder Contract

The decoder is trainable and strategy-bearing.

Input:

```text
rule_hidden_outputs
board_hidden
side_to_move_hidden
workspace_status_hidden
optional trace/context tokens
```

Output:

```text
command_logits
selected_command_hidden
optional sampling metadata
```

Current v1 command names:

```text
QUERY_RULES
READ_WORKSPACE
WRITE_WORKSPACE
SELECT_WORKSPACE
COMMIT_MOVE
NOOP
```

Target command set may expand, but must remain generic motor/attention commands:

```text
APPLY_TO_ACTIVE
BRANCH_FROM_ACTIVE
SELECT_SLOT
COPY_SLOT
INVALIDATE_SLOT
RESET_ACTIVE_TO_ROOT
COMMIT_MOVE
NOOP
```

Forbidden decoder outputs:

```text
hardcoded_depth
hardcoded_beam_width
hardcoded_pruning_rule
external_engine_call
external_value_call
MCTS_node_API
tablebase_probe
human_opening_hint
```

Policy-only means:

```text
command_logits=true
value_head=false
critic_head=false
actor_critic=false
baseline_module=false
```

The decoder may still learn internal evaluation inside its own weights. The architecture must not provide a separately named, supervised, or hardwired critic/evaluator as a required module.

## Inference Loop

High-level loop:

```text
initialize board_hidden from FEN/board_trace
initialize workspace_hidden
while game not terminal:
  repeat internal planning steps:
    frozen_VM(board_hidden,workspace_hidden,side_to_move,command)
      -> rule_hidden_outputs + updated_workspace + command_status
    policy_decoder(rule_hidden_outputs,workspace,side_to_move)
      -> next command logits
    sample/select command
    if command == COMMIT_MOVE:
      frozen_VM applies move to root board
      emit commit trace packet
      break to next ply
```

Host role:

```text
receive trace packets
stream to UI
persist audit logs
run tests/probes
never compute chess rules in production hot path
never choose strategic move
```

## Training Loop

Current intended training:

```text
one_shared_decoder_self_play_game
  -> decoder samples generic commands
  -> frozen VM validates/applies commands
  -> terminal result produced by frozen VM
  -> reward assigned by side to move / player identity
  -> policy gradient update on decoder logprobs
```

Policy loss:

```text
loss = -logprob(selected_command) * reward
```

Current policy-only stance:

```text
baseline=false
value_loss=false
critic_loss=false
```

Future optional variance reduction can be discussed only if it does not become a prescribed critic/evaluator. Default architecture should remain policy-only.

Reward assignment:

```text
white_win -> white command rewards positive, black command rewards negative
black_win -> black command rewards positive, white command rewards negative
draw -> configurable draw reward
illegal_command -> negative command-local penalty possible only as rule/status penalty, not chess evaluation
```

Training data restrictions:

```text
self_play_only=true
engine_labels=false
human_game_labels=false
tablebase_labels=false
handcrafted_eval=false
```

## Search Freedom Principle

The decoder must not be forced into a named search algorithm.

Allowed:

```text
decoder asks VM for legal moves
decoder applies hypothetical moves to workspace
decoder copies/selects workspace slots
decoder commits immediately
decoder spends many internal steps before commit
decoder ignores workspace if learned policy prefers immediate move
decoder learns branch selection implicitly through command logits
```

Forbidden:

```text
external wrapper decides depth
external wrapper expands tree
external wrapper prunes candidates
external wrapper ranks moves
external wrapper computes value
fixed MCTS/beam/negamax controller outside decoder
```

This is the central user-intended advantage:

```text
model has exact chess rules available internally
model learns how much and what to inspect
model learns strategy natively rather than being forced through hand-coded search
```

## Dashboard Contract

Dashboard must be display-only.

Dashboard may show:

```text
board
white readable trace
white token trace
black readable trace
black token trace
current emitter
verification status
terminal status
packets/tokens counters
```

Dashboard must not:

```text
compute legal moves
select moves
call python-chess at runtime
patch illegal behavior
hide failures with fallback display
```

Trace display requirement:

```text
Percepta-style readable log tab
Percepta-style token trace tab
auto-follow only when scrolled to bottom
manual scroll position preserved
```

## Verification Contract

Required native tests:

```text
DenseHardmax2D == HullHardmax2D for tested prefixes
DenseTopK2D == NestedHullTopK2D for tested prefixes
native legal/apply/terminal == python-chess oracle in tests only
target_full_frozen_attention_only=true
full_frozen_attention_only=false
contract_overclaim_fixed=true
python_hot_path=false
fallback_allowed=false
decoder head_dim=2
decoder policy_only=true
value_head_enabled=false
actor_critic=false
all committed moves in VM legal set
trace packets emitted per committed move
dashboard displays trace progressively and preserves manual scroll
```

Recent verification status:

```text
cargo fmt --all -- --check = passed
cargo clippy --workspace --all-targets -- -D warnings = passed
cargo test --workspace = passed
python -m pytest -p no:cacheprovider -q = passed, 112 tests
python -m pytest -p no:cacheprovider -W error -q = passed, 112 tests
```

## Current Gaps

Current architecture is not finished as a strong chess player.

Known remaining work:

```text
self-play training loop for policy-only decoder
native WebSocket dashboard trace streaming
batched trace emission allocation fusion
more CUTLASS-backed score blocks inside candidate/terminal internals where practical
larger trainable decoder architecture
checkpointing and replay pipeline
multi-game training harness
performance profiling after training loop exists
```

Not a gap:

```text
absence_of_value_head = intentional
absence_of_actor_critic = intentional
absence_of_MCTS = intentional
absence_of_handcrafted_eval = intentional
absence_of_human/engine/tablebase labels = intentional
```

## Reviewer Questions

Please review:

```text
1. Does the VM/decoder boundary preserve the user's intent that VM gives only chess rules?
2. Does policy-only decoder avoid prescribing strategy while still being trainable?
3. Does full_frozen_attention_only=true overclaim any remaining non-attention production rule path?
4. Are TracePacket side-channel semantics separated enough from differentiable workspace state?
5. Are generic commands too restrictive or too search-shaped?
6. Does CUDA/CUTLASS usage remain attention substrate rather than external interpreter logic?
7. What minimal self-play training loop should be implemented next without introducing eval/search wrappers?
```

## Short Summary

```text
ChessMachineZero = transformer-computer for chess.
Frozen VM = exact chess rules in frozen 2D attention layers.
Policy decoder = trainable command policy, no required critic/value/evaluator.
Trace = append-only audit and replay side channel.
Workspace = internal scratchpad controlled by decoder.
Training = self-play policy gradient only.
Host = I/O, UI, logging, tests.
```
