# Percepta Article Notes: Can LLMs Be Computers?

- source_url=https://www.percepta.ai/blog/can-llms-be-computers
- access_date=2026-05-26
- storage_policy=Full verbatim article text is not stored in this repository because the public page is copyrighted. This file stores implementation-relevant notes, source attribution, and UI observations.
- project_relevance=ChessMachineZero Percepta-style trace VM, frozen attention rule weights, token-by-token execution trace, dashboard trace display.

## Core Architecture Notes

- program_model=Program logic is represented as weights/circuits inside a transformer-like execution substrate.
- runtime_model=Inference emits execution trace tokens sequentially; host observes/appends/displays trace, not rule-executes program logic.
- trace_model=Execution is inspectable through token traces and a readable trace log.
- speed_model=Long traces remain sequential at the program-step level, while lookup/memory access should be made cheap through structured attention/indexing.
- UI_implication=Dashboard should expose raw token trace and readable log as separate views.

## Token Display Notes From Source UI Reference

- layout=Left pane contains user input, execution stats, and assistant trace; right pane visualizes program state.
- assistant_panel=Scrollable trace output with two modes: `Readable log` and `Token trace`.
- readable_log=Human-readable execution statements such as search attempts, constraint checks, and branch decisions.
- token_trace=Raw compact token rows followed by decoded operation annotations.
- token_row_style=Monospace; raw bytes/fields first; operation annotation second; branch/control tokens visually distinct.
- stats_row=Live speed/volume counters shown above assistant trace: tokens per second, total tokens, lines per second.
- implementation_mapping=ChessMachineZero dashboard now maps trace packets to `Readable log` lines and `Token trace` hexadecimal field rows.

## ChessMachineZero Mapping

- tensor_trace_in=Prompt trace contains board square writes and state registers.
- frozen_attention_blocks=PerceptaFrozenAttentionRuleComputer owns compiled ISA/microprogram weights, frozen attention block stack, and matrix-attention interpreter.
- tensor_trace_out=Legal move enumeration emits CANDIDATE/LEGAL_SET/HALT packets; move execution emits COMMIT_MOVE/WRITE_SQ/state/TERMINAL/HALT packets.
- readable_log_mapping=CANDIDATE becomes a readable `try <uci>` line; LEGAL_SET becomes legality status; COMMIT_MOVE becomes committed move; WRITE_SQ becomes board write; TERMINAL_SET becomes terminal status.
- token_trace_mapping=Each TracePacket is rendered as seven integer fields converted to two-character hexadecimal tokens plus decoded packet annotation.

## Implementation Constraints

- no_external_tree_search=true
- no_human_game_training_data=true
- no_engine_labels=true
- no_tablebase_labels=true
- no_handcrafted_chess_evaluation=true
- python_chess_runtime=false
- python_chess_test_oracle_only=true

## Follow-Up Engineering Targets

- target_1=Keep one-token-at-a-time observable decode semantics.
- target_2=Avoid recomputing full trace continuation for every emitted token by caching compiled continuation state for the active prompt/context.
- target_3=Move frozen attention weights and prompt tensors to explicit CUDA device ownership.
- target_4=Fuse QK/mask/hardmax/select/V/residual operations into CUDA/CUTLASS kernels after CPU recomputation bottleneck is removed.
