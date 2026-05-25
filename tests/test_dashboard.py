from __future__ import annotations

import json

import pytest

from chess_machine_zero.chess.board_io import STARTING_FEN
from chess_machine_zero.chess.rules_oracle import board_after_uci, legal_uci_set
from chess_machine_zero.dashboard.server import DashboardApp
from chess_machine_zero.dashboard.state import CMZDashboardSession, DashboardMoveError
from chess_machine_zero.vm.trace_packet import TraceOp


def test_dashboard_snapshot_exposes_transformer_rule_state_from_trace() -> None:
    session = CMZDashboardSession(seed=20260524)
    snapshot = session.snapshot()

    assert snapshot["fen"] == STARTING_FEN
    assert snapshot["side_to_move"] == "w"
    assert snapshot["legal_count"] == 20
    assert set(snapshot["legal_moves"]) == legal_uci_set(STARTING_FEN)
    assert len(snapshot["board"]["squares"]) == 64
    assert snapshot["engine"]["rules_module"] == "PerceptaFrozenAttentionRuleComputer"
    assert snapshot["engine"]["rule_execution_mode"] == "percepta_frozen_attention_trace_vm"
    assert snapshot["engine"]["attention_backend"] == "logarithmic_2d_attention"
    assert snapshot["engine"]["lookup_complexity"] == "O(log n)"
    assert snapshot["engine"]["rule_core_execution_mode"] == "executable_frozen_attention_layer_graph"
    assert snapshot["engine"]["primitive_kernel_execution_mode"] == "pure_frozen_attention_tensor_layers"
    assert snapshot["engine"]["core_trace_runtime"] == "tensor_trace_in_frozen_attention_blocks_tensor_trace_out"
    assert snapshot["engine"]["core_rule_compute_backend"] == "frozen_transformer_attention_block_stack"
    assert snapshot["engine"]["tensor_kernel_shortcut_runtime"] is False
    assert snapshot["engine"]["compiled_attention_block_stack"] is True
    assert snapshot["engine"]["compiled_attention_block_count"] >= 6
    assert snapshot["engine"]["compiled_attention_head_count"] >= snapshot["engine"]["compiled_attention_block_count"]
    assert snapshot["engine"]["residual_trace_write_count"] >= 2
    assert snapshot["engine"]["percepta_compiler_pipeline"] == "chess_isa_microprogram_to_frozen_attention_weights"
    assert snapshot["engine"]["rule_compiler_backend"] == "rule_microprogram_to_frozen_attention_weights"
    assert snapshot["engine"]["rule_microprogram_source"] == "chess_rule_isa"
    assert snapshot["engine"]["unified_rule_executor_runtime"] is True
    assert snapshot["engine"]["handwritten_stack_primitive_runtime"] is False
    assert snapshot["engine"]["matrix_attention_interpreter_runtime"] is True
    assert snapshot["engine"]["executor_substrate"] == "matrix_attention_interpreter"
    assert snapshot["engine"]["attention_step_operator"] == "QK^T_mask_hardmax_select_V_residual_write"
    assert snapshot["engine"]["pytorch_domain_shortcut_runtime"] is False
    assert snapshot["engine"]["matrix_attention_step_count"] > 0
    assert snapshot["engine"]["compiled_rule_program_weight_count"] >= snapshot["engine"]["rule_microprogram_instruction_count"]
    assert snapshot["engine"]["python_host_boundary_role"] == "display_only"
    assert snapshot["engine"]["tensor_trace_core_runtime"] is True
    assert snapshot["engine"]["tracepacket_core_runtime"] is False
    assert snapshot["engine"]["python_rule_primitive_runtime"] is False
    assert snapshot["engine"]["python_control_flow_rule_primitives"] is False
    assert snapshot["engine"]["compiled_layer_graph_serialized"] is True
    assert set(snapshot["engine"]["compiled_rule_primitives"]) >= {
        "PIECE_DISPATCH",
        "RAY_SCAN",
        "ATTACK_TEST",
        "LEGAL_FILTER",
        "MAKE_MOVE",
        "TERMINAL_PREDICATES",
    }
    assert snapshot["engine"]["tensor_kernel_count"] >= 6
    assert snapshot["engine"]["parametric_rule_weights"] is True
    assert snapshot["engine"]["host_append_only"] is True
    assert snapshot["engine"]["token_streaming"] is True
    assert snapshot["engine"]["uses_mlp"] is False
    assert snapshot["engine"]["position_lookup"] is False
    assert snapshot["engine"]["compiled_prompt_count"] == 0
    assert snapshot["engine"]["trainable_rule_parameters"] == 0
    assert snapshot["engine"]["compiled_rule_parameters"] > 4096
    assert snapshot["engine"]["python_rule_executor_runtime"] is False
    assert snapshot["engine"]["external_tree_search"] is False
    assert snapshot["illegal_commit_count"] == 0
    assert snapshot["transformers"]["mode"] == "two_transformer_selfplay"
    assert snapshot["transformers"]["active"] == "transformer_white"
    assert snapshot["transformers"]["white"]["rules_module"] == "PerceptaFrozenAttentionRuleComputer"
    assert snapshot["transformers"]["black"]["rules_module"] == "PerceptaFrozenAttentionRuleComputer"
    assert snapshot["transformers"]["white"]["executor_substrate"] == "matrix_attention_interpreter"
    assert snapshot["transformers"]["black"]["executor_substrate"] == "matrix_attention_interpreter"
    assert snapshot["trace_legal_verification"]["selected_move_in_legal_set"] is None
    assert snapshot["trace_legal_verification"]["illegal_commit_count"] == 0


def test_dashboard_transformer_selfplay_step_commits_only_trace_legal_move() -> None:
    session = CMZDashboardSession(seed=20260524)
    event = session.step_transformer()
    snapshot = session.snapshot()

    assert event.actor == "transformer_white"
    assert event.transformer_id == "transformer_white"
    assert event.trace_verified_legal is True
    assert event.move_uci in legal_uci_set(STARTING_FEN)
    assert event.trace_op_counts["CANDIDATE"] == 20
    assert event.trace_op_counts["LEGAL_SET"] == 20
    assert event.trace_op_counts["COMMIT_MOVE"] >= 1
    assert snapshot["fen"] == board_after_uci(STARTING_FEN, event.move_uci)
    assert snapshot["ply"] == 1
    assert snapshot["illegal_commit_count"] == 0
    assert snapshot["last_trace"]["op_counts"]["CANDIDATE"] == 20
    assert snapshot["history"][0]["move_uci"] == event.move_uci
    assert snapshot["history"][0]["actor"] == "transformer_white"
    assert snapshot["history"][0]["transformer_id"] == "transformer_white"
    assert snapshot["history"][0]["trace_verified_legal"] is True
    assert snapshot["history"][0]["emitted_token_count"] == len(event.trace)
    assert snapshot["history"][0]["emitted_tokens"][0] == list(event.trace[0].to_tokens())
    assert snapshot["transformer_token_streams"]["white"]["packet_count"] == len(event.trace)
    assert snapshot["transformer_token_streams"]["black"]["packet_count"] == 0
    assert snapshot["trace_legal_verification"]["selected_move_in_legal_set"] is True


def test_dashboard_two_transformers_alternate_and_emit_verified_tokens() -> None:
    session = CMZDashboardSession(seed=20260524)

    white_event = session.step_transformer()
    after_white = session.snapshot()
    black_fen = after_white["fen"]
    black_event = session.step_transformer()
    after_black = session.snapshot()

    assert white_event.actor == "transformer_white"
    assert black_event.actor == "transformer_black"
    assert white_event.move_uci in legal_uci_set(STARTING_FEN)
    assert black_event.move_uci in legal_uci_set(black_fen)
    assert white_event.trace_verified_legal is True
    assert black_event.trace_verified_legal is True
    assert after_black["fen"] == board_after_uci(black_fen, black_event.move_uci)
    assert after_black["ply"] == 2
    assert after_black["illegal_commit_count"] == 0
    assert after_black["history"][0]["actor"] == "transformer_white"
    assert after_black["history"][1]["actor"] == "transformer_black"
    assert after_black["history"][0]["emitted_token_count"] == len(white_event.trace)
    assert after_black["history"][1]["emitted_token_count"] == len(black_event.trace)
    assert after_black["history"][0]["emitted_tokens"][0] == list(white_event.trace[0].to_tokens())
    assert after_black["history"][1]["emitted_tokens"][0] == list(black_event.trace[0].to_tokens())
    assert after_black["transformer_token_streams"]["white"]["packets"][0]["tokens"] == list(white_event.trace[0].to_tokens())
    assert after_black["transformer_token_streams"]["black"]["packets"][0]["tokens"] == list(black_event.trace[0].to_tokens())


def test_dashboard_accepts_human_move_with_parametric_weight_rules() -> None:
    session = CMZDashboardSession(seed=20260524)
    events = session.play_human_move("e2e4", auto_reply=False)
    after = session.snapshot()

    assert len(events) == 1
    assert events[0].actor == "human"
    assert after["fen"] == board_after_uci(STARTING_FEN, "e2e4")
    assert after["ply"] == 1
    assert after["illegal_attempt_count"] == 0


def test_dashboard_rejects_illegal_human_move_without_mutating_board() -> None:
    session = CMZDashboardSession(seed=20260524)
    before = session.snapshot()

    with pytest.raises(DashboardMoveError, match="illegal move"):
        session.play_human_move("e2e5")

    after = session.snapshot()
    assert after["fen"] == before["fen"]
    assert after["ply"] == before["ply"]
    assert after["history"] == before["history"]
    assert after["illegal_attempt_count"] == before["illegal_attempt_count"] + 1


def test_dashboard_reset_replays_same_seeded_transformer_move() -> None:
    session = CMZDashboardSession(seed=20260524)
    first = session.step_transformer()

    session.reset()
    second = session.step_transformer()

    assert second.move_uci == first.move_uci
    assert first.trace_op_counts.get("SAMPLE_SET", 0) == 0
    assert second.trace_op_counts.get("SAMPLE_SET", 0) == 0


def test_dashboard_http_api_exposes_exact_state_transitions() -> None:
    app = DashboardApp(seed=20260524)

    status, headers, body = app.handle("GET", "/api/snapshot", b"")
    initial = json.loads(body.decode("utf-8"))
    assert status == 200
    assert headers["content-type"] == "application/json; charset=utf-8"
    assert initial["legal_count"] == 20

    status, _, body = app.handle("POST", "/api/step", json.dumps({"count": 1}).encode("utf-8"))
    replied = json.loads(body.decode("utf-8"))
    assert status == 200
    assert replied["ply"] == 1
    assert replied["history"][-1]["actor"] == "transformer_white"
    assert replied["history"][-1]["emitted_token_count"] == len(replied["history"][-1]["emitted_tokens"])


def test_dashboard_http_api_rejects_bad_move_with_error_payload() -> None:
    app = DashboardApp(seed=20260524)

    status, _, body = app.handle("POST", "/api/move", json.dumps({"move": "e2e5"}).encode("utf-8"))
    payload = json.loads(body.decode("utf-8"))

    assert status == 400
    assert payload["error"]["code"] == "illegal_move"
    assert payload["snapshot"]["fen"] == STARTING_FEN
    assert payload["snapshot"]["illegal_attempt_count"] == 1


def test_dashboard_static_assets_are_local_and_route_backed() -> None:
    app = DashboardApp(seed=20260524)

    index_status, index_headers, index_body = app.handle("GET", "/", b"")
    js_status, js_headers, js_body = app.handle("GET", "/static/dashboard.js", b"")
    css_status, css_headers, css_body = app.handle("GET", "/static/dashboard.css", b"")
    combined = b"\n".join((index_body, js_body, css_body)).decode("utf-8")

    assert index_status == 200
    assert js_status == 200
    assert css_status == 200
    assert index_headers["content-type"] == "text/html; charset=utf-8"
    assert js_headers["content-type"] == "text/javascript; charset=utf-8"
    assert css_headers["content-type"] == "text/css; charset=utf-8"
    assert 'id="board"' in combined
    assert "/api/snapshot" in combined
    assert "http://" not in combined
    assert "https://" not in combined
    assert "cdn" not in combined.lower()


def test_dashboard_frontend_autoplays_sequential_transformer_selfplay() -> None:
    app = DashboardApp(seed=20260524)
    status, _, js_body = app.handle("GET", "/static/dashboard.js", b"")
    source = js_body.decode("utf-8")

    assert status == 200
    assert "state.busy" in source
    assert "window.setTimeout(autoPlayLoop, 250)" in source
    assert "window.setInterval" not in source
    assert "computing move" in source
    assert "startAutoPlay()" in source
